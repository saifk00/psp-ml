//! Operator fusion pass.
//!
//! Runs after constant folding, before codegen lowering.
//! Pattern-matches op sequences and replaces them with fused ops.

use std::collections::HashMap;

use super::graph::DType;
use super::psp::{Activation, PspModel, PspOp};
use crate::ir::graph::TensorId;

/// Fusion rules that need folded constants and inferred shapes.
///
/// Run *after* `const_fold`, unlike [`fuse`]: `Pad`'s padding amounts are a
/// tensor, and they are only guaranteed to be a readable constant once folding
/// has run.
pub fn fuse_late(model: &mut PspModel) {
    let count = fuse_pad_conv(model);
    if count > 0 {
        eprintln!("fuse: folded {count} Pad ops into the following convolution");
    }
    let count = fuse_pow_const(model);
    if count > 0 {
        eprintln!("fuse: rewrote {count} Pow ops with a scalar exponent into PowConst");
    }
}

/// Fold `Pad(x → p)` + `Conv(p → o)` into `Conv(x → o)` with that padding.
///
/// TFLite emits an explicit `Pad` when a conv's padding is not expressible as
/// SAME/VALID, but our conv kernels take explicit per-edge padding and skip the
/// border taps, so the padded copy is pure overhead — 236 ms across BirdNET's
/// four sites, materialising up to 468 KB per call.
///
/// This has to happen before the arena is planned, which is why it lives here
/// rather than in codegen. Fusing changes liveness: `x` must now survive until
/// the conv, where before it died at the `Pad`. Applied post-hoc to generated
/// code it silently corrupts output — measured on BirdNET, where the planner
/// had placed one conv's output at the same arena offset as its would-be input,
/// so the conv overwrote its own input mid-computation.
///
/// Only fires when the padded tensor has exactly one consumer, the padding is
/// spatial (no batch or channel padding), and the conv is not already padding
/// itself.
fn fuse_pad_conv(model: &mut PspModel) -> usize {
    let mut use_count: HashMap<TensorId, usize> = HashMap::new();
    for op in &model.graph.ops {
        for tid in op.inputs() {
            *use_count.entry(tid).or_insert(0) += 1;
        }
    }

    // padded tensor -> (source tensor, [top, bottom, left, right])
    let mut folds: HashMap<TensorId, (TensorId, [usize; 4])> = HashMap::new();
    for op in &model.graph.ops {
        let PspOp::Pad { input, paddings, output } = op else { continue };
        // One consumer, and not exported: `use_count` only counts op inputs, so
        // a Pad feeding both a conv and the model's output would otherwise lose
        // its only producer.
        if use_count.get(output) != Some(&1) || model.graph.outputs.contains(output) {
            continue;
        }
        let Some(p) = read_pad_amounts(model, *paddings) else { continue };
        // Batch/channel padding has no conv equivalent.
        if p[0] != [0, 0] || p[3] != [0, 0] {
            continue;
        }
        folds.insert(*output, (*input, [p[1][0], p[1][1], p[2][0], p[2][1]]));
    }
    if folds.is_empty() {
        return 0;
    }

    // NOTE: `infer_shapes` treats any nonzero conv padding as SAME and
    // re-derives it from the input shape. The padding written here is by
    // construction *not* SAME — that is why TFLite emitted a separate Pad — so
    // this pass must stay after the last `infer` call. Moving it earlier, or
    // adding a second inference pass, would silently rewrite the fold to SAME:
    // same output shape, different tap alignment, wrong numbers, no error.
    let mut fused = 0;
    // The *padded* tensors whose consumer took over the padding. Keying this on
    // the Pad's source instead would delete every Pad sharing that source, even
    // ones whose own consumer was skipped — their output would then never be
    // written and the consumer would read an uninitialised arena slot.
    let mut fused_pads: Vec<TensorId> = Vec::new();
    for op in &mut model.graph.ops {
        let (input, params) = match op {
            PspOp::Conv2d { input, params, .. } => (input, params),
            PspOp::DepthwiseConv2d { input, params, .. } => (input, params),
            _ => continue,
        };
        let padded = *input;
        let Some(&(src, pad)) = folds.get(&padded) else { continue };
        // Adding to a conv that already pads would need the two to compose;
        // not worth the risk for a case that does not occur.
        if params.pad_top != 0 || params.pad_bottom != 0 || params.pad_left != 0 || params.pad_right != 0 {
            continue;
        }
        *input = src;
        params.pad_top = pad[0];
        params.pad_bottom = pad[1];
        params.pad_left = pad[2];
        params.pad_right = pad[3];
        fused_pads.push(padded);
        fused += 1;
    }
    if fused == 0 {
        return 0;
    }

    // Drop exactly the Pad ops whose consumer took over the padding.
    model.graph.ops.retain(|op| match op {
        PspOp::Pad { output, .. } => !fused_pads.contains(output),
        _ => true,
    });
    fused
}

/// Read a single-element F32 constant, if it is one.
fn read_f32_const(model: &PspModel, tid: TensorId) -> Option<f32> {
    use super::graph::TensorKind;
    let TensorKind::Constant { offset, len } = model.graph.tensor(tid).kind else {
        return None;
    };
    let b = model.model_data.get(offset..offset + len)?;
    if b.len() != 4 {
        return None;
    }
    Some(f32::from_le_bytes(b.try_into().ok()?))
}

/// Whether `tid` is produced by an op that cannot emit a negative value.
///
/// Deliberately narrow: a self-multiply is the only form needed today, and a
/// wrong answer here turns into NaN on device that no host check would catch.
fn is_non_negative(model: &PspModel, tid: TensorId) -> bool {
    use super::psp::BinaryOp;
    model.graph.ops.iter().any(|op| match op {
        PspOp::ElementWise { op: BinaryOp::Mul, input_a, input_b, output } => {
            *output == tid && input_a == input_b
        }
        _ => false,
    })
}

/// Read a `[4,2]` INT32 padding constant, if it is one.
fn read_pad_amounts(model: &PspModel, paddings: TensorId) -> Option<[[usize; 2]; 4]> {
    use super::graph::TensorKind;
    let TensorKind::Constant { offset, len } = model.graph.tensor(paddings).kind else {
        return None;
    };
    let vals: Vec<i32> = model.model_data.get(offset..offset + len)?
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if vals.len() != 8 || vals.iter().any(|&v| v < 0) {
        return None;
    }
    Some([
        [vals[0] as usize, vals[1] as usize],
        [vals[2] as usize, vals[3] as usize],
        [vals[4] as usize, vals[5] as usize],
        [vals[6] as usize, vals[7] as usize],
    ])
}

/// Rewrite `Pow(x, c)` with a single-element constant exponent into `PowConst`.
///
/// `libm::powf` costs ~678 cycles per element; `x^c = 2^(c log2 x)` is three
/// VFPU ops over four lanes. Worth 200 ms on BirdNET's two spectrogram
/// compression ops.
///
/// `vlog2` of a negative is NaN, so this only fires where that cannot silently
/// disagree with `libm::powf`:
///
/// - **Non-integer exponent** — `powf(x, c)` is NaN for `x < 0` there too, so
///   both paths agree *provided* `vlog2` of a negative is also NaN. That is
///   asserted by `test_pow_const_matches_libm` in `kernels::checks`, which runs
///   on device against the real instruction.
/// - **Provably non-negative base** — currently a self-multiply (`x * x`),
///   which is what BirdNET's two spectrogram-compression sites are fed by.
///
/// Without this guard the divergence is invisible to every check we have: the
/// host mirror of `pow_const` uses `libm::powf`, so `cargo test`, `--features
/// local` and the `BIRDNET_TAP` diff would all pass while the device produced
/// NaN. Widen it only alongside a device-side check.
fn fuse_pow_const(model: &mut PspModel) -> usize {
    use super::graph::TensorKind;
    use super::psp::BinaryOp;

    let mut rewritten = 0;
    for i in 0..model.graph.ops.len() {
        let PspOp::ElementWise { op: BinaryOp::Pow, input_a, input_b, output } =
            model.graph.ops[i]
        else {
            continue;
        };
        let (a, b) = (input_a, input_b);
        // The exponent must be a single constant float.
        let t = model.graph.tensor(b);
        if t.shape.iter().product::<usize>() != 1 || t.dtype != DType::F32 {
            continue;
        }
        if !matches!(t.kind, TensorKind::Constant { .. }) {
            continue;
        }
        let Some(c) = read_f32_const(model, b) else { continue };
        // x^0 is the one case a non-negative base does not rescue: libm gives
        // 1.0 at x == 0, the VFPU gives 2^(0 * -inf) = 2^NaN = NaN, and a
        // self-multiply base hits 0 routinely.
        if c == 0.0 {
            continue;
        }
        if c.fract() == 0.0 && !is_non_negative(model, a) {
            continue;
        }
        model.graph.ops[i] = PspOp::PowConst { input: a, exponent: b, output };
        rewritten += 1;
    }
    rewritten
}

/// Run all fusion rules on the model.
pub fn fuse(model: &mut PspModel) {
    let count = fuse_rfft(model);
    if count > 0 {
        eprintln!("fuse: fused {} RFFT2D+SQUEEZE+CAST chains", count);
    }
    let count = fuse_swish(model);
    if count > 0 {
        eprintln!("fuse: fused {} Logistic+Mul chains into Swish", count);
    }
    let count = fuse_conv_swish(model);
    if count > 0 {
        eprintln!("fuse: fused {} Swish ops into the preceding conv", count);
    }
}

/// Fold `Conv2d/DepthwiseConv2d(… → t)` + `Swish(t → o)` into the conv as
/// `fused_activation = Swish`, writing `o` directly.
///
/// The convs are lowered to a GEMM (or the depthwise kernel) whose store
/// already applies the bias and an optional ReLU; swish costs the same
/// kernel four more VFPU ops per quad while the tile is still in registers,
/// versus a separate pass that re-reads and re-writes the whole tensor from
/// DRAM. BirdNET has 44 of these after every conv (213 ms of `swish` passes
/// on top of 176 ms of `bias_add` passes at the 3783 ms baseline).
///
/// Only fires when the conv output has the swish as its sole consumer, the
/// conv has no other fused activation, and nobody asked for `t` itself.
fn fuse_conv_swish(model: &mut PspModel) -> usize {
    let consumers = consumer_map(model);
    let producers = producer_map(model);
    let mut to_remove: Vec<usize> = Vec::new();
    // (conv op index, new output tensor)
    let mut rewrites: Vec<(usize, TensorId)> = Vec::new();

    for (idx, op) in model.graph.ops.iter().enumerate() {
        let PspOp::Swish { input: t, output: o } = op else {
            continue;
        };
        let Some(&conv_idx) = producers.get(t) else {
            continue;
        };
        let params = match &model.graph.ops[conv_idx] {
            PspOp::Conv2d { params, output, .. }
            | PspOp::DepthwiseConv2d { params, output, .. }
                if output == t =>
            {
                params
            }
            _ => continue,
        };
        if params.fused_activation.is_some() {
            continue;
        }
        if consumers.get(t).map(|c| c.len()) != Some(1) {
            continue;
        }
        if model.graph.outputs.contains(t) {
            continue;
        }
        rewrites.push((conv_idx, *o));
        to_remove.push(idx);
    }

    let count = rewrites.len();
    for (conv_idx, new_out) in rewrites {
        match &mut model.graph.ops[conv_idx] {
            PspOp::Conv2d { params, output, .. }
            | PspOp::DepthwiseConv2d { params, output, .. } => {
                params.fused_activation = Some(Activation::Swish);
                *output = new_out;
            }
            _ => unreachable!("rewrite target was checked to be a conv"),
        }
    }
    to_remove.sort_unstable();
    for idx in to_remove.into_iter().rev() {
        model.graph.ops.remove(idx);
    }
    count
}

/// Fuse `Logistic(x → s)` + `Mul(x, s → o)` into `Swish(x → o)`.
///
/// This is the swish/SiLU activation, which TFLite emits as two primitive ops
/// because the format has no opcode for it. BirdNET has 45 such pairs over 3.7
/// M elements; fusing removes an entire pass over the tensor (the intermediate
/// `s` never has to be written or re-read) on top of letting one VFPU kernel
/// do the whole chain.
///
/// Only fires when `s` has exactly one consumer, otherwise the sigmoid is
/// needed in its own right.
fn fuse_swish(model: &mut PspModel) -> usize {
    use super::psp::{BinaryOp, UnaryOp};
    let consumers = consumer_map(model);
    let producers = producer_map(model);
    let mut to_remove: Vec<usize> = Vec::new();
    let mut replacements: Vec<(usize, PspOp)> = Vec::new();

    for (idx, op) in model.graph.ops.iter().enumerate() {
        let PspOp::ElementWise {
            op: BinaryOp::Mul,
            input_a,
            input_b,
            output,
        } = op
        else {
            continue;
        };

        // One operand must be a Logistic whose input is the other operand.
        for (sig, x) in [(*input_a, *input_b), (*input_b, *input_a)] {
            let Some(&prod_idx) = producers.get(&sig) else {
                continue;
            };
            let PspOp::UnaryElementWise {
                op: UnaryOp::Logistic,
                input: logistic_in,
                output: logistic_out,
            } = &model.graph.ops[prod_idx]
            else {
                continue;
            };
            if *logistic_in != x || *logistic_out != sig {
                continue;
            }
            // The sigmoid must feed nothing else, or we still have to compute it.
            if consumers.get(&sig).map(|c| c.len()) != Some(1) {
                continue;
            }
            // Don't swallow a value the caller asked for.
            if model.graph.outputs.contains(&sig) {
                continue;
            }
            replacements.push((
                idx,
                PspOp::Swish {
                    input: x,
                    output: *output,
                },
            ));
            to_remove.push(prod_idx);
            break;
        }
    }

    let count = replacements.len();
    for (idx, op) in replacements {
        model.graph.ops[idx] = op;
    }
    to_remove.sort_unstable();
    to_remove.dedup();
    for idx in to_remove.into_iter().rev() {
        model.graph.ops.remove(idx);
    }
    count
}

/// Build map from tensor ID → vec of op indices that consume it.
fn consumer_map(model: &PspModel) -> HashMap<TensorId, Vec<usize>> {
    let mut map: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, op) in model.graph.ops.iter().enumerate() {
        for id in op.inputs() {
            map.entry(id).or_default().push(idx);
        }
    }
    map
}

/// Build map from tensor ID → op index that produces it.
fn producer_map(model: &PspModel) -> HashMap<TensorId, usize> {
    let mut map: HashMap<TensorId, usize> = HashMap::new();
    for (idx, op) in model.graph.ops.iter().enumerate() {
        for id in op.all_outputs() {
            map.insert(id, idx);
        }
    }
    map
}

/// Fuse RFFT2D → Reshape(SQUEEZE) → Cast(C64→F32) into a single Rfft op.
///
/// Returns the number of fused chains.
fn fuse_rfft(model: &mut PspModel) -> usize {
    let consumers = consumer_map(model);
    let producers = producer_map(model);
    let mut to_remove: Vec<usize> = Vec::new();
    let mut replacements: Vec<(usize, PspOp)> = Vec::new();

    for (i, op) in model.graph.ops.iter().enumerate() {
        let (input, fft_length_id, complex_out) = match op {
            PspOp::Rfft2d {
                input,
                fft_length,
                output,
            } => (*input, *fft_length, *output),
            _ => continue,
        };

        // Look backwards: if the input comes from a Reshape that inserts a
        // singleton dim (3D→4D for TFLite's Rfft2d), absorb it so the fused
        // Rfft operates directly on the 3D tensor.
        let single_consumer = consumers.get(&input).map_or(false, |cs| cs.len() == 1);
        let real_input = if single_consumer {
            if let Some(&prod_idx) = producers.get(&input) {
                match &model.graph.ops[prod_idx] {
                    PspOp::ExpandDims { input: expand_in, .. } => {
                        to_remove.push(prod_idx);
                        *expand_in
                    }
                    PspOp::Reshape { input: reshape_in, output: reshape_out, .. } => {
                        let in_rank = model.graph.tensor(*reshape_in).shape.len();
                        let out_rank = model.graph.tensor(*reshape_out).shape.len();
                        if out_rank == in_rank + 1 {
                            to_remove.push(prod_idx);
                            *reshape_in
                        } else {
                            input
                        }
                    }
                    _ => input,
                }
            } else { input }
        } else {
            input
        };

        // Read fft_length constant: expect [1, N] (1D FFT along last dim)
        let fft_length_vals = match model.read_i32_const(fft_length_id) {
            Some(v) => v,
            None => continue,
        };
        if fft_length_vals.len() != 2 || fft_length_vals[0] != 1 {
            eprintln!(
                "fuse: skipping RFFT2D with non-1D fft_length: {:?}",
                fft_length_vals
            );
            continue;
        }
        let n = fft_length_vals[1] as usize;

        // Trace: complex_out → optional Reshape(SQUEEZE) → Cast(C64→F32)
        // The SQUEEZE may be aliased away during TFLite lowering, so handle both:
        //   Rfft2d → Reshape → Cast   (3-op chain)
        //   Rfft2d → Cast             (2-op chain, SQUEEZE was aliased)
        let next_consumers = match consumers.get(&complex_out) {
            Some(cs) if cs.len() == 1 => cs,
            _ => continue,
        };
        let next_idx = next_consumers[0];
        let cast_idx = match &model.graph.ops[next_idx] {
            // Direct Cast (SQUEEZE was aliased away during TFLite lowering)
            PspOp::Cast { .. } => next_idx,
            // Squeeze/Reshape (SQUEEZE still present), then Cast
            PspOp::Squeeze { output, .. } | PspOp::Reshape { output, .. } => {
                let squeezed_out = *output;
                let cast_consumers = match consumers.get(&squeezed_out) {
                    Some(cs) if cs.len() == 1 => cs,
                    _ => continue,
                };
                match &model.graph.ops[cast_consumers[0]] {
                    PspOp::Cast { .. } => {
                        to_remove.push(next_idx); // remove the Reshape
                        cast_consumers[0]
                    }
                    _ => continue,
                }
            }
            _ => continue,
        };
        let final_out = match &model.graph.ops[cast_idx] {
            PspOp::Cast { output, .. } => *output,
            _ => continue,
        };

        // Verify the final output is F32
        if model.graph.tensor(final_out).dtype != DType::F32 {
            continue;
        }

        // Replace RFFT2D with fused Rfft, remove Cast (Reshape already handled above)
        replacements.push((
            i,
            PspOp::Rfft {
                input: real_input,
                output: final_out,
                fft_length: n,
            },
        ));
        to_remove.push(cast_idx);
    }

    let count = replacements.len();

    // Apply replacements
    for (idx, new_op) in replacements {
        model.graph.ops[idx] = new_op;
    }

    // Remove consumed ops in reverse order to preserve indices
    to_remove.sort_unstable();
    to_remove.dedup();
    for idx in to_remove.into_iter().rev() {
        model.graph.ops.remove(idx);
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::{DType, Graph, TensorKind};
    use crate::ir::psp::{BinaryOp, Conv2dParams};

    fn pad_bytes(p: [[i32; 2]; 4]) -> Vec<u8> {
        p.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn conv_params() -> Conv2dParams {
        Conv2dParams {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            pad_top: 0,
            pad_bottom: 0,
            pad_left: 0,
            pad_right: 0,
            fused_activation: None,
        }
    }

    /// Pad(x -> p) + DepthwiseConv(p -> o) becomes one padded conv reading x.
    #[test]
    fn pad_folds_into_the_following_conv() {
        let data = pad_bytes([[0, 0], [1, 1], [1, 1], [0, 0]]);
        let mut g = Graph::<PspOp>::new();
        let x = g.add_tensor(vec![1, 6, 16, 4], DType::F32, TensorKind::Input);
        g.inputs.push(x);
        let pads = g.add_tensor(
            vec![4, 2],
            DType::I32,
            TensorKind::Constant { offset: 0, len: data.len() },
        );
        let p = g.add_tensor(vec![1, 8, 18, 4], DType::F32, TensorKind::Intermediate);
        let w = g.add_tensor(vec![1, 3, 3, 4], DType::F32, TensorKind::Constant { offset: 0, len: 4 });
        let o = g.add_tensor(vec![1, 3, 8, 4], DType::F32, TensorKind::Output);
        g.outputs.push(o);
        g.ops.push(PspOp::Pad { input: x, paddings: pads, output: p });
        g.ops.push(PspOp::DepthwiseConv2d {
            input: p,
            weights: w,
            bias: None,
            output: o,
            params: conv_params(),
        });

        let mut model = PspModel { graph: g, model_data: data };
        assert_eq!(fuse_pad_conv(&mut model), 1);

        assert!(
            !model.graph.ops.iter().any(|op| matches!(op, PspOp::Pad { .. })),
            "the Pad should be gone"
        );
        match &model.graph.ops[0] {
            PspOp::DepthwiseConv2d { input, params, .. } => {
                assert_eq!(*input, x, "the conv must now read the unpadded tensor");
                assert_eq!(
                    (params.pad_top, params.pad_bottom, params.pad_left, params.pad_right),
                    (1, 1, 1, 1)
                );
            }
            other => panic!("expected a depthwise conv, got {other:?}"),
        }
    }

    /// Conv(x -> t) + Swish(t -> o) becomes Conv(x -> o) with Swish fused.
    #[test]
    fn swish_folds_into_the_preceding_conv() {
        let mut g = Graph::<PspOp>::new();
        let x = g.add_tensor(vec![1, 8, 8, 4], DType::F32, TensorKind::Input);
        g.inputs.push(x);
        let w = g.add_tensor(vec![8, 1, 1, 4], DType::F32, TensorKind::Constant { offset: 0, len: 128 });
        let t = g.add_tensor(vec![1, 8, 8, 8], DType::F32, TensorKind::Intermediate);
        let o = g.add_tensor(vec![1, 8, 8, 8], DType::F32, TensorKind::Output);
        g.outputs.push(o);
        g.ops.push(PspOp::Conv2d {
            input: x,
            weights: w,
            bias: None,
            output: t,
            weight_scales: None,
            params: Conv2dParams { kernel_h: 1, kernel_w: 1, stride_h: 1, stride_w: 1, ..conv_params() },
        });
        g.ops.push(PspOp::Swish { input: t, output: o });

        let mut model = PspModel { graph: g, model_data: vec![] };
        assert_eq!(fuse_conv_swish(&mut model), 1);
        assert_eq!(model.graph.ops.len(), 1, "the Swish should be gone");
        match &model.graph.ops[0] {
            PspOp::Conv2d { output, params, .. } => {
                assert_eq!(*output, o, "the conv must now write the swish's output");
                assert_eq!(params.fused_activation, Some(Activation::Swish));
            }
            other => panic!("expected a conv, got {other:?}"),
        }
    }

    /// The pre-activation tensor is needed elsewhere, so the Swish stays.
    #[test]
    fn swish_is_not_folded_when_the_conv_output_has_another_reader() {
        let mut g = Graph::<PspOp>::new();
        let x = g.add_tensor(vec![1, 8, 8, 4], DType::F32, TensorKind::Input);
        g.inputs.push(x);
        let w = g.add_tensor(vec![8, 1, 1, 4], DType::F32, TensorKind::Constant { offset: 0, len: 128 });
        let t = g.add_tensor(vec![1, 8, 8, 8], DType::F32, TensorKind::Intermediate);
        let s = g.add_tensor(vec![1, 8, 8, 8], DType::F32, TensorKind::Intermediate);
        let o = g.add_tensor(vec![1, 8, 8, 8], DType::F32, TensorKind::Output);
        g.outputs.push(o);
        g.ops.push(PspOp::Conv2d {
            input: x,
            weights: w,
            bias: None,
            output: t,
            weight_scales: None,
            params: Conv2dParams { kernel_h: 1, kernel_w: 1, stride_h: 1, stride_w: 1, ..conv_params() },
        });
        g.ops.push(PspOp::Swish { input: t, output: s });
        // The residual add reads the pre-activation value too.
        g.ops.push(PspOp::ElementWise { op: BinaryOp::Add, input_a: t, input_b: s, output: o });

        let mut model = PspModel { graph: g, model_data: vec![] };
        assert_eq!(fuse_conv_swish(&mut model), 0);
        assert_eq!(model.graph.ops.len(), 3);
    }

    /// Channel padding has no conv equivalent, so it must be left alone.
    #[test]
    fn channel_padding_is_not_folded() {
        let data = pad_bytes([[0, 0], [1, 1], [1, 1], [2, 2]]);
        let mut g = Graph::<PspOp>::new();
        let x = g.add_tensor(vec![1, 6, 16, 4], DType::F32, TensorKind::Input);
        g.inputs.push(x);
        let pads = g.add_tensor(
            vec![4, 2],
            DType::I32,
            TensorKind::Constant { offset: 0, len: data.len() },
        );
        let p = g.add_tensor(vec![1, 8, 18, 8], DType::F32, TensorKind::Intermediate);
        let w = g.add_tensor(vec![1, 3, 3, 8], DType::F32, TensorKind::Constant { offset: 0, len: 4 });
        let o = g.add_tensor(vec![1, 3, 8, 8], DType::F32, TensorKind::Output);
        g.outputs.push(o);
        g.ops.push(PspOp::Pad { input: x, paddings: pads, output: p });
        g.ops.push(PspOp::DepthwiseConv2d {
            input: p,
            weights: w,
            bias: None,
            output: o,
            params: conv_params(),
        });

        let mut model = PspModel { graph: g, model_data: data };
        assert_eq!(fuse_pad_conv(&mut model), 0);
        assert!(model.graph.ops.iter().any(|op| matches!(op, PspOp::Pad { .. })));
    }

    /// A padded tensor read by two ops cannot be folded away.
    #[test]
    fn a_pad_with_two_consumers_is_not_folded() {
        let data = pad_bytes([[0, 0], [1, 1], [1, 1], [0, 0]]);
        let mut g = Graph::<PspOp>::new();
        let x = g.add_tensor(vec![1, 6, 16, 4], DType::F32, TensorKind::Input);
        g.inputs.push(x);
        let pads = g.add_tensor(
            vec![4, 2],
            DType::I32,
            TensorKind::Constant { offset: 0, len: data.len() },
        );
        let p = g.add_tensor(vec![1, 8, 18, 4], DType::F32, TensorKind::Intermediate);
        let w = g.add_tensor(vec![1, 3, 3, 4], DType::F32, TensorKind::Constant { offset: 0, len: 4 });
        let o = g.add_tensor(vec![1, 3, 8, 4], DType::F32, TensorKind::Output);
        let o2 = g.add_tensor(vec![1, 8, 18, 4], DType::F32, TensorKind::Output);
        g.outputs.push(o);
        g.ops.push(PspOp::Pad { input: x, paddings: pads, output: p });
        g.ops.push(PspOp::DepthwiseConv2d {
            input: p,
            weights: w,
            bias: None,
            output: o,
            params: conv_params(),
        });
        g.ops.push(PspOp::Swish { input: p, output: o2 });

        let mut model = PspModel { graph: g, model_data: data };
        assert_eq!(fuse_pad_conv(&mut model), 0);
    }

    fn f32_bytes(v: f32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    /// The PowConst rewrite must not fire where `vlog2` could disagree with
    /// `libm::powf` — i.e. an integer exponent over a base that might be
    /// negative. That divergence is device-only, so no host check would see it.
    #[test]
    fn pow_const_only_fires_where_the_vfpu_path_cannot_diverge() {
        // (exponent, base is a square, should fuse)
        let cases = [
            (0.2199f32, false, true),  // non-integer: powf is NaN for x<0 too
            (2.0f32, false, false),    // integer, base not provably >= 0
            (3.0f32, true, true),      // integer, but base is x*x
        ];
        for (exp, squared, expect) in cases {
            let data = f32_bytes(exp);
            let mut g = Graph::<PspOp>::new();
            let x = g.add_tensor(vec![1, 8], DType::F32, TensorKind::Input);
            g.inputs.push(x);
            let e = g.add_tensor(
                vec![1],
                DType::F32,
                TensorKind::Constant { offset: 0, len: 4 },
            );
            let o = g.add_tensor(vec![1, 8], DType::F32, TensorKind::Output);
            g.outputs.push(o);
            let base = if squared {
                let sq = g.add_tensor(vec![1, 8], DType::F32, TensorKind::Intermediate);
                g.ops.push(PspOp::ElementWise {
                    op: BinaryOp::Mul,
                    input_a: x,
                    input_b: x,
                    output: sq,
                });
                sq
            } else {
                x
            };
            g.ops.push(PspOp::ElementWise {
                op: BinaryOp::Pow,
                input_a: base,
                input_b: e,
                output: o,
            });

            let mut model = PspModel { graph: g, model_data: data };
            let n = fuse_pow_const(&mut model);
            assert_eq!(n, expect as usize, "exp={exp} squared={squared}");
        }
    }

    /// A multi-element exponent is an ordinary elementwise Pow.
    #[test]
    fn tensor_exponent_pow_is_left_alone() {
        let mut g = Graph::<PspOp>::new();
        let x = g.add_tensor(vec![1, 8], DType::F32, TensorKind::Input);
        g.inputs.push(x);
        let e = g.add_tensor(vec![8], DType::F32, TensorKind::Constant { offset: 0, len: 32 });
        let o = g.add_tensor(vec![1, 8], DType::F32, TensorKind::Output);
        g.outputs.push(o);
        g.ops.push(PspOp::ElementWise { op: BinaryOp::Pow, input_a: x, input_b: e, output: o });
        let mut model = PspModel { graph: g, model_data: vec![0u8; 32] };
        assert_eq!(fuse_pow_const(&mut model), 0);
    }

    /// Two Pads sharing one source: folding the first must not delete the
    /// second, whose consumer still needs its output.
    #[test]
    fn folding_one_pad_does_not_delete_its_sibling() {
        let data = pad_bytes([[0, 0], [1, 1], [1, 1], [0, 0]]);
        let mut g = Graph::<PspOp>::new();
        let x = g.add_tensor(vec![1, 6, 16, 4], DType::F32, TensorKind::Input);
        g.inputs.push(x);
        let pads = g.add_tensor(
            vec![4, 2],
            DType::I32,
            TensorKind::Constant { offset: 0, len: data.len() },
        );
        let w = g.add_tensor(vec![1, 3, 3, 4], DType::F32, TensorKind::Constant { offset: 0, len: 4 });

        // Pad #1 -> a conv with no padding of its own: foldable.
        let p1 = g.add_tensor(vec![1, 8, 18, 4], DType::F32, TensorKind::Intermediate);
        let o1 = g.add_tensor(vec![1, 3, 8, 4], DType::F32, TensorKind::Output);
        // Pad #2 -> a conv that already pads: must be left intact.
        let p2 = g.add_tensor(vec![1, 8, 18, 4], DType::F32, TensorKind::Intermediate);
        let o2 = g.add_tensor(vec![1, 3, 8, 4], DType::F32, TensorKind::Output);
        g.outputs.push(o1);
        g.outputs.push(o2);

        g.ops.push(PspOp::Pad { input: x, paddings: pads, output: p1 });
        g.ops.push(PspOp::DepthwiseConv2d {
            input: p1, weights: w, bias: None, output: o1, params: conv_params(),
        });
        g.ops.push(PspOp::Pad { input: x, paddings: pads, output: p2 });
        let mut already = conv_params();
        already.pad_top = 1;
        g.ops.push(PspOp::DepthwiseConv2d {
            input: p2, weights: w, bias: None, output: o2, params: already,
        });

        let mut model = PspModel { graph: g, model_data: data };
        assert_eq!(fuse_pad_conv(&mut model), 1);

        let pads_left: Vec<TensorId> = model.graph.ops.iter().filter_map(|op| match op {
            PspOp::Pad { output, .. } => Some(*output),
            _ => None,
        }).collect();
        assert_eq!(pads_left, vec![p2], "the un-fused Pad must survive to write p2");
    }

    /// A padded tensor that is also a model output still needs its producer.
    #[test]
    fn a_pad_that_is_a_graph_output_is_not_folded() {
        let data = pad_bytes([[0, 0], [1, 1], [1, 1], [0, 0]]);
        let mut g = Graph::<PspOp>::new();
        let x = g.add_tensor(vec![1, 6, 16, 4], DType::F32, TensorKind::Input);
        g.inputs.push(x);
        let pads = g.add_tensor(
            vec![4, 2],
            DType::I32,
            TensorKind::Constant { offset: 0, len: data.len() },
        );
        let p = g.add_tensor(vec![1, 8, 18, 4], DType::F32, TensorKind::Output);
        let w = g.add_tensor(vec![1, 3, 3, 4], DType::F32, TensorKind::Constant { offset: 0, len: 4 });
        let o = g.add_tensor(vec![1, 3, 8, 4], DType::F32, TensorKind::Output);
        g.outputs.push(p);
        g.outputs.push(o);
        g.ops.push(PspOp::Pad { input: x, paddings: pads, output: p });
        g.ops.push(PspOp::DepthwiseConv2d {
            input: p, weights: w, bias: None, output: o, params: conv_params(),
        });

        let mut model = PspModel { graph: g, model_data: data };
        assert_eq!(fuse_pad_conv(&mut model), 0);
        assert!(model.graph.ops.iter().any(|op| matches!(op, PspOp::Pad { .. })));
    }
}
