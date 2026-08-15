//! Operator fusion pass.
//!
//! Runs after constant folding, before codegen lowering.
//! Pattern-matches op sequences and replaces them with fused ops.

use std::collections::HashMap;

use super::graph::DType;
use super::psp::{PspModel, PspOp};
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
        if use_count.get(output) != Some(&1) {
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

    let mut fused = 0;
    let mut consumed: Vec<TensorId> = Vec::new();
    for op in &mut model.graph.ops {
        let (input, params) = match op {
            PspOp::Conv2d { input, params, .. } => (input, params),
            PspOp::DepthwiseConv2d { input, params, .. } => (input, params),
            _ => continue,
        };
        let Some(&(src, pad)) = folds.get(input) else { continue };
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
        consumed.push(src);
        fused += 1;
    }
    if fused == 0 {
        return 0;
    }

    // Drop the Pad ops whose consumer took over the padding.
    let folded: Vec<TensorId> = folds
        .iter()
        .filter(|(_, (src, _))| consumed.contains(src))
        .map(|(out, _)| *out)
        .collect();
    model.graph.ops.retain(|op| match op {
        PspOp::Pad { output, .. } => !folded.contains(output),
        _ => true,
    });
    fused
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
/// Requires `x >= 0` (log2 of a negative is NaN). Both BirdNET sites are fed by
/// a square, but that is not checkable here, so the kernel documents it and the
/// scalar tail uses `libm::powf`, which agrees for non-negative inputs.
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

    /// Pow with a 1-element constant exponent becomes PowConst; a tensor
    /// exponent stays an ordinary elementwise Pow.
    #[test]
    fn scalar_exponent_pow_becomes_pow_const() {
        for (n, expect) in [(1usize, true), (8usize, false)] {
            let mut g = Graph::<PspOp>::new();
            let x = g.add_tensor(vec![1, 8], DType::F32, TensorKind::Input);
            g.inputs.push(x);
            let e = g.add_tensor(
                vec![n],
                DType::F32,
                TensorKind::Constant { offset: 0, len: n * 4 },
            );
            let o = g.add_tensor(vec![1, 8], DType::F32, TensorKind::Output);
            g.outputs.push(o);
            g.ops.push(PspOp::ElementWise {
                op: BinaryOp::Pow,
                input_a: x,
                input_b: e,
                output: o,
            });

            let mut model = PspModel { graph: g, model_data: vec![0u8; n * 4] };
            assert_eq!(fuse_pow_const(&mut model), expect as usize, "n={n}");
            assert_eq!(
                matches!(model.graph.ops[0], PspOp::PowConst { .. }),
                expect,
                "n={n}"
            );
        }
    }
}
