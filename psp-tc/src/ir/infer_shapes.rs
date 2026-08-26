//! Combined I32 evaluation + forward shape propagation pass.
//!
//! Since psp-rt targets fixed-size inputs (no dynamic batching), every tensor
//! shape is statically determinable. This pass iterates ops in graph order
//! (already topological) and:
//!
//! 1. **Evaluates I32 shape-computation ops** (Shape, SplitV, FloorDiv, etc.)
//!    and stores results as constants. This is done IN graph order so that
//!    `Shape(X)` reads X's already-corrected shape, not TFLite's stale annotation.
//!
//! 2. **Infers F32 tensor shapes** from input shapes + op parameters. Uses the
//!    I32 constants produced in step 1 for ops like StridedSlice (begin/end)
//!    and Reshape (shape_tensor).
//!
//! Runs BEFORE `const_fold` so that Shape ops produce correct values.

use super::const_fold::{eval_strided_slice, store_i32, store_i32_shaped};
use super::graph::DType;
use super::psp::{PspModel, PspOp, ReduceOp};
use crate::parse::tflite::lowering::compute_same_padding;

pub fn infer(model: &mut PspModel) {
    let ops = model.graph.ops.clone();
    let mut shape_changes = 0usize;
    let mut i32_evals = 0usize;

    for (op_idx, op) in ops.iter().enumerate() {
        match op {
            // ══════════════════════════════════════════════════════════
            // I32 shape-computation ops: evaluate and store as constants
            // ══════════════════════════════════════════════════════════

            // Builder-provided ops: shapes are declared explicitly at
            // construction, nothing to infer.
            PspOp::StridedViewStft { .. }
            | PspOp::FullyConnectedCB { .. }
            | PspOp::FirDecimate { .. }
            | PspOp::SquarePow { .. } => {}

            PspOp::Shape { input, output } => {
                let vals: Vec<i32> = model.graph.tensor(*input).shape
                    .iter().map(|&s| s as i32).collect();
                store_i32(&mut model.model_data, &mut model.graph, *output, &vals);
                i32_evals += 1;
            }

            PspOp::Pack { inputs, output, axis } => {
                if *axis == 0 && inputs.iter().all(|id| model.read_i32_const(*id).is_some()) {
                    let mut result = Vec::new();
                    for id in inputs {
                        result.extend(model.read_i32_const(*id).unwrap());
                    }
                    store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                    i32_evals += 1;
                }
            }

            PspOp::Range { start, limit, delta, output } => {
                if let (Some(s), Some(l), Some(d)) = (
                    model.read_i32_const(*start),
                    model.read_i32_const(*limit),
                    model.read_i32_const(*delta),
                ) {
                    let (s, l, d) = (s[0], l[0], d[0]);
                    let mut result = Vec::new();
                    let mut v = s;
                    while (d > 0 && v < l) || (d < 0 && v > l) {
                        result.push(v);
                        v += d;
                    }
                    store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                    i32_evals += 1;
                }
            }

            // ══════════════════════════════════════════════════════════
            // Ops that can be EITHER I32 evaluation or F32 shape inference
            // ══════════════════════════════════════════════════════════

            PspOp::SplitV { input, size_splits, axis, outputs } => {
                if model.graph.tensor(*input).dtype == DType::I32 {
                    // I32 evaluation
                    if let (Some(data), Some(splits)) = (
                        model.read_i32_const(*input),
                        model.read_i32_const(*size_splits),
                    ) {
                        let ax = if let Some(ax_vals) = model.read_i32_const(*axis) {
                            ax_vals[0] as usize
                        } else {
                            continue;
                        };
                        // For 1-D data, split directly
                        if ax == 0 {
                            let mut offset = 0usize;
                            for (i, &sz) in splits.iter().enumerate() {
                                let sz = sz as usize;
                                let chunk: Vec<i32> = data[offset..offset + sz].to_vec();
                                store_i32(&mut model.model_data, &mut model.graph, outputs[i], &chunk);
                                offset += sz;
                            }
                            i32_evals += 1;
                        }
                    }
                } else {
                    // F32 shape inference
                    if let (Some(splits), Some(ax_vals)) = (
                        model.read_i32_const(*size_splits),
                        model.read_i32_const(*axis),
                    ) {
                        let ax = ax_vals[0] as usize;
                        let in_s = model.graph.tensor(*input).shape.clone();
                        if ax < in_s.len() {
                            for (i, &sz) in splits.iter().enumerate() {
                                let mut shape = in_s.clone();
                                shape[ax] = sz as usize;
                                shape_changes += set_shape(model, outputs[i], shape);
                            }
                        }
                    }
                }
            }

            PspOp::Concatenation { inputs, output, axis } => {
                let first = model.graph.tensor(inputs[0]);
                if first.dtype == DType::I32 {
                    // I32 evaluation
                    if inputs.iter().all(|id| model.read_i32_const(*id).is_some()) {
                        let mut result = Vec::new();
                        for id in inputs {
                            result.extend(model.read_i32_const(*id).unwrap());
                        }
                        store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                        i32_evals += 1;
                    }
                } else {
                    // F32 shape inference
                    let ax = *axis as usize;
                    let first_shape = model.graph.tensor(inputs[0]).shape.clone();
                    if ax >= first_shape.len() { continue; }
                    if inputs.iter().any(|id| model.graph.tensor(*id).shape.len() <= ax) {
                        continue;
                    }
                    let axis_sum: usize = inputs.iter()
                        .map(|id| model.graph.tensor(*id).shape[ax])
                        .sum();
                    let mut shape = first_shape;
                    shape[ax] = axis_sum;
                    shape_changes += set_shape(model, *output, shape);
                }
            }

            PspOp::Gather { input, indices, output, axis } => {
                if model.graph.tensor(*input).dtype == DType::I32 {
                    // I32 evaluation
                    if *axis == 0 {
                        if let (Some(data), Some(idxs)) = (
                            model.read_i32_const(*input),
                            model.read_i32_const(*indices),
                        ) {
                            let result: Vec<i32> = idxs.iter().map(|&i| data[i as usize]).collect();
                            store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                            i32_evals += 1;
                        }
                    }
                } else {
                    // F32 shape inference
                    let in_s = &model.graph.tensor(*input).shape;
                    let idx_s = &model.graph.tensor(*indices).shape;
                    let ax = *axis as usize;
                    if ax < in_s.len() {
                        let mut shape = in_s[..ax].to_vec();
                        shape.extend_from_slice(idx_s);
                        shape.extend_from_slice(&in_s[ax + 1..]);
                        shape_changes += set_shape(model, *output, shape);
                    }
                }
            }

            PspOp::Reduce { op: reduce_op, input, axes, output, .. } => {
                if model.graph.tensor(*input).dtype == DType::I32 {
                    // I32 evaluation
                    if let Some(data) = model.read_i32_const(*input) {
                        let result = match reduce_op {
                            ReduceOp::Prod => data.iter().copied().product::<i32>(),
                            ReduceOp::Max => data.iter().copied().max().unwrap_or(0),
                            ReduceOp::Min => data.iter().copied().min().unwrap_or(0),
                            ReduceOp::Mean => data.iter().copied().sum::<i32>() / data.len() as i32,
                        };
                        store_i32(&mut model.model_data, &mut model.graph, *output, &[result]);
                        i32_evals += 1;
                    }
                } else {
                    // F32 shape inference
                    let in_s = &model.graph.tensor(*input).shape;
                    match reduce_op {
                        ReduceOp::Mean => {
                            if let Some(ax) = model.read_i32_const(*axes) {
                                let out_s = &model.graph.tensor(*output).shape;
                                let axes_norm: Vec<usize> = ax.iter()
                                    .map(|&a| if a < 0 { (a + in_s.len() as i32) as usize } else { a as usize })
                                    .collect();
                                if out_s.len() == in_s.len() {
                                    // keepDims=true: set reduced axes to 1
                                    let mut shape = in_s.clone();
                                    for &a in &axes_norm {
                                        if a < shape.len() { shape[a] = 1; }
                                    }
                                    shape_changes += set_shape(model, *output, shape);
                                } else {
                                    // keepDims=false: remove reduced axes
                                    let mut shape = Vec::new();
                                    for (i, &d) in in_s.iter().enumerate() {
                                        if !axes_norm.contains(&i) {
                                            shape.push(d);
                                        }
                                    }
                                    shape_changes += set_shape(model, *output, shape);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            PspOp::ElementWise { op: binary_op, input_a, input_b, output } => {
                if model.graph.tensor(*input_a).dtype == DType::I32 {
                    // I32 evaluation with proper multi-dimensional broadcasting
                    if let (Some(a), Some(b)) = (
                        model.read_i32_const(*input_a),
                        model.read_i32_const(*input_b),
                    ) {
                        let a_shape = &model.graph.tensor(*input_a).shape;
                        let b_shape = &model.graph.tensor(*input_b).shape;
                        let (result, out_shape) = eval_broadcast_i32(&a, a_shape, &b, b_shape, *binary_op);
                        store_i32_shaped(&mut model.model_data, &mut model.graph, *output, &result, out_shape);
                        i32_evals += 1;
                    }
                } else {
                    // F32 shape inference (broadcast)
                    let a = &model.graph.tensor(*input_a).shape;
                    let b = &model.graph.tensor(*input_b).shape;
                    let shape = broadcast_shape(a, b);
                    shape_changes += set_shape(model, *output, shape);
                }
            }

            PspOp::StridedSlice {
                input, begin, end, strides, output,
                begin_mask, end_mask, shrink_axis_mask,
            } => {
                if model.graph.tensor(*input).dtype == DType::I32 {
                    // I32 evaluation
                    if let (Some(data), Some(begins), Some(ends), Some(strs)) = (
                        model.read_i32_const(*input),
                        model.read_i32_const(*begin),
                        model.read_i32_const(*end),
                        model.read_i32_const(*strides),
                    ) {
                        if let Some(result) = eval_strided_slice(
                            &data, &begins, &ends, &strs,
                            *begin_mask, *end_mask, *shrink_axis_mask,
                        ) {
                            store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                            i32_evals += 1;
                        }
                    }
                } else {
                    // F32 shape inference
                    let Some(begins) = model.read_i32_const(*begin) else { continue };
                    let Some(ends) = model.read_i32_const(*end) else { continue };
                    let Some(strs) = model.read_i32_const(*strides) else { continue };
                    let in_s = &model.graph.tensor(*input).shape;

                    let mut shape = Vec::new();
                    for d in 0..in_s.len() {
                        if d >= begins.len() { break; }
                        let len = in_s[d] as i32;
                        let stride = strs[d];

                        let b = if begin_mask & (1 << d) != 0 {
                            if stride > 0 { 0 } else { len - 1 }
                        } else {
                            let mut v = begins[d];
                            if v < 0 { v += len; }
                            v
                        };

                        let e = if end_mask & (1 << d) != 0 {
                            if stride > 0 { len } else { -1 }
                        } else {
                            let mut v = ends[d];
                            if v < 0 { v += len; }
                            v
                        };

                        if shrink_axis_mask & (1 << d) != 0 {
                            continue; // dimension removed
                        }

                        let dim = if stride > 0 {
                            ((e - b + stride - 1) / stride).max(0) as usize
                        } else {
                            ((b - e - stride - 1) / (-stride)).max(0) as usize
                        };
                        shape.push(dim);
                    }
                    shape_changes += set_shape(model, *output, shape);
                }
            }

            PspOp::Cast { input, output } => {
                // I32→I32 evaluation (passthrough with correct shape)
                let in_t = model.graph.tensor(*input);
                let out_dtype = model.graph.tensor(*output).dtype;
                if in_t.dtype == DType::I32 && out_dtype == DType::I32 {
                    if let Some(data) = model.read_i32_const(*input) {
                        store_i32(&mut model.model_data, &mut model.graph, *output, &data);
                        i32_evals += 1;
                    }
                }
                // Shape inference (identity) — always applies
                let shape = model.graph.tensor(*input).shape.clone();
                shape_changes += set_shape(model, *output, shape);
            }

            PspOp::Reshape { input, output, shape_tensor, builtin_shape } => {
                if model.graph.tensor(*input).dtype == DType::I32 {
                    // I32 evaluation: pass through data with correct output shape.
                    if let Some(data) = model.read_i32_const(*input) {
                        let in_size = data.len();
                        // Resolve output shape using same priority as F32 Reshape
                        let out_shape = resolve_i32_reshape_shape(
                            model, *output, in_size, shape_tensor, builtin_shape,
                        );
                        store_i32_shaped(&mut model.model_data, &mut model.graph, *output, &data, out_shape);
                        i32_evals += 1;
                    }
                } else {
                    // F32 shape inference
                    let in_size: usize = model.graph.tensor(*input).shape.iter().product();
                    let out_size: usize = model.graph.tensor(*output).shape.iter().product();

                    // Priority 1: TFLite output shape if element count matches
                    // (TFLite's shape inference is correct when inputs haven't changed)
                    if out_size == in_size {
                        continue; // TFLite output is correct
                    }

                    // Element count mismatch — upstream was corrected. Use shape spec.

                    // Priority 2: shape_tensor (from TFLite RESHAPE input[1], may have -1).
                    // Only real RESHAPE ops have shape_tensor set; SQUEEZE/EXPAND_DIMS
                    // are lowered with shape_tensor = None.
                    if let Some(st) = shape_tensor {
                        if let Some(dims) = model.read_i32_const(*st) {
                            if let Some(shape) = resolve_reshape(&dims, in_size) {
                                shape_changes += set_shape(model, *output, shape);
                                continue;
                            }
                        }
                    }

                    // Priority 3: builtin_shape (from TFLite ReshapeOptions, has -1)
                    if let Some(dims) = builtin_shape {
                        if let Some(shape) = resolve_reshape(dims, in_size) {
                            shape_changes += set_shape(model, *output, shape);
                            continue;
                        }
                    }

                    // Priority 4: Expand a singleton dim in TFLite's output shape
                    // to account for upstream corrections (e.g., unsqueeze ops where
                    // TFLite wrote [1,1,1,2048] but input grew from 2048 to 511*2048).
                    let in_shape = &model.graph.tensor(*input).shape;
                    let out_shape = &model.graph.tensor(*output).shape;
                    if let Some(shape) = expand_singleton(in_shape, out_shape, in_size) {
                        shape_changes += set_shape(model, *output, shape);
                        continue;
                    }

                    eprintln!("  WARN: Reshape t{}→t{}: input has {} elements but output shape has {}, no usable shape spec",
                        input, output, in_size, out_size);
                }
            }

            // ══════════════════════════════════════════════════════════
            // F32-only shape inference
            // ══════════════════════════════════════════════════════════

            // Identity shape
            PspOp::UnaryElementWise { input, output, .. }
            | PspOp::Softmax { input, output }
            | PspOp::FakeQuant { input, output }
            | PspOp::Dequantize { input, output }
            | PspOp::PowConst { input, output, .. }
            | PspOp::Swish { input, output }
            | PspOp::ReverseV2 { input, output, .. } => {
                let shape = model.graph.tensor(*input).shape.clone();
                shape_changes += set_shape(model, *output, shape);
            }

            // Squeeze / ExpandDims — deterministic rank changes
            PspOp::Squeeze { input, output, axis } => {
                let mut shape = model.graph.tensor(*input).shape.clone();
                if *axis < shape.len() {
                    shape.remove(*axis);
                }
                shape_changes += set_shape(model, *output, shape);
            }
            PspOp::ExpandDims { input, output, axis } => {
                let mut shape = model.graph.tensor(*input).shape.clone();
                shape.insert(*axis, 1);
                shape_changes += set_shape(model, *output, shape);
            }

            // Conv2d / DepthwiseConv2d / Pool2d
            //
            // Lowering bakes SAME padding into explicit values using TFLite's
            // static shapes — which for models with dynamic framing (BirdNET's
            // SHAPE/RANGE/GATHER front-end) describe a single-frame trace, not
            // the real runtime shapes. Once upstream corrections fix the input
            // shape, any nonzero padding is re-derived as SAME padding for the
            // corrected shape; all-zero padding means VALID and stays.
            PspOp::Conv2d { input, weights, output, params, .. } => {
                let in_s = model.graph.tensor(*input).shape.clone();
                let w_s = model.graph.tensor(*weights).shape.clone();
                if in_s.len() >= 4 && w_s.len() >= 4 {
                    let mut p = params.clone();
                    let was_same =
                        p.pad_top + p.pad_bottom + p.pad_left + p.pad_right > 0;
                    if was_same {
                        let (pt, pb, pl, pr) = compute_same_padding(
                            in_s[1], in_s[2], p.kernel_h, p.kernel_w, p.stride_h, p.stride_w,
                        );
                        (p.pad_top, p.pad_bottom, p.pad_left, p.pad_right) = (pt, pb, pl, pr);
                        if let PspOp::Conv2d { params, .. } = &mut model.graph.ops[op_idx] {
                            *params = p.clone();
                        }
                    }
                    let out_h = conv_dim(in_s[1], p.kernel_h, p.stride_h, p.pad_top, p.pad_bottom);
                    let out_w = conv_dim(in_s[2], p.kernel_w, p.stride_w, p.pad_left, p.pad_right);
                    if out_h > 0 && out_w > 0 {
                        let shape = vec![in_s[0], out_h, out_w, w_s[0]];
                        shape_changes += set_shape(model, *output, shape);
                    } else {
                        shape_changes += propagate_batch(model, *input, *output);
                    }
                }
            }
            PspOp::DepthwiseConv2d { input, output, params, .. } => {
                let in_s = model.graph.tensor(*input).shape.clone();
                if in_s.len() >= 4 {
                    let mut p = params.clone();
                    let was_same =
                        p.pad_top + p.pad_bottom + p.pad_left + p.pad_right > 0;
                    if was_same {
                        let (pt, pb, pl, pr) = compute_same_padding(
                            in_s[1], in_s[2], p.kernel_h, p.kernel_w, p.stride_h, p.stride_w,
                        );
                        (p.pad_top, p.pad_bottom, p.pad_left, p.pad_right) = (pt, pb, pl, pr);
                        if let PspOp::DepthwiseConv2d { params, .. } = &mut model.graph.ops[op_idx] {
                            *params = p.clone();
                        }
                    }
                    let out_h = conv_dim(in_s[1], p.kernel_h, p.stride_h, p.pad_top, p.pad_bottom);
                    let out_w = conv_dim(in_s[2], p.kernel_w, p.stride_w, p.pad_left, p.pad_right);
                    if out_h > 0 && out_w > 0 {
                        let shape = vec![in_s[0], out_h, out_w, in_s[3]];
                        shape_changes += set_shape(model, *output, shape);
                    } else {
                        shape_changes += propagate_batch(model, *input, *output);
                    }
                }
            }
            PspOp::Pool2d { input, output, filter, stride, padding, .. } => {
                let in_s = model.graph.tensor(*input).shape.clone();
                if in_s.len() >= 4 {
                    let mut pad = *padding;
                    let was_same = pad.iter().sum::<usize>() > 0;
                    if was_same {
                        let (pt, pb, pl, pr) = compute_same_padding(
                            in_s[1], in_s[2], filter[0], filter[1], stride[0], stride[1],
                        );
                        pad = [pt, pb, pl, pr];
                        if let PspOp::Pool2d { padding, .. } = &mut model.graph.ops[op_idx] {
                            *padding = pad;
                        }
                    }
                    let out_h = conv_dim(in_s[1], filter[0], stride[0], pad[0], pad[1]);
                    let out_w = conv_dim(in_s[2], filter[1], stride[1], pad[2], pad[3]);
                    if out_h > 0 && out_w > 0 {
                        let shape = vec![in_s[0], out_h, out_w, in_s[3]];
                        shape_changes += set_shape(model, *output, shape);
                    } else {
                        shape_changes += propagate_batch(model, *input, *output);
                    }
                }
            }

            // FullyConnected
            PspOp::FullyConnected { input, weights, output, .. } => {
                let in_s = &model.graph.tensor(*input).shape;
                let w_s = &model.graph.tensor(*weights).shape;
                if !in_s.is_empty() && !w_s.is_empty() {
                    let mut shape = in_s[..in_s.len() - 1].to_vec();
                    shape.push(w_s[0]);
                    shape_changes += set_shape(model, *output, shape);
                }
            }

            // Transpose
            PspOp::Transpose { input, perm, output } => {
                let in_s = &model.graph.tensor(*input).shape;
                if let Some(p) = model.read_i32_const(*perm) {
                    let shape: Vec<usize> = p.iter().map(|&i| in_s[i as usize]).collect();
                    shape_changes += set_shape(model, *output, shape);
                }
            }

            // Pad
            PspOp::Pad { input, paddings, output } => {
                let in_s = &model.graph.tensor(*input).shape;
                if let Some(pads) = model.read_i32_const(*paddings) {
                    let shape: Vec<usize> = in_s.iter().enumerate().map(|(d, &dim)| {
                        dim + pads[d * 2] as usize + pads[d * 2 + 1] as usize
                    }).collect();
                    shape_changes += set_shape(model, *output, shape);
                }
            }

            // Rfft (fused from Rfft2d + Reshape + Cast)
            //
            // The fusion absorbs a Reshape that may have changed rank (4D→3D).
            // We must preserve the TFLite output RANK while scaling the batch
            // dims for corrected upstream shapes.
            PspOp::Rfft { input, output, fft_length } => {
                let in_s = model.graph.tensor(*input).shape.clone();
                let batch: usize = in_s[..in_s.len() - 1].iter().product();
                let freq_bins = fft_length / 2 + 1;
                let expected_elems = batch * freq_bins;
                let out_s = &model.graph.tensor(*output).shape;
                let out_elems: usize = out_s.iter().product();
                if out_elems != expected_elems {
                    // The FFT preserves the input's batch dims in order and
                    // replaces the last dim with freq bins — e.g. [1,511,2048]
                    // → [1,511,1025], matching TFLite's runtime shape. (An
                    // earlier heuristic expanded the *first* singleton of the
                    // static shape instead, yielding [511,1,1025] — which put
                    // the frame count in the batch dim and silently turned the
                    // whole downstream CNN into a per-frame model.)
                    if in_s.len() == out_s.len() {
                        let mut shape = in_s[..in_s.len() - 1].to_vec();
                        shape.push(freq_bins);
                        shape_changes += set_shape(model, *output, shape);
                    } else if let Some(shape) = rfft_output_shape(out_s, batch, freq_bins) {
                        shape_changes += set_shape(model, *output, shape);
                    } else {
                        // Fallback: compute from input shape (changes rank)
                        let mut shape = in_s[..in_s.len() - 1].to_vec();
                        shape.push(freq_bins);
                        shape_changes += set_shape(model, *output, shape);
                    }
                }
            }

            // Already handled above or no shape change needed
            PspOp::Rfft2d { .. } => {}
        }
    }

    if shape_changes > 0 || i32_evals > 0 {
        eprintln!("infer_shapes: {} shape corrections, {} I32 evaluations", shape_changes, i32_evals);
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn set_shape(model: &mut PspModel, id: usize, shape: Vec<usize>) -> usize {
    let old = &model.graph.tensor(id).shape;
    if *old != shape {
        eprintln!("  t{}: {:?} -> {:?}", id, old, shape);
        model.graph.tensor_mut(id).shape = shape;
        1
    } else {
        0
    }
}

/// When conv_dim fails (stale padding after upstream shape changes), we can't
/// recompute H/W but can still propagate the batch dimension.
fn propagate_batch(model: &mut PspModel, input: usize, output: usize) -> usize {
    let in_s = &model.graph.tensor(input).shape;
    let out_s = &model.graph.tensor(output).shape;
    if !out_s.is_empty() && !in_s.is_empty() && out_s[0] != in_s[0] {
        let mut shape = out_s.clone();
        shape[0] = in_s[0];
        set_shape(model, output, shape)
    } else {
        0
    }
}

fn conv_dim(input: usize, kernel: usize, stride: usize, pad_before: usize, pad_after: usize) -> usize {
    let padded = input + pad_before + pad_after;
    if padded < kernel {
        return 0; // degenerate case
    }
    (padded - kernel) / stride + 1
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let len = a.len().max(b.len());
    let mut result = vec![1; len];
    for i in 0..len {
        let da = if i < len - a.len() { 1 } else { a[i - (len - a.len())] };
        let db = if i < len - b.len() { 1 } else { b[i - (len - b.len())] };
        result[i] = da.max(db);
    }
    result
}

/// Resolve the output shape for an I32 Reshape during evaluation.
/// Priority: TFLite output if element count matches, then shape_tensor, then builtin_shape, then flat.
fn resolve_i32_reshape_shape(
    model: &PspModel,
    output: usize,
    in_size: usize,
    shape_tensor: &Option<usize>,
    builtin_shape: &Option<Vec<i32>>,
) -> Vec<usize> {
    // Priority 1: TFLite output shape if element count matches
    let tflite_shape = &model.graph.tensor(output).shape;
    let tflite_elems: usize = tflite_shape.iter().product();
    if tflite_elems == in_size {
        return tflite_shape.clone();
    }
    // Priority 2: shape_tensor
    if let Some(st) = shape_tensor {
        if let Some(dims) = model.read_i32_const(*st) {
            if let Some(shape) = resolve_reshape(&dims, in_size) {
                return shape;
            }
        }
    }
    // Priority 3: builtin_shape
    if let Some(dims) = builtin_shape {
        if let Some(shape) = resolve_reshape(dims, in_size) {
            return shape;
        }
    }
    // Fallback: flat
    vec![in_size]
}

/// Multi-dimensional broadcast evaluation for I32 binary ops.
///
/// Computes the output shape via NumPy-style broadcasting rules, then
/// evaluates the op element-by-element using proper multi-dim indexing.
fn eval_broadcast_i32(
    a: &[i32], a_shape: &[usize],
    b: &[i32], b_shape: &[usize],
    op: super::psp::BinaryOp,
) -> (Vec<i32>, Vec<usize>) {
    use super::psp::BinaryOp;
    let op_fn: fn(i32, i32) -> i32 = match op {
        BinaryOp::Add => |a, b| a + b,
        BinaryOp::Mul => |a, b| a * b,
        BinaryOp::Sub => |a, b| a - b,
        BinaryOp::Div => |a, b| a / b,
        BinaryOp::FloorDiv => |a, b| a.div_euclid(b),
        BinaryOp::Max => |a, b| a.max(b),
        BinaryOp::Pow => |a, b| a.pow(b as u32),
    };

    let out_shape = broadcast_shape(a_shape, b_shape);
    let out_size: usize = out_shape.iter().product();

    // Compute strides for indexing into a and b
    let a_strides = broadcast_strides(a_shape, &out_shape);
    let b_strides = broadcast_strides(b_shape, &out_shape);

    let result: Vec<i32> = (0..out_size).map(|flat| {
        let mut a_idx = 0usize;
        let mut b_idx = 0usize;
        let mut rem = flat;
        for d in (0..out_shape.len()).rev() {
            let coord = rem % out_shape[d];
            rem /= out_shape[d];
            a_idx += coord * a_strides[d];
            b_idx += coord * b_strides[d];
        }
        op_fn(a[a_idx], b[b_idx])
    }).collect();

    (result, out_shape)
}

/// Compute strides for a tensor broadcast into `out_shape`.
/// Dimensions of size 1 in `shape` get stride 0 (broadcast), others get
/// the standard row-major stride.
fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let rank = out_shape.len();
    let offset = rank - shape.len();
    let mut strides = vec![0usize; rank];
    let mut stride = 1usize;
    for d in (0..rank).rev() {
        let dim = if d < offset { 1 } else { shape[d - offset] };
        strides[d] = if dim == 1 { 0 } else { stride };
        stride *= dim;
    }
    strides
}

/// Compute Rfft output shape preserving TFLite's output rank.
///
/// The fuse pass absorbs both pre- and post-Rfft Reshapes, so the fused Rfft
/// operates on 3D input and targets TFLite's 3D output. When upstream
/// corrections change the batch dim, expand a singleton to hold the corrected
/// `batch` while keeping the last dim as `freq_bins`.
fn rfft_output_shape(tflite_out: &[usize], batch: usize, freq_bins: usize) -> Option<Vec<usize>> {
    if tflite_out.is_empty() {
        return None;
    }
    // Last dim should be freq_bins (or close to it — keep as is)
    let mut shape = tflite_out.to_vec();
    let last = shape.len() - 1;
    shape[last] = freq_bins;
    // Remaining dims must multiply to batch
    let current_batch: usize = shape[..last].iter().product();
    if current_batch == batch {
        return Some(shape);
    }
    // Find a singleton in the batch dims to expand
    if batch % current_batch != 0 || current_batch == 0 {
        return None;
    }
    let ratio = batch / current_batch;
    for i in 0..last {
        if shape[i] == 1 {
            shape[i] = ratio;
            if shape[..last].iter().product::<usize>() == batch {
                return Some(shape);
            }
            shape[i] = 1; // restore and try next
        }
    }
    None
}

/// Rank-preserving shape correction for Reshape ops.
///
/// TFLite's output *rank* is always correct — it knows whether a Reshape is
/// an unsqueeze, a flatten, etc. But when upstream shape corrections change
/// the element count, TFLite's specific dim sizes become stale. This function
/// preserves the rank by expanding exactly one singleton dim to absorb the
/// growth.
///
/// Example: input grew from `[1,2048]` to `[511,2048]`. TFLite's output was
/// `[1,1,1,2048]` (unsqueeze). We expand dim 1 → 511, giving `[1,511,1,2048]`.
///
/// For disambiguation, dims are matched from the END (trailing dims like
/// channels/fft_length are typically fixed), then the leftmost unmatched
/// singleton is expanded.
fn expand_singleton(in_shape: &[usize], out_shape: &[usize], in_size: usize) -> Option<Vec<usize>> {
    let out_size: usize = out_shape.iter().product();
    if out_size == 0 || in_size % out_size != 0 {
        return None;
    }
    let ratio = in_size / out_size;
    if ratio <= 1 {
        return None;
    }

    // Find singleton dims in out_shape that could absorb the ratio
    let candidates: Vec<usize> = out_shape.iter().enumerate()
        .filter(|&(_, &d)| d == 1)
        .map(|(i, _)| i)
        .collect();

    if candidates.len() == 1 {
        let mut shape = out_shape.to_vec();
        shape[candidates[0]] = ratio;
        return Some(shape);
    }

    // Multiple singletons: match dims from the end to find which singleton
    // should absorb the ratio. E.g., input [1,511,2048], output [1,1,1,2048]:
    // matching from end: 2048↔2048, then 511 must go somewhere. The dim at
    // position 1 in the output (which is 1) should become 511, giving
    // [1,511,1,2048].
    let mut shape = out_shape.to_vec();
    let mut in_idx = in_shape.len(); // walk backwards through input
    let mut out_idx = out_shape.len(); // walk backwards through output
    let mut expand_pos = None;
    while out_idx > 0 {
        out_idx -= 1;
        if in_idx > 0 && shape[out_idx] == in_shape[in_idx - 1] {
            // Dims match — consume both
            in_idx -= 1;
        } else if shape[out_idx] == 1 && in_idx > 0 && in_shape[in_idx - 1] != 1 {
            // Singleton in output, non-singleton in input — this is our candidate
            shape[out_idx] = in_shape[in_idx - 1];
            expand_pos = Some(out_idx);
            in_idx -= 1;
        }
        // else: singleton in output, either no more input dims or input also 1
    }

    if expand_pos.is_some() {
        let total: usize = shape.iter().product();
        if total == in_size {
            return Some(shape);
        }
    }

    // Fallback: expand the first singleton that gives correct element count
    for &c in &candidates {
        let mut shape = out_shape.to_vec();
        shape[c] = ratio;
        let total: usize = shape.iter().product();
        if total == in_size {
            return Some(shape);
        }
    }
    None
}

/// Resolve a reshape target shape, returning `None` if incompatible.
fn resolve_reshape(dims: &[i32], in_size: usize) -> Option<Vec<usize>> {
    let neg_idx = dims.iter().position(|&d| d == -1);
    if let Some(idx) = neg_idx {
        let known: usize = dims.iter().enumerate()
            .filter(|&(i, _)| i != idx)
            .map(|(_, d)| *d as usize)
            .product();
        if known == 0 || in_size % known != 0 {
            return None;
        }
        let mut shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        shape[idx] = in_size / known;
        Some(shape)
    } else {
        let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let total: usize = shape.iter().product();
        if total != in_size {
            return None;
        }
        Some(shape)
    }
}
