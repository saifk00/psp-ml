use crate::ir::graph::{DType, TensorId, TensorKind};
use crate::ir::psp::{Activation, PoolType, PspModel, PspOp, ReduceOp};

use super::plan::*;

const VFPU_Q: usize = 4;

const fn ceil_vfpu_q(x: usize) -> usize {
    (x + VFPU_Q - 1) & !(VFPU_Q - 1)
}

struct Conv2dKernelParams {
    input: Tensor4d,
    filter: Tensor4d,
    bias: Option<TensorId>,
    output: Tensor4d,
    stride: [usize; 2],
    padding: [usize; 4],
    has_relu: bool,
    /// Per-output-channel dequant scales when the filter is int8.
    weight_scales: Option<TensorId>,
}

impl Conv2dKernelParams {
    fn is_vfpu_eligible(&self) -> bool {
        let [n, _, _, _] = self.input.shape;
        let [co, _, _, _] = self.filter.shape;
        let [_, ho, wo, _] = self.output.shape;
        let gemm_m = n * ho * wo;
        gemm_m % VFPU_Q == 0 && co % VFPU_Q == 0
    }
}

/**
 * Lower a PspModel into a CodegenPlan.
 *
 * If the graph was rewritten by `ir::stream::rewrite()`, the lowerer sees
 * a clean batch=1 graph — no streaming logic needed here.
 *
 * `streamed` is the set of constant tensors the caller decided to leave on
 * disk and read chunkwise at op time (see `memory_planner::streamed_weights`);
 * everything else is packed into the resident blob. Residency *policy* lives
 * in `generate_code`, which needs the resulting `blob_bytes` to decide — this
 * function only mechanises the decision.
 *
 * **Append-only contract.** The only mutations `lower` makes to `model` are
 * `append_constant_f32` calls (repacked VFPU weight matrices, FFT twiddles),
 * which push onto the ends of `model.model_data` and `model.graph.tensors`
 * and nothing else. `generate_code` relies on this to roll a speculative
 * lowering back by truncating both to their prior lengths; `lower_is_append_only`
 * guards it.
 */
pub fn lower(
    model: &mut PspModel,
    streamed: &std::collections::HashSet<TensorId>,
) -> Result<CodegenPlan, String> {
    if model.graph.inputs.len() != 1 {
        return Err(format!(
            "Expected 1 input tensor, found {}",
            model.graph.inputs.len()
        ));
    }
    if model.graph.outputs.len() != 1 {
        return Err(format!(
            "Expected 1 output tensor, found {}",
            model.graph.outputs.len()
        ));
    }

    let input_id = model.graph.inputs[0];
    let output_id = model.graph.outputs[0];
    let input_size = model.graph.tensor(input_id).shape.iter().product::<usize>();
    let output_size = model
        .graph
        .tensor(output_id)
        .shape
        .iter()
        .product::<usize>();

    // TODO: expose as compiler flag
    let use_vfpu_conv2d = true;

    let mut allocs = lower_allocs(model, streamed)?;
    let ops = lower_ops(model, use_vfpu_conv2d, &mut allocs, streamed)?;
    prune_dead_constants(&mut allocs, &ops);

    // Compute blob size AFTER lower_ops, which may append twiddle constants
    let blob_bytes = model.model_data.len();
    let blob_floats = blob_bytes / std::mem::size_of::<f32>();

    Ok(CodegenPlan {
        input_id,
        output_id,
        input_size,
        output_size,
        blob_bytes,
        blob_floats,
        allocs,
        ops,
        arena: None, // filled in by generate_code() after lowering
        stream: None, // filled in by generate_code() from stream rewrite
    })
}

/// Drop constants no kernel call or scratch load actually reads.
///
/// `lower_allocs` runs before `lower_ops` and so admits every constant the IR
/// referenced. Lowering can then make one dead — the VFPU FC path replaces its
/// weight matrix with a repacked copy — and without this the blob would carry
/// both (an extra 394 KB per mel projection on BirdNET).
fn prune_dead_constants(allocs: &mut Vec<TensorAlloc>, ops: &[OpPlan]) {
    let mut live: std::collections::HashSet<TensorId> = std::collections::HashSet::new();
    for op in ops {
        for scratch in &op.scratch {
            if let Some(load) = &scratch.load_from {
                live.insert(load.source);
                if let Some(extra) = load.copy.extra_tensor_refs() {
                    live.insert(extra);
                }
            }
        }
        for sub in &op.sub_ops {
            for kernel in &sub.kernels {
                for r in super::arena::extract_tensor_refs(kernel) {
                    live.insert(r.tensor_id);
                }
            }
        }
    }
    allocs.retain(|a| match a {
        // Streamed constants are reached through their file offset, not through
        // a tensor ref (`extract_tensor_refs` omits them on purpose), and their
        // presence is also what makes render emit the streaming helper.
        TensorAlloc::Constant { streamed: true, .. } => true,
        TensorAlloc::Constant { id, .. } => live.contains(id),
        _ => true,
    });
}

fn lower_allocs(
    model: &PspModel,
    streamed: &std::collections::HashSet<TensorId>,
) -> Result<Vec<TensorAlloc>, String> {
    let mut allocs = Vec::new();
    let sz = std::mem::size_of::<f32>();

    // Collect TensorIds actually referenced by ops, graph inputs, or graph outputs.
    // Tensors made dead by buffer aliasing (e.g. reshape output) will be skipped.
    let mut referenced: std::collections::HashSet<TensorId> = std::collections::HashSet::new();
    for id in &model.graph.inputs {
        referenced.insert(*id);
    }
    for id in &model.graph.outputs {
        referenced.insert(*id);
    }
    for op in &model.graph.ops {
        for id in op.inputs() {
            referenced.insert(id);
        }
        for id in op.all_outputs() {
            referenced.insert(id);
        }
    }

    for tensor in &model.graph.tensors {
        match &tensor.kind {
            TensorKind::Constant { offset, len } => {
                if !referenced.contains(&tensor.id) {
                    continue;
                }
                if offset % sz != 0 {
                    return Err(format!(
                        "Tensor {} constant offset {} not 4-byte aligned",
                        tensor.id, offset
                    ));
                }
                if len % sz != 0 {
                    return Err(format!(
                        "Tensor {} constant len {} not 4-byte aligned",
                        tensor.id, len
                    ));
                }
                allocs.push(TensorAlloc::Constant {
                    id: tensor.id,
                    float_offset: offset / sz,
                    float_len: len / sz,
                    dtype: tensor.dtype,
                    streamed: streamed.contains(&tensor.id),
                });
            }
            TensorKind::Intermediate => {
                if !referenced.contains(&tensor.id) {
                    continue;
                }
                let size = tensor.shape.iter().product::<usize>();
                allocs.push(TensorAlloc::Intermediate {
                    id: tensor.id,
                    size,
                });
            }
            TensorKind::Output => {
                let size = tensor.shape.iter().product::<usize>();
                allocs.push(TensorAlloc::Output {
                    id: tensor.id,
                    size,
                });
            }
            TensorKind::Input => {}
        }
    }

    Ok(allocs)
}

fn lower_ops(
    model: &mut PspModel,
    use_vfpu_conv2d: bool,
    allocs: &mut Vec<TensorAlloc>,
    streamed: &std::collections::HashSet<TensorId>,
) -> Result<Vec<OpPlan>, String> {
    let mut ops = Vec::new();

    for i in 0..model.graph.ops.len() {
        let op = model.graph.ops[i].clone();

        // FullyConnected with a big batch: repack the weights at compile time
        // and use the blocked VFPU GEMM. Handled here, before the immutable
        // `graph` borrow below, because it appends a constant to `model`.
        if let PspOp::FullyConnected {
            input,
            weights,
            bias,
            output,
            fused_activation,
        } = &op
        {
            if let Some(plan) = lower_fc_vfpu(
                model, allocs, streamed, *input, *weights, *bias, *output, fused_activation,
            )? {
                ops.push(plan);
                continue;
            }
        }

        // Handle RFFT separately: it needs &mut model to append twiddle constants
        if let PspOp::Rfft {
            input,
            output,
            fft_length,
        } = &op
        {
            ops.push(lower_rfft(model, allocs, i, *input, *output, *fft_length)?);
            continue;
        }

        let graph = &model.graph;
        let plan = match &op {
            PspOp::Conv2d {
                input,
                weights,
                bias,
                output,
                weight_scales,
                params,
            } => {
                let in_shape = &graph.tensor(*input).shape;
                let w_shape = &graph.tensor(*weights).shape;
                if graph.tensor(*weights).dtype == DType::I8 && weight_scales.is_none() {
                    return Err(format!(
                        "Op {i}: int8 conv weights without scales (quant rewrite not run?)"
                    ));
                }
                let out_shape = &graph.tensor(*output).shape;

                if in_shape.len() != 4 || w_shape.len() != 4 || out_shape.len() != 4 {
                    return Err(format!(
                        "Op {i}: Conv2d expects 4D tensors (input={}, weights={}, output={})",
                        in_shape.len(),
                        w_shape.len(),
                        out_shape.len()
                    ));
                }

                if let Some(Activation::Relu6) = params.fused_activation {
                    return Err(format!("Op {i}: Relu6 not supported for Conv2d"));
                }

                let in4 = Tensor4d {
                    id: *input,
                    shape: [in_shape[0], in_shape[1], in_shape[2], in_shape[3]],
                };
                let w4 = Tensor4d {
                    id: *weights,
                    shape: [w_shape[0], w_shape[1], w_shape[2], w_shape[3]],
                };
                let out4 = Tensor4d {
                    id: *output,
                    shape: [out_shape[0], out_shape[1], out_shape[2], out_shape[3]],
                };
                let stride = [params.stride_h, params.stride_w];
                let padding = [
                    params.pad_top,
                    params.pad_bottom,
                    params.pad_left,
                    params.pad_right,
                ];
                let has_relu = matches!(params.fused_activation, Some(Activation::Relu));
                let conv2d_params = Conv2dKernelParams {
                    input: in4,
                    filter: w4,
                    bias: *bias,
                    output: out4,
                    stride,
                    padding,
                    has_relu,
                    weight_scales: *weight_scales,
                };
                if use_vfpu_conv2d {
                    lower_conv2d_vfpu(conv2d_params)
                } else {
                    lower_conv2d_naive(conv2d_params)
                }
            }

            PspOp::DepthwiseConv2d {
                input,
                weights,
                bias,
                output,
                params,
            } => {
                let in_shape = &graph.tensor(*input).shape;
                let w_shape = &graph.tensor(*weights).shape;
                let out_shape = &graph.tensor(*output).shape;

                if in_shape.len() != 4 || w_shape.len() != 4 || out_shape.len() != 4 {
                    return Err(format!(
                        "Op {i}: DepthwiseConv2d expects 4D tensors (input={}, weights={}, output={})",
                        in_shape.len(), w_shape.len(), out_shape.len()
                    ));
                }

                let in4 = Tensor4d {
                    id: *input,
                    shape: [in_shape[0], in_shape[1], in_shape[2], in_shape[3]],
                };
                let w4 = Tensor4d {
                    id: *weights,
                    shape: [w_shape[0], w_shape[1], w_shape[2], w_shape[3]],
                };
                let out4 = Tensor4d {
                    id: *output,
                    shape: [out_shape[0], out_shape[1], out_shape[2], out_shape[3]],
                };

                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: "depthwise_conv2d".into(),
                        kernels: vec![KernelCall::DepthwiseConv2d {
                            input: in4,
                            filter: w4,
                            bias: *bias,
                            stride: [params.stride_h, params.stride_w],
                            padding: [
                                params.pad_top,
                                params.pad_bottom,
                                params.pad_left,
                                params.pad_right,
                            ],
                            output: out4,
                        }],
                    }],
                }
            }

            PspOp::FullyConnected {
                input,
                weights,
                bias,
                output,
                fused_activation,
            } => {
                let in_shape = &graph.tensor(*input).shape;
                let in_features = *in_shape.last().unwrap_or(&0);
                let batch_size: usize = in_shape[..in_shape.len() - 1].iter().product::<usize>().max(1);
                let out_features = *graph.tensor(*weights).shape.first().unwrap_or(&0);
                let has_relu = matches!(fused_activation.fused_activation, Some(Activation::Relu));

                if let Some(Activation::Relu6) = fused_activation.fused_activation {
                    return Err(format!("Op {i}: Relu6 not supported for FullyConnected"));
                }

                let name = if has_relu {
                    "fully_connected_relu"
                } else {
                    "fully_connected"
                };

                if streamed.contains(weights) {
                    if batch_size != 1 {
                        return Err(format!(
                            "Op {i}: streamed FullyConnected requires batch 1, got {batch_size}"
                        ));
                    }
                    // Row chunks sized to the hostfs sweet spot (~64 KiB —
                    // measured 21.5 MB/s at 64 KiB vs 3.9 MB/s at 4 KiB).
                    let chunk_rows = (65536 / (in_features * 4)).max(1);
                    OpPlan {
                        scratch: vec![ScratchBuffer {
                            size: chunk_rows * in_features,
                            load_from: None,
                        }],
                        sub_ops: vec![SubOpPlan {
                            name: format!("{name}_streamed"),
                            kernels: vec![KernelCall::FullyConnectedStreamed {
                                input: *input,
                                in_features,
                                weights: *weights,
                                bias: *bias,
                                output: *output,
                                out_features,
                                has_relu,
                                scratch: 0,
                                chunk_rows,
                            }],
                        }],
                    }
                } else {
                    OpPlan {
                        scratch: vec![],
                        sub_ops: vec![SubOpPlan {
                            name: name.into(),
                            kernels: vec![KernelCall::FullyConnected {
                                input: *input,
                                in_features,
                                weights: *weights,
                                bias: *bias,
                                output: *output,
                                out_features,
                                has_relu,
                                batch_size,
                            }],
                        }],
                    }
                }
            }

            PspOp::Pool2d {
                pool_type,
                input,
                output,
                filter,
                stride,
                padding,
            } => {
                let in_shape = &graph.tensor(*input).shape;
                let out_shape = &graph.tensor(*output).shape;

                if in_shape.len() != 4 || out_shape.len() != 4 {
                    return Err(format!(
                        "Op {i}: Pool2d expects 4D tensors (input={}, output={})",
                        in_shape.len(),
                        out_shape.len()
                    ));
                }

                let name = match pool_type {
                    PoolType::Max => "max_pool2d",
                    PoolType::Average => "average_pool2d",
                };

                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: name.into(),
                        kernels: vec![KernelCall::Pool2d {
                            input: Tensor4d {
                                id: *input,
                                shape: [in_shape[0], in_shape[1], in_shape[2], in_shape[3]],
                            },
                            output: Tensor4d {
                                id: *output,
                                shape: [out_shape[0], out_shape[1], out_shape[2], out_shape[3]],
                            },
                            filter: *filter,
                            stride: *stride,
                            padding: *padding,
                            pool_type: *pool_type,
                        }],
                    }],
                }
            }

            PspOp::Reshape { input, output, .. }
            | PspOp::Squeeze { input, output, .. }
            | PspOp::ExpandDims { input, output, .. } => OpPlan {
                scratch: vec![],
                sub_ops: vec![SubOpPlan {
                    name: "reshape".into(),
                    kernels: vec![KernelCall::Reshape {
                        input: *input,
                        output: *output,
                    }],
                }],
            },

            PspOp::ElementWise {
                op,
                input_a,
                input_b,
                output,
            } => {
                let b_len = graph.tensor(*input_b).shape.iter().product::<usize>();
                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: op.name().into(),
                        kernels: vec![KernelCall::ElementWise {
                            op: *op,
                            input_a: *input_a,
                            input_b: *input_b,
                            output: *output,
                            b_len,
                        }],
                    }],
                }
            }

            PspOp::UnaryElementWise { op, input, output } => OpPlan {
                scratch: vec![],
                sub_ops: vec![SubOpPlan {
                    name: op.name().into(),
                    kernels: vec![KernelCall::UnaryElementWise {
                        op: *op,
                        input: *input,
                        output: *output,
                    }],
                }],
            },

            PspOp::Swish { input, output } => OpPlan {
                scratch: vec![],
                sub_ops: vec![SubOpPlan {
                    name: "swish".into(),
                    kernels: vec![KernelCall::Swish {
                        input: *input,
                        output: *output,
                    }],
                }],
            },

            PspOp::FakeQuant { input, output } => {
                let (scale, zero_point) = graph
                    .tensor(*output)
                    .quant
                    .as_ref()
                    .ok_or_else(|| format!("Op {i}: FakeQuant output missing quantization"))?
                    .scalar()
                    .map_err(|e| format!("Op {i}: {e}"))?;
                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: "fake_quant".into(),
                        kernels: vec![KernelCall::FakeQuant {
                            input: *input,
                            output: *output,
                            scale,
                            zero_point,
                        }],
                    }],
                }
            }

            PspOp::Dequantize { .. } => {
                return Err(format!(
                    "Op {i}: Dequantize should have been eliminated by the quant rewrite"
                ));
            }

            PspOp::Softmax { .. } => {
                return Err(format!("Op {i}: Softmax kernel not yet implemented"));
            }

            PspOp::Reduce {
                op,
                input,
                axes: _,
                output,
            } => {
                let in_shape = &graph.tensor(*input).shape;
                let out_shape = &graph.tensor(*output).shape;

                // Compute batch size: leading dim that matches between in/out
                let batch_size: usize = if out_shape.len() >= 2
                    && in_shape.first() == out_shape.first()
                    && *in_shape.first().unwrap_or(&1) > 1
                {
                    *out_shape.first().unwrap_or(&1)
                } else {
                    1
                };
                let in_total: usize = in_shape.iter().product();
                let out_total: usize = out_shape.iter().product();
                let frame_in_size = in_total / batch_size.max(1);
                let frame_out_size = out_total / batch_size.max(1);

                if *op == ReduceOp::Mean {
                    let last_matches = out_shape.last() == in_shape.last();
                    let others_are_one = out_shape.iter().rev().skip(1)
                        .all(|&d| d == 1 || d == batch_size);
                    if !last_matches || !others_are_one {
                        return Err(format!(
                            "Op {i}: reduce_mean_hw requires output to keep only the last dim \
                             (input shape={:?}, output shape={:?})",
                            in_shape, out_shape,
                        ));
                    }
                }
                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: op.name().into(),
                        kernels: vec![KernelCall::Reduce {
                            op: *op,
                            input: *input,
                            output: *output,
                            batch_size,
                            frame_in_size,
                            frame_out_size,
                        }],
                    }],
                }
            }

            PspOp::ReverseV2 {
                input,
                axis,
                output,
            } => {
                let in_shape = graph.tensor(*input).shape.clone();
                let ndim = in_shape.len();

                if ndim > 4 {
                    return Err(format!(
                        "Op {i}: ReverseV2 only supports up to 4D tensors (got {}D)",
                        ndim
                    ));
                }

                let axis_tensor = graph.tensor(*axis);
                let axis_val = if let TensorKind::Constant { offset, .. } = axis_tensor.kind {
                    let val = i32::from_le_bytes(
                        model.model_data[offset..offset + 4].try_into().unwrap(),
                    );
                    let a = if val < 0 {
                        (ndim as i32 + val) as usize
                    } else {
                        val as usize
                    };
                    if a >= ndim {
                        return Err(format!(
                            "Op {i}: ReverseV2 axis {} out of bounds for {}D tensor",
                            val, ndim
                        ));
                    }
                    a
                } else {
                    return Err(format!("Op {i}: ReverseV2 requires constant axis tensor"));
                };

                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: "reverse_v2".into(),
                        kernels: vec![KernelCall::ReverseV2 {
                            input: *input,
                            output: *output,
                            input_shape: in_shape,
                            axis: axis_val,
                        }],
                    }],
                }
            }

            PspOp::Transpose {
                input,
                perm,
                output,
            } => {
                let in_shape = graph.tensor(*input).shape.clone();
                let out_shape = graph.tensor(*output).shape.clone();
                let ndim = in_shape.len();

                if ndim > 4 {
                    return Err(format!(
                        "Op {i}: Transpose only supports up to 4D tensors (got {}D)",
                        ndim
                    ));
                }

                let perm_tensor = graph.tensor(*perm);
                let perm_vals = if let TensorKind::Constant { offset, len } = perm_tensor.kind {
                    let vals: Vec<i32> = model.model_data[offset..offset + len]
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    if vals.len() != ndim {
                        return Err(format!(
                            "Op {i}: Transpose perm length {} doesn't match input rank {}",
                            vals.len(),
                            ndim
                        ));
                    }
                    vals.iter().map(|&v| v as usize).collect::<Vec<_>>()
                } else {
                    return Err(format!("Op {i}: Transpose requires constant perm tensor"));
                };

                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: "transpose".into(),
                        kernels: vec![KernelCall::Transpose {
                            input: *input,
                            output: *output,
                            input_shape: in_shape,
                            output_shape: out_shape,
                            perm: perm_vals,
                        }],
                    }],
                }
            }

            PspOp::Pad {
                input,
                paddings,
                output,
            } => {
                let in_shape = &graph.tensor(*input).shape;
                let out_shape = &graph.tensor(*output).shape;

                if in_shape.len() != 4 || out_shape.len() != 4 {
                    return Err(format!(
                        "Op {i}: Pad expects 4D tensors (input={}, output={})",
                        in_shape.len(),
                        out_shape.len()
                    ));
                }

                // Read the padding constant at compile time: [4, 2] INT32
                let pad_tensor = graph.tensor(*paddings);
                let padding = if let TensorKind::Constant { offset, len } = pad_tensor.kind {
                    let vals: Vec<i32> = model.model_data[offset..offset + len]
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    if vals.len() != 8 {
                        return Err(format!(
                            "Op {i}: Pad expects [4,2] padding tensor, got {} values",
                            vals.len()
                        ));
                    }
                    [
                        [vals[0] as usize, vals[1] as usize],
                        [vals[2] as usize, vals[3] as usize],
                        [vals[4] as usize, vals[5] as usize],
                        [vals[6] as usize, vals[7] as usize],
                    ]
                } else {
                    return Err(format!("Op {i}: Pad requires constant padding tensor"));
                };

                let in4 = Tensor4d {
                    id: *input,
                    shape: [in_shape[0], in_shape[1], in_shape[2], in_shape[3]],
                };
                let out4 = Tensor4d {
                    id: *output,
                    shape: [out_shape[0], out_shape[1], out_shape[2], out_shape[3]],
                };

                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: "pad".into(),
                        kernels: vec![KernelCall::Pad {
                            input: in4,
                            output: out4,
                            padding,
                        }],
                    }],
                }
            }

            PspOp::Rfft { .. } => unreachable!("handled above"),

            PspOp::StridedSlice {
                input,
                begin,
                end,
                strides,
                output,
                begin_mask,
                end_mask,
                shrink_axis_mask,
            } => {
                let in_shape = graph.tensor(*input).shape.clone();
                let out_shape = graph.tensor(*output).shape.clone();

                let begin_vals = model.read_i32_const(*begin)
                    .ok_or_else(|| format!("Op {i}: StridedSlice requires constant begin"))?;
                let end_vals = model.read_i32_const(*end)
                    .ok_or_else(|| format!("Op {i}: StridedSlice requires constant end"))?;
                let stride_vals = model.read_i32_const(*strides)
                    .ok_or_else(|| format!("Op {i}: StridedSlice requires constant strides"))?;

                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: "strided_slice".into(),
                        kernels: vec![KernelCall::StridedSlice {
                            input: *input,
                            output: *output,
                            input_shape: in_shape,
                            output_shape: out_shape,
                            begin: begin_vals,
                            end: end_vals,
                            strides: stride_vals,
                            begin_mask: *begin_mask,
                            end_mask: *end_mask,
                            shrink_axis_mask: *shrink_axis_mask,
                        }],
                    }],
                }
            }

            PspOp::Gather {
                input,
                indices,
                output,
                axis,
            } => {
                let in_shape = graph.tensor(*input).shape.clone();
                let out_shape = graph.tensor(*output).shape.clone();

                // Validate indices is constant
                if !matches!(graph.tensor(*indices).kind, TensorKind::Constant { .. }) {
                    return Err(format!("Op {i}: Gather requires constant indices tensor"));
                }

                let axis_val = if *axis < 0 {
                    (in_shape.len() as i32 + *axis) as usize
                } else {
                    *axis as usize
                };

                let indices_len = graph.tensor(*indices).shape.iter().product::<usize>();

                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: "gather".into(),
                        kernels: vec![KernelCall::Gather {
                            input: *input,
                            output: *output,
                            indices: *indices,
                            indices_len,
                            input_shape: in_shape,
                            output_shape: out_shape,
                            axis: axis_val,
                        }],
                    }],
                }
            }

            PspOp::Concatenation {
                inputs,
                output,
                axis,
            } => {
                let out_shape = graph.tensor(*output).shape.clone();
                let in_shapes: Vec<Vec<usize>> = inputs
                    .iter()
                    .map(|id| graph.tensor(*id).shape.clone())
                    .collect();

                let axis_val = if *axis < 0 {
                    (out_shape.len() as i32 + *axis) as usize
                } else {
                    *axis as usize
                };

                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: "concatenation".into(),
                        kernels: vec![KernelCall::Concatenation {
                            inputs: inputs.clone(),
                            output: *output,
                            input_shapes: in_shapes,
                            output_shape: out_shape,
                            axis: axis_val,
                        }],
                    }],
                }
            }

            PspOp::Shape { .. }
            | PspOp::Pack { .. }
            | PspOp::Range { .. }
            | PspOp::SplitV { .. }
            | PspOp::Cast { .. }
            | PspOp::Rfft2d { .. } => {
                return Err(format!(
                    "Op {i}: {:?} should have been constant-folded or fused",
                    op
                ));
            }
        };

        ops.push(plan);
    }

    Ok(ops)
}

fn lower_conv2d_naive(conv2d: Conv2dKernelParams) -> OpPlan {
    let base = if conv2d.has_relu {
        "conv2d_relu"
    } else {
        "conv2d"
    };
    let name = if conv2d.weight_scales.is_some() {
        format!("{base}_q8")
    } else {
        base.to_string()
    };
    OpPlan {
        scratch: vec![],
        sub_ops: vec![SubOpPlan {
            name,
            kernels: vec![KernelCall::Conv2d {
                input: conv2d.input,
                filter: conv2d.filter,
                bias: conv2d.bias,
                stride: conv2d.stride,
                padding: conv2d.padding,
                output: conv2d.output,
                has_relu: conv2d.has_relu,
                weight_scales: conv2d.weight_scales,
            }],
        }],
    }
}

fn lower_conv2d_vfpu(conv2d: Conv2dKernelParams) -> OpPlan {
    let [n, _, _, ci] = conv2d.input.shape;
    let [co, kh, kw, _] = conv2d.filter.shape;
    let [_, ho, wo, _] = conv2d.output.shape;

    let gemm_m = n * ho * wo;
    let gemm_k = kh * kw * ci;
    // im2col writes rows padded to a multiple of 4 and leaves the pad columns
    // untouched; passing the logical `gemm_k` alongside this stride means the
    // GEMM never reads them.
    let k_padded = ceil_vfpu_q(gemm_k);
    let m_padded = ceil_vfpu_q(gemm_m);

    let scratch = vec![
        // 0: im2col output
        ScratchBuffer {
            size: m_padded * k_padded,
            load_from: None,
        },
        // 1: weights repacked into the micro-kernel's B layout
        ScratchBuffer {
            size: psp_rt::kernels::gemm_bp_len(co, gemm_k),
            load_from: Some(ScratchLoad {
                source: conv2d.filter.id,
                copy: match conv2d.weight_scales {
                    Some(scales) => CopyStrategy::PackBDequantI8 {
                        n: co,
                        k: gemm_k,
                        scales,
                    },
                    None => CopyStrategy::PackB { n: co, k: gemm_k },
                },
            }),
        },
        // 2/3: GEMM packing and accumulator scratch
        ScratchBuffer {
            size: psp_rt::kernels::gemm_ap_len(GEMM_MC, GEMM_KC),
            load_from: None,
        },
        ScratchBuffer {
            size: psp_rt::kernels::gemm_cp_len(GEMM_MC, co),
            load_from: None,
        },
    ];

    let mut sub_ops = vec![
        SubOpPlan {
            name: "im2col".into(),
            kernels: vec![KernelCall::Im2colPadded {
                input: conv2d.input,
                kernel_size: [kh, kw],
                stride: conv2d.stride,
                padding: conv2d.padding,
                output_hw: [ho, wo],
                output: 0,
            }],
        },
        SubOpPlan {
            name: "gemm_vfpu".into(),
            kernels: vec![KernelCall::GemmBtPacked {
                a: GemmOperand::Scratch(0),
                lda: k_padded,
                b: GemmOperand::Scratch(1),
                output: conv2d.output.id,
                m: gemm_m,
                k: gemm_k,
                n: co,
                ap: 2,
                cp: 3,
                mc: GEMM_MC,
                kc: GEMM_KC,
            }],
        },
    ];

    if conv2d.bias.is_some() || conv2d.has_relu {
        let name = if conv2d.has_relu {
            "bias_add_relu"
        } else {
            "bias_add"
        };
        let mut kernels = Vec::new();
        if let Some(bias_id) = conv2d.bias {
            kernels.push(KernelCall::BiasAdd {
                output: conv2d.output.id,
                bias: bias_id,
                rows: gemm_m,
                cols: co,
            });
        }
        if conv2d.has_relu {
            kernels.push(KernelCall::Relu {
                output: conv2d.output.id,
            });
        }
        sub_ops.push(SubOpPlan {
            name: name.into(),
            kernels,
        });
    }

    OpPlan { scratch, sub_ops }
}

/// L1 blocking factors for the VFPU GEMM, chosen by sweeping `examples/fc-bench`
/// on hardware: (32, 64) was a flat optimum, with everything within +/-8 in mc
/// or 2x in kc landing inside 3%.
const GEMM_MC: usize = 32;
const GEMM_KC: usize = 64;

/// Minimum batch size worth the pack/unpack overhead. Below this the batched
/// GEMV path wins; BirdNET's mel projections have batch 511.
const GEMM_MIN_BATCH: usize = 8;

/// Try to lower a FullyConnected to the blocked VFPU GEMM, repacking its
/// weights into the micro-kernel layout at compile time.
///
/// Returns `Ok(None)` when the op should fall through to the existing paths:
/// small batches, streamed weights (a different, I/O-bound shape), non-f32 or
/// non-constant weights.
#[allow(clippy::too_many_arguments)]
fn lower_fc_vfpu(
    model: &mut PspModel,
    allocs: &mut Vec<TensorAlloc>,
    streamed: &std::collections::HashSet<TensorId>,
    input: TensorId,
    weights: TensorId,
    bias: Option<TensorId>,
    output: TensorId,
    fused_activation: &crate::ir::psp::FullyConnectedParams,
) -> Result<Option<OpPlan>, String> {
    if streamed.contains(&weights) {
        return Ok(None);
    }
    let w = model.graph.tensor(weights);
    if w.dtype != DType::F32 {
        return Ok(None);
    }
    let TensorKind::Constant { offset, len } = w.kind else {
        return Ok(None);
    };

    let in_shape = &model.graph.tensor(input).shape;
    let k = *in_shape.last().unwrap_or(&0);
    let m: usize = in_shape[..in_shape.len() - 1].iter().product::<usize>().max(1);
    let n = *w.shape.first().unwrap_or(&0);
    if m < GEMM_MIN_BATCH || k == 0 || n == 0 {
        return Ok(None);
    }
    if len != n * k * 4 {
        return Ok(None);
    }

    let has_relu = match fused_activation.fused_activation {
        Some(Activation::Relu) => true,
        Some(Activation::Relu6) => return Err("Relu6 not supported for FullyConnected".into()),
        None => false,
    };

    // Repack B once, here, using the runtime crate's own packer so the two
    // cannot drift apart.
    let src: Vec<f32> = model.model_data[offset..offset + len]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut packed = vec![0.0f32; psp_rt::kernels::gemm_bp_len(n, k)];
    psp_rt::kernels::pack_b_panel(&src, &mut packed, n, k);
    let packed_id = append_constant_f32(model, allocs, vec![packed.len()], &packed);

    let scratch = vec![
        ScratchBuffer {
            size: psp_rt::kernels::gemm_ap_len(GEMM_MC, GEMM_KC),
            load_from: None,
        },
        ScratchBuffer {
            size: psp_rt::kernels::gemm_cp_len(GEMM_MC, n),
            load_from: None,
        },
    ];

    let mut sub_ops = vec![SubOpPlan {
        name: "gemm_vfpu".into(),
        kernels: vec![KernelCall::GemmBtPacked {
            a: GemmOperand::Tensor(input),
            lda: k,
            b: GemmOperand::Tensor(packed_id),
            output,
            m,
            k,
            n,
            ap: 0,
            cp: 1,
            mc: GEMM_MC,
            kc: GEMM_KC,
        }],
    }];

    if bias.is_some() || has_relu {
        let name = if has_relu { "bias_add_relu" } else { "bias_add" };
        let mut kernels = Vec::new();
        if let Some(bias_id) = bias {
            kernels.push(KernelCall::BiasAdd {
                output,
                bias: bias_id,
                rows: m,
                cols: n,
            });
        }
        if has_relu {
            kernels.push(KernelCall::Relu { output });
        }
        sub_ops.push(SubOpPlan {
            name: name.into(),
            kernels,
        });
    }

    Ok(Some(OpPlan { scratch, sub_ops }))
}

/// Append a float slice to model_data as a constant tensor, returning its TensorId.
///
/// This is the *only* way lowering mutates `model`, and it is strictly
/// append-only on both `model_data` and `graph.tensors` (`add_tensor` derives
/// ids from `tensors.len()`). `generate_code` rolls a speculative lowering back
/// by truncating both to their pre-lowering lengths — keep it that way.
fn append_constant_f32(
    model: &mut PspModel,
    allocs: &mut Vec<TensorAlloc>,
    shape: Vec<usize>,
    data: &[f32],
) -> TensorId {
    let sz = std::mem::size_of::<f32>();
    // Align to 4 bytes (should already be aligned since model_data is byte vec)
    let offset = model.model_data.len();
    let len = data.len() * sz;
    for &val in data {
        model.model_data.extend_from_slice(&val.to_le_bytes());
    }
    let id = model
        .graph
        .add_tensor(shape, DType::F32, TensorKind::Constant { offset, len });
    allocs.push(TensorAlloc::Constant {
        id,
        float_offset: offset / sz,
        float_len: data.len(),
        dtype: DType::F32,
        streamed: false,
    });
    id
}

/// Lower a fused RFFT op into a single batched kernel call
/// (per-frame pack → butterfly stages → unpack, looped over all frames).
fn lower_rfft(
    model: &mut PspModel,
    allocs: &mut Vec<TensorAlloc>,
    _op_idx: usize,
    input: TensorId,
    output: TensorId,
    fft_length: usize,
) -> Result<OpPlan, String> {
    let n = fft_length;
    let n_complex = n / 2;

    // Batched over all leading dims: input is [.., frames, n] flattened.
    let in_elems: usize = model.graph.tensor(input).shape.iter().product();
    if in_elems % n != 0 {
        return Err(format!(
            "Rfft: input size {in_elems} not a multiple of fft_length {n}"
        ));
    }
    let frames = in_elems / n;

    // Number of butterfly stages = log2(n_complex)
    let num_stages = n_complex.trailing_zeros() as usize;

    // Scratch buffer: n floats (n_complex interleaved complex pairs), reused
    // across frames.
    let scratch = vec![ScratchBuffer {
        size: n,
        load_from: None,
    }];

    // Stage twiddles, concatenated in stage order: stage s holds 2^s complex
    // entries exp(-2πi·j / 2^(s+1)) at float offset (2^s - 1) * 2.
    // Split layout: per stage, all cosines then all sines. The kernel keeps
    // real and imaginary parts in separate arrays so every butterfly term is a
    // plain elementwise multiply — see the FFT section of psp-rt's kernels.
    let mut stage_twiddles = Vec::new();
    for stage in 0..num_stages {
        let half_size = 1usize << stage;
        let start = stage_twiddles.len();
        for j in 0..half_size {
            let angle = -2.0 * std::f64::consts::PI * (j as f64) / (2.0 * half_size as f64);
            stage_twiddles.push(angle.cos() as f32);
        }
        for j in 0..half_size {
            let angle = -2.0 * std::f64::consts::PI * (j as f64) / (2.0 * half_size as f64);
            stage_twiddles.push(angle.sin() as f32);
        }
        // Pad so the next stage starts 4-float aligned; the kernel loads these
        // with `lv.q` and the natural packing lands at 8 mod 16 bytes.
        let block = psp_rt::kernels::stage_tw_block(stage);
        stage_twiddles.resize(start + block, 0.0);
    }
    debug_assert_eq!(stage_twiddles.len(), psp_rt::kernels::stage_tw_len(n));
    let stage_tw_id =
        append_constant_f32(model, allocs, vec![stage_twiddles.len()], &stage_twiddles);

    // Unpack twiddles: W_N^k = exp(2πi·k/N) stored as [cos, -sin] for k = 1..n_complex-1
    let mut unpack_twiddles = Vec::with_capacity((n_complex - 1) * 2);
    for k in 1..n_complex {
        let angle = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
        unpack_twiddles.push(angle.cos() as f32);
    }
    for k in 1..n_complex {
        let angle = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
        unpack_twiddles.push(-(angle.sin() as f32));
    }
    let unpack_tw_id =
        append_constant_f32(model, allocs, vec![(n_complex - 1) * 2], &unpack_twiddles);

    Ok(OpPlan {
        scratch,
        sub_ops: vec![SubOpPlan {
            name: "rfft".into(),
            kernels: vec![KernelCall::RfftBatch {
                input,
                stage_twiddles: stage_tw_id,
                unpack_twiddles: unpack_tw_id,
                output,
                scratch: 0,
                n,
                frames,
            }],
        }],
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::{DType, Graph, TensorKind};
    use crate::ir::psp::{Conv2dParams, FullyConnectedParams, PspModel};

    /// Shadows `super::lower` so the tests below (none of which exercise weight
    /// streaming) keep reading as `lower(&mut model)`. An explicit item wins
    /// over the `use super::*` glob.
    fn lower(model: &mut PspModel) -> Result<CodegenPlan, String> {
        super::lower(model, &std::collections::HashSet::new())
    }

    /// Helper: build a minimal model with one Conv2d op for VFPU testing.
    /// Input: [1, H, W, CI], Filter: [CO, KH, KW, CI], Output: [1, OH, OW, CO]
    fn make_conv2d_model(
        h: usize,
        w: usize,
        ci: usize,
        co: usize,
        kh: usize,
        kw: usize,
        pad: usize,
        stride: usize,
        has_bias: bool,
        has_relu: bool,
    ) -> PspModel {
        let oh = (h + 2 * pad - kh) / stride + 1;
        let ow = (w + 2 * pad - kw) / stride + 1;

        let mut graph = Graph::new();

        let input = graph.add_tensor(vec![1, h, w, ci], DType::F32, TensorKind::Input);
        graph.inputs.push(input);

        let weight_floats = co * kh * kw * ci;
        let weight_bytes = weight_floats * 4;
        let weights = graph.add_tensor(
            vec![co, kh, kw, ci],
            DType::F32,
            TensorKind::Constant {
                offset: 0,
                len: weight_bytes,
            },
        );

        let bias = if has_bias {
            let bias_bytes = co * 4;
            Some(graph.add_tensor(
                vec![co],
                DType::F32,
                TensorKind::Constant {
                    offset: weight_bytes,
                    len: bias_bytes,
                },
            ))
        } else {
            None
        };

        let output = graph.add_tensor(vec![1, oh, ow, co], DType::F32, TensorKind::Output);
        graph.outputs.push(output);

        let fused_activation = if has_relu {
            Some(Activation::Relu)
        } else {
            None
        };

        graph.ops.push(PspOp::Conv2d {
            input,
            weights,
            bias,
            output,
            weight_scales: None,
            params: Conv2dParams {
                kernel_h: kh,
                kernel_w: kw,
                stride_h: stride,
                stride_w: stride,
                pad_top: pad,
                pad_bottom: pad,
                pad_left: pad,
                pad_right: pad,
                fused_activation,
            },
        });

        let total_bytes = weight_bytes + if has_bias { co * 4 } else { 0 };
        PspModel {
            graph,
            model_data: vec![0u8; total_bytes],
        }
    }

    // --- VFPU Conv2d GEMM dimension tests ---

    #[test]
    fn vfpu_conv2d_gemm_dimensions() {
        // MNIST Conv1: [1,28,28,1] * [8,5,5,1] pad=2 → [1,28,28,8]
        // M=784, K=25, K_pad=28, N=8
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        let plan = lower(&mut model).unwrap();
        let op = &plan.ops[0];
        // gemm is sub_op[1]
        match &op.sub_ops[1].kernels[0] {
            KernelCall::GemmBtPacked { m, k, lda, n, .. } => {
                assert_eq!(*m, 784);
                assert_eq!(*k, 25); // logical K
                assert_eq!(*lda, 28); // im2col row stride, padded to a multiple of 4
                assert_eq!(*n, 8);
            }
            other => panic!("Expected GemmBtPacked, got {:?}", other),
        }
    }

    #[test]
    fn vfpu_conv2d_packs_f32_weights() {
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        let plan = lower(&mut model).unwrap();
        let scratch = &plan.ops[0].scratch[1]; // weight scratch
        match &scratch.load_from {
            Some(ScratchLoad {
                copy: CopyStrategy::PackB { n, k },
                ..
            }) => {
                assert_eq!(*n, 8);
                assert_eq!(*k, 25);
            }
            other => panic!("Expected PackB, got {:?}", other),
        }
        assert_eq!(scratch.size, psp_rt::kernels::gemm_bp_len(8, 25));
    }

    #[test]
    fn vfpu_conv2d_scratch_sizes() {
        // Conv1: M=784, K=25, K_pad=28 → im2col=784*28=21952.
        // The weight scratch is now a packed B panel, which rounds K up to a
        // multiple of 8 and N up to a multiple of 8: 1 nb * 8 kt * 8 * 4 = 256.
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        let plan = lower(&mut model).unwrap();
        assert_eq!(plan.ops[0].scratch[0].size, 21952);
        assert_eq!(plan.ops[0].scratch[1].size, psp_rt::kernels::gemm_bp_len(8, 25));

        // Conv2: M=196, K_pad=200 → im2col=196*200=39200, weights=16*200=3200
        let mut model = make_conv2d_model(14, 14, 8, 16, 5, 5, 2, 1, true, true);
        let plan = lower(&mut model).unwrap();
        assert_eq!(plan.ops[0].scratch[0].size, 39200);
        assert_eq!(plan.ops[0].scratch[1].size, 3200);
    }

    #[test]
    fn vfpu_conv2d_strided() {
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 2, true, true);
        let plan = lower(&mut model).unwrap();
        // Verify im2col uses stride [2,2]
        match &plan.ops[0].sub_ops[0].kernels[0] {
            KernelCall::Im2colPadded { stride, .. } => assert_eq!(*stride, [2, 2]),
            other => panic!("expected Im2colPadded, got: {other:?}"),
        }
    }

    #[test]
    fn vfpu_conv2d_handles_m_not_divisible_by_four() {
        // M = 1*3*3 = 9. The blocked GEMM absorbs the M tail in its pack, so
        // this no longer falls back to the naive kernel — which is what moves
        // BirdNET's conv2d_q8 ops onto the fast path.
        let mut model = make_conv2d_model(5, 5, 1, 4, 3, 3, 0, 1, true, true);
        let plan = lower(&mut model).unwrap();
        match &plan.ops[0].sub_ops[1].kernels[0] {
            KernelCall::GemmBtPacked { m, n, .. } => {
                assert_eq!(*m, 9);
                assert_eq!(*n, 4);
            }
            other => panic!("expected GemmBtPacked, got: {other:?}"),
        }
    }

    #[test]
    fn asymmetric_padding_accepted() {
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        if let PspOp::Conv2d { params, .. } = &mut model.graph.ops[0] {
            params.pad_bottom = 3; // asymmetric
        }
        let plan = lower(&mut model).unwrap();
        // Verify the padding is [2, 3, 2, 2] (top, bottom, left, right)
        match &plan.ops[0].sub_ops[0].kernels[0] {
            KernelCall::Im2colPadded { padding, .. } => {
                assert_eq!(*padding, [2, 3, 2, 2]);
            }
            other => panic!("expected Im2colPadded, got: {other:?}"),
        }
    }

    #[test]
    fn relu6_rejected() {
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        if let PspOp::Conv2d { params, .. } = &mut model.graph.ops[0] {
            params.fused_activation = Some(Activation::Relu6);
        }
        let err = lower(&mut model).unwrap_err();
        assert!(err.contains("Relu6"), "got: {err}");
    }

    /// Build an FC model with the given batch, k and n.
    fn make_fc_model(batch: usize, k: usize, n: usize, has_bias: bool) -> PspModel {
        let mut graph = Graph::new();
        let input = graph.add_tensor(vec![batch, k], DType::F32, TensorKind::Input);
        graph.inputs.push(input);
        let wbytes = n * k * 4;
        let weights = graph.add_tensor(
            vec![n, k],
            DType::F32,
            TensorKind::Constant { offset: 0, len: wbytes },
        );
        let bias = has_bias.then(|| {
            graph.add_tensor(
                vec![n],
                DType::F32,
                TensorKind::Constant { offset: wbytes, len: n * 4 },
            )
        });
        let output = graph.add_tensor(vec![batch, n], DType::F32, TensorKind::Output);
        graph.outputs.push(output);
        graph.ops.push(PspOp::FullyConnected {
            input,
            weights,
            bias,
            output,
            fused_activation: FullyConnectedParams { fused_activation: None },
        });
        let total = wbytes + if has_bias { n * 4 } else { 0 };
        PspModel { graph, model_data: vec![0u8; total] }
    }

    /// Single-op RFFT model — the other caller of `append_constant_f32`.
    fn make_rfft_model(frames: usize, n: usize) -> PspModel {
        let mut graph = Graph::new();
        let input = graph.add_tensor(vec![frames, n], DType::F32, TensorKind::Input);
        graph.inputs.push(input);
        let output = graph.add_tensor(vec![frames, n / 2 + 1], DType::F32, TensorKind::Output);
        graph.outputs.push(output);
        graph.ops.push(PspOp::Rfft {
            input,
            output,
            fft_length: n,
        });
        PspModel { graph, model_data: Vec::new() }
    }

    #[test]
    fn lower_is_append_only() {
        // `generate_code` rolls a speculative lowering back by truncating
        // `model_data` and `graph.tensors` to their prior lengths. That is only
        // exact if lowering appends and never rewrites — these are the two
        // paths that append (repacked VFPU weights, FFT twiddles).
        for mut model in [make_fc_model(511, 1025, 96, false), make_rfft_model(4, 512)] {
            let data_before = model.model_data.clone();
            let tensors_before = model.graph.tensors.len();

            let first = lower(&mut model).unwrap();
            assert!(
                model.model_data.len() > data_before.len(),
                "expected this model to append constants, or it tests nothing"
            );
            assert_eq!(
                &model.model_data[..data_before.len()],
                &data_before[..],
                "lowering rewrote pre-existing model_data"
            );
            let data_after_first = model.model_data.clone();

            model.model_data.truncate(data_before.len());
            model.graph.tensors.truncate(tensors_before);

            let second = lower(&mut model).unwrap();
            assert_eq!(first, second, "re-lowering after rollback changed the plan");
            assert_eq!(
                model.model_data, data_after_first,
                "rollback left duplicated appended constants behind"
            );
        }
    }

    #[test]
    fn vfpu_gemm_used_for_batched_fc() {
        // BirdNET's op14 shape.
        let mut model = make_fc_model(511, 1025, 96, false);
        let plan = lower(&mut model).unwrap();
        assert_eq!(plan.ops[0].sub_ops[0].name, "gemm_vfpu");
        match &plan.ops[0].sub_ops[0].kernels[0] {
            KernelCall::GemmBtPacked { m, k, n, mc, kc, .. } => {
                assert_eq!((*m, *k, *n), (511, 1025, 96));
                assert_eq!((*mc, *kc), (GEMM_MC, GEMM_KC));
            }
            other => panic!("expected GemmBtPacked, got: {other:?}"),
        }
        // Two scratch buffers: packed-A block and the packed-C slab.
        assert_eq!(plan.ops[0].scratch.len(), 2);
        assert_eq!(plan.ops[0].scratch[0].size, psp_rt::kernels::gemm_ap_len(GEMM_MC, GEMM_KC));
        assert_eq!(plan.ops[0].scratch[1].size, psp_rt::kernels::gemm_cp_len(GEMM_MC, 96));
    }

    #[test]
    fn vfpu_gemm_repacks_weights_and_drops_the_original() {
        let mut model = make_fc_model(511, 1025, 96, false);
        let original = match &model.graph.ops[0] {
            PspOp::FullyConnected { weights, .. } => *weights,
            _ => unreachable!(),
        };
        let plan = lower(&mut model).unwrap();
        let packed = match &plan.ops[0].sub_ops[0].kernels[0] {
            KernelCall::GemmBtPacked { b: GemmOperand::Tensor(id), .. } => *id,
            other => panic!("expected GemmBtPacked with tensor B, got: {other:?}"),
        };
        assert_ne!(packed, original, "weights should be repacked into a new constant");
        // The unpacked original is now dead and must not reach the blob.
        let ids: Vec<_> = plan
            .allocs
            .iter()
            .filter_map(|a| match a {
                TensorAlloc::Constant { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert!(ids.contains(&packed), "packed weights missing from allocs");
        assert!(!ids.contains(&original), "dead original weights still in the blob");
    }

    #[test]
    fn vfpu_gemm_keeps_bias_as_a_follow_on_sub_op() {
        let mut model = make_fc_model(64, 128, 32, true);
        let plan = lower(&mut model).unwrap();
        let names: Vec<&str> = plan.ops[0].sub_ops.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["gemm_vfpu", "bias_add"]);
    }

    #[test]
    fn small_batch_fc_stays_on_the_scalar_path() {
        // Below GEMM_MIN_BATCH the pack/unpack overhead is not worth it.
        let mut model = make_fc_model(1, 1024, 16, false);
        let plan = lower(&mut model).unwrap();
        match &plan.ops[0].sub_ops[0].kernels[0] {
            KernelCall::FullyConnected { .. } => {}
            other => panic!("expected scalar FullyConnected, got: {other:?}"),
        }
    }

    #[test]
    fn fc_no_bias() {
        let mut graph = Graph::new();
        let input = graph.add_tensor(vec![784], DType::F32, TensorKind::Input);
        graph.inputs.push(input);
        let weights = graph.add_tensor(
            vec![10, 784],
            DType::F32,
            TensorKind::Constant {
                offset: 0,
                len: 10 * 784 * 4,
            },
        );
        let output = graph.add_tensor(vec![10], DType::F32, TensorKind::Output);
        graph.outputs.push(output);
        graph.ops.push(PspOp::FullyConnected {
            input,
            weights,
            bias: None,
            output,
            fused_activation: FullyConnectedParams {
                fused_activation: None,
            },
        });
        let mut model = PspModel {
            graph,
            model_data: vec![0u8; 10 * 784 * 4],
        };
        let plan = lower(&mut model).unwrap();
        match &plan.ops[0].sub_ops[0].kernels[0] {
            KernelCall::FullyConnected { bias, .. } => assert!(bias.is_none()),
            other => panic!("expected FullyConnected, got: {other:?}"),
        }
    }

    #[test]
    fn weight_offset_not_aligned() {
        let mut graph = Graph::new();
        let input = graph.add_tensor(vec![4], DType::F32, TensorKind::Input);
        graph.inputs.push(input);
        let bad = graph.add_tensor(
            vec![2],
            DType::F32,
            TensorKind::Constant {
                offset: 3, // not 4-byte aligned
                len: 8,
            },
        );
        let output = graph.add_tensor(vec![4], DType::F32, TensorKind::Output);
        graph.outputs.push(output);
        // Reference the bad tensor so it isn't pruned by the alloc pass
        graph.ops.push(PspOp::ElementWise {
            op: crate::ir::psp::BinaryOp::Add,
            input_a: input,
            input_b: bad,
            output,
        });
        let mut model = PspModel {
            graph,
            model_data: vec![0u8; 16],
        };
        let err = lower(&mut model).unwrap_err();
        assert!(err.contains("not 4-byte aligned"), "got: {err}");
    }

    #[test]
    fn tensor_allocs_correct() {
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        let plan = lower(&mut model).unwrap();
        // Should have: 1 weight constant + 1 bias constant + 1 output
        let constants: Vec<_> = plan
            .allocs
            .iter()
            .filter(|a| matches!(a, TensorAlloc::Constant { .. }))
            .collect();
        let outputs: Vec<_> = plan
            .allocs
            .iter()
            .filter(|a| matches!(a, TensorAlloc::Output { .. }))
            .collect();
        assert_eq!(constants.len(), 2);
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn multiple_inputs_rejected() {
        let mut graph: Graph<PspOp> = Graph::new();
        let i1 = graph.add_tensor(vec![4], DType::F32, TensorKind::Input);
        let i2 = graph.add_tensor(vec![4], DType::F32, TensorKind::Input);
        graph.inputs.push(i1);
        graph.inputs.push(i2);
        let output = graph.add_tensor(vec![4], DType::F32, TensorKind::Output);
        graph.outputs.push(output);
        let mut model = PspModel {
            graph,
            model_data: vec![],
        };
        let err = lower(&mut model).unwrap_err();
        assert!(err.contains("Expected 1 input"), "got: {err}");
    }

    #[test]
    fn conv2d_no_bias_no_relu_has_2_sub_ops() {
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, false, false);
        let plan = lower(&mut model).unwrap();
        // im2col + gemm, no bias/relu sub-op
        assert_eq!(plan.ops[0].sub_ops.len(), 2);
        assert_eq!(plan.ops[0].sub_ops[0].name, "im2col");
        assert_eq!(plan.ops[0].sub_ops[1].name, "gemm_vfpu");
    }

    #[test]
    fn conv2d_bias_relu_has_3_sub_ops() {
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        let plan = lower(&mut model).unwrap();
        assert_eq!(plan.ops[0].sub_ops.len(), 3);
        assert_eq!(plan.ops[0].sub_ops[0].name, "im2col");
        assert_eq!(plan.ops[0].sub_ops[1].name, "gemm_vfpu");
        assert_eq!(plan.ops[0].sub_ops[2].name, "bias_add_relu");
    }
}
