use crate::ir::graph::{TensorId, TensorKind};
use crate::ir::psp::{Activation, PspModel, PspOp};

use super::plan::*;

const VFPU_Q: usize = 4;

const fn ceil_vfpu_q(x: usize) -> usize {
    (x + VFPU_Q - 1) & !(VFPU_Q - 1)
}

/**
 * Lower a PspModel into a CodegenPlan.
 */
pub fn lower(model: &PspModel) -> Result<CodegenPlan, String> {
    let graph = &model.graph;

    if graph.inputs.len() != 1 {
        return Err(format!(
            "Expected 1 input tensor, found {}",
            graph.inputs.len()
        ));
    }
    if graph.outputs.len() != 1 {
        return Err(format!(
            "Expected 1 output tensor, found {}",
            graph.outputs.len()
        ));
    }

    let input_id = graph.inputs[0];
    let output_id = graph.outputs[0];
    let input_size = graph.tensor(input_id).shape.iter().product::<usize>();
    let output_size = graph.tensor(output_id).shape.iter().product::<usize>();

    let blob_bytes = model.model_data.len();
    let blob_floats = blob_bytes / std::mem::size_of::<f32>();

    let allocs = lower_allocs(model)?;

    // TODO: expose as compiler flag
    let use_vfpu_conv2d = true;
    let ops = lower_ops(model, use_vfpu_conv2d)?;

    Ok(CodegenPlan {
        input_id,
        output_id,
        input_size,
        output_size,
        blob_bytes,
        blob_floats,
        allocs,
        ops,
    })
}

fn lower_allocs(model: &PspModel) -> Result<Vec<TensorAlloc>, String> {
    let mut allocs = Vec::new();
    let sz = std::mem::size_of::<f32>();

    for tensor in &model.graph.tensors {
        match &tensor.kind {
            TensorKind::Constant { offset, len } => {
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
                });
            }
            TensorKind::Intermediate => {
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

fn lower_ops(model: &PspModel, use_vfpu_conv2d: bool) -> Result<Vec<OpPlan>, String> {
    let graph = &model.graph;
    let mut ops = Vec::new();

    for (i, op) in graph.ops.iter().enumerate() {
        let plan = match op {
            PspOp::Conv2d {
                input,
                weights,
                bias,
                output,
                params,
            } => {
                if params.pad_top != params.pad_bottom || params.pad_left != params.pad_right {
                    return Err(format!("Op {i}: asymmetric padding not supported"));
                }

                let in_shape = &graph.tensor(*input).shape;
                let w_shape = &graph.tensor(*weights).shape;
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
                let padding = [params.pad_top, params.pad_left];
                let has_relu = matches!(params.fused_activation, Some(Activation::Relu));

                if use_vfpu_conv2d {
                    lower_conv2d_vfpu(i, in4, w4, *bias, out4, stride, padding, has_relu)?
                } else {
                    lower_conv2d_naive(in4, w4, *bias, out4, stride, padding, has_relu)
                }
            }

            PspOp::FullyConnected {
                input,
                weights,
                bias,
                output,
                fused_activation,
            } => {
                let bias_id =
                    bias.ok_or_else(|| format!("Op {i}: FullyConnected requires bias tensor"))?;

                let in_features = graph.tensor(*input).shape.iter().product::<usize>();
                let out_features = graph.tensor(*output).shape.iter().product::<usize>();
                let has_relu = matches!(fused_activation.fused_activation, Some(Activation::Relu));

                if let Some(Activation::Relu6) = fused_activation.fused_activation {
                    return Err(format!("Op {i}: Relu6 not supported for FullyConnected"));
                }

                let name = if has_relu {
                    "fully_connected_relu"
                } else {
                    "fully_connected"
                };

                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: name.into(),
                        kernels: vec![KernelCall::FullyConnected {
                            input: *input,
                            in_features,
                            weights: *weights,
                            bias: bias_id,
                            output: *output,
                            out_features,
                            has_relu,
                        }],
                    }],
                }
            }

            PspOp::MaxPool2x2 { input, output } => {
                let in_shape = &graph.tensor(*input).shape;
                let out_shape = &graph.tensor(*output).shape;

                if in_shape.len() != 4 || out_shape.len() != 4 {
                    return Err(format!(
                        "Op {i}: MaxPool2x2 expects 4D tensors (input={}, output={})",
                        in_shape.len(),
                        out_shape.len()
                    ));
                }

                OpPlan {
                    scratch: vec![],
                    sub_ops: vec![SubOpPlan {
                        name: "max_pool2d".into(),
                        kernels: vec![KernelCall::MaxPool2d {
                            input: Tensor4d {
                                id: *input,
                                shape: [in_shape[0], in_shape[1], in_shape[2], in_shape[3]],
                            },
                            output: Tensor4d {
                                id: *output,
                                shape: [out_shape[0], out_shape[1], out_shape[2], out_shape[3]],
                            },
                        }],
                    }],
                }
            }

            PspOp::Reshape { input, output } => OpPlan {
                scratch: vec![],
                sub_ops: vec![SubOpPlan {
                    name: "reshape".into(),
                    kernels: vec![KernelCall::Reshape {
                        input: *input,
                        output: *output,
                    }],
                }],
            },

            PspOp::Softmax { .. } => {
                return Err(format!("Op {i}: Softmax kernel not yet implemented"));
            }
        };

        ops.push(plan);
    }

    Ok(ops)
}

fn lower_conv2d_naive(
    input: Tensor4d,
    filter: Tensor4d,
    bias: Option<TensorId>,
    output: Tensor4d,
    stride: [usize; 2],
    padding: [usize; 2],
    has_relu: bool,
) -> OpPlan {
    let name = if has_relu { "conv2d_relu" } else { "conv2d" };
    OpPlan {
        scratch: vec![],
        sub_ops: vec![SubOpPlan {
            name: name.into(),
            kernels: vec![KernelCall::Conv2d {
                input,
                filter,
                bias,
                stride,
                padding,
                output,
                has_relu,
            }],
        }],
    }
}

fn lower_conv2d_vfpu(
    op_idx: usize,
    input: Tensor4d,
    weights: Tensor4d,
    bias: Option<TensorId>,
    output: Tensor4d,
    stride: [usize; 2],
    padding: [usize; 2],
    has_relu: bool,
) -> Result<OpPlan, String> {
    if stride != [1, 1] {
        return Err(format!(
            "Op {op_idx}: VFPU conv2d requires stride [1,1], got {:?}",
            stride
        ));
    }

    let [n, _, _, ci] = input.shape;
    let [co, kh, kw, _] = weights.shape;
    let [_, ho, wo, _] = output.shape;

    let gemm_m = n * ho * wo;
    let gemm_k = kh * kw * ci;
    let k_padded = ceil_vfpu_q(gemm_k);
    let m_padded = ceil_vfpu_q(gemm_m);
    let n_padded = ceil_vfpu_q(co);

    if m_padded != gemm_m {
        return Err(format!(
            "Op {op_idx}: VFPU conv2d requires M ({gemm_m}) to be a multiple of {VFPU_Q}"
        ));
    }
    if n_padded != co {
        return Err(format!(
            "Op {op_idx}: VFPU conv2d requires N ({co}) to be a multiple of {VFPU_Q}"
        ));
    }

    let m_tiles = m_padded / VFPU_Q;
    let k_tiles = k_padded / VFPU_Q;
    let n_tiles = n_padded / VFPU_Q;

    // Scratch 0: im2col output (M × K_padded)
    let im2col_size = m_padded * k_padded;
    // Scratch 1: padded weight copy (CO × K_padded)
    let weight_size = co * k_padded;

    let weight_copy_strategy = if k_padded != gemm_k {
        CopyStrategy::RowPadded {
            num_rows: co,
            src_stride: gemm_k,
            dst_stride: k_padded,
        }
    } else {
        CopyStrategy::BulkCopy
    };

    let weight_load = ScratchLoad {
        source: weights.id,
        copy: weight_copy_strategy,
    };

    let scratch = vec![
        ScratchBuffer {
            size: im2col_size,
            load_from: None,
        },
        ScratchBuffer {
            size: weight_size,
            load_from: Some(weight_load),
        },
    ];

    let mut sub_ops = vec![
        SubOpPlan {
            name: "im2col".into(),
            kernels: vec![KernelCall::Im2colPadded {
                input,
                kernel_size: [kh, kw],
                padding,
                output_hw: [ho, wo],
                // TODO: proper index handling; maybe two way pointers?
                output: 0, // scratch index 0
            }],
        },
        SubOpPlan {
            name: "matmul".into(),
            kernels: vec![KernelCall::MatmulBtTiled {
                a: 0, // scratch index 0
                b: 1, // scratch index 1
                output: output.id,
                m_tiles,
                k_tiles,
                n_tiles,
            }],
        },
    ];

    // Bias + relu sub-op
    if bias.is_some() || has_relu {
        let name = if has_relu {
            "bias_add_relu"
        } else {
            "bias_add"
        };
        let mut kernels = Vec::new();
        if let Some(bias_id) = bias {
            kernels.push(KernelCall::BiasAdd {
                output: output.id,
                bias: bias_id,
                rows: gemm_m,
                cols: co,
            });
        }
        if has_relu {
            kernels.push(KernelCall::Relu { output: output.id });
        }
        sub_ops.push(SubOpPlan {
            name: name.into(),
            kernels,
        });
    }

    Ok(OpPlan { scratch, sub_ops })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::{DType, Graph, TensorKind};
    use crate::ir::psp::{Conv2dParams, FullyConnectedParams, PspModel};

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
        let model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        let plan = lower(&model).unwrap();
        let op = &plan.ops[0];
        // matmul is sub_op[1]
        match &op.sub_ops[1].kernels[0] {
            KernelCall::MatmulBtTiled {
                m_tiles,
                k_tiles,
                n_tiles,
                ..
            } => {
                assert_eq!(*m_tiles, 196); // 784/4
                assert_eq!(*k_tiles, 7); // 28/4
                assert_eq!(*n_tiles, 2); // 8/4
            }
            other => panic!("Expected MatmulBtTiled, got {:?}", other),
        }
    }

    #[test]
    fn vfpu_conv2d_row_padded_when_k_unaligned() {
        // K=25, K_pad=28 → RowPadded
        let model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        let plan = lower(&model).unwrap();
        let scratch = &plan.ops[0].scratch[1]; // weight scratch
        match &scratch.load_from {
            Some(ScratchLoad {
                copy:
                    CopyStrategy::RowPadded {
                        num_rows,
                        src_stride,
                        dst_stride,
                    },
                ..
            }) => {
                assert_eq!(*num_rows, 8);
                assert_eq!(*src_stride, 25);
                assert_eq!(*dst_stride, 28);
            }
            other => panic!("Expected RowPadded, got {:?}", other),
        }
    }

    #[test]
    fn vfpu_conv2d_bulk_copy_when_k_aligned() {
        // MNIST Conv2: [1,14,14,8] * [16,5,5,8] pad=2 → [1,14,14,16]
        // K=200, K_pad=200 (already aligned) → BulkCopy
        let model = make_conv2d_model(14, 14, 8, 16, 5, 5, 2, 1, true, true);
        let plan = lower(&model).unwrap();
        let scratch = &plan.ops[0].scratch[1];
        match &scratch.load_from {
            Some(ScratchLoad {
                copy: CopyStrategy::BulkCopy,
                ..
            }) => {}
            other => panic!("Expected BulkCopy, got {:?}", other),
        }
    }

    #[test]
    fn vfpu_conv2d_scratch_sizes() {
        // Conv1: M=784, K_pad=28 → im2col=784*28=21952, weights=8*28=224
        let model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        let plan = lower(&model).unwrap();
        assert_eq!(plan.ops[0].scratch[0].size, 21952);
        assert_eq!(plan.ops[0].scratch[1].size, 224);

        // Conv2: M=196, K_pad=200 → im2col=196*200=39200, weights=16*200=3200
        let model = make_conv2d_model(14, 14, 8, 16, 5, 5, 2, 1, true, true);
        let plan = lower(&model).unwrap();
        assert_eq!(plan.ops[0].scratch[0].size, 39200);
        assert_eq!(plan.ops[0].scratch[1].size, 3200);
    }

    #[test]
    fn vfpu_conv2d_rejects_non_unit_stride() {
        let model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 2, true, true);
        let err = lower(&model).unwrap_err();
        assert!(err.contains("stride [1,1]"), "got: {err}");
    }

    #[test]
    fn vfpu_conv2d_rejects_unaligned_m() {
        // M = 1*3*3 = 9 (not multiple of 4)
        let model = make_conv2d_model(5, 5, 1, 4, 3, 3, 0, 1, true, true);
        let plan = lower(&model);
        // M=9, not multiple of 4
        assert!(plan.is_err());
        assert!(plan.unwrap_err().contains("multiple of 4"));
    }

    #[test]
    fn asymmetric_padding_rejected() {
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        if let PspOp::Conv2d { params, .. } = &mut model.graph.ops[0] {
            params.pad_bottom = 3; // asymmetric
        }
        let err = lower(&model).unwrap_err();
        assert!(err.contains("asymmetric"), "got: {err}");
    }

    #[test]
    fn relu6_rejected() {
        let mut model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        if let PspOp::Conv2d { params, .. } = &mut model.graph.ops[0] {
            params.fused_activation = Some(Activation::Relu6);
        }
        let err = lower(&model).unwrap_err();
        assert!(err.contains("Relu6"), "got: {err}");
    }

    #[test]
    fn fc_requires_bias() {
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
            bias: None, // no bias
            output,
            fused_activation: FullyConnectedParams {
                fused_activation: None,
            },
        });
        let model = PspModel {
            graph,
            model_data: vec![0u8; 10 * 784 * 4],
        };
        let err = lower(&model).unwrap_err();
        assert!(err.contains("requires bias"), "got: {err}");
    }

    #[test]
    fn weight_offset_not_aligned() {
        let mut graph = Graph::new();
        let input = graph.add_tensor(vec![4], DType::F32, TensorKind::Input);
        graph.inputs.push(input);
        let _bad = graph.add_tensor(
            vec![2],
            DType::F32,
            TensorKind::Constant {
                offset: 3, // not 4-byte aligned
                len: 8,
            },
        );
        let output = graph.add_tensor(vec![4], DType::F32, TensorKind::Output);
        graph.outputs.push(output);
        // No ops — the alloc pass runs first and should catch it
        let model = PspModel {
            graph,
            model_data: vec![0u8; 16],
        };
        let err = lower(&model).unwrap_err();
        assert!(err.contains("not 4-byte aligned"), "got: {err}");
    }

    #[test]
    fn tensor_allocs_correct() {
        let model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        let plan = lower(&model).unwrap();
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
        let model = PspModel {
            graph,
            model_data: vec![],
        };
        let err = lower(&model).unwrap_err();
        assert!(err.contains("Expected 1 input"), "got: {err}");
    }

    #[test]
    fn conv2d_no_bias_no_relu_has_2_sub_ops() {
        let model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, false, false);
        let plan = lower(&model).unwrap();
        // im2col + matmul, no bias/relu sub-op
        assert_eq!(plan.ops[0].sub_ops.len(), 2);
        assert_eq!(plan.ops[0].sub_ops[0].name, "im2col");
        assert_eq!(plan.ops[0].sub_ops[1].name, "matmul");
    }

    #[test]
    fn conv2d_bias_relu_has_3_sub_ops() {
        let model = make_conv2d_model(28, 28, 1, 8, 5, 5, 2, 1, true, true);
        let plan = lower(&model).unwrap();
        assert_eq!(plan.ops[0].sub_ops.len(), 3);
        assert_eq!(plan.ops[0].sub_ops[0].name, "im2col");
        assert_eq!(plan.ops[0].sub_ops[1].name, "matmul");
        assert_eq!(plan.ops[0].sub_ops[2].name, "bias_add_relu");
    }
}
