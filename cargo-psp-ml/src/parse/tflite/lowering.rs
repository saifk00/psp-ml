//! TFLite to PSP IR lowering pass.
//!
//! Converts a TFLite model into our PSP IR representation with zero-copy
//! weight storage via absolute offsets into the original model buffer.

use std::collections::HashSet;

use super::{
    root_as_model, ActivationFunctionType, Buffer, BuiltinOperator, Operator, Padding, TensorType,
};

type Buffers<'a> = flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Buffer<'a>>>;
use crate::ir::graph::{DType, Graph, Tensor, TensorId, TensorKind};
use crate::ir::const_fold;
use crate::ir::psp::{Activation, BinaryOp, Conv2dParams, FullyConnectedParams, PspModel, PspOp, ReduceOp, UnaryOp};

/// Convert a TFLite model buffer into PSP IR.
///
/// Takes ownership of the raw bytes so `PspModel` can pair the graph with
/// its backing weight data.
pub fn to_psp_ir(model_data: Vec<u8>) -> Result<PspModel, String> {
    let graph = lower(&model_data)?;
    let mut model = PspModel { graph, model_data };
    const_fold::fold(&mut model);
    Ok(model)
}

/// Pure lowering pass: borrows model bytes, returns an IR graph with no data ownership.
fn lower(model_data: &[u8]) -> Result<Graph<PspOp>, String> {
    let model =
        root_as_model(model_data).map_err(|e| format!("failed to parse TFLite model: {e}"))?;

    // TODO multiple subgraph support
    assert!(model.subgraphs().unwrap().len() == 1);

    let subgraph = model.subgraphs().unwrap().get(0);
    let buffers = model.buffers().ok_or("no buffers in model")?;
    let opcodes = model.operator_codes().ok_or("no operator codes in model")?;
    let tflite_tensors = subgraph.tensors().ok_or("no tensors in subgraph")?;

    // Collect graph input/output indices
    let tflite_inputs: HashSet<i32> = subgraph
        .inputs()
        .map(|v| v.iter().collect())
        .unwrap_or_default();
    let tflite_outputs: HashSet<i32> = subgraph
        .outputs()
        .map(|v| v.iter().collect())
        .unwrap_or_default();

    // Build tensor map (TFLite index -> our TensorId)
    let mut tensor_map: Vec<TensorId> = Vec::with_capacity(tflite_tensors.len());
    let mut graph_tensors: Vec<Tensor> = Vec::with_capacity(tflite_tensors.len());

    for idx in 0..tflite_tensors.len() {
        let tensor = tflite_tensors.get(idx);
        let shape: Vec<usize> = tensor
            .shape()
            .map(|s| s.iter().map(|x| x as usize).collect())
            .unwrap_or_default();
        let dtype = convert_dtype(tensor.type_());

        // Determine tensor kind based on role
        let kind = if tflite_inputs.contains(&(idx as i32)) {
            TensorKind::Input
        } else if tflite_outputs.contains(&(idx as i32)) {
            TensorKind::Output
        } else if let Some((offset, len)) =
            get_buffer_location(&model_data, &buffers, tensor.buffer())
        {
            TensorKind::Constant { offset, len }
        } else {
            TensorKind::Intermediate
        };

        let id = graph_tensors.len();
        graph_tensors.push(Tensor {
            id,
            shape,
            dtype,
            kind,
        });
        tensor_map.push(id);
    }

    // Convert operators
    let mut graph_ops: Vec<PspOp> = Vec::new();
    let operators = subgraph.operators().ok_or("no operators in subgraph")?;

    for i in 0..operators.len() {
        let op = operators.get(i);
        let opcode = opcodes.get(op.opcode_index() as usize);
        let builtin_code = opcode.builtin_code();

        let psp_op = match builtin_code {
            BuiltinOperator::CONV_2D => Some(lower_conv2d(&op, &tensor_map, &graph_tensors)?),
            BuiltinOperator::FULLY_CONNECTED => Some(lower_fc(&op, &tensor_map)?),
            BuiltinOperator::MAX_POOL_2D => Some(lower_maxpool(&op, &tensor_map)?),
            BuiltinOperator::RESHAPE
            | BuiltinOperator::SQUEEZE
            | BuiltinOperator::EXPAND_DIMS => {
                if try_alias_shape_op(&op, &mut tensor_map, &tflite_outputs)? {
                    None // aliased — zero-cost, no op emitted
                } else {
                    Some(lower_reshape(&op, &tensor_map)?) // graph-output fallback
                }
            }
            BuiltinOperator::ADD
            | BuiltinOperator::MUL
            | BuiltinOperator::SUB
            | BuiltinOperator::DIV
            | BuiltinOperator::FLOOR_DIV
            | BuiltinOperator::MAXIMUM
            | BuiltinOperator::POW => {
                Some(lower_elementwise(&op, &tensor_map, builtin_code)?)
            }
            BuiltinOperator::LOGISTIC => {
                Some(lower_unary_elementwise(&op, &tensor_map, builtin_code)?)
            }
            BuiltinOperator::SOFTMAX => Some(lower_softmax(&op, &tensor_map)?),
            BuiltinOperator::SHAPE => Some(lower_shape(&op, &tensor_map)?),
            BuiltinOperator::PACK => Some(lower_pack(&op, &tensor_map)?),
            BuiltinOperator::STRIDED_SLICE => Some(lower_strided_slice(&op, &tensor_map)?),
            BuiltinOperator::CONCATENATION => Some(lower_concatenation(&op, &tensor_map)?),
            BuiltinOperator::GATHER => Some(lower_gather(&op, &tensor_map)?),
            BuiltinOperator::REDUCE_PROD
            | BuiltinOperator::REDUCE_MAX
            | BuiltinOperator::REDUCE_MIN
            | BuiltinOperator::MEAN => Some(lower_reduce(&op, &tensor_map, builtin_code)?),
            BuiltinOperator::RANGE => Some(lower_range(&op, &tensor_map)?),
            BuiltinOperator::SPLIT_V => Some(lower_split_v(&op, &tensor_map)?),
            BuiltinOperator::CAST => Some(lower_cast(&op, &tensor_map)?),
            BuiltinOperator::PAD => Some(lower_pad(&op, &tensor_map)?),
            other => {
                return Err(format!(
                    "unsupported operator: {:?}",
                    other.variant_name().unwrap_or("unknown")
                ));
            }
        };
        if let Some(op) = psp_op {
            graph_ops.push(op);
        }
    }

    // Collect graph inputs/outputs (in order from subgraph)
    let inputs: Vec<TensorId> = subgraph
        .inputs()
        .map(|v| v.iter().map(|i| tensor_map[i as usize]).collect())
        .unwrap_or_default();
    let outputs: Vec<TensorId> = subgraph
        .outputs()
        .map(|v| v.iter().map(|i| tensor_map[i as usize]).collect())
        .unwrap_or_default();

    Ok(Graph {
        tensors: graph_tensors,
        ops: graph_ops,
        inputs,
        outputs,
    })
}

/// Get (offset, len) of buffer data in the original file bytes.
///
/// Uses pointer arithmetic to compute the absolute offset of the buffer
/// data within the model file.
fn get_buffer_location(
    model_data: &[u8],
    buffers: &Buffers,
    buffer_idx: u32,
) -> Option<(usize, usize)> {
    let buffer = buffers.get(buffer_idx as usize);
    let data = buffer.data()?;
    // Use bytes() to get the underlying &[u8] slice, then get pointer
    let data_bytes = data.bytes();
    if data_bytes.is_empty() {
        return None;
    }

    // FlatBuffers returns a slice into the original buffer.
    // Compute offset via pointer arithmetic.
    let data_ptr = data_bytes.as_ptr() as usize;
    let base_ptr = model_data.as_ptr() as usize;

    // Sanity check: data should be within model_data
    if data_ptr < base_ptr || data_ptr >= base_ptr + model_data.len() {
        return None;
    }

    let offset = data_ptr - base_ptr;
    Some((offset, data_bytes.len()))
}

/// Try to alias a shape-only op (RESHAPE, SQUEEZE, EXPAND_DIMS).
///
/// If the output tensor is not a graph output, redirect it to reuse the input
/// tensor's buffer — zero cost, no op emitted, no extra allocation.
/// Returns `Ok(true)` if aliased, `Ok(false)` if fallback copy is needed.
fn try_alias_shape_op(
    op: &Operator,
    tensor_map: &mut [TensorId],
    tflite_outputs: &HashSet<i32>,
) -> Result<bool, String> {
    let inputs = op.inputs().ok_or("shape op: no inputs")?;
    let outputs = op.outputs().ok_or("shape op: no outputs")?;
    let out_idx = outputs.get(0);

    // Fall back to copy if output is a graph output (needs separate stack buffer)
    if tflite_outputs.contains(&out_idx) {
        return Ok(false);
    }

    // Alias: output tensor now resolves to the same TensorId as input
    tensor_map[out_idx as usize] = tensor_map[inputs.get(0) as usize];
    Ok(true)
}

/// Lower TFLite CONV_2D to PspOp::Conv2d
fn lower_conv2d(
    op: &Operator,
    tensor_map: &[TensorId],
    graph_tensors: &[Tensor],
) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("CONV_2D: no inputs")?;
    let outputs = op.outputs().ok_or("CONV_2D: no outputs")?;
    let options = op
        .builtin_options_as_conv_2_doptions()
        .ok_or("CONV_2D: no options")?;

    // Input tensors: input, weights, optional bias
    let input = tensor_map[inputs.get(0) as usize];
    let weights = tensor_map[inputs.get(1) as usize];
    let bias = if inputs.len() > 2 {
        Some(tensor_map[inputs.get(2) as usize])
    } else {
        None
    };
    let output = tensor_map[outputs.get(0) as usize];

    // Get kernel size from weight tensor shape [out_channels, kernel_h, kernel_w, in_channels]
    let weight_tensor = &graph_tensors[weights];
    let (kernel_h, kernel_w) = if weight_tensor.shape.len() == 4 {
        (weight_tensor.shape[1], weight_tensor.shape[2])
    } else {
        return Err(format!(
            "CONV_2D: unexpected weight shape {:?}",
            weight_tensor.shape
        ));
    };

    let stride_h = options.stride_h() as usize;
    let stride_w = options.stride_w() as usize;

    // Compute padding based on padding type
    let (pad_top, pad_bottom, pad_left, pad_right) = match options.padding() {
        Padding::VALID => (0, 0, 0, 0),
        Padding::SAME => {
            // For SAME padding, compute to maintain output size
            // pad_total = max(0, (out - 1) * stride + kernel - input)
            // We compute symmetric padding assuming input shape is known
            let input_tensor = &graph_tensors[input];
            if input_tensor.shape.len() == 4 {
                let input_h = input_tensor.shape[1];
                let input_w = input_tensor.shape[2];
                compute_same_padding(input_h, input_w, kernel_h, kernel_w, stride_h, stride_w)
            } else {
                // Fallback: symmetric padding to preserve size with stride 1
                let pad_h = (kernel_h - 1) / 2;
                let pad_w = (kernel_w - 1) / 2;
                (pad_h, kernel_h - 1 - pad_h, pad_w, kernel_w - 1 - pad_w)
            }
        }
        _ => (0, 0, 0, 0),
    };

    Ok(PspOp::Conv2d {
        input,
        weights,
        bias,
        output,
        params: Conv2dParams {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            fused_activation: convert_activation(options.fused_activation_function()),
        },
    })
}

/// Lower TFLite FULLY_CONNECTED to PspOp::FullyConnected
fn lower_fc(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("FULLY_CONNECTED: no inputs")?;
    let outputs = op.outputs().ok_or("FULLY_CONNECTED: no outputs")?;
    let options = op
        .builtin_options_as_fully_connected_options()
        .ok_or("FULLY_CONNECTED: no options")?;

    let input = tensor_map[inputs.get(0) as usize];
    let weights = tensor_map[inputs.get(1) as usize];
    let bias = if inputs.len() > 2 {
        Some(tensor_map[inputs.get(2) as usize])
    } else {
        None
    };
    let output = tensor_map[outputs.get(0) as usize];

    Ok(PspOp::FullyConnected {
        input,
        weights,
        bias,
        output,
        fused_activation: FullyConnectedParams {
            fused_activation: convert_activation(options.fused_activation_function()),
        },
    })
}

/// Lower TFLite MAX_POOL_2D to PspOp::MaxPool2x2
fn lower_maxpool(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("MAX_POOL_2D: no inputs")?;
    let outputs = op.outputs().ok_or("MAX_POOL_2D: no outputs")?;
    let options = op
        .builtin_options_as_pool_2_doptions()
        .ok_or("MAX_POOL_2D: no options")?;

    // We only support 2x2 maxpool with stride 2 for now
    let filter_h = options.filter_height();
    let filter_w = options.filter_width();
    let stride_h = options.stride_h();
    let stride_w = options.stride_w();

    if filter_h != 2 || filter_w != 2 || stride_h != 2 || stride_w != 2 {
        return Err(format!(
            "MAX_POOL_2D: only 2x2 stride 2 supported, got {}x{} stride {}x{}",
            filter_h, filter_w, stride_h, stride_w
        ));
    }

    let input = tensor_map[inputs.get(0) as usize];
    let output = tensor_map[outputs.get(0) as usize];

    Ok(PspOp::MaxPool2x2 { input, output })
}

/// Lower TFLite RESHAPE to PspOp::Reshape
fn lower_reshape(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("RESHAPE: no inputs")?;
    let outputs = op.outputs().ok_or("RESHAPE: no outputs")?;

    let input = tensor_map[inputs.get(0) as usize];
    let output = tensor_map[outputs.get(0) as usize];

    Ok(PspOp::Reshape { input, output })
}

/// Lower TFLite SOFTMAX to PspOp::Softmax
fn lower_softmax(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("SOFTMAX: no inputs")?;
    let outputs = op.outputs().ok_or("SOFTMAX: no outputs")?;

    let input = tensor_map[inputs.get(0) as usize];
    let output = tensor_map[outputs.get(0) as usize];

    Ok(PspOp::Softmax { input, output })
}

/// Lower TFLite element-wise binary ops (ADD, MUL, SUB, DIV, MAXIMUM) to PspOp::ElementWise
fn lower_elementwise(
    op: &Operator,
    tensor_map: &[TensorId],
    builtin_code: BuiltinOperator,
) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("elementwise: no inputs")?;
    let outputs = op.outputs().ok_or("elementwise: no outputs")?;

    let binary_op = match builtin_code {
        BuiltinOperator::ADD => BinaryOp::Add,
        BuiltinOperator::MUL => BinaryOp::Mul,
        BuiltinOperator::SUB => BinaryOp::Sub,
        BuiltinOperator::DIV => BinaryOp::Div,
        BuiltinOperator::FLOOR_DIV => BinaryOp::FloorDiv,
        BuiltinOperator::MAXIMUM => BinaryOp::Max,
        BuiltinOperator::POW => BinaryOp::Pow,
        _ => unreachable!(),
    };

    let input_a = tensor_map[inputs.get(0) as usize];
    let input_b = tensor_map[inputs.get(1) as usize];
    let output = tensor_map[outputs.get(0) as usize];

    Ok(PspOp::ElementWise {
        op: binary_op,
        input_a,
        input_b,
        output,
    })
}

/// Lower TFLite unary element-wise ops (LOGISTIC, ...) to PspOp::UnaryElementWise
fn lower_unary_elementwise(
    op: &Operator,
    tensor_map: &[TensorId],
    builtin_code: BuiltinOperator,
) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("unary elementwise: no inputs")?;
    let outputs = op.outputs().ok_or("unary elementwise: no outputs")?;

    let unary_op = match builtin_code {
        BuiltinOperator::LOGISTIC => UnaryOp::Logistic,
        _ => unreachable!(),
    };

    let input = tensor_map[inputs.get(0) as usize];
    let output = tensor_map[outputs.get(0) as usize];

    Ok(PspOp::UnaryElementWise {
        op: unary_op,
        input,
        output,
    })
}

/// Lower TFLite SHAPE to PspOp::Shape
fn lower_shape(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("SHAPE: no inputs")?;
    let outputs = op.outputs().ok_or("SHAPE: no outputs")?;
    let input = tensor_map[inputs.get(0) as usize];
    let output = tensor_map[outputs.get(0) as usize];
    Ok(PspOp::Shape { input, output })
}

/// Lower TFLite PACK to PspOp::Pack
fn lower_pack(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs_vec = op.inputs().ok_or("PACK: no inputs")?;
    let outputs = op.outputs().ok_or("PACK: no outputs")?;
    let opts = op
        .builtin_options_as_pack_options()
        .ok_or("PACK: no options")?;
    let inputs: Vec<TensorId> = (0..inputs_vec.len())
        .map(|i| tensor_map[inputs_vec.get(i) as usize])
        .collect();
    let output = tensor_map[outputs.get(0) as usize];
    Ok(PspOp::Pack {
        inputs,
        output,
        axis: opts.axis(),
    })
}

/// Lower TFLite STRIDED_SLICE to PspOp::StridedSlice
fn lower_strided_slice(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("STRIDED_SLICE: no inputs")?;
    let outputs = op.outputs().ok_or("STRIDED_SLICE: no outputs")?;
    let opts = op
        .builtin_options_as_strided_slice_options()
        .ok_or("STRIDED_SLICE: no options")?;
    let input = tensor_map[inputs.get(0) as usize];
    let begin = tensor_map[inputs.get(1) as usize];
    let end = tensor_map[inputs.get(2) as usize];
    let strides = tensor_map[inputs.get(3) as usize];
    let output = tensor_map[outputs.get(0) as usize];
    Ok(PspOp::StridedSlice {
        input,
        begin,
        end,
        strides,
        output,
        begin_mask: opts.begin_mask(),
        end_mask: opts.end_mask(),
        shrink_axis_mask: opts.shrink_axis_mask(),
    })
}

/// Lower TFLite CONCATENATION to PspOp::Concatenation
fn lower_concatenation(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs_vec = op.inputs().ok_or("CONCATENATION: no inputs")?;
    let outputs = op.outputs().ok_or("CONCATENATION: no outputs")?;
    let opts = op
        .builtin_options_as_concatenation_options()
        .ok_or("CONCATENATION: no options")?;
    let inputs: Vec<TensorId> = (0..inputs_vec.len())
        .map(|i| tensor_map[inputs_vec.get(i) as usize])
        .collect();
    let output = tensor_map[outputs.get(0) as usize];
    Ok(PspOp::Concatenation {
        inputs,
        output,
        axis: opts.axis(),
    })
}

/// Lower TFLite GATHER to PspOp::Gather
fn lower_gather(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("GATHER: no inputs")?;
    let outputs = op.outputs().ok_or("GATHER: no outputs")?;
    let opts = op
        .builtin_options_as_gather_options()
        .ok_or("GATHER: no options")?;
    let input = tensor_map[inputs.get(0) as usize];
    let indices = tensor_map[inputs.get(1) as usize];
    let output = tensor_map[outputs.get(0) as usize];
    Ok(PspOp::Gather {
        input,
        indices,
        output,
        axis: opts.axis(),
    })
}

/// Lower TFLite REDUCE_PROD/REDUCE_MAX/REDUCE_MIN to PspOp::Reduce
fn lower_reduce(op: &Operator, tensor_map: &[TensorId], builtin_code: BuiltinOperator) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("reduce: no inputs")?;
    let outputs = op.outputs().ok_or("reduce: no outputs")?;
    let reduce_op = match builtin_code {
        BuiltinOperator::REDUCE_PROD => ReduceOp::Prod,
        BuiltinOperator::REDUCE_MAX => ReduceOp::Max,
        BuiltinOperator::REDUCE_MIN => ReduceOp::Min,
        BuiltinOperator::MEAN => ReduceOp::Mean,
        _ => unreachable!(),
    };
    let input = tensor_map[inputs.get(0) as usize];
    let axes = tensor_map[inputs.get(1) as usize];
    let output = tensor_map[outputs.get(0) as usize];
    Ok(PspOp::Reduce {
        op: reduce_op,
        input,
        axes,
        output,
    })
}

/// Lower TFLite RANGE to PspOp::Range
fn lower_range(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("RANGE: no inputs")?;
    let outputs = op.outputs().ok_or("RANGE: no outputs")?;
    let start = tensor_map[inputs.get(0) as usize];
    let limit = tensor_map[inputs.get(1) as usize];
    let delta = tensor_map[inputs.get(2) as usize];
    let output = tensor_map[outputs.get(0) as usize];
    Ok(PspOp::Range {
        start,
        limit,
        delta,
        output,
    })
}

/// Lower TFLite SPLIT_V to PspOp::SplitV
fn lower_split_v(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("SPLIT_V: no inputs")?;
    let output_vec = op.outputs().ok_or("SPLIT_V: no outputs")?;
    let input = tensor_map[inputs.get(0) as usize];
    let size_splits = tensor_map[inputs.get(1) as usize];
    let axis = tensor_map[inputs.get(2) as usize];
    let outputs: Vec<TensorId> = (0..output_vec.len())
        .map(|i| tensor_map[output_vec.get(i) as usize])
        .collect();
    Ok(PspOp::SplitV {
        input,
        size_splits,
        axis,
        outputs,
    })
}

/// Lower TFLite CAST to PspOp::Cast
fn lower_cast(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("CAST: no inputs")?;
    let outputs = op.outputs().ok_or("CAST: no outputs")?;
    let input = tensor_map[inputs.get(0) as usize];
    let output = tensor_map[outputs.get(0) as usize];
    Ok(PspOp::Cast { input, output })
}

/// Lower TFLite PAD to PspOp::Pad
fn lower_pad(op: &Operator, tensor_map: &[TensorId]) -> Result<PspOp, String> {
    let inputs = op.inputs().ok_or("PAD: no inputs")?;
    let outputs = op.outputs().ok_or("PAD: no outputs")?;
    let input = tensor_map[inputs.get(0) as usize];
    let paddings = tensor_map[inputs.get(1) as usize];
    let output = tensor_map[outputs.get(0) as usize];
    Ok(PspOp::Pad { input, paddings, output })
}

/// Convert TFLite tensor type to our DType
fn convert_dtype(t: TensorType) -> DType {
    match t {
        TensorType::FLOAT32 => DType::F32,
        TensorType::INT32 => DType::I32,
        TensorType::INT8 => DType::I8,
        TensorType::UINT8 => DType::U8,
        // Map unknown types to I32 as a fallback — they'll be caught
        // by the op lowering pass if any op actually uses them.
        _ => DType::I32,
    }
}

/// Convert TFLite activation function to our Activation
fn convert_activation(a: ActivationFunctionType) -> Option<Activation> {
    match a {
        ActivationFunctionType::NONE => None,
        ActivationFunctionType::RELU => Some(Activation::Relu),
        ActivationFunctionType::RELU6 => Some(Activation::Relu6),
        other => panic!(
            "unsupported activation function: {:?}",
            other.variant_name().unwrap_or("unknown")
        ),
    }
}

/// Compute SAME padding for convolution.
///
/// Returns (pad_top, pad_bottom, pad_left, pad_right)
fn compute_same_padding(
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> (usize, usize, usize, usize) {
    // Output size for SAME: ceil(input / stride)
    let out_h = (input_h + stride_h - 1) / stride_h;
    let out_w = (input_w + stride_w - 1) / stride_w;

    // Total padding needed
    let pad_h_total = ((out_h - 1) * stride_h + kernel_h).saturating_sub(input_h);
    let pad_w_total = ((out_w - 1) * stride_w + kernel_w).saturating_sub(input_w);

    // Distribute padding (more on bottom/right if odd)
    let pad_top = pad_h_total / 2;
    let pad_bottom = pad_h_total - pad_top;
    let pad_left = pad_w_total / 2;
    let pad_right = pad_w_total - pad_left;

    (pad_top, pad_bottom, pad_left, pad_right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_padding_3x3_stride1() {
        // 28x28 input, 3x3 kernel, stride 1 -> output 28x28
        // Total padding = 3 - 1 = 2, split as 1 top, 1 bottom
        let (top, bottom, left, right) = compute_same_padding(28, 28, 3, 3, 1, 1);
        assert_eq!((top, bottom, left, right), (1, 1, 1, 1));
    }

    #[test]
    fn test_same_padding_5x5_stride1() {
        // 28x28 input, 5x5 kernel, stride 1 -> output 28x28
        // Total padding = 5 - 1 = 4, split as 2 top, 2 bottom
        let (top, bottom, left, right) = compute_same_padding(28, 28, 5, 5, 1, 1);
        assert_eq!((top, bottom, left, right), (2, 2, 2, 2));
    }

    #[test]
    fn test_same_padding_3x3_stride2() {
        // 28x28 input, 3x3 kernel, stride 2 -> output 14x14
        // out = ceil(28/2) = 14
        // pad_total = (14-1)*2 + 3 - 28 = 26 + 3 - 28 = 1
        let (top, bottom, left, right) = compute_same_padding(28, 28, 3, 3, 2, 2);
        assert_eq!((top, bottom, left, right), (0, 1, 0, 1));
    }

    #[test]
    fn test_same_padding_asymmetric() {
        // 7x7 input, 3x3 kernel, stride 2 -> output 4x4
        // out = ceil(7/2) = 4
        // pad_total = (4-1)*2 + 3 - 7 = 6 + 3 - 7 = 2
        let (top, bottom, left, right) = compute_same_padding(7, 7, 3, 3, 2, 2);
        assert_eq!((top, bottom, left, right), (1, 1, 1, 1));
    }

    #[test]
    fn test_same_padding_no_padding_needed() {
        // When kernel fits exactly with no padding
        // 4x4 input, 1x1 kernel, stride 1 -> no padding needed
        let (top, bottom, left, right) = compute_same_padding(4, 4, 1, 1, 1, 1);
        assert_eq!((top, bottom, left, right), (0, 0, 0, 0));
    }

    #[test]
    fn test_same_padding_rectangular() {
        // Non-square input/kernel
        // 28x14 input, 3x5 kernel, stride 1
        let (top, bottom, left, right) = compute_same_padding(28, 14, 3, 5, 1, 1);
        assert_eq!(top + bottom, 2); // 3 - 1 = 2
        assert_eq!(left + right, 4); // 5 - 1 = 4
    }
}
