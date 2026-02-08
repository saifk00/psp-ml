use crate::schema_generated::tflite;
use proc_macro2::{Literal, TokenStream};
use quote::quote;

pub type GenResult<T> = Result<T, String>;

pub struct Generated {
    pub tokens: TokenStream,
    pub data_bytes: Vec<u8>,
    pub data_path: String,
}

struct WeightData {
    statics: TokenStream,
    views: TokenStream,
    data_bytes: Vec<u8>,
    data_path: String,
}

/// Generate the complete MNIST inference code.
pub fn generate_mnist_inference(
    model: &tflite::Model,
    data_path: &str,
) -> GenResult<Generated> {
    // Weight statics (embedded from TFLite buffers)
    let weight_data = generate_weight_statics(model, data_path)?;
    let weight_statics = weight_data.statics;
    let weight_views = weight_data.views;

    // Tensor allocations
    let tensor_allocs = generate_tensor_allocs();

    // Kernel calls
    let kernel_calls = generate_kernel_calls();

    let tokens = quote! {
        //! Generated MNIST inference module

        use psp_ml::kernels::naive::{conv2d_relu, max_pool2d, reshape, fully_connected, fully_connected_relu};

        /// Run MNIST inference on a 28x28 grayscale image
        ///
        /// Input: 784 floats (28x28x1 image, NHWC layout)
        /// Output: 10 floats (logits for digits 0-9)
        pub fn forward(input: &[f32; 784]) -> [f32; 10] {
            #tensor_allocs

            #weight_views

            #kernel_calls

            output
        }

        #weight_statics
    };

    Ok(Generated {
        tokens,
        data_bytes: weight_data.data_bytes,
        data_path: weight_data.data_path,
    })
}

/// Generate static weight declarations.
fn generate_weight_statics(model: &tflite::Model, data_path: &str) -> GenResult<WeightData> {
    let subgraphs = model
        .subgraphs()
        .ok_or_else(|| "Model has no subgraphs".to_string())?;
    if subgraphs.len() == 0 {
        return Err("Model has no subgraph 0".to_string());
    }
    let subgraph = subgraphs.get(0);
    let tensors = subgraph
        .tensors()
        .ok_or_else(|| "Subgraph 0 has no tensors".to_string())?;
    let ops = subgraph
        .operators()
        .ok_or_else(|| "Subgraph 0 has no operators".to_string())?;
    let opcodes = model
        .operator_codes()
        .ok_or_else(|| "Model has no operator codes".to_string())?;
    let buffers = model
        .buffers()
        .ok_or_else(|| "Model has no buffers".to_string())?;

    #[derive(Clone, Copy)]
    struct OpWeights {
        weight: usize,
        bias: Option<usize>,
    }

    let mut convs = Vec::new();
    let mut fcs = Vec::new();

    for (op_index, op) in ops.iter().enumerate() {
        let opcode_index = op.opcode_index() as usize;
        if opcode_index >= opcodes.len() {
            return Err(format!(
                "Operator {op_index} references missing opcode {opcode_index}"
            ));
        }
        let op_name = opcodes
            .get(opcode_index)
            .builtin_code()
            .variant_name()
            .unwrap_or("?");
        let inputs: Vec<i32> = op.inputs().map(|v| v.iter().collect()).unwrap_or_default();

        let parse_weights = |op_name: &str| -> GenResult<OpWeights> {
            if inputs.len() < 2 {
                return Err(format!(
                    "{op_name} op {op_index} expects at least 2 inputs, got {}",
                    inputs.len()
                ));
            }
            let weight_idx = inputs[1];
            if weight_idx < 0 {
                return Err(format!(
                    "{op_name} op {op_index} has invalid weight tensor index {weight_idx}"
                ));
            }
            let bias_idx = if inputs.len() > 2 { inputs[2] } else { -1 };
            let bias = if bias_idx < 0 {
                None
            } else {
                Some(bias_idx as usize)
            };
            Ok(OpWeights {
                weight: weight_idx as usize,
                bias,
            })
        };

        match op_name {
            "CONV_2D" => convs.push(parse_weights(op_name)?),
            "FULLY_CONNECTED" => fcs.push(parse_weights(op_name)?),
            _ => {}
        }
    }

    if convs.len() != 2 {
        return Err(format!("Expected 2 CONV_2D ops, found {}", convs.len()));
    }
    if fcs.len() != 2 {
        return Err(format!(
            "Expected 2 FULLY_CONNECTED ops, found {}",
            fcs.len()
        ));
    }

    struct TensorBytes {
        bytes: Vec<u8>,
        float_len: usize,
    }

    let load_tensor = |tensor_index: usize, label: &str| -> GenResult<TensorBytes> {
        if tensor_index >= tensors.len() {
            return Err(format!(
                "{label} tensor index {tensor_index} out of range (len={})",
                tensors.len()
            ));
        }
        let tensor = tensors.get(tensor_index);
        let tensor_type = tensor.type_();
        if tensor_type != tflite::TensorType::FLOAT32 {
            return Err(format!(
                "{label} tensor {tensor_index} expected FLOAT32, got {}",
                tensor_type.variant_name().unwrap_or("?")
            ));
        }
        let shape: Vec<usize> = tensor
            .shape()
            .map(|s| s.iter().map(|v| v as usize).collect())
            .unwrap_or_default();
        if shape.is_empty() {
            return Err(format!("{label} tensor {tensor_index} has no shape"));
        }
        let expected_len = shape.iter().product::<usize>();
        let buffer_index = tensor.buffer() as usize;
        if buffer_index >= buffers.len() {
            return Err(format!(
                "{label} tensor {tensor_index} refers to missing buffer {buffer_index}"
            ));
        }
        let buffer = buffers.get(buffer_index);
        let data = buffer
            .data()
            .ok_or_else(|| format!("{label} buffer {buffer_index} has no data"))?;
        if data.len() == 0 {
            return Err(format!("{label} buffer {buffer_index} data is empty"));
        }
        let expected_bytes = expected_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{label} tensor {tensor_index} size overflow"))?;
        if data.len() != expected_bytes {
            return Err(format!(
                "{label} tensor {tensor_index} expects {expected_bytes} bytes, got {}",
                data.len()
            ));
        }
        let bytes: Vec<u8> = data.iter().collect();
        Ok(TensorBytes {
            bytes,
            float_len: expected_len,
        })
    };

    let conv1_weight = load_tensor(convs[0].weight, "CONV1_FILTER")?;
    let conv1_bias_idx = convs[0]
        .bias
        .ok_or_else(|| "CONV1_BIAS missing bias tensor".to_string())?;
    let conv1_bias = load_tensor(conv1_bias_idx, "CONV1_BIAS")?;

    let conv2_weight = load_tensor(convs[1].weight, "CONV2_FILTER")?;
    let conv2_bias_idx = convs[1]
        .bias
        .ok_or_else(|| "CONV2_BIAS missing bias tensor".to_string())?;
    let conv2_bias = load_tensor(conv2_bias_idx, "CONV2_BIAS")?;

    let fc1_weight = load_tensor(fcs[0].weight, "FC1_WEIGHTS")?;
    let fc1_bias_idx = fcs[0]
        .bias
        .ok_or_else(|| "FC1_BIAS missing bias tensor".to_string())?;
    let fc1_bias = load_tensor(fc1_bias_idx, "FC1_BIAS")?;

    let fc2_weight = load_tensor(fcs[1].weight, "FC2_WEIGHTS")?;
    let fc2_bias_idx = fcs[1]
        .bias
        .ok_or_else(|| "FC2_BIAS missing bias tensor".to_string())?;
    let fc2_bias = load_tensor(fc2_bias_idx, "FC2_BIAS")?;

    let mut combined = Vec::new();
    let mut offset_floats = 0usize;

    let mut push_tensor = |tensor: &TensorBytes| -> (usize, usize) {
        let offset = offset_floats;
        let len = tensor.float_len;
        combined.extend_from_slice(&tensor.bytes);
        offset_floats += len;
        (offset, len)
    };

    let (conv1_filter_offset, conv1_filter_len) = push_tensor(&conv1_weight);
    let (conv1_bias_offset, conv1_bias_len) = push_tensor(&conv1_bias);
    let (conv2_filter_offset, conv2_filter_len) = push_tensor(&conv2_weight);
    let (conv2_bias_offset, conv2_bias_len) = push_tensor(&conv2_bias);
    let (fc1_weights_offset, fc1_weights_len) = push_tensor(&fc1_weight);
    let (fc1_bias_offset, fc1_bias_len) = push_tensor(&fc1_bias);
    let (fc2_weights_offset, fc2_weights_len) = push_tensor(&fc2_weight);
    let (fc2_bias_offset, fc2_bias_len) = push_tensor(&fc2_bias);

    let total_bytes = combined.len();
    if total_bytes % std::mem::size_of::<f32>() != 0 {
        return Err("Combined weight bytes not aligned to f32".to_string());
    }
    let total_floats = total_bytes / std::mem::size_of::<f32>();

    let data_path_lit = Literal::string(data_path);

    let statics = quote! {
        #[allow(dead_code)]
        #[repr(align(4))]
        struct AlignedBytes<const N: usize>([u8; N]);

        static TENSOR_DATA_BYTES: AlignedBytes<#total_bytes> =
            AlignedBytes(*include_bytes!(#data_path_lit));
        const TENSOR_DATA_FLOATS: usize = #total_floats;

        const CONV1_FILTER_OFFSET: usize = #conv1_filter_offset;
        const CONV1_FILTER_LEN: usize = #conv1_filter_len;
        const CONV1_BIAS_OFFSET: usize = #conv1_bias_offset;
        const CONV1_BIAS_LEN: usize = #conv1_bias_len;
        const CONV2_FILTER_OFFSET: usize = #conv2_filter_offset;
        const CONV2_FILTER_LEN: usize = #conv2_filter_len;
        const CONV2_BIAS_OFFSET: usize = #conv2_bias_offset;
        const CONV2_BIAS_LEN: usize = #conv2_bias_len;
        const FC1_WEIGHTS_OFFSET: usize = #fc1_weights_offset;
        const FC1_WEIGHTS_LEN: usize = #fc1_weights_len;
        const FC1_BIAS_OFFSET: usize = #fc1_bias_offset;
        const FC1_BIAS_LEN: usize = #fc1_bias_len;
        const FC2_WEIGHTS_OFFSET: usize = #fc2_weights_offset;
        const FC2_WEIGHTS_LEN: usize = #fc2_weights_len;
        const FC2_BIAS_OFFSET: usize = #fc2_bias_offset;
        const FC2_BIAS_LEN: usize = #fc2_bias_len;

        fn tensor_data_f32() -> &'static [f32] {
            unsafe {
                core::slice::from_raw_parts(
                    TENSOR_DATA_BYTES.0.as_ptr() as *const f32,
                    TENSOR_DATA_FLOATS,
                )
            }
        }
    };

    let views = quote! {
        let tensor_data = tensor_data_f32();
        let conv1_filter = &tensor_data
            [CONV1_FILTER_OFFSET..CONV1_FILTER_OFFSET + CONV1_FILTER_LEN];
        let conv1_bias = &tensor_data[CONV1_BIAS_OFFSET..CONV1_BIAS_OFFSET + CONV1_BIAS_LEN];
        let conv2_filter = &tensor_data
            [CONV2_FILTER_OFFSET..CONV2_FILTER_OFFSET + CONV2_FILTER_LEN];
        let conv2_bias = &tensor_data[CONV2_BIAS_OFFSET..CONV2_BIAS_OFFSET + CONV2_BIAS_LEN];
        let fc1_weights =
            &tensor_data[FC1_WEIGHTS_OFFSET..FC1_WEIGHTS_OFFSET + FC1_WEIGHTS_LEN];
        let fc1_bias = &tensor_data[FC1_BIAS_OFFSET..FC1_BIAS_OFFSET + FC1_BIAS_LEN];
        let fc2_weights =
            &tensor_data[FC2_WEIGHTS_OFFSET..FC2_WEIGHTS_OFFSET + FC2_WEIGHTS_LEN];
        let fc2_bias = &tensor_data[FC2_BIAS_OFFSET..FC2_BIAS_OFFSET + FC2_BIAS_LEN];
    };

    Ok(WeightData {
        statics,
        views,
        data_bytes: combined,
        data_path: data_path.to_string(),
    })
}

/// Generate intermediate tensor allocations.
fn generate_tensor_allocs() -> TokenStream {
    // Tensor sizes from MNIST model graph (SAME padding):
    // t_conv1_out: [1, 28, 28, 8] = 6272
    // t_pool1_out: [1, 14, 14, 8] = 1568
    // t_conv2_out: [1, 14, 14, 16] = 3136
    // t_pool2_out: [1, 7, 7, 16] = 784
    // t_flatten: [784] = 784
    // t_fc1_out: [64] = 64
    // output: [10] = 10

    let conv1_out_size = 1usize * 28 * 28 * 8;
    let pool1_out_size = 1usize * 14 * 14 * 8;
    let conv2_out_size = 1usize * 14 * 14 * 16;
    let pool2_out_size = 1usize * 7 * 7 * 16;
    let flatten_size = 784usize;
    let fc1_out_size = 64usize;
    let output_size = 10usize;

    quote! {
        // Intermediate tensors
        let mut t_conv1_out = [0.0f32; #conv1_out_size];
        let mut t_pool1_out = [0.0f32; #pool1_out_size];
        let mut t_conv2_out = [0.0f32; #conv2_out_size];
        let mut t_pool2_out = [0.0f32; #pool2_out_size];
        let mut t_flatten = [0.0f32; #flatten_size];
        let mut t_fc1_out = [0.0f32; #fc1_out_size];
        let mut output = [0.0f32; #output_size];
    }
}

/// Generate kernel call sequence (hardcoded for now).
fn generate_kernel_calls() -> TokenStream {
    quote! {
        // Layer 1: Conv2D + ReLU (28x28x1 -> 28x28x8) SAME padding
        conv2d_relu(
            input, [1, 28, 28, 1],
            conv1_filter, [8, 5, 5, 1],
            Some(conv1_bias),
            [1, 1],
            [2, 2],  // SAME padding for 5x5 kernel
            &mut t_conv1_out, [1, 28, 28, 8]
        );

        // Layer 2: MaxPool2D (28x28x8 -> 14x14x8)
        max_pool2d(
            &t_conv1_out, [1, 28, 28, 8],
            [2, 2], [2, 2],
            &mut t_pool1_out, [1, 14, 14, 8]
        );

        // Layer 3: Conv2D + ReLU (14x14x8 -> 14x14x16) SAME padding
        conv2d_relu(
            &t_pool1_out, [1, 14, 14, 8],
            conv2_filter, [16, 5, 5, 8],
            Some(conv2_bias),
            [1, 1],
            [2, 2],  // SAME padding for 5x5 kernel
            &mut t_conv2_out, [1, 14, 14, 16]
        );

        // Layer 4: MaxPool2D (14x14x16 -> 7x7x16)
        max_pool2d(
            &t_conv2_out, [1, 14, 14, 16],
            [2, 2], [2, 2],
            &mut t_pool2_out, [1, 7, 7, 16]
        );

        // Layer 5: Reshape/Flatten (7x7x16 -> 784)
        reshape(&t_pool2_out, &mut t_flatten);

        // Layer 6: FullyConnected + ReLU (784 -> 64)
        fully_connected_relu(
            &t_flatten, 784,
            fc1_weights, fc1_bias,
            &mut t_fc1_out, 64
        );

        // Layer 7: FullyConnected (64 -> 10) - no ReLU on output layer
        fully_connected(
            &t_fc1_out, 64,
            fc2_weights, fc2_bias,
            &mut output, 10
        );
    }
}
