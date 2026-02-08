use crate::ir::PspModel;
use crate::ir::graph::{Graph, TensorId, TensorKind};
use crate::ir::psp::{Activation, PspOp};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub type GenResult<T> = Result<T, String>;

pub struct Generated {
    pub tokens: TokenStream,
    pub data_bytes: Vec<u8>,
    pub data_path: String,
}

pub fn generate_code(model: &PspModel) -> GenResult<Generated> {
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

    let total_bytes = model.model_data.len();
    let total_floats = total_bytes / std::mem::size_of::<f32>();

    let (weight_consts, weight_views) = gen_weight_code(graph, total_bytes, total_floats)?;
    let tensor_allocs = gen_allocs(graph)?;
    let kernel_calls = gen_kernel_calls(graph, input_id)?;

    let output_ident = tensor_ident(output_id);

    let tokens = quote! {
        //! Generated inference module

        #[allow(unused_imports)]
        use psp_ml::kernels::naive::{conv2d, conv2d_relu, max_pool2d, reshape, fully_connected, fully_connected_relu};

        pub fn forward(input: &[f32; #input_size]) -> [f32; #output_size] {
            #tensor_allocs

            #weight_views

            #kernel_calls

            #output_ident
        }

        #weight_consts
    };

    Ok(Generated {
        tokens,
        data_bytes: model.model_data.clone(),
        data_path: "weights.bin".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Helpers for generate_code
// ---------------------------------------------------------------------------

fn tensor_ident(id: TensorId) -> Ident {
    Ident::new(&format!("t_{id}"), Span::call_site())
}

fn shape_tokens(shape: &[usize]) -> TokenStream {
    quote!([#(#shape),*])
}

/// Read-reference expression for a tensor in a kernel call.
///
/// - Input tensor → `input` (the function parameter)
/// - Constant tensor → `t_{id}` (already a `&[f32]` slice)
/// - Intermediate/Output → `&t_{id}` (borrow local array)
fn tensor_read_expr(graph: &Graph<PspOp>, id: TensorId, input_id: TensorId) -> TokenStream {
    if id == input_id {
        return quote!(input);
    }
    let ident = tensor_ident(id);
    match &graph.tensor(id).kind {
        TensorKind::Constant { .. } => quote!(#ident),
        _ => quote!(&#ident),
    }
}

/// Write-reference expression: `&mut t_{id}`.
fn tensor_write_expr(id: TensorId) -> TokenStream {
    let ident = tensor_ident(id);
    quote!(&mut #ident)
}

/// Generate statics (weight embedding) and weight view bindings.
fn gen_weight_code(
    graph: &Graph<PspOp>,
    total_bytes: usize,
    total_floats: usize,
) -> GenResult<(TokenStream, TokenStream)> {
    let mut const_entries = Vec::new();
    let mut view_entries = Vec::new();

    for tensor in &graph.tensors {
        if let TensorKind::Constant { offset, len } = &tensor.kind {
            let sz = std::mem::size_of::<f32>();
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

            let float_offset = offset / sz;
            let float_len = len / sz;

            let offset_ident = Ident::new(&format!("T_{}_OFFSET", tensor.id), Span::call_site());
            let len_ident = Ident::new(&format!("T_{}_LEN", tensor.id), Span::call_site());
            let var_ident = tensor_ident(tensor.id);

            const_entries.push(quote! {
                const #offset_ident: usize = #float_offset;
                const #len_ident: usize = #float_len;
            });

            view_entries.push(quote! {
                let #var_ident = &tensor_data[#offset_ident..#offset_ident + #len_ident];
            });
        }
    }

    let statics = quote! {
        #[allow(dead_code)]
        #[repr(align(4))]
        struct AlignedBytes<const N: usize>([u8; N]);

        static TENSOR_DATA_BYTES: AlignedBytes<#total_bytes> =
            AlignedBytes(*include_bytes!("weights.bin"));
        const TENSOR_DATA_FLOATS: usize = #total_floats;

        #(#const_entries)*

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
        #(#view_entries)*
    };

    Ok((statics, views))
}

/// Generate intermediate and output tensor allocations as zero-filled arrays.
fn gen_allocs(graph: &Graph<PspOp>) -> GenResult<TokenStream> {
    let mut entries = Vec::new();

    for tensor in &graph.tensors {
        match &tensor.kind {
            TensorKind::Intermediate | TensorKind::Output => {
                let ident = tensor_ident(tensor.id);
                let size = tensor.shape.iter().product::<usize>();
                entries.push(quote! {
                    let mut #ident = [0.0f32; #size];
                });
            }
            _ => {}
        }
    }

    Ok(quote!(#(#entries)*))
}

/// Generate the kernel call sequence from the op list.
fn gen_kernel_calls(graph: &Graph<PspOp>, input_id: TensorId) -> GenResult<TokenStream> {
    let mut calls = Vec::new();

    for (i, op) in graph.ops.iter().enumerate() {
        let call = match op {
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

                let input_expr = tensor_read_expr(graph, *input, input_id);
                let input_shape = shape_tokens(in_shape);
                let filter_expr = tensor_read_expr(graph, *weights, input_id);
                let filter_shape = shape_tokens(w_shape);
                let output_expr = tensor_write_expr(*output);
                let output_shape = shape_tokens(out_shape);

                let stride_h = params.stride_h;
                let stride_w = params.stride_w;
                let pad_h = params.pad_top;
                let pad_w = params.pad_left;

                let bias_expr = match bias {
                    Some(b) => {
                        let b_expr = tensor_read_expr(graph, *b, input_id);
                        quote!(Some(#b_expr))
                    }
                    None => quote!(None),
                };

                match params.fused_activation {
                    Some(Activation::Relu) => {
                        quote! {
                            conv2d_relu(
                                #input_expr, #input_shape,
                                #filter_expr, #filter_shape,
                                #bias_expr,
                                [#stride_h, #stride_w],
                                [#pad_h, #pad_w],
                                #output_expr, #output_shape
                            );
                        }
                    }
                    None => {
                        quote! {
                            conv2d(
                                #input_expr, #input_shape,
                                #filter_expr, #filter_shape,
                                #bias_expr,
                                [#stride_h, #stride_w],
                                [#pad_h, #pad_w],
                                #output_expr, #output_shape
                            );
                        }
                    }
                    Some(Activation::Relu6) => {
                        return Err(format!("Op {i}: Relu6 not supported for Conv2d"));
                    }
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

                let input_expr = tensor_read_expr(graph, *input, input_id);
                let in_features = graph.tensor(*input).shape.iter().product::<usize>();
                let weight_expr = tensor_read_expr(graph, *weights, input_id);
                let bias_expr = tensor_read_expr(graph, bias_id, input_id);
                let output_expr = tensor_write_expr(*output);
                let out_features = graph.tensor(*output).shape.iter().product::<usize>();

                match fused_activation.fused_activation {
                    Some(Activation::Relu) => {
                        quote! {
                            fully_connected_relu(
                                #input_expr, #in_features,
                                #weight_expr, #bias_expr,
                                #output_expr, #out_features
                            );
                        }
                    }
                    None => {
                        quote! {
                            fully_connected(
                                #input_expr, #in_features,
                                #weight_expr, #bias_expr,
                                #output_expr, #out_features
                            );
                        }
                    }
                    Some(Activation::Relu6) => {
                        return Err(format!("Op {i}: Relu6 not supported for FullyConnected"));
                    }
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

                let input_expr = tensor_read_expr(graph, *input, input_id);
                let input_shape = shape_tokens(in_shape);
                let output_expr = tensor_write_expr(*output);
                let output_shape = shape_tokens(out_shape);

                quote! {
                    max_pool2d(
                        #input_expr, #input_shape,
                        [2, 2], [2, 2],
                        #output_expr, #output_shape
                    );
                }
            }

            PspOp::Reshape { input, output } => {
                let input_expr = tensor_read_expr(graph, *input, input_id);
                let output_expr = tensor_write_expr(*output);

                quote! {
                    reshape(#input_expr, #output_expr);
                }
            }

            PspOp::Softmax { .. } => {
                return Err(format!("Op {i}: Softmax kernel not yet implemented"));
            }
        };

        calls.push(call);
    }

    Ok(quote!(#(#calls)*))
}
