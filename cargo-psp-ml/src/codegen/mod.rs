use crate::ir::graph::{Graph, TensorId, TensorKind};
use crate::ir::psp::{Activation, PspOp};
use crate::ir::PspModel;
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
    let op_infos = gen_op_infos(graph, input_id)?;

    let plain_calls: Vec<&TokenStream> = op_infos.iter().map(|i| &i.call).collect();
    let timed_calls = gen_timed_calls(&op_infos);
    let op_metadata = gen_op_metadata(&op_infos);

    let output_ident = tensor_ident(output_id);

    let tokens = quote! {
        //! Generated inference module

        #[allow(unused_imports)]
        use psp_ml::kernels::naive::{conv2d, conv2d_relu, max_pool2d, reshape, fully_connected, fully_connected_relu};
        #[allow(unused_imports)]
        use psp_ml::kernels::{im2col, im2col_padded, matmul_bt, matmul_bt_tiled, bias_add, relu};

        pub fn forward(input: &[f32; #input_size]) -> [f32; #output_size] {
            #tensor_allocs

            #weight_views

            #(#plain_calls)*

            #output_ident
        }

        /// Instrumented inference: accumulates per-op tick deltas into `op_ticks`.
        pub fn forward_timed(
            input: &[f32; #input_size],
            op_ticks: &mut [u64; NUM_OPS],
            get_tick: fn() -> u64,
        ) -> [f32; #output_size] {
            #tensor_allocs

            #weight_views

            #timed_calls

            #output_ident
        }

        #op_metadata

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

struct OpInfo {
    name: String,
    call: TokenStream,
}

const VFPU_Q: usize = 4;

const fn ceil_vfpu_q(x: usize) -> usize {
    (x + VFPU_Q - 1) & !(VFPU_Q - 1)
}

/// Shared state extracted from a Conv2d op for codegen.
struct Conv2dArgs {
    input_expr: TokenStream,
    input_shape: TokenStream,
    filter_expr: TokenStream,
    output_expr: TokenStream,
    output_id: TensorId,
    in_shape: [usize; 4],
    w_shape: [usize; 4],
    out_shape: [usize; 4],
    stride: [usize; 2],
    padding: [usize; 2],
    has_relu: bool,
    bias_expr: Option<TokenStream>,
}

/// Generate naive conv2d/conv2d_relu call.
fn gen_conv2d_naive(args: &Conv2dArgs) -> (String, TokenStream) {
    let Conv2dArgs {
        input_expr, input_shape, filter_expr, output_expr,
        w_shape, out_shape, stride, padding, has_relu, bias_expr, ..
    } = args;
    let filter_shape = shape_tokens(w_shape);
    let output_shape = shape_tokens(out_shape);
    let [stride_h, stride_w] = stride;
    let [pad_h, pad_w] = padding;

    let bias_tok = match bias_expr {
        Some(b) => quote!(Some(#b)),
        None => quote!(None),
    };

    if *has_relu {
        ("conv2d_relu".into(), quote! {
            conv2d_relu(
                #input_expr, #input_shape,
                #filter_expr, #filter_shape,
                #bias_tok,
                [#stride_h, #stride_w],
                [#pad_h, #pad_w],
                #output_expr, #output_shape
            );
        })
    } else {
        ("conv2d".into(), quote! {
            conv2d(
                #input_expr, #input_shape,
                #filter_expr, #filter_shape,
                #bias_tok,
                [#stride_h, #stride_w],
                [#pad_h, #pad_w],
                #output_expr, #output_shape
            );
        })
    }
}

/// Generate VFPU im2col_padded + matmul_bt_tiled conv2d path.
///
/// All GEMM dimensions are padded to multiples of VFPU_Q at compile time.
/// Stride must be [1,1] — non-unit strides are not supported for the VFPU path.
fn gen_conv2d_vfpu(args: &Conv2dArgs, op_idx: usize) -> GenResult<(String, TokenStream)> {
    if args.stride != [1, 1] {
        return Err(format!(
            "Op {op_idx}: VFPU conv2d requires stride [1,1], got {:?}",
            args.stride
        ));
    }

    let Conv2dArgs {
        input_expr, input_shape, filter_expr, output_id,
        in_shape, w_shape, out_shape, padding, has_relu, bias_expr, ..
    } = args;

    let [n, _, _, ci] = in_shape;
    let [co, kh, kw, _] = w_shape;
    let [_, ho, wo, _] = out_shape;
    let [pad_h, pad_w] = padding;

    let gemm_m = n * ho * wo;
    let gemm_k = kh * kw * ci;
    let k_padded = ceil_vfpu_q(gemm_k);
    let m_padded = ceil_vfpu_q(gemm_m);
    let n_padded = ceil_vfpu_q(*co);

    // Assert M and N are already VFPU_Q-aligned (defer padding support)
    if m_padded != gemm_m {
        return Err(format!(
            "Op {op_idx}: VFPU conv2d requires M ({gemm_m}) to be a multiple of {VFPU_Q}"
        ));
    }
    if n_padded != *co {
        return Err(format!(
            "Op {op_idx}: VFPU conv2d requires N ({co}) to be a multiple of {VFPU_Q}"
        ));
    }

    let m_tiles = m_padded / VFPU_Q;
    let k_tiles = k_padded / VFPU_Q;
    let n_tiles = n_padded / VFPU_Q;

    let scratch_size = m_padded * k_padded;
    let scratch_ident = Ident::new(&format!("conv_scratch_{op_idx}"), Span::call_site());
    let scratch_static = Ident::new(&format!("CONV_SCRATCH_{op_idx}"), Span::call_site());

    let output_ident = tensor_ident(*output_id);

    // Weight expression: pad K if needed, otherwise use original tensor directly
    let (weight_setup, weight_ref) = if k_padded != gemm_k {
        let padded_size = co * k_padded;
        let padded_static = Ident::new(&format!("PADDED_W_{op_idx}"), Span::call_site());
        let padded_ident = Ident::new(&format!("padded_w_{op_idx}"), Span::call_site());

        let setup = quote! {
            static mut #padded_static: [f32; #padded_size] = [0.0f32; #padded_size];
            let #padded_ident = unsafe { &mut *::core::ptr::addr_of_mut!(#padded_static) };
            for row in 0..#co {
                #padded_ident[row * #k_padded..row * #k_padded + #gemm_k]
                    .copy_from_slice(&#filter_expr[row * #gemm_k..(row + 1) * #gemm_k]);
            }
        };
        (setup, quote!(#padded_ident))
    } else {
        (quote! {}, quote!(#filter_expr))
    };

    let bias_code = match bias_expr {
        Some(b) => quote! { bias_add(&mut #output_ident, #b, #gemm_m, #co); },
        None => quote! {},
    };

    let relu_code = if *has_relu {
        quote! { relu(&mut #output_ident); }
    } else {
        quote! {}
    };

    let name = if *has_relu { "conv2d_relu" } else { "conv2d" };

    Ok((name.into(), quote! {
        {
            static mut #scratch_static: [f32; #scratch_size] = [0.0f32; #scratch_size];
            let #scratch_ident = unsafe { &mut *::core::ptr::addr_of_mut!(#scratch_static) };
            #weight_setup
            im2col_padded(
                #input_expr, #input_shape,
                [#kh, #kw], [#pad_h, #pad_w], [#ho, #wo],
                #scratch_ident
            );
            matmul_bt_tiled(
                #scratch_ident, #weight_ref,
                &mut #output_ident,
                #m_tiles, #k_tiles, #n_tiles
            );
        }
        #bias_code
        #relu_code
    }))
}

/// Generate per-op metadata and kernel calls.
fn gen_op_infos(graph: &Graph<PspOp>, input_id: TensorId) -> GenResult<Vec<OpInfo>> {
    let mut infos = Vec::new();
    let use_vfpu_conv2d = true;

    for (i, op) in graph.ops.iter().enumerate() {
        let (name, call) = match op {
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

                let args = Conv2dArgs {
                    input_expr: tensor_read_expr(graph, *input, input_id),
                    input_shape: shape_tokens(in_shape),
                    filter_expr: tensor_read_expr(graph, *weights, input_id),
                    output_expr: tensor_write_expr(*output),
                    output_id: *output,
                    in_shape: [in_shape[0], in_shape[1], in_shape[2], in_shape[3]],
                    w_shape: [w_shape[0], w_shape[1], w_shape[2], w_shape[3]],
                    out_shape: [out_shape[0], out_shape[1], out_shape[2], out_shape[3]],
                    stride: [params.stride_h, params.stride_w],
                    padding: [params.pad_top, params.pad_left],
                    has_relu: matches!(params.fused_activation, Some(Activation::Relu)),
                    bias_expr: bias.map(|b| tensor_read_expr(graph, b, input_id)),
                };

                if use_vfpu_conv2d {
                    gen_conv2d_vfpu(&args, i)?
                } else {
                    gen_conv2d_naive(&args)
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
                        ("fully_connected_relu".into(), quote! {
                            fully_connected_relu(
                                #input_expr, #in_features,
                                #weight_expr, #bias_expr,
                                #output_expr, #out_features
                            );
                        })
                    }
                    None => {
                        ("fully_connected".into(), quote! {
                            fully_connected(
                                #input_expr, #in_features,
                                #weight_expr, #bias_expr,
                                #output_expr, #out_features
                            );
                        })
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

                ("max_pool2d".into(), quote! {
                    max_pool2d(
                        #input_expr, #input_shape,
                        [2, 2], [2, 2],
                        #output_expr, #output_shape
                    );
                })
            }

            PspOp::Reshape { input, output } => {
                let input_expr = tensor_read_expr(graph, *input, input_id);
                let output_expr = tensor_write_expr(*output);

                ("reshape".into(), quote! {
                    reshape(#input_expr, #output_expr);
                })
            }

            PspOp::Softmax { .. } => {
                return Err(format!("Op {i}: Softmax kernel not yet implemented"));
            }
        };

        infos.push(OpInfo { name, call });
    }

    Ok(infos)
}

/// Wrap each op call with timing instrumentation.
fn gen_timed_calls(infos: &[OpInfo]) -> TokenStream {
    let entries: Vec<TokenStream> = infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            let call = &info.call;
            quote! {
                let __t0 = get_tick();
                #call
                op_ticks[#i] += get_tick() - __t0;
            }
        })
        .collect();
    quote!(#(#entries)*)
}

/// Generate NUM_OPS and OP_NAMES constants.
fn gen_op_metadata(infos: &[OpInfo]) -> TokenStream {
    let num_ops = infos.len();
    let names: Vec<&str> = infos.iter().map(|i| i.name.as_str()).collect();
    quote! {
        pub const NUM_OPS: usize = #num_ops;
        pub const OP_NAMES: [&str; NUM_OPS] = [#(#names),*];
    }
}
