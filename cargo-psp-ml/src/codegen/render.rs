//! Renders a `CodegenPlan` into a `TokenStream`.
//!
//! This is the only codegen file that depends on proc_macro2/quote.

use crate::ir::graph::Graph;
use crate::ir::psp::PspOp;

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

use super::plan::*;
use super::tensor_expr::TensorExprWriter;

pub fn render(plan: &CodegenPlan, graph: &Graph<PspOp>) -> TokenStream {
    let writer = TensorExprWriter::new(graph);

    let weight_statics = render_weight_statics(plan);
    let weight_views = render_weight_views(plan, &writer);
    let tensor_allocs = render_tensor_allocs(plan, &writer);

    let op_tokens: Vec<TokenStream> = plan
        .ops
        .iter()
        .enumerate()
        .map(|(i, op)| render_op_plain(op, i, &writer))
        .collect();
    let plain_calls = quote!(#(#op_tokens)*);

    let timed_calls = render_timed_calls(plan, &writer);
    let op_metadata = render_op_metadata(plan);

    let input_size = plan.input_size;
    let output_size = plan.output_size;
    let output_ident = writer.ident(plan.output_id);

    quote! {
        //! Generated inference module

        #[allow(unused_imports)]
        use psp_ml::kernels::naive::{conv2d, conv2d_relu, max_pool2d, reshape, fully_connected, fully_connected_relu};
        #[allow(unused_imports)]
        use psp_ml::kernels::{im2col, im2col_padded, matmul_bt, matmul_bt_tiled, bias_add, relu};

        pub fn forward(input: &[f32; #input_size]) -> [f32; #output_size] {
            #tensor_allocs

            #weight_views

            #plain_calls

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

        #weight_statics
    }
}

// ---------------------------------------------------------------------------
// Weight statics
// ---------------------------------------------------------------------------

fn render_weight_statics(plan: &CodegenPlan) -> TokenStream {
    let total_bytes = plan.blob_bytes;
    let total_floats = plan.blob_floats;

    let mut const_entries = Vec::new();
    for alloc in &plan.allocs {
        if let TensorAlloc::Constant {
            id,
            float_offset,
            float_len,
        } = alloc
        {
            let offset_ident = Ident::new(&format!("T_{id}_OFFSET"), Span::call_site());
            let len_ident = Ident::new(&format!("T_{id}_LEN"), Span::call_site());
            const_entries.push(quote! {
                const #offset_ident: usize = #float_offset;
                const #len_ident: usize = #float_len;
            });
        }
    }

    quote! {
        #[allow(dead_code)]
        #[repr(align(16))]
        struct AlignedBytes<const N: usize>([u8; N]);

        /// 16-byte aligned f32 array for VFPU `lv.q`/`sv.q`.
        #[repr(C, align(16))]
        struct Aligned16<const N: usize>([f32; N]);

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
    }
}

// ---------------------------------------------------------------------------
// Weight views
// ---------------------------------------------------------------------------

fn render_weight_views(plan: &CodegenPlan, writer: &TensorExprWriter) -> TokenStream {
    let mut view_entries = Vec::new();
    for alloc in &plan.allocs {
        if let TensorAlloc::Constant { id, .. } = alloc {
            let var_ident = writer.ident(*id);
            let offset_ident = writer.offset_const(*id);
            let len_ident = writer.len_const(*id);
            view_entries.push(quote! {
                let #var_ident = &tensor_data[#offset_ident..#offset_ident + #len_ident];
            });
        }
    }

    quote! {
        let tensor_data = tensor_data_f32();
        #(#view_entries)*
    }
}

// ---------------------------------------------------------------------------
// Tensor allocations (intermediates + output)
// ---------------------------------------------------------------------------

fn render_tensor_allocs(plan: &CodegenPlan, writer: &TensorExprWriter) -> TokenStream {
    let mut entries = Vec::new();
    for alloc in &plan.allocs {
        match alloc {
            TensorAlloc::Intermediate { id, size } => {
                let buf_static = writer.buf_static(*id);
                let ident = writer.ident(*id);
                entries.push(quote! {
                    static mut #buf_static: Aligned16<#size> = Aligned16([0.0f32; #size]);
                    let #ident = unsafe {
                        core::slice::from_raw_parts_mut(
                            core::ptr::addr_of_mut!(#buf_static) as *mut f32,
                            #size,
                        )
                    };
                });
            }
            TensorAlloc::Output { id, size } => {
                let ident = writer.ident(*id);
                entries.push(quote! {
                    let mut #ident = [0.0f32; #size];
                });
            }
            TensorAlloc::Constant { .. } => {}
        }
    }
    quote!(#(#entries)*)
}

// ---------------------------------------------------------------------------
// Scratch buffer rendering
// ---------------------------------------------------------------------------

fn render_scratch(op: &OpPlan, op_idx: usize, writer: &TensorExprWriter) -> TokenStream {
    let mut entries = Vec::new();

    for (s_idx, scratch) in op.scratch.iter().enumerate() {
        let size = scratch.size;

        // Choose ident names based on scratch purpose
        // Scratch 0 = im2col scratch, Scratch 1 = padded weights
        let (static_name, local_name) = if s_idx == 0 {
            (
                format!("CONV_SCRATCH_{op_idx}"),
                format!("conv_scratch_{op_idx}"),
            )
        } else {
            (format!("PADDED_W_{op_idx}"), format!("padded_w_{op_idx}"))
        };

        let static_ident = Ident::new(&static_name, Span::call_site());
        let local_ident = Ident::new(&local_name, Span::call_site());

        entries.push(quote! {
            static mut #static_ident: Aligned16<#size> = Aligned16([0.0f32; #size]);
            let #local_ident = unsafe {
                core::slice::from_raw_parts_mut(
                    core::ptr::addr_of_mut!(#static_ident) as *mut f32,
                    #size,
                )
            };
        });

        // Load data if needed
        if let Some(load) = &scratch.load_from {
            let src_expr = writer.read(load.source);
            match &load.copy {
                CopyStrategy::BulkCopy => {
                    entries.push(quote! {
                        #local_ident.copy_from_slice(#src_expr);
                    });
                }
                CopyStrategy::RowPadded {
                    num_rows,
                    src_stride,
                    dst_stride,
                } => {
                    entries.push(quote! {
                        for row in 0..#num_rows {
                            #local_ident[row * #dst_stride..row * #dst_stride + #src_stride]
                                .copy_from_slice(&#src_expr[row * #src_stride..(row + 1) * #src_stride]);
                        }
                    });
                }
            }
        }
    }

    quote!(#(#entries)*)
}

// ---------------------------------------------------------------------------
// Kernel call rendering
// ---------------------------------------------------------------------------

fn render_kernel_call(
    kernel: &KernelCall,
    op_scratch: &[ScratchBuffer],
    op_idx: usize,
    writer: &TensorExprWriter,
) -> TokenStream {
    let _ = op_scratch; // used for scratch ident naming context
    match kernel {
        KernelCall::Conv2d {
            input,
            filter,
            bias,
            stride,
            padding,
            output,
            has_relu,
        } => {
            let input_expr = writer.read(input.id);
            let input_shape_tok = shape_tokens(&input.shape);
            let filter_expr = writer.read(filter.id);
            let filter_shape_tok = shape_tokens(&filter.shape);
            let output_expr = writer.write(output.id);
            let output_shape_tok = shape_tokens(&output.shape);
            let [sh, sw] = stride;
            let [ph, pw] = padding;

            let bias_tok = match bias {
                Some(b) => {
                    let b_expr = writer.read(*b);
                    quote!(Some(#b_expr))
                }
                None => quote!(None),
            };

            if *has_relu {
                quote! {
                    conv2d_relu(
                        #input_expr, #input_shape_tok,
                        #filter_expr, #filter_shape_tok,
                        #bias_tok,
                        [#sh, #sw],
                        [#ph, #pw],
                        #output_expr, #output_shape_tok
                    );
                }
            } else {
                quote! {
                    conv2d(
                        #input_expr, #input_shape_tok,
                        #filter_expr, #filter_shape_tok,
                        #bias_tok,
                        [#sh, #sw],
                        [#ph, #pw],
                        #output_expr, #output_shape_tok
                    );
                }
            }
        }

        KernelCall::Im2colPadded {
            input,
            kernel_size,
            padding,
            output_hw,
            output: _scratch_idx,
        } => {
            let input_expr = writer.read(input.id);
            let input_shape_tok = shape_tokens(&input.shape);
            let [kh, kw] = kernel_size;
            let [ph, pw] = padding;
            let [ho, wo] = output_hw;
            let scratch_ident =
                Ident::new(&format!("conv_scratch_{op_idx}"), Span::call_site());

            quote! {
                im2col_padded(
                    #input_expr, #input_shape_tok,
                    [#kh, #kw], [#ph, #pw], [#ho, #wo],
                    #scratch_ident
                );
            }
        }

        KernelCall::MatmulBtTiled {
            a: _a_scratch,
            b: _b_scratch,
            output,
            m_tiles,
            k_tiles,
            n_tiles,
        } => {
            let a_ident = Ident::new(&format!("conv_scratch_{op_idx}"), Span::call_site());
            let b_ident = Ident::new(&format!("padded_w_{op_idx}"), Span::call_site());
            let output_expr = writer.write(*output);

            quote! {
                matmul_bt_tiled(#a_ident, #b_ident, #output_expr, #m_tiles, #k_tiles, #n_tiles);
            }
        }

        KernelCall::BiasAdd {
            output,
            bias,
            rows,
            cols,
        } => {
            let output_expr = writer.write(*output);
            let bias_expr = writer.read(*bias);
            quote! { bias_add(#output_expr, #bias_expr, #rows, #cols); }
        }

        KernelCall::Relu { output } => {
            let output_expr = writer.write(*output);
            quote! { relu(#output_expr); }
        }

        KernelCall::MaxPool2d {
            input,
            output,
        } => {
            let input_expr = writer.read(input.id);
            let input_shape_tok = shape_tokens(&input.shape);
            let output_expr = writer.write(output.id);
            let output_shape_tok = shape_tokens(&output.shape);
            quote! {
                max_pool2d(
                    #input_expr, #input_shape_tok,
                    [2, 2], [2, 2],
                    #output_expr, #output_shape_tok
                );
            }
        }

        KernelCall::Reshape { input, output } => {
            let input_expr = writer.read(*input);
            let output_expr = writer.write(*output);
            quote! { reshape(#input_expr, #output_expr); }
        }

        KernelCall::FullyConnected {
            input,
            in_features,
            weights,
            bias,
            output,
            out_features,
            has_relu,
        } => {
            let input_expr = writer.read(*input);
            let weight_expr = writer.read(*weights);
            let bias_expr = writer.read(*bias);
            let output_expr = writer.write(*output);
            if *has_relu {
                quote! {
                    fully_connected_relu(
                        #input_expr, #in_features,
                        #weight_expr, #bias_expr,
                        #output_expr, #out_features
                    );
                }
            } else {
                quote! {
                    fully_connected(
                        #input_expr, #in_features,
                        #weight_expr, #bias_expr,
                        #output_expr, #out_features
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Plain (untimed) op rendering
// ---------------------------------------------------------------------------

fn render_op_plain(op: &OpPlan, op_idx: usize, writer: &TensorExprWriter) -> TokenStream {
    let scratch = render_scratch(op, op_idx, writer);
    let calls: Vec<TokenStream> = op
        .sub_ops
        .iter()
        .flat_map(|sub| {
            sub.kernels
                .iter()
                .map(|k| render_kernel_call(k, &op.scratch, op_idx, writer))
        })
        .collect();
    quote! {
        #scratch
        #(#calls)*
    }
}

// ---------------------------------------------------------------------------
// Timed op rendering
// ---------------------------------------------------------------------------

fn render_timed_calls(plan: &CodegenPlan, writer: &TensorExprWriter) -> TokenStream {
    let mut sub_op_idx: usize = 0;
    let mut entries = Vec::new();

    for (op_idx, op) in plan.ops.iter().enumerate() {
        // Scratch setup is untimed
        let scratch = render_scratch(op, op_idx, writer);
        entries.push(scratch);

        for sub in &op.sub_ops {
            let calls: Vec<TokenStream> = sub
                .kernels
                .iter()
                .map(|k| render_kernel_call(k, &op.scratch, op_idx, writer))
                .collect();
            let i = sub_op_idx;
            sub_op_idx += 1;
            entries.push(quote! {
                let __t0 = get_tick();
                #(#calls)*
                op_ticks[#i] += get_tick() - __t0;
            });
        }
    }

    quote!(#(#entries)*)
}

// ---------------------------------------------------------------------------
// Op metadata
// ---------------------------------------------------------------------------

fn render_op_metadata(plan: &CodegenPlan) -> TokenStream {
    let names: Vec<&str> = plan
        .ops
        .iter()
        .flat_map(|op| op.sub_ops.iter().map(|s| s.name.as_str()))
        .collect();
    let num_ops = names.len();
    quote! {
        pub const NUM_OPS: usize = #num_ops;
        pub const OP_NAMES: [&str; NUM_OPS] = [#(#names),*];
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn shape_tokens(shape: &[usize]) -> TokenStream {
    quote!([#(#shape),*])
}
