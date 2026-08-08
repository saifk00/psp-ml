//! Renders a `CodegenPlan` into a `TokenStream`.
//!
//! This is the only codegen file that depends on proc_macro2/quote.

use crate::ir::graph::Graph;
use crate::ir::psp::{PoolType, PspOp};

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

use std::collections::HashSet;

use super::arena::{extract_tensor_refs, ArenaLayout, ArenaSlot, RefKind};
use super::plan::*;
use super::tensor_expr::TensorExprWriter;

pub fn render(plan: &CodegenPlan, graph: &Graph<PspOp>) -> TokenStream {
    let writer = TensorExprWriter::new(graph);
    let arena = plan.arena.as_ref();

    let weight_statics = render_weight_statics(plan);
    let weight_views = render_weight_views(plan, &writer);
    let arena_static = render_arena_static(arena);
    let tensor_allocs = render_tensor_allocs(plan, &writer, arena, plan.stream.as_ref());

    let plain_calls = render_all_ops_plain(plan, &writer, arena);
    let timed_calls = render_all_ops_timed(plan, &writer, arena);
    let profiled_calls = render_all_ops_profiled(plan, &writer, arena);
    let debug_calls = render_all_ops_debug(plan, &writer, arena);
    let op_metadata = render_op_metadata(plan);

    let input_size = plan.input_size;
    let output_size = plan.output_size;
    let output_ident = writer.ident(plan.output_id);

    quote! {
        // Generated inference module

        #[allow(unused_imports)]
        use psp_rt::kernels::naive::*;
        #[allow(unused_imports)]
        use psp_rt::kernels::*;

        #arena_static

        pub const OUTPUT_SIZE: usize = #output_size;

        pub fn forward(input: &[f32; #input_size], output: &mut [f32; #output_size]) {
            #tensor_allocs

            #weight_views

            #plain_calls

            output.copy_from_slice(&#output_ident);
        }

        /// Instrumented inference: accumulates per-op tick deltas into `op_ticks`.
        pub fn forward_timed(
            input: &[f32; #input_size],
            output: &mut [f32; #output_size],
            op_ticks: &mut [u64; NUM_OPS],
            get_tick: fn() -> u64,
        ) {
            #tensor_allocs

            #weight_views

            #timed_calls

            output.copy_from_slice(&#output_ident);
        }

        /// Instrumented inference with per-op hardware profiling counters.
        pub fn forward_profiled(
            input: &[f32; #input_size],
            output: &mut [f32; #output_size],
            op_ticks: &mut [u64; NUM_OPS],
            #[allow(unused)] op_profile: &mut [psp_rt::profiler::OpProfileStats; NUM_OPS],
            get_tick: fn() -> u64,
        ) {
            #tensor_allocs

            #weight_views

            #profiled_calls

            output.copy_from_slice(&#output_ident);
        }

        /// Debug inference: invokes `tap(op_idx, tensor_id, values)` after
        /// each op for every tensor it wrote. Tensor ids correspond to TFLite
        /// tensor indices, so taps can be diffed against a TFLite reference
        /// run layer by layer. Host-only (code size).
        #[cfg(not(target_os = "psp"))]
        #[allow(dead_code)]
        pub fn forward_debug(
            input: &[f32; #input_size],
            output: &mut [f32; #output_size],
            tap: &mut dyn FnMut(usize, usize, &[f32]),
        ) {
            #tensor_allocs

            #weight_views

            #debug_calls

            output.copy_from_slice(&#output_ident);
        }

        #op_metadata

        #weight_statics
    }
}

/// Render all ops (debug mode): plain calls plus a `tap` after each op for
/// every tensor it wrote. Falls back to plain (no taps) for streamed plans.
fn render_all_ops_debug(
    plan: &CodegenPlan,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
) -> TokenStream {
    if plan.stream.is_some() {
        return render_all_ops_plain(plan, writer, arena);
    }
    let op_tokens: Vec<TokenStream> = plan
        .ops
        .iter()
        .enumerate()
        .map(|(i, op)| {
            let body = render_op_plain(op, i, writer, arena);
            let mut seen = HashSet::new();
            let mut taps = Vec::new();
            for sub in &op.sub_ops {
                for kernel in &sub.kernels {
                    for r in extract_tensor_refs(kernel) {
                        if matches!(r.kind, RefKind::Write) && seen.insert(r.tensor_id) {
                            let expr = writer.read(r.tensor_id);
                            let tid = r.tensor_id;
                            taps.push(quote! { tap(#i, #tid, &#expr); });
                        }
                    }
                }
            }
            quote! { #body #(#taps)* }
        })
        .collect();
    quote!(#(#op_tokens)*)
}

// ---------------------------------------------------------------------------
// Op rendering with frame streaming support
// ---------------------------------------------------------------------------

/// Render all ops (plain mode). Wraps frame-section ops in a streaming loop
/// when streaming is active.
fn render_all_ops_plain(
    plan: &CodegenPlan,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
) -> TokenStream {
    if let Some(stream) = &plan.stream {
        render_streamed_ops(plan, stream, writer, arena, TimingMode::Plain)
    } else {
        let op_tokens: Vec<TokenStream> = plan
            .ops
            .iter()
            .enumerate()
            .map(|(i, op)| render_op_plain(op, i, writer, arena))
            .collect();
        quote!(#(#op_tokens)*)
    }
}

/// Render all ops (timed mode).
fn render_all_ops_timed(
    plan: &CodegenPlan,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
) -> TokenStream {
    if let Some(stream) = &plan.stream {
        render_streamed_ops(plan, stream, writer, arena, TimingMode::Timed)
    } else {
        render_timed_calls(plan, writer, arena)
    }
}

/// Render all ops (profiled mode).
fn render_all_ops_profiled(
    plan: &CodegenPlan,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
) -> TokenStream {
    if let Some(stream) = &plan.stream {
        render_streamed_ops(plan, stream, writer, arena, TimingMode::Profiled)
    } else {
        render_profiled_calls(plan, writer, arena)
    }
}

#[derive(Clone, Copy)]
enum TimingMode {
    Plain,
    Timed,
    Profiled,
}

/// Render ops with frame streaming: pre-ops, frame loop, post-ops.
fn render_streamed_ops(
    plan: &CodegenPlan,
    stream: &StreamPlan,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
    mode: TimingMode,
) -> TokenStream {
    let mut sub_op_idx: usize = 0;

    // Pre-ops (before frame loop)
    let mut pre_tokens = Vec::new();
    for op_idx in 0..stream.frame_start {
        let op = &plan.ops[op_idx];
        render_op_with_timing(op, op_idx, writer, arena, mode, &mut sub_op_idx, &mut pre_tokens);
    }

    // Frame input slicing: bind view tensor to a slice of the boundary tensor
    let frame_count = stream.frame_count;
    let mut input_slices = Vec::new();
    for fi in &stream.frame_inputs {
        let view_ident = writer.ident(fi.view_id);
        let boundary_ident = writer.ident(fi.id);
        let stride = fi.frame_stride;
        input_slices.push(quote! {
            let #view_ident = &#boundary_ident[_frame_idx * #stride..(_frame_idx + 1) * #stride];
        });
    }

    // Frame output slicing: bind view tensor to a mutable slice of the boundary tensor
    let mut output_slices = Vec::new();
    for fo in &stream.frame_outputs {
        let view_ident = writer.ident(fo.view_id);
        let boundary_ident = writer.ident(fo.id);
        let stride = fo.frame_stride;
        output_slices.push(quote! {
            let #view_ident = &mut #boundary_ident[_frame_idx * #stride..(_frame_idx + 1) * #stride];
        });
    }

    // Frame-section arena tensor rebindings (inside loop body).
    // Exclude boundary tensors and view tensors (views are bound by slicing above).
    let skip_ids: HashSet<usize> = stream.frame_inputs.iter()
        .flat_map(|fi| [fi.id, fi.view_id])
        .chain(stream.frame_outputs.iter().flat_map(|fo| [fo.id, fo.view_id]))
        .collect();
    let frame_ids: HashSet<usize> = stream.rewritten_tensor_ids.iter()
        .copied()
        .filter(|id| !skip_ids.contains(id))
        .collect();
    let mut frame_arena_bindings = Vec::new();
    if let Some(layout) = arena {
        for alloc in &plan.allocs {
            if let TensorAlloc::Intermediate { id, size } = alloc {
                if frame_ids.contains(id) {
                    let ident = writer.ident(*id);
                    let offset = layout.offsets[&ArenaSlot::Tensor(*id)];
                    frame_arena_bindings.push(quote! {
                        let #ident = unsafe {
                            core::slice::from_raw_parts_mut(
                                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(#offset),
                                #size,
                            )
                        };
                    });
                }
            }
        }
    }

    // Frame-section ops (inside loop)
    let mut frame_tokens = Vec::new();
    for op_idx in stream.frame_start..=stream.frame_end {
        let op = &plan.ops[op_idx];
        render_op_with_timing(op, op_idx, writer, arena, mode, &mut sub_op_idx, &mut frame_tokens);
    }

    // Post-ops (after frame loop)
    let mut post_tokens = Vec::new();
    for op_idx in (stream.frame_end + 1)..plan.ops.len() {
        let op = &plan.ops[op_idx];
        render_op_with_timing(op, op_idx, writer, arena, mode, &mut sub_op_idx, &mut post_tokens);
    }

    quote! {
        #(#pre_tokens)*

        for _frame_idx in 0..#frame_count {
            #(#frame_arena_bindings)*
            #(#input_slices)*
            #(#output_slices)*
            #(#frame_tokens)*
        }

        #(#post_tokens)*
    }
}

/// Render a single op with the appropriate timing wrapper.
fn render_op_with_timing(
    op: &OpPlan,
    op_idx: usize,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
    mode: TimingMode,
    sub_op_idx: &mut usize,
    out: &mut Vec<TokenStream>,
) {
    let scratch = render_scratch(op, op_idx, writer, arena);
    out.push(scratch);

    for sub in &op.sub_ops {
        let calls: Vec<TokenStream> = sub
            .kernels
            .iter()
            .map(|k| render_kernel_call(k, &op.scratch, op_idx, writer))
            .collect();
        let i = *sub_op_idx;
        *sub_op_idx += 1;

        match mode {
            TimingMode::Plain => {
                out.push(quote!(#(#calls)*));
            }
            TimingMode::Timed => {
                out.push(quote! {
                    let __t0 = get_tick();
                    #(#calls)*
                    op_ticks[#i] += get_tick() - __t0;
                });
            }
            TimingMode::Profiled => {
                out.push(quote! {
                    #[cfg(target_os = "psp")]
                    unsafe {
                        psp_rt::profiler::ProfileClear();
                        psp_rt::profiler::ProfileEnable();
                    }
                    let __t0 = get_tick();
                    #(#calls)*
                    op_ticks[#i] += get_tick() - __t0;
                    #[cfg(target_os = "psp")]
                    unsafe {
                        psp_rt::profiler::ProfileDisable();
                        let mut __regs = core::mem::MaybeUninit::<psp_rt::profiler::ProfileRegs>::zeroed();
                        psp_rt::profiler::ProfileGetRegs(__regs.as_mut_ptr());
                        op_profile[#i].accumulate(__regs.assume_init_ref());
                    }
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Weight statics
// ---------------------------------------------------------------------------

/// Models above this threshold use runtime weight loading (pointer path)
/// instead of `include_bytes!` (embedded path). The PSP module loader can
/// handle PRXs up to ~28 MB; 16 MB leaves comfortable headroom for code +
/// intermediate buffers.
const EXTERNAL_WEIGHT_THRESHOLD: usize = 16 * 1024 * 1024;

fn render_weight_statics(plan: &CodegenPlan) -> TokenStream {
    let total_bytes = plan.blob_bytes;
    let total_floats = plan.blob_floats;

    let has_streamed = plan.allocs.iter().any(
        |a| matches!(a, TensorAlloc::Constant { streamed: true, .. }),
    );

    let mut const_entries = Vec::new();
    for alloc in &plan.allocs {
        if let TensorAlloc::Constant {
            id,
            float_offset,
            float_len,
            ..
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

    // Chunked reader for weights that stay on disk (streamed constants).
    // Opens the weight file once per call, seeks to `byte_offset`, and feeds
    // `consume(bytes_done, chunk_floats)` per chunk read into `scratch`.
    let stream_helper = if has_streamed {
        quote! {
            #[cfg(target_os = "psp")]
            fn stream_weight_rows(
                byte_offset: usize,
                total_bytes: usize,
                chunk_bytes: usize,
                scratch: &mut [f32],
                consume: &mut dyn FnMut(usize, &[f32]),
            ) {
                use psp::sys::{sceIoClose, sceIoLseek, sceIoOpen, sceIoRead, IoOpenFlags, IoWhence};
                let fd = unsafe {
                    sceIoOpen(b"host0:/weights.bin\0".as_ptr(), IoOpenFlags::RD_ONLY, 0)
                };
                if fd.0 < 0 {
                    psp_rt::dprintln!("FATAL: could not open host0:/weights.bin for streaming");
                    panic!("weight stream open failed");
                }
                unsafe { sceIoLseek(fd, byte_offset as i64, IoWhence::Set) };
                let mut done = 0usize;
                while done < total_bytes {
                    let want = core::cmp::min(chunk_bytes, total_bytes - done);
                    let dst = scratch.as_mut_ptr() as *mut u8;
                    let mut got = 0usize;
                    while got < want {
                        let n = unsafe {
                            sceIoRead(
                                fd,
                                dst.add(got) as *mut core::ffi::c_void,
                                (want - got) as u32,
                            )
                        };
                        if n <= 0 {
                            psp_rt::dprintln!("FATAL: streamed weight read failed");
                            panic!("weight stream read failed");
                        }
                        got += n as usize;
                    }
                    consume(done, &scratch[..want / 4]);
                    done += want;
                }
                unsafe { sceIoClose(fd) };
            }

            #[cfg(not(target_os = "psp"))]
            fn stream_weight_rows(
                byte_offset: usize,
                total_bytes: usize,
                chunk_bytes: usize,
                scratch: &mut [f32],
                consume: &mut dyn FnMut(usize, &[f32]),
            ) {
                use std::io::{Read, Seek, SeekFrom};
                let mut f = std::fs::File::open(concat!(env!("OUT_DIR"), "/weights.bin"))
                    .expect("failed to open weights.bin for streaming");
                f.seek(SeekFrom::Start(byte_offset as u64)).unwrap();
                let mut done = 0usize;
                while done < total_bytes {
                    let want = core::cmp::min(chunk_bytes, total_bytes - done);
                    let dst = unsafe {
                        core::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut u8, want)
                    };
                    f.read_exact(dst).expect("streamed weight read failed");
                    consume(done, &scratch[..want / 4]);
                    done += want;
                }
            }
        }
    } else {
        quote!()
    };

    if plan.blob_bytes <= EXTERNAL_WEIGHT_THRESHOLD && !has_streamed {
        // Embedded path: weights compiled into the binary via include_bytes!
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

            /// No-op for embedded weights. Provided so callers can
            /// unconditionally call `init()` regardless of model size.
            pub fn init() {}
        }
    } else {
        // Pointer path: weights loaded from file at runtime
        quote! {
            #[allow(dead_code)]
            #[repr(align(16))]
            struct AlignedBytes<const N: usize>([u8; N]);

            /// 16-byte aligned f32 array for VFPU `lv.q`/`sv.q`.
            #[repr(C, align(16))]
            struct Aligned16<const N: usize>([f32; N]);

            const WEIGHT_BYTES: usize = #total_bytes;
            const TENSOR_DATA_FLOATS: usize = #total_floats;

            static mut WEIGHT_PTR: *const u8 = core::ptr::null();

            #(#const_entries)*

            #stream_helper

            fn tensor_data_f32() -> &'static [f32] {
                unsafe {
                    core::slice::from_raw_parts(
                        WEIGHT_PTR as *const f32,
                        TENSOR_DATA_FLOATS,
                    )
                }
            }

            /// Initialize the model by loading weights from file.
            /// Must be called once before `forward()` or `forward_timed()`.
            /// The weight block is registered with `psp_rt::mem` and freed
            /// automatically when the module exits (clean or panicking).
            #[cfg(target_os = "psp")]
            pub fn init() {
                use psp::sys::{sceIoClose, sceIoOpen, sceIoRead, IoOpenFlags};

                let mut alloc_err = 0u32;
                let ptr = psp_rt::mem::alloc_partition(
                    b"weights\0",
                    WEIGHT_BYTES,
                    Some(&mut alloc_err),
                );
                if ptr.is_null() {
                    psp_rt::dprintln!("FATAL: weight alloc failed (0x{:08X})", alloc_err);
                    panic!("weight alloc failed"); // surfaces as the panic exit sentinel
                }

                let fd = unsafe {
                    sceIoOpen(
                        b"host0:/weights.bin\0".as_ptr(),
                        IoOpenFlags::RD_ONLY,
                        0,
                    )
                };
                if fd.0 < 0 {
                    psp_rt::dprintln!("FATAL: could not open host0:/weights.bin");
                    panic!("weights.bin open failed"); // surfaces as the panic exit sentinel
                }
                let mut loaded = 0usize;
                while loaded < WEIGHT_BYTES {
                    let chunk = if WEIGHT_BYTES - loaded < 65536 {
                        WEIGHT_BYTES - loaded
                    } else {
                        65536
                    };
                    let n = unsafe {
                        sceIoRead(
                            fd,
                            ptr.add(loaded) as *mut core::ffi::c_void,
                            chunk as u32,
                        )
                    };
                    if n <= 0 {
                        break;
                    }
                    loaded += n as usize;
                }
                unsafe { sceIoClose(fd) };
                if loaded != WEIGHT_BYTES {
                    psp_rt::dprintln!("FATAL: incomplete weight load: {} / {} bytes", loaded, WEIGHT_BYTES);
                    panic!("incomplete weight load"); // surfaces as the panic exit sentinel
                }
                unsafe { WEIGHT_PTR = ptr };
                psp_rt::dprintln!("Loaded weights: {} bytes", WEIGHT_BYTES);
            }

            /// Initialize the model by loading the resident weight prefix.
            /// `weights.bin` sits next to the generated source in OUT_DIR;
            /// any streamed constants live past WEIGHT_BYTES and are read at
            /// op execution time.
            #[cfg(not(target_os = "psp"))]
            pub fn init() {
                let mut data = std::fs::read(
                    concat!(env!("OUT_DIR"), "/weights.bin"),
                )
                .expect("failed to read weights.bin");
                assert!(data.len() >= WEIGHT_BYTES, "weights.bin smaller than resident size");
                data.truncate(WEIGHT_BYTES);
                let ptr = data.as_ptr();
                std::mem::forget(data);
                unsafe { WEIGHT_PTR = ptr };
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
        if let TensorAlloc::Constant { id, dtype, streamed, .. } = alloc {
            if *streamed {
                // Not memory-resident: ops read it chunkwise from the weight
                // file via stream_weight_rows; no view binding exists.
                continue;
            }
            let var_ident = writer.ident(*id);
            let offset_ident = writer.offset_const(*id);
            let len_ident = writer.len_const(*id);
            if *dtype == crate::ir::graph::DType::I8 {
                // Int8 constants live in the same blob; lengths are stored in
                // 4-byte units, so the byte length is LEN * 4.
                view_entries.push(quote! {
                    let #var_ident: &[i8] = unsafe {
                        core::slice::from_raw_parts(
                            tensor_data[#offset_ident..].as_ptr() as *const i8,
                            #len_ident * 4,
                        )
                    };
                });
            } else {
                view_entries.push(quote! {
                    let #var_ident = &tensor_data[#offset_ident..#offset_ident + #len_ident];
                });
            }
        }
    }

    quote! {
        let tensor_data = tensor_data_f32();
        #(#view_entries)*
    }
}

// ---------------------------------------------------------------------------
// Arena static (module-level, shared between forward and forward_timed)
// ---------------------------------------------------------------------------

fn render_arena_static(arena: Option<&ArenaLayout>) -> TokenStream {
    match arena {
        Some(layout) => {
            let size = layout.arena_size_floats;
            quote! {
                static mut ARENA: Aligned16<#size> = Aligned16([0.0f32; #size]);
            }
        }
        None => quote! {},
    }
}

// ---------------------------------------------------------------------------
// Tensor allocations (intermediates + output)
// ---------------------------------------------------------------------------

fn render_tensor_allocs(
    plan: &CodegenPlan,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
    stream: Option<&StreamPlan>,
) -> TokenStream {
    // Frame boundary tensors get static mut allocation; view tensors get no allocation
    // (they're bound as slices of boundary tensors inside the frame loop)
    let frame_boundary_ids: HashSet<usize> = stream
        .map(|s| {
            s.frame_outputs.iter().map(|fo| fo.id)
                .chain(s.frame_inputs.iter().map(|fi| fi.id))
                .collect()
        })
        .unwrap_or_default();
    let view_ids: HashSet<usize> = stream
        .map(|s| {
            s.frame_inputs.iter().map(|fi| fi.view_id)
                .chain(s.frame_outputs.iter().map(|fo| fo.view_id))
                .collect()
        })
        .unwrap_or_default();

    // Frame-section intermediates (rewritten by stream::rewrite): skip here,
    // they'll be rebound inside the loop. Exclude boundary tensors and views.
    let frame_section_ids: HashSet<usize> = stream
        .map(|s| {
            s.rewritten_tensor_ids.iter()
                .copied()
                .filter(|id| !frame_boundary_ids.contains(id) && !view_ids.contains(id))
                .collect()
        })
        .unwrap_or_default();

    let mut entries = Vec::new();
    for alloc in &plan.allocs {
        match alloc {
            TensorAlloc::Intermediate { id, size } => {
                if frame_section_ids.contains(id) || view_ids.contains(id) {
                    continue; // bound inside frame loop (arena) or as slice (view)
                }
                let ident = writer.ident(*id);
                if frame_boundary_ids.contains(id) {
                    // Frame boundary tensor: full-batch static allocation (not in arena).
                    // Use total_size from FrameBoundaryTensor (pre-rewrite full-batch size),
                    // not the alloc's size (which is now batch=1 after stream rewrite).
                    let full_size = stream
                        .and_then(|s| {
                            s.frame_inputs.iter()
                                .chain(s.frame_outputs.iter())
                                .find(|bt| bt.id == *id)
                                .map(|bt| bt.total_size)
                        })
                        .unwrap_or(*size);
                    let buf_static = writer.buf_static(*id);
                    entries.push(quote! {
                        static mut #buf_static: Aligned16<#full_size> = Aligned16([0.0f32; #full_size]);
                        let #ident = unsafe {
                            core::slice::from_raw_parts_mut(
                                core::ptr::addr_of_mut!(#buf_static) as *mut f32,
                                #full_size,
                            )
                        };
                    });
                } else if let Some(layout) = arena {
                    let offset = layout.offsets[&ArenaSlot::Tensor(*id)];
                    entries.push(quote! {
                        let #ident = unsafe {
                            core::slice::from_raw_parts_mut(
                                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(#offset),
                                #size,
                            )
                        };
                    });
                } else {
                    let buf_static = writer.buf_static(*id);
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
            }
            TensorAlloc::Output { id, size } => {
                let ident = writer.ident(*id);
                let buf_static = writer.buf_static(*id);
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
            TensorAlloc::Constant { .. } => {}
        }
    }
    quote!(#(#entries)*)
}

// ---------------------------------------------------------------------------
// Scratch buffer rendering
// ---------------------------------------------------------------------------

fn render_scratch(
    op: &OpPlan,
    op_idx: usize,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
) -> TokenStream {
    let mut entries = Vec::new();

    for (s_idx, scratch) in op.scratch.iter().enumerate() {
        let size = scratch.size;
        let local_name = format!("scratch_{op_idx}_{s_idx}");
        let local_ident = Ident::new(&local_name, Span::call_site());

        if let Some(layout) = arena {
            let slot = ArenaSlot::Scratch { op_idx, scratch_idx: s_idx };
            let offset = layout.offsets[&slot];
            entries.push(quote! {
                let #local_ident = unsafe {
                    core::slice::from_raw_parts_mut(
                        (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(#offset),
                        #size,
                    )
                };
            });
        } else {
            let static_name = format!("SCRATCH_{op_idx}_{s_idx}");
            let static_ident = Ident::new(&static_name, Span::call_site());
            entries.push(quote! {
                static mut #static_ident: Aligned16<#size> = Aligned16([0.0f32; #size]);
                let #local_ident = unsafe {
                    core::slice::from_raw_parts_mut(
                        core::ptr::addr_of_mut!(#static_ident) as *mut f32,
                        #size,
                    )
                };
            });
        }

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
                CopyStrategy::RowPaddedDequantI8 {
                    num_rows,
                    src_stride,
                    dst_stride,
                    scales,
                } => {
                    let scales_expr = writer.read(*scales);
                    entries.push(quote! {
                        for row in 0..#num_rows {
                            let s = #scales_expr[row];
                            let src = &#src_expr[row * #src_stride..(row + 1) * #src_stride];
                            let dst = &mut #local_ident[row * #dst_stride..row * #dst_stride + #src_stride];
                            for (d, &q) in dst.iter_mut().zip(src.iter()) {
                                *d = q as f32 * s;
                            }
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
            weight_scales,
        } => {
            let input_expr = writer.read(input.id);
            let input_shape_tok = shape_tokens(&input.shape);
            let filter_expr = writer.read(filter.id);
            let filter_shape_tok = shape_tokens(&filter.shape);
            let output_expr = writer.write(output.id);
            let output_shape_tok = shape_tokens(&output.shape);
            let [sh, sw] = stride;
            let [pt, pb, pl, pr] = padding;

            let bias_tok = match bias {
                Some(b) => {
                    let b_expr = writer.read(*b);
                    quote!(Some(#b_expr))
                }
                None => quote!(None),
            };

            let fn_name = match (weight_scales.is_some(), *has_relu) {
                (true, true) => "conv2d_relu_q8",
                (true, false) => "conv2d_q8",
                (false, true) => "conv2d_relu",
                (false, false) => "conv2d",
            };
            let fn_ident = Ident::new(fn_name, Span::call_site());
            let scales_tok = match weight_scales {
                Some(s) => {
                    let s_expr = writer.read(*s);
                    quote!(#s_expr,)
                }
                None => quote!(),
            };

            quote! {
                #fn_ident(
                    #input_expr, #input_shape_tok,
                    #filter_expr, #filter_shape_tok,
                    #scales_tok
                    #bias_tok,
                    [#sh, #sw],
                    [#pt, #pb, #pl, #pr],
                    #output_expr, #output_shape_tok
                );
            }
        }

        KernelCall::Im2colPadded {
            input,
            kernel_size,
            stride,
            padding,
            output_hw,
            output: _scratch_idx,
        } => {
            let input_expr = writer.read(input.id);
            let input_shape_tok = shape_tokens(&input.shape);
            let [kh, kw] = kernel_size;
            let [sh, sw] = stride;
            let [pt, pb, pl, pr] = padding;
            let [ho, wo] = output_hw;
            let scratch_ident = Ident::new(&format!("scratch_{op_idx}_{}", _scratch_idx), Span::call_site());

            quote! {
                im2col_padded(
                    #input_expr, #input_shape_tok,
                    [#kh, #kw], [#sh, #sw], [#pt, #pb, #pl, #pr], [#ho, #wo],
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
            let a_ident = Ident::new(&format!("scratch_{op_idx}_{}", _a_scratch), Span::call_site());
            let b_ident = Ident::new(&format!("scratch_{op_idx}_{}", _b_scratch), Span::call_site());
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

        KernelCall::Pool2d {
            input,
            output,
            filter,
            stride,
            padding,
            pool_type,
        } => {
            let fn_name = match pool_type {
                PoolType::Max => "max_pool2d",
                PoolType::Average => "average_pool2d",
            };
            let fn_ident = Ident::new(fn_name, Span::call_site());
            let input_expr = writer.read(input.id);
            let input_shape_tok = shape_tokens(&input.shape);
            let output_expr = writer.write(output.id);
            let output_shape_tok = shape_tokens(&output.shape);
            let [fh, fw] = filter;
            let [sh, sw] = stride;
            let [pt, pb, pl, pr] = padding;
            quote! {
                #fn_ident(
                    #input_expr, #input_shape_tok,
                    [#fh, #fw], [#sh, #sw],
                    [#pt, #pb, #pl, #pr],
                    #output_expr, #output_shape_tok
                );
            }
        }

        KernelCall::Reshape { input, output } => {
            let input_expr = writer.read(*input);
            let output_expr = writer.write(*output);
            quote! { reshape(#input_expr, #output_expr); }
        }

        KernelCall::FakeQuant {
            input,
            output,
            scale,
            zero_point,
        } => {
            let input_expr = writer.read(*input);
            let output_expr = writer.write(*output);
            quote! { fake_quant(#input_expr, #output_expr, #scale, #zero_point); }
        }

        KernelCall::FullyConnectedStreamed {
            input,
            in_features,
            weights,
            bias,
            output,
            out_features,
            has_relu,
            scratch,
            chunk_rows,
        } => {
            let input_expr = writer.read(*input);
            let output_expr = writer.write(*output);
            let w_offset = writer.offset_const(*weights);
            let scratch_ident =
                Ident::new(&format!("scratch_{op_idx}_{scratch}"), Span::call_site());
            let fc_fn = if *has_relu {
                quote!(fully_connected_relu)
            } else {
                quote!(fully_connected)
            };
            let bias_tok = match bias {
                Some(b) => {
                    let b_expr = writer.read(*b);
                    quote!(Some(&#b_expr[_row0.._row0 + _rows]))
                }
                None => quote!(None),
            };
            quote! {
                stream_weight_rows(
                    #w_offset * 4,
                    #out_features * #in_features * 4,
                    #chunk_rows * #in_features * 4,
                    #scratch_ident,
                    &mut |_byte_off, _chunk| {
                        let _row0 = _byte_off / (#in_features * 4);
                        let _rows = _chunk.len() / #in_features;
                        #fc_fn(
                            #input_expr, #in_features,
                            _chunk, #bias_tok,
                            &mut #output_expr[_row0.._row0 + _rows], _rows
                        );
                    },
                );
            }
        }

        KernelCall::RfftBatch {
            input,
            stage_twiddles,
            unpack_twiddles,
            output,
            scratch,
            n,
            frames,
        } => {
            let input_expr = writer.read(*input);
            let stw_expr = writer.read(*stage_twiddles);
            let utw_expr = writer.read(*unpack_twiddles);
            let output_expr = writer.write(*output);
            let scratch_ident =
                Ident::new(&format!("scratch_{op_idx}_{scratch}"), Span::call_site());
            quote! {
                rfft_batch(#input_expr, #stw_expr, #utw_expr, #scratch_ident, #output_expr, #n, #frames);
            }
        }

        KernelCall::FullyConnected {
            input,
            in_features,
            weights,
            bias,
            output,
            out_features,
            has_relu,
            batch_size,
        } => {
            let input_expr = writer.read(*input);
            let weight_expr = writer.read(*weights);
            let output_expr = writer.write(*output);
            let bias_tok = match bias {
                Some(b) => {
                    let b_expr = writer.read(*b);
                    quote!(Some(#b_expr))
                }
                None => quote!(None),
            };
            let fc_fn = if *has_relu {
                quote!(fully_connected_relu)
            } else {
                quote!(fully_connected)
            };
            if *batch_size > 1 {
                let batch = *batch_size;
                quote! {
                    for _batch_idx in 0..#batch {
                        let _in_off = _batch_idx * #in_features;
                        let _out_off = _batch_idx * #out_features;
                        #fc_fn(
                            &#input_expr[_in_off.._in_off + #in_features], #in_features,
                            #weight_expr, #bias_tok,
                            &mut #output_expr[_out_off.._out_off + #out_features], #out_features
                        );
                    }
                }
            } else {
                quote! {
                    #fc_fn(
                        #input_expr, #in_features,
                        #weight_expr, #bias_tok,
                        #output_expr, #out_features
                    );
                }
            }
        }

        KernelCall::ElementWise {
            op,
            input_a,
            input_b,
            output,
            b_len,
        } => {
            let fn_ident = Ident::new(op.name(), Span::call_site());
            let a_expr = writer.read(*input_a);
            let b_expr = writer.read(*input_b);
            let out_expr = writer.write(*output);
            quote! { #fn_ident(#a_expr, #b_expr, #out_expr, #b_len); }
        }

        KernelCall::UnaryElementWise { op, input, output } => {
            let fn_ident = Ident::new(&format!("unary_{}", op.name()), Span::call_site());
            let in_expr = writer.read(*input);
            let out_expr = writer.write(*output);
            quote! { #fn_ident(#in_expr, #out_expr); }
        }

        KernelCall::Reduce { op, input, output, batch_size, frame_in_size, frame_out_size } => {
            let fn_ident = Ident::new(op.name(), Span::call_site());
            let in_expr = writer.read(*input);
            let out_expr = writer.write(*output);
            if *batch_size > 1 {
                let batch = *batch_size;
                let fin = *frame_in_size;
                let fout = *frame_out_size;
                quote! {
                    for _batch_idx in 0..#batch {
                        let _in_off = _batch_idx * #fin;
                        let _out_off = _batch_idx * #fout;
                        #fn_ident(
                            &#in_expr[_in_off.._in_off + #fin],
                            &mut #out_expr[_out_off.._out_off + #fout]
                        );
                    }
                }
            } else {
                quote! { #fn_ident(#in_expr, #out_expr); }
            }
        }

        KernelCall::Pad {
            input,
            output,
            padding,
        } => {
            let input_expr = writer.read(input.id);
            let input_shape_tok = shape_tokens(&input.shape);
            let output_expr = writer.write(output.id);
            let output_shape_tok = shape_tokens(&output.shape);
            let p00 = padding[0][0];
            let p01 = padding[0][1];
            let p10 = padding[1][0];
            let p11 = padding[1][1];
            let p20 = padding[2][0];
            let p21 = padding[2][1];
            let p30 = padding[3][0];
            let p31 = padding[3][1];
            quote! {
                pad(
                    #input_expr, #input_shape_tok,
                    #output_expr, #output_shape_tok,
                    [[#p00, #p01], [#p10, #p11], [#p20, #p21], [#p30, #p31]]
                );
            }
        }

        KernelCall::Transpose {
            input,
            output,
            input_shape,
            output_shape,
            perm,
        } => {
            let input_expr = writer.read(*input);
            let output_expr = writer.write(*output);
            let is_tok = shape_tokens(input_shape);
            let os_tok = shape_tokens(output_shape);
            let p_tok = shape_tokens(perm);
            quote! {
                transpose(
                    #input_expr, &#is_tok,
                    #output_expr, &#os_tok,
                    &#p_tok
                );
            }
        }

        KernelCall::ReverseV2 {
            input,
            output,
            input_shape,
            axis,
        } => {
            let input_expr = writer.read(*input);
            let output_expr = writer.write(*output);
            let is_tok = shape_tokens(input_shape);
            quote! {
                reverse_v2(#input_expr, &#is_tok, #output_expr, #axis);
            }
        }

        KernelCall::DepthwiseConv2d {
            input,
            filter,
            bias,
            stride,
            padding,
            output,
        } => {
            let input_expr = writer.read(input.id);
            let input_shape_tok = shape_tokens(&input.shape);
            let filter_expr = writer.read(filter.id);
            let filter_shape_tok = shape_tokens(&filter.shape);
            let output_expr = writer.write(output.id);
            let output_shape_tok = shape_tokens(&output.shape);
            let [sh, sw] = stride;
            let [pt, pb, pl, pr] = padding;

            let bias_tok = match bias {
                Some(b) => {
                    let b_expr = writer.read(*b);
                    quote!(Some(#b_expr))
                }
                None => quote!(None),
            };

            quote! {
                depthwise_conv2d(
                    #input_expr, #input_shape_tok,
                    #filter_expr, #filter_shape_tok,
                    #bias_tok,
                    [#sh, #sw],
                    [#pt, #pb, #pl, #pr],
                    #output_expr, #output_shape_tok
                );
            }
        }

        KernelCall::RfftPack { input, output: scratch_idx, n } => {
            let input_expr = writer.read(*input);
            let scratch_ident = Ident::new(&format!("scratch_{op_idx}_{scratch_idx}"), Span::call_site());
            let n = *n;
            quote! { rfft_pack(#input_expr, #scratch_ident, #n); }
        }

        KernelCall::FftButterflyStage { data: scratch_idx, twiddles, n_complex, half_size } => {
            let scratch_ident = Ident::new(&format!("scratch_{op_idx}_{scratch_idx}"), Span::call_site());
            let tw_expr = writer.read(*twiddles);
            let nc = *n_complex;
            let hs = *half_size;
            quote! { fft_butterfly_stage(#scratch_ident, #tw_expr, #nc, #hs); }
        }

        KernelCall::RfftUnpack { data: scratch_idx, twiddles, output, n } => {
            let scratch_ident = Ident::new(&format!("scratch_{op_idx}_{scratch_idx}"), Span::call_site());
            let tw_expr = writer.read(*twiddles);
            let out_expr = writer.write(*output);
            let n = *n;
            quote! { rfft_unpack(#scratch_ident, #tw_expr, #out_expr, #n); }
        }

        KernelCall::StridedSlice {
            input, output,
            input_shape, output_shape: _,
            begin, end, strides,
            begin_mask, end_mask, shrink_axis_mask,
        } => {
            let input_expr = writer.read(*input);
            let output_expr = writer.write(*output);
            let is_tok = shape_tokens(input_shape);
            let begin_vals: Vec<_> = begin.iter().map(|v| quote!(#v)).collect();
            let end_vals: Vec<_> = end.iter().map(|v| quote!(#v)).collect();
            let stride_vals: Vec<_> = strides.iter().map(|v| quote!(#v)).collect();
            quote! {
                strided_slice(
                    #input_expr, &#is_tok,
                    #output_expr,
                    &[#(#begin_vals),*], &[#(#end_vals),*], &[#(#stride_vals),*],
                    #begin_mask, #end_mask, #shrink_axis_mask
                );
            }
        }

        KernelCall::Gather {
            input, output, indices, indices_len,
            input_shape, output_shape,
            axis,
        } => {
            let input_expr = writer.read(*input);
            let output_expr = writer.write(*output);
            let is_tok = shape_tokens(input_shape);
            let os_tok = shape_tokens(output_shape);
            // indices is an I32 constant tensor in the weights blob.
            // Read as f32 slice and reinterpret as i32 at runtime.
            let indices_expr = writer.read(*indices);
            let indices_len = *indices_len;
            quote! {
                gather(
                    #input_expr, &#is_tok,
                    unsafe { core::slice::from_raw_parts(#indices_expr.as_ptr() as *const i32, #indices_len) },
                    #output_expr, &#os_tok,
                    #axis
                );
            }
        }

        KernelCall::Concatenation {
            inputs, output,
            input_shapes, output_shape,
            axis,
        } => {
            let output_expr = writer.write(*output);
            let os_tok = shape_tokens(output_shape);
            let axis = *axis;

            // Generate sequential copy: for each input, compute offset along axis and copy
            let mut copies = Vec::new();
            let mut axis_offset: usize = 0;
            for (idx, (tensor_id, in_shape)) in inputs.iter().zip(input_shapes.iter()).enumerate() {
                let input_expr = writer.read(*tensor_id);
                let _ = idx;

                // Compute suffix size (product of dims after axis)
                let suffix: usize = in_shape[axis + 1..].iter().product();
                let prefix: usize = in_shape[..axis].iter().product();
                let axis_len = in_shape[axis];
                let out_axis_len = output_shape[axis];

                copies.push(quote! {
                    {
                        let src = #input_expr;
                        for p in 0..#prefix {
                            for a in 0..#axis_len {
                                let src_off = p * (#axis_len * #suffix) + a * #suffix;
                                let dst_off = p * (#out_axis_len * #suffix) + (#axis_offset + a) * #suffix;
                                #output_expr[dst_off..dst_off + #suffix].copy_from_slice(&src[src_off..src_off + #suffix]);
                            }
                        }
                    }
                });
                axis_offset += axis_len;
            }

            let _ = os_tok;
            quote! { #(#copies)* }
        }
    }
}

// ---------------------------------------------------------------------------
// Plain (untimed) op rendering
// ---------------------------------------------------------------------------

fn render_op_plain(
    op: &OpPlan,
    op_idx: usize,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
) -> TokenStream {
    let scratch = render_scratch(op, op_idx, writer, arena);
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

fn render_timed_calls(
    plan: &CodegenPlan,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
) -> TokenStream {
    let mut sub_op_idx: usize = 0;
    let mut entries = Vec::new();

    for (op_idx, op) in plan.ops.iter().enumerate() {
        // Scratch setup is untimed
        let scratch = render_scratch(op, op_idx, writer, arena);
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
// Profiled op rendering (timed + hardware counters)
// ---------------------------------------------------------------------------

fn render_profiled_calls(
    plan: &CodegenPlan,
    writer: &TensorExprWriter,
    arena: Option<&ArenaLayout>,
) -> TokenStream {
    let mut sub_op_idx: usize = 0;
    let mut entries = Vec::new();

    for (op_idx, op) in plan.ops.iter().enumerate() {
        let scratch = render_scratch(op, op_idx, writer, arena);
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
                #[cfg(target_os = "psp")]
                unsafe {
                    psp_rt::profiler::ProfileClear();
                    psp_rt::profiler::ProfileEnable();
                }
                let __t0 = get_tick();
                #(#calls)*
                op_ticks[#i] += get_tick() - __t0;
                #[cfg(target_os = "psp")]
                unsafe {
                    psp_rt::profiler::ProfileDisable();
                    let mut __regs = core::mem::MaybeUninit::<psp_rt::profiler::ProfileRegs>::zeroed();
                    psp_rt::profiler::ProfileGetRegs(__regs.as_mut_ptr());
                    op_profile[#i].accumulate(__regs.assume_init_ref());
                }
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
