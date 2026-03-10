pub mod arena;
mod lower;
pub mod plan;
mod render;
mod tensor_expr;

use crate::ir::stream;
use crate::ir::PspModel;
use proc_macro2::TokenStream;

pub type GenResult<T> = Result<T, String>;

pub struct Generated {
    pub tokens: TokenStream,
    pub data_bytes: Vec<u8>,
    pub data_path: String,
}

pub fn generate_code(
    model: &mut PspModel,
    stream_batch: Option<(usize, usize)>,
) -> GenResult<Generated> {
    // Stream analysis + rewrite (before lowering, so lowerer sees clean batch=1 graph)
    let stream_plan = if let Some((start, end)) = stream_batch {
        let regions = stream::analyze(model);
        for (i, r) in regions.iter().enumerate() {
            eprintln!(
                "Stream region {}: ops [{}..={}] ({} ops, N={})",
                i,
                r.start,
                r.end,
                r.end - r.start + 1,
                r.frame_count,
            );
        }
        let region = stream::select_region(model, &regions, start, end)?;
        eprintln!(
            "Stream: selected ops [{}..={}], N={}, {} frame inputs, {} frame outputs",
            region.start,
            region.end,
            region.frame_count,
            region.frame_inputs.len(),
            region.frame_outputs.len(),
        );
        for fi in &region.frame_inputs {
            eprintln!("  input: t{} ({}×{} floats)", fi.id, region.frame_count, fi.frame_stride);
        }
        for fo in &region.frame_outputs {
            eprintln!("  output: t{} ({}×{} floats)", fo.id, region.frame_count, fo.frame_stride);
        }
        Some(stream::rewrite(model, &region))
    } else {
        None
    };

    // Lowerer sees a clean graph (batch=1 if streaming was applied)
    let mut plan = lower::lower(model)?;
    plan.stream = stream_plan;

    plan.arena = Some(arena::compute_and_pack(&plan));

    // Compact the weight blob: pack only referenced constant byte ranges
    let data_bytes = compact_blob(&mut plan, &model.model_data);

    let tokens = render::render(&plan, &model.graph);

    Ok(Generated {
        tokens,
        data_bytes,
        data_path: "weights.bin".to_string(),
    })
}

/// Build a compact weight blob containing only referenced constant data.
///
/// Iterates over all `TensorAlloc::Constant` entries, copies their byte ranges
/// into a new packed buffer, and remaps `float_offset` values. Updates
/// `plan.blob_bytes` and `plan.blob_floats` to the compact size.
fn compact_blob(plan: &mut plan::CodegenPlan, model_data: &[u8]) -> Vec<u8> {
    let sz = std::mem::size_of::<f32>();
    // 16-byte alignment (4 floats) so VFPU-aligned base stays aligned for all entries
    const ALIGN_FLOATS: usize = 4;

    let mut packed = Vec::new();

    for alloc in plan.allocs.iter_mut() {
        if let plan::TensorAlloc::Constant {
            float_offset,
            float_len,
            ..
        } = alloc
        {
            let byte_offset = *float_offset * sz;
            let byte_len = *float_len * sz;

            // Align packed position to 16 bytes
            let packed_floats = packed.len() / sz;
            let aligned_floats = (packed_floats + ALIGN_FLOATS - 1) & !(ALIGN_FLOATS - 1);
            let pad_bytes = (aligned_floats - packed_floats) * sz;
            packed.extend(std::iter::repeat(0u8).take(pad_bytes));

            let new_float_offset = packed.len() / sz;
            packed.extend_from_slice(&model_data[byte_offset..byte_offset + byte_len]);

            *float_offset = new_float_offset;
        }
    }

    plan.blob_bytes = packed.len();
    plan.blob_floats = packed.len() / sz;

    packed
}
