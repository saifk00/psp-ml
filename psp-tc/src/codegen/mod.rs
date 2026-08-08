pub mod arena;
mod lower;
pub mod plan;
mod render;
mod tensor_expr;

use std::fmt;

use crate::ir::stream;
use crate::ir::PspModel;
use proc_macro2::TokenStream;

pub type GenResult<T> = Result<T, String>;

/// Compilation statistics for the generated model.
pub struct ModelStats {
    pub num_ops: usize,
    pub num_sub_ops: usize,
    /// Memory-resident weight bytes (what `init()` loads).
    pub blob_bytes: usize,
    /// Weight bytes left on disk and streamed at op execution time.
    pub streamed_bytes: usize,
    pub arena_size_floats: usize,
    pub input_size_floats: usize,
    pub output_size_floats: usize,
    pub frame_boundary_bytes: usize,
}

impl ModelStats {
    fn peak_memory_bytes(&self) -> usize {
        self.blob_bytes
            + self.arena_size_floats * 4
            + self.output_size_floats * 4
            + self.frame_boundary_bytes
    }
}

fn fmt_bytes(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.1} KiB", bytes as f64 / 1_024.0)
    } else {
        format!("{} B", bytes)
    }
}

impl fmt::Display for ModelStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arena_bytes = self.arena_size_floats * 4;
        writeln!(f, "Model Statistics:")?;
        writeln!(
            f,
            "  Ops:          {} ({} sub-ops)",
            self.num_ops, self.num_sub_ops
        )?;
        writeln!(f, "  Weights:      {} resident", fmt_bytes(self.blob_bytes))?;
        if self.streamed_bytes > 0 {
            writeln!(f, "  Streamed:     {} (from weight file at runtime)", fmt_bytes(self.streamed_bytes))?;
        }
        writeln!(
            f,
            "  Arena:        {} ({} floats)",
            fmt_bytes(arena_bytes),
            self.arena_size_floats
        )?;
        writeln!(f, "  Peak memory:  {}", fmt_bytes(self.peak_memory_bytes()))?;
        writeln!(
            f,
            "  Input:        {} floats ({})",
            self.input_size_floats,
            fmt_bytes(self.input_size_floats * 4)
        )?;
        write!(
            f,
            "  Output:       {} floats ({})",
            self.output_size_floats,
            fmt_bytes(self.output_size_floats * 4)
        )
    }
}

pub struct Generated {
    pub tokens: TokenStream,
    pub data_bytes: Vec<u8>,
    pub data_path: String,
    pub stats: ModelStats,
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
            eprintln!(
                "  input: t{} ({}×{} floats)",
                fi.id, region.frame_count, fi.frame_stride
            );
        }
        for fo in &region.frame_outputs {
            eprintln!(
                "  output: t{} ({}×{} floats)",
                fo.id, region.frame_count, fo.frame_stride
            );
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

    // Collect statistics before rendering
    let frame_boundary_bytes = plan
        .stream
        .as_ref()
        .map(|s| {
            s.frame_inputs
                .iter()
                .chain(s.frame_outputs.iter())
                .map(|t| t.total_size * 4)
                .sum()
        })
        .unwrap_or(0);

    let streamed_bytes = data_bytes.len() - plan.blob_bytes;
    let stats = ModelStats {
        num_ops: plan.ops.len(),
        num_sub_ops: plan.ops.iter().map(|op| op.sub_ops.len()).sum(),
        blob_bytes: plan.blob_bytes,
        streamed_bytes,
        arena_size_floats: plan.arena.as_ref().map_or(0, |a| a.arena_size_floats),
        input_size_floats: plan.input_size,
        output_size_floats: plan.output_size,
        frame_boundary_bytes,
    };

    let tokens = render::render(&plan, &model.graph);

    Ok(Generated {
        tokens,
        data_bytes,
        data_path: "weights.bin".to_string(),
        stats,
    })
}

/// Build a compact weight blob containing only referenced constant data.
///
/// Resident constants are packed first and their `float_offset` remapped to
/// resident-memory offsets; `init()` loads exactly this prefix
/// (`plan.blob_bytes`). Streamed constants are appended *after* the resident
/// prefix — their `float_offset` becomes a file offset and they are read
/// chunkwise at op execution time, never held in memory.
fn compact_blob(plan: &mut plan::CodegenPlan, model_data: &[u8]) -> Vec<u8> {
    let sz = std::mem::size_of::<f32>();
    // 16-byte alignment (4 floats) so VFPU-aligned base stays aligned for all entries
    const ALIGN_FLOATS: usize = 4;

    let mut packed = Vec::new();

    let mut pack = |packed: &mut Vec<u8>, float_offset: &mut usize, float_len: usize| {
        let byte_offset = *float_offset * sz;
        let byte_len = float_len * sz;

        // Align packed position to 16 bytes
        let packed_floats = packed.len() / sz;
        let aligned_floats = (packed_floats + ALIGN_FLOATS - 1) & !(ALIGN_FLOATS - 1);
        let pad_bytes = (aligned_floats - packed_floats) * sz;
        packed.extend(std::iter::repeat(0u8).take(pad_bytes));

        let new_float_offset = packed.len() / sz;
        packed.extend_from_slice(&model_data[byte_offset..byte_offset + byte_len]);
        *float_offset = new_float_offset;
    };

    for alloc in plan.allocs.iter_mut() {
        if let plan::TensorAlloc::Constant {
            float_offset,
            float_len,
            streamed: false,
            ..
        } = alloc
        {
            pack(&mut packed, float_offset, *float_len);
        }
    }

    // Resident prefix ends here — this is what init() loads.
    plan.blob_bytes = packed.len();
    plan.blob_floats = packed.len() / sz;

    for alloc in plan.allocs.iter_mut() {
        if let plan::TensorAlloc::Constant {
            float_offset,
            float_len,
            streamed: true,
            ..
        } = alloc
        {
            pack(&mut packed, float_offset, *float_len);
        }
    }

    packed
}
