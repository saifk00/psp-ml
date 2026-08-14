pub mod arena;
mod lower;
pub mod plan;
mod render;
mod tensor_expr;

use std::collections::HashSet;
use std::fmt;

use crate::ir::graph::TensorId;
use crate::ir::stream;
use crate::ir::PspModel;
use crate::ir::residency::{self, fmt_bytes};
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
    /// Resident-blob budget the residency decision was made against.
    pub resident_budget: usize,
}

impl ModelStats {
    fn peak_memory_bytes(&self) -> usize {
        self.blob_bytes
            + self.arena_size_floats * 4
            + self.output_size_floats * 4
            + self.frame_boundary_bytes
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
        match residency::device_fit(self.blob_bytes, self.resident_budget) {
            Ok(headroom) => writeln!(
                f,
                "  Device fit:   {} resident of a {} budget ({} headroom)",
                fmt_bytes(self.blob_bytes),
                fmt_bytes(self.resident_budget),
                fmt_bytes(headroom)
            )?,
            Err(shortfall) => writeln!(
                f,
                "  Device fit:   OVER BUDGET by {} ({} resident, {} budget)",
                fmt_bytes(shortfall),
                fmt_bytes(self.blob_bytes),
                fmt_bytes(self.resident_budget)
            )?,
        }
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

/// Generated statics that land in `.bss` — the arena, the output buffer, and
/// any frame-boundary buffers.
///
/// These matter for residency planning because the module loader claims them
/// from partition-2 memory *before* `init()` ever runs, so they come straight
/// off the budget available for the weight blob.
fn generated_static_bytes(plan: &plan::CodegenPlan) -> usize {
    plan.arena.as_ref().map_or(0, |a| a.arena_size_floats) * 4
        + plan.output_size * 4
        + frame_boundary_bytes(plan)
}

fn frame_boundary_bytes(plan: &plan::CodegenPlan) -> usize {
    plan.stream
        .as_ref()
        .map(|s| {
            s.frame_inputs
                .iter()
                .chain(s.frame_outputs.iter())
                .map(|t| t.total_size * 4)
                .sum()
        })
        .unwrap_or(0)
}

/// One full lowering pass: lower → arena → compact blob.
///
/// Factored out so the speculative and re-planned passes are provably the same
/// pipeline. Order matters: `plan.stream` must be set before
/// `arena::compute_and_pack` reads it.
fn lower_pack_and_compact(
    model: &mut PspModel,
    streamed: &HashSet<TensorId>,
    stream_plan: Option<plan::StreamPlan>,
) -> GenResult<(plan::CodegenPlan, Vec<u8>)> {
    let mut plan = lower::lower(model, streamed)?;
    plan.stream = stream_plan;
    plan.arena = Some(arena::compute_and_pack(&plan));
    let data_bytes = compact_blob(&mut plan, &model.model_data);
    Ok((plan, data_bytes))
}

pub fn generate_code(
    model: &mut PspModel,
    stream_batch: Option<(usize, usize)>,
    residency_choice: Option<usize>,
    resident_budget: Option<usize>,
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

    // Weight residency: enumerate candidates, then select one — same shape as
    // the stream planner above. `ir::residency` owns the policy; this closure
    // is the measurement it cannot make for itself.
    let candidates = residency::analyze(model);
    residency::report_candidates(&candidates);
    let chosen = residency::select(
        model,
        &candidates,
        residency_choice,
        resident_budget,
        |model, streamed| {
            let (plan, data_bytes) = lower_pack_and_compact(model, streamed, stream_plan.clone())?;
            let (blob, statics) = (plan.blob_bytes, generated_static_bytes(&plan));
            Ok(((plan, data_bytes), blob, statics))
        },
    )?;
    residency::report_selection(model, &chosen);
    let budget = chosen.budget;
    let (plan, data_bytes) = chosen.payload;

    let streamed_bytes = data_bytes.len() - plan.blob_bytes;
    let stats = ModelStats {
        num_ops: plan.ops.len(),
        num_sub_ops: plan.ops.iter().map(|op| op.sub_ops.len()).sum(),
        blob_bytes: plan.blob_bytes,
        streamed_bytes,
        arena_size_floats: plan.arena.as_ref().map_or(0, |a| a.arena_size_floats),
        input_size_floats: plan.input_size,
        output_size_floats: plan.output_size,
        frame_boundary_bytes: frame_boundary_bytes(&plan),
        resident_budget: budget,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::{DType, Graph, TensorKind};
    use crate::ir::psp::{FullyConnectedParams, PspOp};

    const FC_OUT: usize = 1 << 20;
    const FC_IN: usize = 13;
    const FFT: usize = 1024;

    /// A model whose weights fit the partition on paper (50 MiB of a 51.2 MiB
    /// optimistic budget) but not once its arena and output buffer are
    /// subtracted — so only the second planning pass gets it right.
    ///
    /// The trailing Rfft is there to make lowering append constants (twiddles),
    /// which is what the rollback has to undo.
    fn overcommitted_model() -> PspModel {
        let mut g = Graph::<PspOp>::new();

        let input = g.add_tensor(vec![1, FC_IN], DType::F32, TensorKind::Input);
        g.inputs.push(input);
        let wbytes = FC_OUT * FC_IN * 4;
        let weights = g.add_tensor(
            vec![FC_OUT, FC_IN],
            DType::F32,
            TensorKind::Constant { offset: 0, len: wbytes },
        );
        let hidden = g.add_tensor(vec![1, FC_OUT], DType::F32, TensorKind::Intermediate);
        g.ops.push(PspOp::FullyConnected {
            input,
            weights,
            bias: None,
            output: hidden,
            fused_activation: FullyConnectedParams { fused_activation: None },
        });

        let frames = FC_OUT / FFT;
        let out = g.add_tensor(
            vec![1, frames * (FFT / 2 + 1)],
            DType::F32,
            TensorKind::Output,
        );
        g.outputs.push(out);
        g.ops.push(PspOp::Rfft {
            input: hidden,
            output: out,
            fft_length: FFT,
        });

        PspModel { graph: g, model_data: vec![0u8; wbytes] }
    }

    #[test]
    fn replans_with_streaming_when_statics_eat_the_budget() {
        // What a single lowering appends, to compare the rollback against.
        let mut probe = overcommitted_model();
        let base_len = probe.model_data.len();
        lower::lower(&mut probe, &HashSet::new()).unwrap();
        let appended_once = probe.model_data.len() - base_len;
        assert!(appended_once > 0, "expected the Rfft to append twiddles");
        drop(probe);

        let mut model = overcommitted_model();
        let generated = generate_code(&mut model, None, None, None).unwrap();

        // Pass 1 (which sees no statics) fits and streams nothing; pass 2 must
        // notice the ~6 MiB of arena + output and evict the FC weights.
        assert!(
            generated.stats.streamed_bytes > 40 * 1024 * 1024,
            "expected the FC weights to be streamed, got {} B",
            generated.stats.streamed_bytes
        );
        assert!(
            residency::device_fit(
                generated.stats.blob_bytes,
                generated.stats.resident_budget
            )
            .is_ok(),
            "re-planned model still does not fit"
        );

        // The rolled-back pass must not have left a second copy of its
        // appended twiddles in the blob.
        assert_eq!(
            model.model_data.len() - base_len,
            appended_once,
            "rollback left duplicated appended constants behind"
        );
    }

    #[test]
    fn analyze_ladders_from_all_resident_to_all_streamed() {
        let model = overcommitted_model();
        let candidates = residency::analyze(&model);
        assert_eq!(candidates.len(), 2, "one FC weight => two rungs");
        assert!(candidates[0].streamed.is_empty());
        assert_eq!(candidates[0].streamed_estimate, 0);
        assert_eq!(candidates[1].streamed.len(), 1);
        // The ladder conserves bytes: resident + streamed is constant.
        let total = candidates[0].resident_estimate;
        for c in &candidates {
            assert_eq!(c.resident_estimate + c.streamed_estimate, total);
        }
    }

    #[test]
    fn a_forced_candidate_is_used_even_when_it_does_not_fit() {
        // Candidate 0 keeps 50 MiB resident, which does not fit once the arena
        // is subtracted — selection would reject it, forcing must not.
        let mut model = overcommitted_model();
        let generated = generate_code(&mut model, None, Some(0), None).unwrap();
        assert_eq!(generated.stats.streamed_bytes, 0);
        assert!(residency::device_fit(
            generated.stats.blob_bytes,
            generated.stats.resident_budget
        )
        .is_err());
    }

    #[test]
    fn an_out_of_range_candidate_is_an_error() {
        let mut model = overcommitted_model();
        let err = match generate_code(&mut model, None, Some(99), None) {
            Err(e) => e,
            Ok(_) => panic!("expected an out-of-range candidate to be rejected"),
        };
        assert!(err.contains("does not exist"), "unhelpful error: {err}");
    }

    #[test]
    fn a_generous_budget_override_keeps_everything_resident() {
        let mut model = overcommitted_model();
        let generated = generate_code(&mut model, None, None, Some(usize::MAX)).unwrap();
        assert_eq!(generated.stats.streamed_bytes, 0);
    }
}
