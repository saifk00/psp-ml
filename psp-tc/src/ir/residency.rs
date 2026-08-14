//! Weight-residency planning: which constants stay in RAM, and which are read
//! from the weight file at op execution time.
//!
//! Same `analyze` / `select` shape as [`crate::ir::stream`]: `analyze`
//! enumerates candidates without committing to one, and `select` chooses.
//!
//! `select` is generic over what the caller's measurement produces, so this
//! module never names a codegen type: residency owns the *policy* (the ladder,
//! the budget, the search order, the rollback) and the caller supplies the
//! *measurement*.

use std::collections::{HashMap, HashSet};

use crate::ir::graph::{TensorId, TensorKind};
use crate::ir::psp::PspOp;
use crate::ir::PspModel;

/// Largest contiguous partition-2 block on a clean-booted PSP running psplink,
/// measured by `cargo run -p meminfo-host --release` (see `docs/PROGRESS.md`).
///
/// This is measured *with* meminfo's own ~72 KiB module resident, so it is
/// mildly conservative for a smaller module and mildly optimistic for a larger
/// one — `MODULE_IMAGE_RESERVE` covers the difference. It also assumes a clean
/// boot: partition memory leaked by an earlier run is not reclaimed on module
/// unload, so a polluted pool has strictly less than this.
pub const DEVICE_PARTITION_BYTES: usize = 55_734_016;

/// Device-side bytes codegen cannot see: the loaded PRX image (`.text`,
/// `.rodata`, `.data`) plus whatever statics the device crate itself declares.
///
/// Measured on BirdNET's release PRX: 1,056,640 B of image + 160,508 B of
/// device-crate statics = 1.16 MiB. Rounded up to 2 MiB so a moderately larger
/// example crate does not silently overrun the partition.
pub const MODULE_IMAGE_RESERVE: usize = 2 * 1024 * 1024;

/// How many bytes of resident weights fit alongside `generated_static_bytes`
/// of generated statics (arena + output + frame boundary buffers), which live
/// in `.bss` and are therefore claimed by the module loader *before* `init()`
/// allocates the weight blob.
pub fn resident_budget(generated_static_bytes: usize) -> usize {
    DEVICE_PARTITION_BYTES
        .saturating_sub(generated_static_bytes)
        .saturating_sub(MODULE_IMAGE_RESERVE)
}

/// Does a resident weight blob fit its budget?
///
/// `Ok(headroom)` / `Err(shortfall)`, both in bytes.
pub fn device_fit(blob_bytes: usize, budget: usize) -> Result<usize, usize> {
    if blob_bytes <= budget {
        Ok(budget - blob_bytes)
    } else {
        Err(blob_bytes - budget)
    }
}

/// One candidate weight-residency plan.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidencyCandidate {
    /// Constants left on disk and read chunkwise at op execution time.
    pub streamed: HashSet<TensorId>,
    /// IR estimate of the resident constant bytes.
    ///
    /// Only an estimate: the packed blob differs after VFPU pack-B
    /// substitution, dead-constant pruning, 16-byte alignment and appended FFT
    /// twiddles. Good enough to *order* candidates, never good enough to decide
    /// whether one fits — that needs the lowered `CodegenPlan::blob_bytes`.
    pub resident_estimate: usize,
    /// IR estimate of the bytes this candidate leaves on disk.
    pub streamed_estimate: usize,
}

/// Enumerate residency candidates, most-resident first.
///
/// Candidate 0 keeps every constant in memory. Each subsequent candidate also
/// streams the next-largest eligible weight, so the list is a ladder from
/// "everything resident" down to "everything streamable is streamed".
///
/// A weight is eligible only if it is the weight of a *batch-1* FullyConnected
/// used by exactly one op:
///
/// - **Single use**, because a weight read by two ops would be re-read twice.
/// - **FullyConnected**, because its access pattern is one sequential pass per
///   forward, so chunked reads pipeline with the matmul (hostfs sustains
///   ~21.5 MB/s at ≥64 KiB chunks).
/// - **Batch 1**, because that is what the streaming lowering supports — the
///   kernel walks output rows and cannot revisit a chunk for a second batch
///   row. Offering a candidate the lowerer will reject turns a planning
///   decision into a build failure, so the filter has to match `lower_ops`.
///
/// Largest-first because every streamed tensor costs a full re-read on *every*
/// inference, so freeing a given number of bytes is cheapest with the fewest,
/// biggest evictions.
pub fn analyze(model: &PspModel) -> Vec<ResidencyCandidate> {
    let mut const_total = 0usize;
    let mut use_count: HashMap<TensorId, usize> = HashMap::new();
    let mut fc_weights: HashSet<TensorId> = HashSet::new();

    let mut counted: HashSet<TensorId> = HashSet::new();
    for op in &model.graph.ops {
        for tid in op.inputs() {
            if matches!(model.graph.tensor(tid).kind, TensorKind::Constant { .. }) {
                *use_count.entry(tid).or_insert(0) += 1;
                if counted.insert(tid) {
                    const_total += model.graph.tensor(tid).size_bytes();
                }
            }
        }
        if let PspOp::FullyConnected { input, weights, .. } = op {
            let in_shape = &model.graph.tensor(*input).shape;
            let batch: usize = in_shape[..in_shape.len().saturating_sub(1)]
                .iter()
                .product::<usize>()
                .max(1);
            if batch == 1 {
                fc_weights.insert(*weights);
            }
        }
    }

    let mut eligible: Vec<(usize, TensorId)> = fc_weights
        .iter()
        .filter(|tid| use_count.get(tid) == Some(&1))
        .map(|&tid| (model.graph.tensor(tid).size_bytes(), tid))
        .collect();
    eligible.sort_unstable_by(|a, b| b.cmp(a));

    let mut candidates = vec![ResidencyCandidate {
        streamed: HashSet::new(),
        resident_estimate: const_total,
        streamed_estimate: 0,
    }];

    let mut streamed = HashSet::new();
    let mut streamed_bytes = 0usize;
    for (size, tid) in eligible {
        streamed.insert(tid);
        streamed_bytes += size;
        candidates.push(ResidencyCandidate {
            streamed: streamed.clone(),
            resident_estimate: const_total - streamed_bytes,
            streamed_estimate: streamed_bytes,
        });
    }
    candidates
}

/// The least-streamed candidate whose *estimated* resident footprint fits
/// `budget`, or the most-streamed one if none does.
///
/// Estimate-based, so this is for callers that cannot lower (and for tests).
/// `codegen` selects on the measured blob instead.
pub fn streamed_weights(model: &PspModel, budget: usize) -> HashSet<TensorId> {
    let candidates = analyze(model);
    candidates
        .iter()
        .find(|c| c.resident_estimate <= budget)
        .or_else(|| candidates.last())
        .map(|c| c.streamed.clone())
        .unwrap_or_default()
}

/// A candidate that has been measured, and the budget it was judged against.
pub struct Selected<P> {
    /// Whatever the caller's measurement produced for the winning candidate.
    pub payload: P,
    /// Measured resident bytes (not the IR estimate).
    pub blob_bytes: usize,
    /// Measured generated statics, which come off the budget.
    pub static_bytes: usize,
    pub budget: usize,
    /// Rung on the ladder, for reporting.
    pub index: usize,
    pub streamed: HashSet<TensorId>,
}

/// Choose a residency candidate, measuring each in turn until one fits.
///
/// `measure` applies a candidate and returns `(payload, resident_bytes,
/// generated_static_bytes)`. It has to be a callback because the decision is
/// made on the *lowered* blob — post pack-B substitution, dead-constant
/// pruning, 16-byte alignment and appended FFT twiddles — which differs from
/// [`ResidencyCandidate::resident_estimate`] by more than BirdNET's entire
/// margin. The estimate orders the ladder; only measurement decides.
///
/// Rejected candidates are rolled back by truncating `model_data` and
/// `graph.tensors`, which is exact because lowering is append-only (see
/// `codegen::lower`). Truncating to the mark is idempotent, so the first
/// attempt is a no-op.
///
/// `forced` pins a rung by index and skips the fit test.
pub fn select<P>(
    model: &mut PspModel,
    candidates: &[ResidencyCandidate],
    forced: Option<usize>,
    budget_override: Option<usize>,
    mut measure: impl FnMut(&mut PspModel, &HashSet<TensorId>) -> Result<(P, usize, usize), String>,
) -> Result<Selected<P>, String> {
    if let Some(i) = forced {
        if i >= candidates.len() {
            return Err(format!(
                "residency candidate {i} does not exist ({} available)",
                candidates.len()
            ));
        }
    }
    let mark = (model.model_data.len(), model.graph.tensors.len());
    let mut last: Option<Selected<P>> = None;

    for (index, cand) in candidates.iter().enumerate() {
        if forced.is_some_and(|f| f != index) {
            continue;
        }
        model.model_data.truncate(mark.0);
        model.graph.tensors.truncate(mark.1);

        let (payload, blob_bytes, static_bytes) = measure(model, &cand.streamed)?;
        let budget = budget_override.unwrap_or_else(|| resident_budget(static_bytes));
        let fits = device_fit(blob_bytes, budget).is_ok();
        let selected = Selected {
            payload,
            blob_bytes,
            static_bytes,
            budget,
            index,
            streamed: cand.streamed.clone(),
        };
        if fits || forced.is_some() {
            return Ok(selected);
        }
        last = Some(selected);
    }

    // Nothing fits. Keep the most-streamed plan and let `report_selection` say
    // so — a model that cannot fit is still worth emitting, because the device
    // failure is legible and the build log explains it.
    last.ok_or_else(|| "no residency candidates".to_string())
}

/// Print the ladder. Only when there is an actual choice to report.
pub fn report_candidates(candidates: &[ResidencyCandidate]) {
    if candidates.len() < 2 {
        return;
    }
    for (i, c) in candidates.iter().enumerate() {
        eprintln!(
            "Residency candidate {i}: ~{} resident, ~{} streamed ({} tensor{})",
            fmt_bytes(c.resident_estimate),
            fmt_bytes(c.streamed_estimate),
            c.streamed.len(),
            if c.streamed.len() == 1 { "" } else { "s" },
        );
    }
}

/// Print what was chosen and what it costs.
///
/// Unconditional: at BirdNET's ~1.6% margin a silent flip to streaming would
/// otherwise surface later as a mystery 1.2 s regression.
pub fn report_selection<P>(model: &PspModel, s: &Selected<P>) {
    for &tid in &s.streamed {
        eprintln!(
            "residency: streaming t{tid} ({}) from the weight file at runtime",
            fmt_bytes(model.graph.tensor(tid).size_bytes())
        );
    }
    match device_fit(s.blob_bytes, s.budget) {
        Ok(headroom) => eprintln!(
            "residency: candidate {} — resident {} + statics {} + reserve {} = {} of {} \
             partition ({} headroom)",
            s.index,
            fmt_bytes(s.blob_bytes),
            fmt_bytes(s.static_bytes),
            fmt_bytes(MODULE_IMAGE_RESERVE),
            fmt_bytes(s.blob_bytes + s.static_bytes + MODULE_IMAGE_RESERVE),
            fmt_bytes(DEVICE_PARTITION_BYTES),
            fmt_bytes(headroom),
        ),
        Err(shortfall) => eprintln!(
            "residency: WARNING no candidate fits — resident {} exceeds the {} budget by {}; \
             this model will fail to allocate on device",
            fmt_bytes(s.blob_bytes),
            fmt_bytes(s.budget),
            fmt_bytes(shortfall),
        ),
    }
}

/// Byte formatting, shared with `ModelStats`. Lives here because residency
/// reporting is its heaviest user.
pub(crate) fn fmt_bytes(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.1} KiB", bytes as f64 / 1_024.0)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::{DType, Graph, TensorKind};
    use crate::ir::psp::{FullyConnectedParams, PspOp};

    fn model_from(graph: Graph<PspOp>) -> PspModel {
        PspModel { graph, model_data: Vec::new() }
    }

    fn fc(graph: &mut Graph<PspOp>, input: TensorId, weights: TensorId, out_f: usize) -> TensorId {
        let output = graph.add_tensor(vec![1, out_f], DType::F32, TensorKind::Intermediate);
        graph.ops.push(PspOp::FullyConnected {
            input,
            weights,
            bias: None,
            output,
            fused_activation: FullyConnectedParams { fused_activation: None },
        });
        output
    }

    /// Chain of FullyConnected ops; `shapes` are `(out_features, in_features)`.
    /// Returns the model and the weight tensor of each op, in order.
    fn fc_chain(shapes: &[(usize, usize)]) -> (PspModel, Vec<TensorId>) {
        let mut g = Graph::<PspOp>::new();
        let mut act = g.add_tensor(vec![1, shapes[0].1], DType::F32, TensorKind::Input);
        let mut weights = Vec::new();
        let mut offset = 0usize;
        for &(out_f, in_f) in shapes {
            let len = out_f * in_f * 4;
            let w = g.add_tensor(
                vec![out_f, in_f],
                DType::F32,
                TensorKind::Constant { offset, len },
            );
            offset += len;
            act = fc(&mut g, act, w, out_f);
            weights.push(w);
        }
        (model_from(g), weights)
    }

    #[test]
    fn resident_budget_subtracts_statics_and_reserve() {
        assert_eq!(
            resident_budget(0),
            DEVICE_PARTITION_BYTES - MODULE_IMAGE_RESERVE
        );
        assert_eq!(
            resident_budget(1 << 20),
            DEVICE_PARTITION_BYTES - MODULE_IMAGE_RESERVE - (1 << 20)
        );
        // An arena larger than the whole partition yields no budget, not a panic.
        assert_eq!(resident_budget(usize::MAX), 0);
    }

    #[test]
    fn device_fit_reports_headroom_and_shortfall() {
        assert_eq!(device_fit(100, 256), Ok(156));
        assert_eq!(device_fit(256, 256), Ok(0));
        assert_eq!(device_fit(300, 256), Err(44));
    }

    #[test]
    fn nothing_streams_when_the_budget_is_generous() {
        let (model, _) = fc_chain(&[(100, 100), (200, 100)]);
        assert!(streamed_weights(&model, usize::MAX).is_empty());
    }

    #[test]
    fn largest_single_use_weight_streams_first() {
        // 40,000 B + 80,000 B of weights against a 100,000 B budget: evicting
        // the larger one is enough, and it is the one that must go.
        let (model, w) = fc_chain(&[(100, 100), (200, 100)]);
        let streamed = streamed_weights(&model, 100_000);
        assert_eq!(streamed, HashSet::from([w[1]]));
    }

    #[test]
    fn every_candidate_streams_at_a_zero_budget() {
        let (model, w) = fc_chain(&[(100, 100), (200, 100)]);
        let streamed = streamed_weights(&model, 0);
        assert_eq!(streamed, w.iter().copied().collect::<HashSet<_>>());
    }

    #[test]
    fn batched_fc_weights_are_never_candidates() {
        // The streaming lowering only supports batch 1. Offering a batched FC
        // as a candidate produced a lowering error ("streamed FullyConnected
        // requires batch 1, got 511") instead of a plan — BirdNET has three
        // such weights and they were showing up as rungs on the ladder.
        let mut g = Graph::<PspOp>::new();
        let input = g.add_tensor(vec![511, 100], DType::F32, TensorKind::Input);
        let w = g.add_tensor(
            vec![100, 100],
            DType::F32,
            TensorKind::Constant { offset: 0, len: 40_000 },
        );
        fc(&mut g, input, w, 100);
        let candidates = analyze(&model_from(g));
        assert_eq!(candidates.len(), 1, "batched FC must not add a rung");
        assert!(candidates[0].streamed.is_empty());
    }

    #[test]
    fn weights_shared_by_two_ops_are_never_streamed() {
        // Streaming reads a weight once per op from disk, so a tensor used
        // twice would be re-read twice — the heuristic must skip it.
        let mut g = Graph::<PspOp>::new();
        let input = g.add_tensor(vec![1, 100], DType::F32, TensorKind::Input);
        let shared = g.add_tensor(
            vec![100, 100],
            DType::F32,
            TensorKind::Constant { offset: 0, len: 40_000 },
        );
        let a = fc(&mut g, input, shared, 100);
        fc(&mut g, a, shared, 100);
        assert!(streamed_weights(&model_from(g), 0).is_empty());
    }

}
