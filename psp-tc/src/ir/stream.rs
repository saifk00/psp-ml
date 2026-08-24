//! Batch stream analysis and graph rewrite.
//!
//! Two passes, run after `const_fold` and before codegen lowering:
//!
//! 1. **`analyze`** — walks the graph, finds maximal contiguous regions of
//!    batch-independent ops eligible for frame streaming.
//! 2. **`rewrite`** — permanently transforms the graph for a selected region,
//!    setting batch dim to 1 so the lowerer sees a clean single-frame graph.

use std::collections::HashSet;

use crate::ir::graph::{TensorId, TensorKind};
use crate::ir::psp::{PspModel, PspOp};

// Re-use the codegen plan types that arena + render already consume.
use crate::codegen::plan::{FrameBoundaryTensor, StreamPlan};

// ─── Data structures ─────────────────────────────────────────────

/// A contiguous region of batch-independent ops eligible for frame streaming.
#[derive(Debug, Clone)]
pub struct StreamRegion {
    /// Index of first op in the region.
    pub start: usize,
    /// Index of last op in the region (inclusive).
    pub end: usize,
    /// Batch dimension (N), e.g. 511.
    pub frame_count: usize,
    /// Tensors produced before the region, consumed inside it.
    pub frame_inputs: Vec<BoundaryTensor>,
    /// Tensors produced inside the region, consumed after it.
    pub frame_outputs: Vec<BoundaryTensor>,
}

/// A tensor that crosses the streaming region boundary.
#[derive(Debug, Clone)]
pub struct BoundaryTensor {
    pub id: TensorId,
    /// Total element count at full batch (N × frame_stride).
    pub total_elems: usize,
    /// Elements per frame (product of dims 1+).
    pub frame_stride: usize,
}

// ─── Analysis pass ───────────────────────────────────────────────

/// Walk the graph and return all maximal contiguous regions of batch-independent ops.
///
/// Regions are sorted by start index. Only regions with `frame_count > 1` are returned.
pub fn analyze(model: &PspModel) -> Vec<StreamRegion> {
    let num_ops = model.graph.ops.len();

    // Step 1: classify each op
    let independent: Vec<bool> = model
        .graph
        .ops
        .iter()
        .map(|op| is_batch_independent(model, op))
        .collect();

    // Step 2: find maximal contiguous runs
    let mut regions = Vec::new();
    let mut i = 0;
    while i < num_ops {
        if independent[i] {
            let start = i;
            while i < num_ops && independent[i] {
                i += 1;
            }
            let end = i - 1; // inclusive

            // Step 3: detect frame_count from output tensor shapes
            if let Some(frame_count) = detect_frame_count(model, start, end) {
                if frame_count > 1 {
                    let (frame_inputs, frame_outputs) =
                        compute_boundary_tensors(model, start, end, frame_count);
                    regions.push(StreamRegion {
                        start,
                        end,
                        frame_count,
                        frame_inputs,
                        frame_outputs,
                    });
                }
            }
        } else {
            i += 1;
        }
    }

    regions
}

/// Check whether a single PspOp is batch-independent (treats dim 0 as independent batch).
///
/// Conservative: returns `false` when we can't determine statically.
fn is_batch_independent(model: &PspModel, op: &PspOp) -> bool {
    match op {
        // Always batch-independent: spatial / pointwise / per-element
        PspOp::Conv2d { .. }
        | PspOp::DepthwiseConv2d { .. }
        | PspOp::Pool2d { .. }
        | PspOp::ElementWise { .. }
        | PspOp::PowConst { .. }
        | PspOp::UnaryElementWise { .. }
        | PspOp::FakeQuant { .. }
        | PspOp::Dequantize { .. }
        | PspOp::Swish { .. }
        | PspOp::FullyConnected { .. }
        | PspOp::Softmax { .. }
        | PspOp::Cast { .. }
        | PspOp::Squeeze { .. }
        | PspOp::ExpandDims { .. } => true,

        // Reduce: only if axis 0 is not being reduced
        PspOp::Reduce { axes, .. } => match model.read_i32_const(*axes) {
            Some(axes_vals) => !axes_vals.contains(&0),
            None => false,
        },

        // Reshape: batch-independent if output shape[0] == input shape[0]
        PspOp::Reshape { input, output, .. } => {
            let in_shape = &model.graph.tensor(*input).shape;
            let out_shape = &model.graph.tensor(*output).shape;
            !in_shape.is_empty() && !out_shape.is_empty() && in_shape[0] == out_shape[0]
        }

        // Pad: batch-independent if padding[0] == [0, 0]
        PspOp::Pad { paddings, .. } => match model.read_i32_const(*paddings) {
            Some(vals) => vals.len() >= 2 && vals[0] == 0 && vals[1] == 0,
            None => false,
        },

        // Transpose: batch-independent if perm[0] == 0
        PspOp::Transpose { perm, .. } => match model.read_i32_const(*perm) {
            Some(perm_vals) => !perm_vals.is_empty() && perm_vals[0] == 0,
            None => false,
        },

        // Concatenation: batch-independent if not concatenating along dim 0
        PspOp::Concatenation { axis, .. } => *axis != 0,

        // Gather: batch-independent if not gathering along dim 0
        PspOp::Gather { axis, .. } => *axis != 0,

        // StridedViewStft: frames overlap in the input signal, so dim 0 of
        // the output is not independently sliceable from dim 0 of the input.
        PspOp::StridedViewStft { .. } => false,

        // ReverseV2: batch-independent if not reversing dim 0
        PspOp::ReverseV2 { axis, .. } => match model.read_i32_const(*axis) {
            Some(axis_vals) => !axis_vals.contains(&0),
            None => false,
        },

        // StridedSlice: batch-independent if dim 0 is kept as full range
        PspOp::StridedSlice {
            shrink_axis_mask,
            begin_mask,
            end_mask,
            ..
        } => {
            // If shrink_axis_mask bit 0 is set, dim 0 is removed → batch-dependent
            if *shrink_axis_mask & 1 != 0 {
                return false;
            }
            // If both begin_mask and end_mask have bit 0 → full range on dim 0
            *begin_mask & 1 != 0 && *end_mask & 1 != 0
        }

        // Preprocessing / should be eliminated by const_fold+fuse
        PspOp::Rfft { .. }
        | PspOp::Rfft2d { .. }
        | PspOp::Shape { .. }
        | PspOp::Range { .. }
        | PspOp::Pack { .. }
        | PspOp::SplitV { .. } => false,
    }
}

/// Detect frame count from output tensor shapes in a region.
///
/// Scans ops in [start..=end] for the first output tensor with shape[0] > 1.
fn detect_frame_count(model: &PspModel, start: usize, end: usize) -> Option<usize> {
    for op_idx in start..=end {
        for id in model.graph.ops[op_idx].all_outputs() {
            let shape = &model.graph.tensors[id].shape;
            if shape.len() >= 2 && shape[0] > 1 {
                return Some(shape[0]);
            }
        }
    }
    None
}

/// Compute boundary tensors for a region [start..=end] with the given frame_count.
fn compute_boundary_tensors(
    model: &PspModel,
    start: usize,
    end: usize,
    frame_count: usize,
) -> (Vec<BoundaryTensor>, Vec<BoundaryTensor>) {
    let num_ops = model.graph.ops.len();

    // Tensors written/read by frame-section ops
    let mut frame_written: HashSet<TensorId> = HashSet::new();
    let mut frame_read: HashSet<TensorId> = HashSet::new();
    for op_idx in start..=end {
        let op = &model.graph.ops[op_idx];
        for id in op.inputs() {
            frame_read.insert(id);
        }
        for id in op.all_outputs() {
            frame_written.insert(id);
        }
    }

    // Tensors available before the region (pre-written)
    let mut pre_written: HashSet<TensorId> = HashSet::new();
    for &id in &model.graph.inputs {
        pre_written.insert(id);
    }
    for op_idx in 0..start {
        for id in model.graph.ops[op_idx].all_outputs() {
            pre_written.insert(id);
        }
    }

    // Tensors consumed after the region (post-read)
    let mut post_read: HashSet<TensorId> = HashSet::new();
    for op_idx in (end + 1)..num_ops {
        for id in model.graph.ops[op_idx].inputs() {
            post_read.insert(id);
        }
    }

    // Frame inputs: produced before, consumed inside, non-constant, batched
    let mut frame_inputs = Vec::new();
    for &id in &frame_read {
        let tensor = &model.graph.tensors[id];
        let is_pre = pre_written.contains(&id);
        let is_fw = frame_written.contains(&id);
        let is_const = matches!(tensor.kind, TensorKind::Constant { .. });
        if is_pre && !is_fw && !is_const {
            let shape = &tensor.shape;
            if !shape.is_empty() && shape[0] == frame_count {
                let total_elems: usize = shape.iter().product();
                let frame_stride = total_elems / frame_count;
                frame_inputs.push(BoundaryTensor {
                    id,
                    total_elems,
                    frame_stride,
                });
            }
        }
    }

    // Frame outputs: produced inside, consumed after, batched
    let mut frame_outputs = Vec::new();
    for &id in &frame_written {
        if post_read.contains(&id) {
            let shape = &model.graph.tensors[id].shape;
            if !shape.is_empty() && shape[0] == frame_count {
                let total_elems: usize = shape.iter().product();
                let frame_stride = total_elems / frame_count;
                frame_outputs.push(BoundaryTensor {
                    id,
                    total_elems,
                    frame_stride,
                });
            }
        }
    }

    (frame_inputs, frame_outputs)
}

// ─── Rewrite pass ────────────────────────────────────────────────

/// Permanently rewrite the graph for streaming and return a `StreamPlan`.
///
/// - Internal intermediates: shape[0] set to 1 permanently.
/// - Boundary tensors: shape stays full-batch. A batch=1 **view tensor** is
///   created and ops inside the frame region are rewired to use it.
///   The lowerer sees consistent shapes; the render layer maps views to
///   frame-offset slices of the boundary buffers.
pub fn rewrite(model: &mut PspModel, region: &StreamRegion) -> StreamPlan {
    let mut rewritten_ids = HashSet::new();

    // Collect boundary tensor IDs (both inputs and outputs)
    let boundary_ids: HashSet<TensorId> = region
        .frame_inputs
        .iter()
        .chain(region.frame_outputs.iter())
        .map(|bt| bt.id)
        .collect();

    // Rewrite internal intermediates (not boundary tensors)
    for op_idx in region.start..=region.end {
        for tid in model.graph.ops[op_idx].all_outputs() {
            if boundary_ids.contains(&tid) {
                continue;
            }
            let tensor = &mut model.graph.tensors[tid];
            if !tensor.shape.is_empty() && tensor.shape[0] == region.frame_count {
                tensor.shape[0] = 1;
                rewritten_ids.insert(tid);
            }
        }
    }

    // Create view tensors for boundary tensors and rewire ops
    let create_view = |model: &mut PspModel, bt: &BoundaryTensor, rewritten_ids: &mut HashSet<TensorId>| -> TensorId {
        let orig = &model.graph.tensors[bt.id];
        let mut view_shape = orig.shape.clone();
        view_shape[0] = 1;
        let dtype = orig.dtype;
        let view_id = model.graph.add_tensor(view_shape, dtype, TensorKind::Intermediate);
        rewritten_ids.insert(view_id);

        // Rewire ops inside the frame region
        for op_idx in region.start..=region.end {
            model.graph.ops[op_idx].replace_tensor_id(bt.id, view_id);
        }
        view_id
    };

    let frame_inputs: Vec<FrameBoundaryTensor> = region
        .frame_inputs
        .iter()
        .map(|bt| {
            let view_id = create_view(model, bt, &mut rewritten_ids);
            FrameBoundaryTensor {
                id: bt.id,
                view_id,
                total_size: bt.total_elems,
                frame_stride: bt.frame_stride,
            }
        })
        .collect();

    let frame_outputs: Vec<FrameBoundaryTensor> = region
        .frame_outputs
        .iter()
        .map(|bt| {
            let view_id = create_view(model, bt, &mut rewritten_ids);
            FrameBoundaryTensor {
                id: bt.id,
                view_id,
                total_size: bt.total_elems,
                frame_stride: bt.frame_stride,
            }
        })
        .collect();

    StreamPlan {
        frame_count: region.frame_count,
        frame_start: region.start,
        frame_end: region.end,
        frame_inputs,
        frame_outputs,
        rewritten_tensor_ids: rewritten_ids,
    }
}

/// Find the region that contains the user-specified [start..=end] range.
///
/// The user's range must be fully contained within one analyzed region.
/// Boundary tensors are recomputed for the exact sub-range.
pub fn select_region(
    model: &PspModel,
    regions: &[StreamRegion],
    user_start: usize,
    user_end: usize,
) -> Result<StreamRegion, String> {
    for region in regions {
        if region.start <= user_start && region.end >= user_end {
            let (frame_inputs, frame_outputs) =
                compute_boundary_tensors(model, user_start, user_end, region.frame_count);
            return Ok(StreamRegion {
                start: user_start,
                end: user_end,
                frame_count: region.frame_count,
                frame_inputs,
                frame_outputs,
            });
        }
    }

    let region_summary: Vec<String> = regions
        .iter()
        .map(|r| format!("[{}..={}] (N={}, {} ops)", r.start, r.end, r.frame_count, r.end - r.start + 1))
        .collect();

    Err(format!(
        "No eligible streaming region contains [{}..={}].\n\
         Discovered regions:\n  {}",
        user_start,
        user_end,
        if region_summary.is_empty() {
            "(none)".to_string()
        } else {
            region_summary.join("\n  ")
        }
    ))
}
