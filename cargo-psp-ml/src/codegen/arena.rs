//! Arena-based buffer reuse via compile-time liveness analysis.
//!
//! Two layers:
//! 1. **Liveness analysis** — generic, computes [first_op, last_op] for every tensor.
//!    Reusable for weight partitioning in the future.
//! 2. **Arena packing** — greedy first-fit allocator that packs intermediate tensors
//!    and scratch buffers into a single contiguous arena.

use std::collections::HashMap;

use crate::ir::graph::TensorId;

use super::plan::*;

// ─── Layer 1: Liveness Analysis ────────────────────────────────────

/// A reference to a tensor from a kernel call.
#[derive(Debug, Clone, Copy)]
pub struct TensorRef {
    pub tensor_id: TensorId,
    pub kind: RefKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefKind {
    Read,
    Write,
}

/// Liveness interval for a tensor: the range of op indices that reference it.
#[derive(Debug, Clone)]
pub struct LiveRange {
    pub tensor_id: TensorId,
    pub first_op: usize,
    pub last_op: usize,
}

/// Extract all tensor references from a kernel call.
///
/// This is the single maintenance point — update when new KernelCall variants are added.
/// ScratchId fields are ignored here (handled separately in layer 2).
pub fn extract_tensor_refs(kernel: &KernelCall) -> Vec<TensorRef> {
    let mut refs = Vec::new();
    let r = |id: TensorId| TensorRef { tensor_id: id, kind: RefKind::Read };
    let w = |id: TensorId| TensorRef { tensor_id: id, kind: RefKind::Write };

    match kernel {
        KernelCall::DepthwiseConv2d { input, filter, bias, output, .. } => {
            refs.push(r(input.id));
            refs.push(r(filter.id));
            if let Some(b) = bias { refs.push(r(*b)); }
            refs.push(w(output.id));
        }
        KernelCall::Conv2d { input, filter, bias, output, .. } => {
            refs.push(r(input.id));
            refs.push(r(filter.id));
            if let Some(b) = bias { refs.push(r(*b)); }
            refs.push(w(output.id));
        }
        KernelCall::Im2colPadded { input, .. } => {
            refs.push(r(input.id));
            // output is ScratchId, not TensorId
        }
        KernelCall::MatmulBtTiled { output, .. } => {
            // a, b are ScratchIds
            refs.push(w(*output));
        }
        KernelCall::BiasAdd { output, bias, .. } => {
            // output is read+write (in-place addition)
            refs.push(w(*output));
            refs.push(r(*output));
            refs.push(r(*bias));
        }
        KernelCall::Relu { output } => {
            // in-place
            refs.push(w(*output));
            refs.push(r(*output));
        }
        KernelCall::Pool2d { input, output, .. } => {
            refs.push(r(input.id));
            refs.push(w(output.id));
        }
        KernelCall::Reshape { input, output } => {
            refs.push(r(*input));
            refs.push(w(*output));
        }
        KernelCall::FullyConnected { input, weights, bias, output, .. } => {
            refs.push(r(*input));
            refs.push(r(*weights));
            if let Some(b) = bias { refs.push(r(*b)); }
            refs.push(w(*output));
        }
        KernelCall::ElementWise { input_a, input_b, output, .. } => {
            refs.push(r(*input_a));
            refs.push(r(*input_b));
            refs.push(w(*output));
        }
        KernelCall::UnaryElementWise { input, output, .. } => {
            refs.push(r(*input));
            refs.push(w(*output));
        }
        KernelCall::Reduce { input, output, .. } => {
            refs.push(r(*input));
            refs.push(w(*output));
        }
        KernelCall::Pad { input, output, .. } => {
            refs.push(r(input.id));
            refs.push(w(output.id));
        }
        KernelCall::Transpose { input, output, .. } => {
            refs.push(r(*input));
            refs.push(w(*output));
        }
        KernelCall::ReverseV2 { input, output, .. } => {
            refs.push(r(*input));
            refs.push(w(*output));
        }
        KernelCall::RfftPack { input, .. } => {
            refs.push(r(*input));
            // output is ScratchId
        }
        KernelCall::FftButterflyStage { twiddles, .. } => {
            refs.push(r(*twiddles));
            // data is ScratchId
        }
        KernelCall::RfftUnpack { twiddles, output, .. } => {
            refs.push(r(*twiddles));
            refs.push(w(*output));
            // data is ScratchId
        }
        KernelCall::StridedSlice { input, output, .. } => {
            refs.push(r(*input));
            refs.push(w(*output));
        }
        KernelCall::Gather { input, output, indices, .. } => {
            refs.push(r(*input));
            refs.push(r(*indices));
            refs.push(w(*output));
        }
        KernelCall::Concatenation { inputs, output, .. } => {
            for id in inputs { refs.push(r(*id)); }
            refs.push(w(*output));
        }
    }
    refs
}

/// Compute liveness intervals for all tensors referenced in the plan.
///
/// Returns a LiveRange per unique TensorId, covering the full span of ops
/// that reference it. Covers all tensor kinds (constants, intermediates,
/// input, output) — callers filter by kind as needed.
pub fn compute_liveness(plan: &CodegenPlan) -> Vec<LiveRange> {
    let mut map: HashMap<TensorId, (usize, usize)> = HashMap::new();

    for (op_idx, op) in plan.ops.iter().enumerate() {
        for sub_op in &op.sub_ops {
            for kernel in &sub_op.kernels {
                for tref in extract_tensor_refs(kernel) {
                    map.entry(tref.tensor_id)
                        .and_modify(|(first, last)| {
                            *first = (*first).min(op_idx);
                            *last = (*last).max(op_idx);
                        })
                        .or_insert((op_idx, op_idx));
                }
            }
        }
    }

    map.into_iter()
        .map(|(tensor_id, (first_op, last_op))| LiveRange { tensor_id, first_op, last_op })
        .collect()
}

// ─── Layer 2: Arena Packing ────────────────────────────────────────

/// Identifies a buffer slot in the arena.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArenaSlot {
    Tensor(TensorId),
    Scratch { op_idx: usize, scratch_idx: usize },
}

/// An item to pack into the arena.
#[derive(Debug, Clone)]
struct PackItem {
    slot: ArenaSlot,
    size_floats: usize,
    first_op: usize,
    last_op: usize,
}

/// Result of arena packing: per-slot offset assignments and total arena size.
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaLayout {
    pub arena_size_floats: usize,
    /// Offset (in f32s) for each slot within the arena.
    pub offsets: HashMap<ArenaSlot, usize>,
}

const ALIGN_FLOATS: usize = 4; // 4 × f32 = 16 bytes

fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// Build the list of items that need arena slots from intermediate tensors
/// and scratch buffers.
fn build_pack_items(plan: &CodegenPlan, liveness: &[LiveRange]) -> Vec<PackItem> {
    let live_map: HashMap<TensorId, &LiveRange> =
        liveness.iter().map(|lr| (lr.tensor_id, lr)).collect();

    let mut items = Vec::new();

    // Intermediate tensors
    for alloc in &plan.allocs {
        if let TensorAlloc::Intermediate { id, size } = alloc {
            if let Some(lr) = live_map.get(id) {
                items.push(PackItem {
                    slot: ArenaSlot::Tensor(*id),
                    size_floats: *size,
                    first_op: lr.first_op,
                    last_op: lr.last_op,
                });
            }
        }
    }

    // Scratch buffers (scoped to their owning op)
    for (op_idx, op) in plan.ops.iter().enumerate() {
        for (scratch_idx, scratch) in op.scratch.iter().enumerate() {
            items.push(PackItem {
                slot: ArenaSlot::Scratch { op_idx, scratch_idx },
                size_floats: scratch.size,
                first_op: op_idx,
                last_op: op_idx,
            });
        }
    }

    items
}

/// Pack items into a contiguous arena using greedy first-fit with gap coalescing.
///
/// Returns the layout with per-slot offsets and total arena size.
fn pack_arena(items: &mut [PackItem]) -> ArenaLayout {
    // Sort by first_op, then by descending size (large items first for better packing)
    items.sort_by(|a, b| {
        a.first_op.cmp(&b.first_op)
            .then(b.size_floats.cmp(&a.size_floats))
    });

    // Free list: sorted by offset, each entry is (offset, size)
    let mut free_list: Vec<(usize, usize)> = Vec::new();
    // Track assigned items so we can free them when they die
    let mut assigned: Vec<(usize, usize, usize)> = Vec::new(); // (offset, size, last_op)
    let mut offsets = HashMap::new();
    let mut arena_end: usize = 0;

    for item in items.iter() {
        let needed = align_up(item.size_floats, ALIGN_FLOATS);

        // Free all items whose last_op < current first_op
        let mut freed = Vec::new();
        assigned.retain(|&(offset, size, last_op)| {
            if last_op < item.first_op {
                freed.push((offset, size));
                false
            } else {
                true
            }
        });

        // Return freed blocks to the free list and coalesce
        for (offset, size) in freed {
            insert_and_coalesce(&mut free_list, offset, size);
        }

        // First-fit: find a gap that can hold this item (aligned)
        let mut chosen_offset = None;
        for (i, &(gap_off, gap_size)) in free_list.iter().enumerate() {
            let aligned_off = align_up(gap_off, ALIGN_FLOATS);
            let padding = aligned_off - gap_off;
            if gap_size >= padding + needed {
                chosen_offset = Some((i, aligned_off));
                break;
            }
        }

        let offset = if let Some((gap_idx, aligned_off)) = chosen_offset {
            let (gap_off, gap_size) = free_list[gap_idx];
            let padding_before = aligned_off - gap_off;
            let remaining = gap_size - padding_before - needed;

            // Remove the used gap
            free_list.remove(gap_idx);

            // Re-insert any leftover fragments
            if padding_before > 0 {
                insert_and_coalesce(&mut free_list, gap_off, padding_before);
            }
            if remaining > 0 {
                insert_and_coalesce(&mut free_list, aligned_off + needed, remaining);
            }

            aligned_off
        } else {
            // No gap fits — extend the arena
            let aligned_off = align_up(arena_end, ALIGN_FLOATS);
            arena_end = aligned_off + needed;
            aligned_off
        };

        offsets.insert(item.slot.clone(), offset);
        assigned.push((offset, needed, item.last_op));
    }

    ArenaLayout {
        arena_size_floats: arena_end,
        offsets,
    }
}

/// Insert a freed block into the sorted free list and coalesce adjacent gaps.
fn insert_and_coalesce(free_list: &mut Vec<(usize, usize)>, offset: usize, size: usize) {
    // Find insertion point (keep sorted by offset)
    let pos = free_list.partition_point(|&(o, _)| o < offset);
    free_list.insert(pos, (offset, size));

    // Coalesce with next
    if pos + 1 < free_list.len() {
        let (o1, s1) = free_list[pos];
        let (o2, s2) = free_list[pos + 1];
        if o1 + s1 == o2 {
            free_list[pos] = (o1, s1 + s2);
            free_list.remove(pos + 1);
        }
    }

    // Coalesce with previous
    if pos > 0 {
        let (o_prev, s_prev) = free_list[pos - 1];
        let (o_cur, s_cur) = free_list[pos];
        if o_prev + s_prev == o_cur {
            free_list[pos - 1] = (o_prev, s_prev + s_cur);
            free_list.remove(pos);
        }
    }
}

/// Compute liveness and pack all intermediate + scratch buffers into a shared arena.
pub fn compute_and_pack(plan: &CodegenPlan) -> ArenaLayout {
    let liveness = compute_liveness(plan);
    let mut items = build_pack_items(plan, &liveness);
    pack_arena(&mut items)
}
