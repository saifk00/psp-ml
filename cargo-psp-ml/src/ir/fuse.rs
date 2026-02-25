//! Operator fusion pass.
//!
//! Runs after constant folding, before codegen lowering.
//! Pattern-matches op sequences and replaces them with fused ops.

use std::collections::HashMap;

use super::graph::{DType, TensorKind};
use super::psp::{PspModel, PspOp};
use crate::ir::graph::TensorId;

/// Run all fusion rules on the model.
pub fn fuse(model: &mut PspModel) {
    let count = fuse_rfft(model);
    if count > 0 {
        eprintln!("fuse: fused {} RFFT2D+SQUEEZE+CAST chains", count);
    }
}

/// Build map from tensor ID → vec of op indices that consume it.
fn consumer_map(model: &PspModel) -> HashMap<TensorId, Vec<usize>> {
    let mut map: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, op) in model.graph.ops.iter().enumerate() {
        for id in op.inputs() {
            map.entry(id).or_default().push(idx);
        }
    }
    map
}

/// Read an INT32 constant tensor as a Vec<i32>.
fn read_i32(model: &PspModel, id: TensorId) -> Option<Vec<i32>> {
    let t = model.graph.tensor(id);
    if let TensorKind::Constant { offset, len } = t.kind {
        Some(
            model.model_data[offset..offset + len]
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    } else {
        None
    }
}

/// Fuse RFFT2D → Reshape(SQUEEZE) → Cast(C64→F32) into a single Rfft op.
///
/// Returns the number of fused chains.
fn fuse_rfft(model: &mut PspModel) -> usize {
    let consumers = consumer_map(model);
    let mut to_remove: Vec<usize> = Vec::new();
    let mut replacements: Vec<(usize, PspOp)> = Vec::new();

    for (i, op) in model.graph.ops.iter().enumerate() {
        let (input, fft_length_id, complex_out) = match op {
            PspOp::Rfft2d {
                input,
                fft_length,
                output,
            } => (*input, *fft_length, *output),
            _ => continue,
        };

        // Read fft_length constant: expect [1, N] (1D FFT along last dim)
        let fft_length_vals = match read_i32(model, fft_length_id) {
            Some(v) => v,
            None => continue,
        };
        if fft_length_vals.len() != 2 || fft_length_vals[0] != 1 {
            eprintln!(
                "fuse: skipping RFFT2D with non-1D fft_length: {:?}",
                fft_length_vals
            );
            continue;
        }
        let n = fft_length_vals[1] as usize;

        // Trace: complex_out → optional Reshape(SQUEEZE) → Cast(C64→F32)
        // The SQUEEZE may be aliased away during TFLite lowering, so handle both:
        //   Rfft2d → Reshape → Cast   (3-op chain)
        //   Rfft2d → Cast             (2-op chain, SQUEEZE was aliased)
        let next_consumers = match consumers.get(&complex_out) {
            Some(cs) if cs.len() == 1 => cs,
            _ => continue,
        };
        let next_idx = next_consumers[0];
        let cast_idx = match &model.graph.ops[next_idx] {
            // Direct Cast (SQUEEZE was aliased away during TFLite lowering)
            PspOp::Cast { .. } => next_idx,
            // Reshape (SQUEEZE still present), then Cast
            PspOp::Reshape { output, .. } => {
                let squeezed_out = *output;
                let cast_consumers = match consumers.get(&squeezed_out) {
                    Some(cs) if cs.len() == 1 => cs,
                    _ => continue,
                };
                match &model.graph.ops[cast_consumers[0]] {
                    PspOp::Cast { .. } => {
                        to_remove.push(next_idx); // remove the Reshape
                        cast_consumers[0]
                    }
                    _ => continue,
                }
            }
            _ => continue,
        };
        let final_out = match &model.graph.ops[cast_idx] {
            PspOp::Cast { output, .. } => *output,
            _ => continue,
        };

        // Verify the final output is F32
        if model.graph.tensor(final_out).dtype != DType::F32 {
            continue;
        }

        // Replace RFFT2D with fused Rfft, remove Cast (Reshape already handled above)
        replacements.push((
            i,
            PspOp::Rfft {
                input,
                output: final_out,
                fft_length: n,
            },
        ));
        to_remove.push(cast_idx);
    }

    let count = replacements.len();

    // Apply replacements
    for (idx, new_op) in replacements {
        model.graph.ops[idx] = new_op;
    }

    // Remove consumed ops in reverse order to preserve indices
    to_remove.sort_unstable();
    to_remove.dedup();
    for idx in to_remove.into_iter().rev() {
        model.graph.ops.remove(idx);
    }

    count
}
