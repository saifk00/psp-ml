//! Pretty-printer for PspModel IR graphs.

use std::fmt::Write;

use super::graph::{DType, TensorKind};
use super::psp::PspModel;

/// Return a human-readable string representation of the model's IR graph.
pub fn dump(model: &PspModel) -> String {
    let mut out = String::new();
    let g = &model.graph;

    // --- Tensor summary ---
    let mut n_input = 0usize;
    let mut n_output = 0usize;
    let mut n_const = 0usize;
    let mut n_inter = 0usize;
    for t in &g.tensors {
        match t.kind {
            TensorKind::Input => n_input += 1,
            TensorKind::Output => n_output += 1,
            TensorKind::Constant { .. } => n_const += 1,
            TensorKind::Intermediate => n_inter += 1,
        }
    }
    let _ = writeln!(
        out,
        "Tensors ({} total, {} inputs, {} outputs, {} constants, {} intermediates):",
        g.tensors.len(),
        n_input,
        n_output,
        n_const,
        n_inter
    );
    for t in &g.tensors {
        let kind = match &t.kind {
            TensorKind::Input => "Input".to_string(),
            TensorKind::Output => "Output".to_string(),
            TensorKind::Constant { len, .. } => format!("Constant({} bytes)", len),
            TensorKind::Intermediate => "Intermediate".to_string(),
        };
        let dtype = match t.dtype {
            DType::F32 => "F32",
            DType::I32 => "I32",
            DType::I8 => "I8",
            DType::U8 => "U8",
        };
        let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        let _ = writeln!(out, "  t{:<4} {:<22} {} [{}]", t.id, kind, dtype, shape.join(", "));
    }

    // --- Ops ---
    let _ = writeln!(out, "\nOps ({} total):", g.ops.len());
    for (i, op) in g.ops.iter().enumerate() {
        let _ = writeln!(out, "  [{:<3}] {}", i, op);
    }

    // --- Graph I/O ---
    let inputs: Vec<String> = g.inputs.iter().map(|id| format!("t{}", id)).collect();
    let outputs: Vec<String> = g.outputs.iter().map(|id| format!("t{}", id)).collect();
    let _ = writeln!(out, "\nGraph: inputs=[{}] outputs=[{}]", inputs.join(", "), outputs.join(", "));

    out
}
