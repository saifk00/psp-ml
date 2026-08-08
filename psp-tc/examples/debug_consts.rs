//! Debug tool: run lowering→quant→fuse→infer_shapes (NO const_fold, so the
//! I32 shape-computation ops are still present) and print every op whose
//! output got evaluated to an I32 constant, with its value. Compare against
//! the TFLite interpreter's runtime tensors to find where evaluation diverges.

use psp_tc::ir::graph::{DType, TensorKind};
use psp_tc::ir::psp::PspModel;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: debug_consts <model.tflite>");
    let data = std::fs::read(&path).unwrap();

    let graph = psp_tc::parse::tflite::lowering::lower(&data).unwrap();
    let mut model = PspModel {
        graph,
        model_data: data,
    };
    psp_tc::ir::quant::rewrite(&mut model).unwrap();
    psp_tc::ir::fuse::fuse(&mut model);
    psp_tc::ir::infer_shapes::infer(&mut model);

    for (i, op) in model.graph.ops.iter().enumerate() {
        for out in op.all_outputs() {
            let t = model.graph.tensor(out);
            if t.dtype != DType::I32 {
                continue;
            }
            if let TensorKind::Constant { offset, len } = t.kind {
                let vals: Vec<i32> = model.model_data[offset..offset + len]
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let n = vals.len();
                let head = &vals[..n.min(8)];
                let tail = if n > 8 { &vals[n - 4..] } else { &[][..] };
                println!(
                    "[{i:3}] {} => t{out} shape={:?} n={n} head={head:?} tail={tail:?}",
                    op, t.shape
                );
            }
        }
    }
}
