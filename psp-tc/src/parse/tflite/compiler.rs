use crate::ir::const_fold;
use crate::ir::dump;
use crate::ir::fuse;
use crate::ir::infer_shapes;
use crate::ir::PspModel;
use crate::memory_planner;
use crate::parse::tflite::lowering;

/// Convert a TFLite model buffer into PSP IR.
///
/// Takes ownership of the raw bytes so `PspModel` can pair the graph with
/// its backing weight data. When `dump_ir` is true, prints the IR graph
/// to stderr after each pipeline stage.
pub fn to_psp_ir(model_data: Vec<u8>, dump_ir: bool) -> Result<PspModel, String> {
    let graph = lowering::lower(&model_data)?;
    let mut model = PspModel { graph, model_data };
    if dump_ir {
        eprintln!("=== IR: after lowering ===\n{}", dump::dump(&model));
    }
    crate::ir::quant::rewrite(&mut model)?;
    if dump_ir {
        eprintln!("=== IR: after quant rewrite ===\n{}", dump::dump(&model));
    }
    fuse::fuse(&mut model);
    if dump_ir {
        eprintln!("=== IR: after fusion ===\n{}", dump::dump(&model));
    }
    infer_shapes::infer(&mut model);
    if dump_ir {
        eprintln!("=== IR: after infer_shapes ===\n{}", dump::dump(&model));
    }
    const_fold::fold(&mut model);
    // Needs folded constants (Pad amounts) and inferred shapes, so it runs
    // here rather than with the other fusions.
    fuse::fuse_late(&mut model);
    if dump_ir {
        eprintln!("=== IR: after const_fold ===\n{}", dump::dump(&model));
    }

    // IO planning; some models dont entirely fit in main memory (~54MB of user memory with high RAM enabled)
    // so we need to store some weights on disk and load them in time for their respective ops.
    let cumulative_weight_bytes = memory_planner::swap_analysis(&model);
    println!("Maximum constant tensor footprint: {} bytes", cumulative_weight_bytes.last().expect("no ops?"));

    Ok(model)
}
