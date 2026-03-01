use crate::ir::const_fold;
use crate::ir::dump;
use crate::ir::fuse;
use crate::ir::infer_shapes;
use crate::ir::PspModel;
use crate::parse::tflite::lowering;

/// Convert a TFLite model buffer into PSP IR.
///
/// Takes ownership of the raw bytes so `PspModel` can pair the graph with
/// its backing weight data. When `dump_ir` is true, prints the IR graph
/// to stderr after each pipeline stage.
pub fn to_psp_ir(model_data: Vec<u8>, dump_ir: bool) -> Result<PspModel, String> {
    let graph = lowering::lower(&model_data)?;
    let mut model = PspModel { graph, model_data };
    if dump_ir { eprintln!("=== IR: after lowering ===\n{}", dump::dump(&model)); }
    fuse::fuse(&mut model);
    if dump_ir { eprintln!("=== IR: after fusion ===\n{}", dump::dump(&model)); }
    infer_shapes::infer(&mut model);
    if dump_ir { eprintln!("=== IR: after infer_shapes ===\n{}", dump::dump(&model)); }
    const_fold::fold(&mut model);
    if dump_ir { eprintln!("=== IR: after const_fold ===\n{}", dump::dump(&model)); }
    Ok(model)
}
