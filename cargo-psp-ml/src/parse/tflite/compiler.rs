use crate::ir::const_fold;
use crate::ir::fuse;
use crate::ir::PspModel;
use crate::parse::tflite::lowering;

/// Convert a TFLite model buffer into PSP IR.
///
/// Takes ownership of the raw bytes so `PspModel` can pair the graph with
/// its backing weight data.
pub fn to_psp_ir(model_data: Vec<u8>) -> Result<PspModel, String> {
    let graph = lowering::lower(&model_data)?;
    let mut model = PspModel { graph, model_data };
    fuse::fuse(&mut model);
    const_fold::fold(&mut model);
    Ok(model)
}
