//! TFLite model compiler — generates Rust code targeting the `psp-rt` runtime library.

pub mod builder;
pub mod codegen;
pub mod ir;
pub mod memory_planner;
pub mod parse;

pub use builder::PspModelBuilder;
pub use codegen::ModelStats;

use ir::PspModel;
use std::fs;
use std::path::Path;

/// Compiles a TFLite model into `generated.rs` + `weights.bin` in `out_dir`.
///
/// This is the entry point `build.rs` scripts call — see `psp-tc`'s CLI
/// binary (`src/bin/psp-tc.rs`) for a `--dump-ir`-capable alternative that
/// drives the lower-level `parse`/`codegen` modules directly for
/// interactive/diagnostic use.
///
/// `weights.bin` needs no special handling by the caller: generated code
/// references it via `include_bytes!("weights.bin")`, a path resolved
/// relative to the including file, so as long as both files land in the
/// same `out_dir` it resolves correctly regardless of where `out_dir` is.
pub fn compile_tflite(
    model_path: &Path,
    out_dir: &Path,
    stream_batch: Option<(usize, usize)>,
) -> Result<ModelStats, String> {
    let data = fs::read(model_path)
        .map_err(|err| format!("failed to read {}: {err}", model_path.display()))?;

    let mut psp_model = parse::tflite::to_psp_ir(data, false)
        .map_err(|err| format!("error lowering to IR: {err}"))?;

    let generated = codegen::generate_code(&mut psp_model, stream_batch, None, None)
        .map_err(|err| format!("codegen error: {err}"))?;

    write_generated(generated, out_dir, "generated.rs")
}

/// `compile_tflite` under a caller-chosen module name: emits `<name>.rs` +
/// `<name>_weights.bin` into `out_dir`, so a crate can embed several
/// generated modules (`mod a { include!(...a.rs) }`) without their blob
/// names colliding. `include_bytes!` resolves next to the generated file;
/// if the blob exceeds the embed threshold the device loads
/// `host0:/<name>_weights.bin` (mount-root) and the host `$OUT_DIR/<name>_weights.bin`.
pub fn compile_tflite_named(
    model_path: &Path,
    out_dir: &Path,
    stream_batch: Option<(usize, usize)>,
    name: &str,
) -> Result<ModelStats, String> {
    let data = fs::read(model_path)
        .map_err(|err| format!("failed to read {}: {err}", model_path.display()))?;

    let mut psp_model = parse::tflite::to_psp_ir(data, false)
        .map_err(|err| format!("error lowering to IR: {err}"))?;

    let generated = codegen::generate_code_named(
        &mut psp_model,
        stream_batch,
        None,
        None,
        &format!("{name}_weights.bin"),
    )
    .map_err(|err| format!("codegen error: {err}"))?;

    write_generated(generated, out_dir, &format!("{name}.rs"))
}

/// Compile a hand-built graph (see [`PspModelBuilder`]) into `<name>.rs` +
/// `<name>_weights.bin` in `out_dir`. The graph is taken as-is — no TFLite
/// pipeline passes run; shapes are the builder's declarations.
pub fn compile_graph(
    model: &mut PspModel,
    out_dir: &Path,
    name: &str,
) -> Result<ModelStats, String> {
    let generated =
        codegen::generate_code_named(model, None, None, None, &format!("{name}_weights.bin"))
            .map_err(|err| format!("codegen error: {err}"))?;

    write_generated(generated, out_dir, &format!("{name}.rs"))
}

fn write_generated(
    generated: codegen::Generated,
    out_dir: &Path,
    rs_name: &str,
) -> Result<ModelStats, String> {
    let weights_path = out_dir.join(&generated.data_path);
    fs::write(&weights_path, &generated.data_bytes)
        .map_err(|err| format!("failed to write {}: {err}", weights_path.display()))?;

    let syntax_tree = syn::parse2::<syn::File>(generated.tokens)
        .map_err(|err| format!("failed to parse generated code: {err}"))?;
    let formatted = prettyplease::unparse(&syntax_tree);

    let generated_path = out_dir.join(rs_name);
    fs::write(&generated_path, formatted)
        .map_err(|err| format!("failed to write {}: {err}", generated_path.display()))?;

    Ok(generated.stats)
}
