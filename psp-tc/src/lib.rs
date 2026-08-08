//! TFLite model compiler — generates Rust code targeting the `psp-rt` runtime library.

pub mod codegen;
pub mod ir;
pub mod memory_planner;
pub mod parse;

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
) -> Result<(), String> {
    let data = fs::read(model_path)
        .map_err(|err| format!("failed to read {}: {err}", model_path.display()))?;

    let mut psp_model = parse::tflite::to_psp_ir(data, false)
        .map_err(|err| format!("error lowering to IR: {err}"))?;

    let generated = codegen::generate_code(&mut psp_model, stream_batch)
        .map_err(|err| format!("codegen error: {err}"))?;

    let weights_path = out_dir.join("weights.bin");
    fs::write(&weights_path, &generated.data_bytes)
        .map_err(|err| format!("failed to write {}: {err}", weights_path.display()))?;

    let syntax_tree = syn::parse2::<syn::File>(generated.tokens)
        .map_err(|err| format!("failed to parse generated code: {err}"))?;
    let formatted = prettyplease::unparse(&syntax_tree);

    let generated_path = out_dir.join("generated.rs");
    fs::write(&generated_path, formatted)
        .map_err(|err| format!("failed to write {}: {err}", generated_path.display()))?;

    Ok(())
}
