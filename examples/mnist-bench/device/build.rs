//! Generates `generated.rs` + `weights.bin` from `mnist_cnn.tflite` into
//! `OUT_DIR`. Runs unconditionally (not gated on the psp target) since the
//! `local` host-testing feature also needs the same generated inference
//! code, just linked against psp-rt's portable (non-VFPU) kernel fallback.

use std::env;
use std::path::Path;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let model = Path::new(&manifest_dir).join("../mnist_cnn.tflite");

    psp_tc::compile_tflite(&model, Path::new(&out_dir), None).unwrap_or_else(|e| {
        panic!("psp-tc codegen failed: {e}");
    });

    println!("cargo:rerun-if-changed={}", model.display());
}
