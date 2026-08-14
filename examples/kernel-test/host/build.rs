//! Cross-compiles `psp-rt`'s `test-kernels` binary for the PSP.
//!
//! Unlike the other examples there is no sibling `device/` crate: the tests
//! live in `psp-rt` itself, because they are the same `kernels::checks` list
//! that `cargo test` runs against the scalar fallbacks. This build produces
//! the variant that runs them against the real VFPU assembly.

use std::path::Path;
use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let psp_rt_dir = Path::new(&manifest_dir).join("../../../psp-rt");
    let target_dir = psp_rt_dir.join("target");

    let profile = std::env::var("PROFILE").unwrap();

    let mut cmd = Command::new("cargo");
    cmd.arg("+nightly").arg("psp");
    // psp-rt's ML device crates default to `local`; this build needs the real
    // no_std device target, and `test-kernels` gates the binary itself.
    cmd.arg("--no-default-features")
        .arg("--features")
        .arg("test-kernels")
        .arg("--bin")
        .arg("test-kernels");
    if profile == "release" {
        cmd.arg("--release");
    }
    cmd.arg("--target-dir").arg(&target_dir);
    cmd.current_dir(&psp_rt_dir);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!("failed to run `cargo psp` ({e}). Is cargo-psp installed? `cargo install cargo-psp`")
    });
    assert!(status.success(), "cargo psp failed to build test-kernels");

    let prx_path = target_dir.join(format!("mipsel-sony-psp/{profile}/test-kernels.prx"));
    assert!(
        prx_path.exists(),
        "expected a prx at {} but it wasn't produced",
        prx_path.display()
    );

    println!("cargo:rustc-env=PRX_PATH={}", prx_path.display());
    println!("cargo:rerun-if-changed=../../../psp-rt/src");
    println!("cargo:rerun-if-changed=../../../psp-rt/Cargo.toml");
}
