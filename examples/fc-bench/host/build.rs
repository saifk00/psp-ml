//! Builds the sibling `device/` PSP crate into a `.prx` via the plain,
//! unmodified `cargo psp` (community `cargo-psp` tool — no fork, nothing
//! special here), then hands the resulting path to `main()` via
//! `PRX_PATH`. Uses an isolated `--target-dir` (under `device/`, not
//! shared with this crate's own `target/`) so this nested `cargo psp`
//! invocation never contends for the build lock the outer `cargo build`
//! already holds.

use std::path::Path;
use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let device_dir = Path::new(&manifest_dir).join("../device");
    let target_dir = device_dir.join("target");

    let profile = std::env::var("PROFILE").unwrap(); // "debug" or "release"

    let mut cmd = Command::new("cargo");
    cmd.arg("+nightly").arg("psp");
    if profile == "release" {
        cmd.arg("--release");
    }
    cmd.arg("--target-dir").arg(&target_dir);
    cmd.current_dir(&device_dir);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "failed to run `cargo psp` ({e}). Is cargo-psp installed? `cargo install cargo-psp`"
        )
    });
    assert!(status.success(), "cargo psp failed to build the device crate");

    let prx_path = target_dir.join(format!("mipsel-sony-psp/{profile}/fc_bench.prx"));
    assert!(
        prx_path.exists(),
        "expected a prx at {} but it wasn't produced",
        prx_path.display()
    );

    println!("cargo:rustc-env=PRX_PATH={}", prx_path.display());
    println!("cargo:rerun-if-changed=../device/src");
    println!("cargo:rerun-if-changed=../device/Cargo.toml");
}
