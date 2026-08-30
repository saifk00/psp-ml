//! Builds the `pspbird` binary of the sibling `device/` crate into a `.prx`
//! via plain `cargo psp`, same shape as birdnet/host/build.rs, and hands the
//! path to `main()` via `PRX_PATH`. Shares `device/target` with the
//! benchmark runner so the two builds reuse each other's artifacts.

use std::path::Path;
use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let device_dir = Path::new(&manifest_dir).join("../device");
    let target_dir = device_dir.join("target");
    let profile = std::env::var("PROFILE").unwrap();

    let mut cmd = Command::new("cargo");
    cmd.arg("+nightly").arg("psp");
    cmd.arg("--no-default-features");
    cmd.arg("--features").arg("app");
    cmd.arg("--bin").arg("pspbird");
    if profile == "release" {
        cmd.arg("--release");
    }
    // TOPK fixes the classifier width for every region; BIRDNET_MODEL
    // defaults to the FP32 build inside device/build.rs.
    for var in ["TOPK", "BIRDNET_PYTHON", "BIRDNET_MODEL"] {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
        println!("cargo:rerun-if-env-changed={var}");
    }
    cmd.arg("--target-dir").arg(&target_dir);
    cmd.current_dir(&device_dir);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!("failed to run `cargo psp` ({e}). Is cargo-psp installed? `cargo install cargo-psp`")
    });
    assert!(status.success(), "cargo psp failed to build pspbird");

    let prx_path = target_dir.join(format!("mipsel-sony-psp/{profile}/pspbird.prx"));
    assert!(prx_path.exists(), "expected a prx at {} but it wasn't produced", prx_path.display());

    println!("cargo:rustc-env=PRX_PATH={}", prx_path.display());
    println!("cargo:rerun-if-changed=../device/src");
    println!("cargo:rerun-if-changed=../device/Cargo.toml");
    println!("cargo:rerun-if-changed=../device/build.rs");
    // XMB presentation: pack-pbp bakes these into the EBOOT.
    println!("cargo:rerun-if-changed=../device/Psp.toml");
    println!("cargo:rerun-if-changed=../device/icon0.png");
    println!("cargo:rerun-if-changed=../device/pic1.png");
    println!("cargo:rerun-if-changed=../prune_classifier.py");
    // The nested cargo tracks these itself, but this build script must
    // re-run for that to happen at all.
    println!("cargo:rerun-if-changed=../../../toolchain/psp-tc/src");
    println!("cargo:rerun-if-changed=../../../toolchain/psp-rt/src");
}
