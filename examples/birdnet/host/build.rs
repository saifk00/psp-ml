//! Builds the sibling `device/` PSP crate into a `.prx` via the plain,
//! unmodified `cargo psp`, then hands the resulting path to `main()` via
//! `PRX_PATH`. Isolated `--target-dir` avoids lock contention with the
//! outer `cargo build`'s own `target/`.

use std::path::Path;
use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let device_dir = Path::new(&manifest_dir).join("../device");
    let target_dir = device_dir.join("target");

    let profile = std::env::var("PROFILE").unwrap(); // "debug" or "release"

    let mut cmd = Command::new("cargo");
    cmd.arg("+nightly").arg("psp");
    // The device crate defaults to its `local` (host) feature so a bare
    // `cargo test` can build it; the real device build must opt out.
    cmd.arg("--no-default-features");
    if profile == "release" {
        cmd.arg("--release");
    }
    let profiling = std::env::var("PSP_PROFILE").is_ok_and(|v| v != "0");
    if profiling {
        cmd.arg("--features").arg("hwprofile");
    }
    // BIRDNET_CUSTOM_FRONTEND=1: sever the model at the branch concat and run
    // the custom-op frontend (StridedViewStft + banded mel) ahead of the
    // backbone. See examples/birdnet/device's `custom-frontend` feature.
    let custom_frontend = std::env::var("BIRDNET_CUSTOM_FRONTEND").is_ok_and(|v| v != "0");
    if custom_frontend {
        cmd.arg("--features").arg("custom-frontend");
    }
    println!("cargo:rerun-if-env-changed=BIRDNET_CUSTOM_FRONTEND");
    // BIRDNET_SMALL_FFT composes with the custom frontend (FFT pruning pass).
    if let Ok(val) = std::env::var("BIRDNET_SMALL_FFT") {
        cmd.env("BIRDNET_SMALL_FFT", val);
    }
    println!("cargo:rerun-if-env-changed=BIRDNET_SMALL_FFT");

    // Forward the species-pruning knobs to the device build, which is what
    // actually invokes prune_classifier.py. Cargo would pass these through
    // anyway, but being explicit keeps the contract visible from the host side.
    for var in ["TOPK", "BIRDNET_REGION", "BIRDNET_BBOX", "BIRDNET_PYTHON"] {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
        println!("cargo:rerun-if-env-changed={var}");
    }

    cmd.arg("--target-dir").arg(&target_dir);
    cmd.current_dir(&device_dir);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "failed to run `cargo psp` ({e}). Is cargo-psp installed? `cargo install cargo-psp`"
        )
    });
    assert!(status.success(), "cargo psp failed to build the device crate");

    let prx_path = target_dir.join(format!("mipsel-sony-psp/{profile}/birdnet.prx"));
    assert!(
        prx_path.exists(),
        "expected a prx at {} but it wasn't produced",
        prx_path.display()
    );

    println!("cargo:rustc-env=PRX_PATH={}", prx_path.display());
    println!("cargo:rerun-if-env-changed=PSP_PROFILE");
    println!("cargo:rerun-if-changed=../device/src");
    println!("cargo:rerun-if-changed=../device/Cargo.toml");
    // Without this, editing the device build script does not re-run `cargo
    // psp`, so the staged weights.bin and the .prx can drift apart.
    println!("cargo:rerun-if-changed=../device/build.rs");
    println!("cargo:rerun-if-changed=../prune_classifier.py");
}
