//! Builds the sibling `device/` crate TWICE — once per mode — so each mode's
//! memory footprint is measurable on its own (same reasoning as the STFT
//! benchmark's host build). Both PRXes are staged into one deploy dir under
//! distinct names.

use std::path::Path;
use std::process::Command;

const MODES: [(&str, &str); 2] = [("mode-dense", "mel_dense"), ("mode-cb", "mel_cb")];

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let device_dir = Path::new(&manifest_dir).join("../device");
    let profile = std::env::var("PROFILE").unwrap();

    let deploy_dir = device_dir.join("target/deploy");
    std::fs::create_dir_all(&deploy_dir).unwrap();

    for (feature, staged_name) in MODES {
        let target_dir = device_dir.join("target").join(feature);

        let mut cmd = Command::new("cargo");
        cmd.arg("+nightly").arg("psp");
        cmd.arg("--no-default-features");
        cmd.arg("--features").arg(feature);
        if profile == "release" {
            cmd.arg("--release");
        }
        cmd.arg("--target-dir").arg(&target_dir);
        cmd.current_dir(&device_dir);
        for var in ["BIRDNET_PYTHON"] {
            if let Ok(val) = std::env::var(var) {
                cmd.env(var, val);
            }
            println!("cargo:rerun-if-env-changed={var}");
        }

        let status = cmd.status().unwrap_or_else(|e| {
            panic!("failed to run `cargo psp` ({e}). Is cargo-psp installed? `cargo install cargo-psp`")
        });
        assert!(
            status.success(),
            "cargo psp failed to build the device crate ({feature})"
        );

        let prx = target_dir.join(format!(
            "mipsel-sony-psp/{profile}/birdnet-mel-benchmark.prx"
        ));
        assert!(
            prx.exists(),
            "expected a prx at {} but it wasn't produced",
            prx.display()
        );
        std::fs::copy(&prx, deploy_dir.join(format!("{staged_name}.prx"))).unwrap();
    }

    println!("cargo:rustc-env=PRX_DEPLOY_DIR={}", deploy_dir.display());
    println!("cargo:rerun-if-changed=../device/src");
    println!("cargo:rerun-if-changed=../device/build.rs");
    println!("cargo:rerun-if-changed=../device/Cargo.toml");
}
