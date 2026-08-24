//! Builds the sibling `device/` crate TWICE — once per mode — because each
//! mode's memory footprint must be measurable on its own: a PRX containing
//! both frontends would carry both arenas and both weight blobs, and the
//! on-device free-memory numbers could not attribute anything.
//!
//! Separate --target-dir per mode (a shared one would thrash the feature
//! unification cache on every alternation); both PRXes are then staged into
//! one deploy dir under distinct names so a single `host1:` mount serves
//! whichever the runner loads.

use std::path::Path;
use std::process::Command;

const MODES: [(&str, &str); 2] = [
    ("mode-dense", "stft_dense"),
    ("mode-strided", "stft_strided"),
];

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
        // The device crate defaults to its `local` (host) feature so a bare
        // `cargo test` can build it; the real device build must opt out.
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
            "mipsel-sony-psp/{profile}/birdnet-stft-benchmark.prx"
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
    println!("cargo:rerun-if-changed=../slice_stft.py");
}
