//! Cross-compiles `psp-rt`'s `device-tests` binary for the PSP.
//!
//! Unlike the examples there is no sibling `device/` crate: the checks live in
//! `psp-rt` itself, because they are the same `device_test::SUITES` list that
//! `cargo test` runs against the scalar fallbacks. This build produces the
//! variant that runs them against the real VFPU assembly.

use std::path::Path;
use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let psp_rt_dir = Path::new(&manifest_dir).join("../psp-rt");
    let target_dir = psp_rt_dir.join("target");

    // Always release, regardless of the host test profile. `cargo test` sets
    // PROFILE=debug, and an unoptimised VFPU build is slow enough to risk
    // `module!`'s 240 s watchdog — a check would "fail" by timing out.
    let debug = std::env::var("PSP_TEST_DEBUG").is_ok_and(|v| v != "0");
    println!("cargo:rerun-if-env-changed=PSP_TEST_DEBUG");
    let profile = if debug { "debug" } else { "release" };

    let mut cmd = Command::new("cargo");
    cmd.arg("+nightly").arg("psp");
    // psp-rt's default feature set is empty, but the ML device crates in the
    // workspace default to `local`; passing this keeps the invocation
    // identical to every other device build in the tree.
    cmd.arg("--no-default-features")
        .arg("--features")
        .arg("device-tests")
        .arg("--bin")
        .arg("device-tests");
    if !debug {
        cmd.arg("--release");
    }
    // A separate target dir under psp-rt/, so this nested cargo doesn't
    // contend with the outer one's build lock.
    cmd.arg("--target-dir").arg(&target_dir);
    cmd.current_dir(&psp_rt_dir);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!("failed to run `cargo psp` ({e}). Is cargo-psp installed? `cargo install cargo-psp`")
    });
    assert!(status.success(), "cargo psp failed to build device-tests");

    let prx_path = target_dir.join(format!("mipsel-sony-psp/{profile}/device-tests.prx"));
    assert!(
        prx_path.exists(),
        "expected a prx at {} but it wasn't produced",
        prx_path.display()
    );

    println!("cargo:rustc-env=PRX_PATH={}", prx_path.display());
    println!("cargo:rerun-if-changed=../psp-rt/src");
    println!("cargo:rerun-if-changed=../psp-rt/Cargo.toml");
}
