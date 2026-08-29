//! Installs the standalone PSPBird app onto a USB-connected PSP's Memory
//! Stick. Builds the app first (`cargo build -p pspbird-host --release`, so
//! `pspbird.EBOOT.PBP`, `weights.bin` and `blobs/` are staged in the
//! device build dir), stages the species-image pack into `blobs/` if the
//! photos have been fetched, mounts that dir as `host0:`, and runs the
//! device-side copier. Afterwards the app is at ms0:/PSP/GAME/PSPBIRD/ and launches
//! from the XMB with no host attached.

use psplink_connection::{LoadOutcome, PSPConnection};
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().expect("PRX_PATH has no parent directory");
    let prx_name = prx_path.file_name().unwrap().to_str().unwrap();

    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    eprintln!("==> Building the app (cargo build -p pspbird-host --release)");
    let status = Command::new("cargo")
        .args(["build", "-p", "pspbird-host", "--release"])
        .current_dir(&repo)
        .status()
        .expect("failed to run cargo");
    assert!(status.success(), "pspbird build failed");

    // The species-image pack is built by pspbird-host at run time (it
    // depends on the gitignored images/ manifest, not on the build), so
    // run its stage-only mode: writes or removes blobs/birds.img beside
    // the PRX without touching the PSP.
    eprintln!("==> Staging species images (pspbird-host --stage-images)");
    let status = Command::new("cargo")
        .args(["run", "-p", "pspbird-host", "--release", "--", "--stage-images"])
        .current_dir(&repo)
        .status()
        .expect("failed to run cargo");
    assert!(status.success(), "pspbird image staging failed");

    let app_dir = repo.join("examples/birdnet/device/target/mipsel-sony-psp/release");
    for f in ["pspbird.EBOOT.PBP", "weights.bin", "blobs"] {
        assert!(app_dir.join(f).exists(), "{} missing in {}", f, app_dir.display());
    }

    eprintln!("==> Connecting (host0:{}, host1:{})...", app_dir.display(), prx_dir.display());
    let conn = PSPConnection::connect(&app_dir, prx_dir, Default::default()).unwrap_or_else(|e| {
        eprintln!("error: failed to connect to PSP: {e}");
        std::process::exit(1);
    });

    eprintln!("==> Loading host1:{prx_name}");
    let outcome = conn
        .load_program(&format!("host1:{prx_name}"), |bytes| {
            std::io::stdout().write_all(bytes).ok();
            std::io::stdout().flush().ok();
        })
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    conn.disconnect();
    match outcome {
        LoadOutcome::Success => eprintln!("==> Done"),
        other => {
            eprintln!("==> Install failed: {other:?}");
            std::process::exit(1);
        }
    }
}
