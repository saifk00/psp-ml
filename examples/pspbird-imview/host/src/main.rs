//! Host runner for `pspbird-imview`: packs the species photos fetched by
//! `examples/birdnet/fetch_images.py` into a `PBIM` file next to the PRX,
//! mounts that directory as `host0:`, and runs the viewer.
//!
//! ```text
//! uv run examples/birdnet/fetch_images.py <labels.txt>   # once
//! cargo run -p pspbird-imview-host --release
//! ```
//!
//! `IMVIEW_MANIFEST` overrides the manifest path, `IMVIEW_SIZE` the
//! (square) image size in pixels (default 96).

use birdnet::imfile;
use psplink_connection::{LoadOutcome, PSPConnection};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

const PACK_NAME: &str = "birds.img";
/// Interactive: matches the device's IDLE_QUIT_US.
const IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60 + 30);

fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().expect("PRX_PATH has no parent directory");
    let prx_name = prx_path.file_name().unwrap().to_str().unwrap();

    let manifest = std::env::var_os("IMVIEW_MANIFEST").map(PathBuf::from).unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../birdnet/images/manifest.toml")
    });
    let size: u32 = std::env::var("IMVIEW_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(96);

    eprintln!("==> Packing {} at {size}x{size}", manifest.display());
    let t = std::time::Instant::now();
    // No region maps: the viewer picks by image index, not class.
    let packed = imfile::pack_images(&manifest, size, size, &[]).unwrap_or_else(|e| {
        eprintln!("error: {e}\n(run `uv run examples/birdnet/fetch_images.py <labels.txt>` first)");
        std::process::exit(1);
    });
    for (label, why) in &packed.fallbacks {
        eprintln!("    placeholder for {label}: {why}");
    }
    for (i, l) in packed.labels.iter().enumerate() {
        eprintln!("    {i:3}  {l}");
    }
    let pack_path = prx_dir.join(PACK_NAME);
    std::fs::write(&pack_path, &packed.bytes).expect("write pack");
    eprintln!(
        "    {} images, {} bytes -> {} ({:.1?})",
        packed.labels.len(),
        packed.bytes.len(),
        pack_path.display(),
        t.elapsed()
    );

    eprintln!("==> Connecting (host0:{})...", prx_dir.display());
    let conn = PSPConnection::connect(prx_dir, prx_dir, Default::default()).unwrap_or_else(|e| {
        eprintln!("error: failed to connect to PSP: {e}");
        std::process::exit(1);
    });

    eprintln!("==> Loading host1:{prx_name}  (UP/DOWN pick, X load, SELECT quit)");
    let outcome = conn
        .load_program_with_idle_timeout(&format!("host1:{prx_name}"), IDLE_TIMEOUT, |bytes| {
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
            eprintln!("==> Failed: {other:?}");
            std::process::exit(1);
        }
    }
}
