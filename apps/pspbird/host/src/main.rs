//! Host runner for `pspbird`. Publishes the classifier blobs staged beside
//! the PRX into `host0:/blobs/` (the device installs them to ms0 on first
//! use), deploys the app PRX and streams its log until the user exits it on
//! the device (HOME). `host0:` is the birdnet example root.

use birdnet::imfile;
use psplink_connection::{LoadOutcome, PSPConnection};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Assemble a standalone install: `EBOOT.PBP`, `weights.bin` and `blobs/`
/// (classifier blobs plus `birds.img` when species images were fetched)
/// in `dest`, laid out the way the device code resolves them relative to
/// the launch directory. Copy `dest` to `ms0:/PSP/GAME/PSPBIRD/`.
fn pack(prx_dir: &Path, dest: &Path) {
    let eboot = prx_dir.join("pspbird.EBOOT.PBP");
    if !eboot.exists() {
        eprintln!("error: {} missing — cargo-psp did not produce an EBOOT", eboot.display());
        std::process::exit(1);
    }
    std::fs::create_dir_all(dest.join("blobs")).expect("failed to create the pack dir");
    std::fs::copy(&eboot, dest.join("EBOOT.PBP")).unwrap();
    std::fs::copy(prx_dir.join("weights.bin"), dest.join("weights.bin")).unwrap();
    let mut total = std::fs::metadata(dest.join("weights.bin")).unwrap().len();
    for entry in std::fs::read_dir(prx_dir.join("blobs")).expect("no staged blobs") {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dest.join("blobs").join(entry.file_name())).unwrap();
        total += entry.metadata().unwrap().len();
    }
    eprintln!(
        "==> Packed {} ({:.1} MB); copy it to ms0:/PSP/GAME/PSPBIRD/",
        dest.display(),
        total as f64 / 1e6
    );
}

/// Must be at least the device side's `timeout_secs` in `psp_rt::module!`,
/// or the host gives up on a user who is just sitting on a menu.
const IDLE_TIMEOUT: Duration = Duration::from_secs(3600);

/// Species thumbnails: 3 rows of 2 beside the results text, so 80 px is
/// the largest square that fits three high on the 272-line screen.
const IMAGE_SIZE: u32 = 80;
const IMAGE_PACK: &str = "birds.img";

/// Build `blobs/birds.img` beside the PRX from the photos
/// `examples/birdnet/fetch_images.py` fetched, with one class -> image map
/// per staged blob (keyed by the blob's file stem, the region name the
/// device selects by). Skipped with a warning when no manifest exists:
/// the app runs without pictures. Lives in the staged `blobs/` so both
/// the host0: publish and `--pack` carry it along.
fn pack_images(prx_dir: &Path) {
    let staged = prx_dir.join("blobs");
    let manifest = std::env::var_os("IMVIEW_MANIFEST")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../images/manifest.toml"));
    if !manifest.exists() {
        eprintln!(
            "==> No species images ({} missing): run `uv run apps/pspbird/fetch_images.py <labels.txt>`",
            manifest.display()
        );
        let _ = std::fs::remove_file(staged.join(IMAGE_PACK));
        return;
    }
    let mut regions = Vec::new();
    for entry in std::fs::read_dir(&staged).expect("no staged blobs beside the prx") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        let name = path.file_stem().unwrap().to_str().unwrap().to_string();
        let labels = imfile::pbrd_labels(&path).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
        regions.push((name, labels));
    }
    regions.sort_by(|a, b| a.0.cmp(&b.0));
    let t = std::time::Instant::now();
    let packed = imfile::pack_images(&manifest, IMAGE_SIZE, IMAGE_SIZE, &regions).unwrap_or_else(|e| {
        eprintln!("error: packing species images: {e}");
        std::process::exit(1);
    });
    std::fs::write(staged.join(IMAGE_PACK), &packed.bytes).expect("failed to write birds.img");
    eprintln!(
        "==> Packed {} species images at {}x{} into blobs/{IMAGE_PACK} ({} bytes, {:.1?})",
        packed.labels.len(),
        packed.width,
        packed.height,
        packed.bytes.len(),
        t.elapsed()
    );
    for (label, why) in &packed.fallbacks {
        eprintln!("    placeholder for {label}: {why}");
    }
    for (region, misses) in &packed.unmapped {
        let n = regions.iter().find(|r| &r.0 == region).map(|r| r.1.len()).unwrap_or(0);
        eprintln!("    {region}: {}/{n} classes have a picture", n - misses);
    }
}

fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().expect("PRX_PATH has no parent directory");
    pack_images(prx_dir);

    let args: Vec<String> = std::env::args().collect();
    // Stage only: pspbird-install runs this so the image pack beside the
    // PRX is current before it copies `blobs/` to the Memory Stick.
    if args.iter().any(|a| a == "--stage-images") {
        return;
    }
    if let Some(i) = args.iter().position(|a| a == "--pack") {
        let dest = args.get(i + 1).unwrap_or_else(|| {
            eprintln!("usage: pspbird-host --pack <dir>   (e.g. /media/ms0/PSP/GAME/PSPBIRD)");
            std::process::exit(2);
        });
        pack(prx_dir, Path::new(dest));
        return;
    }
    let prx_name = prx_path.file_name().unwrap().to_str().unwrap();
    let mount_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");

    // The backbone's weights.bin: too large to embed, so the device reads
    // it from host0:/weights.bin (see birdnet-host for why it is copied at
    // deploy time and from beside the .prx).
    let staged_weights = prx_dir.join("weights.bin");
    std::fs::copy(&staged_weights, mount_dir.join("weights.bin")).unwrap_or_else(|e| {
        eprintln!("error: failed to publish weights.bin to the host0: mount: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "==> Published weights.bin ({} bytes)",
        std::fs::metadata(&staged_weights).map(|m| m.len()).unwrap_or(0)
    );

    // Blobs and .prx come from the same build directory, so they agree on
    // the classifier width.
    let staged = prx_dir.join("blobs");
    let published = mount_dir.join("blobs");
    let _ = std::fs::remove_dir_all(&published);
    std::fs::create_dir_all(&published).expect("failed to create the blobs mount dir");
    let mut n = 0;
    for entry in std::fs::read_dir(&staged).expect("no staged blobs beside the prx") {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), published.join(entry.file_name())).unwrap();
        n += 1;
    }
    eprintln!("==> Published {n} classifier blobs to {}", published.display());

    eprintln!("==> Connecting (host0:{}, host1:{})...", mount_dir.display(), prx_dir.display());
    let conn = PSPConnection::connect(&mount_dir, prx_dir, Default::default()).unwrap_or_else(|e| {
        eprintln!("error: failed to connect to PSP: {e}");
        std::process::exit(1);
    });

    eprintln!("==> Loading host1:{prx_name} (exit with HOME on the device)");
    let outcome = conn
        .load_program_with_idle_timeout(&format!("host1:{prx_name}"), IDLE_TIMEOUT, |bytes| {
            std::io::stdout().write_all(bytes).ok();
            std::io::stdout().flush().ok();
        })
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });

    match outcome {
        LoadOutcome::Success => eprintln!("==> Done"),
        LoadOutcome::TimedOut => {
            eprintln!("==> Timed out: app_main exceeded its budget and was terminated");
            std::process::exit(1);
        }
        LoadOutcome::Panicked => {
            eprintln!("==> Program panicked");
            std::process::exit(1);
        }
        LoadOutcome::ShellError(v) => {
            eprintln!("==> Shell error loading PRX: 0x{v:08X}");
            std::process::exit(1);
        }
        LoadOutcome::KernelError(v) => {
            eprintln!("==> Kernel error: 0x{v:08X}");
            std::process::exit(1);
        }
    }
    conn.disconnect();
}
