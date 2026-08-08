//! Host runner for `mnist-bench`. `build.rs` cross-compiled the sibling
//! `device/` crate into a `.prx` and handed us its path via `PRX_PATH`.
//! `host0:` is mounted to the mnist-bench/ directory (one level up from
//! here) so the device's `host0:/benchmarks.json` write lands next to
//! `mnist_cnn.tflite`, matching where the old `run.sh` left it.

use psplink_connection::{LoadOutcome, PSPConnection};
use std::io::Write;
use std::path::Path;

fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().expect("PRX_PATH has no parent directory");
    let prx_name = prx_path
        .file_name()
        .expect("PRX_PATH has no file name")
        .to_str()
        .expect("PRX_PATH is not valid UTF-8");

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let bench_dir = Path::new(manifest_dir).join("..");
    let bench_json = bench_dir.join("benchmarks.json");
    let _ = std::fs::remove_file(&bench_json); // drop any stale result from a previous run

    eprintln!(
        "==> Connecting (host0:{}, host1:{})...",
        bench_dir.display(),
        prx_dir.display()
    );
    let conn = PSPConnection::connect(&bench_dir, prx_dir, Default::default()).unwrap_or_else(|e| {
        eprintln!("error: failed to connect to PSP: {e}");
        std::process::exit(1);
    });

    eprintln!("==> Loading host1:{prx_name}");
    let outcome = conn
        .load_program(&format!("host1:{prx_name}"), |bytes| {
            std::io::stdout().write_all(bytes).ok();
        })
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });

    match outcome {
        LoadOutcome::Success => eprintln!("==> Done"),
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

    match std::fs::read_to_string(&bench_json) {
        Ok(contents) => {
            eprintln!("\n==> {}:", bench_json.display());
            println!("{contents}");
        }
        Err(e) => eprintln!("==> Warning: could not read {}: {e}", bench_json.display()),
    }
}
