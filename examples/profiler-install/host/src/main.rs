//! Host runner for the one-time profiler-plugin install.
//!
//! Mounts `kernel-plugin/` as `host0:` so the device side can read
//! `psp_ml_kernel.prx` straight off the build directory, and the PRX directory
//! as `host1:` as usual.

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

    let plugin_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../kernel-plugin");
    let plugin = plugin_dir.join("psp_ml_kernel.prx");
    if !plugin.exists() {
        eprintln!("error: {} not found — run `make -C kernel-plugin` first", plugin.display());
        std::process::exit(1);
    }

    eprintln!("==> Connecting (host0:{}, host1:{})...", plugin_dir.display(), prx_dir.display());
    let conn = PSPConnection::connect(&plugin_dir, prx_dir, Default::default()).unwrap_or_else(|e| {
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

    conn.disconnect();

    match outcome {
        LoadOutcome::Success => {
            eprintln!("==> Done");
            eprintln!();
            eprintln!("Power-cycle the PSP now — kernel plugins are only loaded at boot.");
        }
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
}
