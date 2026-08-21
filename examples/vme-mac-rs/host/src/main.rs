//! Host runner for the pure-Rust VME MAC. Mounts the example dir as `host0:`
//! (for any result file) and the PRX dir as `host1:`, loads the PRX, and
//! streams stdout.

use psplink_connection::{LoadOutcome, PSPConnection};
use std::io::Write;
use std::path::Path;

fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().expect("PRX_PATH has no parent directory");
    let prx_name = prx_path.file_name().unwrap().to_str().unwrap();

    let example_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");

    eprintln!("==> Connecting...");
    let conn = PSPConnection::connect(&example_dir, prx_dir, Default::default())
        .unwrap_or_else(|e| {
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
        LoadOutcome::Success => eprintln!("==> Done"),
        LoadOutcome::TimedOut => {
            eprintln!("==> Timed out");
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
