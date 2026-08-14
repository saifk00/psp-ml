//! Runs psp-rt's kernel checks on real hardware, against the VFPU assembly.
//!
//! `cargo test` runs the same `kernels::checks` list on the host, but there it
//! compiles to the scalar fallbacks — every `#[cfg(target_os = "psp")]` VFPU
//! block is excluded. So the host suite validates the *algorithms* and this
//! validates the *assembly*, which is where the bugs have actually been.
//!
//! Exits nonzero if any check fails, so it can gate a release the way
//! `cargo test` does.

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

    // The device writes host0:/test-results.json; mount this crate's directory
    // to receive it.
    let mount_dir = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    let results_path = mount_dir.join("test-results.json");
    // Never report a stale pass: if the run dies before writing, the absence of
    // the file has to be visible.
    let _ = std::fs::remove_file(&results_path);

    eprintln!(
        "==> Connecting (host0:{}, host1:{})...",
        mount_dir.display(),
        prx_dir.display()
    );
    let conn = PSPConnection::connect(&mount_dir, prx_dir, Default::default()).unwrap_or_else(|e| {
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
        LoadOutcome::Success => {}
        LoadOutcome::TimedOut => {
            eprintln!("==> Timed out: a kernel check hung and the worker was terminated");
            std::process::exit(1);
        }
        LoadOutcome::Panicked => {
            eprintln!("==> Panicked on device — see the log above");
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

    // The device always exits cleanly, pass or fail — the verdict is in the
    // JSON, so a missing file is itself a failure.
    let json = std::fs::read_to_string(&results_path).unwrap_or_else(|e| {
        eprintln!(
            "==> No {} ({e}). The run did not reach the end.",
            results_path.display()
        );
        std::process::exit(1);
    });

    let (passed, failed) = (field(&json, "passed"), field(&json, "failed"));
    println!("\n==> Device kernel checks: {passed} passed, {failed} failed");
    if failed != 0 {
        for name in failing_names(&json) {
            println!("    FAILED: {name}");
        }
        std::process::exit(1);
    }
    if passed == 0 {
        println!("    but nothing ran — the check list is empty?");
        std::process::exit(1);
    }
}

/// Pull a top-level integer out of the device's JSON.
///
/// Hand-rolled to keep this crate dependency-free; the device writes the file
/// with a fixed writer, so the shape is known.
fn field(json: &str, key: &str) -> u32 {
    json.split(&format!("\"{key}\":"))
        .nth(1)
        .and_then(|s| s.trim_start().split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            eprintln!("==> malformed test-results.json: no \"{key}\"");
            std::process::exit(1);
        })
}

fn failing_names(json: &str) -> Vec<&str> {
    json.lines()
        .filter(|l| l.contains("\"passed\": false"))
        .filter_map(|l| l.split("\"name\": \"").nth(1))
        .filter_map(|l| l.split('"').next())
        .collect()
}
