use std::path::PathBuf;
use std::process;
use std::time::Duration;

use psp_runner::{ExitReason, PspRunner};

fn main() {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let mut prx_path: Option<String> = None;
    let mut root_dir = PathBuf::from(".");
    let mut timeout_secs: u64 = 60;
    let mut wait_for: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                root_dir = PathBuf::from(args.next().unwrap_or_else(|| {
                    eprintln!("--root requires a directory path");
                    process::exit(1);
                }));
            }
            "--timeout" => {
                timeout_secs = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--timeout requires a number of seconds");
                        process::exit(1);
                    });
            }
            "--wait-for" => {
                wait_for = Some(args.next().unwrap_or_else(|| {
                    eprintln!("--wait-for requires a filename");
                    process::exit(1);
                }));
            }
            "--help" | "-h" => {
                eprintln!("Usage: psp-runner [OPTIONS] <PRX_PATH>");
                eprintln!();
                eprintln!("Execute a PSP PRX module via psplink USB protocol.");
                eprintln!("Replaces usbhostfs_pc + pspsh with a single command.");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --root <DIR>       HostFS root directory (default: .)");
                eprintln!("  --timeout <SECS>   Timeout in seconds (default: 60)");
                eprintln!("  --wait-for <FILE>  Exit when this file is written via host0:/");
                eprintln!();
                eprintln!("Examples:");
                eprintln!("  psp-runner target/mipsel-sony-psp/release/mnist-bench.prx \\");
                eprintln!("    --wait-for benchmarks.json");
                eprintln!();
                eprintln!("  psp-runner target/mipsel-sony-psp/release/test-kernels.prx \\");
                eprintln!("    --wait-for test-results.json --timeout 30");
                eprintln!();
                eprintln!("Environment:");
                eprintln!("  RUST_LOG=info    Show protocol activity");
                eprintln!("  RUST_LOG=debug   Show all HostFS commands");
                process::exit(0);
            }
            _ => {
                if prx_path.is_none() {
                    prx_path = Some(arg);
                } else {
                    eprintln!("unexpected argument: {arg}");
                    process::exit(1);
                }
            }
        }
    }

    let prx_path = prx_path.unwrap_or_else(|| {
        eprintln!("Usage: psp-runner [OPTIONS] <PRX_PATH>");
        eprintln!("       psp-runner --help for more info");
        process::exit(1);
    });

    eprintln!("==> Connecting to PSP...");
    let runner = PspRunner::connect(root_dir).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        process::exit(1);
    });

    let host0_path = format!("host0:/{prx_path}");
    eprintln!("==> Executing: {host0_path}");

    let result = runner
        .execute(
            &host0_path,
            wait_for.as_deref(),
            Duration::from_secs(timeout_secs),
        )
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            process::exit(1);
        });

    match &result.exit_reason {
        ExitReason::FileReceived(path) => {
            eprintln!("==> Received: {}", path.display());
        }
        ExitReason::Timeout => {
            eprintln!("==> Timed out after {timeout_secs}s");
            if !result.files_written.is_empty() {
                eprintln!("    Files written before timeout:");
                for f in &result.files_written {
                    eprintln!("      {}", f.display());
                }
            }
            process::exit(1);
        }
        ExitReason::Disconnected(msg) => {
            eprintln!("==> PSP disconnected: {msg}");
            process::exit(1);
        }
    }
}
