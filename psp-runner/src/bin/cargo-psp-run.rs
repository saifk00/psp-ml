//! Cargo subcommand: `cargo psp-run`
//!
//! Builds a PSP PRX via `cargo psp` and deploys it to a PSP running psplink.
//!
//! Usage:
//!   cargo psp-run -p hello-psp --release
//!   cargo psp-run -p mnist-bench --release

use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};
use std::time::Duration;

use psp_runner::{ExitReason, PspRunner};

fn main() {
    env_logger::init();

    // When invoked as `cargo psp-run`, cargo passes "psp-run" as the first arg.
    let args: Vec<String> = std::env::args().collect();
    let args = if args.get(1).map(|s| s.as_str()) == Some("psp-run") {
        &args[2..]
    } else {
        &args[1..]
    };

    let mut package: Option<String> = None;
    let mut release = false;
    let mut timeout_secs: u64 = 60;
    let mut root_dir = PathBuf::from(".");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-p" | "--package" => {
                i += 1;
                package = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("-p requires a package name");
                    process::exit(1);
                }).clone());
            }
            "--release" => release = true,
            "--timeout" => {
                i += 1;
                timeout_secs = args.get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--timeout requires a number of seconds");
                        process::exit(1);
                    });
            }
            "--root" => {
                i += 1;
                root_dir = PathBuf::from(args.get(i).unwrap_or_else(|| {
                    eprintln!("--root requires a directory path");
                    process::exit(1);
                }));
            }
            "--help" | "-h" => {
                eprintln!("Usage: cargo psp-run [OPTIONS] -p <PACKAGE>");
                eprintln!();
                eprintln!("Build a PSP PRX and deploy it to a PSP running psplink.");
                eprintln!();
                eprintln!("Build options:");
                eprintln!("  -p, --package <PKG>  Package to build and run");
                eprintln!("  --release            Build in release mode");
                eprintln!();
                eprintln!("Runner options:");
                eprintln!("  --timeout <SECS>     Timeout in seconds (default: 60)");
                eprintln!("  --root <DIR>         HostFS root directory (default: workspace root)");
                eprintln!();
                eprintln!("Examples:");
                eprintln!("  cargo psp-run -p hello-psp --release");
                eprintln!("  cargo psp-run -p mnist-bench --release");
                eprintln!();
                eprintln!("Environment:");
                eprintln!("  RUST_LOG=info    Show protocol activity");
                eprintln!("  RUST_LOG=debug   Show all HostFS commands");
                process::exit(0);
            }
            other => {
                eprintln!("unexpected argument: {other}");
                eprintln!("       cargo psp-run --help for usage");
                process::exit(1);
            }
        }
        i += 1;
    }

    // Use `cargo metadata` to get workspace info
    let metadata = cargo_metadata();
    let workspace_root = PathBuf::from(
        metadata["workspace_root"].as_str().unwrap_or_else(|| {
            eprintln!("error: cargo metadata missing workspace_root");
            process::exit(1);
        })
    );
    let target_directory = PathBuf::from(
        metadata["target_directory"].as_str().unwrap_or_else(|| {
            eprintln!("error: cargo metadata missing target_directory");
            process::exit(1);
        })
    );

    // Infer package name from cargo metadata if not specified
    let package = package.unwrap_or_else(|| {
        infer_package_name(&metadata).unwrap_or_else(|| {
            eprintln!("error: could not determine package name");
            eprintln!("       use -p <PACKAGE> or run from a package directory");
            process::exit(1);
        })
    });

    // --- Step 1: Build via `cargo psp` ---
    let mut build_cmd = Command::new("cargo");
    build_cmd.arg("+nightly").arg("psp");
    if release {
        build_cmd.arg("--release");
    }
    build_cmd.arg("-p").arg(&package);

    eprintln!("==> Building: cargo +nightly psp{} -p {package}",
        if release { " --release" } else { "" });

    // Capture output so we can parse the PRX path from cargo-psp's output.
    // cargo-psp prints: [6]  NNNN bytes | /absolute/path/to/foo.prx
    let output = build_cmd.output().unwrap_or_else(|e| {
        eprintln!("error: failed to run `cargo psp`: {e}");
        eprintln!("       is cargo-psp installed? (`cargo install cargo-psp`)");
        process::exit(1);
    });
    // Forward build stderr (warnings, errors, progress) to our stderr
    let build_stderr = String::from_utf8_lossy(&output.stderr);
    if !build_stderr.is_empty() {
        eprint!("{build_stderr}");
    }
    if !output.status.success() {
        eprintln!("error: build failed");
        process::exit(output.status.code().unwrap_or(1));
    }

    // --- Step 2: Find the PRX from build output ---
    let build_stdout = String::from_utf8_lossy(&output.stdout);
    // Also print build stdout (the PBP table) to stderr for visibility
    if !build_stdout.is_empty() {
        eprint!("{build_stdout}");
    }
    let prx_path = build_stdout
        .lines()
        .find_map(|line| {
            // Match lines like: [6]  227964 bytes | /abs/path/to/foo.prx
            let path = line.split('|').nth(1)?.trim();
            if path.ends_with(".prx") { Some(path.to_string()) } else { None }
        })
        .unwrap_or_else(|| {
            // Fallback: guess from package name + target_directory
            let profile = if release { "release" } else { "debug" };
            target_directory.join(format!("mipsel-sony-psp/{profile}/{package}.prx"))
                .to_string_lossy().into_owned()
        });

    let prx_abs = fs::canonicalize(&prx_path).unwrap_or_else(|_| {
        eprintln!("error: PRX not found at {prx_path}");
        process::exit(1);
    });

    // Use workspace root as HostFS root, unless --root was explicitly set.
    let root_dir = if root_dir == PathBuf::from(".") {
        workspace_root
    } else {
        root_dir
    };
    let host0_rel = prx_abs.strip_prefix(&root_dir).unwrap_or(&prx_abs);

    // --- Step 3: Deploy to PSP ---
    eprintln!("==> Connecting to PSP...");
    let runner = PspRunner::connect(root_dir).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        process::exit(1);
    });

    let host0_path = format!("host0:/{}", host0_rel.display());
    eprintln!("==> Executing: {host0_path}");

    let result = runner
        .execute(&host0_path, Duration::from_secs(timeout_secs))
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            process::exit(1);
        });

    match &result.exit_reason {
        ExitReason::ModuleExited => {
            eprintln!("==> Done");
        }
        ExitReason::Timeout => {
            eprintln!("==> Timed out after {timeout_secs}s");
            process::exit(1);
        }
        ExitReason::Disconnected(msg) => {
            eprintln!("==> PSP disconnected: {msg}");
            process::exit(1);
        }
    }
}

/// Run `cargo metadata` and return the parsed JSON.
fn cargo_metadata() -> serde_json::Value {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()
        .unwrap_or_else(|e| {
            eprintln!("error: failed to run `cargo metadata`: {e}");
            process::exit(1);
        });
    if !output.status.success() {
        eprintln!("error: `cargo metadata` failed");
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }
    serde_json::from_slice(&output.stdout).unwrap_or_else(|e| {
        eprintln!("error: failed to parse cargo metadata: {e}");
        process::exit(1);
    })
}

/// Infer the package name from cargo metadata by matching manifest_path to CWD.
fn infer_package_name(metadata: &serde_json::Value) -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let packages = metadata["packages"].as_array()?;
    for pkg in packages {
        let manifest = pkg["manifest_path"].as_str()?;
        let pkg_dir = PathBuf::from(manifest).parent()?.to_path_buf();
        if cwd == pkg_dir {
            return pkg["name"].as_str().map(String::from);
        }
    }
    // If only one package in the workspace, use that
    if packages.len() == 1 {
        return packages[0]["name"].as_str().map(String::from);
    }
    None
}
