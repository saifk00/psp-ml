//! Cargo subcommand: `cargo psp-ml`
//!
//! Usage:
//!   cargo psp-ml compile model.tflite -o src/
//!   cargo psp-ml run -p hello-psp --release

use std::fs;
use std::path::PathBuf;
use std::process;

fn main() {
    env_logger::init();

    // When invoked as `cargo psp-ml`, cargo passes "psp-ml" as argv[1]. Strip it.
    let args: Vec<String> = std::env::args().collect();
    let args = if args.get(1).map(|s| s.as_str()) == Some("psp-ml") {
        &args[2..]
    } else {
        &args[1..]
    };

    match args.first().map(|s| s.as_str()) {
        Some("compile") => cmd_compile(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("--help") | Some("-h") | None => print_usage(),
        Some(other) => {
            eprintln!("error: unknown subcommand '{other}'");
            eprintln!();
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("cargo psp-ml — PSP ML toolchain");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  cargo psp-ml compile <model.tflite> [-o <dir>]");
    eprintln!("  cargo psp-ml run -p <package> [--release] [--timeout <secs>]");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  compile   Compile a TFLite model into Rust code + weights");
    eprintln!("  run       Build, deploy, and run on a PSP over USB");
}

// ---------------------------------------------------------------------------
// compile
// ---------------------------------------------------------------------------

fn cmd_compile(args: &[String]) {
    use psp_ml_compiler::codegen::generate_code;
    use psp_ml_compiler::parse::tflite;

    let mut model_path: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" | "-o" => {
                i += 1;
                out_dir = Some(PathBuf::from(args.get(i).unwrap_or_else(|| {
                    eprintln!("Usage: cargo psp-ml compile <model.tflite> [-o <dir>]");
                    process::exit(1);
                })));
            }
            "--help" | "-h" => {
                eprintln!("Usage: cargo psp-ml compile <model.tflite> [-o <dir>]");
                eprintln!();
                eprintln!("Compile a TFLite model into Rust code targeting the psp-ml runtime.");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -o, --out <DIR>  Output directory (default: current directory)");
                process::exit(0);
            }
            _ => {
                if model_path.is_none() {
                    model_path = Some(args[i].clone());
                } else {
                    eprintln!("Unexpected argument: {}", args[i]);
                    eprintln!("Usage: cargo psp-ml compile <model.tflite> [-o <dir>]");
                    process::exit(1);
                }
            }
        }
        i += 1;
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Usage: cargo psp-ml compile <model.tflite> [-o <dir>]");
        process::exit(1);
    });

    let out_dir = out_dir.unwrap_or_else(|| PathBuf::from("."));

    let data = fs::read(&model_path).expect("Failed to read model");
    let psp_model = tflite::to_psp_ir(data).unwrap_or_else(|err| {
        eprintln!("Error lowering to IR: {err}");
        process::exit(1);
    });

    let generated = generate_code(&psp_model).unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        process::exit(1);
    });

    let weights_path = out_dir.join("weights.bin");
    if let Err(err) = fs::write(&weights_path, &generated.data_bytes) {
        eprintln!("Error writing {}: {err}", weights_path.display());
        process::exit(1);
    }

    let syntax_tree =
        syn::parse2::<syn::File>(generated.tokens).expect("Failed to parse generated code");
    let formatted = prettyplease::unparse(&syntax_tree);

    let generated_path = out_dir.join("generated.rs");
    if let Err(err) = fs::write(&generated_path, formatted) {
        eprintln!("Error writing {}: {err}", generated_path.display());
        process::exit(1);
    }

    eprintln!(
        "Generated {} and {}",
        generated_path.display(),
        weights_path.display()
    );
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: &[String]) {
    use psp_ml_compiler::runner::{ExitReason, PspRunner};
    use std::process::Command;
    use std::time::Duration;

    let mut package: Option<String> = None;
    let mut bin: Option<String> = None;
    let mut features: Option<String> = None;
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
            "--bin" => {
                i += 1;
                bin = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("--bin requires a binary name");
                    process::exit(1);
                }).clone());
            }
            "--features" => {
                i += 1;
                features = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("--features requires a feature list");
                    process::exit(1);
                }).clone());
            }
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
                eprintln!("Usage: cargo psp-ml run [OPTIONS] -p <PACKAGE>");
                eprintln!();
                eprintln!("Build a PSP PRX and deploy it to a PSP running psplink.");
                eprintln!();
                eprintln!("Build options:");
                eprintln!("  -p, --package <PKG>    Package to build and run");
                eprintln!("  --bin <NAME>           Binary target (default: package name)");
                eprintln!("  --features <FEATURES>  Comma-separated features to activate");
                eprintln!("  --release              Build in release mode");
                eprintln!();
                eprintln!("Runner options:");
                eprintln!("  --timeout <SECS>       Timeout in seconds (default: 60)");
                eprintln!("  --root <DIR>           HostFS root directory (default: workspace root)");
                eprintln!();
                eprintln!("Examples:");
                eprintln!("  cargo psp-ml run -p hello-psp --release");
                eprintln!("  cargo psp-ml run -p psp-ml --bin test-kernels --features test-kernels --release");
                eprintln!();
                eprintln!("Environment:");
                eprintln!("  RUST_LOG=info    Show protocol activity");
                eprintln!("  RUST_LOG=debug   Show all HostFS commands");
                process::exit(0);
            }
            other => {
                eprintln!("unexpected argument: {other}");
                eprintln!("       cargo psp-ml run --help for usage");
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

    let package = package.unwrap_or_else(|| {
        eprintln!("error: -p <PACKAGE> is required");
        eprintln!("       cargo psp-ml run --help for usage");
        process::exit(1);
    });

    // The PRX filename comes from the binary name, not the package name.
    let prx_name = bin.as_deref().unwrap_or(&package);

    // --- Step 1: Build via `cargo psp` ---
    let mut build_cmd = Command::new("cargo");
    build_cmd.arg("+nightly").arg("psp");
    if release {
        build_cmd.arg("--release");
    }
    build_cmd.arg("-p").arg(&package);
    if let Some(b) = &bin {
        build_cmd.arg("--bin").arg(b);
    }
    if let Some(f) = &features {
        build_cmd.arg("--features").arg(f);
    }

    eprintln!("==> Building: cargo +nightly psp{} -p {package}{}{}",
        if release { " --release" } else { "" },
        bin.as_ref().map(|b| format!(" --bin {b}")).unwrap_or_default(),
        features.as_ref().map(|f| format!(" --features {f}")).unwrap_or_default(),
    );

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

    // --- Step 2: Find the PRX ---
    // Forward build stdout (PBP table) to stderr for visibility
    let build_stdout = String::from_utf8_lossy(&output.stdout);
    if !build_stdout.is_empty() {
        eprint!("{build_stdout}");
    }
    let profile = if release { "release" } else { "debug" };
    let prx_path = target_directory
        .join(format!("mipsel-sony-psp/{profile}/{prx_name}.prx"));

    let prx_abs = fs::canonicalize(&prx_path).unwrap_or_else(|_| {
        eprintln!("error: PRX not found at {}", prx_path.display());
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

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn cargo_metadata() -> serde_json::Value {
    use std::process::Command;

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
