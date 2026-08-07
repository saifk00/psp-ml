//! Cargo subcommand: `cargo psp-ml`
//!
//! Usage:
//!   cargo psp-ml compile model.tflite -o src/
//!   cargo psp-ml run -p hello-psp --release

use std::fs;
use std::path::PathBuf;
use std::process;

fn main() {
    // When invoked as `cargo psp-ml`, cargo passes "psp-ml" as argv[1]. Strip it.
    let args: Vec<String> = std::env::args().collect();
    let args = if args.get(1).map(|s| s.as_str()) == Some("psp-ml") {
        &args[2..]
    } else {
        &args[1..]
    };

    match args.first().map(|s| s.as_str()) {
        Some("compile") => cmd_compile(&args[1..]),
        Some("info") => cmd_info(&args[1..]),
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
    eprintln!("  cargo psp-ml info <model.tflite>");
    eprintln!("  cargo psp-ml run -p <package> [--release]");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  compile   Compile a TFLite model into Rust code + weights");
    eprintln!("  info      Dump model ops, shapes, dtypes, and supported/missing analysis");
    eprintln!("  run       Build and deploy a PRX to a PSP running psplink");
}

// ---------------------------------------------------------------------------
// compile
// ---------------------------------------------------------------------------

fn cmd_compile(args: &[String]) {
    use cargo_psp_ml::codegen::generate_code;
    use cargo_psp_ml::parse::tflite;

    let mut model_path: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut dump_ir = false;
    let mut stream_batch: Option<(usize, usize)> = None;

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
            "--dump-ir" => dump_ir = true,
            "--stream-batch" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("--stream-batch requires START:END argument");
                    process::exit(1);
                });
                let parts: Vec<&str> = val.split(':').collect();
                if parts.len() != 2 {
                    eprintln!("--stream-batch format: START:END (e.g. 38:278)");
                    process::exit(1);
                }
                let start: usize = parts[0].parse().unwrap_or_else(|_| {
                    eprintln!("--stream-batch START must be a number");
                    process::exit(1);
                });
                let end: usize = parts[1].parse().unwrap_or_else(|_| {
                    eprintln!("--stream-batch END must be a number");
                    process::exit(1);
                });
                stream_batch = Some((start, end));
            }
            "--help" | "-h" => {
                eprintln!("Usage: cargo psp-ml compile <model.tflite> [-o <dir>] [--dump-ir]");
                eprintln!();
                eprintln!("Compile a TFLite model into Rust code targeting the psp-ml runtime.");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -o, --out <DIR>           Output directory (default: current directory)");
                eprintln!("  --dump-ir                 Print IR graph after each pipeline stage");
                eprintln!("  --stream-batch START:END  Process batch frames one at a time (op indices)");
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
    let mut psp_model = tflite::to_psp_ir(data, dump_ir).unwrap_or_else(|err| {
        eprintln!("Error lowering to IR: {err}");
        process::exit(1);
    });

    let generated = generate_code(&mut psp_model, stream_batch).unwrap_or_else(|err| {
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
    eprintln!();
    eprintln!("{}", generated.stats);
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

fn cmd_info(args: &[String]) {
    use cargo_psp_ml::parse::tflite;

    let model_path = match args.first() {
        Some(p) if p != "--help" && p != "-h" => p,
        _ => {
            eprintln!("Usage: cargo psp-ml info <model.tflite>");
            eprintln!();
            eprintln!("Dump model operators, tensor shapes, data types, and");
            eprintln!("supported/missing op analysis without compiling.");
            process::exit(if args.first().map(|s| s.as_str()) == Some("--help") { 0 } else { 1 });
        }
    };

    let data = fs::read(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {model_path}: {e}");
        process::exit(1);
    });

    tflite::dump_model_info(&data);
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: &[String]) {
    use std::process::Command;

    let mut package: Option<String> = None;
    let mut bin: Option<String> = None;
    let mut features: Option<String> = None;
    let mut release = false;

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
            "--help" | "-h" => {
                eprintln!("Usage: cargo psp-ml run [OPTIONS] -p <PACKAGE>");
                eprintln!();
                eprintln!("Build a PSP PRX and deploy it to a PSP running psplink over USB.");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -p, --package <PKG>    Package to build and run");
                eprintln!("  --bin <NAME>           Binary target (default: package name)");
                eprintln!("  --features <FEATURES>  Comma-separated features to activate");
                eprintln!("  --release              Build in release mode");
                eprintln!();
                eprintln!("Examples:");
                eprintln!("  cargo psp-ml run -p hello-psp --release");
                eprintln!("  cargo psp-ml run -p psp-ml --bin test-kernels --features test-kernels --release");
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
    let _workspace_root = PathBuf::from(
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
    let build_stderr = String::from_utf8_lossy(&output.stderr);
    if !build_stderr.is_empty() {
        eprint!("{build_stderr}");
    }
    if !output.status.success() {
        eprintln!("error: build failed");
        process::exit(output.status.code().unwrap_or(1));
    }

    // --- Step 2: Find the PRX ---
    let build_stdout = String::from_utf8_lossy(&output.stdout);
    if !build_stdout.is_empty() {
        eprint!("{build_stdout}");
    }
    let profile = if release { "release" } else { "debug" };

    // host1: is mounted to the profile's target dir (where the PRX lives)
    let host1_target = target_directory.join(format!("mipsel-sony-psp/{profile}"));
    let prx_path = format!("host1:{prx_name}.prx");

    // host0: = CWD (where user code writes files)
    // host1: = workspace root (where PRX lives, used by ld command)
    let host0_target = std::env::current_dir().unwrap_or_else(|e| {
        eprintln!("error: cannot determine current directory: {e}");
        process::exit(1);
    });

    // Connects directly over USB (native FFI, via psplink-connection) —
    // no usbhostfs_pc subprocess, no TCP bridge, no interactive-shell mount
    // race to work around. host0/host1 are configured before the handshake,
    // same as psplink-connection's own connect().
    eprintln!("==> Connecting: host0:{} host1:{}", host0_target.display(), host1_target.display());
    let conn = psplink_connection::PSPConnection::connect(&host0_target, &host1_target, Default::default())
        .unwrap_or_else(|e| {
            eprintln!("error: failed to connect to PSP: {e}");
            process::exit(1);
        });

    eprintln!("==> Loading {prx_path}");
    let outcome = conn
        .load_program(&prx_path, |bytes| {
            use std::io::Write;
            let _ = std::io::stdout().write_all(bytes);
        })
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            process::exit(1);
        });

    use psplink_connection::LoadOutcome;
    match outcome {
        LoadOutcome::Success => eprintln!("==> Done"),
        LoadOutcome::Panicked => {
            eprintln!("==> Program panicked");
            process::exit(1);
        }
        LoadOutcome::ShellError(v) => {
            eprintln!("==> Shell error loading PRX: 0x{v:08X}");
            process::exit(1);
        }
        LoadOutcome::KernelError(v) => {
            eprintln!("==> Kernel error: 0x{v:08X}");
            process::exit(1);
        }
    }

    conn.disconnect();
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
