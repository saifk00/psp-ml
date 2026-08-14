//! `psp-tc` — standalone CLI for the TFLite→Rust compiler.
//!
//! Usage:
//!   psp-tc compile model.tflite -o src/
//!   psp-tc info model.tflite
//!
//! `build.rs` scripts should call `psp_tc::compile_tflite` directly instead
//! of shelling out to this binary — this CLI exists for manual/diagnostic
//! use (in particular `--dump-ir`, not exposed by the library entry point).

use std::fs;
use std::path::PathBuf;
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.first().map(|s| s.as_str()) {
        Some("compile") => cmd_compile(&args[1..]),
        Some("info") => cmd_info(&args[1..]),
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
    eprintln!("psp-tc — TFLite→Rust compiler for the psp-rt runtime");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  psp-tc compile <model.tflite> [-o <dir>]");
    eprintln!("  psp-tc info <model.tflite>");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  compile   Compile a TFLite model into Rust code + weights");
    eprintln!("  info      Dump model ops, shapes, dtypes, and supported/missing analysis");
}

// ---------------------------------------------------------------------------
// compile
// ---------------------------------------------------------------------------

fn cmd_compile(args: &[String]) {
    use psp_tc::codegen::generate_code;
    use psp_tc::parse::tflite;

    let mut model_path: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut dump_ir = false;
    let mut stream_batch: Option<(usize, usize)> = None;
    let mut resident_budget: Option<usize> = None;
    let mut residency: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" | "-o" => {
                i += 1;
                out_dir = Some(PathBuf::from(args.get(i).unwrap_or_else(|| {
                    eprintln!("Usage: psp-tc compile <model.tflite> [-o <dir>]");
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
            "--residency" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("--residency requires a candidate index");
                    process::exit(1);
                });
                residency = Some(val.parse().unwrap_or_else(|_| {
                    eprintln!("--residency must be a candidate index");
                    process::exit(1);
                }));
            }
            "--help" | "-h" => {
                eprintln!("Usage: psp-tc compile <model.tflite> [-o <dir>] [--dump-ir]");
                eprintln!();
                eprintln!("Compile a TFLite model into Rust code targeting the psp-rt runtime.");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -o, --out <DIR>           Output directory (default: current directory)");
                eprintln!("  --dump-ir                 Print IR graph after each pipeline stage");
                eprintln!("  --stream-batch START:END  Process batch frames one at a time (op indices)");
                eprintln!("  --residency N             Force weight-residency candidate N (0 keeps");
                process::exit(0);
            }
            _ => {
                if model_path.is_none() {
                    model_path = Some(args[i].clone());
                } else {
                    eprintln!("Unexpected argument: {}", args[i]);
                    eprintln!("Usage: psp-tc compile <model.tflite> [-o <dir>]");
                    process::exit(1);
                }
            }
        }
        i += 1;
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Usage: psp-tc compile <model.tflite> [-o <dir>]");
        process::exit(1);
    });

    let out_dir = out_dir.unwrap_or_else(|| PathBuf::from("."));

    let data = fs::read(&model_path).expect("Failed to read model");
    let mut psp_model = tflite::to_psp_ir(data, dump_ir).unwrap_or_else(|err| {
        eprintln!("Error lowering to IR: {err}");
        process::exit(1);
    });

    let generated = generate_code(&mut psp_model, stream_batch, residency, resident_budget).unwrap_or_else(|err| {
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
    use psp_tc::parse::tflite;

    let model_path = match args.first() {
        Some(p) if p != "--help" && p != "-h" => p,
        _ => {
            eprintln!("Usage: psp-tc info <model.tflite>");
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
