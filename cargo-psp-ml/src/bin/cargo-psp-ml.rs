use psp_ml::codegen::generate_code;
use psp_ml::parse::tflite;
use std::fs;
use std::path::PathBuf;
use std::process;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut model_path: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" | "-o" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("Usage: cargo psp-ml <model.tflite> [--out <dir>]");
                    process::exit(1);
                });
                out_dir = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                eprintln!("Usage: cargo psp-ml <model.tflite> [--out <dir>]");
                process::exit(0);
            }
            _ => {
                if model_path.is_none() {
                    model_path = Some(arg);
                } else {
                    eprintln!("Unexpected argument: {arg}");
                    eprintln!("Usage: cargo psp-ml <model.tflite> [--out <dir>]");
                    process::exit(1);
                }
            }
        }
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Usage: cargo psp-ml <model.tflite> [--out <dir>]");
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
