//! Build pipeline for the birdnet device crate. Each step feeds the next:
//!
//!   1. pick the model      full .tflite, or a TOPK-pruned copy in OUT_DIR
//!   2. psp-tc compile      -> generated.rs + weights.bin
//!   3. emit classes.rs     -> OUTPUT_CLASSES, so main.rs matches the model
//!   4. embed the audio     cardinal_3s.wav -> audio_f32.bin
//!   5. stage for deploy    weights.bin next to the .prx
//!
//! Steps 1 and 3 exist only because pruning changes the model's output width;
//! everything downstream (labels, class count, blob) is derived from whichever
//! model step 1 selected, so the three can never disagree.
//!
//! Knobs: TOPK, BIRDNET_REGION, BIRDNET_BBOX, BIRDNET_PYTHON,
//! BIRDNET_GENERATED_OVERRIDE. See examples/birdnet/README.md.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const FULL_MODEL: &str = "models/birdnet/audio-model-int8.tflite";
const FULL_LABELS: &str = "models/birdnet/labels/en_us.txt";
const PRUNER: &str = "examples/birdnet/prune_classifier.py";
const INPUT_SAMPLES: usize = 144_000;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let repo_root = manifest_dir.join("../../..");

    for var in [
        "TOPK",
        "BIRDNET_REGION",
        "BIRDNET_BBOX",
        "BIRDNET_PYTHON",
        "BIRDNET_GENERATED_OVERRIDE",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed={}", repo_root.join(FULL_MODEL).display());
    println!("cargo:rerun-if-changed={}", repo_root.join(FULL_LABELS).display());
    println!("cargo:rerun-if-changed={}", repo_root.join(PRUNER).display());
    println!("cargo:rerun-if-changed=../cardinal_3s.wav");

    // 1. Model selection. Also writes labels.txt (and, when pruned,
    //    kept_indices.txt) into OUT_DIR alongside it.
    let model = select_model(&repo_root, &out_dir);

    // 2. Codegen. Produces generated.rs and weights.bin in OUT_DIR.
    psp_tc::compile_tflite(&model, &out_dir, None)
        .unwrap_or_else(|e| panic!("psp-tc codegen failed: {e}"));
    apply_generated_override(&out_dir);

    // 3. Class count, read back from the labels the selected model implies.
    let num_classes = read_lines(&out_dir.join("labels.txt")).len();
    write(
        &out_dir.join("classes.rs"),
        format!("pub const OUTPUT_CLASSES: usize = {num_classes};\n").as_bytes(),
    );
    println!("cargo:warning=birdnet: {num_classes} output classes");

    // 4. Model input, pre-converted so the device does not need a WAV parser.
    embed_audio(&manifest_dir, &out_dir);

    // 5. Hand the blob to the host runner.
    stage_for_deploy(&out_dir);
}

/// Returns the .tflite psp-tc should compile, and writes the labels that go
/// with it (plus `kept_indices.txt` when pruned) into `out_dir`.
///
/// Unset TOPK means the stock 6522-class model and the full labels file.
fn select_model(repo_root: &Path, out_dir: &Path) -> PathBuf {
    let full_model = repo_root.join(FULL_MODEL);
    let full_labels = repo_root.join(FULL_LABELS);

    let topk = match std::env::var("TOPK") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => {
            // Unpruned: labels are the full list, and no index map -- deleting
            // it matters, or a stale map from a previous pruned build would be
            // applied to full-width output.
            std::fs::copy(&full_labels, out_dir.join("labels.txt"))
                .expect("failed to read models/birdnet/labels/en_us.txt");
            let _ = std::fs::remove_file(out_dir.join("kept_indices.txt"));
            return full_model;
        }
    };
    let topk: u32 = topk
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("TOPK must be a positive integer, got {topk:?} ({e})"));
    assert!(topk > 0, "TOPK must be > 0");

    let pruned = out_dir.join("audio-model-pruned.tflite");
    let mut cmd = pruner_command(&repo_root.join(PRUNER));
    cmd.arg(&full_model)
        .arg(&full_labels)
        .arg("--top-n")
        .arg(topk.to_string())
        .arg("-o")
        .arg(&pruned)
        .arg("--write-labels")
        .arg(out_dir.join("labels.txt"))
        .arg("--write-indices")
        .arg(out_dir.join("kept_indices.txt"));
    // An explicit bbox wins over the named region.
    match std::env::var("BIRDNET_BBOX") {
        Ok(s) if !s.trim().is_empty() => {
            let corners: Vec<&str> = s.split([',', ' ']).filter(|p| !p.is_empty()).collect();
            assert_eq!(
                corners.len(),
                4,
                "BIRDNET_BBOX must be 'lat0,lat1,lon0,lon1', got {s:?}"
            );
            cmd.arg("--bbox").args(corners);
        }
        _ => {
            let region = std::env::var("BIRDNET_REGION").unwrap_or_else(|_| "eastern-na".into());
            cmd.arg("--region").arg(region);
        }
    }

    let status = cmd
        .current_dir(repo_root)
        .status()
        .unwrap_or_else(|e| panic!("failed to launch the pruner ({e}). {PYTHON_HELP}"));
    assert!(status.success(), "prune_classifier.py failed. {PYTHON_HELP}");
    pruned
}

const PYTHON_HELP: &str = "Install uv, or set BIRDNET_PYTHON to a python that \
                           already has the pruner's deps (ai-edge-litert or \
                           tensorflow, plus numpy).";

/// `uv run` resolves the pruner's PEP 723 inline deps, matching how the
/// reference scripts in models/ are run. BIRDNET_PYTHON bypasses uv with an
/// interpreter that already has them.
fn pruner_command(script: &Path) -> Command {
    match std::env::var("BIRDNET_PYTHON") {
        Ok(py) if !py.trim().is_empty() => {
            let mut c = Command::new(py);
            c.arg(script);
            c
        }
        _ => {
            let mut c = Command::new("uv");
            c.arg("run").arg(script);
            c
        }
    }
}

/// Prototyping hook: swap in a hand-edited `generated.rs` to measure a codegen
/// optimisation before implementing it in psp-tc. The model is still compiled,
/// so weights.bin and the blob layout stay consistent — only the emitted code
/// is replaced, and it must match that same blob.
fn apply_generated_override(out_dir: &Path) {
    if let Ok(over) = std::env::var("BIRDNET_GENERATED_OVERRIDE") {
        std::fs::copy(&over, out_dir.join("generated.rs"))
            .unwrap_or_else(|e| panic!("failed to apply {over}: {e}"));
        println!("cargo:warning=using hand-edited generated.rs from {over}");
        println!("cargo:rerun-if-changed={over}");
    }
}

/// Convert the 3 s window to f32 exactly as models/birdnet_reference.py does
/// (i16 / 32768.0), which is what makes the device's input bit-identical to the
/// Python golden run.
fn embed_audio(manifest_dir: &Path, out_dir: &Path) {
    let wav_path = manifest_dir.join("../cardinal_3s.wav");
    let wav = std::fs::read(&wav_path).unwrap_or_else(|e| {
        panic!(
            "{} not found ({e}); run `uv run models/birdnet_reference.py` first",
            wav_path.display()
        )
    });

    let pcm = &wav[44..]; // skip WAV header
    let mut floats = vec![0.0f32; INPUT_SAMPLES];
    for (i, slot) in floats.iter_mut().enumerate().take(pcm.len() / 2) {
        *slot = i16::from_le_bytes([pcm[i * 2], pcm[i * 2 + 1]]) as f32 / 32768.0;
    }

    let bytes: Vec<u8> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
    write(&out_dir.join("audio_f32.bin"), &bytes);
}

/// Stage weights.bin (and the index map) beside the .prx this build produces.
///
/// Deliberately *not* copied into the host0: mount here. Several build trees
/// compile this crate — the workspace `cargo build -p birdnet` and cargo-psp's
/// own `--target-dir` — each with its own OUT_DIR and its own blob. If they all
/// wrote to the shared mount, the last one to run would win and could leave a
/// blob that does not match the deployed .prx (an unpruned 42.8 MB blob against
/// a TOPK .prx expecting 17.8 MB). `init()` reads exactly WEIGHT_BYTES without
/// validating the file, so that mismatch is silent: the run computes on wrong
/// offsets and produces garbage. The host runner copies from here at deploy
/// time instead, taking blob and .prx from the same directory.
fn stage_for_deploy(out_dir: &Path) {
    // Locate the profile dir by name, not by depth: plain cargo nests OUT_DIR as
    // <profile>/build/<pkg>-<hash>/out, while cargo-psp splits package and hash
    // into separate components, so a fixed ancestor count fits only one of them.
    let profile = std::env::var("PROFILE").unwrap();
    let stage_dir = out_dir
        .ancestors()
        .find(|p| p.file_name().is_some_and(|n| n == profile.as_str()))
        .unwrap_or_else(|| panic!("no `{profile}` component in OUT_DIR {}", out_dir.display()));

    std::fs::copy(out_dir.join("weights.bin"), stage_dir.join("weights.bin"))
        .expect("failed to stage weights.bin beside the prx");

    let kept = out_dir.join("kept_indices.txt");
    if kept.exists() {
        std::fs::copy(&kept, stage_dir.join("kept_indices.txt"))
            .expect("failed to stage kept_indices.txt");
    } else {
        let _ = std::fs::remove_file(stage_dir.join("kept_indices.txt"));
    }
}

fn read_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
        .lines()
        .map(str::to_owned)
        .collect()
}

fn write(path: &Path, bytes: &[u8]) {
    let mut f = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
    f.write_all(bytes)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}
