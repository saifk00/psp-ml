//! Build pipeline for the whole-frontend benchmark — the STFT and mel pieces
//! tied together, one generated module per mode:
//!
//!   dense_tflite:  models/birdnet/stft/frontend_mel.tflite (the frontend as
//!                  BirdNET expresses it, sliced whole: dense gathers,
//!                  materialised windows, dense mel FCs, MUL+POW).
//!   custom_ops:    psp_tc::stft_mel_frontend — per branch a StridedViewStft
//!                  then FullyConnectedCB (make_mel filterbank) + SquarePow.
//!
//! Input is the normalised signal (samples.bin) so both modes and the TFLite
//! goldens see bit-identical input bytes.

use std::path::{Path, PathBuf};
use std::process::Command;

const SLICER: &str = "examples/birdnet-stft-benchmark/slice_stft.py";
const ASSETS: &str = "models/birdnet/stft";
const N_SAMPLES: usize = 144_000;
const N_WINDOWS: usize = 511;
const N_BANKS: usize = 96;

/// (fft_length, fmin, fmax) — reverse-engineered filterbank ranges, verified
/// against the stored matrices by psp-tc's mel tests.
const BRANCHES: [(usize, f64, f64); 2] = [(2048, 0.0, 3000.0), (1024, 500.0, 15000.0)];
const SAMPLING_RATE: f64 = 48_000.0;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let repo_root = manifest_dir.join("../../..");
    let assets = repo_root.join(ASSETS);

    println!("cargo:rerun-if-env-changed=BIRDNET_PYTHON");
    println!("cargo:rerun-if-changed={}", repo_root.join(SLICER).display());
    for f in [
        "frontend_mel.tflite",
        "samples.bin",
        "window_2048.bin",
        "window_1024.bin",
        "golden_mel_2048.bin",
        "golden_mel_1024.bin",
        "manifest.json",
    ] {
        println!("cargo:rerun-if-changed={}", assets.join(f).display());
    }

    ensure_assets(&repo_root, &assets);

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(assets.join("manifest.json")).unwrap(),
    )
    .unwrap();
    let pow_exponent = |fft_length: usize| -> f32 {
        manifest["branches"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["fft_length"].as_u64() == Some(fft_length as u64))
            .unwrap_or_else(|| panic!("no manifest entry for L={fft_length}"))["pow_exponent"]
            .as_f64()
            .unwrap() as f32
    };

    // Dense mode: the whole sliced frontend through the standard pipeline.
    let dense_dir = out_dir.join("dense");
    std::fs::create_dir_all(&dense_dir).unwrap();
    let dense = psp_tc::compile_tflite_named(
        &assets.join("frontend_mel.tflite"),
        &dense_dir,
        None,
        "frontend_dense",
    )
    .unwrap_or_else(|e| panic!("psp-tc codegen (dense) failed: {e}"));

    // Custom mode: both branches on the builder.
    let branches: Vec<psp_tc::mel::FrontendBranch> = BRANCHES
        .iter()
        .map(|&(l, fmin, fmax)| psp_tc::mel::FrontendBranch {
            fft_length: l,
            window: read_f32_bin(&assets.join(format!("window_{l}.bin"))),
            fmin,
            fmax,
            pow_exponent: pow_exponent(l),
        })
        .collect();
    let mut b = psp_tc::PspModelBuilder::new();
    let samples = b.input(vec![1, N_SAMPLES]);
    let outs = psp_tc::mel::stft_mel_frontend(
        &mut b,
        samples,
        N_WINDOWS,
        SAMPLING_RATE,
        N_BANKS,
        &branches,
    );
    for out in outs {
        b.output(out);
    }
    let mut model = b.finish();
    let custom_dir = out_dir.join("custom");
    std::fs::create_dir_all(&custom_dir).unwrap();
    let custom = psp_tc::compile_graph(&mut model, &custom_dir, "frontend_custom")
        .unwrap_or_else(|e| panic!("psp-tc codegen (custom) failed: {e}"));

    // Small-FFT mode: the same custom frontend with the FFT-pruning pass —
    // where the mel banks read only a low prefix of the bins, the FFT
    // shrinks to the rounded needed size and the signal is anti-alias
    // filtered + read at a stride so the bin grid is unchanged.
    let mut b = psp_tc::PspModelBuilder::new();
    let samples = b.input(vec![1, N_SAMPLES]);
    let outs = psp_tc::stft_mel_frontend_small_fft(
        &mut b,
        samples,
        N_WINDOWS,
        SAMPLING_RATE,
        N_BANKS,
        &branches,
    );
    for out in outs {
        b.output(out);
    }
    let mut model = b.finish();
    let small_dir = out_dir.join("small");
    std::fs::create_dir_all(&small_dir).unwrap();
    let small = psp_tc::compile_graph(&mut model, &small_dir, "frontend_small")
        .unwrap_or_else(|e| panic!("psp-tc codegen (small) failed: {e}"));

    std::fs::copy(assets.join("samples.bin"), out_dir.join("samples.bin")).unwrap();

    let output_bytes = 2 * N_WINDOWS * N_BANKS * 4;
    let config = format!(
        "pub const N_WINDOWS: usize = {N_WINDOWS};\n\
         pub const N_BANKS: usize = {N_BANKS};\n\
         pub const DENSE_ARENA_BYTES: usize = {};\n\
         pub const DENSE_BLOB_BYTES: usize = {};\n\
         pub const CUSTOM_ARENA_BYTES: usize = {};\n\
         pub const CUSTOM_BLOB_BYTES: usize = {};\n\
         pub const SMALL_ARENA_BYTES: usize = {};\n\
         pub const SMALL_BLOB_BYTES: usize = {};\n\
         pub const OUTPUT_BYTES: usize = {output_bytes};\n",
        dense.arena_size_floats * 4,
        dense.blob_bytes,
        custom.arena_size_floats * 4,
        custom.blob_bytes,
        small.arena_size_floats * 4,
        small.blob_bytes,
    );
    std::fs::write(out_dir.join("bench_config.rs"), config).unwrap();

    println!(
        "cargo:warning=frontend-bench dense:  arena {} B + blob {} B",
        dense.arena_size_floats * 4,
        dense.blob_bytes
    );
    println!(
        "cargo:warning=frontend-bench custom: arena {} B + blob {} B",
        custom.arena_size_floats * 4,
        custom.blob_bytes
    );
    println!(
        "cargo:warning=frontend-bench small:  arena {} B + blob {} B",
        small.arena_size_floats * 4,
        small.blob_bytes
    );
}

fn ensure_assets(repo_root: &Path, assets: &Path) {
    let needed = [
        "frontend_mel.tflite",
        "samples.bin",
        "window_2048.bin",
        "window_1024.bin",
        "golden_mel_2048.bin",
        "golden_mel_1024.bin",
        "manifest.json",
    ];
    if needed.iter().all(|f| assets.join(f).exists()) {
        return;
    }

    let slicer = repo_root.join(SLICER);
    let mut cmd = match std::env::var("BIRDNET_PYTHON") {
        Ok(py) if !py.trim().is_empty() => {
            let mut c = Command::new(py);
            c.arg(&slicer);
            c
        }
        _ => {
            let mut c = Command::new("uv");
            c.arg("run").arg(&slicer);
            c
        }
    };
    let status = cmd.current_dir(repo_root).status().unwrap_or_else(|e| {
        panic!("failed to run {} ({e}). {PYTHON_HELP}", slicer.display())
    });
    assert!(status.success(), "slice_stft.py failed. {PYTHON_HELP}");
}

const PYTHON_HELP: &str = "Install uv, or set BIRDNET_PYTHON to a python that \
has ai-edge-litert (or tensorflow), numpy and flatbuffers installed. Requires \
models/birdnet/audio-model-int8.tflite (gitignored, downloaded separately).";

fn read_f32_bin(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
