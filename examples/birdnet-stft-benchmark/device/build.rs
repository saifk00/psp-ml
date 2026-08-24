//! Build pipeline for the STFT frontend benchmark. Two generated modules from
//! the same frontend, one per mode:
//!
//!   dense_gather:  models/birdnet/stft/frontend.tflite (sliced out of
//!                  BirdNET by slice_stft.py) through the normal psp-tc
//!                  pipeline — Range/Gather/window-MUL/Rfft, the 2.3 MiB
//!                  index constant and the ~9 MiB of materialised windows.
//!   strided_view:  the same two branches hand-built with PspModelBuilder as
//!                  StridedViewStft ops — windows are strided views into the
//!                  signal, so only the FFT scratch and the outputs exist.
//!
//! Both land in their own OUT_DIR subdir under their own module name
//! (`stft_dense.rs` + `stft_dense_weights.bin`, likewise `stft_strided`), so
//! one crate can include both. `bench_config.rs` carries each mode's
//! compile-time memory numbers for the runtime report.

use std::path::{Path, PathBuf};
use std::process::Command;

const SLICER: &str = "examples/birdnet-stft-benchmark/slice_stft.py";
const ASSETS: &str = "models/birdnet/stft";
const N_SAMPLES: usize = 144_000;
const N_WINDOWS: usize = 511;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let repo_root = manifest_dir.join("../../..");
    let assets = repo_root.join(ASSETS);

    println!("cargo:rerun-if-env-changed=BIRDNET_PYTHON");
    println!("cargo:rerun-if-changed={}", repo_root.join(SLICER).display());
    for f in ["frontend.tflite", "window_2048.bin", "window_1024.bin", "samples.bin"] {
        println!("cargo:rerun-if-changed={}", assets.join(f).display());
    }

    ensure_assets(&repo_root, &assets);

    // Dense mode: the sliced .tflite through the standard pipeline.
    let dense_dir = out_dir.join("dense");
    std::fs::create_dir_all(&dense_dir).unwrap();
    let dense = psp_tc::compile_tflite_named(
        &assets.join("frontend.tflite"),
        &dense_dir,
        None,
        "stft_dense",
    )
    .unwrap_or_else(|e| panic!("psp-tc codegen (dense) failed: {e}"));

    // Strided mode: hand-built graph, same branch order as the slice
    // (L=2048 first) so output0/output1 line up across modes.
    let mut b = psp_tc::PspModelBuilder::new();
    let samples = b.input(vec![N_SAMPLES]);
    for fft_length in [2048usize, 1024] {
        let window = read_f32_bin(&assets.join(format!("window_{fft_length}.bin")));
        assert_eq!(window.len(), fft_length);
        let w = b.constant_f32(vec![fft_length], &window);
        let stft = b.strided_view_stft(samples, Some(w), fft_length, N_WINDOWS);
        b.output(stft);
    }
    let mut model = b.finish();
    let strided_dir = out_dir.join("strided");
    std::fs::create_dir_all(&strided_dir).unwrap();
    let strided = psp_tc::compile_graph(&mut model, &strided_dir, "stft_strided")
        .unwrap_or_else(|e| panic!("psp-tc codegen (strided) failed: {e}"));

    // The model input, staged into OUT_DIR so include_bytes! has a stable path.
    std::fs::copy(assets.join("samples.bin"), out_dir.join("samples.bin")).unwrap();

    // Compile-time memory numbers, per mode, for the runtime report. Statics
    // = arena + output buffers: what the module loader claims from partition
    // memory before anything runs. The blob is embedded (<16 MiB), so it
    // arrives as part of the module image.
    let config = format!(
        "pub const N_WINDOWS: usize = {N_WINDOWS};\n\
         pub const DENSE_ARENA_BYTES: usize = {};\n\
         pub const DENSE_BLOB_BYTES: usize = {};\n\
         pub const STRIDED_ARENA_BYTES: usize = {};\n\
         pub const STRIDED_BLOB_BYTES: usize = {};\n\
         pub const OUTPUT_BYTES: usize = {};\n",
        dense.arena_size_floats * 4,
        dense.blob_bytes,
        strided.arena_size_floats * 4,
        strided.blob_bytes,
        dense.output_size_floats * 4,
    );
    std::fs::write(out_dir.join("bench_config.rs"), config).unwrap();

    println!(
        "cargo:warning=stft-bench dense:   arena {} B + blob {} B",
        dense.arena_size_floats * 4,
        dense.blob_bytes
    );
    println!(
        "cargo:warning=stft-bench strided: arena {} B + blob {} B",
        strided.arena_size_floats * 4,
        strided.blob_bytes
    );
}

/// Regenerate the sliced frontend + fixtures when missing. Same convention as
/// birdnet's pruner: `uv run` resolves the slicer's PEP 723 deps;
/// BIRDNET_PYTHON bypasses uv with a ready interpreter.
fn ensure_assets(repo_root: &Path, assets: &Path) {
    let needed = [
        "frontend.tflite",
        "window_2048.bin",
        "window_1024.bin",
        "samples.bin",
        "golden_2048.bin",
        "golden_1024.bin",
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
