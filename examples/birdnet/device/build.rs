use std::io::Write;
use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // TFLite -> Rust codegen. stream_batch is load-bearing, not optional:
    // processes 511 frames one-at-a-time through the main conv block,
    // reducing the arena from ~161 MiB to ~9.6 MiB (see the removed
    // run.sh's own comment to this effect).
    let model = Path::new(&manifest_dir).join("../../models/BirdNET_v2.4_tflite/audio-model.tflite");
    psp_tc::compile_tflite(&model, Path::new(&out_dir), Some((27, 279))).unwrap_or_else(|e| {
        panic!("psp-tc codegen failed: {e}");
    });
    println!("cargo:rerun-if-changed={}", model.display());

    let wav = std::fs::read(std::path::Path::new(&manifest_dir).join("recording.wav"))
        .expect("recording.wav not found");

    let pcm = &wav[44..]; // skip WAV header
    let num_samples = pcm.len() / 2;
    let count = num_samples.min(144000);

    let mut floats = vec![0.0f32; 144000];
    for i in 0..count {
        let sample = i16::from_le_bytes([pcm[i * 2], pcm[i * 2 + 1]]);
        floats[i] = sample as f32 / 32768.0;
    }

    let out_path = std::path::Path::new(&out_dir).join("audio_f32.bin");
    let mut f = std::fs::File::create(&out_path).unwrap();
    for &val in &floats {
        f.write_all(&val.to_le_bytes()).unwrap();
    }

    println!("cargo:rerun-if-changed=recording.wav");
}
