use std::io::Write;
use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let model =
        Path::new(&manifest_dir).join("../../../models/birdnet/audio-model-int8.tflite");
    psp_tc::compile_tflite(&model, Path::new(&out_dir), None).unwrap_or_else(|e| {
        panic!("psp-tc codegen failed: {e}");
    });
    println!("cargo:rerun-if-changed={}", model.display());

    // Prototyping hook: swap in a hand-edited `generated.rs` to measure a
    // codegen optimisation before implementing it in psp-tc. The model is still
    // compiled above, so `weights.bin` and the blob layout stay consistent —
    // only the emitted code is replaced, and it must match that same blob.
    if let Ok(over) = std::env::var("BIRDNET_GENERATED_OVERRIDE") {
        std::fs::copy(&over, Path::new(&out_dir).join("generated.rs"))
            .unwrap_or_else(|e| panic!("failed to apply {over}: {e}"));
        println!("cargo:warning=using hand-edited generated.rs from {over}");
        println!("cargo:rerun-if-changed={over}");
    }
    println!("cargo:rerun-if-env-changed=BIRDNET_GENERATED_OVERRIDE");

    // The 3 s cardinal window chosen by models/birdnet_reference.py — using
    // its exact i16 samples (and the same i16/32768.0 conversion) makes the
    // model input bit-identical to the Python golden run.
    let wav_path = Path::new(&manifest_dir).join("../cardinal_3s.wav");
    let wav = std::fs::read(&wav_path).unwrap_or_else(|e| {
        panic!(
            "{} not found ({e}); run `uv run models/birdnet_reference.py` first",
            wav_path.display()
        )
    });

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

    // The device loads resident weights (and streams the classifier) from
    // host0:/weights.bin at runtime; the host runner mounts the example root
    // as host0:, so publish the blob there.
    std::fs::copy(
        Path::new(&out_dir).join("weights.bin"),
        Path::new(&manifest_dir).join("../weights.bin"),
    )
    .expect("failed to copy weights.bin to the host0: mount dir");

    println!("cargo:rerun-if-changed=../cardinal_3s.wav");
}
