use std::io::Write;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
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

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = std::path::Path::new(&out_dir).join("audio_f32.bin");
    let mut f = std::fs::File::create(&out_path).unwrap();
    for &val in &floats {
        f.write_all(&val.to_le_bytes()).unwrap();
    }

    println!("cargo:rerun-if-changed=recording.wav");
}
