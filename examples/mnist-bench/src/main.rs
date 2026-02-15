#![cfg_attr(not(feature = "local"), no_std)]
#![cfg_attr(not(feature = "local"), no_main)]

#[cfg(not(feature = "local"))]
use psp::sys::{sceRtcGetCurrentTick, sceRtcGetTickResolution};

#[cfg(not(feature = "local"))]
psp_ml::module!("mnist_bench", 1, 0);

mod generated;

use psp_ml::bench::{self, BenchmarkResult};

static IMAGES_RAW: &[u8] = include_bytes!("../t10k-images-idx3-ubyte");
static LABELS_RAW: &[u8] = include_bytes!("../t10k-labels-idx1-ubyte");

const NUM_IMAGES: usize = 100;

fn get_image(index: usize, images: &[u8]) -> [f32; 784] {
    let mut img = [0.0f32; 784];
    let offset = index * 784;
    for i in 0..784 {
        img[i] = images[offset + i] as f32 / 255.0;
    }
    img
}

/// Run timed inference, return (total_ticks, correct, per-op ticks).
fn run_benchmark(get_tick: fn() -> u64) -> (u64, u32, [u64; generated::NUM_OPS]) {
    let images = &IMAGES_RAW[16..];
    let labels = &LABELS_RAW[8..];

    let mut op_ticks = [0u64; generated::NUM_OPS];
    let mut correct = 0u32;

    let start = get_tick();
    for i in 0..NUM_IMAGES {
        let img = get_image(i, images);
        let output = generated::forward_timed(&img, &mut op_ticks, get_tick);
        if bench::argmax(&output) == labels[i] as usize {
            correct += 1;
        }
    }
    let total_ticks = get_tick() - start;

    (total_ticks, correct, op_ticks)
}

fn make_result(id: &str, tick_res: u64, total_ticks: u64, correct: u32, op_ticks: &[u64]) -> BenchmarkResult {
    BenchmarkResult {
        id,
        num_samples: NUM_IMAGES,
        tick_resolution: tick_res,
        total_ticks,
        correct,
        op_names: &generated::OP_NAMES,
        op_ticks,
    }
}

// ============================================================================
// PSP entry point
// ============================================================================

#[cfg(not(feature = "local"))]
fn get_tick() -> u64 {
    let mut tick = 0u64;
    unsafe { sceRtcGetCurrentTick(&mut tick) };
    tick
}

#[cfg(not(feature = "local"))]
fn app_main() {
    psp::enable_home_button();

    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;

    psp_ml::dprintln!("MNIST benchmark: {} images", NUM_IMAGES);
    let (total_ticks, correct, op_ticks) = run_benchmark(get_tick);
    let result = make_result("vfpu", tick_res, total_ticks, correct, &op_ticks);

    let total_us = (total_ticks * 1_000_000) / tick_res;
    psp_ml::dprintln!("  {}ms total, {}us/image, {}/{} correct",
        total_us / 1000, total_us / NUM_IMAGES as u64, correct, NUM_IMAGES);

    for (i, name) in generated::OP_NAMES.iter().enumerate() {
        let us = (op_ticks[i] * 1_000_000) / tick_res;
        psp_ml::dprintln!("  [{}] {}: {}us", i, name, us);
    }

    let json = bench::format_results(&result);
    bench::write_hostfs("host0:/benchmarks.json", json.as_bytes());
    psp_ml::dprintln!("Wrote benchmarks.json to host0:/");
}

// ============================================================================
// Local (host CPU) entry point
// ============================================================================

#[cfg(feature = "local")]
static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

#[cfg(feature = "local")]
fn local_get_tick() -> u64 {
    EPOCH.get().expect("epoch not set").elapsed().as_nanos() as u64
}

#[cfg(feature = "local")]
fn main() {
    EPOCH.set(std::time::Instant::now()).unwrap();
    let tick_res: u64 = 1_000_000_000; // 1 tick = 1 ns

    println!("MNIST benchmark: {} images", NUM_IMAGES);
    let (total_ticks, correct, op_ticks) = run_benchmark(local_get_tick);
    let result = make_result("local", tick_res, total_ticks, correct, &op_ticks);

    let total_us = (total_ticks * 1_000_000) / tick_res;
    println!("  {}ms total, {}us/image, {}/{} correct",
        total_us / 1000, total_us / NUM_IMAGES as u64, correct, NUM_IMAGES);

    for (i, name) in generated::OP_NAMES.iter().enumerate() {
        let us = (op_ticks[i] * 1_000_000) / tick_res;
        let pct = if total_us > 0 { (us * 100) / total_us } else { 0 };
        println!("  [{i}] {name}: {us}us ({pct}%)");
    }

    let json = bench::format_results(&result);
    let out_path = concat!(env!("CARGO_MANIFEST_DIR"), "/benchmarks.json");
    std::fs::write(out_path, json.as_bytes()).expect("failed to write benchmarks.json");
    println!("Wrote {out_path}");
}
