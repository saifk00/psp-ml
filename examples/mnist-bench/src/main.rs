#![no_std]
#![no_main]

use psp::dprintln;
use psp::sys::{sceRtcGetCurrentTick, sceRtcGetTickResolution};

psp::module!("mnist_bench", 1, 0);

mod generated;

// Embed MNIST test dataset (includes headers)
static IMAGES_RAW: &[u8] = include_bytes!("../t10k-images-idx3-ubyte");
static LABELS_RAW: &[u8] = include_bytes!("../t10k-labels-idx1-ubyte");

const NUM_IMAGES: usize = 100;

fn get_tick() -> u64 {
    let mut tick = 0u64;
    unsafe { sceRtcGetCurrentTick(&mut tick) };
    tick
}

fn ticks_to_us(ticks: u64, resolution: u32) -> u64 {
    (ticks * 1_000_000) / resolution as u64
}

fn ticks_to_ms(ticks: u64, resolution: u32) -> u64 {
    (ticks * 1_000) / resolution as u64
}

/// Extract image at given index and normalize to f32 [0.0, 1.0]
fn get_image(index: usize, images: &[u8]) -> [f32; 784] {
    let mut img = [0.0f32; 784];
    let offset = index * 784;
    for i in 0..784 {
        img[i] = images[offset + i] as f32 / 255.0;
    }
    img
}

fn psp_main() {
    psp::enable_home_button();

    dprintln!("MNIST Inference Benchmark");
    dprintln!("=========================");
    dprintln!("");

    let tick_res = unsafe { sceRtcGetTickResolution() };

    // Skip headers: 16 bytes for images, 8 bytes for labels
    let images = &IMAGES_RAW[16..];
    let labels = &LABELS_RAW[8..];

    // Store predictions for all images
    let mut predictions = [0u8; NUM_IMAGES];

    dprintln!("Running inference on {} images...", NUM_IMAGES);

    let start = get_tick();

    for i in 0..NUM_IMAGES {
        let img = get_image(i, images);
        let output = generated::forward(&img);

        // Argmax to find predicted digit
        let mut max_idx = 0u8;
        let mut max_val = output[0];
        for j in 1..10 {
            if output[j] > max_val {
                max_val = output[j];
                max_idx = j as u8;
            }
        }
        predictions[i] = max_idx;
    }

    let elapsed = get_tick() - start;

    // Calculate accuracy (outside timed region)
    let mut correct = 0u32;
    for i in 0..NUM_IMAGES {
        if predictions[i] == labels[i] {
            correct += 1;
        }
    }

    let elapsed_ms = ticks_to_ms(elapsed, tick_res);
    let elapsed_us = ticks_to_us(elapsed, tick_res);
    let per_image_us = elapsed_us / NUM_IMAGES as u64;
    let accuracy_pct = (correct as f32 / NUM_IMAGES as f32) * 100.0;

    dprintln!("");
    dprintln!("Results:");
    dprintln!("  Total time: {} ms", elapsed_ms);
    dprintln!("  Per image:  {} us", per_image_us);
    dprintln!(
        "  Accuracy:   {}/{} ({:.1}%)",
        correct,
        NUM_IMAGES,
        accuracy_pct
    );

    dprintln!("");
    dprintln!("Done. Press HOME to exit.");

    loop {
        unsafe { psp::sys::sceKernelDelayThread(100_000) };
    }
}
