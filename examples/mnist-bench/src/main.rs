#![cfg_attr(not(feature = "local"), no_std)]
#![cfg_attr(not(feature = "local"), no_main)]

#[cfg(not(feature = "local"))]
use core::ffi::c_void;
#[cfg(not(feature = "local"))]
use psp::sys::{
    sceIoClose, sceIoOpen, sceIoWrite, sceRtcGetCurrentTick, sceRtcGetTickResolution,
    IoOpenFlags,
};

#[cfg(not(feature = "local"))]
psp_ml::module!("mnist_bench", 1, 0);

mod generated;

// Embed MNIST test dataset (includes headers)
static IMAGES_RAW: &[u8] = include_bytes!("../t10k-images-idx3-ubyte");
static LABELS_RAW: &[u8] = include_bytes!("../t10k-labels-idx1-ubyte");

const NUM_IMAGES: usize = 100;
const KERNEL_TYPE: &str = "vfpu";

/// Extract image at given index and normalize to f32 [0.0, 1.0]
fn get_image(index: usize, images: &[u8]) -> [f32; 784] {
    let mut img = [0.0f32; 784];
    let offset = index * 784;
    for i in 0..784 {
        img[i] = images[offset + i] as f32 / 255.0;
    }
    img
}

// ============================================================================
// Benchmark result collection (shared between PSP and local)
// ============================================================================

struct BenchResult {
    total_us: u64,
    per_image_us: u64,
    correct: u32,
    total: u32,
    op_ticks: [u64; generated::NUM_OPS],
    op_profile: [psp_ml::profiler::OpProfileStats; generated::NUM_OPS],
}

/// Run the timed benchmark and return results.
///
/// `get_tick` returns a monotonic tick value; `tick_res` is ticks per second.
fn run_benchmark(get_tick: fn() -> u64, tick_res: u64) -> BenchResult {
    let images = &IMAGES_RAW[16..];
    let labels = &LABELS_RAW[8..];

    let mut predictions = [0u8; NUM_IMAGES];
    let mut op_ticks = [0u64; generated::NUM_OPS];

    let start = get_tick();

    for i in 0..NUM_IMAGES {
        let img = get_image(i, images);
        let mut output = [0.0f32; generated::OUTPUT_SIZE];
        generated::forward_timed(&img, &mut output, &mut op_ticks, get_tick);

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

    let elapsed_ticks = get_tick() - start;

    let mut correct = 0u32;
    for i in 0..NUM_IMAGES {
        if predictions[i] == labels[i] {
            correct += 1;
        }
    }

    let total_us = (elapsed_ticks * 1_000_000) / tick_res;
    let per_image_us = total_us / NUM_IMAGES as u64;

    BenchResult {
        total_us,
        per_image_us,
        correct,
        total: NUM_IMAGES as u32,
        op_ticks,
        op_profile: [psp_ml::profiler::OpProfileStats::zero(); generated::NUM_OPS],
    }
}

/// Run benchmark with hardware profiling counters per sub-op.
#[cfg(feature = "profile")]
fn run_profiled_benchmark(get_tick: fn() -> u64, tick_res: u64) -> BenchResult {
    let images = &IMAGES_RAW[16..];
    let labels = &LABELS_RAW[8..];

    let mut predictions = [0u8; NUM_IMAGES];
    let mut op_ticks = [0u64; generated::NUM_OPS];
    let mut op_profile = [psp_ml::profiler::OpProfileStats::zero(); generated::NUM_OPS];

    let start = get_tick();

    for i in 0..NUM_IMAGES {
        let img = get_image(i, images);
        let mut output = [0.0f32; generated::OUTPUT_SIZE];
        generated::forward_profiled(&img, &mut output, &mut op_ticks, &mut op_profile, get_tick);

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

    let elapsed_ticks = get_tick() - start;

    let mut correct = 0u32;
    for i in 0..NUM_IMAGES {
        if predictions[i] == labels[i] {
            correct += 1;
        }
    }

    let total_us = (elapsed_ticks * 1_000_000) / tick_res;
    let per_image_us = total_us / NUM_IMAGES as u64;

    BenchResult {
        total_us,
        per_image_us,
        correct,
        total: NUM_IMAGES as u32,
        op_ticks,
        op_profile,
    }
}

// ============================================================================
// no_std JSON writer (works on both PSP and host)
// ============================================================================

/// Minimal JSON formatter that writes into a fixed-size byte buffer.
struct JsonBuf {
    buf: [u8; 16384],
    pos: usize,
}

impl JsonBuf {
    fn new() -> Self {
        JsonBuf {
            buf: [0u8; 16384],
            pos: 0,
        }
    }

    fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.pos]
    }

    fn push_byte(&mut self, b: u8) {
        if self.pos < self.buf.len() {
            self.buf[self.pos] = b;
            self.pos += 1;
        }
    }

    fn push_str(&mut self, s: &str) {
        for &b in s.as_bytes() {
            self.push_byte(b);
        }
    }

    fn push_u64(&mut self, mut val: u64) {
        if val == 0 {
            self.push_byte(b'0');
            return;
        }
        let start = self.pos;
        while val > 0 {
            self.push_byte(b'0' + (val % 10) as u8);
            val /= 10;
        }
        // Reverse the digits in-place
        let end = self.pos;
        let mut i = start;
        let mut j = end - 1;
        while i < j {
            self.buf.swap(i, j);
            i += 1;
            j -= 1;
        }
    }

    fn push_u32(&mut self, val: u32) {
        self.push_u64(val as u64);
    }
}

/// Write benchmark results as JSON into a buffer.
fn format_json(result: &BenchResult, tick_res: u64) -> JsonBuf {
    let mut j = JsonBuf::new();
    j.push_str("{\n");

    j.push_str("  \"model\": \"mnist_cnn\",\n");

    j.push_str("  \"config\": {\n");
    j.push_str("    \"kernel_type\": \"");
    j.push_str(KERNEL_TYPE);
    j.push_str("\"\n");
    j.push_str("  },\n");

    j.push_str("  \"inference\": {\n");
    j.push_str("    \"num_images\": ");
    j.push_u32(result.total);
    j.push_str(",\n");
    j.push_str("    \"total_us\": ");
    j.push_u64(result.total_us);
    j.push_str(",\n");
    j.push_str("    \"per_image_us\": ");
    j.push_u64(result.per_image_us);
    j.push_str(",\n");
    j.push_str("    \"correct\": ");
    j.push_u32(result.correct);
    j.push_str(",\n");
    j.push_str("    \"total\": ");
    j.push_u32(result.total);
    j.push_str("\n");
    j.push_str("  },\n");

    j.push_str("  \"ops\": [\n");
    for (idx, name) in generated::OP_NAMES.iter().enumerate() {
        let op_us = (result.op_ticks[idx] * 1_000_000) / tick_res;
        j.push_str("    { \"index\": ");
        j.push_u32(idx as u32);
        j.push_str(", \"name\": \"");
        j.push_str(name);
        j.push_str("\", \"total_us\": ");
        j.push_u64(op_us);
        j.push_str(", \"calls\": ");
        j.push_u32(result.total);

        let p = &result.op_profile[idx];
        j.push_str(", \"profile\": { ");
        j.push_str("\"systemck\": "); j.push_u64(p.systemck);
        j.push_str(", \"cpuck\": "); j.push_u64(p.cpuck);
        j.push_str(", \"internal\": "); j.push_u64(p.internal);
        j.push_str(", \"memory\": "); j.push_u64(p.memory);
        j.push_str(", \"copz\": "); j.push_u64(p.copz);
        j.push_str(", \"vfpu_stalls\": "); j.push_u64(p.vfpu_stalls);
        j.push_str(", \"sleep\": "); j.push_u64(p.sleep);
        j.push_str(", \"bus_access\": "); j.push_u64(p.bus_access);
        j.push_str(", \"uncached_load\": "); j.push_u64(p.uncached_load);
        j.push_str(", \"uncached_store\": "); j.push_u64(p.uncached_store);
        j.push_str(", \"cached_load\": "); j.push_u64(p.cached_load);
        j.push_str(", \"cached_store\": "); j.push_u64(p.cached_store);
        j.push_str(", \"i_miss\": "); j.push_u64(p.i_miss);
        j.push_str(", \"d_miss\": "); j.push_u64(p.d_miss);
        j.push_str(", \"d_writeback\": "); j.push_u64(p.d_writeback);
        j.push_str(", \"cop0_inst\": "); j.push_u64(p.cop0_inst);
        j.push_str(", \"fpu_inst\": "); j.push_u64(p.fpu_inst);
        j.push_str(", \"vfpu_inst\": "); j.push_u64(p.vfpu_inst);
        j.push_str(", \"local_bus\": "); j.push_u64(p.local_bus);
        j.push_str(" }");

        j.push_str(" }");
        if idx + 1 < generated::NUM_OPS {
            j.push_byte(b',');
        }
        j.push_byte(b'\n');
    }
    j.push_str("  ]\n");

    j.push_str("}\n");
    j
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

    psp_ml::dprintln!("MNIST Inference Benchmark");
    psp_ml::dprintln!("=========================");
    psp_ml::dprintln!("");

    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;

    generated::init();

    #[cfg(feature = "profile")]
    psp_ml::dprintln!("Running PROFILED inference on {} images...", NUM_IMAGES);
    #[cfg(not(feature = "profile"))]
    psp_ml::dprintln!("Running inference on {} images...", NUM_IMAGES);

    #[cfg(feature = "profile")]
    let result = run_profiled_benchmark(get_tick, tick_res);
    #[cfg(not(feature = "profile"))]
    let result = run_benchmark(get_tick, tick_res);

    psp_ml::dprintln!("");
    psp_ml::dprintln!("Results:");
    psp_ml::dprintln!("  Total time: {} ms", result.total_us / 1000);
    psp_ml::dprintln!("  Per image:  {} us", result.per_image_us);
    psp_ml::dprintln!(
        "  Accuracy:   {}/{} ({}%)",
        result.correct,
        result.total,
        (result.correct * 100) / result.total
    );

    psp_ml::dprintln!("");
    psp_ml::dprintln!("Per-op breakdown:");
    for (idx, name) in generated::OP_NAMES.iter().enumerate() {
        let op_us = (result.op_ticks[idx] * 1_000_000) / tick_res;
        let p = &result.op_profile[idx];
        psp_ml::dprintln!(
            "  [{}] {}: {} us | cpuck={} d_miss={} vfpu_inst={} mem_stall={}",
            idx, name, op_us, p.cpuck, p.d_miss, p.vfpu_inst, p.memory
        );
    }

    // Write JSON to host0:/benchmarks.json
    let json = format_json(&result, tick_res);
    let path = b"host0:/benchmarks.json\0";
    let fd = unsafe {
        sceIoOpen(
            path.as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o644,
        )
    };
    if fd.0 >= 0 {
        unsafe {
            sceIoWrite(fd, json.as_bytes().as_ptr() as *const c_void, json.as_bytes().len());
            sceIoClose(fd);
        }
        psp_ml::dprintln!("");
        psp_ml::dprintln!("Wrote benchmarks.json to host0:/");
    } else {
        psp_ml::dprintln!("");
        psp_ml::dprintln!("Warning: could not write to host0:/ (psplink not connected?)");
    }
}

// ============================================================================
// Local (host CPU) entry point
// ============================================================================

#[cfg(feature = "local")]
static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

#[cfg(feature = "local")]
fn local_get_tick() -> u64 {
    EPOCH
        .get()
        .expect("epoch not set")
        .elapsed()
        .as_nanos() as u64
}

#[cfg(feature = "local")]
fn main() {
    EPOCH.set(std::time::Instant::now()).unwrap();
    // Tick resolution: 1 tick = 1 nanosecond -> 1_000_000_000 ticks/sec
    let tick_res: u64 = 1_000_000_000;

    println!("MNIST Inference Benchmark");
    println!("=========================");
    println!();

    generated::init();

    println!("Running inference on {} images...", NUM_IMAGES);
    let result = run_benchmark(local_get_tick, tick_res);

    println!();
    println!("Results:");
    println!("  Total time: {} ms", result.total_us / 1000);
    println!("  Per image:  {} us", result.per_image_us);
    println!(
        "  Accuracy:   {}/{} ({:.1}%)",
        result.correct,
        result.total,
        (result.correct as f32 / result.total as f32) * 100.0
    );

    println!();
    println!("Per-op breakdown:");
    for (idx, name) in generated::OP_NAMES.iter().enumerate() {
        let op_us = (result.op_ticks[idx] * 1_000_000) / tick_res;
        let pct = if result.total_us > 0 {
            (op_us * 100) / result.total_us
        } else {
            0
        };
        let p = &result.op_profile[idx];
        println!(
            "  [{idx}] {name}: {op_us} us ({pct}%) | cpuck={} d_miss={} vfpu_inst={} mem_stall={}",
            p.cpuck, p.d_miss, p.vfpu_inst, p.memory
        );
    }

    // Write JSON to benchmarks.json in the example directory
    let json = format_json(&result, tick_res);
    let out_path = concat!(env!("CARGO_MANIFEST_DIR"), "/benchmarks.json");
    std::fs::write(out_path, json.as_bytes()).expect("failed to write benchmarks.json");
    println!();
    println!("Wrote {}", out_path);
}
