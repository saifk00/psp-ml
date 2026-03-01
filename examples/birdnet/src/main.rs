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
psp_ml::module!("birdnet", 1, 0);

mod generated;

const INPUT_SAMPLES: usize = 144000;
const OUTPUT_CLASSES: usize = 6522;

// Audio pre-converted to f32 by build.rs (avoids 562KB stack allocation)
#[repr(C, align(16))]
struct AlignedAudio([u8; INPUT_SAMPLES * 4]);
static AUDIO_F32_BYTES: AlignedAudio =
    AlignedAudio(*include_bytes!(concat!(env!("OUT_DIR"), "/audio_f32.bin")));

fn audio_input() -> &'static [f32; INPUT_SAMPLES] {
    unsafe { &*(AUDIO_F32_BYTES.0.as_ptr() as *const [f32; INPUT_SAMPLES]) }
}

// ============================================================================
// Benchmark result collection (shared between PSP and local)
// ============================================================================

struct BenchResult {
    total_us: u64,
    per_image_us: u64,
    op_ticks: [u64; generated::NUM_OPS],
    output: [f32; OUTPUT_CLASSES],
}

/// Run timed inference and return results.
///
/// `get_tick` returns a monotonic tick value; `tick_res` is ticks per second.
fn run_benchmark(get_tick: fn() -> u64, tick_res: u64) -> BenchResult {
    let input = audio_input();
    let mut op_ticks = [0u64; generated::NUM_OPS];

    let start = get_tick();
    let output = generated::forward_timed(input, &mut op_ticks, get_tick);
    let elapsed_ticks = get_tick() - start;

    let total_us = (elapsed_ticks * 1_000_000) / tick_res;

    BenchResult {
        total_us,
        per_image_us: total_us, // single inference
        op_ticks,
        output,
    }
}

// ============================================================================
// no_std JSON writer (works on both PSP and host)
// ============================================================================

/// Minimal JSON formatter that writes into a fixed-size byte buffer.
struct JsonBuf {
    buf: [u8; 32768],
    pos: usize,
}

impl JsonBuf {
    fn new() -> Self {
        JsonBuf {
            buf: [0u8; 32768],
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

    j.push_str("  \"model\": \"birdnet_v2.4\",\n");

    j.push_str("  \"inference\": {\n");
    j.push_str("    \"total_us\": ");
    j.push_u64(result.total_us);
    j.push_str(",\n");
    j.push_str("    \"per_inference_us\": ");
    j.push_u64(result.per_image_us);
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
// Results writer — dump raw output scores as text
// ============================================================================

/// Write output scores as text, one per line. Uses a static buffer since
/// 6522 floats × ~15 chars ≈ 100KB.
fn write_results_to_buf(output: &[f32; OUTPUT_CLASSES], buf: &mut [u8]) -> usize {
    let mut pos = 0;

    for &val in output.iter() {
        // Format float with 6 decimal places using integer math
        let negative = val < 0.0;
        let abs_val = if negative { -val } else { val };
        let integer_part = abs_val as u64;
        let frac_part = ((abs_val - integer_part as f32) * 1_000_000.0) as u64;

        if negative && pos < buf.len() {
            buf[pos] = b'-';
            pos += 1;
        }

        // Write integer part
        let start = pos;
        if integer_part == 0 {
            if pos < buf.len() {
                buf[pos] = b'0';
                pos += 1;
            }
        } else {
            let mut tmp = integer_part;
            while tmp > 0 && pos < buf.len() {
                buf[pos] = b'0' + (tmp % 10) as u8;
                tmp /= 10;
                pos += 1;
            }
            // Reverse integer digits
            let mut i = start;
            let mut j = pos - 1;
            while i < j {
                buf.swap(i, j);
                i += 1;
                j -= 1;
            }
        }

        // Decimal point
        if pos < buf.len() {
            buf[pos] = b'.';
            pos += 1;
        }

        // Write fractional part (6 digits, zero-padded)
        let mut frac = frac_part;
        let frac_start = pos;
        for _ in 0..6 {
            if pos < buf.len() {
                buf[pos] = b'0';
                pos += 1;
            }
        }
        // Fill from right to left
        let mut fi = pos - 1;
        while frac > 0 && fi >= frac_start {
            buf[fi] = b'0' + (frac % 10) as u8;
            frac /= 10;
            if fi == 0 {
                break;
            }
            fi -= 1;
        }

        // Newline
        if pos < buf.len() {
            buf[pos] = b'\n';
            pos += 1;
        }
    }

    pos
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
fn write_file(path: &[u8], data: &[u8]) {
    let fd = unsafe {
        sceIoOpen(
            path.as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o644,
        )
    };
    if fd.0 >= 0 {
        unsafe {
            sceIoWrite(fd, data.as_ptr() as *const c_void, data.len());
            sceIoClose(fd);
        }
    } else {
        psp_ml::dprintln!("Warning: could not write file (psplink not connected?)");
    }
}

#[cfg(not(feature = "local"))]
fn app_main() {
    psp::enable_home_button();

    generated::init();

    psp_ml::dprintln!("Running BirdNET...");

    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;
    let result = run_benchmark(get_tick, tick_res);

    psp_ml::dprintln!("");
    psp_ml::dprintln!("Results:");
    psp_ml::dprintln!("  Total time: {} ms", result.total_us / 1000);
    psp_ml::dprintln!("");

    psp_ml::dprintln!("Per-op breakdown:");
    for (idx, name) in generated::OP_NAMES.iter().enumerate() {
        let op_us = (result.op_ticks[idx] * 1_000_000) / tick_res;
        psp_ml::dprintln!("  [{}] {}: {} us", idx, name, op_us);
    }

    // Write benchmarks.json
    let json = format_json(&result, tick_res);
    write_file(b"host0:/benchmarks.json\0", json.as_bytes());
    psp_ml::dprintln!("");
    psp_ml::dprintln!("Wrote benchmarks.json");

    // Write results.txt (raw scores)
    static mut RESULT_BUF: [u8; 131072] = [0u8; 131072]; // 128KB
    let len = write_results_to_buf(&result.output, unsafe { &mut RESULT_BUF });
    write_file(b"host0:/results.txt\0", unsafe { &RESULT_BUF[..len] });
    psp_ml::dprintln!("Wrote results.txt ({} bytes)", len);
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
    let tick_res: u64 = 1_000_000_000;

    println!("BirdNET Inference Benchmark");
    println!("===========================");
    println!();

    generated::init();

    println!("Running inference...");
    let result = run_benchmark(local_get_tick, tick_res);

    println!();
    println!("Results:");
    println!("  Total time: {} ms", result.total_us / 1000);

    println!();
    println!("Per-op breakdown:");
    for (idx, name) in generated::OP_NAMES.iter().enumerate() {
        let op_us = (result.op_ticks[idx] * 1_000_000) / tick_res;
        let pct = if result.total_us > 0 {
            (op_us * 100) / result.total_us
        } else {
            0
        };
        println!("  [{idx}] {name}: {op_us} us ({pct}%)");
    }

    // Write benchmarks.json
    let json = format_json(&result, tick_res);
    let out_dir = env!("CARGO_MANIFEST_DIR");
    let json_path = format!("{}/benchmarks.json", out_dir);
    std::fs::write(&json_path, json.as_bytes()).expect("failed to write benchmarks.json");
    println!();
    println!("Wrote {}", json_path);

    // Write results.txt (raw scores)
    let mut result_buf = vec![0u8; 131072];
    let len = write_results_to_buf(&result.output, &mut result_buf);
    let results_path = format!("{}/results.txt", out_dir);
    std::fs::write(&results_path, &result_buf[..len]).expect("failed to write results.txt");
    println!("Wrote {}", results_path);
}
