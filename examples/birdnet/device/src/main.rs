#![cfg_attr(not(feature = "local"), no_std)]
#![cfg_attr(not(feature = "local"), no_main)]

#[cfg(not(feature = "local"))]
use core::ffi::c_void;
#[cfg(not(feature = "local"))]
use psp::sys::{
    sceIoClose, sceIoOpen, sceIoWrite, sceRtcGetCurrentTick, sceRtcGetTickResolution, IoOpenFlags,
};

#[cfg(not(feature = "local"))]
psp_rt::module!("birdnet", 1, 0);

mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated.rs"));
}

const INPUT_SAMPLES: usize = 144000;
const OUTPUT_CLASSES: usize = 6522;
const TOP_K: usize = 5;

// Audio pre-converted to f32 by build.rs (avoids 562KB stack allocation)
#[repr(C, align(16))]
struct AlignedAudio([u8; INPUT_SAMPLES * 4]);
static AUDIO_F32_BYTES: AlignedAudio =
    AlignedAudio(*include_bytes!(concat!(env!("OUT_DIR"), "/audio_f32.bin")));

fn audio_input() -> &'static [f32; INPUT_SAMPLES] {
    unsafe { &*(AUDIO_F32_BYTES.0.as_ptr() as *const [f32; INPUT_SAMPLES]) }
}

// BirdNET species labels, one per line, index-aligned with the output vector.
static LABELS: &str = include_str!("../../../../models/birdnet/labels/en_us.txt");

fn label(index: usize) -> &'static str {
    LABELS.lines().nth(index).unwrap_or("?")
}

/// Indices of the k largest scores, by repeated argmax (k * 6522 is cheap).
fn top_k(output: &[f32; OUTPUT_CLASSES]) -> [usize; TOP_K] {
    let mut picked = [usize::MAX; TOP_K];
    for slot in 0..TOP_K {
        let mut best = usize::MAX;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in output.iter().enumerate() {
            if v > best_v && !picked[..slot].contains(&i) {
                best = i;
                best_v = v;
            }
        }
        picked[slot] = best;
    }
    picked
}

// ============================================================================
// Benchmark result collection (shared between PSP and local)
// ============================================================================

struct BenchResult {
    total_us: u64,
    op_ticks: [u64; generated::NUM_OPS],
}

/// Run timed inference and return results.
///
/// `get_tick` returns a monotonic tick value; `tick_res` is ticks per second.
/// Output is written into the `output` buffer.
fn run_benchmark(
    get_tick: fn() -> u64,
    tick_res: u64,
    output: &mut [f32; OUTPUT_CLASSES],
) -> BenchResult {
    let input = audio_input();
    let mut op_ticks = [0u64; generated::NUM_OPS];

    let start = get_tick();
    generated::forward_timed(input, output, &mut op_ticks, get_tick);
    let elapsed_ticks = get_tick() - start;

    let total_us = (elapsed_ticks * 1_000_000) / tick_res;

    BenchResult { total_us, op_ticks }
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

    j.push_str("  \"model\": \"birdnet_v2.4_int8\",\n");

    j.push_str("  \"inference\": {\n");
    j.push_str("    \"total_us\": ");
    j.push_u64(result.total_us);
    j.push_str("\n  },\n");

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

/// Format raw output scores as text, one per line, 6 decimal places.
fn write_results_to_buf(output: &[f32; OUTPUT_CLASSES], buf: &mut [u8]) -> usize {
    let mut pos = 0;

    for &val in output.iter() {
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
            let mut i = start;
            let mut j = pos - 1;
            while i < j {
                buf.swap(i, j);
                i += 1;
                j -= 1;
            }
        }

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
        let mut fi = pos - 1;
        while frac > 0 && fi >= frac_start {
            buf[fi] = b'0' + (frac % 10) as u8;
            frac /= 10;
            if fi == 0 {
                break;
            }
            fi -= 1;
        }

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
    // forward_timed calls this between ops; use it as a progress heartbeat so
    // the host's per-event timeout never starves during a long inference.
    static mut CALLS: u32 = 0;
    unsafe {
        CALLS += 1;
        if CALLS % 64 == 0 {
            psp_rt::dprint!(".");
        }
    }
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
        psp_rt::dprintln!("Warning: could not write file (psplink not connected?)");
    }
}

#[cfg(not(feature = "local"))]
fn app_main() {
    psp::enable_home_button();

    // Run at full clock: psplink boots at 222/111 MHz (measured by
    // examples/roofline; 333/166 is a 1.5x speedup).
    unsafe { psp::sys::scePowerSetClockFrequency(333, 333, 166) };

    psp_rt::dprintln!("birdnet: started, loading weights...");
    generated::init();

    psp_rt::dprintln!("Running BirdNET (int8)...");

    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;
    static mut OUTPUT_BUF: [f32; OUTPUT_CLASSES] = [0.0f32; OUTPUT_CLASSES];
    let output = unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT_BUF) };
    let result = run_benchmark(get_tick, tick_res, output);

    psp_rt::dprintln!("");
    psp_rt::dprintln!("Total time: {} ms", result.total_us / 1000);
    psp_rt::dprintln!("");
    psp_rt::dprintln!("Top-{} species:", TOP_K);
    for &i in top_k(output).iter() {
        psp_rt::dprintln!("  [{}] {} raw={}", i, label(i), output[i]);
    }

    // Raw scores for host-side comparison against the Python golden run.
    static mut RESULT_BUF: [u8; 131072] = [0u8; 131072];
    let result_buf = unsafe { &mut *core::ptr::addr_of_mut!(RESULT_BUF) };
    let len = write_results_to_buf(output, result_buf);
    write_file(b"host0:/results.txt\0", &result_buf[..len]);

    let json = format_json(&result, tick_res);
    write_file(b"host0:/benchmarks.json\0", json.as_bytes());
    psp_rt::dprintln!("");
    psp_rt::dprintln!("Wrote results.txt and benchmarks.json");
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
    let tick_res: u64 = 1_000_000_000;

    println!("BirdNET Inference (local, int8-as-f32)");
    println!("======================================");

    generated::init();

    // Debug tap mode: dump every op's output tensor to tap/t<ID>.bin for
    // layer-by-layer comparison against a TFLite reference run.
    if std::env::var("BIRDNET_TAP").is_ok() {
        let tap_dir = format!("{}/tap", env!("CARGO_MANIFEST_DIR"));
        std::fs::create_dir_all(&tap_dir).unwrap();
        let mut output = Box::new([0.0f32; OUTPUT_CLASSES]);
        let mut manifest = String::new();
        generated::forward_debug(
            audio_input(),
            &mut output,
            &mut |op_idx, tensor_id, values| {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{tap_dir}/t{tensor_id}.bin"), bytes).unwrap();
                manifest.push_str(&format!("{op_idx} {tensor_id} {}\n", values.len()));
            },
        );
        std::fs::write(format!("{tap_dir}/manifest.txt"), manifest).unwrap();
        println!("Wrote taps to {tap_dir}");
        return;
    }

    let mut output = Box::new([0.0f32; OUTPUT_CLASSES]);
    let result = run_benchmark(local_get_tick, tick_res, &mut output);

    println!();
    println!("Total time: {} ms", result.total_us / 1000);
    println!();
    println!("Top-{TOP_K} species:");
    for &i in top_k(&output).iter() {
        println!("  [{i:4}] raw={:9.4}  {}", output[i], label(i));
    }

    let out_dir = env!("CARGO_MANIFEST_DIR");

    let mut result_buf = vec![0u8; 131072];
    let len = write_results_to_buf(&output, &mut result_buf);
    let results_path = format!("{}/results.txt", out_dir);
    std::fs::write(&results_path, &result_buf[..len]).expect("failed to write results.txt");

    let json = format_json(&result, tick_res);
    let json_path = format!("{}/benchmarks.json", out_dir);
    std::fs::write(&json_path, json.as_bytes()).expect("failed to write benchmarks.json");
    println!();
    println!("Wrote {results_path} and {json_path}");
}
