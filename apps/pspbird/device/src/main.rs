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

// dead_code: with the custom frontend the backbone's OUTPUT_SIZE/debug entry
// points go unused (main.rs sizes buffers from classes.rs's OUTPUT_CLASSES).
#[allow(dead_code)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated.rs"));
}

// Custom frontend (BIRDNET_CUSTOM_FRONTEND=1): `generated` above is only the
// conv backbone (severed at the branch-merge CONCAT), and this module is the
// spectrogram frontend — normalisation, strided-view STFTs, banded mel
// projections — built by build.rs with PspModelBuilder.
#[cfg(feature = "custom-frontend")]
#[allow(dead_code)]
mod frontend {
    include!(concat!(env!("OUT_DIR"), "/frontend/custom_frontend.rs"));
}

#[cfg(feature = "custom-frontend")]
const N_BANKS: usize = 96;
#[cfg(feature = "custom-frontend")]
const N_WINDOWS: usize = 511;
#[cfg(feature = "custom-frontend")]
const MEL_LEN: usize = N_BANKS * N_WINDOWS;
#[cfg(feature = "custom-frontend")]
static mut FRONTEND_MEL_2048: [f32; MEL_LEN] = [0.0; MEL_LEN];
#[cfg(feature = "custom-frontend")]
static mut FRONTEND_MEL_1024: [f32; MEL_LEN] = [0.0; MEL_LEN];
#[cfg(feature = "custom-frontend")]
static mut BACKBONE_INPUT: [f32; MEL_LEN * 2] = [0.0; MEL_LEN * 2];

const INPUT_SAMPLES: usize = 144000;
// 6522 for the full model, or the surviving species count when TOPK is set.
// Emitted by build.rs so it always matches the model psp-tc actually compiled.
include!(concat!(env!("OUT_DIR"), "/classes.rs"));
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
// build.rs writes either the full en_us.txt or the pruned subset, so this stays
// aligned with OUTPUT_CLASSES.
static LABELS: &str = include_str!(concat!(env!("OUT_DIR"), "/labels.txt"));

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
    /// Time inside the custom frontend + input assembly (0 without the
    /// `custom-frontend` feature, where nothing reads it). Included in
    /// `total_us`.
    #[cfg_attr(not(feature = "custom-frontend"), allow(dead_code))]
    frontend_us: u64,
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
    let mut op_ticks = [0u64; generated::NUM_OPS];

    let start = get_tick();

    // Custom frontend: run the builder-generated module, then assemble the
    // backbone's [1, 96, 511, 2] input from its bank-major outputs — each
    // branch mel-axis reversed (the model's REVERSE_V2, axis 2) and
    // channel-interleaved (2048 first), which is what the severed model's
    // CONCAT consumed. Folding this shuffle into the frontend itself is
    // deliberately deferred.
    #[cfg(feature = "custom-frontend")]
    let (input, frontend_us) = {
        assert!(frontend::OUTPUT_SIZES == [MEL_LEN, MEL_LEN]);
        let m2048 = unsafe { &mut *core::ptr::addr_of_mut!(FRONTEND_MEL_2048) };
        let m1024 = unsafe { &mut *core::ptr::addr_of_mut!(FRONTEND_MEL_1024) };
        frontend::forward(audio_input(), m2048, m1024);
        let bb = unsafe { &mut *core::ptr::addr_of_mut!(BACKBONE_INPUT) };
        for q in 0..N_BANKS {
            let src = (N_BANKS - 1 - q) * N_WINDOWS;
            for t in 0..N_WINDOWS {
                let i = (q * N_WINDOWS + t) * 2;
                bb[i] = m2048[src + t];
                bb[i + 1] = m1024[src + t];
            }
        }
        let us = (get_tick() - start) * 1_000_000 / tick_res;
        (unsafe { &*core::ptr::addr_of!(BACKBONE_INPUT) }, us)
    };
    #[cfg(not(feature = "custom-frontend"))]
    let (input, frontend_us) = (audio_input(), 0u64);
    #[cfg(not(feature = "hwprofile"))]
    generated::forward_timed(input, output, &mut op_ticks, get_tick);
    // Hardware-counter pass. The generated `forward_profiled` brackets every
    // sub-op with the Allegrex profiler MMIO (clear -> enable -> kernel ->
    // disable -> read), which needs the psp_ml_kernel plugin installed or the
    // counters read zero.
    #[cfg(feature = "hwprofile")]
    {
        static mut PROFILE: [psp_rt::profiler::OpProfileStats; generated::NUM_OPS] =
            [psp_rt::profiler::OpProfileStats::zero(); generated::NUM_OPS];
        generated::forward_profiled(
            input,
            output,
            &mut op_ticks,
            unsafe { &mut *core::ptr::addr_of_mut!(PROFILE) },
            get_tick,
        );
        report_hw_profile(unsafe { &*core::ptr::addr_of!(PROFILE) }, &op_ticks);
    }
    let elapsed_ticks = get_tick() - start;

    let total_us = (elapsed_ticks * 1_000_000) / tick_res;

    BenchResult {
        total_us,
        frontend_us,
        op_ticks,
    }
}

/// Print the ops that cost the most CPU cycles, with the counters that say why.
///
/// Cycles rather than wall time because these come from the hardware counter
/// and are immune to the timing drift that per-op tick accounting has had.
#[cfg(feature = "hwprofile")]
fn report_hw_profile(
    profile: &[psp_rt::profiler::OpProfileStats; generated::NUM_OPS],
    _op_ticks: &[u64; generated::NUM_OPS],
) {
    let total: u64 = profile.iter().map(|p| p.cpuck).sum();
    psp_rt::dprintln!("");
    psp_rt::dprintln!("=== hardware profile: {} total cpu cycles ===", total);
    psp_rt::dprintln!("op  %cyc  Mcycles  i-miss  d-miss  cop0-stall  mem-stall  name");

    // Selection sort over indices: no allocator, and 15 passes over 271 entries
    // is nothing next to the inference itself.
    let mut used = [false; generated::NUM_OPS];
    for _ in 0..20 {
        let mut best = usize::MAX;
        for i in 0..generated::NUM_OPS {
            if !used[i] && (best == usize::MAX || profile[i].cpuck > profile[best].cpuck) {
                best = i;
            }
        }
        if best == usize::MAX || profile[best].cpuck == 0 {
            break;
        }
        used[best] = true;
        let p = &profile[best];
        let pct = if total > 0 { p.cpuck * 100 / total } else { 0 };
        psp_rt::dprintln!(
            "{:3} {:3}% {:8} {:7} {:7} {:11} {:10} {}",
            best,
            pct,
            p.cpuck / 1_000_000,
            p.i_miss,
            p.d_miss,
            p.copz,
            p.memory,
            generated::OP_NAMES[best]
        );
    }

    // Aggregate by kernel name: where the whole run's cycles and stalls go,
    // in milliseconds at 333 MHz. The per-op table above shows the flat
    // profile; this is the sum that actually answers "how much time is
    // quantization / memory waits".
    const MAX_KINDS: usize = 32;
    let mut names: [&str; MAX_KINDS] = [""; MAX_KINDS];
    let mut agg = [[0u64; 4]; MAX_KINDS]; // cpuck, mem-stall, cop0-stall, d-miss
    let mut n_kinds = 0usize;
    for i in 0..generated::NUM_OPS {
        let name = generated::OP_NAMES[i];
        let mut k = n_kinds;
        for (j, n) in names.iter().enumerate().take(n_kinds) {
            if *n == name {
                k = j;
                break;
            }
        }
        if k == n_kinds && n_kinds < MAX_KINDS {
            names[k] = name;
            n_kinds += 1;
        }
        if k < MAX_KINDS {
            agg[k][0] += profile[i].cpuck;
            agg[k][1] += profile[i].memory;
            agg[k][2] += profile[i].copz;
            agg[k][3] += profile[i].d_miss;
        }
    }
    const KHZ: u64 = 333_000; // cycles per millisecond
    let (mut tot_mem, mut tot_cop) = (0u64, 0u64);
    for k in 0..n_kinds {
        tot_mem += agg[k][1];
        tot_cop += agg[k][2];
    }
    psp_rt::dprintln!("");
    psp_rt::dprintln!(
        "=== by kernel (ms at 333 MHz): total {} ms, mem-stall {} ms, cop0-stall {} ms ===",
        total / KHZ,
        tot_mem / KHZ,
        tot_cop / KHZ
    );
    psp_rt::dprintln!("kernel                 ms   mem-stall  cop0-stall   d-miss(k)");
    let mut printed = [false; MAX_KINDS];
    for _ in 0..n_kinds {
        let mut best = usize::MAX;
        for k in 0..n_kinds {
            if !printed[k] && (best == usize::MAX || agg[k][0] > agg[best][0]) {
                best = k;
            }
        }
        if best == usize::MAX {
            break;
        }
        printed[best] = true;
        psp_rt::dprintln!(
            "{:18} {:6} {:9} {:11} {:10}",
            names[best],
            agg[best][0] / KHZ,
            agg[best][1] / KHZ,
            agg[best][2] / KHZ,
            agg[best][3] / 1000
        );
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
    // No heartbeat here. `forward_timed` calls this twice per sub-op, so a
    // periodic USB print lands *inside* the timed region and is attributed to
    // whichever op it interrupts. It existed to keep the host's per-event
    // timeout fed back when inference took 20 s; at ~7 s it is pure overhead.
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
    psp_rt::enable_home_button();

    // Run at full clock: psplink boots at 222/111 MHz (measured by
    // examples/roofline; 333/166 is a 1.5x speedup).
    unsafe { psp::sys::scePowerSetClockFrequency(333, 333, 166) };

    psp_rt::dprintln!("birdnet: started, loading weights...");
    generated::init();
    #[cfg(feature = "custom-frontend")]
    frontend::init();

    #[cfg(not(feature = "custom-frontend"))]
    psp_rt::dprintln!("Running BirdNET (int8)...");
    #[cfg(feature = "custom-frontend")]
    psp_rt::dprintln!("Running BirdNET (int8, custom frontend)...");

    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;
    static mut OUTPUT_BUF: [f32; OUTPUT_CLASSES] = [0.0f32; OUTPUT_CLASSES];
    let output = unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT_BUF) };
    let result = run_benchmark(get_tick, tick_res, output);

    psp_rt::dprintln!("");
    psp_rt::dprintln!("Total time: {} ms", result.total_us / 1000);
    #[cfg(feature = "custom-frontend")]
    psp_rt::dprintln!("  frontend (incl. assembly): {} ms", result.frontend_us / 1000);
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
    #[cfg(feature = "custom-frontend")]
    if std::env::var("BIRDNET_TAP").is_ok() {
        eprintln!("BIRDNET_TAP is not supported with the custom frontend (tensor ids");
        eprintln!("only line up against TFLite for the whole-model compile); rerun");
        eprintln!("without --features custom-frontend.");
        std::process::exit(2);
    }
    #[cfg(not(feature = "custom-frontend"))]
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
    #[cfg(feature = "custom-frontend")]
    println!("  frontend (incl. assembly): {} ms", result.frontend_us / 1000);
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
