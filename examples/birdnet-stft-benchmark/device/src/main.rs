//! BirdNET STFT frontend benchmark: dense-gather (the sliced .tflite as
//! TFLite expresses it) vs strided-view (`PspOp::StridedViewStft`).
//!
//! A device PRX contains exactly ONE mode (`--features mode-dense` or
//! `mode-strided`), so its module image, weight blob and arena are that
//! mode's alone and the on-device free-memory numbers attribute cleanly.
//! The `local` build carries both and takes `--mode` at runtime; it also
//! checks both against the TFLite goldens and each other.
//!
//! Every run prints a machine-readable `#stftbench ...` line the host runner
//! parses for its comparison table.

#![cfg_attr(not(feature = "local"), no_std)]
#![cfg_attr(not(feature = "local"), no_main)]

#[cfg(all(
    not(feature = "local"),
    not(any(feature = "mode-dense", feature = "mode-strided"))
))]
compile_error!(
    "device build needs a mode: --no-default-features --features mode-dense (or mode-strided)"
);
#[cfg(all(feature = "mode-dense", feature = "mode-strided"))]
compile_error!("mode-dense and mode-strided are one-per-PRX; build twice instead");

#[cfg(not(feature = "local"))]
use core::ffi::c_void;
#[cfg(not(feature = "local"))]
use psp::sys::{
    sceIoClose, sceIoOpen, sceIoWrite, sceKernelMaxFreeMemSize, sceKernelTotalFreeMemSize,
    scePowerSetClockFrequency, sceRtcGetCurrentTick, sceRtcGetTickResolution, IoOpenFlags,
};

#[cfg(not(feature = "local"))]
psp_rt::module!("stft_bench", 1, 0);

include!(concat!(env!("OUT_DIR"), "/bench_config.rs"));

// dead_code: the benchmark drives forward/forward_timed only, never the
// profiled/debug entry points every generated module carries.
#[cfg(any(feature = "local", feature = "mode-dense"))]
#[allow(dead_code)]
mod dense {
    include!(concat!(env!("OUT_DIR"), "/dense/stft_dense.rs"));
}

#[cfg(any(feature = "local", feature = "mode-strided"))]
#[allow(dead_code)]
mod strided {
    include!(concat!(env!("OUT_DIR"), "/strided/stft_strided.rs"));
}

const N_SAMPLES: usize = 144_000;
const OUT_2048: usize = 511 * 1025;
const OUT_1024: usize = 511 * 513;
/// Timed iterations (after one warmup).
const RUNS: u64 = 3;

/// Normalised signal, bit-identical to the TFLite reference run's tap
/// (slice_stft.py wrote it), staged by build.rs.
static SAMPLES_RAW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/samples.bin"));

static mut INPUT: [f32; N_SAMPLES] = [0.0; N_SAMPLES];
static mut OUT0: [f32; OUT_2048] = [0.0; OUT_2048];
static mut OUT1: [f32; OUT_1024] = [0.0; OUT_1024];

fn decode_input() -> &'static [f32; N_SAMPLES] {
    let input = unsafe { &mut *core::ptr::addr_of_mut!(INPUT) };
    for (i, c) in SAMPLES_RAW.chunks_exact(4).enumerate() {
        input[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    input
}

struct BenchOut {
    avg_us: u64,
}

/// Warm up once, then time RUNS full frontend passes of module `$m`,
/// printing a per-op breakdown of the final pass.
macro_rules! run_frontend {
    ($m:ident, $print:ident, $get_tick:expr, $tick_res:expr) => {{
        let input = decode_input();
        let out0 = unsafe { &mut *core::ptr::addr_of_mut!(OUT0) };
        let out1 = unsafe { &mut *core::ptr::addr_of_mut!(OUT1) };
        assert!($m::OUTPUT_SIZES == [OUT_2048, OUT_1024]);

        let get_tick: fn() -> u64 = $get_tick;
        let tick_res: u64 = $tick_res;
        $m::forward(input, out0, out1); // warmup

        let mut op_ticks = [0u64; $m::NUM_OPS];
        let start = get_tick();
        for _ in 0..RUNS {
            $m::forward_timed(input, out0, out1, &mut op_ticks, get_tick);
        }
        let total_us = (get_tick() - start) * 1_000_000 / tick_res;

        $print!("  per-op (avg over {} runs):", RUNS);
        for (i, name) in $m::OP_NAMES.iter().enumerate() {
            let us = op_ticks[i] * 1_000_000 / tick_res / RUNS;
            if us > 0 {
                $print!("    {:>8} us  {}", us, name);
            }
        }
        BenchOut { avg_us: total_us / RUNS }
    }};
}

// ============================================================================
// Device entry point: one mode per PRX
// ============================================================================

#[cfg(not(feature = "local"))]
fn get_tick() -> u64 {
    let mut tick = 0u64;
    unsafe { sceRtcGetCurrentTick(&mut tick) };
    tick
}

#[cfg(all(not(feature = "local"), feature = "mode-dense"))]
const MODE: &str = "dense_gather";
#[cfg(all(not(feature = "local"), feature = "mode-dense"))]
const MODE_ARENA: usize = DENSE_ARENA_BYTES;
#[cfg(all(not(feature = "local"), feature = "mode-dense"))]
const MODE_BLOB: usize = DENSE_BLOB_BYTES;

#[cfg(all(not(feature = "local"), feature = "mode-strided"))]
const MODE: &str = "strided_view";
#[cfg(all(not(feature = "local"), feature = "mode-strided"))]
const MODE_ARENA: usize = STRIDED_ARENA_BYTES;
#[cfg(all(not(feature = "local"), feature = "mode-strided"))]
const MODE_BLOB: usize = STRIDED_BLOB_BYTES;

#[cfg(not(feature = "local"))]
fn write_file(path: &[u8], data: &[u8]) {
    let fd = unsafe {
        sceIoOpen(
            path.as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o777,
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
macro_rules! dprint_line {
    ($($t:tt)*) => { psp_rt::dprintln!($($t)*) };
}

#[cfg(not(feature = "local"))]
fn app_main() {
    psp_rt::enable_home_button();
    unsafe { scePowerSetClockFrequency(333, 333, 166) };

    // Free partition memory AFTER this PRX loaded: its image (embedded weight
    // blob included) and .bss (arena + output buffers) are already claimed,
    // so the dense and strided PRXes report different numbers — that
    // difference IS the frontend's memory watermark, measured on hardware.
    let total_free = unsafe { sceKernelTotalFreeMemSize() };
    let max_block = unsafe { sceKernelMaxFreeMemSize() };

    psp_rt::dprintln!("STFT frontend benchmark — mode {}", MODE);
    psp_rt::dprintln!(
        "  compile-time: arena {} B, weight blob {} B, outputs {} B",
        MODE_ARENA,
        MODE_BLOB,
        OUTPUT_BYTES
    );
    psp_rt::dprintln!(
        "  after load: {} B free ({} B max block)",
        total_free,
        max_block
    );

    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;

    #[cfg(feature = "mode-dense")]
    let result = {
        dense::init();
        run_frontend!(dense, dprint_line, get_tick, tick_res)
    };
    #[cfg(feature = "mode-strided")]
    let result = {
        strided::init();
        run_frontend!(strided, dprint_line, get_tick, tick_res)
    };

    psp_rt::dprintln!("  frontend: {} us avg over {} runs", result.avg_us, RUNS);

    // Outputs for host-side comparison: bit-exact across modes, rel-RMS vs
    // the TFLite golden.
    let out0 = unsafe { &*core::ptr::addr_of!(OUT0) };
    let out1 = unsafe { &*core::ptr::addr_of!(OUT1) };
    #[cfg(feature = "mode-dense")]
    let (p0, p1): (&[u8], &[u8]) = (b"host0:/out_dense_2048.bin\0", b"host0:/out_dense_1024.bin\0");
    #[cfg(feature = "mode-strided")]
    let (p0, p1): (&[u8], &[u8]) = (
        b"host0:/out_strided_2048.bin\0",
        b"host0:/out_strided_1024.bin\0",
    );
    write_file(p0, unsafe {
        core::slice::from_raw_parts(out0.as_ptr() as *const u8, OUT_2048 * 4)
    });
    write_file(p1, unsafe {
        core::slice::from_raw_parts(out1.as_ptr() as *const u8, OUT_1024 * 4)
    });

    psp_rt::dprintln!(
        "#stftbench mode={} arena_bytes={} blob_bytes={} output_bytes={} free_bytes={} max_block_bytes={} avg_us={}",
        MODE,
        MODE_ARENA,
        MODE_BLOB,
        OUTPUT_BYTES,
        total_free,
        max_block,
        result.avg_us
    );
}

// ============================================================================
// Local (host CPU) entry point: both modes, checked against the goldens
// ============================================================================

#[cfg(feature = "local")]
static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

#[cfg(feature = "local")]
fn local_get_tick() -> u64 {
    EPOCH.get().expect("epoch not set").elapsed().as_nanos() as u64
}

#[cfg(feature = "local")]
macro_rules! println_line {
    ($($t:tt)*) => { println!($($t)*) };
}

#[cfg(feature = "local")]
fn load_golden(name: &str, len: usize) -> Vec<f32> {
    let path = format!(
        "{}/../../../models/birdnet/stft/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals.len(), len, "{name} has the wrong size");
    vals
}

/// Max per-frame error normalised by that frame's RMS, against the TFLite
/// golden — same yardstick as psp-rt's rfft tests (their bound is 1e-4; the
/// full 2048-point frontend accumulates a little more).
#[cfg(feature = "local")]
fn check_golden(got: &[f32], golden: &[f32], bins: usize, label: &str) -> bool {
    let mut worst = 0.0f64;
    for (g_row, w_row) in got.chunks_exact(bins).zip(golden.chunks_exact(bins)) {
        let rms = (w_row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / bins as f64)
            .sqrt()
            .max(1e-9);
        for (g, w) in g_row.iter().zip(w_row.iter()) {
            worst = worst.max(((g - w).abs() as f64) / rms);
        }
    }
    let ok = worst < 1e-3;
    println!(
        "  {label}: max err/frame-RMS vs TFLite golden = {worst:.2e} {}",
        if ok { "(ok)" } else { "EXCEEDS 1e-3" }
    );
    ok
}

#[cfg(feature = "local")]
fn main() {
    EPOCH.set(std::time::Instant::now()).unwrap();
    let tick_res: u64 = 1_000_000_000;

    let mode = {
        let args: Vec<String> = std::env::args().collect();
        match args.iter().position(|a| a == "--mode") {
            Some(i) => args
                .get(i + 1)
                .unwrap_or_else(|| panic!("--mode needs a value"))
                .clone(),
            None => "both".to_string(),
        }
    };
    if !["dense_gather", "strided_view", "both"].contains(&mode.as_str()) {
        panic!("--mode must be dense_gather, strided_view or both, got {mode}");
    }

    let golden_2048 = load_golden("golden_2048.bin", OUT_2048);
    let golden_1024 = load_golden("golden_1024.bin", OUT_1024);
    let mut ok = true;
    let mut dense_out: Option<(Vec<f32>, Vec<f32>)> = None;

    if mode != "strided_view" {
        println!("dense_gather (arena {DENSE_ARENA_BYTES} B, blob {DENSE_BLOB_BYTES} B):");
        dense::init();
        let r = run_frontend!(dense, println_line, local_get_tick, tick_res);
        println!("  frontend: {} us avg over {RUNS} runs", r.avg_us);
        let (o0, o1) = unsafe { (&*core::ptr::addr_of!(OUT0), &*core::ptr::addr_of!(OUT1)) };
        ok &= check_golden(o0, &golden_2048, 1025, "L=2048");
        ok &= check_golden(o1, &golden_1024, 513, "L=1024");
        dense_out = Some((o0.to_vec(), o1.to_vec()));
    }

    if mode != "dense_gather" {
        println!("strided_view (arena {STRIDED_ARENA_BYTES} B, blob {STRIDED_BLOB_BYTES} B):");
        strided::init();
        let r = run_frontend!(strided, println_line, local_get_tick, tick_res);
        println!("  frontend: {} us avg over {RUNS} runs", r.avg_us);
        let (o0, o1) = unsafe { (&*core::ptr::addr_of!(OUT0), &*core::ptr::addr_of!(OUT1)) };
        ok &= check_golden(o0, &golden_2048, 1025, "L=2048");
        ok &= check_golden(o1, &golden_1024, 513, "L=1024");

        if let Some((d0, d1)) = &dense_out {
            let exact = d0
                .iter()
                .zip(o0.iter())
                .chain(d1.iter().zip(o1.iter()))
                .all(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "  dense vs strided: {}",
                if exact { "bit-identical" } else { "MISMATCH" }
            );
            ok &= exact;
        }
    }

    if !ok {
        std::process::exit(1);
    }
    println!("PASS");
}
