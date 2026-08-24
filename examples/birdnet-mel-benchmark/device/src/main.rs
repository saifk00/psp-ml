//! BirdNET mel projection benchmark: dense_fc (the sliced .tflite's
//! [96, bins] matmul, 99%+ zero MACs) vs banded_cb (`FullyConnectedCB` over
//! `make_mel`'s column-banded filterbank + the fused square-pow).
//!
//! A device PRX contains exactly ONE mode (`--features mode-dense` or
//! `mode-cb`) so its arena and blob attribute cleanly; each mode runs both
//! branches (L=2048 and L=1024). Inputs are the STFT goldens — bit-exactly
//! what feeds this subgraph in the full model. The `local` build carries
//! both modes, takes `--mode` at runtime, and checks everything against the
//! TFLite goldens.
//!
//! banded_cb's output is `[96, 511]` — transposed relative to the dense
//! mode's `[511, 96]` — because the VFPU kernel writes each bank's per-4-row
//! GEMV result contiguously (vtfm4 tiles, see
//! `psp_rt::kernels::fully_connected_cb`); the checks index the golden
//! accordingly. Bank-major is also what the full model's downstream
//! TRANSPOSE wants.
//!
//! Every run prints a machine-readable `#melbench ...` line for the host.

#![cfg_attr(not(feature = "local"), no_std)]
#![cfg_attr(not(feature = "local"), no_main)]

#[cfg(all(
    not(feature = "local"),
    not(any(feature = "mode-dense", feature = "mode-cb"))
))]
compile_error!(
    "device build needs a mode: --no-default-features --features mode-dense (or mode-cb)"
);
#[cfg(all(feature = "mode-dense", feature = "mode-cb"))]
compile_error!("mode-dense and mode-cb are one-per-PRX; build twice instead");

#[cfg(not(feature = "local"))]
use core::ffi::c_void;
#[cfg(not(feature = "local"))]
use psp::sys::{
    sceIoClose, sceIoOpen, sceIoWrite, sceKernelMaxFreeMemSize, sceKernelTotalFreeMemSize,
    scePowerSetClockFrequency, sceRtcGetCurrentTick, sceRtcGetTickResolution, IoOpenFlags,
};

#[cfg(not(feature = "local"))]
psp_rt::module!("mel_bench", 1, 0);

include!(concat!(env!("OUT_DIR"), "/bench_config.rs"));

// dead_code: the benchmark drives forward/forward_timed only.
#[cfg(any(feature = "local", feature = "mode-dense"))]
#[allow(dead_code)]
mod dense_2048 {
    include!(concat!(env!("OUT_DIR"), "/dense/mel_dense_2048.rs"));
}
#[cfg(any(feature = "local", feature = "mode-dense"))]
#[allow(dead_code)]
mod dense_1024 {
    include!(concat!(env!("OUT_DIR"), "/dense/mel_dense_1024.rs"));
}
#[cfg(any(feature = "local", feature = "mode-cb"))]
#[allow(dead_code)]
mod cb_2048 {
    include!(concat!(env!("OUT_DIR"), "/cb/mel_cb_2048.rs"));
}
#[cfg(any(feature = "local", feature = "mode-cb"))]
#[allow(dead_code)]
mod cb_1024 {
    include!(concat!(env!("OUT_DIR"), "/cb/mel_cb_1024.rs"));
}

const IN_2048: usize = 511 * 1025;
const IN_1024: usize = 511 * 513;
const OUT_LEN: usize = N_WINDOWS * N_BANKS;
/// Timed iterations (after one warmup).
const RUNS: u64 = 3;

/// The branch inputs (STFT goldens), embedded 16-byte aligned so the
/// generated code can treat them as `&[f32; N]` in place — no decode copy.
#[repr(align(16))]
struct AlignedBytes<const N: usize>([u8; N]);

static INPUT_2048_RAW: AlignedBytes<{ IN_2048 * 4 }> =
    AlignedBytes(*include_bytes!(concat!(env!("OUT_DIR"), "/input_2048.bin")));
static INPUT_1024_RAW: AlignedBytes<{ IN_1024 * 4 }> =
    AlignedBytes(*include_bytes!(concat!(env!("OUT_DIR"), "/input_1024.bin")));

fn input_2048() -> &'static [f32; IN_2048] {
    unsafe { &*(INPUT_2048_RAW.0.as_ptr() as *const [f32; IN_2048]) }
}
fn input_1024() -> &'static [f32; IN_1024] {
    unsafe { &*(INPUT_1024_RAW.0.as_ptr() as *const [f32; IN_1024]) }
}

static mut OUT_2048: [f32; OUT_LEN] = [0.0; OUT_LEN];
static mut OUT_1024: [f32; OUT_LEN] = [0.0; OUT_LEN];

/// Warm up once, then time RUNS passes of one branch module, printing its
/// per-op breakdown. Returns the average microseconds per pass.
macro_rules! run_branch {
    ($m:ident, $print:ident, $input:expr, $out:expr, $get_tick:expr, $tick_res:expr) => {{
        assert!($m::OUTPUT_SIZE == OUT_LEN);
        let get_tick: fn() -> u64 = $get_tick;
        let tick_res: u64 = $tick_res;
        $m::init();
        $m::forward($input, $out); // warmup

        let mut op_ticks = [0u64; $m::NUM_OPS];
        let start = get_tick();
        for _ in 0..RUNS {
            $m::forward_timed($input, $out, &mut op_ticks, get_tick);
        }
        let total_us = (get_tick() - start) * 1_000_000 / tick_res;

        $print!("  per-op (avg over {} runs):", RUNS);
        for (i, name) in $m::OP_NAMES.iter().enumerate() {
            let us = op_ticks[i] * 1_000_000 / tick_res / RUNS;
            if us > 0 {
                $print!("    {:>8} us  {}", us, name);
            }
        }
        total_us / RUNS
    }};
}

/// Run both branches of one mode.
macro_rules! run_mode {
    ($m2048:ident, $m1024:ident, $print:ident, $get_tick:expr, $tick_res:expr) => {{
        let out0 = unsafe { &mut *core::ptr::addr_of_mut!(OUT_2048) };
        $print!("  L=2048:");
        let us_2048 = run_branch!($m2048, $print, input_2048(), out0, $get_tick, $tick_res);
        let out1 = unsafe { &mut *core::ptr::addr_of_mut!(OUT_1024) };
        $print!("  L=1024:");
        let us_1024 = run_branch!($m1024, $print, input_1024(), out1, $get_tick, $tick_res);
        (us_2048, us_1024)
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
const MODE: &str = "dense_fc";
#[cfg(all(not(feature = "local"), feature = "mode-dense"))]
const MODE_ARENA: usize = DENSE_ARENA_BYTES;
#[cfg(all(not(feature = "local"), feature = "mode-dense"))]
const MODE_BLOB: usize = DENSE_BLOB_BYTES;

#[cfg(all(not(feature = "local"), feature = "mode-cb"))]
const MODE: &str = "banded_cb";
#[cfg(all(not(feature = "local"), feature = "mode-cb"))]
const MODE_ARENA: usize = CB_ARENA_BYTES;
#[cfg(all(not(feature = "local"), feature = "mode-cb"))]
const MODE_BLOB: usize = CB_BLOB_BYTES;

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

    // Free partition memory AFTER this PRX loaded: its image (embedded blob
    // + inputs) and .bss (arena + outputs) are already claimed, so dense and
    // cb PRXes report different numbers — the mode's memory watermark,
    // measured on hardware.
    let total_free = unsafe { sceKernelTotalFreeMemSize() };
    let max_block = unsafe { sceKernelMaxFreeMemSize() };

    psp_rt::dprintln!("mel projection benchmark — mode {}", MODE);
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
    let (us_2048, us_1024) = run_mode!(dense_2048, dense_1024, dprint_line, get_tick, tick_res);
    #[cfg(feature = "mode-cb")]
    let (us_2048, us_1024) = run_mode!(cb_2048, cb_1024, dprint_line, get_tick, tick_res);

    psp_rt::dprintln!(
        "  mel: {} us (L=2048) + {} us (L=1024) avg over {} runs",
        us_2048,
        us_1024,
        RUNS
    );

    #[cfg(feature = "mode-dense")]
    let (p0, p1): (&[u8], &[u8]) = (
        b"host0:/out_dense_mel_2048.bin\0",
        b"host0:/out_dense_mel_1024.bin\0",
    );
    #[cfg(feature = "mode-cb")]
    let (p0, p1): (&[u8], &[u8]) = (
        b"host0:/out_cb_mel_2048.bin\0",
        b"host0:/out_cb_mel_1024.bin\0",
    );
    let out0 = unsafe { &*core::ptr::addr_of!(OUT_2048) };
    let out1 = unsafe { &*core::ptr::addr_of!(OUT_1024) };
    write_file(p0, unsafe {
        core::slice::from_raw_parts(out0.as_ptr() as *const u8, OUT_LEN * 4)
    });
    write_file(p1, unsafe {
        core::slice::from_raw_parts(out1.as_ptr() as *const u8, OUT_LEN * 4)
    });

    psp_rt::dprintln!(
        "#melbench mode={} arena_bytes={} blob_bytes={} output_bytes={} free_bytes={} max_block_bytes={} us_2048={} us_1024={}",
        MODE,
        MODE_ARENA,
        MODE_BLOB,
        OUTPUT_BYTES,
        total_free,
        max_block,
        us_2048,
        us_1024
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
fn load_golden(name: &str) -> Vec<f32> {
    let path = format!(
        "{}/../../../models/birdnet/stft/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals.len(), OUT_LEN, "{name} has the wrong size");
    vals
}

/// Max per-frame error normalised by that frame's RMS. `tol` is per mode:
/// dense_fc runs the stored matrix (1e-3); banded_cb runs `make_mel`'s
/// regenerated filterbank, whose ~2e-5 weight deltas the x^~0.22 compression
/// amplifies to a few 1e-3 (5e-3; MEL_CB_USE_STORED=1 at build time swaps in
/// the stored matrix and brings it back to dense levels).
///
/// The golden is `[N_WINDOWS, N_BANKS]` row-major; `transposed` says the
/// checked output is `[N_BANKS, N_WINDOWS]` (banded_cb's layout).
#[cfg(feature = "local")]
fn check_golden(got: &[f32], golden: &[f32], label: &str, tol: f64, transposed: bool) -> bool {
    let mut worst = 0.0f64;
    for (m, w_row) in golden.chunks_exact(N_BANKS).enumerate() {
        let rms = (w_row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
            / N_BANKS as f64)
            .sqrt()
            .max(1e-9);
        for (b, w) in w_row.iter().enumerate() {
            let g = if transposed {
                got[b * N_WINDOWS + m]
            } else {
                got[m * N_BANKS + b]
            };
            worst = worst.max(((g - w).abs() as f64) / rms);
        }
    }
    let ok = worst < tol;
    println!(
        "  {label}: max err/frame-RMS vs TFLite golden = {worst:.2e} {}",
        if ok { "(ok)" } else { "EXCEEDS TOLERANCE" }
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
    if !["dense_fc", "banded_cb", "both"].contains(&mode.as_str()) {
        panic!("--mode must be dense_fc, banded_cb or both, got {mode}");
    }

    let golden_2048 = load_golden("golden_mel_2048.bin");
    let golden_1024 = load_golden("golden_mel_1024.bin");
    let mut ok = true;
    let mut dense_out: Option<(Vec<f32>, Vec<f32>)> = None;

    if mode != "banded_cb" {
        println!("dense_fc (arena {DENSE_ARENA_BYTES} B, blob {DENSE_BLOB_BYTES} B):");
        let (us_2048, us_1024) =
            run_mode!(dense_2048, dense_1024, println_line, local_get_tick, tick_res);
        println!("  mel: {us_2048} us (L=2048) + {us_1024} us (L=1024) avg over {RUNS} runs");
        let (o0, o1) = unsafe { (&*core::ptr::addr_of!(OUT_2048), &*core::ptr::addr_of!(OUT_1024)) };
        ok &= check_golden(o0, &golden_2048, "L=2048", 1e-3, false);
        ok &= check_golden(o1, &golden_1024, "L=1024", 1e-3, false);
        dense_out = Some((o0.to_vec(), o1.to_vec()));
    }

    if mode != "dense_fc" {
        println!("banded_cb (arena {CB_ARENA_BYTES} B, blob {CB_BLOB_BYTES} B):");
        let (us_2048, us_1024) =
            run_mode!(cb_2048, cb_1024, println_line, local_get_tick, tick_res);
        println!("  mel: {us_2048} us (L=2048) + {us_1024} us (L=1024) avg over {RUNS} runs");
        let (o0, o1) = unsafe { (&*core::ptr::addr_of!(OUT_2048), &*core::ptr::addr_of!(OUT_1024)) };
        ok &= check_golden(o0, &golden_2048, "L=2048", 5e-3, true);
        ok &= check_golden(o1, &golden_1024, "L=1024", 5e-3, true);

        // Informational: how far the regenerated filterbank lands from the
        // stored one, end to end (both modes already gate against the
        // golden). dense is [511, 96], cb is [96, 511].
        if let Some((d0, d1)) = &dense_out {
            let mut worst = 0.0f32;
            for m in 0..N_WINDOWS {
                for b in 0..N_BANKS {
                    let a = (d0[m * N_BANKS + b] - o0[b * N_WINDOWS + m]).abs();
                    let c = (d1[m * N_BANKS + b] - o1[b * N_WINDOWS + m]).abs();
                    worst = worst.max(a).max(c);
                }
            }
            println!("  dense vs cb: max |diff| = {worst:.2e}");
        }
    }

    if !ok {
        std::process::exit(1);
    }
    println!("PASS");
}
