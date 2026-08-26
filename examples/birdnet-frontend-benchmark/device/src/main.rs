//! Whole-frontend benchmark: BirdNET's spectrogram pipeline from the
//! normalised signal to both branches' compressed mel spectrograms —
//! dense_tflite (the frontend as the model expresses it: dense gathers,
//! materialised windows, dense mel FCs) vs custom_ops (StridedViewStft +
//! FullyConnectedCB + SquarePow, tied together by
//! `psp_tc::stft_mel_frontend`).
//!
//! One mode per PRX so arena/blob attribute cleanly. custom_ops's outputs
//! are `[96, 511]` bank-major (transposed — the CB kernel's layout, which is
//! also what the full model's downstream TRANSPOSE wants); dense outputs are
//! `[511, 96]`. The checks index the golden accordingly.
//!
//! Every run prints a machine-readable `#frontbench ...` line for the host.

#![cfg_attr(not(feature = "local"), no_std)]
#![cfg_attr(not(feature = "local"), no_main)]

#[cfg(all(
    not(feature = "local"),
    not(any(feature = "mode-dense", feature = "mode-custom", feature = "mode-small"))
))]
compile_error!(
    "device build needs a mode: --no-default-features --features mode-dense (or mode-custom, mode-small)"
);
#[cfg(any(
    all(feature = "mode-dense", feature = "mode-custom"),
    all(feature = "mode-dense", feature = "mode-small"),
    all(feature = "mode-custom", feature = "mode-small"),
))]
compile_error!("mode features are one-per-PRX; build separately instead");

#[cfg(not(feature = "local"))]
use core::ffi::c_void;
#[cfg(not(feature = "local"))]
use psp::sys::{
    sceIoClose, sceIoOpen, sceIoWrite, sceKernelMaxFreeMemSize, sceKernelTotalFreeMemSize,
    scePowerSetClockFrequency, sceRtcGetCurrentTick, sceRtcGetTickResolution, IoOpenFlags,
};

#[cfg(not(feature = "local"))]
psp_rt::module!("front_bench", 1, 0);

include!(concat!(env!("OUT_DIR"), "/bench_config.rs"));

// dead_code: the benchmark drives forward/forward_timed only.
#[cfg(any(feature = "local", feature = "mode-dense"))]
#[allow(dead_code)]
mod dense {
    include!(concat!(env!("OUT_DIR"), "/dense/frontend_dense.rs"));
}
#[cfg(any(feature = "local", feature = "mode-custom"))]
#[allow(dead_code)]
mod custom {
    include!(concat!(env!("OUT_DIR"), "/custom/frontend_custom.rs"));
}

#[cfg(any(feature = "local", feature = "mode-small"))]
#[allow(dead_code)]
mod small {
    include!(concat!(env!("OUT_DIR"), "/small/frontend_small.rs"));
}

const N_SAMPLES: usize = 144_000;
const OUT_LEN: usize = N_WINDOWS * N_BANKS;
/// Timed iterations (after one warmup).
const RUNS: u64 = 3;

/// Normalised signal (TFLite's own tap for the cardinal fixture), embedded
/// 16-byte aligned and used in place.
#[repr(align(16))]
struct AlignedBytes<const N: usize>([u8; N]);

static SAMPLES_RAW: AlignedBytes<{ N_SAMPLES * 4 }> =
    AlignedBytes(*include_bytes!(concat!(env!("OUT_DIR"), "/samples.bin")));

fn samples() -> &'static [f32; N_SAMPLES] {
    unsafe { &*(SAMPLES_RAW.0.as_ptr() as *const [f32; N_SAMPLES]) }
}

static mut OUT_2048: [f32; OUT_LEN] = [0.0; OUT_LEN];
static mut OUT_1024: [f32; OUT_LEN] = [0.0; OUT_LEN];

/// Warm up once, then time RUNS full frontend passes of module `$m`,
/// printing a per-op breakdown of the final pass. Returns avg us per pass.
macro_rules! run_frontend {
    ($m:ident, $print:ident, $get_tick:expr, $tick_res:expr) => {{
        assert!($m::OUTPUT_SIZES == [OUT_LEN, OUT_LEN]);
        let get_tick: fn() -> u64 = $get_tick;
        let tick_res: u64 = $tick_res;
        let out0 = unsafe { &mut *core::ptr::addr_of_mut!(OUT_2048) };
        let out1 = unsafe { &mut *core::ptr::addr_of_mut!(OUT_1024) };
        $m::init();
        $m::forward(samples(), out0, out1); // warmup

        let mut op_ticks = [0u64; $m::NUM_OPS];
        let start = get_tick();
        for _ in 0..RUNS {
            $m::forward_timed(samples(), out0, out1, &mut op_ticks, get_tick);
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
const MODE: &str = "dense_tflite";
#[cfg(all(not(feature = "local"), feature = "mode-dense"))]
const MODE_ARENA: usize = DENSE_ARENA_BYTES;
#[cfg(all(not(feature = "local"), feature = "mode-dense"))]
const MODE_BLOB: usize = DENSE_BLOB_BYTES;

#[cfg(all(not(feature = "local"), feature = "mode-custom"))]
const MODE: &str = "custom_ops";
#[cfg(all(not(feature = "local"), feature = "mode-custom"))]
const MODE_ARENA: usize = CUSTOM_ARENA_BYTES;
#[cfg(all(not(feature = "local"), feature = "mode-custom"))]
const MODE_BLOB: usize = CUSTOM_BLOB_BYTES;

#[cfg(all(not(feature = "local"), feature = "mode-small"))]
const MODE: &str = "small_fft";
#[cfg(all(not(feature = "local"), feature = "mode-small"))]
const MODE_ARENA: usize = SMALL_ARENA_BYTES;
#[cfg(all(not(feature = "local"), feature = "mode-small"))]
const MODE_BLOB: usize = SMALL_BLOB_BYTES;

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

    let total_free = unsafe { sceKernelTotalFreeMemSize() };
    let max_block = unsafe { sceKernelMaxFreeMemSize() };

    psp_rt::dprintln!("whole-frontend benchmark — mode {}", MODE);
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
    let avg_us = run_frontend!(dense, dprint_line, get_tick, tick_res);
    #[cfg(feature = "mode-custom")]
    let avg_us = run_frontend!(custom, dprint_line, get_tick, tick_res);
    #[cfg(feature = "mode-small")]
    let avg_us = run_frontend!(small, dprint_line, get_tick, tick_res);

    psp_rt::dprintln!("  frontend: {} us avg over {} runs", avg_us, RUNS);

    #[cfg(feature = "mode-dense")]
    let (p0, p1): (&[u8], &[u8]) = (
        b"host0:/out_dense_front_2048.bin\0",
        b"host0:/out_dense_front_1024.bin\0",
    );
    #[cfg(feature = "mode-custom")]
    let (p0, p1): (&[u8], &[u8]) = (
        b"host0:/out_custom_front_2048.bin\0",
        b"host0:/out_custom_front_1024.bin\0",
    );
    #[cfg(feature = "mode-small")]
    let (p0, p1): (&[u8], &[u8]) = (
        b"host0:/out_small_front_2048.bin\0",
        b"host0:/out_small_front_1024.bin\0",
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
        "#frontbench mode={} arena_bytes={} blob_bytes={} output_bytes={} free_bytes={} max_block_bytes={} avg_us={}",
        MODE,
        MODE_ARENA,
        MODE_BLOB,
        OUTPUT_BYTES,
        total_free,
        max_block,
        avg_us
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

/// Max per-frame error normalised by that frame's RMS. Both modes gate at
/// 1e-2 for the whole frontend: our FFT differs from TFLite's in the last
/// bits (~4e-6 in STFT units), and the x^~0.17-0.22 compression amplifies
/// tiny absolute differences on near-zero mel values (measured: dense
/// 5.4e-3, custom 4.4e-3 worst-frame). custom additionally runs make_mel's
/// regenerated filterbank and is bank-major (`transposed`). The birdnet
/// end-to-end golden check is the decisive correctness gate.
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
    if !["dense_tflite", "custom_ops", "small_fft", "all", "both"].contains(&mode.as_str()) {
        panic!("--mode must be dense_tflite, custom_ops, small_fft or all, got {mode}");
    }
    let run_all = mode == "all" || mode == "both";

    let golden_2048 = load_golden("golden_mel_2048.bin");
    let golden_1024 = load_golden("golden_mel_1024.bin");
    let mut ok = true;

    if mode == "dense_tflite" || run_all {
        println!("dense_tflite (arena {DENSE_ARENA_BYTES} B, blob {DENSE_BLOB_BYTES} B):");
        let us = run_frontend!(dense, println_line, local_get_tick, tick_res);
        println!("  frontend: {us} us avg over {RUNS} runs");
        let (o0, o1) = unsafe { (&*core::ptr::addr_of!(OUT_2048), &*core::ptr::addr_of!(OUT_1024)) };
        ok &= check_golden(o0, &golden_2048, "L=2048", 1e-2, false);
        ok &= check_golden(o1, &golden_1024, "L=1024", 1e-2, false);
    }

    if mode == "custom_ops" || run_all {
        println!("custom_ops (arena {CUSTOM_ARENA_BYTES} B, blob {CUSTOM_BLOB_BYTES} B):");
        let us = run_frontend!(custom, println_line, local_get_tick, tick_res);
        println!("  frontend: {us} us avg over {RUNS} runs");
        let (o0, o1) = unsafe { (&*core::ptr::addr_of!(OUT_2048), &*core::ptr::addr_of!(OUT_1024)) };
        ok &= check_golden(o0, &golden_2048, "L=2048", 1e-2, true);
        ok &= check_golden(o1, &golden_1024, "L=1024", 1e-2, true);
    }

    if mode == "small_fft" || run_all {
        println!("small_fft (arena {SMALL_ARENA_BYTES} B, blob {SMALL_BLOB_BYTES} B):");
        let us = run_frontend!(small, println_line, local_get_tick, tick_res);
        println!("  frontend: {us} us avg over {RUNS} runs");
        let (o0, o1) = unsafe { (&*core::ptr::addr_of!(OUT_2048), &*core::ptr::addr_of!(OUT_1024)) };
        // The pruned L=2048 branch adds the anti-alias filter's passband
        // ripple and (attenuated) alias residue on top of custom_ops's
        // error; the L=1024 branch is untouched by the pass.
        ok &= check_golden(o0, &golden_2048, "L=2048", 3e-2, true);
        ok &= check_golden(o1, &golden_1024, "L=1024", 1e-2, true);
    }

    if !ok {
        std::process::exit(1);
    }
    println!("PASS");
}
