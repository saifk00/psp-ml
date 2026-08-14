//! Why BirdNET's classifier FC is slow, measured at its real shape.
//!
//! The hardware counters say op 451 (`fully_connected`, `[1,1024] x
//! [6522,1024]^T`) takes 131 Mcycles and 427,203 d-cache misses. That miss
//! count is only **2.1% above the compulsory minimum**: 26.7 MB of weights
//! divided by the 64-byte line is 417,408 misses, and with m=1 every weight is
//! touched exactly once. There is no reuse, so no amount of blocking or tiling
//! can reduce the misses — the data simply has to come in.
//!
//! What is not forced is the *stalling*. 131 Mcycles over 417,408 lines is 314
//! cycles per line, against a measured DRAM latency of ~73 cycles and a plain
//! sequential scalar read that sustains 191 MB/s (111 cycles/line). The
//! classifier runs at 68 MB/s. So the loop is not bandwidth-bound and not
//! compute-bound (6.7 MFLOP would be ~58 ms of scalar FPU); it is waiting on
//! one miss at a time.
//!
//! Phase 1's roofline saw the machine overlap misses when given the chance —
//! `read_stride64` hit 372 MB/s and `lv.q` loads hit 326 MB/s against 191 MB/s
//! for scalar sequential. The variants here test whether that carries over to
//! a real GEMV, by attacking the two things the baseline lacks: wider loads
//! and independent miss streams.
//!
//! So the variants here sweep the number of *independent weight streams*: R
//! output rows are accumulated together, giving R sequential reads in flight
//! instead of one. Nothing else changes — same bytes, same order within each
//! stream, same arithmetic.
//!
//! Every variant reads exactly the same bytes in the same order, so **the miss
//! counts must come out equal**. If they do, and the times differ, then the
//! win is latency hiding rather than miss reduction — which is the claim being
//! tested. Correctness is gated against the scalar reference.

#![no_std]
#![no_main]
#![cfg_attr(target_os = "psp", feature(asm_experimental_arch))]

use core::ffi::c_void;
use psp::sys::{
    sceIoClose, sceIoOpen, sceIoWrite, scePowerSetClockFrequency, sceRtcGetCurrentTick,
    sceRtcGetTickResolution, IoOpenFlags,
};
use psp_rt::dprintln;
use psp_rt::profiler::{OpProfileStats, ProfileClear, ProfileDisable, ProfileEnable, ProfileGetRegs,
                       ProfileRegs};

psp_rt::module!("gemv_bench", 1, 0);

/// BirdNET's classifier, exactly: `[1,1024] x [6522,1024]^T`.
const K: usize = 1024;
const N: usize = 6522;
const CACHE_LINE: usize = 64;

fn tick() -> u64 {
    let mut t = 0u64;
    unsafe { sceRtcGetCurrentTick(&mut t) };
    t
}

fn alloc_f32(len: usize) -> &'static mut [f32] {
    let mut err = 0u32;
    let raw = psp_rt::mem::alloc_partition(b"gemv\0", len * 4 + CACHE_LINE, Some(&mut err));
    assert!(!raw.is_null(), "alloc of {} floats failed: 0x{:08X}", len, err);
    // 16-byte alignment is required by `lv.q`; take a full line for good measure.
    let aligned = (((raw as usize) + CACHE_LINE - 1) & !(CACHE_LINE - 1)) as *mut f32;
    unsafe { core::slice::from_raw_parts_mut(aligned, len) }
}

fn fill(buf: &mut [f32], seed: u32) {
    let mut s = seed;
    for v in buf.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = ((s >> 8) as f32 / 8388608.0) - 1.0;
    }
}

// ---------------------------------------------------------------------------
// Variants. All compute out[n] = dot(input, weights[n*K .. n*K+K]).
// ---------------------------------------------------------------------------

/// Baseline: byte-for-byte what `psp_rt::kernels::naive::fully_connected` does.
#[inline(never)]
fn scalar_1row(input: &[f32], w: &[f32], out: &mut [f32]) {
    for n in 0..N {
        let mut sum = 0.0f32;
        for k in 0..K {
            sum += input[k] * w[n * K + k];
        }
        out[n] = sum;
    }
}

/// Four output rows at once: four independent sequential streams through the
/// weights, so four misses can be outstanding instead of one. Same loads, same
/// bytes, same order within each stream.
#[inline(never)]
fn scalar_4row(input: &[f32], w: &[f32], out: &mut [f32]) {
    let mut n = 0;
    while n + 4 <= N {
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (b0, b1, b2, b3) = (n * K, (n + 1) * K, (n + 2) * K, (n + 3) * K);
        for k in 0..K {
            let x = input[k];
            s0 += x * w[b0 + k];
            s1 += x * w[b1 + k];
            s2 += x * w[b2 + k];
            s3 += x * w[b3 + k];
        }
        out[n] = s0;
        out[n + 1] = s1;
        out[n + 2] = s2;
        out[n + 3] = s3;
        n += 4;
    }
    for n in n..N {
        let mut sum = 0.0f32;
        for k in 0..K {
            sum += input[k] * w[n * K + k];
        }
        out[n] = sum;
    }
}

/// Generate an R-independent-stream variant: R output rows computed together,
/// so R sequential weight streams are in flight at once instead of one.
///
/// Identical bytes in an identical order within each stream, so the compulsory
/// miss count cannot change — only how many of those misses the memory system
/// is allowed to work on concurrently.
macro_rules! rows_variant {
    ($name:ident, $r:expr) => {
        #[inline(never)]
        fn $name(input: &[f32], w: &[f32], out: &mut [f32]) {
            const R: usize = $r;
            let mut n = 0;
            while n + R <= N {
                let mut acc = [0.0f32; R];
                for k in 0..K {
                    let x = input[k];
                    for r in 0..R {
                        acc[r] += x * w[(n + r) * K + k];
                    }
                }
                out[n..n + R].copy_from_slice(&acc);
                n += R;
            }
            while n < N {
                let mut sum = 0.0f32;
                for k in 0..K {
                    sum += input[k] * w[n * K + k];
                }
                out[n] = sum;
                n += 1;
            }
        }
    };
}

rows_variant!(scalar_2row, 2);
rows_variant!(scalar_8row, 8);
rows_variant!(scalar_16row, 16);

/// Interleave every group of 4 rows in place: `w[g][k][r]` instead of
/// `w[g*4+r][k]`.
///
/// The 4-stream variant is faster but takes *more* misses than compulsory,
/// because its four streams are 4 KB apart and collide in the cache — the
/// excess matches the input vector being evicted and re-read once per group.
/// Interleaving turns those four strided streams into one sequential one,
/// which should keep the ILP of four accumulators while returning the miss
/// count to compulsory.
///
/// Done in place with a 16 KB scratch because a second copy of the weights
/// would not fit next to the first.
fn pack_groups4(w: &mut [f32], scratch: &mut [f32]) {
    let groups = N / 4;
    for g in 0..groups {
        let base = g * 4 * K;
        scratch[..4 * K].copy_from_slice(&w[base..base + 4 * K]);
        for k in 0..K {
            for r in 0..4 {
                w[base + k * 4 + r] = scratch[r * K + k];
            }
        }
    }
}

/// Four accumulators over the interleaved layout: one sequential stream.
#[inline(never)]
fn packed_4row(input: &[f32], w: &[f32], out: &mut [f32]) {
    let groups = N / 4;
    for g in 0..groups {
        let base = g * 4 * K;
        let mut acc = [0.0f32; 4];
        for k in 0..K {
            let x = input[k];
            let row = &w[base + k * 4..base + k * 4 + 4];
            acc[0] += x * row[0];
            acc[1] += x * row[1];
            acc[2] += x * row[2];
            acc[3] += x * row[3];
        }
        out[g * 4..g * 4 + 4].copy_from_slice(&acc);
    }
    for n in (groups * 4)..N {
        let mut sum = 0.0f32;
        for k in 0..K {
            sum += input[k] * w[n * K + k];
        }
        out[n] = sum;
    }
}

// ---------------------------------------------------------------------------

struct Run {
    ms: u64,
    stats: OpProfileStats,
}

/// Time one variant and capture its hardware counters in the same pass.
fn measure(tick_res: u64, f: &mut dyn FnMut()) -> Run {
    let mut regs = OpProfileStats::zero();
    let mut raw: ProfileRegs = unsafe { core::mem::zeroed() };
    unsafe {
        ProfileClear();
        ProfileEnable();
    }
    let t0 = tick();
    f();
    let dt = tick() - t0;
    unsafe {
        ProfileDisable();
        ProfileGetRegs(&mut raw);
    }
    regs.accumulate(&raw);
    Run { ms: (dt * 1000) / tick_res, stats: regs }
}

/// Max error against the reference, normalised by the reference's largest
/// magnitude. Elementwise relative error is meaningless here: a 1024-term dot
/// product can cancel to near zero, which makes a tiny absolute difference
/// look enormous.
fn max_err(a: &[f32], b: &[f32]) -> f32 {
    let mut scale = 1e-12f32;
    for &v in b {
        if v.abs() > scale {
            scale = v.abs();
        }
    }
    let rms = scale;
    let mut worst = 0.0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs() / rms;
        if d > worst {
            worst = d;
        }
    }
    worst
}

fn app_main() {
    psp::enable_home_button();
    unsafe { scePowerSetClockFrequency(333, 333, 166) };
    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;

    let bytes = N * K * 4;
    dprintln!("=== GEMV bench: [1,{}] x [{},{}]^T ({} MB of weights) ===", K, N, K, bytes / 1048576);
    dprintln!("compulsory misses = {} B / {} B line = {}", bytes, CACHE_LINE, bytes / CACHE_LINE);

    let input = alloc_f32(K);
    let w = alloc_f32(N * K);
    let out = alloc_f32(N);
    let reference = alloc_f32(N);
    fill(input, 12345);
    fill(w, 999);

    // Reference first, so every variant can be gated against it.
    scalar_1row(input, w, reference);

    dprintln!("");
    dprintln!("variant         ms    MB/s   d-miss   i-miss  cyc/line  err");

    let report = |name: &str, r: &Run, err: f32| {
        let mbps = if r.ms > 0 { (bytes as u64 * 1000) / (r.ms * 1_048_576) } else { 0 };
        let cpl = if r.stats.d_miss > 0 { r.stats.cpuck / r.stats.d_miss } else { 0 };
        dprintln!(
            "{:<12} {:>5} {:>7} {:>8} {:>8} {:>9}  {}",
            name, r.ms, mbps, r.stats.d_miss, r.stats.i_miss, cpl, (err * 1000.0) as u32
        );
    };

    let r = measure(tick_res, &mut || scalar_1row(input, w, out));
    report("scalar_1row", &r, 0.0);
    let base_ms = r.ms;

    let r = measure(tick_res, &mut || scalar_4row(input, w, out));
    let e = max_err(out, reference);
    report("scalar_4row", &r, e);
    let s4 = r.ms;

    let r = measure(tick_res, &mut || scalar_2row(input, w, out));
    let e = max_err(out, reference);
    report("scalar_2row", &r, e);
    let s2 = r.ms;

    let r = measure(tick_res, &mut || scalar_8row(input, w, out));
    let e = max_err(out, reference);
    report("scalar_8row", &r, e);
    let s8 = r.ms;

    let r = measure(tick_res, &mut || scalar_16row(input, w, out));
    let e = max_err(out, reference);
    report("scalar_16row", &r, e);
    let s16 = r.ms;

    // Packing is destructive, so it goes last.
    let scratch = alloc_f32(4 * K);
    let t0 = tick();
    pack_groups4(w, scratch);
    let pack_ms = ((tick() - t0) * 1000) / tick_res;
    let r = measure(tick_res, &mut || packed_4row(input, w, out));
    let e = max_err(out, reference);
    report("packed_4row", &r, e);
    let sp = r.ms;
    dprintln!("(one-time repack cost {} ms; in a model it happens at compile time)", pack_ms);

    dprintln!("");
    dprintln!("err = max|delta| / max|ref|, x1000 (0 = matches scalar exactly)");
    dprintln!("packed_4row speedup x100 = {}", base_ms * 100 / sp.max(1));
    dprintln!("speedup x100 vs 1 stream: 2row={} 4row={} 8row={} 16row={}",
        base_ms * 100 / s2.max(1), base_ms * 100 / s4.max(1),
        base_ms * 100 / s8.max(1), base_ms * 100 / s16.max(1));

    // Machine-readable copy for the host.
    let mut buf = [0u8; 512];
    let mut len = 0;
    let mut put = |s: &str, buf: &mut [u8], len: &mut usize| {
        for &b in s.as_bytes() {
            if *len < buf.len() {
                buf[*len] = b;
                *len += 1;
            }
        }
    };
    put("gemv-bench done\n", &mut buf, &mut len);
    let fd = unsafe {
        sceIoOpen(
            b"host0:/gemv-bench.txt\0".as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o777,
        )
    };
    if fd.0 >= 0 {
        unsafe {
            sceIoWrite(fd, buf.as_ptr() as *const c_void, len);
            sceIoClose(fd);
        }
    }
}
