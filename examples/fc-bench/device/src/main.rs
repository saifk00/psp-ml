//! FullyConnected kernel benchmark at BirdNET's real shapes.
//!
//! BirdNET's two mel-projection FCs are 30% of inference time (2978 ms + 1493
//! ms of 19.97 s). Both are `C[511,96] = A[511,K] @ B[96,K]^T` with K=1025 and
//! K=513. This measures every candidate kernel at exactly those shapes and
//! gates each on correctness against the scalar reference.
//!
//! The host-side unit tests in `psp-rt` validate the *algorithm* against an f64
//! reference, but they exercise the scalar mirror. This binary is what
//! validates the actual VFPU assembly, on hardware.
//!
//! Results print to stdout and land in `host0:/fc-bench.json`.

#![no_std]
#![no_main]

use core::ffi::c_void;
use psp::sys::{
    sceIoClose, sceIoOpen, sceIoWrite, scePowerGetCpuClockFrequencyInt, scePowerSetClockFrequency,
    sceRtcGetCurrentTick, sceRtcGetTickResolution, IoOpenFlags,
};
use psp_rt::dprintln;
use psp_rt::kernels::{gemm_ap_len, gemm_bp_len, gemm_bt_packed, gemm_cp_len, pack_b_panel};
use psp_rt::kernels::{matmul_bt_tiled, naive};

psp_rt::module!("fc_bench", 1, 0);

/// The two shapes that matter, straight out of the BirdNET graph.
const SHAPES: [(usize, usize, usize, &str); 2] = [
    (511, 1025, 96, "op14"),
    (511, 513, 96, "op28"),
];

const M_MAX: usize = 511;
const K_MAX: usize = 1025;
const N_MAX: usize = 96;
const CACHE_LINE: usize = 64;
/// A representative BirdNET elementwise tensor: [1,12,32,288].
const ELEM_N: usize = 110592;

/// Micro-kernel ceiling: `vmmul.q` peaks at 2309 MFLOP/s but the VFPU has no
/// matrix multiply-accumulate, so each one costs 4 `vadd.q`. 128 FLOP per
/// (15.4 + 4*1.5) cycles at 333 MHz.
const CEILING_MFLOPS: u64 = 1992;

fn tick() -> u64 {
    let mut t = 0u64;
    unsafe { sceRtcGetCurrentTick(&mut t) };
    t
}

fn alloc_f32(len: usize) -> &'static mut [f32] {
    let mut err = 0u32;
    let raw = psp_rt::mem::alloc_partition(b"fcbench\0", len * 4 + CACHE_LINE, Some(&mut err));
    assert!(!raw.is_null(), "alloc of {} floats failed: 0x{:08X}", len, err);
    let aligned = (((raw as usize) + CACHE_LINE - 1) & !(CACHE_LINE - 1)) as *mut f32;
    unsafe { core::slice::from_raw_parts_mut(aligned, len) }
}

/// Deterministic pseudo-random fill in [-1, 1).
fn fill(buf: &mut [f32], seed: u32) {
    let mut s = seed | 1;
    for v in buf.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = ((s >> 8) as f32 / 8388608.0) - 1.0;
    }
}

struct Json {
    buf: [u8; 4096],
    pos: usize,
}

impl Json {
    fn new() -> Self {
        Json { buf: [0; 4096], pos: 0 }
    }
    fn push(&mut self, s: &[u8]) {
        for &b in s {
            if self.pos < self.buf.len() {
                self.buf[self.pos] = b;
                self.pos += 1;
            }
        }
    }
    fn push_u64(&mut self, mut v: u64) {
        if v == 0 {
            self.push(b"0");
            return;
        }
        let mut tmp = [0u8; 20];
        let mut n = 0;
        while v > 0 {
            tmp[n] = b'0' + (v % 10) as u8;
            v /= 10;
            n += 1;
        }
        for i in (0..n).rev() {
            self.push(&[tmp[i]]);
        }
    }
}

/// Max |got - want| normalised by the RMS of the reference, in parts per
/// million. Elementwise *relative* error is meaningless for a long dot product
/// over signed data — results can cancel to near zero — so compare against the
/// scale of the result matrix instead.
fn error_ppm(got: &[f32], want: &[f32]) -> u64 {
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f32;
    for i in 0..want.len() {
        let d = got[i] - want[i];
        let d = if d < 0.0 { -d } else { d };
        if d > max_abs {
            max_abs = d;
        }
        sum_sq += want[i] * want[i];
    }
    let rms = libm::sqrtf(sum_sq / want.len() as f32);
    if rms <= 0.0 {
        return 0;
    }
    (max_abs / rms * 1_000_000.0) as u64
}

#[allow(clippy::too_many_arguments)]
fn record(
    json: &mut Json,
    first: &mut bool,
    shape: &str,
    variant: &str,
    m: usize,
    k: usize,
    n: usize,
    us: u64,
    ppm: u64,
) {
    let flops = 2 * (m as u64) * (k as u64) * (n as u64);
    let mflops = if us > 0 { flops / us } else { 0 };
    let pct = mflops * 100 / CEILING_MFLOPS;
    let status = if ppm <= 1000 { "ok" } else { "BAD" };
    dprintln!(
        "  {:<22} {:>8} us  {:>6} MFLOP/s  {:>3}% ceil  err {} ppm {}",
        variant,
        us,
        mflops,
        pct,
        ppm,
        status
    );
    if !*first {
        json.push(b",\n");
    }
    *first = false;
    json.push(b"  {\"shape\":\"");
    json.push(shape.as_bytes());
    json.push(b"\",\"variant\":\"");
    json.push(variant.as_bytes());
    json.push(b"\",\"us\":");
    json.push_u64(us);
    json.push(b",\"mflops\":");
    json.push_u64(mflops);
    json.push(b",\"pct_ceiling\":");
    json.push_u64(pct);
    json.push(b",\"err_ppm\":");
    json.push_u64(ppm);
    json.push(b"}");
}

/// Baseline: exactly what codegen emits today — a batch of scalar GEMVs.
#[inline(never)]
fn run_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        naive::fully_connected(&a[i * k..(i + 1) * k], k, b, None, &mut c[i * n..(i + 1) * n], n);
    }
}

fn app_main() {
    psp::enable_home_button();
    // psplink boots at 222/111 MHz.
    unsafe { scePowerSetClockFrequency(333, 333, 166) };
    let cpu = unsafe { scePowerGetCpuClockFrequencyInt() };
    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;
    assert!(tick_res == 1_000_000);

    dprintln!("=== FC kernel benchmark ({} MHz) ===", cpu);

    // Worst-case buffers, reused across shapes.
    let a = alloc_f32(M_MAX * K_MAX);
    let b = alloc_f32(N_MAX * K_MAX);
    let bp = alloc_f32(gemm_bp_len(N_MAX, K_MAX));
    let c = alloc_f32(M_MAX * N_MAX);
    let c_ref = alloc_f32(M_MAX * N_MAX);
    // Padded copies for the existing `matmul_bt_tiled` (needs all dims /4).
    let a_pad = alloc_f32(512 * 1028);
    let b_pad = alloc_f32(96 * 1028);
    let c_pad = alloc_f32(512 * 96);
    // Blocking scratch, sized for the largest (mc, kc) we try.
    let ap = alloc_f32(gemm_ap_len(64, 256));
    let cp = alloc_f32(gemm_cp_len(64, N_MAX));
    // Dedicated elementwise buffers: `c` is only M_MAX*N_MAX floats.
    let e_in = alloc_f32(ELEM_N);
    let e_out = alloc_f32(ELEM_N);
    let e_ref = alloc_f32(ELEM_N);

    fill(a, 12345);
    fill(b, 999);

    let mut json = Json::new();
    let mut first = true;
    json.push(b"{\n \"cpu_mhz\":");
    json.push_u64(cpu as u64);
    json.push(b",\n \"results\":[\n");

    // ---- Elementwise: what BirdNET actually spends 30% of its time on ----
    // 6.62 M mul elements and 3.71 M logistic elements per inference. The
    // scalar loop disassembles to 11 instructions/element (~33 ns at 1 IPC)
    // yet measures 352 ns, so the interesting question is how much of that is
    // memory and how much the VFPU can recover.
    dprintln!("");
    dprintln!("--- elementwise (n = {} floats, 3 streams) ---", ELEM_N);
    {
        fill(e_in, 4242);
        let iters = 40usize;
        let total_elems = ELEM_N as u64 * iters as u64;

        let t0 = tick();
        for _ in 0..iters {
            naive::binary_mul(e_in, e_ref, e_out, ELEM_N);
        }
        let us_mul = tick() - t0;

        let t0 = tick();
        for _ in 0..iters {
            naive::unary_logistic(e_in, e_out);
        }
        let us_log = tick() - t0;

        let t0 = tick();
        for _ in 0..iters {
            psp_rt::kernels::logistic(e_in, e_out);
        }
        let us_vlog = tick() - t0;

        let t0 = tick();
        for _ in 0..iters {
            psp_rt::kernels::swish(e_in, e_out);
        }
        let us_swish = tick() - t0;

        // Accuracy: VFPU chain vs libm, and fused swish vs logistic-then-mul.
        naive::unary_logistic(e_in, e_ref);
        psp_rt::kernels::logistic(e_in, e_out);
        let log_ppm = error_ppm(e_out, e_ref);
        for i in 0..ELEM_N {
            e_ref[i] *= e_in[i];
        }
        psp_rt::kernels::swish(e_in, e_out);
        let swish_ppm = error_ppm(e_out, e_ref);

        let ns = |us: u64| us * 1000 / total_elems;
        dprintln!("  binary_mul (scalar)      {:>5} ns/elem", ns(us_mul));
        dprintln!("  logistic (libm expf)     {:>5} ns/elem", ns(us_log));
        dprintln!("  logistic (VFPU)          {:>5} ns/elem   err {} ppm", ns(us_vlog), log_ppm);
        dprintln!("  swish FUSED (VFPU)       {:>5} ns/elem   err {} ppm", ns(us_swish), swish_ppm);
        dprintln!("  today logistic+mul =     {:>5} ns/elem", ns(us_log) + ns(us_mul));
    }

    // ---- Micro-kernel in isolation ----
    // Everything L1-resident and reused, so this measures pure issue rate with
    // no cache misses. If this lands near the 1992 MFLOP/s ceiling the blocked
    // driver is memory-bound; if it lands near the blocked figure the asm
    // itself is the limit.
    dprintln!("");
    dprintln!("--- micro-kernel in isolation (L1-resident) ---");
    for &ktc in [4usize, 8, 32, 128].iter() {
        let iters: usize = 400_000 / ktc;
        // ap: ktc*16, bp: ktc*32, cp: 32 — a few KB total.
        let t0 = tick();
        for _ in 0..iters {
            unsafe {
                psp_rt::kernels::micro_4x8(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), ktc);
            }
        }
        let us = tick() - t0;
        let flops = (iters as u64) * (ktc as u64) * 256;
        let mflops = if us > 0 { flops / us } else { 0 };
        dprintln!(
            "  ktc={:<4} {:>8} us  {:>6} MFLOP/s  {:>3}% ceil",
            ktc,
            us,
            mflops,
            mflops * 100 / CEILING_MFLOPS
        );
        if !first {
            json.push(b",\n");
        }
        first = false;
        json.push(b"  {\"shape\":\"micro\",\"variant\":\"micro_ktc_");
        json.push_u64(ktc as u64);
        json.push(b"\",\"us\":");
        json.push_u64(us);
        json.push(b",\"mflops\":");
        json.push_u64(mflops);
        json.push(b",\"pct_ceiling\":");
        json.push_u64(mflops * 100 / CEILING_MFLOPS);
        json.push(b",\"err_ppm\":0}");
    }

    for &(m, k, n, shape) in SHAPES.iter() {
        dprintln!("");
        dprintln!("--- {} : C[{},{}] = A[{},{}] @ B[{},{}]^T ---", shape, m, n, m, k, n, k);

        // ---- v0: scalar baseline, also the correctness reference ----
        let t0 = tick();
        run_naive(a, b, c_ref, m, k, n);
        let us = tick() - t0;
        record(&mut json, &mut first, shape, "v0_scalar_gemv", m, k, n, us, 0);

        // ---- v1: existing matmul_bt_tiled on padded scratch ----
        // Pad M 511->512 and K to a multiple of 4; N=96 is already aligned.
        let kp = (k + 3) / 4 * 4;
        let mp = (m + 3) / 4 * 4;
        for i in 0..mp {
            for j in 0..kp {
                a_pad[i * kp + j] = if i < m && j < k { a[i * k + j] } else { 0.0 };
            }
        }
        for i in 0..n {
            for j in 0..kp {
                b_pad[i * kp + j] = if j < k { b[i * k + j] } else { 0.0 };
            }
        }
        let t0 = tick();
        matmul_bt_tiled(a_pad, b_pad, c_pad, mp / 4, kp / 4, n / 4);
        let us = tick() - t0;
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = c_pad[i * n + j];
            }
        }
        record(&mut json, &mut first, shape, "v1_matmul_bt_tiled", m, k, n, us, error_ppm(c, c_ref));

        // ---- Where does the blocked driver's time actually go? ----
        // Same nest as gemm_bt_packed, with the three phases timed apart.
        pack_b_panel(b, bp, n, k);
        {
            let (mc, kc) = (32usize, 64usize);
            let kt_total = (k + 7) / 8 * 2;
            let nb_total = (n + 7) / 8;
            let ktc_max = kc / 4;
            let (mut t_pack, mut t_micro, mut t_unpack, mut t_zero) = (0u64, 0u64, 0u64, 0u64);
            let mut m0 = 0;
            while m0 < m {
                let rows = if m - m0 < mc { m - m0 } else { mc };
                let mb_count = (rows + 3) / 4;
                let t = tick();
                for v in cp[..mb_count * nb_total * 32].iter_mut() {
                    *v = 0.0;
                }
                t_zero += tick() - t;
                let mut kt0 = 0;
                while kt0 < kt_total {
                    let ktc = if kt_total - kt0 < ktc_max { kt_total - kt0 } else { ktc_max };
                    let t = tick();
                    psp_rt::kernels::pack_a_block(a, ap, m, k, m0, mb_count, kt0 * 4, ktc);
                    t_pack += tick() - t;
                    let t = tick();
                    for nb in 0..nb_total {
                        let b_off = (nb * kt_total + kt0) * 32;
                        for mb in 0..mb_count {
                            unsafe {
                                psp_rt::kernels::micro_4x8(
                                    ap.as_ptr().add(mb * ktc * 16),
                                    bp.as_ptr().add(b_off),
                                    cp.as_mut_ptr().add((nb * mb_count + mb) * 32),
                                    ktc,
                                );
                            }
                        }
                    }
                    t_micro += tick() - t;
                    kt0 += ktc;
                }
                let t = tick();
                psp_rt::kernels::unpack_c_block(cp, c, m, n, m0, mb_count, nb_total);
                t_unpack += tick() - t;
                m0 += mc;
            }
            let flops = 2 * (m as u64) * (k as u64) * (n as u64);
            dprintln!(
                "  [breakdown m32_k64] pack {} us, micro {} us ({} MFLOP/s), unpack {} us, zero {} us",
                t_pack,
                t_micro,
                if t_micro > 0 { flops / t_micro } else { 0 },
                t_unpack,
                t_zero
            );
        }

        // A spread around the chosen default so the shape of the optimum
        // stays visible; psp-tc uses (32, 64).
        for &(mc, kc) in [(32usize, 64usize), (32, 32), (48, 16), (16, 64), (32, 128)].iter() {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            let t0 = tick();
            gemm_bt_packed(a, bp, c, ap, cp, m, k, n, mc, kc);
            let us = tick() - t0;
            let ppm = error_ppm(c, c_ref);
            let name: &str = match (mc, kc) {
                (32, 64) => "v2_m32_k64_DEFAULT",
                (32, 32) => "v2_m32_k32",
                (48, 16) => "v2_m48_k16",
                (16, 64) => "v2_m16_k64",
                _ => "v2_m32_k128",
            };
            record(&mut json, &mut first, shape, name, m, k, n, us, ppm);
        }
    }

    json.push(b"\n ]\n}\n");
    let fd = unsafe {
        sceIoOpen(
            b"host0:/fc-bench.json\0".as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o644,
        )
    };
    if fd.0 >= 0 {
        unsafe {
            sceIoWrite(fd, json.buf.as_ptr() as *const c_void, json.pos);
            sceIoClose(fd);
        }
        dprintln!("");
        dprintln!("Wrote fc-bench.json");
    }
}
