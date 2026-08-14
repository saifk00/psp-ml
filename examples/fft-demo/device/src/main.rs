//! What the PSP's VFPU butterfly instructions actually do, and how to build an
//! FFT out of them.
//!
//! `vbfy1.q` / `vbfy2.q` are the VFPU's radix-2 butterfly ops.
//!
//! Run with `cargo run -p fft-demo-host --release`.

#![no_std]
#![no_main]
#![feature(asm_experimental_arch)]

use psp::sys::{
    scePowerSetClockFrequency, sceRtcGetCurrentTick, sceRtcGetTickResolution,
};
use psp::vfpu_asm;
use psp_rt::dprintln;

psp_rt::module!("fft_demo", 1, 0);

#[repr(align(16))]
struct Q([f32; 4]);

#[repr(align(16))]
struct Buf<const N: usize>([f32; N]);

fn tick() -> u64 {
    let mut t = 0u64;
    unsafe { sceRtcGetCurrentTick(&mut t) };
    t
}

// ---------------------------------------------------------------------------
// 1. What do the butterfly instructions compute?
// ---------------------------------------------------------------------------

fn run_vbfy1(input: [f32; 4]) -> [f32; 4] {
    let src = Q(input);
    let mut dst = Q([0.0; 4]);
    unsafe {
        vfpu_asm!(
            "lv.q R000, 0({i})",
            "vbfy1.q R100, R000",
            "sv.q R100, 0({o})",
            i = in(reg) (src.0.as_ptr()),
            o = in(reg) (dst.0.as_mut_ptr()),
            options(nostack),
        );
    }
    dst.0
}

fn run_vbfy2(input: [f32; 4]) -> [f32; 4] {
    let src = Q(input);
    let mut dst = Q([0.0; 4]);
    unsafe {
        vfpu_asm!(
            "lv.q R000, 0({i})",
            "vbfy2.q R100, R000",
            "sv.q R100, 0({o})",
            i = in(reg) (src.0.as_ptr()),
            o = in(reg) (dst.0.as_mut_ptr()),
            options(nostack),
        );
    }
    dst.0
}

/// Apply an arbitrary prefix-decorated op and return the result.
///
/// Prefixes (`vpfxs`/`vpfxt`/`vpfxd`) are written as bracket decorations on the
/// operands and the assembler emits the prefix instruction for us. They cost
/// one cycle with no added latency, and give free lane swizzle, negation,
/// absolute value and constant substitution.
fn run_swizzle(input: [f32; 4]) -> [f32; 4] {
    let src = Q(input);
    let mut dst = Q([0.0; 4]);
    unsafe {
        vfpu_asm!(
            "lv.q R000, 0({i})",
            // reverse the lanes, and negate the middle two
            "vmov.q R100, R000[W,-Z,-Y,X]",
            "sv.q R100, 0({o})",
            i = in(reg) (src.0.as_ptr()),
            o = in(reg) (dst.0.as_mut_ptr()),
            options(nostack),
        );
    }
    dst.0
}

/// Two complex multiplies on *interleaved* data `[re0,im0,re1,im1]`, using
/// prefixes to get the cross terms without touching memory:
///   p = bot[x,x,z,z] * tw[x,y,z,w]      -> [br*twr, br*twi, ...]
///   q = bot[y,y,w,w] * tw[-y,x,-w,z]    -> [-bi*twi, bi*twr, ...]
///   t = p + q                            -> [t_re, t_im, ...]
fn run_cmul(bot: [f32; 4], tw: [f32; 4]) -> [f32; 4] {
    let b = Q(bot);
    let t = Q(tw);
    let mut dst = Q([0.0; 4]);
    unsafe {
        vfpu_asm!(
            "lv.q R000, 0({b})",
            "lv.q R100, 0({t})",
            "vmul.q R200, R000[X,X,Z,Z], R100",
            "vmul.q R201, R000[Y,Y,W,W], R100[-Y,X,-W,Z]",
            "vadd.q R202, R200, R201",
            "sv.q R202, 0({o})",
            b = in(reg) (b.0.as_ptr()),
            t = in(reg) (t.0.as_ptr()),
            o = in(reg) (dst.0.as_mut_ptr()),
            options(nostack),
        );
    }
    dst.0
}

/// Name each output lane in terms of the input lanes a,b,c,d — this is the
/// whole point of the exercise, so print it as a formula, not just numbers.
fn describe(name: &str, f: fn([f32; 4]) -> [f32; 4]) {
    // Powers of 10 make each output lane's decomposition unambiguous:
    // any sum/difference of a,b,c,d reads straight off the digits.
    let out = f([1.0, 10.0, 100.0, 1000.0]);
    dprintln!("  {}([a,b,c,d]) with a=1 b=10 c=100 d=1000:", name);
    dprintln!(
        "    -> [{}, {}, {}, {}]",
        out[0] as i32,
        out[1] as i32,
        out[2] as i32,
        out[3] as i32
    );
    // Cross-check on a second vector so the reading isn't a coincidence.
    let o2 = f([3.0, 5.0, 7.0, 11.0]);
    dprintln!(
        "    [3,5,7,11] -> [{}, {}, {}, {}]",
        o2[0] as i32,
        o2[1] as i32,
        o2[2] as i32,
        o2[3] as i32
    );
}

// ---------------------------------------------------------------------------
// 2. A verified FFT built on them
// ---------------------------------------------------------------------------

const DEMO_N: usize = 16; // complex points

/// Naive DFT in f64-ish f32, the oracle for section 2.
fn dft(re: &[f32], im: &[f32], out_re: &mut [f32], out_im: &mut [f32], n: usize) {
    for k in 0..n {
        let mut sr = 0.0f32;
        let mut si = 0.0f32;
        for t in 0..n {
            let ang = -2.0 * core::f32::consts::PI * (k * t % n) as f32 / n as f32;
            let (c, s) = (libm::cosf(ang), libm::sinf(ang));
            sr += re[t] * c - im[t] * s;
            si += re[t] * s + im[t] * c;
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut r = 0;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

fn app_main() {
    psp::enable_home_button();
    unsafe { scePowerSetClockFrequency(333, 333, 166) };
    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;
    assert!(tick_res == 1_000_000);

    dprintln!("=== PSP VFPU butterfly demo ===");
    dprintln!("");
    dprintln!("1. Instruction semantics (measured, not from a manual)");
    describe("vbfy1.q", run_vbfy1);
    describe("vbfy2.q", run_vbfy2);

    dprintln!("");
    dprintln!("1b. Operand prefixes (swizzle / negate), emitted by bracket syntax");
    let sw = run_swizzle([1.0, 10.0, 100.0, 1000.0]);
    dprintln!(
        "  vmov.q rd, rs[W,-Z,-Y,X] on [1,10,100,1000] -> [{}, {}, {}, {}]",
        sw[0] as i32, sw[1] as i32, sw[2] as i32, sw[3] as i32
    );
    // (1+2i)*(3+4i) = -5+10i ; (5+6i)*(7+8i) = -13+82i
    let cm = run_cmul([1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]);
    dprintln!(
        "  2 complex muls via prefixes: (1+2i)(3+4i)=[{},{}]  (5+6i)(7+8i)=[{},{}]",
        cm[0] as i32, cm[1] as i32, cm[2] as i32, cm[3] as i32
    );
    dprintln!("    expected: [-5,10] and [-13,82]");

    // ---- 2. split-format radix-2 FFT, first stage via vbfy ----
    dprintln!("");
    dprintln!("2. {}-point complex FFT vs naive DFT", DEMO_N);

    let mut re = Buf::<DEMO_N>([0.0; DEMO_N]);
    let mut im = Buf::<DEMO_N>([0.0; DEMO_N]);
    for i in 0..DEMO_N {
        // an arbitrary but reproducible signal
        re.0[i] = libm::cosf(i as f32 * 0.7) + 0.3 * i as f32;
        im.0[i] = libm::sinf(i as f32 * 1.3);
    }

    let mut want_re = [0.0f32; DEMO_N];
    let mut want_im = [0.0f32; DEMO_N];
    dft(&re.0, &im.0, &mut want_re, &mut want_im, DEMO_N);

    // bit-reverse into split arrays
    let bits = DEMO_N.trailing_zeros() as usize;
    let mut gr = Buf::<DEMO_N>([0.0; DEMO_N]);
    let mut gi = Buf::<DEMO_N>([0.0; DEMO_N]);
    for k in 0..DEMO_N {
        let b = bit_reverse(k, bits);
        gr.0[k] = re.0[b];
        gi.0[k] = im.0[b];
    }

    // Stage 0 (half_size = 1, twiddle = 1): pure add/sub on adjacent pairs.
    // That is exactly a butterfly instruction — one per 4 values, applied to
    // the real and imaginary arrays independently because we keep them split.
    unsafe {
        let mut o = 0;
        while o + 4 <= DEMO_N {
            vfpu_asm!(
                "lv.q R000, 0({r})",
                "vbfy1.q R100, R000",
                "sv.q R100, 0({r})",
                "lv.q R001, 0({i})",
                "vbfy1.q R101, R001",
                "sv.q R101, 0({i})",
                r = in(reg) (gr.0.as_mut_ptr().add(o)),
                i = in(reg) (gi.0.as_mut_ptr().add(o)),
                options(nostack),
            );
            o += 4;
        }
    }

    // Remaining stages, scalar (the demo's point is stage 0 + correctness).
    let mut half = 2;
    while half < DEMO_N {
        let full = half * 2;
        let mut base = 0;
        while base < DEMO_N {
            for j in 0..half {
                let ang = -core::f32::consts::PI * j as f32 / half as f32;
                let (wr, wi) = (libm::cosf(ang), libm::sinf(ang));
                let (t, b) = (base + j, base + j + half);
                let tr = wr * gr.0[b] - wi * gi.0[b];
                let ti = wr * gi.0[b] + wi * gr.0[b];
                let (ur, ui) = (gr.0[t], gi.0[t]);
                gr.0[t] = ur + tr;
                gi.0[t] = ui + ti;
                gr.0[b] = ur - tr;
                gi.0[b] = ui - ti;
            }
            base += full;
        }
        half = full;
    }

    let mut worst = 0.0f32;
    for k in 0..DEMO_N {
        let dr = gr.0[k] - want_re[k];
        let di = gi.0[k] - want_im[k];
        let e = libm::sqrtf(dr * dr + di * di);
        if e > worst {
            worst = e;
        }
    }
    dprintln!("  bin 0 : fft {} vs dft {}", gr.0[0] as i32, want_re[0] as i32);
    dprintln!("  bin 1 : fft {} vs dft {}", gr.0[1] as i32, want_re[1] as i32);
    dprintln!("  worst |error| x1e6 = {}", (worst * 1e6) as i32);
    dprintln!(
        "  {}",
        if worst < 1e-3 {
            "MATCHES the DFT — stage 0 via butterfly instruction is correct"
        } else {
            "MISMATCH"
        }
    );

    // ---- 3. how much is stage 0 worth at BirdNET's size? ----
    // Static, not a stack local: `repr(align(16))` on a large local is not a
    // guarantee I want to lean on for `lv.q`, and a static in .bss is
    // unambiguously aligned.
    dprintln!("");
    dprintln!("3. Stage-0 cost at BirdNET's size (n_complex=1024)");
    const NC: usize = 1024;
    static mut BIG: Buf<NC> = Buf([1.0; NC]);
    let big: &mut [f32] = unsafe { &mut (*core::ptr::addr_of_mut!(BIG)).0 };
    let iters = 200usize;

    let t0 = tick();
    for _ in 0..iters {
        let mut o = 0;
        while o + 2 <= NC {
            let a = big[o];
            let b = big[o + 1];
            big[o] = a + b;
            big[o + 1] = a - b;
            o += 2;
        }
    }
    let us_scalar = tick() - t0;
    dprintln!("  scalar add/sub pairs : {} us", us_scalar);

    let t0 = tick();
    for _ in 0..iters {
        unsafe {
            let base = big.as_mut_ptr();
            let mut o = 0;
            while o + 4 <= NC {
                vfpu_asm!(
                    "lv.q R000, 0({p})",
                    "vbfy1.q R100, R000",
                    "sv.q R100, 0({p})",
                    p = in(reg) (base.add(o)),
                    options(nostack),
                );
                o += 4;
            }
        }
    }
    let us_vfpu = tick() - t0;
    dprintln!("  vbfy1.q              : {} us", us_vfpu);
    if us_vfpu > 0 {
        dprintln!("  speedup x100         : {}", us_scalar * 100 / us_vfpu);
    }

    dprintln!("");
    dprintln!("done");
}
