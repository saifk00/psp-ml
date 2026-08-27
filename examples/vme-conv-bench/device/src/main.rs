//! Hardware bench for the VME int8 1x1 conv offload: measures the
//! invocation overhead the psp-tc heuristic needs, and verifies the array
//! path bit-for-bit against the scalar integer reference at BirdNET's real
//! conv shapes.
//!
//! Prints machine-readable `#vmeconv` lines:
//!   micro  — per-`vme::run()` cost at two stream lengths (fixed job
//!            overhead vs stream-length slope)
//!   shape  — whole-conv wall time through `vme_conv1x1_i8`, with the
//!            job count and effective MACs/us
//!   verify — exact comparison against the scalar reference

#![no_std]
#![no_main]

use psp::sys::{sceRtcGetCurrentTick, sceRtcGetTickResolution};
use psp_rt::kernels::{vme_conv1x1_i8, vme_conv1x1_i8_reference};
use psp_rt::{dprintln, vme};

psp_rt::module!("vme_conv_bench", 1, 0);

include!(concat!(env!("OUT_DIR"), "/shapes.rs"));

const MAX_W: usize = 1536 * 108;
const MAX_IN: usize = 6144 * 24;
const MAX_OUT: usize = 6144 * 72;

static mut WEIGHTS: [i8; MAX_W] = [0; MAX_W];
static mut INPUT: [f32; MAX_IN] = [0.0; MAX_IN];
static mut OUTPUT: [f32; MAX_OUT] = [0.0; MAX_OUT];
static mut W_SCALES: [f32; 1536] = [0.0; 1536];
static mut BIAS: [f32; 1536] = [0.0; 1536];

const IN_SCALE: f32 = 0.05;
const IN_ZP: i32 = 3;

fn get_tick() -> u64 {
    let mut t = 0u64;
    unsafe { sceRtcGetCurrentTick(&mut t) };
    t
}

fn lcg(s: &mut u32) -> u32 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    *s
}

fn app_main() {
    psp_rt::enable_home_button();
    unsafe { psp::sys::scePowerSetClockFrequency(333, 333, 166) };
    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;

    dprintln!("=== vme-conv-bench ===");
    if vme::init() < 0 {
        dprintln!("#vmeconv fatal init failed (no plugin? not power-cycled?)");
        return;
    }
    let Some(job) = vme::Job::get() else {
        dprintln!("#vmeconv fatal no shared job");
        return;
    };
    if !job.has_image_mode() {
        dprintln!("#vmeconv fatal plugin pre-v1.1");
        return;
    }
    if !psp_rt::kernels::vme_conv_available() {
        dprintln!("#vmeconv fatal kernel setup failed (image alloc?)");
        return;
    }
    dprintln!("#vmeconv kernel ready");

    // (An earlier micro probe at k = 8 hit the plugin's run timeout — tiny
    // streams never signal completion; see vme_conv::MIN_K. Per-job cost is
    // read off the real shapes below instead.)
    let (input, weights, w_scales, bias, output) = unsafe {
        (
            &*core::ptr::addr_of!(INPUT),
            &*core::ptr::addr_of!(WEIGHTS),
            &*core::ptr::addr_of!(W_SCALES),
            &*core::ptr::addr_of!(BIAS),
            &mut *core::ptr::addr_of_mut!(OUTPUT),
        )
    };
    {
        let ws = unsafe { &mut *core::ptr::addr_of_mut!(W_SCALES) };
        let bs = unsafe { &mut *core::ptr::addr_of_mut!(BIAS) };
        for i in 0..1536 {
            ws[i] = 0.01 + (i % 7) as f32 * 0.003;
            bs[i] = (i % 5) as f32 * 0.1 - 0.2;
        }
    }
    let _ = (input, weights, w_scales, bias, &output);

    // ── the real shapes ─────────────────────────────────────────────────
    for shape in SHAPES.iter() {
        let (k, co, pixels) = (shape.k, shape.co, shape.pixels);
        let mut s = 0xB1DDu32 ^ (k as u32) << 8;
        let inp = unsafe { &mut *core::ptr::addr_of_mut!(INPUT) };
        let w = unsafe { &mut *core::ptr::addr_of_mut!(WEIGHTS) };
        for x in inp.iter_mut().take(pixels * k) {
            *x = ((lcg(&mut s) >> 20) as f32 / 1024.0 - 2.0) * 3.0;
        }
        for v in w.iter_mut().take(co * k) {
            *v = ((lcg(&mut s) >> 22) as i32 - 512).clamp(-128, 127) as i8;
        }

        dprintln!("#vmeconv running k={} co={} px={}", k, co, pixels);
        let jobs = pixels.div_ceil(shape.p_full) * co.div_ceil(4);
        let t0 = get_tick();
        vme_conv1x1_i8(
            &input[..pixels * k], pixels, k, IN_SCALE, IN_ZP, &weights[..co * k],
            &w_scales[..co], Some(&bias[..co]), &mut output[..pixels * co], co,
            shape.ctx_full, shape.ctx_rem, shape.p_full, shape.weights_off,
        );
        let us = (get_tick() - t0) * 1_000_000 / tick_res;
        let macs = (pixels * k * co) as u64;
        dprintln!(
            "#vmeconv shape k={} co={} px={} jobs={} total_us={} per_job_us={} macs_per_us={}",
            k, co, pixels, jobs, us, us / jobs as u64, macs / us.max(1)
        );

        // Exact verification against the scalar integer reference — the
        // array path must reproduce it bit-for-bit (same int dots, same
        // f32 dequant expression).
        let mut ref_row = [0.0f32; 1536];
        let mut bad = 0u32;
        for p in 0..pixels {
            vme_conv1x1_i8_reference(
                &input[p * k..(p + 1) * k], 1, k, IN_SCALE, IN_ZP, &weights[..co * k],
                &w_scales[..co], Some(&bias[..co]), &mut ref_row[..co], co,
            );
            for c in 0..co {
                if output[p * co + c].to_bits() != ref_row[c].to_bits() {
                    if bad == 0 {
                        dprintln!(
                            "  first mismatch p={} c={}: vme={} ref={}",
                            p, c, output[p * co + c], ref_row[c]
                        );
                    }
                    bad += 1;
                }
            }
        }
        dprintln!(
            "#vmeconv verify k={} {} mismatches={}",
            k,
            if bad == 0 { "PASS" } else { "FAIL" },
            bad
        );
    }

    job.restore_image_caps();
    dprintln!("#vmeconv done");
}
