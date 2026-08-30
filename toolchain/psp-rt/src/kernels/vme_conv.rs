//! Runtime for the VME-offloaded int8 1x1 convolution.
//!
//! The compiler (psp-tc's `vme_conv` module) bakes a per-conv plan: a
//! 106-word array context per job shape, the ring layout, and the tiling
//! (4 output channels x P pixels per job). This kernel drives it: quantize
//! the f32 activations into the shared 1 MB machine image, loop jobs
//! (rewrite the 4 weight rows per channel-quad, `vme::run()`, difference
//! the cumulative MacI stream back into exact int32 dots), and dequantize
//! into the f32 output the fake-quant pipeline expects.
//!
//! Integer math is exact (int8 x int8 accumulated in wrapping 24-bit,
//! recovered by differencing — RTL-verified in psp-tc's vme_conv_e2e), so
//! the host mirror below computes the identical result with plain scalar
//! code, and the device asserts against it in `examples/vme-conv-bench`.
//!
//! If the `psp_vme_kernel` plugin is missing (or pre-v1.1), the device
//! falls back to the same scalar integer path — slow but correct — and
//! says so once.

/// Context words per job shape (vme-assembler layout, loaded from the
/// image's mapped offset by the plugin).
pub const VME_CTX_WORDS: usize = 106;
/// Ring capacity in samples.
pub const RING_WORDS: usize = 2048;
const LANES: usize = 4;

/// Quantize one activation to the int8 grid and remove the zero point:
/// the array streams `q - zp`, so dots need no zero-point correction.
#[inline]
fn quantize(x: f32, inv_scale: f32, zp: i32) -> i32 {
    let q = libm::roundf(x * inv_scale) as i32 + zp;
    q.clamp(-128, 127) - zp
}

/// Difference one lane's cumulative stream back into dots (wrapping
/// 24-bit — the write path truncates mod 2^24, verified on the RTL).
#[cfg_attr(not(target_os = "psp"), allow(dead_code))]
#[inline]
fn dot_from_cumulative(cum: i32, prev: i32) -> i32 {
    let d = ((cum as i64 - prev as i64) & 0xFF_FFFF) as i32;
    (d ^ 0x80_0000) - 0x80_0000
}

/// The scalar mirror: identical integer math, no array. Host builds always
/// use it; the device uses it only when the plugin is absent — and
/// `examples/vme-conv-bench` asserts the array path against it on hardware.
#[allow(clippy::too_many_arguments)]
pub fn vme_conv1x1_i8_reference(
    input: &[f32],
    pixels: usize,
    k: usize,
    in_scale: f32,
    in_zp: i32,
    weights: &[i8],
    w_scales: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    co: usize,
) {
    let inv = 1.0 / in_scale;
    // Quantized activations for one pixel at a time (k <= 1024).
    let mut q = [0i32; RING_WORDS / 2];
    for p in 0..pixels {
        let row = &input[p * k..(p + 1) * k];
        for (j, x) in row.iter().enumerate() {
            q[j] = quantize(*x, inv, in_zp);
        }
        for c in 0..co {
            let wrow = &weights[c * k..(c + 1) * k];
            let mut acc = 0i32;
            for j in 0..k {
                acc += q[j] * wrow[j] as i32;
            }
            let b = bias.map_or(0.0, |b| b[c]);
            output[p * co + c] = acc as f32 * (in_scale * w_scales[c]) + b;
        }
    }
}

/// VME-offloaded int8 1x1 conv: `output[p, c] = dequant(Σ_j q(in[p, j]) ·
/// w[c, j])` over `[pixels, k]` channel-last input (NHWC is exactly that
/// for a 1x1). `ctx_full` covers `p_full`-pixel jobs; `ctx_rem` (may be
/// empty) covers the final `pixels % p_full` batch.
/// Whether the VME offload is actually usable on this device (plugin
/// booted or bootable, v1.1, image allocated). Host builds: always false.
pub fn vme_conv_available() -> bool {
    #[cfg(target_os = "psp")]
    {
        return device::available();
    }
    #[cfg(not(target_os = "psp"))]
    false
}

#[allow(clippy::too_many_arguments)]
pub fn vme_conv1x1_i8(
    input: &[f32],
    pixels: usize,
    k: usize,
    in_scale: f32,
    in_zp: i32,
    weights: &[i8],
    w_scales: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    co: usize,
    ctx_full: &[i32],
    ctx_rem: &[i32],
    p_full: usize,
    weights_off: usize,
) {
    #[cfg(target_os = "psp")]
    {
        if device::conv1x1_i8_vme(
            input, pixels, k, in_scale, in_zp, weights, w_scales, bias, output, co, ctx_full,
            ctx_rem, p_full, weights_off,
        ) {
            return;
        }
        crate::dprintln!("vme_conv: plugin unavailable, scalar fallback");
    }
    let _ = (ctx_full, ctx_rem, p_full, weights_off);
    vme_conv1x1_i8_reference(
        input, pixels, k, in_scale, in_zp, weights, w_scales, bias, output, co,
    );
}

#[cfg(target_os = "psp")]
mod device {
    use super::{dot_from_cumulative, quantize, LANES, VME_CTX_WORDS};
    use crate::vme;
    use core::sync::atomic::{AtomicI32, Ordering};

    const IMAGE_BYTES: usize = 0x100000;
    const CTX_BYTE_OFF: usize = 0xF8000;
    /// Image byte offsets: BASE_0..3 then TOP_0..3 (vme-assembler layout).
    const BASE_OFF: [usize; 4] = [0x0000, 0x2000, 0x4000, 0x6000];
    const TOP0_OFF: usize = 0x20000;

    /// 0 = untried, 1 = ready, -1 = unavailable.
    static STATE: AtomicI32 = AtomicI32::new(0);
    static mut IMAGE: *mut u32 = core::ptr::null_mut();

    fn setup() -> bool {
        match STATE.load(Ordering::Relaxed) {
            1 => return true,
            -1 => return false,
            _ => {}
        }
        let ok = (|| {
            // The plugin may already be booted by the host program (its
            // second VmeInit is not guaranteed to be idempotent), so only
            // init when the shared job is absent.
            if vme::Job::get().is_none() && vme::init() < 0 {
                return false;
            }
            let Some(job) = vme::Job::get() else { return false };
            if !job.has_image_mode() {
                return false;
            }
            let base = crate::mem::alloc_partition(b"vme_conv\0", IMAGE_BYTES + 64, None);
            if base.is_null() {
                return false;
            }
            let aligned = ((base as usize + 63) & !63) | 0x4000_0000;
            // The plugin restages every ring from the image each run; zero it
            // once so untouched regions are deterministic.
            let p = aligned as *mut u32;
            for i in 0..IMAGE_BYTES / 4 {
                unsafe { core::ptr::write_volatile(p.add(i), 0) };
            }
            unsafe { IMAGE = p };
            job.set_image(aligned as u32);
            true
        })();
        STATE.store(if ok { 1 } else { -1 }, Ordering::Relaxed);
        ok
    }

    #[inline]
    fn wr(img: *mut u32, word_off: usize, v: i32) {
        unsafe { core::ptr::write_volatile(img.add(word_off), v as u32) };
    }

    #[inline]
    fn rd(img: *const u32, word_off: usize) -> i32 {
        unsafe { core::ptr::read_volatile(img.add(word_off)) as i32 }
    }

    /// Whether the offload path is available (for benches that must not
    /// silently grind the scalar fallback).
    pub(super) fn available() -> bool {
        setup()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn conv1x1_i8_vme(
        input: &[f32],
        pixels: usize,
        k: usize,
        in_scale: f32,
        in_zp: i32,
        weights: &[i8],
        w_scales: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        co: usize,
        ctx_full: &[i32],
        ctx_rem: &[i32],
        p_full: usize,
        weights_off: usize,
    ) -> bool {
        if !setup() {
            return false;
        }
        let img = unsafe { IMAGE };
        let inv = 1.0 / in_scale;
        let quads = co.div_ceil(LANES);

        let mut p0 = 0usize;
        let mut ctx_loaded: Option<bool> = None; // Some(is_full_batch)
        while p0 < pixels {
            let pb = (pixels - p0).min(p_full);
            let full = pb == p_full;
            let ctx: &[i32] = if full { ctx_full } else { ctx_rem };
            debug_assert_eq!(ctx.len(), VME_CTX_WORDS);
            // Context: rewrite when the batch shape changes (once, plus
            // possibly once more for the final partial batch).
            if ctx_loaded != Some(full) {
                for (i, w) in ctx.iter().enumerate() {
                    wr(img, CTX_BYTE_OFF / 4 + i, *w);
                }
                ctx_loaded = Some(full);
            }
            // Activations for this pixel batch -> TOP_0, quantized.
            let top = TOP0_OFF / 4;
            for (i, x) in input[p0 * k..(p0 + pb) * k].iter().enumerate() {
                wr(img, top + i, quantize(*x, inv, in_zp));
            }

            for quad in 0..quads {
                // The 4 weight rows for this quad (zeros pad the tail quad).
                for lane in 0..LANES {
                    let c = quad * LANES + lane;
                    let woff = BASE_OFF[lane] / 4 + weights_off;
                    if c < co {
                        let row = &weights[c * k..(c + 1) * k];
                        for (j, w) in row.iter().enumerate() {
                            wr(img, woff + j, *w as i32);
                        }
                    } else {
                        for j in 0..k {
                            wr(img, woff + j, 0);
                        }
                    }
                }

                if vme::run() < 0 {
                    return false;
                }

                // Readback: difference each lane's cumulative stream.
                for lane in 0..LANES {
                    let c = quad * LANES + lane;
                    if c >= co {
                        break;
                    }
                    let scale = in_scale * w_scales[c];
                    let b = bias.map_or(0.0, |b| b[c]);
                    let base = BASE_OFF[lane] / 4;
                    let mut prev = 0i32;
                    for p in 0..pb {
                        let cum = rd(img, base + p);
                        let dot = dot_from_cumulative(cum, prev);
                        prev = cum;
                        output[(p0 + p) * co + c] = dot as f32 * scale + b;
                    }
                }
            }
            p0 += pb;
        }
        true
    }
}
