//! VME-accelerated int8 1x1 convolution: the context generator, the tiling
//! plan, and the VFPU-vs-VME cost heuristic the conv2d lowering consults.
//!
//! The mapping (silicon constraints from `vme-assembler`'s calibrated
//! model): one job computes **4 output channels x P pixels** of an int8
//! 1x1 conv as multiply-accumulates on the Media Engine's array —
//!
//! * `TOP_0` holds P·K activation samples (zero-point-subtracted int8 in
//!   i32 words, channel-last, exactly NHWC's layout for a 1x1), streamed
//!   linearly as every lane's *front* operand (TOP reads have no lane
//!   affinity).
//! * `BASE_n` holds lane n's K weight samples at the top of the ring
//!   (offset 2048−K), replayed per pixel (`Replay { seg_len: K,
//!   stride: 0 }`) as the *back* operand — base-bank reads are own-lane
//!   only, which is exactly one output channel per lane.
//! * FU0 runs `MacI` (running inner product). The write AGU walks with
//!   step 0 and `Replay { seg_len: K, stride: 1 }`, so each pixel's K
//!   writes land on one word and the survivor is the cumulative sum at
//!   that pixel's end: `BASE_n[p] = Σ_{q<=p} dot(q)` in wrapping 24-bit
//!   arithmetic. The CPU recovers exact dots by differencing — each dot
//!   fits 24 bits for K ≤ ~500 worst-case, so the wrap cancels.
//!
//! Capacity per job is the hard wall: activations P·K ≤ 2048 words (one
//! 8 KB ring at one sample per 32-bit word — the array has no packed int8
//! streams), so a job carries at most `4 · 2048 = 8192` MACs. That number,
//! against the measured per-job invocation overhead, is what the
//! [`vme_conv1x1_profitable`] heuristic weighs.

use vme_assembler::assemble::CTX_WORDS;
use vme_assembler::{
    context_words, validate, AguParams, Buffer, Opcode, Operation, Pe, Replay, Source, VmeConfig,
};

/// Ring-buffer capacity in samples (one per 32-bit word).
pub const RING_WORDS: usize = 2048;
/// Lanes = processing elements = output channels per job.
pub const LANES: usize = 4;

/// The per-conv plan the lowering bakes into the generated code.
#[derive(Debug, Clone, PartialEq)]
pub struct VmeConv1x1Plan {
    /// Dot length (input channels of the 1x1).
    pub k: usize,
    /// Pixels per full job: `floor(RING_WORDS / k)`.
    pub pixels_per_job: usize,
    /// Word offset of each lane's weight row within its BASE ring.
    pub weights_off: usize,
    /// Context for full jobs (count = pixels_per_job · k).
    pub ctx: [u32; CTX_WORDS],
}

/// Smallest dot length verified on silicon: the bench's k >= 24 shapes all
/// pass bit-exact, while a k = 8 probe job never signalled completion (the
/// plugin's run timeout fired). Until that boundary is mapped, the
/// generator refuses dot lengths below the verified floor.
pub const MIN_K: usize = 24;

/// Build the 106-word context for one job shape: 4 lanes of `MacI`, `p`
/// pixels of dot length `k`. Errors if the shape cannot fit the rings.
pub fn vme_conv1x1_ctx(k: usize, p: usize) -> Result<[u32; CTX_WORDS], String> {
    if !(MIN_K..=RING_WORDS / 2).contains(&k) {
        return Err(format!("k = {k} outside the verified range {MIN_K}..=1024"));
    }
    if p == 0 || p * k > RING_WORDS {
        return Err(format!("p = {p} pixels of k = {k} overrun the activation ring"));
    }
    let weights_off = (RING_WORDS - k) as u16;

    let mut vme = VmeConfig::new();
    vme.set_stream_len((p * k) as u32);
    for pe in Pe::ALL {
        let elem = vme.pe_mut(pe);
        elem.fu0().set_front(Source::Buf(Buffer::Top0));
        elem.fu0().set_back(Source::Buf(pe.write_buffer()));
        elem.fu0().set_op(Operation::new(Opcode::MacI));
        // Weights: lane's own BASE ring, top of the buffer, replayed per pixel.
        elem.read_base = AguParams {
            offset: weights_off,
            step: 1,
            replay: Some(Replay { seg_len: k as u16, stride: 0 }),
            ..AguParams::default()
        };
        // Results: one word per pixel — step 0 pins the K writes of a pixel
        // to one offset (last wins = that pixel's cumulative), stride 1
        // advances per segment.
        elem.write = AguParams {
            offset: 0,
            step: 0,
            replay: Some(Replay { seg_len: k as u16, stride: 1 }),
            ..AguParams::default()
        };
        // The write region [0, p) and the weight region [2048-k, 2048) are
        // disjoint (p·k <= 2048 implies p <= 2048-k for k >= 2).
        elem.allow_write_clobber = true;
    }

    let plan = validate(&vme).map_err(|errs| {
        errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ")
    })?;
    Ok(context_words(&vme, &plan))
}

/// Plan a conv's VME execution, or `None` when the shape cannot map
/// (dot length outside the ring-fit range).
pub fn plan_vme_conv1x1(k: usize) -> Option<VmeConv1x1Plan> {
    if !(MIN_K..=RING_WORDS / 2).contains(&k) {
        return None;
    }
    let pixels_per_job = RING_WORDS / k;
    let ctx = vme_conv1x1_ctx(k, pixels_per_job).ok()?;
    Some(VmeConv1x1Plan {
        k,
        pixels_per_job,
        weights_off: RING_WORDS - k,
        ctx,
    })
}

/// Scalar reference for the job's integer math: `dots[c][p]` for 4 weight
/// rows over P pixels of K channel-last activations — what the device
/// kernel's difference pass must reproduce (both are exact int32).
pub fn reference_dots(acts: &[i32], weights: &[&[i32]; LANES], k: usize, p: usize) -> Vec<[i32; LANES]> {
    (0..p)
        .map(|px| {
            let mut out = [0i32; LANES];
            for (c, row) in weights.iter().enumerate() {
                out[c] = acts[px * k..(px + 1) * k]
                    .iter()
                    .zip(row.iter())
                    .map(|(a, w)| a * w)
                    .sum();
            }
            out
        })
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════
// The VFPU-vs-VME heuristic
// ═════════════════════════════════════════════════════════════════════════

/// Measured on retail silicon by `examples/vme-conv-bench` (2026-08-27,
/// plugin v1.1 image mode, 333 MHz main CPU / 166 MHz ME):
///
/// * per-job wall time is dominated by the fixed image-mode round trip
///   (the ME re-stages all eight 8 KB rings from the image and reads them
///   all back every run), plus the main CPU's uncached writes of the
///   changed regions;
/// * the VFPU fake-quant path moves int8-conv MACs at the whole-model
///   in-situ rate measured by the hardware profiler.
///
/// Filled in from the bench's `#vmeconv` lines; see that example's README
/// for the raw table.
pub struct VmeCostModel {
    /// Fixed cost of one image-mode job, microseconds.
    pub job_overhead_us: f64,
    /// Marginal per-MAC cost on the array while streaming, MACs per µs.
    pub stream_macs_per_us: f64,
    /// The VFPU fake-quant conv path's effective rate, MACs per µs.
    pub vfpu_macs_per_us: f64,
}

/// The measured model (examples/vme-conv-bench on retail silicon,
/// 2026-08-27, all four shapes verified bit-exact):
///
/// | k   | co   | px   | jobs | per-job    | MACs/us |
/// |-----|------|------|------|------------|---------|
/// | 24  | 72   | 6144 | 1314 | 2503 us    | 3       |
/// | 36  | 288  | 1536 | 2016 | 1725 us    | 4       |
/// | 72  | 864  | 384  | 3024 | 1252 us    | 6       |
/// | 108 | 1536 | 96   | 2304 | 1488 us    | 4       |
///
/// The fixed image-mode round trip (the ME restages all eight 8 KB rings
/// and reads them all back; the CPU reads results uncached) is 1.25-2.5 ms
/// per job against ~12 us of array compute. `job_overhead_us` uses the
/// *best* measured case, which is already fatal: profitability needs
/// ~400 K MACs per job against the 8 K/job ring-capacity ceiling, so the
/// heuristic selects the VFPU for every real shape until the plugin can
/// stage deltas instead of whole images.
pub const VME_COST: VmeCostModel = VmeCostModel {
    job_overhead_us: 1252.0,   // best measured per-job cost (k=72 shape)
    stream_macs_per_us: 660.0, // 4 MACs/cycle at 166 MHz (RTL: 2065 cyc/job)
    vfpu_macs_per_us: 217.0,   // 178 M MACs / 0.82 s (hardware profile)
};
pub const VME_COST_MEASURED: bool = true;

/// Estimated microseconds for the VME path on a whole conv.
pub fn vme_conv1x1_estimate_us(pixels: usize, k: usize, co: usize) -> Option<f64> {
    let plan_pixels = RING_WORDS / k;
    if !(MIN_K..=RING_WORDS / 2).contains(&k) || plan_pixels == 0 {
        return None;
    }
    let pixel_batches = pixels.div_ceil(plan_pixels);
    let quad_jobs = co.div_ceil(LANES);
    let jobs = (pixel_batches * quad_jobs) as f64;
    let macs = (pixels * k * co) as f64;
    Some(jobs * VME_COST.job_overhead_us + macs / VME_COST.stream_macs_per_us)
}

/// Estimated microseconds for the VFPU fake-quant path on the same conv.
pub fn vfpu_conv_estimate_us(pixels: usize, k: usize, co: usize) -> f64 {
    (pixels * k * co) as f64 / VME_COST.vfpu_macs_per_us
}

/// The lowering's decision: offload an int8 1x1 conv to the VME only when
/// the measured cost model says it is faster than the VFPU fake-quant
/// path. `PSP_TC_FORCE_VME=1` overrides (for validating the offload path
/// end to end on shapes the model would reject).
pub fn vme_conv1x1_profitable(pixels: usize, k: usize, co: usize) -> bool {
    if std::env::var("PSP_TC_FORCE_VME").is_ok_and(|v| v != "0") {
        return plan_vme_conv1x1(k).is_some();
    }
    match vme_conv1x1_estimate_us(pixels, k, co) {
        Some(vme_us) => vme_us < vfpu_conv_estimate_us(pixels, k, co),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_covers_birdnet_int8_shapes() {
        // The int8 1x1 dot lengths in BirdNET's backbone all map; the one
        // 3x3 (K = 1728) exceeds the ring-fit range and stays on the VFPU.
        for k in [24usize, 36, 72, 108] {
            let plan = plan_vme_conv1x1(k).unwrap();
            assert_eq!(plan.pixels_per_job, RING_WORDS / k);
            assert_eq!(plan.weights_off, RING_WORDS - k);
            assert!(plan.pixels_per_job * k <= RING_WORDS);
        }
        assert!(plan_vme_conv1x1(1728).is_none());
    }

    #[test]
    fn ctx_encodes_the_job_shape() {
        let k = 72usize;
        let p = 28usize;
        let ctx = vme_conv1x1_ctx(k, p).unwrap();
        // PE0 write AGU (words 45..51): replay mode, step 0, count p*k.
        assert_eq!(ctx[45] >> 24 & 0x7F, 0x02, "write mode should be segmented");
        assert_eq!(ctx[46] >> 16, 0, "write step must be 0");
        assert_eq!((ctx[46] & 0xFFFF) as usize, p * k - 1, "write count");
        assert_eq!((ctx[47] & 0xFFFF) as usize, k - 1, "write INNER0 segment");
        assert_eq!(ctx[48] as usize, 1, "write stride 1");
        // PE0 base read (words 39..45): offset 2048-k, segment k, stride 0.
        assert_eq!((ctx[39] & 0xFFFF) as usize, RING_WORDS - k);
        assert_eq!((ctx[41] & 0xFFFF) as usize, k - 1);
        assert_eq!(ctx[42], 0);
    }

    #[test]
    fn heuristic_rejects_when_overhead_dominates() {
        // A tiny conv: one job's overhead alone exceeds the VFPU time.
        assert!(!vme_conv1x1_profitable(64, 24, 4));
    }

    #[test]
    fn estimates_are_monotone_in_work() {
        let a = vme_conv1x1_estimate_us(768, 72, 864).unwrap();
        let b = vme_conv1x1_estimate_us(1536, 72, 864).unwrap();
        assert!(b > a);
    }
}
