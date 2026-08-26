//! Mel filterbanks as column-banded matrices.
//!
//! BirdNET's mel projection is stored as a dense FC ([1025, 96] and
//! [513, 96]) that is 99%+ zeros: each mel bank weights one contiguous run
//! of 1–17 frequency bins (a triangle in mel space). `CBMatrix` keeps only
//! those bands, `make_mel` regenerates them from the reverse-engineered
//! parameters (HTK mel, fmin/fmax per branch — verified against the stored
//! matrices to ~2e-5), and [`mel_spectrogram`] assembles the whole
//! FC-CB + fused square-pow subgraph on a [`PspModelBuilder`].
//!
//! `make_mel` takes the sampling rate as a parameter because the PSP records
//! at 44.1 kHz while BirdNET's frontend assumes 48 kHz — regenerating the
//! banks at the true rate is how that mismatch gets fixed without a
//! resampler.

use crate::builder::PspModelBuilder;
use crate::ir::graph::TensorId;
use crate::ir::psp::{BinaryOp, PspOp, ReduceOp};

/// One column of a column-banded matrix: a contiguous run of nonzeros
/// starting at row `start`.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnBand {
    pub start: usize,
    pub data: Vec<f32>,
}

/// A logically `[n_rows, n_cols]` matrix stored as one contiguous band of
/// nonzeros per column (CSC restricted to banded columns). For the mel
/// matrices, rows are frequency bins and columns are banks.
#[derive(Debug, Clone, PartialEq)]
pub struct CBMatrix {
    pub n_rows: usize,
    pub columns: Vec<ColumnBand>,
}

impl CBMatrix {
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    pub fn nnz(&self) -> usize {
        self.columns.iter().map(|c| c.data.len()).sum()
    }

    /// Compress a matrix given as bank rows — the orientation TFLite stores
    /// FC weights in: `rows` is `[n_cols, n_rows]` row-major, one output
    /// column (bank) per row. Errors if any row's nonzeros are not one
    /// contiguous band.
    pub fn from_bank_rows(rows: &[f32], n_cols: usize, n_rows: usize) -> Result<Self, String> {
        if rows.len() != n_cols * n_rows {
            return Err(format!(
                "expected {n_cols}x{n_rows} = {} values, got {}",
                n_cols * n_rows,
                rows.len()
            ));
        }
        let mut columns = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let row = &rows[c * n_rows..(c + 1) * n_rows];
            let first = row.iter().position(|v| *v != 0.0);
            let band = match first {
                None => ColumnBand {
                    start: 0,
                    data: Vec::new(),
                },
                Some(start) => {
                    let end = n_rows - row.iter().rev().position(|v| *v != 0.0).unwrap();
                    if row[start..end].iter().any(|v| *v == 0.0) {
                        return Err(format!("column {c}: nonzeros are not one contiguous band"));
                    }
                    ColumnBand {
                        start,
                        data: row[start..end].to_vec(),
                    }
                }
            };
            columns.push(band);
        }
        Ok(Self {
            n_rows,
            columns,
        })
    }

    /// Expand back to bank rows (`[n_cols, n_rows]` row-major) — the inverse
    /// of `from_bank_rows`, for tests and comparisons.
    pub fn to_bank_rows(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_cols() * self.n_rows];
        for (c, band) in self.columns.iter().enumerate() {
            out[c * self.n_rows + band.start..][..band.data.len()].copy_from_slice(&band.data);
        }
        out
    }

    /// The `[start, len]` pairs the `fully_connected_cb` kernel consumes.
    pub fn band_meta(&self) -> Vec<i32> {
        self.columns
            .iter()
            .flat_map(|b| [b.start as i32, b.data.len() as i32])
            .collect()
    }

    /// All band coefficients concatenated in column order.
    pub fn band_data(&self) -> Vec<f32> {
        self.columns.iter().flat_map(|b| b.data.clone()).collect()
    }
}

/// HTK mel scale.
fn mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Triangular HTK mel filterbank over `n_freqs` real-FFT bins, as a
/// `[n_freqs, n_banks]` `CBMatrix`.
///
/// `B+2` points uniformly spaced in mel between `mel(fmin)` and `mel(fmax)`;
/// bank `b` is the triangle rising from point `b-1` to `b` and falling to
/// `b+1` (evaluated at each bin's mel, i.e. TF's
/// `linear_to_mel_weight_matrix` semantics, no area normalisation).
/// Reproduces BirdNET's stored matrices to ~2e-5 with
/// `(fmin, fmax) = (0, 3000)` for the 2048 window and `(500, 15000)` for
/// the 1024 window at 48 kHz.
pub fn make_mel(
    n_freqs: usize,
    n_banks: usize,
    fmin: f64,
    fmax: f64,
    sampling_rate: f64,
) -> CBMatrix {
    let l_fft = 2 * (n_freqs - 1);
    let m_min = mel(fmin);
    let dm = (mel(fmax) - m_min) / (n_banks + 1) as f64;
    let m_bank: Vec<f64> = (0..n_banks + 2).map(|i| m_min + i as f64 * dm).collect();
    let m_bin: Vec<f64> = (0..n_freqs)
        .map(|k| mel(k as f64 * sampling_rate / l_fft as f64))
        .collect();

    let mut columns = Vec::with_capacity(n_banks);
    for b in 1..=n_banks {
        let weight = |k: usize| -> f64 {
            let rising = (m_bin[k] - m_bank[b - 1]) / (m_bank[b] - m_bank[b - 1]);
            let falling = (m_bank[b + 1] - m_bin[k]) / (m_bank[b + 1] - m_bank[b]);
            rising.min(falling).max(0.0)
        };
        // The triangle is nonzero strictly between m_bank[b-1] and
        // m_bank[b+1]; scan that bin range and trim to the nonzero run.
        let mut start = None;
        let mut data = Vec::new();
        for k in 0..n_freqs {
            if m_bin[k] >= m_bank[b + 1] {
                break;
            }
            let w = weight(k);
            if w > 0.0 {
                start.get_or_insert(k);
                data.push(w as f32);
            } else if start.is_some() {
                break;
            }
        }
        columns.push(ColumnBand {
            start: start.unwrap_or(0),
            data,
        });
    }
    CBMatrix {
        n_rows: n_freqs,
        columns,
    }
}

/// Assemble the mel-spectrogram subgraph on a builder: banded matmul by the
/// generated filterbank, then the fused square-and-pow compression
/// (`(x^2)^p` — the FFT hands us real parts, so squaring comes first).
/// Returns the `[n_banks, n_windows]` output tensor — transposed relative
/// to the dense mel step (see `PspModelBuilder::fully_connected_cb`); the
/// caller marks it as a graph output.
#[allow(clippy::too_many_arguments)]
pub fn mel_spectrogram(
    b: &mut PspModelBuilder,
    input_fft: TensorId,
    sampling_rate: f64,
    fmin: f64,
    fmax: f64,
    n_banks: usize,
    pow_exponent: f32,
) -> TensorId {
    let shape = b.shape(input_fft).to_vec();
    let n_freqs = *shape.last().expect("input_fft has no shape");
    let mel_mat = make_mel(n_freqs, n_banks, fmin, fmax, sampling_rate);
    let matmul = b.fully_connected_cb(input_fft, &mel_mat);
    b.square_pow(matmul, pow_exponent)
}

/// BirdNET's input normalisation, replicated op-for-op from the model's
/// leading chain (verified against the flatbuffer):
///
/// ```text
/// min    = reduce_min(x)
/// shift  = x - min
/// max    = reduce_max(shift)        // of the SHIFTED signal
/// y      = shift / (max + 1e-6)
/// y      = (y - 0.5) * 2.0          // -> [-1, 1]
/// ```
///
/// Uses the same `Reduce`/`ElementWise` ops the dense pipeline lowers, so
/// the result is identical to compiling the model's own prefix. Returns the
/// normalised tensor (input's shape).
pub fn birdnet_normalize(b: &mut PspModelBuilder, x: TensorId) -> TensorId {
    let shape = b.shape(x).to_vec();
    let axes = b.constant_i32(vec![1], &[1]);
    let scalar = |b: &mut PspModelBuilder, v: f32| b.constant_f32(vec![1], &[v]);
    let mut binary = |b: &mut PspModelBuilder, op, a, rhs| {
        let out = b.intermediate(shape.clone());
        b.add_op(PspOp::ElementWise {
            op,
            input_a: a,
            input_b: rhs,
            output: out,
        });
        out
    };

    let min = b.intermediate(vec![1, 1]);
    b.add_op(PspOp::Reduce {
        op: ReduceOp::Min,
        input: x,
        axes,
        output: min,
    });
    let shift = binary(b, BinaryOp::Sub, x, min);

    let max = b.intermediate(vec![1, 1]);
    b.add_op(PspOp::Reduce {
        op: ReduceOp::Max,
        input: shift,
        axes,
        output: max,
    });
    let eps = scalar(b, 1e-6);
    let denom = {
        let out = b.intermediate(vec![1, 1]);
        b.add_op(PspOp::ElementWise {
            op: BinaryOp::Add,
            input_a: max,
            input_b: eps,
            output: out,
        });
        out
    };
    let y = binary(b, BinaryOp::Div, shift, denom);
    let half = scalar(b, 0.5);
    let y = binary(b, BinaryOp::Sub, y, half);
    let two = scalar(b, 2.0);
    binary(b, BinaryOp::Mul, y, two)
}

/// One STFT branch of the full custom frontend.
pub struct FrontendBranch {
    pub fft_length: usize,
    /// Hann window, `fft_length` long (extracted from the model).
    pub window: Vec<f32>,
    pub fmin: f64,
    pub fmax: f64,
    pub pow_exponent: f32,
}

/// The whole custom BirdNET frontend from a (normalised) signal: per
/// branch, a strided-view STFT then the banded mel projection + fused
/// square-pow. Returns one `[n_banks, n_windows]` output per branch
/// (bank-major — see `PspModelBuilder::fully_connected_cb`), in `branches`
/// order. The caller marks them as graph outputs.
pub fn stft_mel_frontend(
    b: &mut PspModelBuilder,
    samples: TensorId,
    n_windows: usize,
    sampling_rate: f64,
    n_banks: usize,
    branches: &[FrontendBranch],
) -> Vec<TensorId> {
    branches
        .iter()
        .map(|br| {
            assert_eq!(br.window.len(), br.fft_length);
            let w = b.constant_f32(vec![br.fft_length], &br.window);
            let stft = b.strided_view_stft(samples, Some(w), br.fft_length, n_windows);
            mel_spectrogram(
                b,
                stft,
                sampling_rate,
                br.fmin,
                br.fmax,
                n_banks,
                br.pow_exponent,
            )
        })
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════
// Small-FFT pruning pass
// ═════════════════════════════════════════════════════════════════════════
//
// The mel banks read a contiguous low prefix of the FFT's bins (BirdNET:
// 128 of 1025 for L=2048, 320 of 513 for L=1024) — most of the transform is
// computed and thrown away. This pass shrinks the FFT while keeping the
// SAME bin grid: reading the signal at stride D and transforming N = L/D
// points yields bins spaced Fs/L exactly as before, just fewer of them —
// provided the signal is anti-alias filtered first, since everything above
// the pruned Nyquist folds back into the kept bins.

/// How one branch's FFT gets pruned. Produced by [`plan_small_fft`].
#[derive(Debug, Clone, PartialEq)]
pub struct SmallFftPlan {
    /// Columns the mel banks actually read (top band's end bin + 1).
    pub needed_cols: usize,
    /// Pruned FFT length: needed_cols rounded up to 2^j + 1 columns
    /// (N = 2·2^j), then doubled while the Nyquist guard demands it.
    pub fft_length: usize,
    /// Total decimation `original_L / fft_length`.
    pub decim: usize,
    /// Decimation of the *stored* signal — the largest power of 2 dividing
    /// `decim` that keeps the hop integer (BirdNET's hop 278 = 2·139 allows
    /// 2, not 4). The rest of the decimation happens as the STFT's
    /// `in_stride` read, aliasing already removed by the filter.
    pub store_decim: usize,
    /// Frame-internal stride at the stored rate: `decim / store_decim`.
    pub in_stride: usize,
    /// Hop at the stored rate: `original_hop / store_decim`.
    pub hop: usize,
    /// Anti-alias FIR (designed at the original rate, applied by
    /// `fir_decimate` with factor `store_decim`).
    pub taps: Vec<f32>,
}

/// Compute the pruned-FFT plan for one branch, or `None` when the rule
/// leaves the FFT unchanged.
///
/// The sizing rule: count the FFT columns the branch's mel banks touch,
/// round up to the nearest `2^j + 1`, and use `N = 2·2^j` as the FFT size.
/// One guard on top: the anti-alias filter needs transition room. Aliases
/// onto a kept frequency f come from `Fs/decim − f` and above, so if the top
/// needed frequency exceeds `NYQUIST_GUARD` of the pruned Nyquist the
/// transition band would be impossibly narrow (BirdNET's L=2048 branch at
/// the rule's N=256: top bank at 2977 Hz vs a 3000 Hz Nyquist — 47 Hz of
/// room) and the size is doubled. At N=512 the transition spans
/// 2977→9023 Hz, a ~30-tap filter, and everything the transition band
/// aliases lands in bins the mel banks never read.
pub fn plan_small_fft(
    fft_length: usize,
    fmin: f64,
    fmax: f64,
    n_banks: usize,
    sampling_rate: f64,
    n_samples: usize,
    n_windows: usize,
) -> Option<SmallFftPlan> {
    const NYQUIST_GUARD: f64 = 0.75;
    const STOPBAND_DB: f64 = 60.0;

    let n_freqs = fft_length / 2 + 1;
    let mel_mat = make_mel(n_freqs, n_banks, fmin, fmax, sampling_rate);
    let needed_cols = mel_mat
        .columns
        .iter()
        .map(|b| b.start + b.data.len())
        .max()
        .unwrap_or(0);
    assert!(needed_cols > 1, "mel banks read no bins");

    // Round needed columns up to 2^j + 1; the FFT size is then 2·2^j.
    let pow = usize::BITS - (needed_cols - 2).leading_zeros();
    let mut n_new = 2usize << pow;
    // Nyquist guard.
    let top_hz = (needed_cols - 1) as f64 * sampling_rate / fft_length as f64;
    while n_new < fft_length
        && top_hz > NYQUIST_GUARD * sampling_rate * n_new as f64 / fft_length as f64 / 2.0
    {
        n_new *= 2;
    }
    if n_new >= fft_length {
        return None;
    }

    let decim = fft_length / n_new;
    let hop_full = (n_samples - fft_length) / (n_windows - 1);
    let mut store_decim = 1usize;
    while store_decim * 2 <= decim
        && decim % (store_decim * 2) == 0
        && hop_full % (store_decim * 2) == 0
        && n_samples % (store_decim * 2) == 0
    {
        store_decim *= 2;
    }

    // Anti-alias filter: pass everything the banks read, stop everything
    // whose alias would land on it.
    let stop_hz = sampling_rate / decim as f64 - top_hz;
    let taps = design_lowpass(sampling_rate, top_hz, stop_hz, STOPBAND_DB);

    Some(SmallFftPlan {
        needed_cols,
        fft_length: n_new,
        decim,
        store_decim,
        in_stride: decim / store_decim,
        hop: hop_full / store_decim,
        taps,
    })
}

/// Kaiser windowed-sinc lowpass: unity passband up to `pass_hz`,
/// `atten_db` of stopband rejection from `stop_hz`. Odd tap count
/// (linear phase). Standard Kaiser design equations.
pub fn design_lowpass(fs: f64, pass_hz: f64, stop_hz: f64, atten_db: f64) -> Vec<f32> {
    assert!(stop_hz > pass_hz && stop_hz < fs / 2.0);
    let delta_w = 2.0 * std::f64::consts::PI * (stop_hz - pass_hz) / fs;
    let mut n = ((atten_db - 7.95) / (2.285 * delta_w)).ceil() as usize + 1;
    if n % 2 == 0 {
        n += 1;
    }
    let beta = if atten_db > 50.0 {
        0.1102 * (atten_db - 8.7)
    } else if atten_db >= 21.0 {
        0.5842 * (atten_db - 21.0).powf(0.4) + 0.07886 * (atten_db - 21.0)
    } else {
        0.0
    };
    let fc = (pass_hz + stop_hz) / 2.0 / fs; // normalized cutoff (cycles/sample)
    let center = (n - 1) as f64 / 2.0;
    let i0_beta = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let m = i as f64 - center;
            let sinc = if m == 0.0 {
                2.0 * fc
            } else {
                (2.0 * std::f64::consts::PI * fc * m).sin() / (std::f64::consts::PI * m)
            };
            let r = 2.0 * m / (n - 1) as f64;
            let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta;
            (sinc * w) as f32
        })
        .collect()
}

/// Modified Bessel function of the first kind, order 0 (power series).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half_x = x / 2.0;
    for k in 1..30 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

/// [`stft_mel_frontend`] with the small-FFT pruning pass applied per branch:
/// where [`plan_small_fft`] shrinks a branch's FFT, the signal is anti-alias
/// filtered and decimated once (`FirDecimate`), the strided STFT reads it at
/// the plan's hop and inner stride with the window subsampled to match, and
/// the mel filterbank is regenerated on the pruned grid — identical bands,
/// since the bin spacing is unchanged. Branches the rule leaves alone
/// compile exactly as in the baseline.
pub fn stft_mel_frontend_small_fft(
    b: &mut PspModelBuilder,
    samples: TensorId,
    n_windows: usize,
    sampling_rate: f64,
    n_banks: usize,
    branches: &[FrontendBranch],
) -> Vec<TensorId> {
    let n_samples: usize = b.shape(samples).iter().product();
    branches
        .iter()
        .map(|br| {
            assert_eq!(br.window.len(), br.fft_length);
            match plan_small_fft(
                br.fft_length,
                br.fmin,
                br.fmax,
                n_banks,
                sampling_rate,
                n_samples,
                n_windows,
            ) {
                None => {
                    let w = b.constant_f32(vec![br.fft_length], &br.window);
                    let stft = b.strided_view_stft(samples, Some(w), br.fft_length, n_windows);
                    mel_spectrogram(
                        b,
                        stft,
                        sampling_rate,
                        br.fmin,
                        br.fmax,
                        n_banks,
                        br.pow_exponent,
                    )
                }
                Some(plan) => {
                    let y = b.fir_decimate(samples, &plan.taps, plan.store_decim);
                    // Subsample the window AND scale by decim: the pruned
                    // DFT sums 1/decim as many terms, so without the scale
                    // every kept bin comes out decim x smaller than the
                    // baseline's (the aliasing identity X'[k] = (1/D)ΣV[..]).
                    let sub_window: Vec<f32> = (0..plan.fft_length)
                        .map(|j| br.window[j * plan.decim] * plan.decim as f32)
                        .collect();
                    let w = b.constant_f32(vec![plan.fft_length], &sub_window);
                    let stft = b.strided_view_stft_with(
                        y,
                        Some(w),
                        plan.fft_length,
                        n_windows,
                        plan.hop,
                        plan.in_stride,
                    );
                    mel_spectrogram(
                        b,
                        stft,
                        sampling_rate / plan.decim as f64,
                        br.fmin,
                        br.fmax,
                        n_banks,
                        br.pow_exponent,
                    )
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_rows_roundtrip() {
        let rows = vec![
            0.0, 0.5, 1.0, 0.0, //
            0.0, 0.0, 0.25, 0.75, //
            0.0, 0.0, 0.0, 0.0, // empty column allowed
        ];
        let m = CBMatrix::from_bank_rows(&rows, 3, 4).unwrap();
        assert_eq!(m.nnz(), 4);
        assert_eq!(m.columns[0].start, 1);
        assert_eq!(m.band_meta(), vec![1, 2, 2, 2, 0, 0]);
        assert_eq!(m.to_bank_rows(), rows);
    }

    #[test]
    fn split_band_is_rejected() {
        let rows = vec![1.0, 0.0, 1.0];
        assert!(CBMatrix::from_bank_rows(&rows, 1, 3).is_err());
    }

    #[test]
    fn small_fft_plan_matches_birdnet_arithmetic() {
        // L=2048 (banks 0–3 kHz, bins 0..127): rule gives N=256 but the top
        // bank sits at 99% of that Nyquist, so the guard doubles to 512.
        // Total decim 4; hop 278 keeps only a factor 2 integer, so the
        // signal stores at half rate and the STFT reads it at stride 2.
        let plan = plan_small_fft(2048, 0.0, 3000.0, 96, 48000.0, 144000, 511).unwrap();
        assert_eq!(plan.needed_cols, 128);
        assert_eq!(plan.fft_length, 512);
        assert_eq!(plan.decim, 4);
        assert_eq!(plan.store_decim, 2);
        assert_eq!(plan.in_stride, 2);
        assert_eq!(plan.hop, 139);
        assert!(plan.taps.len() % 2 == 1 && plan.taps.len() < 64, "{}", plan.taps.len());

        // L=1024 (banks to 15 kHz, bins 0..319): 320 rounds to 513 columns
        // -> N=1024, unchanged -> no plan.
        assert!(plan_small_fft(1024, 500.0, 15000.0, 96, 48000.0, 144000, 511).is_none());
    }

    #[test]
    fn lowpass_response_meets_spec() {
        let plan = plan_small_fft(2048, 0.0, 3000.0, 96, 48000.0, 144000, 511).unwrap();
        let h = |f_hz: f64| -> f64 {
            // magnitude of the FIR's frequency response at f (fs = 48 kHz)
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (t, &tap) in plan.taps.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * f_hz * t as f64 / 48000.0;
                re += tap as f64 * ang.cos();
                im += tap as f64 * ang.sin();
            }
            (re * re + im * im).sqrt()
        };
        // Passband (everything the banks read) within ~0.15 dB of unity.
        for f in [0.0, 1000.0, 2000.0, 2976.6] {
            assert!((h(f) - 1.0).abs() < 0.02, "f={f}: {}", h(f));
        }
        // Stopband: every alias source for the kept band is >= ~60 dB down.
        for f in [9023.4, 12000.0, 15000.0, 21000.0, 23900.0] {
            assert!(h(f) < 2e-3, "f={f}: {}", h(f));
        }
    }

    #[test]
    fn pruned_grid_regenerates_identical_bands() {
        // Bin spacing is unchanged (48000/2048 == 12000/512), so make_mel on
        // the pruned grid must produce the same bands.
        let full = make_mel(1025, 96, 0.0, 3000.0, 48000.0);
        let pruned = make_mel(257, 96, 0.0, 3000.0, 12000.0);
        assert_eq!(full.columns.len(), pruned.columns.len());
        for (a, b) in full.columns.iter().zip(pruned.columns.iter()) {
            assert_eq!(a.start, b.start);
            assert_eq!(a.data.len(), b.data.len());
            for (x, y) in a.data.iter().zip(b.data.iter()) {
                assert!((x - y).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn small_fft_frontend_graph_generates() {
        let mut b = PspModelBuilder::new();
        let raw = b.input(vec![1, 144000]);
        let norm = birdnet_normalize(&mut b, raw);
        let branches = [
            FrontendBranch {
                fft_length: 2048,
                window: (0..2048).map(|i| (i % 7) as f32 * 0.1).collect(),
                fmin: 0.0,
                fmax: 3000.0,
                pow_exponent: 0.2199,
            },
            FrontendBranch {
                fft_length: 1024,
                window: vec![1.0; 1024],
                fmin: 500.0,
                fmax: 15000.0,
                pow_exponent: 0.1720,
            },
        ];
        let outs = stft_mel_frontend_small_fft(&mut b, norm, 511, 48000.0, 96, &branches);
        for out in outs {
            b.output(out);
        }
        let mut model = b.finish();
        let generated = crate::codegen::generate_code_named(
            &mut model,
            None,
            None,
            None,
            "sf_weights.bin",
        )
        .unwrap();
        assert_eq!(generated.stats.output_size_floats, 2 * 96 * 511);
        let code = generated.tokens.to_string();
        assert!(code.contains("fir_decimate"));
        // The pruned branch: 512-point FFT at hop 139, inner stride 2.
        assert!(code.contains("512usize , 139usize , 2usize , 511usize"));
        // The untouched branch keeps its full 1024-point transform.
        assert!(code.contains("1024usize , 280usize , 1usize , 511usize"));
    }

    #[test]
    fn full_frontend_graph_generates() {
        // Raw signal -> normalize -> 2x (strided STFT -> banded mel).
        let mut b = PspModelBuilder::new();
        let raw = b.input(vec![1, 144000]);
        let norm = birdnet_normalize(&mut b, raw);
        let branches = [
            FrontendBranch {
                fft_length: 2048,
                window: vec![1.0; 2048],
                fmin: 0.0,
                fmax: 3000.0,
                pow_exponent: 0.2199,
            },
            FrontendBranch {
                fft_length: 1024,
                window: vec![1.0; 1024],
                fmin: 500.0,
                fmax: 15000.0,
                pow_exponent: 0.1720,
            },
        ];
        let outs = stft_mel_frontend(&mut b, norm, 511, 48000.0, 96, &branches);
        for out in &outs {
            b.output(*out);
        }
        let mut model = b.finish();

        let generated = crate::codegen::generate_code_named(
            &mut model,
            None,
            None,
            None,
            "frontend_weights.bin",
        )
        .unwrap();

        assert_eq!(generated.stats.output_size_floats, 2 * 96 * 511);
        // Arena: the normalised signal ping-pong (2x 144000 + scalars), the
        // live STFT spectra and FC intermediates — nowhere near the dense
        // frontend's ~13 MiB (STFT gathers + mel reshapes).
        assert!(
            generated.stats.arena_size_floats * 4 < 4 * 1024 * 1024,
            "arena unexpectedly large: {} floats",
            generated.stats.arena_size_floats
        );
        let code = generated.tokens.to_string();
        assert!(code.contains("rfft_strided_batch"));
        assert!(code.contains("fully_connected_cb"));
        assert!(code.contains("square_pow"));
    }

    #[test]
    fn mel_graph_generates_cb_code() {
        let mut b = PspModelBuilder::new();
        let fft = b.input(vec![511, 1025]);
        let out = mel_spectrogram(&mut b, fft, 48000.0, 0.0, 3000.0, 96, 0.2199);
        b.output(out);
        let mut model = b.finish();

        let generated = crate::codegen::generate_code_named(
            &mut model,
            None,
            None,
            None,
            "mel_weights.bin",
        )
        .unwrap();

        assert_eq!(generated.stats.output_size_floats, 511 * 96);
        // Arena: just the FC-CB intermediate (square_pow writes the output
        // static). The dense slice of the same subgraph needs 2.2 MiB.
        assert_eq!(generated.stats.arena_size_floats, 511 * 96);
        // Blob: band meta + coefficients — ~1.8 KiB vs 387 KiB dense.
        assert!(
            generated.stats.blob_bytes < 4 * 1024,
            "blob unexpectedly large: {} B",
            generated.stats.blob_bytes
        );

        let code = generated.tokens.to_string();
        assert!(code.contains("fully_connected_cb"));
        assert!(code.contains("square_pow"));
    }

    /// make_mel vs the matrices BirdNET actually ships. Needs the gitignored
    /// fixtures from slice_stft.py, so it self-skips when they are absent —
    /// same convention as vme-assembler's e2e tests.
    #[test]
    fn make_mel_matches_birdnet_stored_matrices() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/birdnet/stft");
        for (bins, fmin, fmax) in [(1025usize, 0.0, 3000.0), (513, 500.0, 15000.0)] {
            let l = (bins - 1) * 2;
            let path = format!("{dir}/mel_dense_{l}.bin");
            let Ok(bytes) = std::fs::read(&path) else {
                eprintln!("skipping: {path} not generated (run slice_stft.py)");
                return;
            };
            let stored: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let ours = make_mel(bins, 96, fmin, fmax, 48000.0).to_bank_rows();
            assert_eq!(ours.len(), stored.len());

            // Same band structure (zero/nonzero pattern must agree exactly)…
            for (i, (a, b)) in ours.iter().zip(stored.iter()).enumerate() {
                assert_eq!(
                    *a == 0.0,
                    *b == 0.0,
                    "L={l}: sparsity pattern differs at {i} (ours {a}, stored {b})"
                );
            }
            // …and values to f32 rounding of the reversed formula.
            let worst = ours
                .iter()
                .zip(stored.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 5e-5, "L={l}: max |diff| = {worst}");
        }
    }
}
