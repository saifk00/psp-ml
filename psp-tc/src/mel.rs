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
