//! Hand-constructed PSP IR graphs — the custom-op path into the compiler.
//!
//! TFLite is one producer of `PspModel`s; this is the other. A `build.rs`
//! that needs an op TFLite cannot express (today: `StridedViewStft`, which
//! replaces the dense-gather STFT frontend with strided views into the
//! signal) assembles the graph directly and hands it to
//! [`crate::compile_graph`]:
//!
//! ```no_run
//! use psp_tc::PspModelBuilder;
//!
//! let window: Vec<f32> = vec![1.0; 1024]; // hann window, really
//! let mut b = PspModelBuilder::new();
//! let samples = b.input(vec![144000]);
//! let w = b.constant_f32(vec![1024], &window);
//! let stft = b.strided_view_stft(samples, Some(w), 1024, 511);
//! b.output(stft);
//! let mut model = b.finish();
//! psp_tc::compile_graph(&mut model, std::path::Path::new("out"), "birdnet_stft").unwrap();
//! ```
//!
//! Builder graphs are taken as-is: shapes are declared, so none of the TFLite
//! pipeline (quant rewrite, fusion, shape inference, const fold) runs on them.

use crate::ir::graph::{DType, Graph, TensorId, TensorKind};
use crate::ir::psp::{PspModel, PspOp};

pub struct PspModelBuilder {
    graph: Graph<PspOp>,
    model_data: Vec<u8>,
}

impl PspModelBuilder {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            model_data: Vec::new(),
        }
    }

    /// Declare a graph input (F32). Codegen currently supports exactly one.
    pub fn input(&mut self, shape: Vec<usize>) -> TensorId {
        let id = self.graph.add_tensor(shape, DType::F32, TensorKind::Input);
        self.graph.inputs.push(id);
        id
    }

    /// Add an F32 constant backed by the model blob.
    pub fn constant_f32(&mut self, shape: Vec<usize>, data: &[f32]) -> TensorId {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "constant shape {shape:?} does not match {} data elements",
            data.len()
        );
        let offset = self.model_data.len();
        for v in data {
            self.model_data.extend_from_slice(&v.to_le_bytes());
        }
        self.graph.add_tensor(
            shape,
            DType::F32,
            TensorKind::Constant {
                offset,
                len: data.len() * 4,
            },
        )
    }

    /// Add an I32 constant backed by the model blob.
    pub fn constant_i32(&mut self, shape: Vec<usize>, data: &[i32]) -> TensorId {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "constant shape {shape:?} does not match {} data elements",
            data.len()
        );
        let offset = self.model_data.len();
        for v in data {
            self.model_data.extend_from_slice(&v.to_le_bytes());
        }
        self.graph.add_tensor(
            shape,
            DType::I32,
            TensorKind::Constant {
                offset,
                len: data.len() * 4,
            },
        )
    }

    /// Shape of a tensor created through this builder.
    pub fn shape(&self, id: TensorId) -> &[usize] {
        &self.graph.tensor(id).shape
    }

    /// Declare an intermediate (activation) tensor, for wiring hand-built
    /// ops via [`Self::add_op`].
    pub fn intermediate(&mut self, shape: Vec<usize>) -> TensorId {
        self.graph
            .add_tensor(shape, DType::F32, TensorKind::Intermediate)
    }

    /// Windowed strided-view STFT over a 1D signal: `n_windows` frames of
    /// `fft_length` samples, hop derived the way BirdNET frames —
    /// `floor((n_samples - fft_length) / (n_windows - 1))`. Returns the
    /// `[n_windows, fft_length/2 + 1]` output tensor (real parts of the
    /// frequency bins, TFLite RFFT2D + CAST semantics).
    pub fn strided_view_stft(
        &mut self,
        input: TensorId,
        window: Option<TensorId>,
        fft_length: usize,
        n_windows: usize,
    ) -> TensorId {
        let n_samples: usize = self.graph.tensor(input).shape.iter().product();
        assert!(
            n_windows > 1 && fft_length <= n_samples,
            "need n_windows > 1 and fft_length {fft_length} <= n_samples {n_samples}"
        );
        let hop = (n_samples - fft_length) / (n_windows - 1);
        self.strided_view_stft_with(input, window, fft_length, n_windows, hop, 1)
    }

    /// `strided_view_stft` with the hop and inner stride given explicitly —
    /// the small-FFT pass frames a decimated signal, where the hop is the
    /// original hop divided by the stored decimation and elements within a
    /// frame are `in_stride` apart (see `psp_rt::kernels::rfft_strided_batch`).
    pub fn strided_view_stft_with(
        &mut self,
        input: TensorId,
        window: Option<TensorId>,
        fft_length: usize,
        n_windows: usize,
        hop: usize,
        in_stride: usize,
    ) -> TensorId {
        let output = self.graph.add_tensor(
            vec![n_windows, fft_length / 2 + 1],
            DType::F32,
            TensorKind::Intermediate,
        );
        self.graph.ops.push(PspOp::StridedViewStft {
            input,
            window,
            output,
            fft_length,
            hop,
            in_stride,
            n_windows,
        });
        output
    }

    /// FIR lowpass + decimation of a 1D signal (see
    /// `psp_rt::kernels::fir_decimate`). Returns the `[len/factor]` output.
    pub fn fir_decimate(&mut self, input: TensorId, taps: &[f32], factor: usize) -> TensorId {
        let n: usize = self.graph.tensor(input).shape.iter().product();
        assert_eq!(n % factor, 0, "signal length {n} not divisible by {factor}");
        let taps_t = self.constant_f32(vec![taps.len()], taps);
        let output = self
            .graph
            .add_tensor(vec![n / factor], DType::F32, TensorKind::Intermediate);
        self.graph.ops.push(PspOp::FirDecimate {
            input,
            taps: taps_t,
            output,
            factor,
        });
        output
    }

    /// Matmul against a column-banded matrix (see [`crate::mel::CBMatrix`]):
    /// `out[b, m] = Σ_k in[m, start_b + k] * band_b[k]`. The matrix is
    /// serialised into the blob as a `[n_banks, 2]` I32 `[start, len]` table
    /// plus the concatenated band coefficients. Returns the
    /// `[n_banks, rows]` output tensor — **transposed** relative to the
    /// dense FC: each bank's per-4-row GEMV result stores contiguously, and
    /// bank-major is the orientation the full model's downstream TRANSPOSE
    /// wants anyway. See `psp_rt::kernels::fully_connected_cb`.
    pub fn fully_connected_cb(
        &mut self,
        input: TensorId,
        matrix: &crate::mel::CBMatrix,
    ) -> TensorId {
        let in_shape = self.graph.tensor(input).shape.clone();
        let (rows, in_features) = match in_shape.as_slice() {
            [rows, cols] => (*rows, *cols),
            other => panic!("fully_connected_cb input must be 2D, got {other:?}"),
        };
        assert_eq!(
            in_features, matrix.n_rows,
            "input has {in_features} features but the matrix has {} rows",
            matrix.n_rows
        );
        let n_banks = matrix.n_cols();
        let band_meta = self.constant_i32(vec![n_banks, 2], &matrix.band_meta());
        let band_data = self.constant_f32(vec![matrix.nnz()], &matrix.band_data());
        let output = self.graph.add_tensor(
            vec![n_banks, rows],
            DType::F32,
            TensorKind::Intermediate,
        );
        self.graph.ops.push(PspOp::FullyConnectedCB {
            input,
            band_meta,
            band_data,
            output,
        });
        output
    }

    /// Fused `(x^2)^p` elementwise — the spectrogram compression applied to
    /// the real-part FFT bins. Returns the output tensor (same shape).
    pub fn square_pow(&mut self, input: TensorId, exponent: f32) -> TensorId {
        let shape = self.graph.tensor(input).shape.clone();
        let output = self
            .graph
            .add_tensor(shape, DType::F32, TensorKind::Intermediate);
        self.graph.ops.push(PspOp::SquarePow {
            input,
            output,
            exponent,
        });
        output
    }

    /// Append an arbitrary op. The caller is responsible for having created
    /// its tensors through this builder.
    pub fn add_op(&mut self, op: PspOp) {
        self.graph.ops.push(op);
    }

    /// Mark a tensor as a graph output. Call once per output, in the order
    /// `forward()`'s output parameters should take.
    pub fn output(&mut self, id: TensorId) {
        self.graph.tensor_mut(id).kind = TensorKind::Output;
        self.graph.outputs.push(id);
    }

    pub fn finish(self) -> PspModel {
        PspModel {
            graph: self.graph,
            model_data: self.model_data,
        }
    }
}

impl Default for PspModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_builder_derives_birdnet_hops() {
        let mut b = PspModelBuilder::new();
        let samples = b.input(vec![144000]);
        let s2048 = b.strided_view_stft(samples, None, 2048, 511);
        let s1024 = b.strided_view_stft(samples, None, 1024, 511);
        b.output(s2048);
        b.output(s1024);
        let model = b.finish();

        assert_eq!(model.graph.tensor(s2048).shape, vec![511, 1025]);
        assert_eq!(model.graph.tensor(s1024).shape, vec![511, 513]);
        let hops: Vec<usize> = model
            .graph
            .ops
            .iter()
            .map(|op| match op {
                PspOp::StridedViewStft { hop, .. } => *hop,
                other => panic!("unexpected op {other:?}"),
            })
            .collect();
        assert_eq!(hops, vec![278, 280]);
    }

    #[test]
    fn stft_graph_generates_multi_output_code_with_no_arena() {
        let hann_2048: Vec<f32> = (0..2048).map(|i| (i % 7) as f32 * 0.1).collect();
        let hann_1024: Vec<f32> = (0..1024).map(|i| (i % 5) as f32 * 0.2).collect();

        let mut b = PspModelBuilder::new();
        let samples = b.input(vec![144000]);
        let w2048 = b.constant_f32(vec![2048], &hann_2048);
        let w1024 = b.constant_f32(vec![1024], &hann_1024);
        let s2048 = b.strided_view_stft(samples, Some(w2048), 2048, 511);
        let s1024 = b.strided_view_stft(samples, Some(w1024), 1024, 511);
        b.output(s2048);
        b.output(s1024);
        let mut model = b.finish();

        let generated = crate::codegen::generate_code_named(
            &mut model,
            None,
            None,
            None,
            "stft_weights.bin",
        )
        .unwrap();

        // The whole point: both STFT branches write straight into the output
        // statics, so the only arena occupant left is the larger branch's
        // n-float FFT scratch — 8 KB, vs the dense frontend's ~9.6 MiB of
        // materialised window matrices.
        assert_eq!(generated.stats.arena_size_floats, 2048);
        assert_eq!(
            generated.stats.output_size_floats,
            511 * 1025 + 511 * 513
        );
        // Blob: the two windows plus both branches' twiddles, nowhere near
        // the dense frontend's 2.35 MB of gather indices.
        assert!(
            generated.stats.blob_bytes < 64 * 1024,
            "blob unexpectedly large: {} B",
            generated.stats.blob_bytes
        );

        let code = generated.tokens.to_string();
        assert!(code.contains("OUTPUT_SIZES"));
        assert!(code.contains("output0"));
        assert!(code.contains("output1"));
        assert!(code.contains("rfft_strided_batch"));
        assert!(code.contains("stft_weights.bin"));
    }
}
