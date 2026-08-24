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
            n_windows,
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
