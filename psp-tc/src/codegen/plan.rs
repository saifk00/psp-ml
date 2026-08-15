//! Codegen plan IR
//!
//! Three core concepts:
//! 1. **TensorAlloc** — model-level buffer allocations (constants, intermediates, output)
//! 2. **ScratchBuffer** — op-level aligned buffers, optionally pre-loaded from a tensor
//! 3. **KernelCall** — actual kernel invocations, referencing tensors and scratch buffers

use std::collections::HashSet;

use crate::ir::graph::{DType, TensorId};
use crate::ir::psp::{BinaryOp, PoolType, ReduceOp, UnaryOp};

use super::arena::ArenaLayout;

/// Index into `OpPlan::scratch`.
pub type ScratchId = usize;

/// A tensor reference bundled with its 4D shape (NHWC).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tensor4d {
    pub id: TensorId,
    pub shape: [usize; 4],
}

/// Complete codegen plan for a model. Produced by `lower()`, consumed by `render()`.
#[derive(Debug, Clone, PartialEq)]
pub struct CodegenPlan {
    pub input_id: TensorId,
    pub output_id: TensorId,
    pub input_size: usize,
    pub output_size: usize,
    pub blob_bytes: usize,
    pub blob_floats: usize,
    pub allocs: Vec<TensorAlloc>,
    pub ops: Vec<OpPlan>,
    pub arena: Option<ArenaLayout>,
    pub stream: Option<StreamPlan>,
}

// ─── (4) Frame streaming ────────────────────────────────────────

/// Frame streaming plan: process N frames one at a time through a batch-
/// independent section of the graph, reducing arena from N× to 1× frame size.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamPlan {
    /// Number of frames (batch dimension, e.g. 511).
    pub frame_count: usize,
    /// Index of first frame-section op in `CodegenPlan::ops`.
    pub frame_start: usize,
    /// Index of last frame-section op (inclusive) in `CodegenPlan::ops`.
    pub frame_end: usize,
    /// Tensors produced before the frame loop but consumed inside it.
    pub frame_inputs: Vec<FrameBoundaryTensor>,
    /// Tensors produced inside the frame loop but consumed after it.
    pub frame_outputs: Vec<FrameBoundaryTensor>,
    /// Tensor IDs rewritten from [N,...] to [1,...] by the stream rewrite.
    /// Used by arena packing and render to identify frame-scoped tensors.
    pub rewritten_tensor_ids: HashSet<TensorId>,
}

/// A tensor that crosses the frame loop boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameBoundaryTensor {
    pub id: TensorId,
    /// View tensor (batch=1 alias) used by ops inside the frame loop.
    pub view_id: TensorId,
    /// Total size in floats (full N×... size).
    pub total_size: usize,
    /// Floats per frame (stride for slicing).
    pub frame_stride: usize,
}

// ─── (1) Tensor allocations ─────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TensorAlloc {
    /// A weight-blob constant. Offsets/lengths are in 4-byte units
    /// regardless of dtype (int8 constants must be a multiple of 4 bytes);
    /// `dtype` controls whether the rendered view is `&[f32]` or `&[i8]`.
    ///
    /// `streamed` constants are packed at the *end* of `weights.bin`, are
    /// never loaded by `init()`, and are read chunkwise from the file when
    /// their op executes; for them `float_offset` is a file offset, not a
    /// resident-memory offset.
    Constant {
        id: TensorId,
        float_offset: usize,
        float_len: usize,
        dtype: DType,
        streamed: bool,
    },
    Intermediate {
        id: TensorId,
        size: usize,
    },
    Output {
        id: TensorId,
        size: usize,
    },
}

// ─── (2) Scratch buffers ────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ScratchBuffer {
    pub size: usize,
    pub load_from: Option<ScratchLoad>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScratchLoad {
    pub source: TensorId,
    pub copy: CopyStrategy,
}

/// Where a GEMM operand lives: a model tensor, or an op-local scratch buffer
/// (im2col output, or weights packed at runtime).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GemmOperand {
    Tensor(TensorId),
    Scratch(ScratchId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum CopyStrategy {
    BulkCopy,
    RowPadded {
        num_rows: usize,
        src_stride: usize,
        dst_stride: usize,
    },
    /// Repack an `[n, k]` f32 weight matrix into `gemm_bt_packed`'s B layout.
    PackB { n: usize, k: usize },
    /// Repack an `[n, k]` int8 weight matrix into `gemm_bt_packed`'s B layout,
    /// dequantising with a per-output-channel scale during the pack so the
    /// weights stay 1 byte each in the blob.
    PackBDequantI8 { n: usize, k: usize, scales: TensorId },
    /// Row-padded copy from an int8 source, dequantizing each row with its
    /// per-output-channel scale: `dst[r][c] = src[r][c] as f32 * scales[r]`.
    RowPaddedDequantI8 {
        num_rows: usize,
        src_stride: usize,
        dst_stride: usize,
        scales: TensorId,
    },
}

impl CopyStrategy {
    /// Tensors this copy reads *besides* the `ScratchLoad::source`.
    ///
    /// Exhaustive on purpose: liveness (`arena`) and dead-constant pruning
    /// (`lower`) both depend on it, and a missing arm here silently drops a
    /// constant from the blob. Add an arm when you add a variant.
    pub fn extra_tensor_refs(&self) -> Option<TensorId> {
        match self {
            CopyStrategy::BulkCopy => None,
            CopyStrategy::RowPadded { .. } => None,
            CopyStrategy::PackB { .. } => None,
            CopyStrategy::RowPaddedDequantI8 { scales, .. } => Some(*scales),
            CopyStrategy::PackBDequantI8 { scales, .. } => Some(*scales),
        }
    }
}

// ─── (3) Kernel calls ───────────────────────────────────────────

/// One logical operator, grouping its scratch buffers and sub-operations.
#[derive(Debug, Clone, PartialEq)]
pub struct OpPlan {
    pub scratch: Vec<ScratchBuffer>,
    pub sub_ops: Vec<SubOpPlan>,
}

/// A named sub-operation within an op (e.g. "im2col", "matmul", "bias_add_relu").
#[derive(Debug, Clone, PartialEq)]
pub struct SubOpPlan {
    pub name: String,
    pub kernels: Vec<KernelCall>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KernelCall {
    DepthwiseConv2d {
        input: Tensor4d,
        filter: Tensor4d,
        bias: Option<TensorId>,
        stride: [usize; 2],
        padding: [usize; 4],
        output: Tensor4d,
    },
    Conv2d {
        input: Tensor4d,
        filter: Tensor4d,
        bias: Option<TensorId>,
        stride: [usize; 2],
        padding: [usize; 4],
        output: Tensor4d,
        has_relu: bool,
        /// When set, `filter` is int8 and the kernel dequantizes with these
        /// per-output-channel scales (`conv2d_q8` / `conv2d_relu_q8`).
        weight_scales: Option<TensorId>,
    },
    Im2colPadded {
        input: Tensor4d,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 4],
        output_hw: [usize; 2],
        output: ScratchId,
    },
    MatmulBtTiled {
        a: ScratchId,
        b: ScratchId,
        output: TensorId,
        m_tiles: usize,
        k_tiles: usize,
        n_tiles: usize,
    },
    BiasAdd {
        output: TensorId,
        bias: TensorId,
        rows: usize,
        cols: usize,
    },
    Relu {
        output: TensorId,
    },
    Pool2d {
        input: Tensor4d,
        output: Tensor4d,
        filter: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 4],
        pool_type: PoolType,
    },
    Reshape {
        input: TensorId,
        output: TensorId,
    },
    FullyConnected {
        input: TensorId,
        in_features: usize,
        weights: TensorId,
        bias: Option<TensorId>,
        output: TensorId,
        out_features: usize,
        has_relu: bool,
        batch_size: usize,
    },
    /// Cache-blocked, VFPU register-blocked FullyConnected:
    /// `output[m,n] = input[m,k] @ weights_packed[n,k]^T`.
    ///
    /// `weights_packed` is a compile-time repacking of the weight constant
    /// into `psp_rt::kernels`' micro-kernel layout, so there is no runtime
    /// weight shuffling. `ap`/`cp` are the packing and accumulator scratch.
    /// Measured 19.7x over the batched-GEMV path at BirdNET's mel shape.
    GemmBtPacked {
        a: GemmOperand,
        /// A's row stride, which may exceed `k` (im2col pads rows to a
        /// multiple of 4 and leaves those columns stale).
        lda: usize,
        b: GemmOperand,
        output: TensorId,
        m: usize,
        k: usize,
        n: usize,
        ap: ScratchId,
        cp: ScratchId,
        mc: usize,
        kc: usize,
    },

    /// FullyConnected whose weight matrix is streamed from the weight file in
    /// row chunks (through a scratch buffer) instead of being memory-resident.
    /// Batch-1 only.
    FullyConnectedStreamed {
        input: TensorId,
        in_features: usize,
        /// The streamed weight constant (render uses its FILE_OFFSET const).
        weights: TensorId,
        bias: Option<TensorId>,
        output: TensorId,
        out_features: usize,
        has_relu: bool,
        /// Scratch sized to `chunk_rows * in_features` floats.
        scratch: ScratchId,
        chunk_rows: usize,
    },
    ElementWise {
        op: BinaryOp,
        input_a: TensorId,
        input_b: TensorId,
        output: TensorId,
        b_len: usize,
    },
    UnaryElementWise {
        op: UnaryOp,
        input: TensorId,
        output: TensorId,
    },
    /// `out = x * sigmoid(x)`, computed 4 lanes at a time on the VFPU.
    Swish { input: TensorId, output: TensorId },
    PowConst { input: TensorId, exponent: TensorId, output: TensorId },
    /// Snap f32 values to an int8 quantization grid (TFLite QUANTIZE
    /// simulated in f32).
    FakeQuant {
        input: TensorId,
        output: TensorId,
        scale: f32,
        zero_point: i32,
    },
    Reduce {
        op: ReduceOp,
        input: TensorId,
        output: TensorId,
        batch_size: usize,
        frame_in_size: usize,
        frame_out_size: usize,
    },
    Pad {
        input: Tensor4d,
        output: Tensor4d,
        padding: [[usize; 2]; 4],
    },
    Transpose {
        input: TensorId,
        output: TensorId,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        perm: Vec<usize>,
    },
    ReverseV2 {
        input: TensorId,
        output: TensorId,
        input_shape: Vec<usize>,
        axis: usize,
    },

    /// Batched real FFT: `frames` contiguous length-`n` frames in, real
    /// frequency-bin parts out. Twiddles precomputed as blob constants.
    RfftBatch {
        input: TensorId,
        stage_twiddles: TensorId,
        unpack_twiddles: TensorId,
        output: TensorId,
        scratch: ScratchId,
        n: usize,
        frames: usize,
    },

    /// Pack N real values into N/2 interleaved complex pairs in bit-reversed order.
    RfftPack {
        input: TensorId,
        output: ScratchId,
        n: usize,
    },

    /// One radix-2 DIT butterfly stage of an FFT.
    FftButterflyStage {
        data: ScratchId,
        twiddles: TensorId,
        n_complex: usize,
        half_size: usize,
    },

    /// Unpack N/2 complex FFT result to N/2+1 real-part frequency bins.
    RfftUnpack {
        data: ScratchId,
        twiddles: TensorId,
        output: TensorId,
        n: usize,
    },

    /// N-dimensional strided slice (begin/end/strides resolved at compile time).
    StridedSlice {
        input: TensorId,
        output: TensorId,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        begin: Vec<i32>,
        end: Vec<i32>,
        strides: Vec<i32>,
        begin_mask: i32,
        end_mask: i32,
        shrink_axis_mask: i32,
    },

    /// Gather elements along an axis using constant indices.
    Gather {
        input: TensorId,
        output: TensorId,
        indices: TensorId,
        indices_len: usize,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        axis: usize,
    },

    /// Concatenate tensors along an axis.
    Concatenation {
        inputs: Vec<TensorId>,
        output: TensorId,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
        axis: usize,
    },
}
