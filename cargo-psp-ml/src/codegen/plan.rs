//! Codegen plan IR
//!
//! Three core concepts:
//! 1. **TensorAlloc** — model-level buffer allocations (constants, intermediates, output)
//! 2. **ScratchBuffer** — op-level aligned buffers, optionally pre-loaded from a tensor
//! 3. **KernelCall** — actual kernel invocations, referencing tensors and scratch buffers

use crate::ir::graph::TensorId;

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
}

// ─── (1) Tensor allocations ─────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TensorAlloc {
    Constant {
        id: TensorId,
        float_offset: usize,
        float_len: usize,
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

#[derive(Debug, Clone, PartialEq)]
pub enum CopyStrategy {
    BulkCopy,
    RowPadded {
        num_rows: usize,
        src_stride: usize,
        dst_stride: usize,
    },
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
    Conv2d {
        input: Tensor4d,
        filter: Tensor4d,
        bias: Option<TensorId>,
        stride: [usize; 2],
        padding: [usize; 2],
        output: Tensor4d,
        has_relu: bool,
    },
    Im2colPadded {
        input: Tensor4d,
        kernel_size: [usize; 2],
        padding: [usize; 2],
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
    MaxPool2d {
        input: Tensor4d,
        output: Tensor4d,
    },
    Reshape {
        input: TensorId,
        output: TensorId,
    },
    FullyConnected {
        input: TensorId,
        in_features: usize,
        weights: TensorId,
        bias: TensorId,
        output: TensorId,
        out_features: usize,
        has_relu: bool,
    },
}
