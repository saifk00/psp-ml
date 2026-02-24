//! PSP-specific IR
//! each op maps to a kernel
//!

use crate::ir::graph::{Graph, TensorId};

/// A lowered PSP model: pure IR graph paired with the raw model bytes.
/// Constant tensors reference into `model_data` via `TensorKind::Constant { offset, len }`.
#[derive(Debug)]
pub struct PspModel {
    pub graph: Graph<PspOp>,
    pub model_data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum PspOp {
    /// Direct convolution (naive, no im2col)
    Conv2d {
        input: TensorId,
        weights: TensorId,
        bias: Option<TensorId>,
        output: TensorId,
        params: Conv2dParams,
    },

    /// Fully connected / dense layer
    FullyConnected {
        input: TensorId,
        weights: TensorId,
        bias: Option<TensorId>,
        output: TensorId,
        fused_activation: FullyConnectedParams,
    },

    /// 2×2 max pooling, stride 2
    MaxPool2x2 { input: TensorId, output: TensorId },

    /// Reshape (zero-cost pointer reinterpret)
    Reshape { input: TensorId, output: TensorId },

    /// Softmax over last dimension
    Softmax { input: TensorId, output: TensorId },

    /// Element-wise binary operation (add, mul, sub, div, max)
    ElementWise {
        op: BinaryOp,
        input_a: TensorId,
        input_b: TensorId,
        output: TensorId,
    },

    /// Element-wise unary operation (logistic, ...)
    UnaryElementWise {
        op: UnaryOp,
        input: TensorId,
        output: TensorId,
    },

    // ─── Constant-foldable ops (eliminated before codegen) ──────

    /// Extract tensor shape as INT32 vector
    Shape { input: TensorId, output: TensorId },

    /// Stack N tensors along a new axis
    Pack { inputs: Vec<TensorId>, output: TensorId, axis: i32 },

    /// Slice with begin/end/strides and masks
    StridedSlice {
        input: TensorId,
        begin: TensorId,
        end: TensorId,
        strides: TensorId,
        output: TensorId,
        begin_mask: i32,
        end_mask: i32,
        shrink_axis_mask: i32,
    },

    /// Concatenate tensors along axis
    Concatenation { inputs: Vec<TensorId>, output: TensorId, axis: i32 },

    /// Index lookup along axis
    Gather { input: TensorId, indices: TensorId, output: TensorId, axis: i32 },

    /// Reduction along axes (product, max, min)
    Reduce { op: ReduceOp, input: TensorId, axes: TensorId, output: TensorId },

    /// Generate integer range [start, limit) with step delta
    Range { start: TensorId, limit: TensorId, delta: TensorId, output: TensorId },

    /// Split tensor into multiple outputs along axis
    SplitV { input: TensorId, size_splits: TensorId, axis: TensorId, outputs: Vec<TensorId> },

    /// Type cast
    Cast { input: TensorId, output: TensorId },
}

#[derive(Debug, Clone)]
pub struct Conv2dParams {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_bottom: usize,
    pub pad_left: usize,
    pub pad_right: usize,
    pub fused_activation: Option<Activation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Relu6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    FloorDiv,
    Max,
    Pow,
}

impl BinaryOp {
    pub fn name(self) -> &'static str {
        match self {
            BinaryOp::Add => "binary_add",
            BinaryOp::Mul => "binary_mul",
            BinaryOp::Sub => "binary_sub",
            BinaryOp::Div => "binary_div",
            BinaryOp::FloorDiv => "binary_floor_div",
            BinaryOp::Max => "binary_max",
            BinaryOp::Pow => "binary_pow",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Logistic,
}

impl UnaryOp {
    pub fn name(self) -> &'static str {
        match self {
            UnaryOp::Logistic => "logistic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Prod,
    Max,
    Min,
    Mean,
}

impl ReduceOp {
    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Prod => "reduce_prod",
            ReduceOp::Max => "reduce_max",
            ReduceOp::Min => "reduce_min",
            ReduceOp::Mean => "reduce_mean_hw",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FullyConnectedParams {
    pub fused_activation: Option<Activation>,
}

impl PspOp {
    pub fn inputs(&self) -> Vec<TensorId> {
        match self {
            PspOp::Conv2d {
                input,
                weights,
                bias,
                ..
            }
            | PspOp::FullyConnected {
                input,
                weights,
                bias,
                ..
            } => {
                let mut v = vec![*input, *weights];
                // bias is optional
                if let Some(b) = bias {
                    v.push(*b);
                }
                v
            }
            PspOp::MaxPool2x2 { input, .. }
            | PspOp::Reshape { input, .. }
            | PspOp::Softmax { input, .. }
            | PspOp::Shape { input, .. }
            | PspOp::Cast { input, .. }
            | PspOp::UnaryElementWise { input, .. } => vec![*input],
            PspOp::ElementWise {
                input_a, input_b, ..
            } => vec![*input_a, *input_b],
            PspOp::Pack { inputs, .. } | PspOp::Concatenation { inputs, .. } => inputs.clone(),
            PspOp::StridedSlice {
                input,
                begin,
                end,
                strides,
                ..
            } => vec![*input, *begin, *end, *strides],
            PspOp::Gather {
                input, indices, ..
            } => vec![*input, *indices],
            PspOp::Reduce { input, axes, .. } => vec![*input, *axes],
            PspOp::Range {
                start,
                limit,
                delta,
                ..
            } => vec![*start, *limit, *delta],
            PspOp::SplitV {
                input,
                size_splits,
                axis,
                ..
            } => vec![*input, *size_splits, *axis],
        }
    }

    pub fn output(&self) -> TensorId {
        match self {
            PspOp::Conv2d { output, .. }
            | PspOp::FullyConnected { output, .. }
            | PspOp::MaxPool2x2 { output, .. }
            | PspOp::Reshape { output, .. }
            | PspOp::Softmax { output, .. }
            | PspOp::ElementWise { output, .. }
            | PspOp::UnaryElementWise { output, .. }
            | PspOp::Shape { output, .. }
            | PspOp::Pack { output, .. }
            | PspOp::StridedSlice { output, .. }
            | PspOp::Concatenation { output, .. }
            | PspOp::Gather { output, .. }
            | PspOp::Reduce { output, .. }
            | PspOp::Range { output, .. }
            | PspOp::Cast { output, .. } => *output,
            PspOp::SplitV { .. } => panic!("SplitV has multiple outputs; use outputs()"),
        }
    }

    pub fn all_outputs(&self) -> Vec<TensorId> {
        match self {
            PspOp::SplitV { outputs, .. } => outputs.clone(),
            other => vec![other.output()],
        }
    }
}
