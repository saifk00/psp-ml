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
            | PspOp::Softmax { input, .. } => vec![*input],
        }
    }

    pub fn output(&self) -> TensorId {
        match self {
            PspOp::Conv2d { output, .. }
            | PspOp::FullyConnected { output, .. }
            | PspOp::MaxPool2x2 { output, .. }
            | PspOp::Reshape { output, .. }
            | PspOp::Softmax { output, .. } => *output,
        }
    }
}
