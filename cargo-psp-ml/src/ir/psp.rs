//! PSP-specific IR
//! each op maps to a kernel
//!

use crate::ir::graph::{Graph, TensorId, TensorKind};

/// A lowered PSP model: pure IR graph paired with the raw model bytes.
/// Constant tensors reference into `model_data` via `TensorKind::Constant { offset, len }`.
#[derive(Debug)]
pub struct PspModel {
    pub graph: Graph<PspOp>,
    pub model_data: Vec<u8>,
}

impl PspModel {
    /// Read an INT32 constant tensor as `Vec<i32>`.
    /// Returns `None` if the tensor is not `TensorKind::Constant`.
    pub fn read_i32_const(&self, tid: TensorId) -> Option<Vec<i32>> {
        let t = self.graph.tensor(tid);
        if let TensorKind::Constant { offset, len } = t.kind {
            Some(
                self.model_data[offset..offset + len]
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        } else {
            None
        }
    }
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

    /// Depthwise 2D convolution (per-channel, depth_multiplier=1)
    DepthwiseConv2d {
        input: TensorId,
        weights: TensorId,
        bias: Option<TensorId>,
        output: TensorId,
        params: Conv2dParams,
    },

    /// 2D pooling (max or average)
    Pool2d {
        pool_type: PoolType,
        input: TensorId,
        output: TensorId,
        filter: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 4],
    },

    /// Reshape (zero-cost pointer reinterpret)
    Reshape {
        input: TensorId,
        output: TensorId,
        /// Optional shape tensor (from TFLite RESHAPE). Used for dynamic shape inference.
        shape_tensor: Option<TensorId>,
        /// TFLite builtin new_shape (may contain -1 for dynamic dim).
        builtin_shape: Option<Vec<i32>>,
    },

    /// Remove a size-1 dimension (zero-cost pointer reinterpret)
    Squeeze {
        input: TensorId,
        output: TensorId,
        axis: usize,
    },

    /// Insert a size-1 dimension (zero-cost pointer reinterpret)
    ExpandDims {
        input: TensorId,
        output: TensorId,
        axis: usize,
    },

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
    Pack {
        inputs: Vec<TensorId>,
        output: TensorId,
        axis: i32,
    },

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
    Concatenation {
        inputs: Vec<TensorId>,
        output: TensorId,
        axis: i32,
    },

    /// Index lookup along axis
    Gather {
        input: TensorId,
        indices: TensorId,
        output: TensorId,
        axis: i32,
    },

    /// Reduction along axes (product, max, min)
    Reduce {
        op: ReduceOp,
        input: TensorId,
        axes: TensorId,
        output: TensorId,
    },

    /// Generate integer range [start, limit) with step delta
    Range {
        start: TensorId,
        limit: TensorId,
        delta: TensorId,
        output: TensorId,
    },

    /// Split tensor into multiple outputs along axis
    SplitV {
        input: TensorId,
        size_splits: TensorId,
        axis: TensorId,
        outputs: Vec<TensorId>,
    },

    /// Type cast
    Cast { input: TensorId, output: TensorId },

    /// Zero-pad an NHWC tensor
    Pad {
        input: TensorId,
        paddings: TensorId,
        output: TensorId,
    },

    /// Permute tensor dimensions (up to 4D)
    Transpose {
        input: TensorId,
        perm: TensorId,
        output: TensorId,
    },

    /// Reverse elements along an axis
    ReverseV2 {
        input: TensorId,
        axis: TensorId,
        output: TensorId,
    },

    // ─── Pre-fusion ops (replaced by fusion pass) ───────────────
    /// Raw RFFT2D from TFLite (pre-fusion, output is COMPLEX64)
    Rfft2d {
        input: TensorId,
        fft_length: TensorId,
        output: TensorId,
    },

    // ─── Fused ops (produced by fusion pass) ────────────────────
    /// Fused real FFT: input F32 → output F32 (real parts of frequency bins)
    Rfft {
        input: TensorId,
        output: TensorId,
        fft_length: usize,
    },
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
pub enum PoolType {
    Max,
    Average,
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

impl std::fmt::Display for PspOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PspOp::Conv2d {
                input,
                weights,
                bias,
                output,
                params,
            } => {
                write!(f, "Conv2d(t{}, t{}", input, weights)?;
                if let Some(b) = bias {
                    write!(f, ", t{}", b)?;
                }
                write!(
                    f,
                    " → t{}; kernel={}x{} stride={}x{} pad={},{},{},{}",
                    output,
                    params.kernel_h,
                    params.kernel_w,
                    params.stride_h,
                    params.stride_w,
                    params.pad_top,
                    params.pad_bottom,
                    params.pad_left,
                    params.pad_right
                )?;
                if let Some(act) = &params.fused_activation {
                    write!(f, " {}", act)?;
                }
                write!(f, ")")
            }
            PspOp::FullyConnected {
                input,
                weights,
                bias,
                output,
                fused_activation,
            } => {
                write!(f, "FullyConnected(t{}, t{}", input, weights)?;
                if let Some(b) = bias {
                    write!(f, ", t{}", b)?;
                }
                write!(f, " → t{}", output)?;
                if let Some(act) = &fused_activation.fused_activation {
                    write!(f, "; {}", act)?;
                }
                write!(f, ")")
            }
            PspOp::DepthwiseConv2d {
                input,
                weights,
                bias,
                output,
                params,
            } => {
                write!(f, "DepthwiseConv2d(t{}, t{}", input, weights)?;
                if let Some(b) = bias {
                    write!(f, ", t{}", b)?;
                }
                write!(
                    f,
                    " → t{}; kernel={}x{} stride={}x{} pad={},{},{},{}",
                    output,
                    params.kernel_h,
                    params.kernel_w,
                    params.stride_h,
                    params.stride_w,
                    params.pad_top,
                    params.pad_bottom,
                    params.pad_left,
                    params.pad_right
                )?;
                if let Some(act) = &params.fused_activation {
                    write!(f, " {}", act)?;
                }
                write!(f, ")")
            }
            PspOp::Pool2d {
                pool_type,
                input,
                output,
                filter,
                stride,
                ..
            } => {
                write!(
                    f,
                    "Pool2d(t{} → t{}; {} filter={}x{} stride={}x{})",
                    input, output, pool_type, filter[0], filter[1], stride[0], stride[1]
                )
            }
            PspOp::Reshape { input, output, .. } => write!(f, "Reshape(t{} → t{})", input, output),
            PspOp::Squeeze {
                input,
                output,
                axis,
            } => write!(f, "Squeeze(t{} → t{}; axis={})", input, output, axis),
            PspOp::ExpandDims {
                input,
                output,
                axis,
            } => write!(f, "ExpandDims(t{} → t{}; axis={})", input, output, axis),
            PspOp::Softmax { input, output } => write!(f, "Softmax(t{} → t{})", input, output),
            PspOp::ElementWise {
                op,
                input_a,
                input_b,
                output,
            } => {
                write!(
                    f,
                    "ElementWise(t{}, t{} → t{}; {})",
                    input_a, input_b, output, op
                )
            }
            PspOp::UnaryElementWise { op, input, output } => {
                write!(f, "UnaryElementWise(t{} → t{}; {})", input, output, op)
            }
            PspOp::Shape { input, output } => write!(f, "Shape(t{} → t{})", input, output),
            PspOp::Pack {
                inputs,
                output,
                axis,
            } => {
                write!(f, "Pack(")?;
                for (i, id) in inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "t{}", id)?;
                }
                write!(f, " → t{}; axis={})", output, axis)
            }
            PspOp::StridedSlice {
                input,
                begin,
                end,
                strides,
                output,
                begin_mask,
                end_mask,
                shrink_axis_mask,
            } => {
                write!(
                    f,
                    "StridedSlice(t{}, t{}, t{}, t{} → t{}; begin_mask={} end_mask={} shrink={})",
                    input, begin, end, strides, output, begin_mask, end_mask, shrink_axis_mask
                )
            }
            PspOp::Concatenation {
                inputs,
                output,
                axis,
            } => {
                write!(f, "Concatenation(")?;
                for (i, id) in inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "t{}", id)?;
                }
                write!(f, " → t{}; axis={})", output, axis)
            }
            PspOp::Gather {
                input,
                indices,
                output,
                axis,
            } => {
                write!(
                    f,
                    "Gather(t{}, t{} → t{}; axis={})",
                    input, indices, output, axis
                )
            }
            PspOp::Reduce {
                op,
                input,
                axes,
                output,
            } => {
                write!(f, "Reduce(t{}, t{} → t{}; {})", input, axes, output, op)
            }
            PspOp::Range {
                start,
                limit,
                delta,
                output,
            } => {
                write!(f, "Range(t{}, t{}, t{} → t{})", start, limit, delta, output)
            }
            PspOp::SplitV {
                input,
                size_splits,
                axis,
                outputs,
            } => {
                write!(f, "SplitV(t{}, t{}, t{} → ", input, size_splits, axis)?;
                for (i, id) in outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "t{}", id)?;
                }
                write!(f, ")")
            }
            PspOp::Cast { input, output } => write!(f, "Cast(t{} → t{})", input, output),
            PspOp::Pad {
                input,
                paddings,
                output,
            } => {
                write!(f, "Pad(t{}, t{} → t{})", input, paddings, output)
            }
            PspOp::Transpose {
                input,
                perm,
                output,
            } => {
                write!(f, "Transpose(t{}, t{} → t{})", input, perm, output)
            }
            PspOp::ReverseV2 {
                input,
                axis,
                output,
            } => {
                write!(f, "ReverseV2(t{}, t{} → t{})", input, axis, output)
            }
            PspOp::Rfft2d {
                input,
                fft_length,
                output,
            } => {
                write!(f, "Rfft2d(t{}, t{} → t{})", input, fft_length, output)
            }
            PspOp::Rfft {
                input,
                output,
                fft_length,
            } => {
                write!(f, "Rfft(t{} → t{}; n={})", input, output, fft_length)
            }
        }
    }
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::Relu => write!(f, "Relu"),
            Activation::Relu6 => write!(f, "Relu6"),
        }
    }
}

impl std::fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "Add"),
            BinaryOp::Mul => write!(f, "Mul"),
            BinaryOp::Sub => write!(f, "Sub"),
            BinaryOp::Div => write!(f, "Div"),
            BinaryOp::FloorDiv => write!(f, "FloorDiv"),
            BinaryOp::Max => write!(f, "Max"),
            BinaryOp::Pow => write!(f, "Pow"),
        }
    }
}

impl std::fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Logistic => write!(f, "Logistic"),
        }
    }
}

impl std::fmt::Display for PoolType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolType::Max => write!(f, "Max"),
            PoolType::Average => write!(f, "Avg"),
        }
    }
}

impl std::fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReduceOp::Prod => write!(f, "Prod"),
            ReduceOp::Max => write!(f, "Max"),
            ReduceOp::Min => write!(f, "Min"),
            ReduceOp::Mean => write!(f, "Mean"),
        }
    }
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
            | PspOp::DepthwiseConv2d {
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
            PspOp::Pool2d { input, .. }
            | PspOp::Reshape { input, .. }
            | PspOp::Squeeze { input, .. }
            | PspOp::ExpandDims { input, .. }
            | PspOp::Softmax { input, .. }
            | PspOp::Shape { input, .. }
            | PspOp::Cast { input, .. }
            | PspOp::UnaryElementWise { input, .. }
            | PspOp::Rfft { input, .. } => vec![*input],
            PspOp::Pad {
                input, paddings, ..
            } => vec![*input, *paddings],
            PspOp::Transpose { input, perm, .. } => vec![*input, *perm],
            PspOp::ReverseV2 { input, axis, .. } => vec![*input, *axis],
            PspOp::Rfft2d {
                input, fft_length, ..
            } => vec![*input, *fft_length],
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
            PspOp::Gather { input, indices, .. } => vec![*input, *indices],
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

    /// Returns IDs of constant tensors (weights and biases) used by this op.
    ///
    /// Used by the IO planner to compute weight-memory budgets per op. Only
    /// weight-bearing ops return non-empty results; other ops are assumed to
    /// have had their small structural constants (axes, paddings, perms)
    /// eliminated by `const_fold` upstream.
    pub fn const_tensors(&self) -> Vec<TensorId> {
        match self {
            PspOp::Conv2d { weights, bias, .. }
            | PspOp::DepthwiseConv2d { weights, bias, .. }
            | PspOp::FullyConnected { weights, bias, .. } => {
                let mut v = vec![*weights];
                if let Some(b) = bias {
                    v.push(*b);
                }
                v
            }
            _ => Vec::new(),
        }
    }

    pub fn output(&self) -> TensorId {
        match self {
            PspOp::Conv2d { output, .. }
            | PspOp::DepthwiseConv2d { output, .. }
            | PspOp::FullyConnected { output, .. }
            | PspOp::Pool2d { output, .. }
            | PspOp::Reshape { output, .. }
            | PspOp::Squeeze { output, .. }
            | PspOp::ExpandDims { output, .. }
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
            | PspOp::Cast { output, .. }
            | PspOp::Pad { output, .. }
            | PspOp::Transpose { output, .. }
            | PspOp::ReverseV2 { output, .. }
            | PspOp::Rfft2d { output, .. }
            | PspOp::Rfft { output, .. } => *output,
            PspOp::SplitV { .. } => panic!("SplitV has multiple outputs; use outputs()"),
        }
    }

    pub fn all_outputs(&self) -> Vec<TensorId> {
        match self {
            PspOp::SplitV { outputs, .. } => outputs.clone(),
            other => vec![other.output()],
        }
    }

    /// Replace all occurrences of `old` with `new` in input and output fields.
    pub fn replace_tensor_id(&mut self, old: TensorId, new: TensorId) {
        let r = |id: &mut TensorId| {
            if *id == old {
                *id = new;
            }
        };
        let r_opt = |id: &mut Option<TensorId>| {
            if *id == Some(old) {
                *id = Some(new);
            }
        };
        match self {
            PspOp::Conv2d {
                input,
                weights,
                bias,
                output,
                ..
            }
            | PspOp::DepthwiseConv2d {
                input,
                weights,
                bias,
                output,
                ..
            }
            | PspOp::FullyConnected {
                input,
                weights,
                bias,
                output,
                ..
            } => {
                r(input);
                r(weights);
                r_opt(bias);
                r(output);
            }
            PspOp::Reshape {
                input,
                output,
                shape_tensor,
                ..
            } => {
                r(input);
                r(output);
                r_opt(shape_tensor);
            }
            PspOp::Pool2d { input, output, .. }
            | PspOp::Squeeze { input, output, .. }
            | PspOp::ExpandDims { input, output, .. }
            | PspOp::Softmax { input, output }
            | PspOp::UnaryElementWise { input, output, .. }
            | PspOp::Shape { input, output }
            | PspOp::Cast { input, output }
            | PspOp::Rfft { input, output, .. } => {
                r(input);
                r(output);
            }
            PspOp::ElementWise {
                input_a,
                input_b,
                output,
                ..
            } => {
                r(input_a);
                r(input_b);
                r(output);
            }
            PspOp::Pad {
                input,
                paddings,
                output,
            } => {
                r(input);
                r(paddings);
                r(output);
            }
            PspOp::Transpose {
                input,
                perm,
                output,
            } => {
                r(input);
                r(perm);
                r(output);
            }
            PspOp::ReverseV2 {
                input,
                axis,
                output,
            } => {
                r(input);
                r(axis);
                r(output);
            }
            PspOp::Rfft2d {
                input,
                fft_length,
                output,
            } => {
                r(input);
                r(fft_length);
                r(output);
            }
            PspOp::StridedSlice {
                input,
                begin,
                end,
                strides,
                output,
                ..
            } => {
                r(input);
                r(begin);
                r(end);
                r(strides);
                r(output);
            }
            PspOp::Gather {
                input,
                indices,
                output,
                ..
            } => {
                r(input);
                r(indices);
                r(output);
            }
            PspOp::Reduce {
                input,
                axes,
                output,
                ..
            } => {
                r(input);
                r(axes);
                r(output);
            }
            PspOp::Range {
                start,
                limit,
                delta,
                output,
            } => {
                r(start);
                r(limit);
                r(delta);
                r(output);
            }
            PspOp::SplitV {
                input,
                size_splits,
                axis,
                outputs,
            } => {
                r(input);
                r(size_splits);
                r(axis);
                for o in outputs {
                    r(o);
                }
            }
            PspOp::Pack { inputs, output, .. } | PspOp::Concatenation { inputs, output, .. } => {
                for i in inputs {
                    r(i);
                }
                r(output);
            }
        }
    }
}
