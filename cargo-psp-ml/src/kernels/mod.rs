//! PSP Neural Network Kernels
//!
//! This module contains the operator implementations that the compiler emits.
//!
//! - `naive`: Reference implementations (7-loop conv, etc.)
//! - Root module: Optimized VFPU-accelerated kernels

pub mod naive;

use core::option::Option;

// ============================================================================
// Optimized Kernels (VFPU-accelerated)
// ============================================================================

/// 2D Convolution using im2col + VFPU matrix multiply
///
/// - `input`:  [N, H, W, Ci]
/// - `filter`: [Co, Kh, Kw, Ci]
/// - `bias`:   [Co]
/// - `output`: [N, Ho, Wo, Co]
/// - `scratch`: workspace for im2col, size = Ho * Wo * Kh * Kw * Ci
pub fn conv2d(
    _input: &[f32],
    _input_shape: [usize; 4],
    _filter: &[f32],
    _filter_shape: [usize; 4],
    _bias: Option<&[f32]>,
    _stride: [usize; 2],
    _output: &mut [f32],
    _output_shape: [usize; 4],
    _scratch: &mut [f32],
) {
    todo!("VFPU conv2d")
}

/// 2D Max Pooling
pub fn max_pool2d(
    _input: &[f32],
    _input_shape: [usize; 4],
    _kernel: [usize; 2],
    _stride: [usize; 2],
    _output: &mut [f32],
    _output_shape: [usize; 4],
) {
    todo!("VFPU max_pool2d")
}

/// Reshape (zero-copy view change, or copy if needed)
pub fn reshape(input: &[f32], output: &mut [f32]) {
    // Reshape is just a memcpy, no VFPU needed
    naive::reshape(input, output);
}

/// Fully Connected with ReLU using VFPU matrix multiply
///
/// - `input`:   [in_features]
/// - `weights`: [out_features, in_features]
/// - `bias`:    [out_features]
/// - `output`:  [out_features]
pub fn fully_connected_relu(
    _input: &[f32],
    _in_features: usize,
    _weights: &[f32],
    _bias: &[f32],
    _output: &mut [f32],
    _out_features: usize,
) {
    todo!("VFPU fully_connected_relu")
}

// ============================================================================
// VFPU Primitives
// ============================================================================

/// Matrix multiply: C[M,N] = A[M,K] @ B[K,N]
///
/// Tiles computation into 4x4 blocks for VFPU registers.
pub fn matmul(_a: &[f32], _b: &[f32], _c: &mut [f32], _m: usize, _k: usize, _n: usize) {
    todo!("VFPU matmul")
}

/// Element-wise ReLU: x = max(0, x)
pub fn relu(_data: &mut [f32]) {
    todo!("VFPU relu")
}

/// Vector dot product
pub fn dot(_a: &[f32], _b: &[f32]) -> f32 {
    todo!("VFPU dot")
}
