//! PSP Neural Network Kernels
//!
//! This module contains the operator implementations that the compiler emits.
//!
//! - `naive`: Reference implementations (7-loop conv, etc.)
//! - Root module: Optimized VFPU-accelerated kernels

pub mod naive;

use core::option::Option;

#[cfg(target_os = "psp")]
use psp::vfpu_asm;

const VFPU_Q: usize = 4;

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

#[inline]
const fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ============================================================================
// VFPU Primitives
// ============================================================================

#[repr(align(16))]
struct Tile([f32; 16]);

/// Load a 4x4 tile from a row-major matrix into an aligned buffer.
/// Zero-pads if the tile extends past matrix boundaries.
#[inline]
fn load_tile(src: &[f32], cols: usize, row: usize, col: usize, max_rows: usize, max_cols: usize, dst: &mut Tile) {
    dst.0 = [0.0; 16];
    let rows = if row + 4 <= max_rows { 4 } else { max_rows - row };
    let cs = if col + 4 <= max_cols { 4 } else { max_cols - col };
    for r in 0..rows {
        for c in 0..cs {
            dst.0[r * 4 + c] = src[(row + r) * cols + col + c];
        }
    }
}

/// Store a 4x4 tile from an aligned buffer to a row-major matrix.
/// Clips to matrix boundaries.
#[inline]
fn store_tile(src: &Tile, dst: &mut [f32], cols: usize, row: usize, col: usize, max_rows: usize, max_cols: usize) {
    let rows = if row + 4 <= max_rows { 4 } else { max_rows - row };
    let cs = if col + 4 <= max_cols { 4 } else { max_cols - col };
    for r in 0..rows {
        for c in 0..cs {
            dst[(row + r) * cols + col + c] = src.0[r * 4 + c];
        }
    }
}

/// VFPU 4x4 multiply-accumulate: acc += a * b
#[cfg(target_os = "psp")]
#[inline]
fn vfpu_mul_acc(a: &Tile, b: &Tile, acc: &mut Tile) {
    unsafe {
        vfpu_asm!(
            // Load A tile into M000
            "lv.q R000,  0({0})",
            "lv.q R001, 16({0})",
            "lv.q R002, 32({0})",
            "lv.q R003, 48({0})",
            // Load B tile into M100
            "lv.q R100,  0({1})",
            "lv.q R101, 16({1})",
            "lv.q R102, 32({1})",
            "lv.q R103, 48({1})",
            // Load accumulator into M200
            "lv.q R200,  0({2})",
            "lv.q R201, 16({2})",
            "lv.q R202, 32({2})",
            "lv.q R203, 48({2})",
            // Multiply: M300 = M000 * M100
            "vmmul.q M300, M000, M100",
            // Accumulate: M200 += M300
            "vadd.q R200, R200, R300",
            "vadd.q R201, R201, R301",
            "vadd.q R202, R202, R302",
            "vadd.q R203, R203, R303",
            // Store accumulator
            "sv.q R200,  0({2})",
            "sv.q R201, 16({2})",
            "sv.q R202, 32({2})",
            "sv.q R203, 48({2})",
            in(reg) a.0.as_ptr(),
            in(reg) b.0.as_ptr(),
            in(reg) acc.0.as_mut_ptr(),
            options(nostack),
        );
    }
}

/// Scalar fallback for host builds.
#[cfg(not(target_os = "psp"))]
#[inline]
fn vfpu_mul_acc(a: &Tile, b: &Tile, acc: &mut Tile) {
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                acc.0[i * 4 + j] += a.0[i * 4 + k] * b.0[k * 4 + j];
            }
        }
    }
}

/// Matrix multiply: C[M,N] = A[M,K] @ B[K,N]
///
/// Tiles computation into 4x4 blocks for VFPU registers.
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let tiles_m = div_ceil(m, VFPU_Q);
    let tiles_n = div_ceil(n, VFPU_Q);
    let tiles_k = div_ceil(k, VFPU_Q);

    for ti in 0..tiles_m {
        for tj in 0..tiles_n {
            let mut acc = Tile([0.0; 16]);

            for tk in 0..tiles_k {
                let mut a_tile = Tile([0.0; 16]);
                let mut b_tile = Tile([0.0; 16]);

                load_tile(a, k, ti * 4, tk * 4, m, k, &mut a_tile);
                load_tile(b, n, tk * 4, tj * 4, k, n, &mut b_tile);

                vfpu_mul_acc(&a_tile, &b_tile, &mut acc);
            }

            store_tile(&acc, c, n, ti * 4, tj * 4, m, n);
        }
    }
}

/// Element-wise ReLU: x = max(0, x)
pub fn relu(_data: &mut [f32]) {
    todo!("VFPU relu")
}

/// Vector dot product
pub fn dot(_a: &[f32], _b: &[f32]) -> f32 {
    todo!("VFPU dot")
}
