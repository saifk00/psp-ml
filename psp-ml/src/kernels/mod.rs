//! PSP Neural Network Kernels
//!
//! This module contains the operator implementations that the compiler emits.
//!
//! - `naive`: Reference implementations that work on any target.
//! - Root module: Optimized VFPU-accelerated kernels (PSP uses VFPU, host uses scalar fallbacks).

pub mod naive;

#[cfg(target_os = "psp")]
use psp::vfpu_asm;

const VFPU_Q: usize = 4;

// ============================================================================
// VFPU Primitives
// ============================================================================

#[repr(align(16))]
struct Tile([f32; 16]);

/// Load a 4x4 tile from a row-major matrix into an aligned buffer.
/// Zero-pads if the tile extends past matrix boundaries.
#[inline]
fn load_tile(
    src: &[f32],
    cols: usize,
    row: usize,
    col: usize,
    max_rows: usize,
    max_cols: usize,
    dst: &mut Tile,
) {
    dst.0 = [0.0; 16];
    let rows = if row + 4 <= max_rows {
        4
    } else {
        max_rows - row
    };
    let cs = if col + 4 <= max_cols {
        4
    } else {
        max_cols - col
    };
    for r in 0..rows {
        for c in 0..cs {
            dst.0[r * 4 + c] = src[(row + r) * cols + col + c];
        }
    }
}

/// Store a 4x4 tile from an aligned buffer to a row-major matrix.
/// Clips to matrix boundaries.
#[inline]
fn store_tile(
    src: &Tile,
    dst: &mut [f32],
    cols: usize,
    row: usize,
    col: usize,
    max_rows: usize,
    max_cols: usize,
) {
    let rows = if row + 4 <= max_rows {
        4
    } else {
        max_rows - row
    };
    let cs = if col + 4 <= max_cols {
        4
    } else {
        max_cols - col
    };
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
            in(reg) (a.0.as_ptr()),
            in(reg) (b.0.as_ptr()),
            in(reg) (acc.0.as_mut_ptr()),
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

/// VFPU 4x4 multiply-accumulate with B transposed: acc += a @ b^T
///
/// Uses E100 (transposed view of M100) to avoid software transpose.
#[cfg(target_os = "psp")]
#[inline]
fn vfpu_mul_acc_bt(a: &Tile, b: &Tile, acc: &mut Tile) {
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
            // Multiply: M300 = M000 @ M100^T (E100 = transposed view)
            "vmmul.q M300, M000, E100",
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
            in(reg) (a.0.as_ptr()),
            in(reg) (b.0.as_ptr()),
            in(reg) (acc.0.as_mut_ptr()),
            options(nostack),
        );
    }
}

/// Scalar fallback for B-transposed multiply-accumulate.
#[cfg(not(target_os = "psp"))]
#[inline]
fn vfpu_mul_acc_bt(a: &Tile, b: &Tile, acc: &mut Tile) {
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                acc.0[i * 4 + j] += a.0[i * 4 + k] * b.0[j * 4 + k];
            }
        }
    }
}

/// VFPU ReLU on 4 aligned floats: buf = max(buf, 0)
#[cfg(target_os = "psp")]
#[inline]
fn vfpu_relu4(buf: &mut Tile) {
    unsafe {
        vfpu_asm!(
            "vzero.q R100",
            "lv.q R000, 0({0})",
            "vmax.q R000, R000, R100",
            "sv.q R000, 0({0})",
            in(reg) (buf.0.as_mut_ptr()),
            options(nostack),
        );
    }
}

#[cfg(not(target_os = "psp"))]
#[inline]
fn vfpu_relu4(buf: &mut Tile) {
    for x in buf.0[..4].iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// Load a VFPU_Q×VFPU_Q tile from a pre-padded row-major matrix.
///
/// All dimensions must be exact multiples of VFPU_Q — no bounds checking.
#[inline]
fn load_tile_direct(src: &[f32], stride: usize, row: usize, col: usize) -> Tile {
    let mut t = Tile([0.0; VFPU_Q * VFPU_Q]);
    for r in 0..VFPU_Q {
        let base = (row + r) * stride + col;
        t.0[r * VFPU_Q..r * VFPU_Q + VFPU_Q]
            .copy_from_slice(&src[base..base + VFPU_Q]);
    }
    t
}

/// Store a VFPU_Q×VFPU_Q tile to a pre-padded row-major matrix.
///
/// All dimensions must be exact multiples of VFPU_Q — no bounds checking.
#[inline]
fn store_tile_direct(src: &Tile, dst: &mut [f32], stride: usize, row: usize, col: usize) {
    for r in 0..VFPU_Q {
        let base = (row + r) * stride + col;
        dst[base..base + VFPU_Q]
            .copy_from_slice(&src.0[r * VFPU_Q..r * VFPU_Q + VFPU_Q]);
    }
}

// ============================================================================
// Public Optimized Kernels
// ============================================================================

#[inline]
const fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Matrix multiply with B transposed: C[M,N] = A[M,K] @ B[N,K]^T
///
/// B is stored as [N, K] in memory. Tiles computation into 4x4 blocks
/// using VFPU E_XXX transposed register views.
pub fn matmul_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let tiles_m = div_ceil(m, VFPU_Q);
    let tiles_n = div_ceil(n, VFPU_Q);
    let tiles_k = div_ceil(k, VFPU_Q);

    for ti in 0..tiles_m {
        for tj in 0..tiles_n {
            let mut acc = Tile([0.0; 16]);

            for tk in 0..tiles_k {
                let mut a_tile = Tile([0.0; 16]);
                let mut b_tile = Tile([0.0; 16]);

                load_tile(a, k, ti * VFPU_Q, tk * VFPU_Q, m, k, &mut a_tile);
                load_tile(b, k, tj * VFPU_Q, tk * VFPU_Q, n, k, &mut b_tile);

                vfpu_mul_acc_bt(&a_tile, &b_tile, &mut acc);
            }

            store_tile(&acc, c, n, ti * VFPU_Q, tj * VFPU_Q, m, n);
        }
    }
}

/// Tiled matmul for pre-padded inputs: C[M,N] = A[M,K] @ B[N,K]^T
///
/// ALL dimensions must be exact multiples of VFPU_Q (guaranteed by codegen padding).
/// No bounds checking, no conditional tile zeroing.
///
/// - `a`: [m_tiles*VFPU_Q, k_tiles*VFPU_Q] — im2col output (row-major)
/// - `b`: [n_tiles*VFPU_Q, k_tiles*VFPU_Q] — padded weights (row-major, transposed via E_XXX)
/// - `c`: [m_tiles*VFPU_Q, n_tiles*VFPU_Q] — output
pub fn matmul_bt_tiled(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m_tiles: usize,
    k_tiles: usize,
    n_tiles: usize,
) {
    let k = k_tiles * VFPU_Q;
    let n = n_tiles * VFPU_Q;

    for ti in 0..m_tiles {
        for tj in 0..n_tiles {
            let mut acc = Tile([0.0; VFPU_Q * VFPU_Q]);

            for tk in 0..k_tiles {
                let a_tile = load_tile_direct(a, k, ti * VFPU_Q, tk * VFPU_Q);
                let b_tile = load_tile_direct(b, k, tj * VFPU_Q, tk * VFPU_Q);
                vfpu_mul_acc_bt(&a_tile, &b_tile, &mut acc);
            }

            store_tile_direct(&acc, c, n, ti * VFPU_Q, tj * VFPU_Q);
        }
    }
}

/// Element-wise ReLU: x = max(0, x)
pub fn relu(data: &mut [f32]) {
    let chunks = data.len() / VFPU_Q;
    let mut buf = Tile([0.0; 16]);

    for i in 0..chunks {
        let off = i * VFPU_Q;
        buf.0[..4].copy_from_slice(&data[off..off + 4]);
        vfpu_relu4(&mut buf);
        data[off..off + 4].copy_from_slice(&buf.0[..4]);
    }

    // Scalar tail for remaining elements
    for x in data[chunks * VFPU_Q..].iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// im2col: Rearrange NHWC input patches into a 2D column matrix.
///
/// - `input`:     [N, H, W, Ci]
/// - `kernel`:    [Kh, Kw]
/// - `stride`:    [Sh, Sw]
/// - `padding`:   [Ph, Pw] — symmetric padding on each side
/// - `output_hw`: [Ho, Wo]
/// - `col`:       output matrix [N*Ho*Wo, Kh*Kw*Ci]
pub fn im2col(
    input: &[f32],
    input_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    output_hw: [usize; 2],
    col: &mut [f32],
) {
    let [n, h, w, ci] = input_shape;
    let [kh, kw] = kernel;
    let [sh, sw] = stride;
    let [pad_h, pad_w] = padding;
    let [ho, wo] = output_hw;
    let k = kh * kw * ci;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                let row = batch * (ho * wo) + oy * wo + ox;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = (oy * sh + ky) as isize - pad_h as isize;
                        let ix = (ox * sw + kx) as isize - pad_w as isize;
                        for ic in 0..ci {
                            let col_idx = ky * (kw * ci) + kx * ci + ic;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                let in_idx = batch * (h * w * ci)
                                    + (iy as usize) * (w * ci)
                                    + (ix as usize) * ci
                                    + ic;
                                col[row * k + col_idx] = input[in_idx];
                            } else {
                                col[row * k + col_idx] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// im2col for stride-1 convolutions with VFPU_Q-padded output columns.
///
/// Output layout: `[N*Ho*Wo, k_padded]` where `k_padded = ceil(Kh*Kw*Ci, VFPU_Q)`.
/// Padding columns are left as-is — caller must provide a zero-initialized buffer.
///
/// - `input`:     [N, H, W, Ci]
/// - `kernel`:    [Kh, Kw]
/// - `padding`:   [Ph, Pw] — symmetric padding on each side
/// - `output_hw`: [Ho, Wo]
/// - `col`:       output matrix [N*Ho*Wo, k_padded]
pub fn im2col_padded(
    input: &[f32],
    input_shape: [usize; 4],
    kernel: [usize; 2],
    padding: [usize; 2],
    output_hw: [usize; 2],
    col: &mut [f32],
) {
    let [n, h, w, ci] = input_shape;
    let [kh, kw] = kernel;
    let [pad_h, pad_w] = padding;
    let [ho, wo] = output_hw;
    let k = kh * kw * ci;
    let k_padded = div_ceil(k, VFPU_Q) * VFPU_Q;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                let row = batch * (ho * wo) + oy * wo + ox;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = (oy + ky) as isize - pad_h as isize;
                        let ix = (ox + kx) as isize - pad_w as isize;
                        for ic in 0..ci {
                            let col_idx = ky * (kw * ci) + kx * ci + ic;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                let in_idx = batch * (h * w * ci)
                                    + (iy as usize) * (w * ci)
                                    + (ix as usize) * ci
                                    + ic;
                                col[row * k_padded + col_idx] = input[in_idx];
                            } else {
                                col[row * k_padded + col_idx] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Add bias to each row: data[row, col] += bias[col]
///
/// - `data`: [rows, cols] mutable
/// - `bias`: [cols]
pub fn bias_add(data: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            data[r * cols + c] += bias[c];
        }
    }
}
