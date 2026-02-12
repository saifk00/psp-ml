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
/// All three buffers must be 16-byte aligned (guaranteed by codegen `Aligned16` wrappers).
///
/// On PSP, the inner k-tile loop is a single asm block:
///   - `lv.q` directly from A/B row pointers (no scalar intermediary)
///   - accumulator stays in VFPU M200 across k iterations
///   - `vzero.q` for acc init (no memset)
///   - `sv.q` directly to C row pointers
///
/// - `a`: [m_tiles*VFPU_Q, k_tiles*VFPU_Q] — im2col output (row-major, 16-byte aligned)
/// - `b`: [n_tiles*VFPU_Q, k_tiles*VFPU_Q] — padded weights (row-major, 16-byte aligned)
/// - `c`: [m_tiles*VFPU_Q, n_tiles*VFPU_Q] — output (16-byte aligned)
#[cfg(target_os = "psp")]
pub fn matmul_bt_tiled(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m_tiles: usize,
    k_tiles: usize,
    n_tiles: usize,
) {
    debug_assert!(a.as_ptr() as usize % 16 == 0, "a must be 16-byte aligned");
    debug_assert!(b.as_ptr() as usize % 16 == 0, "b must be 16-byte aligned");
    debug_assert!(c.as_ptr() as usize % 16 == 0, "c must be 16-byte aligned");

    if k_tiles == 0 {
        return;
    }

    let k = k_tiles * VFPU_Q;
    let n = n_tiles * VFPU_Q;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    for ti in 0..m_tiles {
        for tj in 0..n_tiles {
            unsafe {
                let a0 = a_ptr.add(ti * VFPU_Q * k);
                let a1 = a0.add(k);
                let a2 = a1.add(k);
                let a3 = a2.add(k);

                let b0 = b_ptr.add(tj * VFPU_Q * k);
                let b1 = b0.add(k);
                let b2 = b1.add(k);
                let b3 = b2.add(k);

                let c_base = c_ptr.add(ti * VFPU_Q * n + tj * VFPU_Q);
                let n_stride_bytes = n * core::mem::size_of::<f32>();

                vfpu_asm!(
                    "vzero.q R200",
                    "vzero.q R201",
                    "vzero.q R202",
                    "vzero.q R203",
                    "2:",
                    "lv.q R000, 0({a0})",
                    "lv.q R001, 0({a1})",
                    "lv.q R002, 0({a2})",
                    "lv.q R003, 0({a3})",
                    "lv.q R100, 0({b0})",
                    "lv.q R101, 0({b1})",
                    "lv.q R102, 0({b2})",
                    "lv.q R103, 0({b3})",
                    "vmmul.q M300, M000, E100",
                    "vadd.q R200, R200, R300",
                    "vadd.q R201, R201, R301",
                    "vadd.q R202, R202, R302",
                    "vadd.q R203, R203, R303",
                    "addiu {a0}, {a0}, 16",
                    "addiu {a1}, {a1}, 16",
                    "addiu {a2}, {a2}, 16",
                    "addiu {a3}, {a3}, 16",
                    "addiu {b0}, {b0}, 16",
                    "addiu {b1}, {b1}, 16",
                    "addiu {b2}, {b2}, 16",
                    "addiu {k}, {k}, -1",
                    "bnez {k}, 2b",
                    "addiu {b3}, {b3}, 16",  // branch delay slot
                    // Store accumulator rows to C
                    "sv.q R200, 0({c})",
                    "addu {c}, {c}, {ns}",
                    "sv.q R201, 0({c})",
                    "addu {c}, {c}, {ns}",
                    "sv.q R202, 0({c})",
                    "addu {c}, {c}, {ns}",
                    "sv.q R203, 0({c})",
                    a0 = inout(reg) (a0) => _,
                    a1 = inout(reg) (a1) => _,
                    a2 = inout(reg) (a2) => _,
                    a3 = inout(reg) (a3) => _,
                    b0 = inout(reg) (b0) => _,
                    b1 = inout(reg) (b1) => _,
                    b2 = inout(reg) (b2) => _,
                    b3 = inout(reg) (b3) => _,
                    k = inout(reg) (k_tiles) => _,
                    c = inout(reg) (c_base) => _,
                    ns = in(reg) (n_stride_bytes),
                    options(nostack),
                );
            }
        }
    }
}

/// Scalar fallback for host builds.
#[cfg(not(target_os = "psp"))]
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
            let mut acc = [0.0f32; VFPU_Q * VFPU_Q];

            for tk in 0..k_tiles {
                for r in 0..VFPU_Q {
                    for c_col in 0..VFPU_Q {
                        for kk in 0..VFPU_Q {
                            let a_val = a[(ti * VFPU_Q + r) * k + tk * VFPU_Q + kk];
                            let b_val = b[(tj * VFPU_Q + c_col) * k + tk * VFPU_Q + kk];
                            acc[r * VFPU_Q + c_col] += a_val * b_val;
                        }
                    }
                }
            }

            for r in 0..VFPU_Q {
                let base = (ti * VFPU_Q + r) * n + tj * VFPU_Q;
                c[base..base + VFPU_Q]
                    .copy_from_slice(&acc[r * VFPU_Q..r * VFPU_Q + VFPU_Q]);
            }
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
