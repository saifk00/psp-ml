//! PSP Neural Network Kernels
//!
//! This module contains the operator implementations that the compiler emits.
//!
//! - `naive`: Reference implementations that work on any target.
//! - Root module: Optimized VFPU-accelerated kernels (PSP uses VFPU, host uses scalar fallbacks).

pub mod checks;

pub mod naive;
mod vme_conv;
pub use vme_conv::*;

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
#[inline(never)]
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

    let k_stride_bytes = k * core::mem::size_of::<f32>();

    for ti in 0..m_tiles {
        for tj in 0..n_tiles {
            unsafe {
                let a0 = a_ptr.add(ti * VFPU_Q * k);
                let b0 = b_ptr.add(tj * VFPU_Q * k);
                let c_base = c_ptr.add(ti * VFPU_Q * n + tj * VFPU_Q);
                let n_stride_bytes = n * core::mem::size_of::<f32>();

                // Compute loop: accumulate M200 = sum(A_tile @ B_tile^T)
                vfpu_asm!(
                    "vzero.q R200",
                    "vzero.q R201",
                    "vzero.q R202",
                    "vzero.q R203",
                    "2:",
                    // Load A: 4 rows from a0, a0+ks, a0+2*ks, a0+3*ks
                    "lv.q R000, 0({a0})",
                    "addu {tmp}, {a0}, {ks}",
                    "lv.q R001, 0({tmp})",
                    "addu {tmp}, {tmp}, {ks}",
                    "lv.q R002, 0({tmp})",
                    "addu {tmp}, {tmp}, {ks}",
                    "lv.q R003, 0({tmp})",
                    // Load B: 4 rows from b0, b0+ks, b0+2*ks, b0+3*ks
                    "lv.q R100, 0({b0})",
                    "addu {tmp}, {b0}, {ks}",
                    "lv.q R101, 0({tmp})",
                    "addu {tmp}, {tmp}, {ks}",
                    "lv.q R102, 0({tmp})",
                    "addu {tmp}, {tmp}, {ks}",
                    "lv.q R103, 0({tmp})",
                    // Multiply and accumulate
                    "vmmul.q M300, M000, E100",
                    "vadd.q R200, R200, R300",
                    "vadd.q R201, R201, R301",
                    "vadd.q R202, R202, R302",
                    "vadd.q R203, R203, R303",
                    // Advance base pointers by one float4 column (16 bytes)
                    "addiu {a0}, {a0}, 16",
                    "addiu {k}, {k}, -1",
                    "bnez {k}, 2b",
                    "addiu {b0}, {b0}, 16",  // branch delay slot
                    a0 = inout(reg) (a0) => _,
                    b0 = inout(reg) (b0) => _,
                    ks = in(reg) (k_stride_bytes),
                    k = inout(reg) (k_tiles) => _,
                    tmp = out(reg) _,
                    options(nostack),
                );

                // Store accumulator M200 to C rows (separate block to reduce register pressure)
                vfpu_asm!(
                    "sv.q R200, 0({c})",
                    "addu {c}, {c}, {ns}",
                    "sv.q R201, 0({c})",
                    "addu {c}, {c}, {ns}",
                    "sv.q R202, 0({c})",
                    "addu {c}, {c}, {ns}",
                    "sv.q R203, 0({c})",
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
                c[base..base + VFPU_Q].copy_from_slice(&acc[r * VFPU_Q..r * VFPU_Q + VFPU_Q]);
            }
        }
    }
}

// ============================================================================
// Cache-blocked, register-blocked GEMM: C[M,N] = A[M,K] @ B[N,K]^T
// ============================================================================
//
// `matmul_bt_tiled` above computes one 4x4 output tile at a time over the full
// K extent, so it re-streams all of B for every M-tile — for BirdNET's mel
// projection (M=511, K=1025, N=96) that is ~101 MB against a 32 KB L1, i.e.
// ~6x memory-bound. This path fixes both halves of that:
//
//   * **L1 blocking.** Work on an `mc x n` slab of C at a time, held packed in
//     `cp` (12 KB at mc=32, n=96) and resident across the whole K sweep, while
//     a `n x kc` panel of B stays resident across the M sweep. Traffic drops to
//     ~8.6 MB, which is under the ~17 MB the compute time can hide.
//   * **Register blocking.** A 4x8 micro-tile issues two *independent*
//     `vmmul.q` into separate temps (the configuration `examples/roofline`
//     measured at peak), halving load traffic per FLOP versus 4x4.
//
// The VFPU has no matrix multiply-accumulate, so every `vmmul.q` costs 4
// `vadd.q`; the micro-kernel ceiling is ~1992 MFLOP/s (86% of the 2309
// `vmmul.q` peak) at any block size. Do not expect more.
//
// Register map (4 of 8 matrix registers spare in the 4x4 kernel; here 7 of 8
// are live, M700 spare):
//   M000 = A[m..m+4, k..k+4]
//   M100 = B[n..n+4, k..k+4]      M200 = B[n+4..n+8, k..k+4]
//   M300 = A @ M100^T             M400 = A @ M200^T
//   M500 += M300 (C cols 0..4)    M600 += M400 (C cols 4..8)

/// Rows of C per micro-tile.
pub const GEMM_MR: usize = 4;
/// Columns of C per micro-tile.
pub const GEMM_NR: usize = 8;
/// K is padded to a multiple of this so the k-tile count is always even and
/// the micro-kernel's 2x unroll needs no remainder path.
pub const GEMM_KPAD: usize = 8;

/// Floats in the packed-A scratch for an `mc x kc` block.
pub const fn gemm_ap_len(mc: usize, kc: usize) -> usize {
    div_ceil(mc, GEMM_MR) * div_ceil(kc, 4) * GEMM_MR * 4
}

/// Floats in the packed-C scratch for an `mc x n` slab.
pub const fn gemm_cp_len(mc: usize, n: usize) -> usize {
    div_ceil(mc, GEMM_MR) * div_ceil(n, GEMM_NR) * GEMM_MR * GEMM_NR
}

/// Whether `gemm_bt_packed_fused` finishes every tile in one k-block for
/// this `k`/`kc` (its direct path), which is when `cp` only needs one tile.
pub const fn gemm_single_kblock(k: usize, kc: usize) -> bool {
    div_ceil(k, GEMM_KPAD) * 2 <= kc / 4
}

/// Floats `gemm_bt_packed_fused` needs in `cp` for these blocking factors:
/// one 4x8 tile on the direct path, the `mc x n` slab otherwise.
pub const fn gemm_fused_cp_len(mc: usize, kc: usize, k: usize, n: usize) -> usize {
    if gemm_single_kblock(k, kc) {
        GEMM_MR * GEMM_NR
    } else {
        gemm_cp_len(mc, n)
    }
}

/// Floats in the packed-B panel for a `[n, k]` weight matrix.
pub const fn gemm_bp_len(n: usize, k: usize) -> usize {
    div_ceil(n, GEMM_NR) * (div_ceil(k, GEMM_KPAD) * 2) * GEMM_NR * 4
}

/// Pack `B[n, k]` (row-major) into the micro-kernel's layout, zero-padding N
/// to a multiple of 8 and K to a multiple of 8.
///
/// Layout: `bp[nb][kt][j][kk] = B[nb*8 + j][kt*4 + kk]`. For a fixed `nb` the
/// k-tiles are contiguous, so one k-step of the micro-kernel reads 8
/// consecutive quads and advances 128 bytes with no address arithmetic.
///
/// For model weights this is done once at compile time by psp-tc; the runtime
/// copy exists for benchmarks and tests.
pub fn pack_b_panel(b: &[f32], bp: &mut [f32], n: usize, k: usize) {
    let kt_total = div_ceil(k, GEMM_KPAD) * 2;
    let nb_total = div_ceil(n, GEMM_NR);
    let mut idx = 0;
    for nb in 0..nb_total {
        for kt in 0..kt_total {
            for j in 0..GEMM_NR {
                let row = nb * GEMM_NR + j;
                for kk in 0..4 {
                    let col = kt * 4 + kk;
                    bp[idx] = if row < n && col < k {
                        b[row * k + col]
                    } else {
                        0.0
                    };
                    idx += 1;
                }
            }
        }
    }
}

/// Like `pack_b_panel` but the source is int8 with a per-output-channel scale,
/// dequantising during the pack. Lets int8 conv weights stay 1 byte each in the
/// blob while the micro-kernel still sees f32.
pub fn pack_b_panel_dequant_i8(b: &[i8], scales: &[f32], bp: &mut [f32], n: usize, k: usize) {
    let kt_total = div_ceil(k, GEMM_KPAD) * 2;
    let nb_total = div_ceil(n, GEMM_NR);
    let mut idx = 0;
    for nb in 0..nb_total {
        for kt in 0..kt_total {
            for j in 0..GEMM_NR {
                let row = nb * GEMM_NR + j;
                let s = if row < n { scales[row] } else { 0.0 };
                for kk in 0..4 {
                    let col = kt * 4 + kk;
                    bp[idx] = if row < n && col < k {
                        b[row * k + col] as f32 * s
                    } else {
                        0.0
                    };
                    idx += 1;
                }
            }
        }
    }
}

/// Pack an `mb_count*4 x kt_count*4` block of `A[m, k]` into micro-kernel
/// layout `ap[mb][kt][r][kk]`, zeroing anything past the real M/K extents.
///
/// Written as an explicit indexed loop on purpose: the `psp` crate's `memcpy`
/// is a byte-at-a-time loop (~36 MB/s measured), so `copy_from_slice` here
/// would cost more than the multiply it feeds.
#[allow(clippy::too_many_arguments)]
pub fn pack_a_block(
    a: &[f32],
    ap: &mut [f32],
    m: usize,
    k: usize,
    lda: usize,
    m0: usize,
    mb_count: usize,
    k0: usize,
    kt_count: usize,
) {
    // k-tiles that lie entirely inside the real K extent. Only the tile that
    // straddles `k` (at most one, since K is padded to a multiple of 8 and a
    // tile is 4 wide) needs the per-element bounds test — before this split
    // a row whose *last* tile was partial took the slow path for every
    // element, ~31 cycles/float, on every small-K conv in the model.
    let full_kt = if k0 >= k {
        0
    } else {
        core::cmp::min(kt_count, (k - k0) / 4)
    };
    for mb in 0..mb_count {
        let row0 = m0 + mb * GEMM_MR;
        let block = mb * kt_count * GEMM_MR * 4;
        // Row-outer so each A row is read as one sequential stream. The
        // transposed alternative (kt outer) interleaves four rows 4100 bytes
        // apart and measured ~1.5x slower on the DRAM side.
        for r in 0..GEMM_MR {
            let row = row0 + r;
            if row < m {
                let mut src = row * lda + k0;
                let mut idx = block + r * 4;
                unsafe {
                    let ab = a.as_ptr();
                    let pb = ap.as_mut_ptr();
                    for _ in 0..full_kt {
                        let s = ab.add(src);
                        let d = pb.add(idx);
                        *d = *s;
                        *d.add(1) = *s.add(1);
                        *d.add(2) = *s.add(2);
                        *d.add(3) = *s.add(3);
                        src += 4;
                        idx += GEMM_MR * 4;
                    }
                }
                for kt in full_kt..kt_count {
                    let idx = block + (kt * GEMM_MR + r) * 4;
                    for kk in 0..4 {
                        let col = k0 + kt * 4 + kk;
                        ap[idx + kk] = if col < k { a[row * lda + col] } else { 0.0 };
                    }
                }
            } else {
                for kt in 0..kt_count {
                    let idx = block + (kt * GEMM_MR + r) * 4;
                    ap[idx..idx + 4].fill(0.0);
                }
            }
        }
    }
}

/// Scatter the packed C slab back into row-major `C[m, n]`, dropping the rows
/// and columns that only existed as padding.
pub fn unpack_c_block(
    cp: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    m0: usize,
    mb_count: usize,
    nb_total: usize,
) {
    for mb in 0..mb_count {
        for r in 0..GEMM_MR {
            let row = m0 + mb * GEMM_MR + r;
            if row >= m {
                continue;
            }
            for nb in 0..nb_total {
                let base = ((nb * mb_count + mb) * GEMM_MR + r) * GEMM_NR;
                let col0 = nb * GEMM_NR;
                let take = if col0 + GEMM_NR <= n {
                    GEMM_NR
                } else {
                    n - col0
                };
                for j in 0..take {
                    c[row * n + col0 + j] = cp[base + j];
                }
            }
        }
    }
}

/// What the GEMM / depthwise epilogue does to each output element after the
/// bias add. Fusing this into the store saves a full read-modify-write pass
/// over the output tensor per activation — on BirdNET that pass was memory
/// bound at 57 ns/element and there were 90 of them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Epilogue {
    None,
    Relu,
    Swish,
}

/// Scalar mirror of the epilogue, for tails and the host build.
#[inline]
pub fn apply_epilogue(v: f32, act: Epilogue) -> f32 {
    match act {
        Epilogue::None => v,
        Epilogue::Relu => {
            if v < 0.0 {
                0.0
            } else {
                v
            }
        }
        Epilogue::Swish => v / (1.0 + libm::expf(-v)),
    }
}

/// Sixteen aligned zeros: a bias source for the "no bias" epilogue so the
/// VFPU paths always have something to `lv.q`.
#[cfg(target_os = "psp")]
#[repr(align(16))]
struct Align16F16([f32; 16]);
#[cfg(target_os = "psp")]
static ZEROS16: Align16F16 = Align16F16([0.0; 16]);

/// The 4x8 micro-kernel body. `$init` is the accumulator setup: either the
/// eight `lv.q` that load an existing C tile, or eight `vzero.q`.
#[cfg(target_os = "psp")]
macro_rules! micro_4x8_body {
    ($ap:expr, $bp:expr, $cp:expr, $kt:expr, $($init:tt),* $(,)?) => {
        vfpu_asm!(
            $($init,)*
            "2:",
            // ---- k-tile 0 ----
            "lv.q R000,  0({a})",
            "lv.q R001, 16({a})",
            "lv.q R002, 32({a})",
            "lv.q R003, 48({a})",
            "lv.q R100,   0({b})",
            "lv.q R101,  16({b})",
            "lv.q R102,  32({b})",
            "lv.q R103,  48({b})",
            "lv.q R200,  64({b})",
            "lv.q R201,  80({b})",
            "lv.q R202,  96({b})",
            "lv.q R203, 112({b})",
            // Two independent products -> both pipelines busy
            "vmmul.q M300, M000, E100",
            "vmmul.q M400, M000, E200",
            "vadd.q R500, R500, R300",
            "vadd.q R501, R501, R301",
            "vadd.q R502, R502, R302",
            "vadd.q R503, R503, R303",
            "vadd.q R600, R600, R400",
            "vadd.q R601, R601, R401",
            "vadd.q R602, R602, R402",
            "vadd.q R603, R603, R403",
            // ---- k-tile 1 ----
            "lv.q R000,  64({a})",
            "lv.q R001,  80({a})",
            "lv.q R002,  96({a})",
            "lv.q R003, 112({a})",
            "lv.q R100, 128({b})",
            "lv.q R101, 144({b})",
            "lv.q R102, 160({b})",
            "lv.q R103, 176({b})",
            "lv.q R200, 192({b})",
            "lv.q R201, 208({b})",
            "lv.q R202, 224({b})",
            "lv.q R203, 240({b})",
            "vmmul.q M300, M000, E100",
            "vmmul.q M400, M000, E200",
            "vadd.q R500, R500, R300",
            "vadd.q R501, R501, R301",
            "vadd.q R502, R502, R302",
            "vadd.q R503, R503, R303",
            "vadd.q R600, R600, R400",
            "vadd.q R601, R601, R401",
            "vadd.q R602, R602, R402",
            "vadd.q R603, R603, R403",
            "addiu {a}, {a}, 128",
            "addiu {kp}, {kp}, -1",
            "bnez {kp}, 2b",
            "addiu {b}, {b}, 256", // branch delay slot
            // Accumulator out
            "sv.q R500,   0({c})",
            "sv.q R600,  16({c})",
            "sv.q R501,  32({c})",
            "sv.q R601,  48({c})",
            "sv.q R502,  64({c})",
            "sv.q R602,  80({c})",
            "sv.q R503,  96({c})",
            "sv.q R603, 112({c})",
            a = inout(reg) ($ap) => _,
            b = inout(reg) ($bp) => _,
            c = in(reg) ($cp),
            kp = inout(reg) ($kt / 2) => _,
            options(nostack),
        )
    };
}

/// 4x8 micro-kernel: `cp[4][8] += ap[4][4*kt] @ bp[8][4*kt]^T`.
///
/// `kt_count` must be even (guaranteed by `GEMM_KPAD`). All three pointers
/// must be 16-byte aligned. The accumulator is loaded once, kept in M500/M600
/// across the whole k loop, and stored once.
///
/// # Safety
/// Caller guarantees `ap` holds `kt_count*16`, `bp` holds `kt_count*32`, and
/// `cp` holds 32 floats, all 16-byte aligned.
///
/// Public so benchmarks can measure it in isolation on an L1-resident working
/// set, separating micro-kernel issue rate from memory-system effects.
#[cfg(target_os = "psp")]
#[inline(never)]
pub unsafe fn micro_4x8(ap: *const f32, bp: *const f32, cp: *mut f32, kt_count: usize) {
    debug_assert!(kt_count % 2 == 0 && kt_count > 0);
    micro_4x8_body!(
        ap,
        bp,
        cp,
        kt_count,
        // Accumulator in: the packed C tile is 32 contiguous floats, so the
        // 4x8 block is 8 quads at fixed offsets — no row stride needed.
        "lv.q R500,   0({c})",
        "lv.q R600,  16({c})",
        "lv.q R501,  32({c})",
        "lv.q R601,  48({c})",
        "lv.q R502,  64({c})",
        "lv.q R602,  80({c})",
        "lv.q R503,  96({c})",
        "lv.q R603, 112({c})",
    );
}

/// `micro_4x8` with the accumulator starting at zero: `cp = ap @ bp^T`.
/// Used when the whole K extent fits one k-block, so the tile never needs to
/// be zeroed in memory and re-loaded.
///
/// # Safety
/// As `micro_4x8`.
#[cfg(target_os = "psp")]
#[inline(never)]
pub unsafe fn micro_4x8_fresh(ap: *const f32, bp: *const f32, cp: *mut f32, kt_count: usize) {
    debug_assert!(kt_count % 2 == 0 && kt_count > 0);
    micro_4x8_body!(
        ap,
        bp,
        cp,
        kt_count,
        "vzero.q R500",
        "vzero.q R600",
        "vzero.q R501",
        "vzero.q R601",
        "vzero.q R502",
        "vzero.q R602",
        "vzero.q R503",
        "vzero.q R603",
    );
}

/// Scalar mirror of the 4x8 micro-kernel for host builds.
#[cfg(not(target_os = "psp"))]
pub unsafe fn micro_4x8(ap: *const f32, bp: *const f32, cp: *mut f32, kt_count: usize) {
    for kt in 0..kt_count {
        for r in 0..GEMM_MR {
            for j in 0..GEMM_NR {
                let mut acc = 0.0f32;
                for kk in 0..4 {
                    let a = *ap.add(kt * 16 + r * 4 + kk);
                    let b = *bp.add(kt * 32 + j * 4 + kk);
                    acc += a * b;
                }
                *cp.add(r * GEMM_NR + j) += acc;
            }
        }
    }
}

#[cfg(not(target_os = "psp"))]
pub unsafe fn micro_4x8_fresh(ap: *const f32, bp: *const f32, cp: *mut f32, kt_count: usize) {
    for i in 0..GEMM_MR * GEMM_NR {
        *cp.add(i) = 0.0;
    }
    micro_4x8(ap, bp, cp, kt_count);
}

/// Write one packed 4x8 accumulator tile into row-major `C[m, n]` at
/// (`row0`, `col0`): add `bias[col0..col0+8]`, apply `act`, drop the rows and
/// columns that only existed as padding.
///
/// This replaces `unpack_c_block` + `bias_add` + `swish`/`relu`: the tile is
/// still in L1 when it is stored, so the epilogue costs nothing extra on the
/// memory side.
#[allow(clippy::too_many_arguments)]
#[inline]
fn store_tile_4x8(
    tile: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    row0: usize,
    col0: usize,
    bias: Option<&[f32]>,
    act: Epilogue,
) {
    let rows = core::cmp::min(GEMM_MR, m - row0);
    let cols = core::cmp::min(GEMM_NR, n - col0);

    #[cfg(target_os = "psp")]
    {
        // Full interior tile with every row quad-aligned: straight VFPU.
        let cptr = c.as_mut_ptr().wrapping_add(row0 * n + col0);
        let bptr = match bias {
            Some(b) => b.as_ptr().wrapping_add(col0),
            None => ZEROS16.0.as_ptr(),
        };
        if rows == GEMM_MR
            && cols == GEMM_NR
            && n % 4 == 0
            && cptr as usize % 16 == 0
            && bptr as usize % 16 == 0
        {
            unsafe { store_tile_4x8_vfpu(tile.as_ptr(), cptr, n * 4, bptr, act) };
            return;
        }
    }

    for r in 0..rows {
        for j in 0..cols {
            let mut v = tile[r * GEMM_NR + j];
            if let Some(b) = bias {
                v += b[col0 + j];
            }
            c[(row0 + r) * n + col0 + j] = apply_epilogue(v, act);
        }
    }
}

/// VFPU body of `store_tile_4x8`. `stride_bytes` is C's row pitch.
///
/// # Safety
/// `tile` (32 floats), `bias` (8 floats) and the four C rows must be 16-byte
/// aligned and fully in bounds.
#[cfg(target_os = "psp")]
#[inline(never)]
unsafe fn store_tile_4x8_vfpu(
    tile: *const f32,
    c: *mut f32,
    stride_bytes: usize,
    bias: *const f32,
    act: Epilogue,
) {
    let c0 = c as *mut u8;
    let c1 = c0.add(stride_bytes);
    let c2 = c1.add(stride_bytes);
    let c3 = c2.add(stride_bytes);
    // Tile rows r at byte 32r: M000 rows 0-1 (R000..R003), M100 rows 2-3.
    macro_rules! tile_io {
        ($($mid:tt),* $(,)?) => {
            vfpu_asm!(
                "lv.q R000,   0({t})",
                "lv.q R001,  16({t})",
                "lv.q R002,  32({t})",
                "lv.q R003,  48({t})",
                "lv.q R100,  64({t})",
                "lv.q R101,  80({t})",
                "lv.q R102,  96({t})",
                "lv.q R103, 112({t})",
                "lv.q R200,  0({bias})",
                "lv.q R201, 16({bias})",
                "lv.q R700, 0({k})", // -log2(e), for the swish variant
                "vadd.q R000, R000, R200",
                "vadd.q R001, R001, R201",
                "vadd.q R002, R002, R200",
                "vadd.q R003, R003, R201",
                "vadd.q R100, R100, R200",
                "vadd.q R101, R101, R201",
                "vadd.q R102, R102, R200",
                "vadd.q R103, R103, R201",
                $($mid,)*
                "sv.q R000,  0({c0})",
                "sv.q R001, 16({c0})",
                "sv.q R002,  0({c1})",
                "sv.q R003, 16({c1})",
                "sv.q R100,  0({c2})",
                "sv.q R101, 16({c2})",
                "sv.q R102,  0({c3})",
                "sv.q R103, 16({c3})",
                t = in(reg) (tile),
                bias = in(reg) (bias),
                k = in(reg) (NEG_LOG2_E.0.as_ptr()),
                c0 = in(reg) (c0),
                c1 = in(reg) (c1),
                c2 = in(reg) (c2),
                c3 = in(reg) (c3),
                options(nostack),
            )
        };
    }
    match act {
        Epilogue::None => tile_io!(),
        Epilogue::Relu => tile_io!(
            "vzero.q R300",
            "vmax.q R000, R000, R300",
            "vmax.q R001, R001, R300",
            "vmax.q R002, R002, R300",
            "vmax.q R003, R003, R300",
            "vmax.q R100, R100, R300",
            "vmax.q R101, R101, R300",
            "vmax.q R102, R102, R300",
            "vmax.q R103, R103, R300",
        ),
        // x * sigmoid(x) with sigmoid(x) = 1 / (1 + 2^(-x*log2 e)); the
        // same chain as `swish`, eight quads interleaved so the transcendental
        // latencies overlap.
        Epilogue::Swish => tile_io!(
            "vone.q R701",
            "vmul.q R300, R000, R700",
            "vmul.q R301, R001, R700",
            "vmul.q R302, R002, R700",
            "vmul.q R303, R003, R700",
            "vmul.q R400, R100, R700",
            "vmul.q R401, R101, R700",
            "vmul.q R402, R102, R700",
            "vmul.q R403, R103, R700",
            "vexp2.q R300, R300",
            "vexp2.q R301, R301",
            "vexp2.q R302, R302",
            "vexp2.q R303, R303",
            "vexp2.q R400, R400",
            "vexp2.q R401, R401",
            "vexp2.q R402, R402",
            "vexp2.q R403, R403",
            "vadd.q R300, R300, R701",
            "vadd.q R301, R301, R701",
            "vadd.q R302, R302, R701",
            "vadd.q R303, R303, R701",
            "vadd.q R400, R400, R701",
            "vadd.q R401, R401, R701",
            "vadd.q R402, R402, R701",
            "vadd.q R403, R403, R701",
            "vrcp.q R300, R300",
            "vrcp.q R301, R301",
            "vrcp.q R302, R302",
            "vrcp.q R303, R303",
            "vrcp.q R400, R400",
            "vrcp.q R401, R401",
            "vrcp.q R402, R402",
            "vrcp.q R403, R403",
            "vmul.q R000, R000, R300",
            "vmul.q R001, R001, R301",
            "vmul.q R002, R002, R302",
            "vmul.q R003, R003, R303",
            "vmul.q R100, R100, R400",
            "vmul.q R101, R101, R401",
            "vmul.q R102, R102, R402",
            "vmul.q R103, R103, R403",
        ),
    }
}

/// Cache-blocked GEMM with B pre-packed and a fused epilogue:
/// `C[m,n] = act(A[m,k] @ B[n,k]^T + bias)`.
///
/// `m`, `k` and `n` are arbitrary — padding is absorbed by the packing steps,
/// so there is no alignment requirement on the caller's tensors and no scalar
/// tail path. `bp` must come from `pack_b_panel` (or psp-tc's compile-time
/// equivalent); `ap` and `cp` are 16-byte-aligned scratch.
///
/// `lda` is A's row stride, which may exceed `k` (im2col writes rows padded to
/// a multiple of 4, and those pad columns hold stale arena data — passing the
/// logical `k` separately means they are never read). It is also how a 1x1
/// convolution feeds its NHWC input straight in, with no im2col copy.
///
/// `mc` and `kc` are the L1 blocking factors; `kc` must be a multiple of 8.
/// Two regimes:
///
/// - **K fits one k-block** (`k <= kc`, every conv in BirdNET with K <= 128):
///   each 4x8 tile is computed start-to-finish in registers and written to C
///   through the epilogue. `cp` only needs 32 floats. There is no C slab to
///   zero, no slab re-read, and the B panel is walked once per m-block.
///   Pick `mc` so the packed A block (`mc*kc` floats) plus one B sub-panel
///   stays inside L1 — the A block is re-read for every `nb`.
/// - **K spans several k-blocks**: the classic slab: `cp` holds `mc x n`
///   partial sums across k-blocks (`gemm_cp_len(mc, n)` floats) and the
///   epilogue runs per tile once the last k-block is in.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bt_packed_fused(
    a: &[f32],
    lda: usize,
    bp: &[f32],
    c: &mut [f32],
    ap: &mut [f32],
    cp: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    mc: usize,
    kc: usize,
    bias: Option<&[f32]>,
    act: Epilogue,
) {
    debug_assert!(kc % GEMM_KPAD == 0, "kc must be a multiple of 8");
    debug_assert!(ap.as_ptr() as usize % 16 == 0, "ap must be 16-byte aligned");
    debug_assert!(cp.as_ptr() as usize % 16 == 0, "cp must be 16-byte aligned");

    let kt_total = div_ceil(k, GEMM_KPAD) * 2;
    let nb_total = div_ceil(n, GEMM_NR);
    let ktc_max = kc / 4;
    let tile_len = GEMM_MR * GEMM_NR;
    debug_assert!(cp.len() >= gemm_fused_cp_len(mc, kc, k, n), "cp too small");

    if gemm_single_kblock(k, kc) {
        let mut m0 = 0;
        while m0 < m {
            let rows = core::cmp::min(mc, m - m0);
            let mb_count = div_ceil(rows, GEMM_MR);
            pack_a_block(a, ap, m, k, lda, m0, mb_count, 0, kt_total);
            for nb in 0..nb_total {
                let b_off = nb * kt_total * GEMM_NR * 4;
                for mb in 0..mb_count {
                    unsafe {
                        micro_4x8_fresh(
                            ap.as_ptr().add(mb * kt_total * GEMM_MR * 4),
                            bp.as_ptr().add(b_off),
                            cp.as_mut_ptr(),
                            kt_total,
                        );
                    }
                    store_tile_4x8(
                        &cp[..tile_len],
                        c,
                        m,
                        n,
                        m0 + mb * GEMM_MR,
                        nb * GEMM_NR,
                        bias,
                        act,
                    );
                }
            }
            m0 += mc;
        }
        return;
    }

    let mut m0 = 0;
    while m0 < m {
        let rows = core::cmp::min(mc, m - m0);
        let mb_count = div_ceil(rows, GEMM_MR);

        // Zero the C slab: the arena is reused between ops and never re-zeroed.
        let cp_len = mb_count * nb_total * tile_len;
        cp[..cp_len].fill(0.0);

        let mut kt0 = 0;
        while kt0 < kt_total {
            let ktc = core::cmp::min(ktc_max, kt_total - kt0);
            pack_a_block(a, ap, m, k, lda, m0, mb_count, kt0 * 4, ktc);

            // nb outer, mb inner. With C packed [nb][mb][r][j] this makes all
            // three streams sequential in the inner loop: the B sub-panel is
            // loop-invariant, A walks its packed blocks in order, and C walks
            // contiguously. The other order strides B by `kt_total*32` floats
            // (33 KB here) on every step, which cost ~2x.
            for nb in 0..nb_total {
                let b_off = (nb * kt_total + kt0) * GEMM_NR * 4;
                for mb in 0..mb_count {
                    unsafe {
                        micro_4x8(
                            ap.as_ptr().add(mb * ktc * GEMM_MR * 4),
                            bp.as_ptr().add(b_off),
                            cp.as_mut_ptr().add((nb * mb_count + mb) * tile_len),
                            ktc,
                        );
                    }
                }
            }
            kt0 += ktc;
        }

        for nb in 0..nb_total {
            for mb in 0..mb_count {
                let base = (nb * mb_count + mb) * tile_len;
                store_tile_4x8(
                    &cp[base..base + tile_len],
                    c,
                    m,
                    n,
                    m0 + mb * GEMM_MR,
                    nb * GEMM_NR,
                    bias,
                    act,
                );
            }
        }
        m0 += mc;
    }
}

/// `gemm_bt_packed_fused` without bias or activation: `C = A @ B^T`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bt_packed(
    a: &[f32],
    lda: usize,
    bp: &[f32],
    c: &mut [f32],
    ap: &mut [f32],
    cp: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    mc: usize,
    kc: usize,
) {
    gemm_bt_packed_fused(a, lda, bp, c, ap, cp, m, k, n, mc, kc, None, Epilogue::None);
}

// ============================================================================
// Real FFT
// ============================================================================
//
// The complex data is held **de-interleaved** — all real parts, then all
// imaginary parts — rather than as (re, im) pairs, so a radix-2 butterfly's
// cross products (`tw_re*bot_im`, `tw_im*bot_re`) are plain elementwise
// `vmul.q` on aligned quads: four butterflies per instruction group, 20
// instructions for 4 butterflies.
//
// The alternative is to keep the interleaved layout and use operand prefixes
// for the swizzle and negation, which the VFPU does support and rust-psp does
// expose (bracket decoration, uppercase lanes: `vmul.q R200, R000[X,X,Z,Z],
// R100[-Y,X,-W,Z]` — the assembler emits the `vpfxs`/`vpfxt` for you, one
// cycle, no added latency). That is not worse, but it is not better either:
// one quad holds 4 complex reals when split versus only 2 complex numbers
// interleaved, so the extra cross-term instructions are exactly offset and
// both land at 5 instructions per butterfly. Split wins on not needing a
// layout-specific twiddle order in codegen.
//
// Stage 0 has a twiddle of 1, so it is pure add/sub on adjacent elements —
// exactly `vbfy1.q`, measured on hardware (see `examples/fft-demo`) as:
//
//     vbfy1.q([a,b,c,d]) -> [a+b, a-b, c+d, c-d]
//     vbfy2.q([a,b,c,d]) -> [a+c, b+d, a-c, b-d]
//
// Buffer layouts (sizes are unchanged from the interleaved version, so codegen
// only had to change how it *orders* the twiddle constants):
//   scratch:         [re(nc) | im(nc)]
//   stage_twiddles:  per stage s, at offset 2*(2^s - 1): [re(2^s) | im(2^s)]
//   unpack_twiddles: [re(nc-1) | im(nc-1)]

/// Floats reserved for stage `s`'s twiddles: `2 * 2^s` rounded up to a
/// multiple of 4 so each stage's real and imaginary runs stay 16-byte aligned
/// for `lv.q`. psp-tc's `lower_rfft` emits blocks of exactly this size.
pub const fn stage_tw_block(s: usize) -> usize {
    let need = 2 * (1usize << s);
    if need < 4 {
        4
    } else {
        need
    }
}

/// Total floats in the stage-twiddle constant for an `n`-point real FFT.
pub const fn stage_tw_len(n: usize) -> usize {
    let stages = (n / 2).trailing_zeros() as usize;
    let mut total = 0;
    let mut s = 0;
    while s < stages {
        total += stage_tw_block(s);
        s += 1;
    }
    total
}

/// Bit-reversed pack of a real frame into split complex arrays. `stride`
/// spaces the frame's samples inside `input` (1 = contiguous; the pruned
/// small-FFT frontend reads its decimated signal at stride 2).
fn rfft_pack_split(input: &[f32], re: &mut [f32], im: &mut [f32], nc: usize, stride: usize) {
    // Reversed-counter increment rather than reversing each index from
    // scratch: the naive form costs log2(nc) shifts per element *per frame*
    // (~5 M operations across BirdNET's 511 frames) to recompute the identical
    // permutation every time. Carrying `rev` from one k to the next is O(1)
    // amortised.
    let mut rev = 0usize;
    for k in 0..nc {
        re[k] = input[2 * rev * stride];
        im[k] = input[(2 * rev + 1) * stride];
        let mut bit = nc >> 1;
        while bit != 0 && rev & bit != 0 {
            rev ^= bit;
            bit >>= 1;
        }
        rev |= bit;
    }
}

/// `rfft_pack_split` with a window multiply folded into the pack. Each source
/// sample is multiplied by the matching window coefficient exactly once, so
/// the result is bit-identical to materialising `input[i] * window[i]` first
/// and packing that (the dense gather+mul path this replaces).
fn rfft_pack_split_windowed(
    input: &[f32],
    window: &[f32],
    re: &mut [f32],
    im: &mut [f32],
    nc: usize,
    stride: usize,
) {
    let mut rev = 0usize;
    for k in 0..nc {
        re[k] = input[2 * rev * stride] * window[2 * rev];
        im[k] = input[(2 * rev + 1) * stride] * window[2 * rev + 1];
        let mut bit = nc >> 1;
        while bit != 0 && rev & bit != 0 {
            rev ^= bit;
            bit >>= 1;
        }
        rev |= bit;
    }
}

/// Stage 0: twiddle is 1, so each adjacent pair becomes (a+b, a-b).
#[cfg(target_os = "psp")]
fn fft_stage0(re: &mut [f32], im: &mut [f32], nc: usize) {
    let quads = nc / 4;
    if quads > 0 {
        unsafe {
            vfpu_asm!(
                "2:",
                "lv.q R000, 0({r})",
                "lv.q R001, 0({i})",
                "vbfy1.q R100, R000",
                "vbfy1.q R101, R001",
                "sv.q R100, 0({r})",
                "sv.q R101, 0({i})",
                "addiu {r}, {r}, 16",
                "addiu {n}, {n}, -1",
                "bnez {n}, 2b",
                "addiu {i}, {i}, 16",
                r = inout(reg) (re.as_mut_ptr()) => _,
                i = inout(reg) (im.as_mut_ptr()) => _,
                n = inout(reg) (quads) => _,
                options(nostack),
            );
        }
    }
    for k in (quads * 4..nc).step_by(2) {
        let (ar, ai, br, bi) = (re[k], im[k], re[k + 1], im[k + 1]);
        re[k] = ar + br;
        im[k] = ai + bi;
        re[k + 1] = ar - br;
        im[k + 1] = ai - bi;
    }
}

#[cfg(not(target_os = "psp"))]
fn fft_stage0(re: &mut [f32], im: &mut [f32], nc: usize) {
    for k in (0..nc).step_by(2) {
        let (ar, ai, br, bi) = (re[k], im[k], re[k + 1], im[k + 1]);
        re[k] = ar + br;
        im[k] = ai + bi;
        re[k + 1] = ar - br;
        im[k + 1] = ai - bi;
    }
}

/// One radix-2 stage over four butterflies at a time.
///
/// # Safety
/// `top`/`bot` are `half`-aligned indices into `re`/`im`; all pointers land on
/// 16-byte boundaries because `half >= 4` and groups start at multiples of
/// `2*half`.
#[cfg(target_os = "psp")]
#[inline(never)]
unsafe fn fft_bfly4(
    re: *mut f32,
    im: *mut f32,
    tw_re: *const f32,
    tw_im: *const f32,
    top: usize,
    bot: usize,
    count: usize,
) {
    let rt = re.add(top);
    let rb = re.add(bot);
    let it = im.add(top);
    let ib = im.add(bot);
    vfpu_asm!(
        "2:",
        "lv.q R000, 0({rb})",          // bot.re
        "lv.q R001, 0({ib})",          // bot.im
        "lv.q R100, 0({twr})",
        "lv.q R101, 0({twi})",
        "lv.q R200, 0({rt})",          // top.re
        "lv.q R201, 0({it})",          // top.im
        "vmul.q R300, R100, R000",     // tw_re*bot_re
        "vmul.q R301, R101, R001",     // tw_im*bot_im
        "vsub.q R300, R300, R301",     // t_re
        "vmul.q R302, R100, R001",     // tw_re*bot_im
        "vmul.q R303, R101, R000",     // tw_im*bot_re
        "vadd.q R302, R302, R303",     // t_im
        "vadd.q R400, R200, R300",
        "vsub.q R401, R200, R300",
        "vadd.q R402, R201, R302",
        "vsub.q R403, R201, R302",
        "sv.q R400, 0({rt})",
        "sv.q R401, 0({rb})",
        "sv.q R402, 0({it})",
        "sv.q R403, 0({ib})",
        "addiu {rt}, {rt}, 16",
        "addiu {rb}, {rb}, 16",
        "addiu {it}, {it}, 16",
        "addiu {ib}, {ib}, 16",
        "addiu {twr}, {twr}, 16",
        "addiu {n}, {n}, -1",
        "bnez {n}, 2b",
        "addiu {twi}, {twi}, 16",
        rt = inout(reg) (rt) => _,
        rb = inout(reg) (rb) => _,
        it = inout(reg) (it) => _,
        ib = inout(reg) (ib) => _,
        twr = inout(reg) (tw_re) => _,
        twi = inout(reg) (tw_im) => _,
        n = inout(reg) (count) => _,
        options(nostack),
    );
}

/// One radix-2 DIT stage on split arrays.
fn fft_stage_split(
    re: &mut [f32],
    im: &mut [f32],
    tw_re: &[f32],
    tw_im: &[f32],
    nc: usize,
    half: usize,
) {
    let full = half * 2;
    let mut base = 0;
    while base < nc {
        let mut j = 0;
        #[cfg(target_os = "psp")]
        if half >= 4 {
            let quads = half / 4;
            unsafe {
                fft_bfly4(
                    re.as_mut_ptr(),
                    im.as_mut_ptr(),
                    tw_re.as_ptr(),
                    tw_im.as_ptr(),
                    base,
                    base + half,
                    quads,
                );
            }
            j = quads * 4;
        }
        while j < half {
            let (t, b) = (base + j, base + j + half);
            let (wr, wi) = (tw_re[j], tw_im[j]);
            let t_re = wr * re[b] - wi * im[b];
            let t_im = wr * im[b] + wi * re[b];
            let (ur, ui) = (re[t], im[t]);
            re[t] = ur + t_re;
            im[t] = ui + t_im;
            re[b] = ur - t_re;
            im[b] = ui - t_im;
            j += 1;
        }
        base += full;
    }
}

/// Extract the real parts of a real-input FFT from the packed half-length
/// complex transform. Same identity as the interleaved version, split inputs.
fn rfft_unpack_split(
    re: &[f32],
    im: &[f32],
    utw_re: &[f32],
    utw_im: &[f32],
    output: &mut [f32],
    nc: usize,
) {
    output[0] = re[0] + im[0];
    output[nc] = re[0] - im[0];
    for k in 1..nc {
        let conj = nc - k;
        let a_re = 0.5 * (re[k] + re[conj]);
        let b_re = 0.5 * (im[k] + im[conj]);
        let b_im = -0.5 * (re[k] - re[conj]);
        output[k] = a_re + (utw_re[k - 1] * b_re - utw_im[k - 1] * b_im);
    }
}

/// One frame of the real FFT: pack (optionally windowed), butterfly stages,
/// unpack. Shared by `rfft_batch` and `rfft_strided_batch`.
#[allow(clippy::too_many_arguments)]
fn rfft_frame_split(
    frame: &[f32],
    stride: usize,
    window: Option<&[f32]>,
    stage_twiddles: &[f32],
    utw_re: &[f32],
    utw_im: &[f32],
    re: &mut [f32],
    im: &mut [f32],
    out_row: &mut [f32],
    nc: usize,
    stages: usize,
) {
    match window {
        Some(w) => rfft_pack_split_windowed(frame, w, re, im, nc, stride),
        None => rfft_pack_split(frame, re, im, nc, stride),
    }
    fft_stage0(re, im, nc);
    // Stage blocks are padded to a multiple of 4 floats. The natural
    // packing (block s at offset 2*(2^s - 1)) puts every stage's twiddles
    // at byte offset 8 mod 16, so every `lv.q` of them would be
    // misaligned — which the scalar host mirror cannot detect.
    let mut off = stage_tw_block(0);
    for s in 1..stages {
        let half = 1usize << s;
        fft_stage_split(
            re,
            im,
            &stage_twiddles[off..off + half],
            &stage_twiddles[off + half..off + 2 * half],
            nc,
            half,
        );
        off += stage_tw_block(s);
    }
    rfft_unpack_split(re, im, utw_re, utw_im, out_row, nc);
}

/// Batched real FFT over `frames` contiguous length-`n` frames.
///
/// `input` is `[frames, n]`, `output` is `[frames, n/2 + 1]` (real parts of the
/// frequency bins, matching TFLite RFFT2D + CAST(complex->f32)).
#[allow(clippy::too_many_arguments)]
pub fn rfft_batch(
    input: &[f32],
    stage_twiddles: &[f32],
    unpack_twiddles: &[f32],
    scratch: &mut [f32],
    output: &mut [f32],
    n: usize,
    frames: usize,
) {
    let nc = n / 2;
    let out_bins = nc + 1;
    let stages = nc.trailing_zeros() as usize;
    let (re, im) = scratch.split_at_mut(nc);
    let (utw_re, utw_im) = unpack_twiddles.split_at(nc - 1);

    for f in 0..frames {
        rfft_frame_split(
            &input[f * n..(f + 1) * n],
            1,
            None,
            stage_twiddles,
            utw_re,
            utw_im,
            re,
            im,
            &mut output[f * out_bins..(f + 1) * out_bins],
            nc,
            stages,
        );
    }
}

/// Windowed STFT over strided views of a 1D signal: frame `f` is the slice
/// `input[f*hop .. f*hop + n]`, multiplied elementwise by `window` (if given)
/// during the bit-reversal pack, then real-FFT'd into row `f` of `output`
/// (`[frames, n/2 + 1]`, real parts of the frequency bins).
///
/// This replaces the dense-gather STFT frontend — materialising every
/// (overlapping) window as a `[frames, n]` matrix plus its `[frames, n]`
/// index constant — with reads straight from the signal, so no intermediate
/// larger than the `n`-float scratch exists. Because the window multiply and
/// the FFT are the same f32 operations in the same order, the output is
/// bit-identical to `gather` + `mul` + `rfft_batch`.
/// `in_stride` spaces the samples *within* a frame: element `j` of frame `f`
/// is `input[f*hop + in_stride*j]`. 1 is the plain overlapping-window STFT;
/// the pruned small-FFT frontend reads its half-rate signal at stride 2
/// (every 4th original sample) so the same 23.4 Hz bin grid comes out of a
/// 4x smaller transform. The window still indexes by `j`.
#[allow(clippy::too_many_arguments)]
pub fn rfft_strided_batch(
    input: &[f32],
    window: Option<&[f32]>,
    stage_twiddles: &[f32],
    unpack_twiddles: &[f32],
    scratch: &mut [f32],
    output: &mut [f32],
    n: usize,
    hop: usize,
    in_stride: usize,
    frames: usize,
) {
    let span = in_stride * (n - 1) + 1;
    assert!(
        (frames - 1) * hop + span <= input.len(),
        "rfft_strided_batch: {} frames of {} (stride {}) at hop {} overrun input len {}",
        frames,
        n,
        in_stride,
        hop,
        input.len()
    );
    let nc = n / 2;
    let out_bins = nc + 1;
    let stages = nc.trailing_zeros() as usize;
    let (re, im) = scratch.split_at_mut(nc);
    let (utw_re, utw_im) = unpack_twiddles.split_at(nc - 1);

    for f in 0..frames {
        rfft_frame_split(
            &input[f * hop..f * hop + span],
            in_stride,
            window,
            stage_twiddles,
            utw_re,
            utw_im,
            re,
            im,
            &mut output[f * out_bins..(f + 1) * out_bins],
            nc,
            stages,
        );
    }
}

/// Reduce mean over all dims except the last (channel dim).
///
/// Input has `N*C` elements (NHWC flattened), output has `C` elements:
/// `output[ch] = mean_i input[i*C + ch]`.
///
/// Streams the input row by row, accumulating into the C-sized output (which
/// stays in L1) — the channel-outer order the naive kernel used walked each
/// column with stride C and took one cache miss per element (631k misses on
/// 4 MB of input in BirdNET; 139 ms → 64 ms scalar, ~25 ms on the VFPU).
pub fn reduce_mean_hw(input: &[f32], output: &mut [f32]) {
    let c = output.len();
    if c == 0 {
        return;
    }
    let n = input.len() / c;
    output.fill(0.0);
    let vec_ok =
        c % 4 == 0 && input.as_ptr() as usize % 16 == 0 && output.as_ptr() as usize % 16 == 0;
    for i in 0..n {
        let row = &input[i * c..(i + 1) * c];
        if vec_ok {
            vadd_inplace(output, row, c);
        } else {
            for ch in 0..c {
                output[ch] += row[ch];
            }
        }
    }
    let inv = 1.0 / n as f32;
    for v in output.iter_mut() {
        *v *= inv;
    }
}

// ============================================================================
// Element-wise add / mul with broadcasting
// ============================================================================
//
// `out = a op b` where `b` is either the same length as `out`, a scalar, or a
// row (`b_len` divides `out.len()`, e.g. an NHWC tensor times a `[1,1,1,C]`
// squeeze-excite gate). The naive kernel handled the row case with `b[i %
// b_len]` — an integer divide per element, which on the Allegrex costs more
// than the multiply. BirdNET spends 378 ms in that path.

/// `out[i] = a[i] op b[i]` over one quad-aligned run, for `$vop` in
/// {vadd, vmul}. `n` need not be a multiple of 4; the tail is scalar.
#[cfg(target_os = "psp")]
macro_rules! vbinary_run {
    ($name:ident, $vop:tt, $sop:expr) => {
        #[inline(never)]
        fn $name(a: &[f32], b: &[f32], out: &mut [f32], n: usize) {
            let quads = n / 4;
            if quads > 0 {
                unsafe {
                    vfpu_asm!(
                        "2:",
                        "lv.q R000, 0({a})",
                        "lv.q R100, 0({b})",
                        $vop,
                        "sv.q R200, 0({o})",
                        "addiu {a}, {a}, 16",
                        "addiu {b}, {b}, 16",
                        "addiu {n}, {n}, -1",
                        "bnez {n}, 2b",
                        "addiu {o}, {o}, 16",
                        a = inout(reg) (a.as_ptr()) => _,
                        b = inout(reg) (b.as_ptr()) => _,
                        o = inout(reg) (out.as_mut_ptr()) => _,
                        n = inout(reg) (quads) => _,
                        options(nostack),
                    );
                }
            }
            let op: fn(f32, f32) -> f32 = $sop;
            for i in quads * 4..n {
                out[i] = op(a[i], b[i]);
            }
        }
    };
}

#[cfg(target_os = "psp")]
vbinary_run!(vadd_run, "vadd.q R200, R000, R100", |x, y| x + y);
#[cfg(target_os = "psp")]
vbinary_run!(vmul_run, "vmul.q R200, R000, R100", |x, y| x * y);

#[cfg(not(target_os = "psp"))]
fn vadd_run(a: &[f32], b: &[f32], out: &mut [f32], n: usize) {
    for i in 0..n {
        out[i] = a[i] + b[i];
    }
}
#[cfg(not(target_os = "psp"))]
fn vmul_run(a: &[f32], b: &[f32], out: &mut [f32], n: usize) {
    for i in 0..n {
        out[i] = a[i] * b[i];
    }
}

macro_rules! broadcast_binary {
    ($name:ident, $run:ident, $sop:expr) => {
        /// See the section comment: `b` is full-length, scalar, or a row.
        pub fn $name(a: &[f32], b: &[f32], out: &mut [f32], b_len: usize) {
            let op: fn(f32, f32) -> f32 = $sop;
            let len = out.len();
            let aligned = a.as_ptr() as usize % 16 == 0
                && b.as_ptr() as usize % 16 == 0
                && out.as_ptr() as usize % 16 == 0;
            if b_len == len {
                if aligned {
                    $run(a, b, out, len);
                } else {
                    for i in 0..len {
                        out[i] = op(a[i], b[i]);
                    }
                }
            } else if b_len == 1 {
                let s = b[0];
                for i in 0..len {
                    out[i] = op(a[i], s);
                }
            } else if b_len > 0 && len % b_len == 0 {
                // Row broadcast: b repeats every b_len elements.
                let vec_rows = aligned && b_len % 4 == 0;
                let mut off = 0;
                while off < len {
                    if vec_rows {
                        $run(&a[off..off + b_len], b, &mut out[off..off + b_len], b_len);
                    } else {
                        for j in 0..b_len {
                            out[off + j] = op(a[off + j], b[j]);
                        }
                    }
                    off += b_len;
                }
            } else {
                for i in 0..len {
                    out[i] = op(a[i], b[i % b_len]);
                }
            }
        }
    };
}

broadcast_binary!(binary_add, vadd_run, |x, y| x + y);
broadcast_binary!(binary_mul, vmul_run, |x, y| x * y);

// ============================================================================
// FIR decimation
// ============================================================================

/// FIR lowpass + decimation: `output[n] = Σ_t taps[t] · input[n·factor + t − (T−1)/2]`,
/// out-of-range input treated as zero. The anti-alias step ahead of the
/// small-FFT frontend's strided reads — taps are designed at build time
/// (Kaiser windowed-sinc in psp-tc) so the stopband covers every alias
/// source that would fold onto the mel banks' passband.
///
/// Centering on (T−1)/2 (odd tap count, linear phase) makes the output
/// time-aligned with the input — `output[n]` estimates the lowpassed signal at
/// sample `n·factor` with zero group delay, which matters because the STFT
/// keeps real parts (phase-sensitive), not magnitudes: a delayed read would
/// rotate every bin's phase.
///
/// VFPU path: the tap window for output `n` starts at an arbitrary sample, so
/// `lv.q` on the input needs the window snapped back to a quad boundary. Four
/// copies of the taps, each pre-shifted by 0..3 and zero-padded, let every
/// output read its input from an aligned base with a matching tap table —
/// then it is a plain quad dot product (two accumulator chains, `vfad.q` at
/// the end). Edges that would read outside the input stay scalar.
pub fn fir_decimate(input: &[f32], taps: &[f32], output: &mut [f32], factor: usize) {
    let t = taps.len();
    if t == 0 {
        output.fill(0.0);
        return;
    }
    let center = (t - 1) / 2;

    #[cfg(target_os = "psp")]
    {
        const MAX_TAPS: usize = 128;
        // Padded table length: 3 lead lanes for the shift, then round up to
        // 8 so the two-chain loop needs no remainder.
        let tp = div_ceil(t + 3, 8) * 8;
        if tp <= MAX_TAPS {
            #[repr(align(16))]
            struct Tables([[f32; MAX_TAPS]; 4]);
            let mut tabs = Tables([[0.0; MAX_TAPS]; 4]);
            for (shift, tab) in tabs.0.iter_mut().enumerate() {
                tab[shift..shift + t].copy_from_slice(taps);
            }
            let quads = tp / 4;
            // Quad alignment is relative to the pointer, not the index, so
            // an input that is only 4-byte aligned still works.
            let base_lane = input.as_ptr() as usize / 4;
            for (n, out) in output.iter_mut().enumerate() {
                let s = (n * factor) as isize - center as isize;
                if s >= 0 {
                    let s = s as usize;
                    let shift = (base_lane + s) % 4;
                    if s >= shift && s - shift + tp <= input.len() {
                        let b = s - shift;
                        *out = unsafe {
                            fir_dot_vfpu(input.as_ptr().add(b), tabs.0[shift].as_ptr(), quads)
                        };
                        continue;
                    }
                }
                *out = fir_tap_scalar(input, taps, n * factor, center);
            }
            return;
        }
    }

    for (n, out) in output.iter_mut().enumerate() {
        *out = fir_tap_scalar(input, taps, n * factor, center);
    }
}

/// One output of the FIR with bounds handling: `Σ_t taps[t] · input[base + t − center]`.
#[inline]
fn fir_tap_scalar(input: &[f32], taps: &[f32], base: usize, center: usize) -> f32 {
    let mut sum = 0.0f32;
    for (t, &h) in taps.iter().enumerate() {
        let idx = base + t;
        if idx >= center && idx - center < input.len() {
            sum += h * input[idx - center];
        }
    }
    sum
}

/// Dot product of `quads*4` floats at two 16-byte-aligned pointers; `quads`
/// must be even and non-zero.
#[cfg(target_os = "psp")]
#[inline(never)]
unsafe fn fir_dot_vfpu(x: *const f32, h: *const f32, quads: usize) -> f32 {
    let mut result = 0.0f32;
    vfpu_asm!(
        "vzero.q R300",
        "vzero.q R301",
        "2:",
        "lv.q R000,  0({x})",
        "lv.q R001, 16({x})",
        "lv.q R100,  0({h})",
        "lv.q R101, 16({h})",
        "vmul.q R200, R000, R100",
        "vmul.q R201, R001, R101",
        "vadd.q R300, R300, R200",
        "vadd.q R301, R301, R201",
        "addiu {x}, {x}, 32",
        "addiu {n}, {n}, -1",
        "bnez {n}, 2b",
        "addiu {h}, {h}, 32", // branch delay slot
        "vadd.q R300, R300, R301",
        "vfad.q S400, R300",
        "sv.s S400, 0({o})",
        x = inout(reg) (x) => _,
        h = inout(reg) (h) => _,
        n = inout(reg) (quads / 2) => _,
        o = in(reg) (&mut result as *mut f32),
        options(nostack),
    );
    result
}

// ============================================================================
// Depthwise convolution
// ============================================================================
//
// Depthwise conv has *no* cross-channel contraction — output channel `c`
// depends only on input channel `c` — so unlike a regular conv it cannot be
// turned into a GEMM. What it does have, in NHWC, is the channel index as the
// fastest-varying dimension, which makes the whole thing a contiguous vector
// FMA once the loop nest is right:
//
//     out[oy,ox, 0..C] = act(bias[0..C] + Σ_taps in[iy,ix, 0..C] * filt[ky,kx, 0..C])
//
// The reference version loops channels *outside* the kernel taps, so every
// single multiply-add re-derives two 4D indices (five integer multiplies) and
// re-evaluates four boundary branches — measured at ~71 cycles per MAC, or
// 9 MFLOP/s. Hoisting the tap bookkeeping to once per output pixel leaves a
// pure streaming FMA over C.
//
// The FMA keeps a 16-channel accumulator in M000 across *all* taps of the
// output pixel: per tap it is 8 quad loads and 8 VFPU ops, and the
// accumulator touches memory exactly once (the store). The previous shape —
// one `acc += a*b` pass over the chunk per tap — loaded and stored the
// accumulator nine times per 3x3 output and was bound on that traffic
// (431 ms for 24 MFLOP on BirdNET).
//
// Taps are described to the asm as up to three kernel rows: within a row the
// `kx` taps are a constant `C` floats apart in both the input and the filter,
// so a row is a pointer-bump loop and the rows are unrolled. Kernels taller
// than 3 rows take the reference path. A kernel call covers a *run* of
// consecutive output pixels with the same tap structure (a strip's interior
// columns), so the call and setup cost is paid once per run, not per pixel.
//
// Measured in isolation, in-cache, the generic 16-lane kernel is ~270 cycles
// per pixel (3x3): it reloads the 36 filter quads for every pixel and its
// single accumulator chain serialises the adds. The 8-lane `dw_kernel8_3x3`
// specialisation keeps the 18 filter quads in VFPU registers across the run,
// software-pipelines load → mul → add and splits the accumulation over two
// chains: ~112 cycles per pixel for 8 lanes, i.e. 1.2x the generic kernel's
// throughput per lane, and half the loads. Full 3x3 windows take it; the
// partial windows along the padding go to the generic kernel.
//
// The pixel sweep is tiled so the `kh` input rows it re-reads stay in the
// 16 KB L1: a per-layer channel chunk (whole pixels up to 128 channels, so
// the chunk never straddles cache lines) and a column strip. Even so the
// layers are DRAM-stall bound after these changes (~60% of their cycles on
// BirdNET): the D-cache is 2-way with 8 KB ways, and rows 144 KB apart
// (the [24,64,288] stride-2 layer) alias the same sets.

/// The in-bounds taps of a run of consecutive output pixels, as kernel
/// rows. One call of a `dw_kernel*` instance evaluates `pixels` pixels in a
/// row: they share the row/tap structure (same `ky`/`kx` ranges), and each
/// pixel is `pixel_in_bytes` further into the input and `pixel_out_bytes`
/// further into the output.
struct DwRows {
    /// Input pointer of the first in-bounds tap of each kernel row (channel 0)
    /// for the first pixel of the run.
    input: [*const f32; 3],
    /// Filter pointer of the same tap.
    filter: [*const f32; 3],
    /// Rows present, 1..=3 (0 is handled before the asm is reached).
    row_count: usize,
    /// In-bounds taps per row.
    kx_count: usize,
    /// Byte distance between consecutive `kx` taps (`C * 4`).
    stride_bytes: usize,
    /// Pixels in the run, >= 1.
    pixels: usize,
    /// Input bytes from one pixel of the run to the next (`sw * C * 4`).
    pixel_in_bytes: usize,
    /// Output bytes from one pixel of the run to the next (`C * 4`).
    pixel_out_bytes: usize,
}

/// 16-lane depthwise tap loop over a run of pixels, see the section comment.
/// The activation instructions are spliced in between the tap loop and the
/// store.
#[cfg(target_os = "psp")]
macro_rules! dw_kernel16 {
    ($name:ident $(, $act:tt)* $(,)?) => {
        #[inline(never)]
        unsafe fn $name(bias: *const f32, out: *mut f32, rows: &DwRows, ch_bytes: usize) {
            vfpu_asm!(
                "lv.q R700, 0({k})", // -log2(e), for the swish variant
                "1:", // ---- next output pixel ----
                "lv.q R000,  0({bias})",
                "lv.q R001, 16({bias})",
                "lv.q R002, 32({bias})",
                "lv.q R003, 48({bias})",
                "move {nr}, {nr0}",
                // ---- kernel row 0 (always present) ----
                "addu {ia}, {i0}, {co}",
                "addu {fa}, {f0}, {co}",
                "move {n}, {kx}",
                "2:",
                "lv.q R100,  0({ia})",
                "lv.q R101, 16({ia})",
                "lv.q R102, 32({ia})",
                "lv.q R103, 48({ia})",
                "lv.q R200,  0({fa})",
                "lv.q R201, 16({fa})",
                "lv.q R202, 32({fa})",
                "lv.q R203, 48({fa})",
                "vmul.q R300, R100, R200",
                "vmul.q R301, R101, R201",
                "vmul.q R302, R102, R202",
                "vmul.q R303, R103, R203",
                "vadd.q R000, R000, R300",
                "vadd.q R001, R001, R301",
                "vadd.q R002, R002, R302",
                "vadd.q R003, R003, R303",
                "addu {ia}, {ia}, {cs}",
                "addiu {n}, {n}, -1",
                "bnez {n}, 2b",
                "addu {fa}, {fa}, {cs}", // branch delay slot
                // ---- kernel row 1 ----
                "addiu {nr}, {nr}, -1",
                "beqz {nr}, 9f",
                "addu {ia}, {i1}, {co}", // branch delay slot
                "addu {fa}, {f1}, {co}",
                "move {n}, {kx}",
                "3:",
                "lv.q R100,  0({ia})",
                "lv.q R101, 16({ia})",
                "lv.q R102, 32({ia})",
                "lv.q R103, 48({ia})",
                "lv.q R200,  0({fa})",
                "lv.q R201, 16({fa})",
                "lv.q R202, 32({fa})",
                "lv.q R203, 48({fa})",
                "vmul.q R300, R100, R200",
                "vmul.q R301, R101, R201",
                "vmul.q R302, R102, R202",
                "vmul.q R303, R103, R203",
                "vadd.q R000, R000, R300",
                "vadd.q R001, R001, R301",
                "vadd.q R002, R002, R302",
                "vadd.q R003, R003, R303",
                "addu {ia}, {ia}, {cs}",
                "addiu {n}, {n}, -1",
                "bnez {n}, 3b",
                "addu {fa}, {fa}, {cs}", // branch delay slot
                // ---- kernel row 2 ----
                "addiu {nr}, {nr}, -1",
                "beqz {nr}, 9f",
                "addu {ia}, {i2}, {co}", // branch delay slot
                "addu {fa}, {f2}, {co}",
                "move {n}, {kx}",
                "4:",
                "lv.q R100,  0({ia})",
                "lv.q R101, 16({ia})",
                "lv.q R102, 32({ia})",
                "lv.q R103, 48({ia})",
                "lv.q R200,  0({fa})",
                "lv.q R201, 16({fa})",
                "lv.q R202, 32({fa})",
                "lv.q R203, 48({fa})",
                "vmul.q R300, R100, R200",
                "vmul.q R301, R101, R201",
                "vmul.q R302, R102, R202",
                "vmul.q R303, R103, R203",
                "vadd.q R000, R000, R300",
                "vadd.q R001, R001, R301",
                "vadd.q R002, R002, R302",
                "vadd.q R003, R003, R303",
                "addu {ia}, {ia}, {cs}",
                "addiu {n}, {n}, -1",
                "bnez {n}, 4b",
                "addu {fa}, {fa}, {cs}", // branch delay slot
                "9:",
                $($act,)*
                "sv.q R000,  0({o})",
                "sv.q R001, 16({o})",
                "sv.q R002, 32({o})",
                "sv.q R003, 48({o})",
                // Advance every row to the next output pixel.
                "addu {i0}, {i0}, {pi}",
                "addu {i1}, {i1}, {pi}",
                "addu {i2}, {i2}, {pi}",
                "addiu {np}, {np}, -1",
                "bnez {np}, 1b",
                "addu {o}, {o}, {po}", // branch delay slot
                bias = in(reg) (bias),
                o = inout(reg) (out) => _,
                co = in(reg) (ch_bytes),
                k = in(reg) (NEG_LOG2_E.0.as_ptr()),
                i0 = inout(reg) (rows.input[0]) => _,
                i1 = inout(reg) (rows.input[1]) => _,
                i2 = inout(reg) (rows.input[2]) => _,
                f0 = in(reg) (rows.filter[0]),
                f1 = in(reg) (rows.filter[1]),
                f2 = in(reg) (rows.filter[2]),
                cs = in(reg) (rows.stride_bytes),
                kx = in(reg) (rows.kx_count),
                nr0 = in(reg) (rows.row_count),
                pi = in(reg) (rows.pixel_in_bytes),
                po = in(reg) (rows.pixel_out_bytes),
                np = inout(reg) (rows.pixels) => _,
                nr = out(reg) _,
                n = out(reg) _,
                ia = out(reg) _,
                fa = out(reg) _,
                options(nostack),
            );
        }
    };
}

/// 4-lane depthwise tap loop over a run of pixels, see the section comment.
/// The activation instructions are spliced in between the tap loop and the
/// store.
#[cfg(target_os = "psp")]
macro_rules! dw_kernel4 {
    ($name:ident $(, $act:tt)* $(,)?) => {
        #[inline(never)]
        unsafe fn $name(bias: *const f32, out: *mut f32, rows: &DwRows, ch_bytes: usize) {
            vfpu_asm!(
                "lv.q R700, 0({k})", // -log2(e), for the swish variant
                "1:", // ---- next output pixel ----
                "lv.q R000,  0({bias})",
                "move {nr}, {nr0}",
                // ---- kernel row 0 (always present) ----
                "addu {ia}, {i0}, {co}",
                "addu {fa}, {f0}, {co}",
                "move {n}, {kx}",
                "2:",
                "lv.q R100,  0({ia})",
                "lv.q R200,  0({fa})",
                "vmul.q R300, R100, R200",
                "vadd.q R000, R000, R300",
                "addu {ia}, {ia}, {cs}",
                "addiu {n}, {n}, -1",
                "bnez {n}, 2b",
                "addu {fa}, {fa}, {cs}", // branch delay slot
                // ---- kernel row 1 ----
                "addiu {nr}, {nr}, -1",
                "beqz {nr}, 9f",
                "addu {ia}, {i1}, {co}", // branch delay slot
                "addu {fa}, {f1}, {co}",
                "move {n}, {kx}",
                "3:",
                "lv.q R100,  0({ia})",
                "lv.q R200,  0({fa})",
                "vmul.q R300, R100, R200",
                "vadd.q R000, R000, R300",
                "addu {ia}, {ia}, {cs}",
                "addiu {n}, {n}, -1",
                "bnez {n}, 3b",
                "addu {fa}, {fa}, {cs}", // branch delay slot
                // ---- kernel row 2 ----
                "addiu {nr}, {nr}, -1",
                "beqz {nr}, 9f",
                "addu {ia}, {i2}, {co}", // branch delay slot
                "addu {fa}, {f2}, {co}",
                "move {n}, {kx}",
                "4:",
                "lv.q R100,  0({ia})",
                "lv.q R200,  0({fa})",
                "vmul.q R300, R100, R200",
                "vadd.q R000, R000, R300",
                "addu {ia}, {ia}, {cs}",
                "addiu {n}, {n}, -1",
                "bnez {n}, 4b",
                "addu {fa}, {fa}, {cs}", // branch delay slot
                "9:",
                $($act,)*
                "sv.q R000,  0({o})",
                // Advance every row to the next output pixel.
                "addu {i0}, {i0}, {pi}",
                "addu {i1}, {i1}, {pi}",
                "addu {i2}, {i2}, {pi}",
                "addiu {np}, {np}, -1",
                "bnez {np}, 1b",
                "addu {o}, {o}, {po}", // branch delay slot
                bias = in(reg) (bias),
                o = inout(reg) (out) => _,
                co = in(reg) (ch_bytes),
                k = in(reg) (NEG_LOG2_E.0.as_ptr()),
                i0 = inout(reg) (rows.input[0]) => _,
                i1 = inout(reg) (rows.input[1]) => _,
                i2 = inout(reg) (rows.input[2]) => _,
                f0 = in(reg) (rows.filter[0]),
                f1 = in(reg) (rows.filter[1]),
                f2 = in(reg) (rows.filter[2]),
                cs = in(reg) (rows.stride_bytes),
                kx = in(reg) (rows.kx_count),
                nr0 = in(reg) (rows.row_count),
                pi = in(reg) (rows.pixel_in_bytes),
                po = in(reg) (rows.pixel_out_bytes),
                np = inout(reg) (rows.pixels) => _,
                nr = out(reg) _,
                n = out(reg) _,
                ia = out(reg) _,
                fa = out(reg) _,
                options(nostack),
            );
        }
    };
}

#[cfg(target_os = "psp")]
dw_kernel16!(dw16_none);
#[cfg(target_os = "psp")]
dw_kernel16!(
    dw16_relu,
    "vzero.q R300",
    "vmax.q R000, R000, R300",
    "vmax.q R001, R001, R300",
    "vmax.q R002, R002, R300",
    "vmax.q R003, R003, R300",
);
#[cfg(target_os = "psp")]
dw_kernel16!(
    dw16_swish,
    "vone.q R701",
    "vmul.q R300, R000, R700",
    "vmul.q R301, R001, R700",
    "vmul.q R302, R002, R700",
    "vmul.q R303, R003, R700",
    "vexp2.q R300, R300",
    "vexp2.q R301, R301",
    "vexp2.q R302, R302",
    "vexp2.q R303, R303",
    "vadd.q R300, R300, R701",
    "vadd.q R301, R301, R701",
    "vadd.q R302, R302, R701",
    "vadd.q R303, R303, R701",
    "vrcp.q R300, R300",
    "vrcp.q R301, R301",
    "vrcp.q R302, R302",
    "vrcp.q R303, R303",
    "vmul.q R000, R000, R300",
    "vmul.q R001, R001, R301",
    "vmul.q R002, R002, R302",
    "vmul.q R003, R003, R303",
);
#[cfg(target_os = "psp")]
dw_kernel4!(dw4_none);
#[cfg(target_os = "psp")]
dw_kernel4!(dw4_relu, "vzero.q R300", "vmax.q R000, R000, R300");
#[cfg(target_os = "psp")]
dw_kernel4!(
    dw4_swish,
    "vone.q R701",
    "vmul.q R300, R000, R700",
    "vexp2.q R300, R300",
    "vadd.q R300, R300, R701",
    "vrcp.q R300, R300",
    "vmul.q R000, R000, R300",
);

/// 8-lane depthwise kernel specialised to a full 3x3 tap window
/// (`row_count == kx_count == 3`), over a run of pixels. The 18 filter quads
/// stay in VFPU registers for the whole run, so each pixel costs 18 input
/// loads instead of the generic kernel's 36 input + 36 filter loads per
/// 16 lanes; the tap sequence is software-pipelined. Register map: M0 acc /
/// even-tap inputs, M1-M4 + R500/R501 filter, R502/R503 + R702/R703
/// products, R600/R601 odd-tap inputs, R602/R603 second accumulator, R700
/// -log2(e), R701 one; the epilogue may use R002/R003 as temporaries.
#[cfg(target_os = "psp")]
macro_rules! dw_kernel8_3x3 {
    ($name:ident $(, $act:tt)* $(,)?) => {
        #[inline(never)]
        unsafe fn $name(bias: *const f32, out: *mut f32, rows: &DwRows, ch_bytes: usize) {
            vfpu_asm!(
                "lv.q R700, 0({k})", // -log2(e), for the swish variant
                "vone.q R701",
                // Hold the 3x3 filter for these 8 channels in R100..R501 for the
                // whole run: tap t = (ky*3 + kx) is quads 2t (lanes 0-3) and 2t+1 (4-7).
                "addu {fa}, {f0}, {co}",
                "lv.q R100,  0({fa})",
                "lv.q R101, 16({fa})",
                "addu {fa}, {fa}, {cs}",
                "lv.q R102,  0({fa})",
                "lv.q R103, 16({fa})",
                "addu {fa}, {fa}, {cs}",
                "lv.q R200,  0({fa})",
                "lv.q R201, 16({fa})",
                "addu {fa}, {f1}, {co}",
                "lv.q R202,  0({fa})",
                "lv.q R203, 16({fa})",
                "addu {fa}, {fa}, {cs}",
                "lv.q R300,  0({fa})",
                "lv.q R301, 16({fa})",
                "addu {fa}, {fa}, {cs}",
                "lv.q R302,  0({fa})",
                "lv.q R303, 16({fa})",
                "addu {fa}, {f2}, {co}",
                "lv.q R400,  0({fa})",
                "lv.q R401, 16({fa})",
                "addu {fa}, {fa}, {cs}",
                "lv.q R402,  0({fa})",
                "lv.q R403, 16({fa})",
                "addu {fa}, {fa}, {cs}",
                "lv.q R500,  0({fa})",
                "lv.q R501, 16({fa})",
                "1:", // ---- next output pixel ----
                "lv.q R000,  0({bias})", // even-tap accumulator (taps 1,3,5,7)
                "lv.q R001, 16({bias})",
                // Software-pipelined taps: the loads for tap t+1 are issued before
                // the multiplies of tap t, and the adds of tap t-1 after them, so no
                // instruction waits on the one just before it. Two accumulator chains
                // (R000/R001 for odd taps, R602/R603 for even) halve the add latency
                // on the critical path.
                "addu {ia}, {i0}, {co}",
                "lv.q R002,  0({ia})",
                "lv.q R003, 16({ia})",
                "addu {ia}, {ia}, {cs}",
                "lv.q R600,  0({ia})",
                "lv.q R601, 16({ia})",
                "vmul.q R602, R002, R100",
                "vmul.q R603, R003, R101",
                "addu {ia}, {ia}, {cs}",
                "lv.q R002,  0({ia})",
                "lv.q R003, 16({ia})",
                "vmul.q R702, R600, R102",
                "vmul.q R703, R601, R103",
                "addu {ia}, {i1}, {co}",
                "lv.q R600,  0({ia})",
                "lv.q R601, 16({ia})",
                "vmul.q R502, R002, R200",
                "vmul.q R503, R003, R201",
                "vadd.q R000, R000, R702",
                "vadd.q R001, R001, R703",
                "addu {ia}, {ia}, {cs}",
                "lv.q R002,  0({ia})",
                "lv.q R003, 16({ia})",
                "vmul.q R702, R600, R202",
                "vmul.q R703, R601, R203",
                "vadd.q R602, R602, R502",
                "vadd.q R603, R603, R503",
                "addu {ia}, {ia}, {cs}",
                "lv.q R600,  0({ia})",
                "lv.q R601, 16({ia})",
                "vmul.q R502, R002, R300",
                "vmul.q R503, R003, R301",
                "vadd.q R000, R000, R702",
                "vadd.q R001, R001, R703",
                "addu {ia}, {i2}, {co}",
                "lv.q R002,  0({ia})",
                "lv.q R003, 16({ia})",
                "vmul.q R702, R600, R302",
                "vmul.q R703, R601, R303",
                "vadd.q R602, R602, R502",
                "vadd.q R603, R603, R503",
                "addu {ia}, {ia}, {cs}",
                "lv.q R600,  0({ia})",
                "lv.q R601, 16({ia})",
                "vmul.q R502, R002, R400",
                "vmul.q R503, R003, R401",
                "vadd.q R000, R000, R702",
                "vadd.q R001, R001, R703",
                "addu {ia}, {ia}, {cs}",
                "lv.q R002,  0({ia})",
                "lv.q R003, 16({ia})",
                "vmul.q R702, R600, R402",
                "vmul.q R703, R601, R403",
                "vadd.q R602, R602, R502",
                "vadd.q R603, R603, R503",
                "vmul.q R502, R002, R500",
                "vmul.q R503, R003, R501",
                "vadd.q R000, R000, R702",
                "vadd.q R001, R001, R703",
                "vadd.q R602, R602, R502",
                "vadd.q R603, R603, R503",
                "vadd.q R000, R000, R602",
                "vadd.q R001, R001, R603",
                $($act,)*
                "sv.q R000,  0({o})",
                "sv.q R001, 16({o})",
                // Advance every row to the next output pixel.
                "addu {i0}, {i0}, {pi}",
                "addu {i1}, {i1}, {pi}",
                "addu {i2}, {i2}, {pi}",
                "addiu {np}, {np}, -1",
                "bnez {np}, 1b",
                "addu {o}, {o}, {po}", // branch delay slot
                bias = in(reg) (bias),
                o = inout(reg) (out) => _,
                co = in(reg) (ch_bytes),
                k = in(reg) (NEG_LOG2_E.0.as_ptr()),
                i0 = inout(reg) (rows.input[0]) => _,
                i1 = inout(reg) (rows.input[1]) => _,
                i2 = inout(reg) (rows.input[2]) => _,
                f0 = in(reg) (rows.filter[0]),
                f1 = in(reg) (rows.filter[1]),
                f2 = in(reg) (rows.filter[2]),
                cs = in(reg) (rows.stride_bytes),
                pi = in(reg) (rows.pixel_in_bytes),
                po = in(reg) (rows.pixel_out_bytes),
                np = inout(reg) (rows.pixels) => _,
                ia = out(reg) _,
                fa = out(reg) _,
                options(nostack),
            );
        }
    };
}

#[cfg(target_os = "psp")]
dw_kernel8_3x3!(dw8_none);
#[cfg(target_os = "psp")]
dw_kernel8_3x3!(
    dw8_relu,
    "vzero.q R002",
    "vmax.q R000, R000, R002",
    "vmax.q R001, R001, R002",
);
#[cfg(target_os = "psp")]
dw_kernel8_3x3!(
    dw8_swish,
    "vmul.q R002, R000, R700",
    "vmul.q R003, R001, R700",
    "vexp2.q R002, R002",
    "vexp2.q R003, R003",
    "vadd.q R002, R002, R701",
    "vadd.q R003, R003, R701",
    "vrcp.q R002, R002",
    "vrcp.q R003, R003",
    "vmul.q R000, R000, R002",
    "vmul.q R001, R001, R003",
);

/// One group of `lanes` (16, 8 or 4) channels starting at `ch0`, over the
/// run of pixels in `rows`: for each pixel
/// `out[ch] = act(bias[ch] + Σ_taps input[..+ch] * filter[..+ch])`.
/// `out` is the output slice of the run's first pixel. 8 lanes is the 3x3
/// specialisation and needs `row_count == kx_count == 3`.
#[inline]
fn dw_group(
    lanes: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    rows: &DwRows,
    ch0: usize,
    act: Epilogue,
) {
    #[cfg(target_os = "psp")]
    {
        let bptr = match bias {
            Some(b) => b.as_ptr().wrapping_add(ch0),
            None => ZEROS16.0.as_ptr(),
        };
        debug_assert!(lanes != 8 || (rows.row_count == 3 && rows.kx_count == 3));
        let f: unsafe fn(*const f32, *mut f32, &DwRows, usize) = match (lanes, act) {
            (16, Epilogue::None) => dw16_none,
            (16, Epilogue::Relu) => dw16_relu,
            (16, Epilogue::Swish) => dw16_swish,
            (8, Epilogue::None) => dw8_none,
            (8, Epilogue::Relu) => dw8_relu,
            (8, Epilogue::Swish) => dw8_swish,
            (_, Epilogue::None) => dw4_none,
            (_, Epilogue::Relu) => dw4_relu,
            (_, Epilogue::Swish) => dw4_swish,
        };
        unsafe { f(bptr, out.as_mut_ptr().add(ch0), rows, ch0 * 4) };
    }
    #[cfg(not(target_os = "psp"))]
    {
        let stride = rows.stride_bytes / 4;
        let pin = rows.pixel_in_bytes / 4;
        let pout = rows.pixel_out_bytes / 4;
        for px in 0..rows.pixels {
            for l in 0..lanes {
                let ch = ch0 + l;
                let mut acc = bias.map_or(0.0, |b| b[ch]);
                for r in 0..rows.row_count {
                    for kx in 0..rows.kx_count {
                        // Host mirror of the pointer walk; the pointers were
                        // derived from slices in `depthwise_conv2d`.
                        unsafe {
                            acc += *rows.input[r].add(px * pin + kx * stride + ch)
                                * *rows.filter[r].add(kx * stride + ch);
                        }
                    }
                }
                out[px * pout + ch] = apply_epilogue(acc, act);
            }
        }
    }
}

/// Depthwise 2D convolution (NHWC), depth_multiplier = 1, with the bias and
/// activation fused into the store.
///
/// - `input`:  [N, H, W, C]
/// - `filter`: [1, Kh, Kw, C]
/// - `bias`:   [C]
/// - `padding`: [pad_top, pad_bottom, pad_left, pad_right]
/// - `output`: [N, Ho, Wo, C]
///
/// The fast path needs `C % 4 == 0`, `Kh <= 3` and 16-byte-aligned tensors
/// (always true for arena tensors and blob constants); anything else takes
/// the reference path.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d(
    input: &[f32],
    input_shape: [usize; 4],
    filter: &[f32],
    filter_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
    act: Epilogue,
) {
    let [n, h, w, c] = input_shape;
    let [_, kh, kw, _] = filter_shape;
    let [_, ho, wo, _] = output_shape;
    let [sh, sw] = stride;
    let [pad_top, _pad_bottom, pad_left, _pad_right] = padding;

    let aligned = c % 4 == 0
        && input.as_ptr() as usize % 16 == 0
        && filter.as_ptr() as usize % 16 == 0
        && output.as_ptr() as usize % 16 == 0
        && bias.map_or(true, |b| b.as_ptr() as usize % 16 == 0);
    if kh > 3 || !aligned {
        depthwise_conv2d_ref(
            input,
            input_shape,
            filter,
            filter_shape,
            bias,
            stride,
            padding,
            output,
            output_shape,
        );
        for v in output.iter_mut() {
            *v = apply_epilogue(*v, act);
        }
        return;
    }

    // Channels outermost, in chunks of `chunk` channels; within a chunk the
    // pixel sweep re-reads each input row `kh` times, so the `kh`-row window
    // it walks has to stay inside the 16 KB L1. Two knobs: the chunk width
    // and a column strip `strip` chosen so the window
    // `kh * (strip*sw + kw) * chunk * 4 B` stays within 8 KB, leaving the
    // rest of the cache to the filter and output.
    //
    // A pixel's channels are contiguous, so a chunk that covers the whole
    // pixel is always line-efficient; take that whenever it fits the lanes
    // budget (c <= 128) and let the strip carry the cache budget. Wider
    // tensors are chunked instead, never below 16 channels — one cache line
    // per pixel — or every line is fetched once per chunk. (Those chunks are
    // only line-aligned when `c % 16 == 0`; a 72-channel tensor chunked by 16
    // straddles two lines at most pixels and doubles its read traffic, which
    // is why whole-pixel chunks win below 128.)
    let chunk_max = if c <= 128 {
        c
    } else {
        let per_channel = kh * w * 4;
        let want = 8192 / per_channel.max(1);
        (want / 16 * 16).clamp(16, 128)
    };
    let strip = {
        let cols = 8192 / (kh * chunk_max * 4);
        (cols.saturating_sub(kw) / sw).clamp(4, wo.max(4))
    };

    // Output columns whose taps are all inside the padded width: from the
    // first `ox` with `ox*sw >= pad_left` up to the last with
    // `ox*sw + kw <= w + pad_left`. Those share one tap structure, so a
    // strip's interior goes to the kernel as a single run of pixels; the
    // few edge columns go one at a time. (The kernel's fixed cost is ~120
    // cycles per call and the driver's another ~200 — more than the 3x3 tap
    // loop itself for one pixel.)
    let ox_lo = pad_left.div_ceil(sw).min(wo);
    let ox_hi = ((w + pad_left).saturating_sub(kw) / sw + 1).min(wo);

    let mut c0 = 0;
    while c0 < c {
        let chunk = core::cmp::min(chunk_max, c - c0);
        for batch in 0..n {
            for ox0 in (0..wo).step_by(strip) {
                let ox_end = core::cmp::min(ox0 + strip, wo);
                for oy in 0..ho {
                    // In-bounds kernel rows: pad_top <= oy*sh + ky < h + pad_top.
                    let ky0 = pad_top.saturating_sub(oy * sh);
                    let ky1 = core::cmp::min(kh, (h + pad_top).saturating_sub(oy * sh));
                    let out_base = batch * (ho * wo * c) + oy * (wo * c);
                    let bias_only = |output: &mut [f32], ox: usize| {
                        let out_row = &mut output[out_base + ox * c..out_base + ox * c + c];
                        for ch in c0..c0 + chunk {
                            out_row[ch] = apply_epilogue(bias.map_or(0.0, |b| b[ch]), act);
                        }
                    };
                    if ky1 <= ky0 {
                        for ox in ox0..ox_end {
                            bias_only(output, ox);
                        }
                        continue;
                    }
                    // One kernel call per run: the edge columns singly, the
                    // interior in one go.
                    let mut ox = ox0;
                    while ox < ox_end {
                        let run = if ox >= ox_lo && ox < ox_hi {
                            core::cmp::min(ox_hi, ox_end) - ox
                        } else {
                            1
                        };
                        let kx0 = pad_left.saturating_sub(ox * sw);
                        let kx1 = core::cmp::min(kw, (w + pad_left).saturating_sub(ox * sw));
                        if kx1 <= kx0 {
                            bias_only(output, ox);
                            ox += run;
                            continue;
                        }
                        let mut rows = DwRows {
                            input: [input.as_ptr(); 3],
                            filter: [filter.as_ptr(); 3],
                            row_count: ky1 - ky0,
                            kx_count: kx1 - kx0,
                            stride_bytes: c * 4,
                            pixels: run,
                            pixel_in_bytes: sw * c * 4,
                            pixel_out_bytes: c * 4,
                        };
                        let ix = ox * sw + kx0 - pad_left;
                        for (r, ky) in (ky0..ky1).enumerate() {
                            let iy = oy * sh + ky - pad_top;
                            rows.input[r] =
                                input[batch * (h * w * c) + iy * (w * c) + ix * c..].as_ptr();
                            rows.filter[r] = filter[ky * (kw * c) + kx0 * c..].as_ptr();
                        }
                        let out_run = &mut output[out_base + ox * c..out_base + (ox + run) * c];
                        // Full 3x3 windows go to the register-resident 8-lane
                        // kernel; partial windows (padding rows/columns) to
                        // the generic 16-lane one.
                        let wide = if rows.row_count == 3 && rows.kx_count == 3 {
                            8
                        } else {
                            16
                        };
                        let mut ch = c0;
                        while ch + wide <= c0 + chunk {
                            dw_group(wide, bias, out_run, &rows, ch, act);
                            ch += wide;
                        }
                        while ch < c0 + chunk {
                            dw_group(4, bias, out_run, &rows, ch, act);
                            ch += 4;
                        }
                        ox += run;
                    }
                }
            }
        }
        c0 += chunk;
    }
}

/// `acc[i] = max(acc[i], a[i])` over `n` contiguous floats.
#[cfg(target_os = "psp")]
#[inline(never)]
fn vmax_inplace(acc: &mut [f32], a: &[f32], n: usize) {
    let quads = n / 4;
    if quads > 0 {
        unsafe {
            vfpu_asm!(
                "2:",
                "lv.q R000, 0({a})",
                "lv.q R100, 0({o})",
                "vmax.q R100, R100, R000",
                "sv.q R100, 0({o})",
                "addiu {a}, {a}, 16",
                "addiu {n}, {n}, -1",
                "bnez {n}, 2b",
                "addiu {o}, {o}, 16",
                a = inout(reg) (a.as_ptr()) => _,
                o = inout(reg) (acc.as_mut_ptr()) => _,
                n = inout(reg) (quads) => _,
                options(nostack),
            );
        }
    }
    for i in quads * 4..n {
        if a[i] > acc[i] {
            acc[i] = a[i];
        }
    }
}

#[cfg(not(target_os = "psp"))]
fn vmax_inplace(acc: &mut [f32], a: &[f32], n: usize) {
    for i in 0..n {
        if a[i] > acc[i] {
            acc[i] = a[i];
        }
    }
}

/// `acc[i] += a[i]` over `n` contiguous floats.
#[cfg(target_os = "psp")]
#[inline(never)]
fn vadd_inplace(acc: &mut [f32], a: &[f32], n: usize) {
    let quads = n / 4;
    if quads > 0 {
        unsafe {
            vfpu_asm!(
                "2:",
                "lv.q R000, 0({a})",
                "lv.q R100, 0({o})",
                "vadd.q R100, R100, R000",
                "sv.q R100, 0({o})",
                "addiu {a}, {a}, 16",
                "addiu {n}, {n}, -1",
                "bnez {n}, 2b",
                "addiu {o}, {o}, 16",
                a = inout(reg) (a.as_ptr()) => _,
                o = inout(reg) (acc.as_mut_ptr()) => _,
                n = inout(reg) (quads) => _,
                options(nostack),
            );
        }
    }
    for i in quads * 4..n {
        acc[i] += a[i];
    }
}

#[cfg(not(target_os = "psp"))]
fn vadd_inplace(acc: &mut [f32], a: &[f32], n: usize) {
    for i in 0..n {
        acc[i] += a[i];
    }
}

/// Collect the in-bounds taps for one output pixel as (input, filter) base
/// offsets. Shared by depthwise conv and both pooling kernels — this is the
/// bookkeeping that must not happen per channel.
#[allow(clippy::too_many_arguments)]
#[inline]
fn collect_taps(
    tap_in: &mut [usize],
    tap_f: &mut [usize],
    h: usize,
    w: usize,
    c: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    pad_top: usize,
    pad_left: usize,
    batch: usize,
    oy: usize,
    ox: usize,
) -> usize {
    let mut ntaps = 0;
    for ky in 0..kh {
        for kx in 0..kw {
            let iy_padded = oy * sh + ky;
            let ix_padded = ox * sw + kx;
            if iy_padded < pad_top || ix_padded < pad_left {
                continue;
            }
            let iy = iy_padded - pad_top;
            let ix = ix_padded - pad_left;
            if iy >= h || ix >= w {
                continue;
            }
            tap_in[ntaps] = batch * (h * w * c) + iy * (w * c) + ix * c;
            tap_f[ntaps] = ky * (kw * c) + kx * c;
            ntaps += 1;
        }
    }
    ntaps
}

/// 2D max pooling (NHWC). Same restructuring as `depthwise_conv2d`: the tap
/// bookkeeping happens once per output pixel, leaving a contiguous `vmax` over
/// channels.
pub fn max_pool2d(
    input: &[f32],
    input_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    let [n, h, w, c] = input_shape;
    let [kh, kw] = kernel;
    let [sh, sw] = stride;
    let [_, ho, wo, _] = output_shape;
    let [pad_top, _, pad_left, _] = padding;

    const MAX_TAPS: usize = 64;
    let mut tap_in = [0usize; MAX_TAPS];
    let mut tap_f = [0usize; MAX_TAPS];

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                let ntaps = if kh * kw <= MAX_TAPS {
                    collect_taps(
                        &mut tap_in,
                        &mut tap_f,
                        h,
                        w,
                        c,
                        kh,
                        kw,
                        sh,
                        sw,
                        pad_top,
                        pad_left,
                        batch,
                        oy,
                        ox,
                    )
                } else {
                    0
                };
                let out_base = batch * (ho * wo * c) + oy * (wo * c) + ox * c;
                let out_row = &mut output[out_base..out_base + c];
                if ntaps == 0 {
                    for v in out_row.iter_mut() {
                        *v = f32::NEG_INFINITY;
                    }
                    continue;
                }
                out_row.copy_from_slice(&input[tap_in[0]..tap_in[0] + c]);
                for t in 1..ntaps {
                    vmax_inplace(out_row, &input[tap_in[t]..tap_in[t] + c], c);
                }
            }
        }
    }
}

/// 2D average pooling (NHWC). Divides by the number of in-bounds taps, so
/// padded edges average over fewer elements (matching TFLite).
pub fn average_pool2d(
    input: &[f32],
    input_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    let [n, h, w, c] = input_shape;
    let [kh, kw] = kernel;
    let [sh, sw] = stride;
    let [_, ho, wo, _] = output_shape;
    let [pad_top, _, pad_left, _] = padding;

    const MAX_TAPS: usize = 64;
    let mut tap_in = [0usize; MAX_TAPS];
    let mut tap_f = [0usize; MAX_TAPS];

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                let ntaps = if kh * kw <= MAX_TAPS {
                    collect_taps(
                        &mut tap_in,
                        &mut tap_f,
                        h,
                        w,
                        c,
                        kh,
                        kw,
                        sh,
                        sw,
                        pad_top,
                        pad_left,
                        batch,
                        oy,
                        ox,
                    )
                } else {
                    0
                };
                let out_base = batch * (ho * wo * c) + oy * (wo * c) + ox * c;
                let out_row = &mut output[out_base..out_base + c];
                if ntaps == 0 {
                    for v in out_row.iter_mut() {
                        *v = 0.0;
                    }
                    continue;
                }
                out_row.copy_from_slice(&input[tap_in[0]..tap_in[0] + c]);
                for t in 1..ntaps {
                    vadd_inplace(out_row, &input[tap_in[t]..tap_in[t] + c], c);
                }
                let inv = 1.0 / ntaps as f32;
                for v in out_row.iter_mut() {
                    *v *= inv;
                }
            }
        }
    }
}

/// Straightforward reference used for the oversized-kernel fallback and by
/// tests. Correct but ~20x slower; see the module comment.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d_ref(
    input: &[f32],
    input_shape: [usize; 4],
    filter: &[f32],
    filter_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    let [n, h, w, c] = input_shape;
    let [_, kh, kw, _] = filter_shape;
    let [_, ho, wo, _] = output_shape;
    let [sh, sw] = stride;
    let [pad_top, _pad_bottom, pad_left, _pad_right] = padding;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                for ch in 0..c {
                    let mut sum = bias.map_or(0.0, |b| b[ch]);
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy_padded = oy * sh + ky;
                            let ix_padded = ox * sw + kx;
                            if iy_padded < pad_top || ix_padded < pad_left {
                                continue;
                            }
                            let iy = iy_padded - pad_top;
                            let ix = ix_padded - pad_left;
                            if iy >= h || ix >= w {
                                continue;
                            }
                            let in_idx = batch * (h * w * c) + iy * (w * c) + ix * c + ch;
                            let f_idx = ky * (kw * c) + kx * c + ch;
                            sum += input[in_idx] * filter[f_idx];
                        }
                    }
                    let out_idx = batch * (ho * wo * c) + oy * (wo * c) + ox * c + ch;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// ============================================================================
// Sigmoid / swish
// ============================================================================
//
// `sigmoid(x) = 1/(1 + e^-x) = 1/(1 + 2^(-x*log2 e))`, which the VFPU computes
// 4 lanes at a time with `vexp2.q` + `vrcp.q` — no software `expf` and no
// division. `swish(x) = x*sigmoid(x)` is the same chain plus one `vmul.q`,
// and fusing it saves a whole second pass over the tensor: BirdNET applies
// swish as a separate `Logistic` then `Mul` on 45 tensor pairs.
//
// `vexp2`/`vrcp` are hardware approximations, so results differ from `libm`
// in the last few bits — see `swish_matches_libm` in the tests for the
// measured bound.

#[cfg(target_os = "psp")]
static NEG_LOG2_E: Align16F4 = Align16F4([-core::f32::consts::LOG2_E; 4]);

#[cfg(target_os = "psp")]
#[repr(align(16))]
struct Align16F4([f32; 4]);

/// `out[i] = in[i].powf(c)` for a compile-time-constant exponent.
///
/// `x^c = 2^(c * log2 x)`, which the VFPU does 4 lanes at a time with
/// `vlog2.q` + `vmul.q` + `vexp2.q`. The alternative is `libm::powf` per
/// element, measured at ~678 cycles each on BirdNET's spectrogram compression.
///
/// Inputs must be **non-negative** — `log2` of a negative is NaN. BirdNET's
/// two call sites are fed by a square, so that holds there; the scalar tail and
/// the host fallback use `libm::powf`, which agrees for x >= 0.
///
/// `vlog2`/`vexp2` are hardware approximations, so this is *not* bit-identical
/// to `libm::powf` — same trade already accepted for `logistic`/`swish`.
#[cfg(target_os = "psp")]
#[inline(never)]
pub fn pow_const(input: &[f32], output: &mut [f32], exponent: f32) {
    let n = core::cmp::min(input.len(), output.len());
    let aligned = input.as_ptr() as usize % 16 == 0 && output.as_ptr() as usize % 16 == 0;
    let quads = if aligned { n / 4 } else { 0 };
    if quads > 0 {
        // Broadcast the exponent to all four lanes.
        let c4 = Align16F4([exponent; 4]);
        unsafe {
            vfpu_asm!(
                "lv.q R700, 0({c})",
                "2:",
                "lv.q R000, 0({i})",
                "vlog2.q R001, R000",        // log2 x
                "vmul.q R002, R001, R700",   // c * log2 x
                "vexp2.q R003, R002",        // 2^(c log2 x) = x^c
                "sv.q R003, 0({o})",
                "addiu {i}, {i}, 16",
                "addiu {n}, {n}, -1",
                "bnez {n}, 2b",
                "addiu {o}, {o}, 16",        // branch delay slot
                c = in(reg) (c4.0.as_ptr()),
                i = inout(reg) (input.as_ptr()) => _,
                o = inout(reg) (output.as_mut_ptr()) => _,
                n = inout(reg) (quads) => _,
                options(nostack),
            );
        }
    }
    for i in quads * 4..n {
        output[i] = libm::powf(input[i], exponent);
    }
}

/// Scalar mirror for host builds.
#[cfg(not(target_os = "psp"))]
pub fn pow_const(input: &[f32], output: &mut [f32], exponent: f32) {
    let n = core::cmp::min(input.len(), output.len());
    for i in 0..n {
        output[i] = libm::powf(input[i], exponent);
    }
}

/// `out[i] = (in[i] * in[i]).powf(c)` — the spectrogram compression fused
/// into one pass. BirdNET applies MUL(x,x) then POW(·, c) to each mel
/// projection; fusing saves a full read+write of the tensor, and squaring
/// first makes the base non-negative, so `pow_const`'s precondition holds by
/// construction. One `vmul.q` on top of `pow_const`'s pipeline; numerics are
/// otherwise identical to `mul` + `pow_const` (an IEEE f32 multiply is exact
/// in either engine).
#[cfg(target_os = "psp")]
#[inline(never)]
pub fn square_pow(input: &[f32], output: &mut [f32], exponent: f32) {
    let n = core::cmp::min(input.len(), output.len());
    let aligned = input.as_ptr() as usize % 16 == 0 && output.as_ptr() as usize % 16 == 0;
    let quads = if aligned { n / 4 } else { 0 };
    if quads > 0 {
        let c4 = Align16F4([exponent; 4]);
        unsafe {
            vfpu_asm!(
                "lv.q R700, 0({c})",
                "2:",
                "lv.q R000, 0({i})",
                "vmul.q R000, R000, R000",   // x^2
                "vlog2.q R001, R000",        // log2 x^2
                "vmul.q R002, R001, R700",   // c * log2 x^2
                "vexp2.q R003, R002",        // (x^2)^c
                "sv.q R003, 0({o})",
                "addiu {i}, {i}, 16",
                "addiu {n}, {n}, -1",
                "bnez {n}, 2b",
                "addiu {o}, {o}, 16",        // branch delay slot
                c = in(reg) (c4.0.as_ptr()),
                i = inout(reg) (input.as_ptr()) => _,
                o = inout(reg) (output.as_mut_ptr()) => _,
                n = inout(reg) (quads) => _,
                options(nostack),
            );
        }
    }
    for i in quads * 4..n {
        output[i] = libm::powf(input[i] * input[i], exponent);
    }
}

/// Scalar mirror for host builds.
#[cfg(not(target_os = "psp"))]
pub fn square_pow(input: &[f32], output: &mut [f32], exponent: f32) {
    let n = core::cmp::min(input.len(), output.len());
    for i in 0..n {
        output[i] = libm::powf(input[i] * input[i], exponent);
    }
}

// ============================================================================
// Banded (Compressed-Sparse-Column) fully connected — the mel projection
// ============================================================================

/// `output[b, m] = Σ_k input[m, start_b + k] * band_data[off_b + k]` — a
/// matmul against a column-banded matrix, one contiguous band of nonzeros
/// per output column. BirdNET's mel filterbanks are the motivating case:
/// [1025, 96] and [513, 96] matrices at 0.3% / 1.2% nonzero (bands of 1–17
/// bins), so the dense FC spends >99% of its MACs multiplying by zero.
///
/// `band_meta` holds `[start, len]` per output column (bank), and `band_data`
/// their coefficients concatenated in the same order.
///
/// **The output is `[out_features, rows]` — transposed** relative to the
/// dense FC. Each bank's GEMV over the 4-row group lands in one 4-vector, so
/// writing bank-major stores it contiguously with no transpose pass — and
/// bank-major is what the full model wants anyway (the mel projection is
/// followed by a TRANSPOSE there).
///
/// VFPU path, per bank, per 4 input rows: k GEMVs over 16-coefficient
/// chunks. The band (zero-padded to the chunk) multiplies a 4×16 window of
/// input — M000..M300 hold its four 4×4 tiles, M400's rows the four
/// coefficient subvectors, four `vtfm4.q` produce M500's rows, and their sum
/// accumulates in R600, stored with `sv.s` (the 511-row output stride is not
/// quad-aligned). Input rows aren't quad-aligned either (stride 1025/513 ≡ 1
/// mod 4), so each 4×W window is packed into an aligned scratch first —
/// per-group traffic the band's overlap keeps cache-resident. Memory-bound
/// and low arithmetic intensity by design: this is the correctness baseline,
/// not the end state.
///
/// The scalar mirror (host, tail rows, bands longer than 32) accumulates in
/// ascending k, so on the host the result is bit-identical to the dense
/// matmul with the equivalent mostly-zero dense matrix, transposed. The VFPU
/// path's chunk-tree summation differs in the last bits.
#[allow(clippy::too_many_arguments)]
pub fn fully_connected_cb(
    input: &[f32],
    rows: usize,
    in_features: usize,
    band_meta: &[i32],
    band_data: &[f32],
    output: &mut [f32],
    out_features: usize,
) {
    debug_assert_eq!(band_meta.len(), out_features * 2);

    #[cfg(target_os = "psp")]
    {
        fully_connected_cb_vfpu(
            input,
            rows,
            in_features,
            band_meta,
            band_data,
            output,
            out_features,
        );
    }

    #[cfg(not(target_os = "psp"))]
    {
        let mut off = 0usize;
        for b in 0..out_features {
            let start = band_meta[2 * b] as usize;
            let len = band_meta[2 * b + 1] as usize;
            let band = &band_data[off..off + len];
            off += len;
            let out_row = &mut output[b * rows..(b + 1) * rows];
            for (m, out) in out_row.iter_mut().enumerate() {
                *out = fc_cb_dot(&input[m * in_features + start..], band);
            }
        }
    }
}

/// One band dot in ascending-k order (the scalar reference order).
fn fc_cb_dot(row: &[f32], band: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, w) in row.iter().zip(band.iter()) {
        sum += x * w;
    }
    sum
}

#[cfg(target_os = "psp")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn fully_connected_cb_vfpu(
    input: &[f32],
    rows: usize,
    in_features: usize,
    band_meta: &[i32],
    band_data: &[f32],
    output: &mut [f32],
    out_features: usize,
) {
    // Zero-padded band coefficients (up to 2 chunks) and the packed 4×W
    // input window. Rows of `scratch` are a fixed 32 floats apart so the asm
    // walks chunks with one 64-byte stride for both W=16 and W=32.
    #[repr(align(16))]
    struct AlignedBuf<const N: usize>([f32; N]);
    let mut coef = AlignedBuf([0.0f32; 32]);
    let mut scratch = AlignedBuf([0.0f32; 4 * 32]);

    let mut off = 0usize;
    for b in 0..out_features {
        let start = band_meta[2 * b] as usize;
        let len = band_meta[2 * b + 1] as usize;
        let band = &band_data[off..off + len];
        off += len;
        let out_row = &mut output[b * rows..(b + 1) * rows];

        if len == 0 {
            out_row.fill(0.0);
            continue;
        }
        // BirdNET's bands are 1–17 long; anything past two chunks takes the
        // scalar path rather than growing the fixed buffers.
        if len > 32 {
            for (m, out) in out_row.iter_mut().enumerate() {
                *out = fc_cb_dot(&input[m * in_features + start..], band);
            }
            continue;
        }
        debug_assert!(start + len <= in_features);

        let chunks = if len <= 16 { 1usize } else { 2 };
        let w = chunks * 16;
        for k in 0..len {
            coef.0[k] = band[k];
        }
        for k in len..w {
            coef.0[k] = 0.0;
        }
        // The window may run past the row's end (the padding is zeros in
        // `coef`, but the packed input must not read the next row).
        let copy_w = w.min(in_features - start);

        let groups = rows / 4;
        for g in 0..groups {
            let m0 = g * 4;
            for r in 0..4 {
                let src = &input[(m0 + r) * in_features + start..];
                let dst = &mut scratch.0[r * 32..r * 32 + w];
                for k in 0..copy_w {
                    dst[k] = src[k];
                }
                for k in copy_w..w {
                    dst[k] = 0.0;
                }
            }
            unsafe {
                let s = scratch.0.as_ptr();
                vfpu_asm!(
                    "vzero.q R600",
                    "2:",
                    // M000..M300: the four 4×4 tiles of the 4×16 window.
                    "lv.q R000, 0({r0})",
                    "lv.q R001, 0({r1})",
                    "lv.q R002, 0({r2})",
                    "lv.q R003, 0({r3})",
                    "lv.q R100, 16({r0})",
                    "lv.q R101, 16({r1})",
                    "lv.q R102, 16({r2})",
                    "lv.q R103, 16({r3})",
                    "lv.q R200, 32({r0})",
                    "lv.q R201, 32({r1})",
                    "lv.q R202, 32({r2})",
                    "lv.q R203, 32({r3})",
                    "lv.q R300, 48({r0})",
                    "lv.q R301, 48({r1})",
                    "lv.q R302, 48({r2})",
                    "lv.q R303, 48({r3})",
                    // M400: the four coefficient subvectors.
                    "lv.q R400, 0({c})",
                    "lv.q R401, 16({c})",
                    "lv.q R402, 32({c})",
                    "lv.q R403, 48({c})",
                    // Four GEMVs into M500's rows. E-form: vtfm4's M-form
                    // dots the register matrix's *columns* with the vector
                    // (same implicit transpose as vmmul's first operand,
                    // pinned by a hardware diagnostic); E accesses the
                    // transpose, giving the row dots the lv.q row loads set up.
                    "vtfm4.q R500, E000, R400",
                    "vtfm4.q R501, E100, R401",
                    "vtfm4.q R502, E200, R402",
                    "vtfm4.q R503, E300, R403",
                    // ...sum the rows, accumulate across chunks in R600.
                    "vadd.q R500, R500, R501",
                    "vadd.q R502, R502, R503",
                    "vadd.q R500, R500, R502",
                    "vadd.q R600, R600, R500",
                    "addiu {r0}, {r0}, 64",
                    "addiu {r1}, {r1}, 64",
                    "addiu {r2}, {r2}, 64",
                    "addiu {r3}, {r3}, 64",
                    "addiu {n}, {n}, -1",
                    "bnez {n}, 2b",
                    "addiu {c}, {c}, 64", // branch delay slot
                    // Column-major result: 4 consecutive windows of bank b.
                    "sv.s S600, 0({o})",
                    "sv.s S610, 4({o})",
                    "sv.s S620, 8({o})",
                    "sv.s S630, 12({o})",
                    r0 = inout(reg) (s) => _,
                    r1 = inout(reg) (s.add(32)) => _,
                    r2 = inout(reg) (s.add(64)) => _,
                    r3 = inout(reg) (s.add(96)) => _,
                    c = inout(reg) (coef.0.as_ptr()) => _,
                    o = in(reg) (out_row.as_mut_ptr().add(m0)),
                    n = inout(reg) (chunks) => _,
                    options(nostack),
                );
            }
        }
        for m in groups * 4..rows {
            out_row[m] = fc_cb_dot(&input[m * in_features + start..], band);
        }
    }
}

/// `out[i] = in[i] * sigmoid(in[i])` (swish / SiLU).
#[cfg(target_os = "psp")]
#[inline(never)]
pub fn swish(input: &[f32], output: &mut [f32]) {
    let n = core::cmp::min(input.len(), output.len());
    // lv.q/sv.q need 16-byte alignment; arena tensors always are, but a
    // sub-slice might not be, so fall back rather than fault.
    let aligned = input.as_ptr() as usize % 16 == 0 && output.as_ptr() as usize % 16 == 0;
    let quads = if aligned { n / 4 } else { 0 };
    if quads > 0 {
        unsafe {
            vfpu_asm!(
                "lv.q R700, 0({c})",
                "vone.q R701",
                "2:",
                "lv.q R000, 0({i})",
                "vmul.q R001, R000, R700",   // -x * log2(e)
                "vexp2.q R002, R001",        // e^-x
                "vadd.q R003, R002, R701",   // 1 + e^-x
                "vrcp.q R100, R003",         // sigmoid(x)
                "vmul.q R101, R100, R000",   // x * sigmoid(x)
                "sv.q R101, 0({o})",
                "addiu {i}, {i}, 16",
                "addiu {n}, {n}, -1",
                "bnez {n}, 2b",
                "addiu {o}, {o}, 16",        // branch delay slot
                c = in(reg) (NEG_LOG2_E.0.as_ptr()),
                i = inout(reg) (input.as_ptr()) => _,
                o = inout(reg) (output.as_mut_ptr()) => _,
                n = inout(reg) (quads) => _,
                options(nostack),
            );
        }
    }
    for i in quads * 4..n {
        let x = input[i];
        output[i] = x / (1.0 + libm::expf(-x));
    }
}

#[cfg(not(target_os = "psp"))]
pub fn swish(input: &[f32], output: &mut [f32]) {
    let n = core::cmp::min(input.len(), output.len());
    for i in 0..n {
        let x = input[i];
        output[i] = x / (1.0 + libm::expf(-x));
    }
}

/// `out[i] = sigmoid(in[i])`. Same chain as `swish` without the final multiply.
#[cfg(target_os = "psp")]
#[inline(never)]
pub fn logistic(input: &[f32], output: &mut [f32]) {
    let n = core::cmp::min(input.len(), output.len());
    let aligned = input.as_ptr() as usize % 16 == 0 && output.as_ptr() as usize % 16 == 0;
    let quads = if aligned { n / 4 } else { 0 };
    if quads > 0 {
        unsafe {
            vfpu_asm!(
                "lv.q R700, 0({c})",
                "vone.q R701",
                "2:",
                "lv.q R000, 0({i})",
                "vmul.q R001, R000, R700",
                "vexp2.q R002, R001",
                "vadd.q R003, R002, R701",
                "vrcp.q R100, R003",
                "sv.q R100, 0({o})",
                "addiu {i}, {i}, 16",
                "addiu {n}, {n}, -1",
                "bnez {n}, 2b",
                "addiu {o}, {o}, 16",
                c = in(reg) (NEG_LOG2_E.0.as_ptr()),
                i = inout(reg) (input.as_ptr()) => _,
                o = inout(reg) (output.as_mut_ptr()) => _,
                n = inout(reg) (quads) => _,
                options(nostack),
            );
        }
    }
    for i in quads * 4..n {
        output[i] = 1.0 / (1.0 + libm::expf(-input[i]));
    }
}

#[cfg(not(target_os = "psp"))]
pub fn logistic(input: &[f32], output: &mut [f32]) {
    let n = core::cmp::min(input.len(), output.len());
    for i in 0..n {
        output[i] = 1.0 / (1.0 + libm::expf(-input[i]));
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
    stride: [usize; 2],
    padding: [usize; 4],
    output_hw: [usize; 2],
    col: &mut [f32],
) {
    let [n, h, w, ci] = input_shape;
    let [kh, kw] = kernel;
    let [sh, sw] = stride;
    let [pad_top, _pad_bottom, pad_left, _pad_right] = padding;
    let [ho, wo] = output_hw;
    let k = kh * kw * ci;
    let k_padded = div_ceil(k, VFPU_Q) * VFPU_Q;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                let row = batch * (ho * wo) + oy * wo + ox;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = (oy * sh + ky) as isize - pad_top as isize;
                        let ix = (ox * sw + kx) as isize - pad_left as isize;
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

#[cfg(test)]
mod gemm_tests {
    use super::*;
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    /// Deterministic pseudo-random fill in [-1, 1).
    fn fill(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 8388608.0) - 1.0
            })
            .collect()
    }

    /// f64 reference — comparing two f32 orderings would make the less
    /// accurate one the oracle.
    fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += a[i * k + kk] as f64 * b[j * k + kk] as f64;
                }
                c[i * n + j] = acc as f32;
            }
        }
        c
    }

    fn check(m: usize, k: usize, n: usize, mc: usize, kc: usize) {
        let a = fill(m * k, 12345);
        let b = fill(n * k, 999);
        let want = reference(&a, &b, m, k, n);

        let mut bp = vec![0.0f32; gemm_bp_len(n, k)];
        pack_b_panel(&b, &mut bp, n, k);

        // Dirty the scratch: the arena is reused and never re-zeroed, so any
        // missing pad-zeroing must show up as a failure here.
        let mut ap = vec![7.5f32; gemm_ap_len(mc, kc)];
        let mut cp = vec![-3.25f32; gemm_cp_len(mc, n)];
        let mut got = vec![f32::NAN; m * n];

        gemm_bt_packed(&a, k, &bp, &mut got, &mut ap, &mut cp, m, k, n, mc, kc);

        // Elementwise *relative* error is meaningless here: with random signed
        // inputs a length-K dot product can cancel to near zero, so a tiny
        // absolute error shows up as a huge ratio. Judge against the scale of
        // the result matrix instead — max error normalised by RMS — plus the
        // Frobenius norm, which is what actually catches a mis-indexed tile.
        let mut max_abs = 0.0f32;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..m * n {
            let d = (got[i] - want[i]).abs();
            max_abs = max_abs.max(d);
            num += (d as f64) * (d as f64);
            den += (want[i] as f64) * (want[i] as f64);
        }
        let frob = (num / den.max(1e-30)).sqrt();
        let rms = (den / (m * n) as f64).sqrt().max(1e-9) as f32;
        let max_scaled = max_abs / rms;
        assert!(
            max_scaled < 1e-3 && frob < 1e-5,
            "m={m} k={k} n={n} mc={mc} kc={kc}: \
             max_abs={max_abs:e} (={max_scaled:e} of rms {rms:e}) frobenius={frob:e}"
        );
    }

    #[test]
    fn birdnet_op14_shape() {
        check(511, 1025, 96, 32, 32);
    }

    #[test]
    fn birdnet_op28_shape() {
        check(511, 513, 96, 32, 32);
    }

    #[test]
    fn blocking_factors_do_not_change_result() {
        for (mc, kc) in [(32, 32), (48, 16), (16, 64), (64, 8)] {
            check(511, 1025, 96, mc, kc);
        }
    }

    #[test]
    fn awkward_dims() {
        // every combination of ragged M, K and N
        check(1, 1, 1, 32, 32);
        check(3, 7, 5, 32, 32);
        check(9, 13, 12, 8, 8);
        check(37, 1025, 96, 32, 32);
    }

    #[test]
    fn matches_naive_fully_connected() {
        // The exact call the codegen replaces: batch of GEMVs vs one GEMM.
        let (m, k, n) = (64, 129, 24);
        let a = fill(m * k, 7);
        let b = fill(n * k, 8);

        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            naive::fully_connected(
                &a[i * k..(i + 1) * k],
                k,
                &b,
                None,
                &mut want[i * n..(i + 1) * n],
                n,
            );
        }

        let mut bp = vec![0.0f32; gemm_bp_len(n, k)];
        pack_b_panel(&b, &mut bp, n, k);
        let mut ap = vec![0.0f32; gemm_ap_len(32, 32)];
        let mut cp = vec![0.0f32; gemm_cp_len(32, n)];
        let mut got = vec![0.0f32; m * n];
        gemm_bt_packed(&a, k, &bp, &mut got, &mut ap, &mut cp, m, k, n, 32, 32);

        for i in 0..m * n {
            let scale = got[i].abs().max(want[i].abs()).max(1e-6);
            assert!(
                (got[i] - want[i]).abs() / scale < 1e-3,
                "i={i} got={} want={}",
                got[i],
                want[i]
            );
        }
    }
}

#[cfg(test)]
mod depthwise_tests {
    use super::*;
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    fn fill(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 8388608.0) - 1.0
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn check(
        h: usize,
        w: usize,
        c: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        pad: [usize; 4],
        with_bias: bool,
    ) {
        let ho = (h + pad[0] + pad[1] - kh) / sh + 1;
        let wo = (w + pad[2] + pad[3] - kw) / sw + 1;
        let input = fill(h * w * c, 11);
        let filter = fill(kh * kw * c, 22);
        let bias = fill(c, 33);
        let b = with_bias.then_some(&bias[..]);

        let mut got = vec![f32::NAN; ho * wo * c];
        let mut want = vec![f32::NAN; ho * wo * c];
        depthwise_conv2d(
            &input,
            [1, h, w, c],
            &filter,
            [1, kh, kw, c],
            b,
            [sh, sw],
            pad,
            &mut got,
            [1, ho, wo, c],
            Epilogue::None,
        );
        depthwise_conv2d_ref(
            &input,
            [1, h, w, c],
            &filter,
            [1, kh, kw, c],
            b,
            [sh, sw],
            pad,
            &mut want,
            [1, ho, wo, c],
        );

        for i in 0..got.len() {
            let scale = got[i].abs().max(want[i].abs()).max(1e-6);
            assert!(
                (got[i] - want[i]).abs() / scale < 1e-5,
                "h={h} w={w} c={c} k={kh}x{kw} s={sh}x{sw} pad={pad:?} bias={with_bias}: \
                 i={i} got={} want={}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn birdnet_shapes() {
        // The real depthwise layers: 3x3, SAME and strided, C a multiple of 16.
        check(12, 32, 288, 3, 3, 1, 1, [1, 1, 1, 1], true);
        check(14, 34, 864, 3, 3, 2, 2, [0, 0, 0, 0], true);
        check(6, 16, 864, 3, 3, 1, 1, [1, 1, 1, 1], true);
        check(3, 8, 1536, 3, 3, 1, 1, [1, 1, 1, 1], true);
    }

    #[test]
    fn no_bias() {
        check(6, 16, 288, 3, 3, 1, 1, [1, 1, 1, 1], false);
    }

    #[test]
    fn channel_counts_not_multiples_of_the_vector_width() {
        // Channel counts that are not multiples of 4 take the reference path.
        for c in [1, 3, 4, 7, 16, 17, 20, 33] {
            check(5, 5, c, 3, 3, 1, 1, [1, 1, 1, 1], true);
        }
    }

    #[test]
    fn asymmetric_padding_and_strides() {
        check(7, 7, 32, 3, 3, 2, 2, [1, 1, 1, 1], true);
        check(8, 8, 32, 3, 3, 1, 2, [0, 1, 0, 1], true);
        check(5, 9, 48, 1, 1, 1, 1, [0, 0, 0, 0], false);
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    fn fill(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 8388608.0) - 1.0
            })
            .collect()
    }

    /// Reference: channel loop outside the taps, exactly as TFLite specifies.
    fn pool_ref(
        input: &[f32],
        (h, w, c): (usize, usize, usize),
        (kh, kw): (usize, usize),
        (sh, sw): (usize, usize),
        pad: [usize; 4],
        (ho, wo): (usize, usize),
        is_max: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; ho * wo * c];
        for oy in 0..ho {
            for ox in 0..wo {
                for ch in 0..c {
                    let mut acc = if is_max { f32::NEG_INFINITY } else { 0.0 };
                    let mut count = 0usize;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oy * sh + ky) as isize - pad[0] as isize;
                            let ix = (ox * sw + kx) as isize - pad[2] as isize;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                let v = input[(iy as usize) * (w * c) + (ix as usize) * c + ch];
                                if is_max {
                                    if v > acc {
                                        acc = v;
                                    }
                                } else {
                                    acc += v;
                                }
                                count += 1;
                            }
                        }
                    }
                    out[oy * (wo * c) + ox * c + ch] = if is_max {
                        acc
                    } else if count > 0 {
                        acc / count as f32
                    } else {
                        0.0
                    };
                }
            }
        }
        out
    }

    fn check(
        h: usize,
        w: usize,
        c: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        pad: [usize; 4],
    ) {
        let ho = (h + pad[0] + pad[1] - kh) / sh + 1;
        let wo = (w + pad[2] + pad[3] - kw) / sw + 1;
        let input = fill(h * w * c, 55);

        for is_max in [true, false] {
            let want = pool_ref(&input, (h, w, c), (kh, kw), (sh, sw), pad, (ho, wo), is_max);
            let mut got = vec![f32::NAN; ho * wo * c];
            if is_max {
                max_pool2d(
                    &input,
                    [1, h, w, c],
                    [kh, kw],
                    [sh, sw],
                    pad,
                    &mut got,
                    [1, ho, wo, c],
                );
            } else {
                average_pool2d(
                    &input,
                    [1, h, w, c],
                    [kh, kw],
                    [sh, sw],
                    pad,
                    &mut got,
                    [1, ho, wo, c],
                );
            }
            for i in 0..got.len() {
                let scale = got[i].abs().max(want[i].abs()).max(1e-6);
                assert!(
                    (got[i] - want[i]).abs() / scale < 1e-5,
                    "max={is_max} h={h} w={w} c={c} k={kh}x{kw} s={sh}x{sw} pad={pad:?}: \
                     i={i} got={} want={}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn birdnet_shapes() {
        // [1,48,1,24] with a 1x2 filter, the shapes actually in the graph
        check(48, 2, 24, 1, 2, 1, 2, [0, 0, 0, 0]);
        check(24, 8, 288, 2, 2, 2, 2, [0, 0, 0, 0]);
    }

    #[test]
    fn padding_averages_over_valid_taps_only() {
        check(5, 5, 16, 3, 3, 1, 1, [1, 1, 1, 1]);
        check(7, 7, 32, 3, 3, 2, 2, [1, 1, 1, 1]);
    }

    #[test]
    fn channel_tails() {
        for c in [1, 3, 7, 17, 33] {
            check(5, 5, c, 2, 2, 1, 1, [0, 0, 0, 0]);
        }
    }
}

#[cfg(test)]
mod rfft_tests {
    use super::*;
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    /// Twiddles in the same split layout psp-tc emits.
    fn twiddles(n: usize) -> (Vec<f32>, Vec<f32>) {
        let nc = n / 2;
        let stages = nc.trailing_zeros() as usize;
        let mut stage = Vec::new();
        for s in 0..stages {
            let half = 1usize << s;
            let start = stage.len();
            for j in 0..half {
                let a = -2.0 * core::f64::consts::PI * j as f64 / (2.0 * half as f64);
                stage.push(a.cos() as f32);
            }
            for j in 0..half {
                let a = -2.0 * core::f64::consts::PI * j as f64 / (2.0 * half as f64);
                stage.push(a.sin() as f32);
            }
            stage.resize(start + stage_tw_block(s), 0.0);
        }
        let mut unpack = Vec::new();
        for k in 1..nc {
            let a = 2.0 * core::f64::consts::PI * k as f64 / n as f64;
            unpack.push(a.cos() as f32);
        }
        for k in 1..nc {
            let a = 2.0 * core::f64::consts::PI * k as f64 / n as f64;
            unpack.push(-(a.sin() as f32));
        }
        (stage, unpack)
    }

    /// Oracle: real-input DFT in f64, keeping only the real parts.
    fn dft_real(x: &[f32], n: usize) -> Vec<f32> {
        (0..=n / 2)
            .map(|k| {
                let mut acc = 0.0f64;
                for (t, &v) in x.iter().enumerate().take(n) {
                    acc += v as f64
                        * (-2.0 * core::f64::consts::PI * (k * t % n) as f64 / n as f64).cos();
                }
                acc as f32
            })
            .collect()
    }

    fn check(n: usize, frames: usize) {
        let mut s = 12345u32;
        let input: Vec<f32> = (0..n * frames)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 8388608.0) - 1.0
            })
            .collect();
        let (stage, unpack) = twiddles(n);
        // Dirty scratch: the arena is reused and never re-zeroed.
        let mut scratch = vec![7.5f32; n];
        let mut out = vec![f32::NAN; (n / 2 + 1) * frames];

        rfft_batch(&input, &stage, &unpack, &mut scratch, &mut out, n, frames);

        for f in 0..frames {
            let want = dft_real(&input[f * n..(f + 1) * n], n);
            let bins = n / 2 + 1;
            let rms = (want.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / bins as f64)
                .sqrt()
                .max(1e-9) as f32;
            for k in 0..bins {
                let d = (out[f * bins + k] - want[k]).abs();
                assert!(
                    d / rms < 1e-4,
                    "n={n} frame={f} bin={k}: got {} want {} (rms {rms})",
                    out[f * bins + k],
                    want[k]
                );
            }
        }
    }

    #[test]
    fn birdnet_sizes() {
        // The two real sizes in the graph, a couple of frames each.
        check(2048, 2);
        check(1024, 2);
    }

    /// The bug the scalar host mirror structurally cannot catch: every stage's
    /// twiddle run is loaded with `lv.q`, so it must start 4-float aligned.
    /// The natural packing (offset `2*(2^s - 1)`) is 2 mod 4 for every stage.
    #[test]
    fn stage_twiddle_runs_are_quad_aligned() {
        for n in [64usize, 128, 1024, 2048] {
            let nc = n / 2;
            let stages = nc.trailing_zeros() as usize;
            let mut off = stage_tw_block(0);
            for s in 1..stages {
                let half = 1usize << s;
                // Only stages wide enough for the vector path issue `lv.q`;
                // narrower ones fall to the scalar loop.
                if half >= 4 {
                    assert_eq!(off % 4, 0, "n={n} stage={s}: tw_re at float {off}");
                    assert_eq!((off + half) % 4, 0, "n={n} stage={s}: tw_im misaligned");
                }
                off += stage_tw_block(s);
            }
            assert_eq!(
                off,
                stage_tw_len(n),
                "n={n}: block sizes must tile the buffer"
            );
        }
    }

    #[test]
    fn small_sizes() {
        for n in [8, 16, 32, 64, 128] {
            check(n, 3);
        }
    }

    /// The contract `rfft_strided_batch` exists to honour: bit-identical to
    /// materialising every overlapping window (the dense gather), multiplying
    /// by the window, and running `rfft_batch` on the result.
    fn check_strided(n: usize, hop: usize, frames: usize, windowed: bool) {
        check_strided_at(n, hop, 1, frames, windowed);
    }

    fn check_strided_at(n: usize, hop: usize, in_stride: usize, frames: usize, windowed: bool) {
        let n_samples = (frames - 1) * hop + in_stride * (n - 1) + 4; // a few trailing samples
        let mut s = 99u32;
        let mut rand = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 8388608.0) - 1.0
        };
        let samples: Vec<f32> = (0..n_samples).map(|_| rand()).collect();
        let window: Option<Vec<f32>> = windowed.then(|| (0..n).map(|_| rand()).collect());
        let (stage, unpack) = twiddles(n);
        let bins = n / 2 + 1;

        // Dense reference: gather + window multiply + rfft_batch.
        let mut dense = vec![0.0f32; frames * n];
        for f in 0..frames {
            for j in 0..n {
                let w = window.as_ref().map_or(1.0, |w| w[j]);
                dense[f * n + j] = samples[f * hop + in_stride * j] * w;
            }
        }
        let mut scratch = vec![7.5f32; n];
        let mut want = vec![f32::NAN; bins * frames];
        rfft_batch(&dense, &stage, &unpack, &mut scratch, &mut want, n, frames);

        let mut scratch = vec![-3.25f32; n];
        let mut got = vec![f32::NAN; bins * frames];
        rfft_strided_batch(
            &samples,
            window.as_deref(),
            &stage,
            &unpack,
            &mut scratch,
            &mut got,
            n,
            hop,
            in_stride,
            frames,
        );

        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                g.to_bits() == w.to_bits(),
                "n={n} hop={hop} stride={in_stride} frames={frames} windowed={windowed} elem {i}: {g} != {w}"
            );
        }
    }

    #[test]
    fn strided_matches_dense_small() {
        check_strided(16, 5, 4, true);
        check_strided(32, 32, 3, false); // hop == n: identical to rfft_batch
        check_strided(64, 7, 6, true); // heavily overlapping windows
    }

    #[test]
    fn strided_matches_dense_birdnet_shapes() {
        // The real branches, a handful of frames: L=2048 hop=278, L=1024 hop=280.
        check_strided(2048, 278, 5, true);
        check_strided(1024, 280, 5, true);
    }

    #[test]
    fn inner_stride_matches_gathered_frames() {
        check_strided_at(16, 5, 2, 4, true);
        check_strided_at(32, 7, 3, 5, true);
        // The pruned 2048 branch's real shape: 512-point frames read at
        // stride 2 from the half-rate signal, hop 139.
        check_strided_at(512, 139, 2, 5, true);
    }

    #[test]
    fn fir_decimate_reference() {
        // Impulse: output picks up the taps at the decimated positions.
        let mut x = vec![0.0f32; 32];
        x[10] = 1.0;
        let taps = [0.25f32, 0.5, 1.0, 0.5, 0.25]; // center = 2
        let mut y = vec![0.0f32; 16];
        fir_decimate(&x, &taps, &mut y, 2);
        // y[n] = sum_t taps[t] * x[2n + t - 2] -> hits x[10] when 2n + t = 12.
        let mut want = vec![0.0f32; 16];
        for (t, &h) in taps.iter().enumerate() {
            if (12 - t as i32) % 2 == 0 {
                let n = (12 - t as i32) / 2;
                if (0..16).contains(&n) {
                    want[n as usize] += h;
                }
            }
        }
        assert_eq!(y, want);

        // DC gain: a constant input yields sum(taps) away from the edges.
        let x = vec![1.0f32; 64];
        let mut y = vec![0.0f32; 32];
        fir_decimate(&x, &taps, &mut y, 2);
        let dc: f32 = taps.iter().sum();
        for n in 2..30 {
            assert!((y[n] - dc).abs() < 1e-6, "n={n}: {} vs {dc}", y[n]);
        }
        // Edge outputs see zero-padding (fewer taps in range).
        assert!(y[0] < dc);
    }
}

#[cfg(test)]
mod mel_tests {
    use super::*;
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    /// The CB contract on the host (scalar path): bit-identical to the dense
    /// matmul with the equivalent (mostly-zero) dense matrix — transposed,
    /// since the CB output is `[out_features, rows]` — because zero terms
    /// contribute exactly nothing to an ordered f32 accumulation.
    #[test]
    fn fc_cb_matches_dense_bitwise() {
        let (rows, in_features, out_features) = (7, 40, 9);
        let mut s = 5u32;
        let mut rand = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 8388608.0) - 1.0
        };

        // Random bands, including len-1, a 17-long band (the second-chunk
        // case on device), and a band whose padded window runs past the row.
        let mut band_meta = Vec::new();
        let mut band_data = Vec::new();
        let mut dense = vec![0.0f32; out_features * in_features];
        for b in 0..out_features {
            let (start, len) = match b {
                7 => (5, 17),
                8 => (36, 4),
                _ => ((b * 4) % (in_features - 6), 1 + (b % 6)),
            };
            band_meta.push(start as i32);
            band_meta.push(len as i32);
            for k in 0..len {
                let v = rand();
                band_data.push(v);
                dense[b * in_features + start + k] = v;
            }
        }
        let input: Vec<f32> = (0..rows * in_features).map(|_| rand()).collect();

        let mut want = vec![0.0f32; rows * out_features];
        for m in 0..rows {
            naive::fully_connected(
                &input[m * in_features..(m + 1) * in_features],
                in_features,
                &dense,
                None,
                &mut want[m * out_features..(m + 1) * out_features],
                out_features,
            );
        }

        let mut got = vec![f32::NAN; rows * out_features];
        fully_connected_cb(
            &input,
            rows,
            in_features,
            &band_meta,
            &band_data,
            &mut got,
            out_features,
        );

        for m in 0..rows {
            for b in 0..out_features {
                let (g, w) = (got[b * rows + m], want[m * out_features + b]);
                assert!(g.to_bits() == w.to_bits(), "m={m} b={b}: {g} != {w}");
            }
        }
    }

    /// Fusing the square must change nothing vs the two-op form.
    #[test]
    fn square_pow_matches_mul_then_pow() {
        let input: Vec<f32> = (0..37).map(|i| (i as f32 - 11.0) * 0.3).collect();
        let mut squared = vec![0.0f32; input.len()];
        for (s, x) in squared.iter_mut().zip(input.iter()) {
            *s = x * x;
        }
        let mut want = vec![0.0f32; input.len()];
        pow_const(&squared, &mut want, 0.2199);
        let mut got = vec![f32::NAN; input.len()];
        square_pow(&input, &mut got, 0.2199);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(g.to_bits() == w.to_bits(), "elem {i}: {g} != {w}");
        }
    }
}
