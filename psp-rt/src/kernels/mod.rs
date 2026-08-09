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
    let kk_span = kt_count * 4;
    for mb in 0..mb_count {
        let row0 = m0 + mb * GEMM_MR;
        let block = mb * kt_count * GEMM_MR * 4;
        // Row-outer so each A row is read as one sequential stream. The
        // transposed alternative (kt outer) interleaves four rows 4100 bytes
        // apart and measured ~1.5x slower on the DRAM side.
        for r in 0..GEMM_MR {
            let row = row0 + r;
            // Fast path: this whole row of the block is interior. Hoisting the
            // edge test out of the element loop matters — with a bounds check
            // per element the pack cost 31 cycles/float.
            if row < m && k0 + kk_span <= k {
                let mut src = row * lda + k0;
                let mut idx = block + r * 4;
                unsafe {
                    let ab = a.as_ptr();
                    let pb = ap.as_mut_ptr();
                    for _ in 0..kt_count {
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
            } else {
                for kt in 0..kt_count {
                    let idx = block + (kt * GEMM_MR + r) * 4;
                    for kk in 0..4 {
                        let col = k0 + kt * 4 + kk;
                        ap[idx + kk] = if row < m && col < k {
                            a[row * lda + col]
                        } else {
                            0.0
                        };
                    }
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
    vfpu_asm!(
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
        a = inout(reg) (ap) => _,
        b = inout(reg) (bp) => _,
        c = in(reg) (cp),
        kp = inout(reg) (kt_count / 2) => _,
        options(nostack),
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

/// Cache-blocked GEMM with B pre-packed: `C[m,n] = A[m,k] @ B[n,k]^T`.
///
/// `m`, `k` and `n` are arbitrary — padding is absorbed by the packing steps,
/// so there is no alignment requirement on the caller's tensors and no scalar
/// tail path. `bp` must come from `pack_b_panel` (or psp-tc's compile-time
/// equivalent); `ap` and `cp` are scratch of at least `gemm_ap_len(mc, kc)`
/// and `gemm_cp_len(mc, n)` floats, both 16-byte aligned.
///
/// `lda` is A's row stride, which may exceed `k` (im2col writes rows padded to
/// a multiple of 4, and those pad columns hold stale arena data — passing the
/// logical `k` separately means they are never read).
///
/// `mc` and `kc` are the L1 blocking factors; `kc` must be a multiple of 8.
/// mc=32, kc=32 keeps the B panel (n*kc) and C slab (mc*n) together under
/// ~25 KB of the 32 KB L1.
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
    debug_assert!(kc % GEMM_KPAD == 0, "kc must be a multiple of 8");
    debug_assert!(ap.as_ptr() as usize % 16 == 0, "ap must be 16-byte aligned");
    debug_assert!(cp.as_ptr() as usize % 16 == 0, "cp must be 16-byte aligned");

    let kt_total = div_ceil(k, GEMM_KPAD) * 2;
    let nb_total = div_ceil(n, GEMM_NR);
    let ktc_max = kc / 4;

    let mut m0 = 0;
    while m0 < m {
        let rows = core::cmp::min(mc, m - m0);
        let mb_count = div_ceil(rows, GEMM_MR);

        // Zero the C slab: the arena is reused between ops and never re-zeroed.
        let cp_len = mb_count * nb_total * GEMM_MR * GEMM_NR;
        for v in cp[..cp_len].iter_mut() {
            *v = 0.0;
        }

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
                            cp.as_mut_ptr().add((nb * mb_count + mb) * GEMM_MR * GEMM_NR),
                            ktc,
                        );
                    }
                }
            }
            kt0 += ktc;
        }

        unpack_c_block(cp, c, m, n, m0, mb_count, nb_total);
        m0 += mc;
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
#[repr(align(16))]
struct Align16F4([f32; 4]);

#[cfg(target_os = "psp")]
static NEG_LOG2_E: Align16F4 = Align16F4([-core::f32::consts::LOG2_E; 4]);

/// `out[i] = in[i] * sigmoid(in[i])` (swish / SiLU).
#[cfg(target_os = "psp")]
#[inline(never)]
pub fn swish(input: &[f32], output: &mut [f32]) {
    let n = core::cmp::min(input.len(), output.len());
    // lv.q/sv.q need 16-byte alignment; arena tensors always are, but a
    // sub-slice might not be, so fall back rather than fault.
    let aligned =
        input.as_ptr() as usize % 16 == 0 && output.as_ptr() as usize % 16 == 0;
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
    let aligned =
        input.as_ptr() as usize % 16 == 0 && output.as_ptr() as usize % 16 == 0;
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
