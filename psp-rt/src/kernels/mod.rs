//! PSP Neural Network Kernels
//!
//! This module contains the operator implementations that the compiler emits.
//!
//! - `naive`: Reference implementations that work on any target.
//! - Root module: Optimized VFPU-accelerated kernels (PSP uses VFPU, host uses scalar fallbacks).

pub mod checks;

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

/// Bit-reversed pack of a real frame into split complex arrays.
fn rfft_pack_split(input: &[f32], re: &mut [f32], im: &mut [f32], nc: usize) {
    // Reversed-counter increment rather than reversing each index from
    // scratch: the naive form costs log2(nc) shifts per element *per frame*
    // (~5 M operations across BirdNET's 511 frames) to recompute the identical
    // permutation every time. Carrying `rev` from one k to the next is O(1)
    // amortised.
    let mut rev = 0usize;
    for k in 0..nc {
        re[k] = input[2 * rev];
        im[k] = input[2 * rev + 1];
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
        rfft_pack_split(&input[f * n..(f + 1) * n], re, im, nc);
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
        rfft_unpack_split(
            re,
            im,
            utw_re,
            utw_im,
            &mut output[f * out_bins..(f + 1) * out_bins],
            nc,
        );
    }
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
//     out[oy,ox, 0..C] += in[iy,ix, 0..C] * filt[ky,kx, 0..C]
//
// The reference version loops channels *outside* the kernel taps, so every
// single multiply-add re-derives two 4D indices (five integer multiplies) and
// re-evaluates four boundary branches — measured at ~71 cycles per MAC, or
// 9 MFLOP/s. Hoisting the tap bookkeeping to once per (output pixel, tap)
// leaves a pure streaming FMA over C.

/// `acc[i] += a[i] * b[i]` over `n` contiguous floats.
///
/// 4x unrolled: the per-iteration pointer bumps and branch cost as much as the
/// arithmetic otherwise. All three pointers must be 16-byte aligned.
#[cfg(target_os = "psp")]
#[inline(never)]
fn vfma_inplace(acc: &mut [f32], a: &[f32], b: &[f32], n: usize) {
    let blocks = n / 16;
    if blocks > 0 {
        unsafe {
            vfpu_asm!(
                "2:",
                "lv.q R000,  0({a})",
                "lv.q R001, 16({a})",
                "lv.q R002, 32({a})",
                "lv.q R003, 48({a})",
                "lv.q R100,  0({b})",
                "lv.q R101, 16({b})",
                "lv.q R102, 32({b})",
                "lv.q R103, 48({b})",
                "lv.q R200,  0({o})",
                "lv.q R201, 16({o})",
                "lv.q R202, 32({o})",
                "lv.q R203, 48({o})",
                "vmul.q R300, R000, R100",
                "vmul.q R301, R001, R101",
                "vmul.q R302, R002, R102",
                "vmul.q R303, R003, R103",
                "vadd.q R200, R200, R300",
                "vadd.q R201, R201, R301",
                "vadd.q R202, R202, R302",
                "vadd.q R203, R203, R303",
                "sv.q R200,  0({o})",
                "sv.q R201, 16({o})",
                "sv.q R202, 32({o})",
                "sv.q R203, 48({o})",
                "addiu {a}, {a}, 64",
                "addiu {b}, {b}, 64",
                "addiu {n}, {n}, -1",
                "bnez {n}, 2b",
                "addiu {o}, {o}, 64", // branch delay slot
                a = inout(reg) (a.as_ptr()) => _,
                b = inout(reg) (b.as_ptr()) => _,
                o = inout(reg) (acc.as_mut_ptr()) => _,
                n = inout(reg) (blocks) => _,
                options(nostack),
            );
        }
    }
    for i in blocks * 16..n {
        acc[i] += a[i] * b[i];
    }
}

#[cfg(not(target_os = "psp"))]
fn vfma_inplace(acc: &mut [f32], a: &[f32], b: &[f32], n: usize) {
    for i in 0..n {
        acc[i] += a[i] * b[i];
    }
}

/// Depthwise 2D convolution (NHWC), depth_multiplier = 1.
///
/// - `input`:  [N, H, W, C]
/// - `filter`: [1, Kh, Kw, C]
/// - `bias`:   [C]
/// - `padding`: [pad_top, pad_bottom, pad_left, pad_right]
/// - `output`: [N, Ho, Wo, C]
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
) {
    let [n, h, w, c] = input_shape;
    let [_, kh, kw, _] = filter_shape;
    let [_, ho, wo, _] = output_shape;
    let [sh, sw] = stride;
    let [pad_top, _pad_bottom, pad_left, _pad_right] = padding;

    // Tap bases for one output pixel. 3x3 in practice; the bound keeps this
    // stack-only and any larger kernel falls back to the scalar path.
    const MAX_TAPS: usize = 64;
    if kh * kw > MAX_TAPS {
        depthwise_conv2d_ref(
            input, input_shape, filter, filter_shape, bias, stride, padding, output, output_shape,
        );
        return;
    }
    let mut tap_in = [0usize; MAX_TAPS];
    let mut tap_f = [0usize; MAX_TAPS];

    // Channels outermost, in chunks, so the filter slice for a chunk stays
    // resident across the pixel sweep. Swept on hardware over {64, 128, 256}:
    // 128 wins, but only by ~7% — this kernel is bound by per-element issue
    // cost in `vfma_inplace`, not by cache locality.
    const CHUNK: usize = 128;

    let mut c0 = 0;
    while c0 < c {
        let chunk = if c - c0 < CHUNK { c - c0 } else { CHUNK };
        for batch in 0..n {
            for oy in 0..ho {
                for ox in 0..wo {
                    let ntaps = collect_taps(
                        &mut tap_in, &mut tap_f, h, w, c, kh, kw, sh, sw, pad_top, pad_left,
                        batch, oy, ox,
                    );
                    let out_base = batch * (ho * wo * c) + oy * (wo * c) + ox * c + c0;
                    let out_row = &mut output[out_base..out_base + chunk];
                    match bias {
                        Some(b) => out_row.copy_from_slice(&b[c0..c0 + chunk]),
                        None => {
                            for v in out_row.iter_mut() {
                                *v = 0.0;
                            }
                        }
                    }
                    for t in 0..ntaps {
                        let a = &input[tap_in[t] + c0..tap_in[t] + c0 + chunk];
                        let f = &filter[tap_f[t] + c0..tap_f[t] + c0 + chunk];
                        vfma_inplace(out_row, a, f, chunk);
                    }
                }
            }
        }
        c0 += CHUNK;
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
                        &mut tap_in, &mut tap_f, h, w, c, kh, kw, sh, sw, pad_top, pad_left,
                        batch, oy, ox,
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
                        &mut tap_in, &mut tap_f, h, w, c, kh, kw, sh, sw, pad_top, pad_left,
                        batch, oy, ox,
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
    fn check(h: usize, w: usize, c: usize, kh: usize, kw: usize, sh: usize, sw: usize, pad: [usize; 4], with_bias: bool) {
        let ho = (h + pad[0] + pad[1] - kh) / sh + 1;
        let wo = (w + pad[2] + pad[3] - kw) / sw + 1;
        let input = fill(h * w * c, 11);
        let filter = fill(kh * kw * c, 22);
        let bias = fill(c, 33);
        let b = with_bias.then_some(&bias[..]);

        let mut got = vec![f32::NAN; ho * wo * c];
        let mut want = vec![f32::NAN; ho * wo * c];
        depthwise_conv2d(&input, [1, h, w, c], &filter, [1, kh, kw, c], b, [sh, sw], pad, &mut got, [1, ho, wo, c]);
        depthwise_conv2d_ref(&input, [1, h, w, c], &filter, [1, kh, kw, c], b, [sh, sw], pad, &mut want, [1, ho, wo, c]);

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
        // Exercises the scalar tail of vfma_inplace.
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

    fn check(h: usize, w: usize, c: usize, kh: usize, kw: usize, sh: usize, sw: usize, pad: [usize; 4]) {
        let ho = (h + pad[0] + pad[1] - kh) / sh + 1;
        let wo = (w + pad[2] + pad[3] - kw) / sw + 1;
        let input = fill(h * w * c, 55);

        for is_max in [true, false] {
            let want = pool_ref(&input, (h, w, c), (kh, kw), (sh, sw), pad, (ho, wo), is_max);
            let mut got = vec![f32::NAN; ho * wo * c];
            if is_max {
                max_pool2d(&input, [1, h, w, c], [kh, kw], [sh, sw], pad, &mut got, [1, ho, wo, c]);
            } else {
                average_pool2d(&input, [1, h, w, c], [kh, kw], [sh, sw], pad, &mut got, [1, ho, wo, c]);
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
            let rms = (want.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
                / bins as f64)
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
            assert_eq!(off, stage_tw_len(n), "n={n}: block sizes must tile the buffer");
        }
    }

    #[test]
    fn small_sizes() {
        for n in [8, 16, 32, 64, 128] {
            check(n, 3);
        }
    }
}
