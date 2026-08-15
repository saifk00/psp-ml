//! Kernel correctness checks, shared by the host test suite and the on-device
//! runner.
//!
//! Each check is a pure predicate over the kernels, so the same list runs two
//! ways: `cargo test -p psp-rt` exercises the scalar fallbacks on the host,
//! and `cargo test -p device-tests` runs the identical list on hardware
//! against the real VFPU assembly. Keeping one source of truth is the point —
//! a check that only exists in the device binary never runs in CI, and one
//! that only exists as a `#[test]` never sees the assembly it is meant to
//! validate. See `crate::device_test` for the registry.

use crate::device_test::device_checks;
use crate::kernels;
use crate::kernels::naive;

const EPS: f32 = 1e-4;

/// 16-byte-aligned storage for a check's buffers.
///
/// A plain `[f32; N]` local gets whatever alignment the frame layout happens
/// to hand it, and that is not good enough for the kernels that load with
/// `lv.q`. `matmul_bt_tiled` and `gemm_bt_packed` require alignment outright —
/// only as a `debug_assert!`, so a release device build gets a **CPU fault**
/// instead, which is not a Rust panic, does not unwind, and locks psplink up
/// until the PSP is power-cycled. Others (`pow_const`, `swish`, `logistic`)
/// silently take their scalar fallback when unaligned, so an unaligned check
/// passes without ever running the instructions it exists to validate.
///
/// Either way the host suite is blind to it: alignment only matters on device.
///
/// Derefs to a slice, so a check body reads as ordinary array code and only
/// the declaration marks the guarantee. Don't copy the array back out — a
/// `let x = aligned.0;` hands the copy whatever alignment its own slot has.
#[repr(align(16))]
struct Aligned<const N: usize>([f32; N]);

impl<const N: usize> core::ops::Deref for Aligned<N> {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> core::ops::DerefMut for Aligned<N> {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.0
    }
}

fn approx_eq(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        let diff = if a[i] > b[i] { a[i] - b[i] } else { b[i] - a[i] };
        if diff > EPS {
            return false;
        }
    }
    true
}

// ============================================================================

fn test_relu() -> bool {
    let mut data = [-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5];
    let expected = [0.0f32, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.5];
    kernels::relu(&mut data);
    approx_eq(&data, &expected)
}

fn test_bias_add() -> bool {
    // [3, 4] matrix + [4] bias
    let mut data = [
        1.0f32, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ];
    let bias = [0.1f32, 0.2, 0.3, 0.4];
    let expected = [
        1.1f32, 2.2, 3.3, 4.4,
        5.1, 6.2, 7.3, 8.4,
        9.1, 10.2, 11.3, 12.4,
    ];
    kernels::bias_add(&mut data, &bias, 3, 4);
    approx_eq(&data, &expected)
}

fn test_matmul_bt_identity() -> bool {
    // A[4,4] @ I[4,4]^T = A
    let a = [
        1.0f32, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let b = [
        1.0f32, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let mut c = [0.0f32; 16];
    kernels::matmul_bt(&a, &b, &mut c, 4, 4, 4);
    approx_eq(&c, &a)
}

fn test_matmul_bt_known() -> bool {
    // A[4,8] @ B[4,8]^T = C[4,4]
    // Use simple values so we can verify by hand
    // A: rows of [1,0,0,0, 0,0,0,0], [0,1,0,0, 0,0,0,0], etc.
    // B: rows of [1,1,0,0, 0,0,0,0], [0,0,1,1, 0,0,0,0], etc.
    // C[i,j] = sum_k A[i,k] * B[j,k]
    let a = [
        1.0f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0,
    ];
    let b = [
        1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    // C[0,0] = 1*1 + 2*1 = 3
    // C[0,1] = 0
    // C[0,2] = 1*1 = 1
    // C[0,3] = 2*1 = 2
    // C[1,0] = 3*1 + 4*1 = 7
    // C[1,1] = 0
    // C[1,2] = 0
    // C[1,3] = 0
    // C[2,0] = 0
    // C[2,1] = 5*1 + 6*1 = 11
    // C[2,2] = 0
    // C[2,3] = 0
    // C[3,0] = 0
    // C[3,1] = 7*1 + 8*1 = 15
    // C[3,2] = 8*1 = 8
    // C[3,3] = 7*1 = 7
    let expected = [
        3.0f32, 0.0, 1.0, 2.0,
        7.0, 0.0, 0.0, 0.0,
        0.0, 11.0, 0.0, 0.0,
        0.0, 15.0, 8.0, 7.0,
    ];
    let mut c = [0.0f32; 16];
    kernels::matmul_bt(&a, &b, &mut c, 4, 8, 4);
    approx_eq(&c, &expected)
}

fn test_matmul_bt_non_aligned() -> bool {
    // A[5,6] @ B[3,6]^T = C[5,3]
    // Non-aligned dims test boundary tile handling
    // Use a simple reference: compute expected with naive triple loop
    let a: [f32; 30] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    ];
    let b: [f32; 18] = [
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    ];
    // Compute reference: C[i,j] = sum_k A[i,k] * B[j,k]
    let mut expected = [0.0f32; 15];
    for i in 0..5 {
        for j in 0..3 {
            let mut sum = 0.0f32;
            for k in 0..6 {
                sum += a[i * 6 + k] * b[j * 6 + k];
            }
            expected[i * 3 + j] = sum;
        }
    }
    let mut c = [0.0f32; 15];
    kernels::matmul_bt(&a, &b, &mut c, 5, 6, 3);
    approx_eq(&c, &expected)
}

fn test_matmul_bt_tiled() -> bool {
    // Pre-padded: A[8,8] @ B[4,8]^T = C[8,4]
    // m_tiles=2, k_tiles=2, n_tiles=1
    // Fill with sequential values, compare against matmul_bt
    let mut a = Aligned([0.0f32; 64]);
    for i in 0..64 { a[i] = (i as f32) * 0.1; }
    let mut b = Aligned([0.0f32; 32]);
    for i in 0..32 { b[i] = (i as f32) * 0.05 + 0.01; }

    let mut c_ref = [0.0f32; 32];
    kernels::matmul_bt(&a, &b, &mut c_ref, 8, 8, 4);

    let mut c_tiled = Aligned([0.0f32; 32]);
    kernels::matmul_bt_tiled(&a, &b, &mut c_tiled, 2, 2, 1);

    approx_eq(&c_tiled, &c_ref)
}

fn test_matmul_bt_tiled_large() -> bool {
    // A[16,20] @ B[8,20]^T = C[16,8]
    // m_tiles=4, k_tiles=5, n_tiles=2
    let mut a = Aligned([0.0f32; 320]);
    for i in 0..320 { a[i] = ((i % 17) as f32) * 0.1 - 0.8; }
    let mut b = Aligned([0.0f32; 160]);
    for i in 0..160 { b[i] = ((i % 13) as f32) * 0.07 - 0.4; }

    let mut c_ref = [0.0f32; 128];
    kernels::matmul_bt(&a, &b, &mut c_ref, 16, 20, 8);

    let mut c_tiled = Aligned([0.0f32; 128]);
    kernels::matmul_bt_tiled(&a, &b, &mut c_tiled, 4, 5, 2);

    approx_eq(&c_tiled, &c_ref)
}

fn test_im2col_simple() -> bool {
    // [1,4,4,1] input, [3,3] kernel, no padding, stride 1
    // Output: [1*2*2, 9] = [4, 9]
    let input: [f32; 16] = [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let mut col = [0.0f32; 36]; // 4 rows × 9 cols
    kernels::im2col(
        &input,
        [1, 4, 4, 1],
        [3, 3],
        [1, 1],
        [0, 0],
        [2, 2],
        &mut col,
    );
    // Row 0 (oy=0, ox=0): patch at top-left 3x3
    let expected_row0 = [1.0f32, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0];
    // Row 1 (oy=0, ox=1): patch shifted right by 1
    let expected_row1 = [2.0f32, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0];

    approx_eq(&col[0..9], &expected_row0) && approx_eq(&col[9..18], &expected_row1)
}

fn test_im2col_with_padding() -> bool {
    // [1,4,4,1] input, [3,3] kernel, padding [1,1], stride 1
    // Output: [1*4*4, 9] = [16, 9]
    let input: [f32; 16] = [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let mut col = [0.0f32; 144]; // 16 rows × 9 cols
    kernels::im2col(
        &input,
        [1, 4, 4, 1],
        [3, 3],
        [1, 1],
        [1, 1],
        [4, 4],
        &mut col,
    );
    // Row 0 (oy=0, ox=0): top-left corner with padding
    // ky=0: iy=-1 (pad), all zeros
    // ky=1: iy=0, kx=0: ix=-1 (pad)=0, kx=1: ix=0 -> input[0]=1.0, kx=2: ix=1 -> input[1]=2.0
    // ky=2: iy=1, kx=0: ix=-1 (pad)=0, kx=1: ix=0 -> input[4]=5.0, kx=2: ix=1 -> input[5]=6.0
    let expected_row0 = [0.0f32, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 5.0, 6.0];
    approx_eq(&col[0..9], &expected_row0)
}

fn test_im2col_padded_vs_im2col() -> bool {
    // [1,4,4,2] input, [3,3] kernel, padding [1,1], stride 1
    // K = 3*3*2 = 18, K_padded = 20
    // im2col output: [16, 18]
    // im2col_padded output: [16, 20]
    // First 18 cols should match, cols 18-19 should be zero
    let mut input = [0.0f32; 32];
    for i in 0..32 { input[i] = (i as f32) * 0.1 + 0.05; }

    let mut col_ref = [0.0f32; 288]; // 16 * 18
    kernels::im2col(
        &input,
        [1, 4, 4, 2],
        [3, 3],
        [1, 1],
        [1, 1],
        [4, 4],
        &mut col_ref,
    );

    let mut col_padded = [0.0f32; 320]; // 16 * 20
    kernels::im2col_padded(
        &input,
        [1, 4, 4, 2],
        [3, 3],
        [1, 1],
        [1, 1, 1, 1],
        [4, 4],
        &mut col_padded,
    );

    // Compare each row: first 18 elements should match, last 2 should be zero
    for row in 0..16 {
        let ref_row = &col_ref[row * 18..(row + 1) * 18];
        let pad_row = &col_padded[row * 20..row * 20 + 18];
        if !approx_eq(ref_row, pad_row) {
            return false;
        }
        // Check padding columns are zero
        if col_padded[row * 20 + 18] != 0.0 || col_padded[row * 20 + 19] != 0.0 {
            return false;
        }
    }
    true
}

fn test_conv2d_via_im2col_vs_naive() -> bool {
    // Small conv2d: [1,6,6,2] input, [4,3,3,2] filters
    // stride [1,1], SAME padding
    // Compare full VFPU pipeline (im2col_padded + matmul_bt_tiled + bias_add + relu)
    // against naive::conv2d_relu

    let ci = 2usize;
    let co = 4usize;
    let h = 6usize;
    let w = 6usize;
    let kh = 3usize;
    let kw = 3usize;

    // SAME padding for stride 1: pad = (k - 1) / 2
    let pad_h = (kh - 1) / 2; // = 1
    let pad_w = (kw - 1) / 2; // = 1
    let ho = h; // stride 1, SAME → output same as input
    let wo = w;

    // Generate deterministic input
    let mut input = [0.0f32; 72]; // 1*6*6*2
    for i in 0..72 { input[i] = ((i % 11) as f32) * 0.1 - 0.5; }

    // Generate deterministic filters: [Co, Kh, Kw, Ci] = [4, 3, 3, 2]
    let mut filter = [0.0f32; 72]; // 4*3*3*2
    for i in 0..72 { filter[i] = ((i % 7) as f32) * 0.05 - 0.15; }

    let bias = [0.1f32, -0.2, 0.05, 0.3];

    // --- Naive path ---
    let mut output_naive = [0.0f32; 144]; // 1*6*6*4
    naive::conv2d_relu(
        &input,
        [1, h, w, ci],
        &filter,
        [co, kh, kw, ci],
        Some(&bias),
        [1, 1],
        [pad_h, pad_w, pad_h, pad_w],
        &mut output_naive,
        [1, ho, wo, co],
    );

    // --- VFPU pipeline ---
    // Step 1: im2col_padded
    let gemm_k = kh * kw * ci; // = 18
    let k_padded = ((gemm_k + 3) / 4) * 4; // = 20
    let gemm_m = ho * wo; // = 36
    let m_padded = ((gemm_m + 3) / 4) * 4; // = 36 (already aligned)
    let n_padded = ((co + 3) / 4) * 4; // = 4 (already aligned)

    let mut im2col_buf = Aligned([0.0f32; 720]); // 36 * 20
    kernels::im2col_padded(
        &input,
        [1, h, w, ci],
        [kh, kw],
        [1, 1],
        [pad_h, pad_w, pad_h, pad_w],
        [ho, wo],
        &mut im2col_buf,
    );

    // Step 2: Pad weights [Co, K] → [Co, K_padded]
    // Filter is [Co, Kh, Kw, Ci] = [4, 18] row-major
    let mut weights_padded = Aligned([0.0f32; 80]); // 4 * 20
    for row in 0..co {
        for col in 0..gemm_k {
            weights_padded[row * k_padded + col] = filter[row * gemm_k + col];
        }
    }

    // Step 3: matmul_bt_tiled
    let m_tiles = m_padded / 4;
    let k_tiles = k_padded / 4;
    let n_tiles = n_padded / 4;
    let mut output_vfpu = Aligned([0.0f32; 144]); // 36 * 4
    kernels::matmul_bt_tiled(
        &im2col_buf,
        &weights_padded,
        &mut output_vfpu,
        m_tiles,
        k_tiles,
        n_tiles,
    );

    // Step 4: bias_add + relu
    kernels::bias_add(&mut output_vfpu, &bias, gemm_m, co);
    kernels::relu(&mut output_vfpu);

    approx_eq(&output_vfpu, &output_naive)
}

/// Explicit zero-pad + VALID conv must equal the kernel's own padding.
///
/// `ir::fuse` rewrites `Pad`+`Conv` into a single padded conv, so the two
/// formulations have to agree *exactly* — adding a zero tap is exact in IEEE
/// 754, so this is a bit-equality check, not an approximate one.
///
/// Shaped like BirdNET's 4th depthwise ([1,6,16,C] stride 2, 3x3). The channel
/// counts matter: `depthwise_conv2d` hands `vfma_inplace` a chunk of `C`, and
/// that takes the `lv.q` path only for `chunk >= 16` (`blocks = n / 16`). C=4
/// exercises the scalar tail only, C=16 the vector path only, C=20 both.
fn test_depthwise_padding_matches_explicit_pad() -> bool {
    const H: usize = 6;
    const W: usize = 16;
    const K: usize = 3;
    const S: usize = 2;
    const HO: usize = (H + 2 - K) / S + 1;
    const WO: usize = (W + 2 - K) / S + 1;
    const CMAX: usize = 20;

    // `depthwise_conv2d` reaches `vfma_inplace`, which uses `lv.q` unguarded
    // once a chunk is 16 wide. Every slice it takes starts at a multiple of `c`
    // floats, so aligning these five bases aligns all of them.
    let mut inp = Aligned([0.0f32; H * W * CMAX]);
    let mut filt = Aligned([0.0f32; K * K * CMAX]);
    let mut padded = Aligned([0.0f32; (H + 2) * (W + 2) * CMAX]);
    let mut a = Aligned([0.0f32; HO * WO * CMAX]);
    let mut b = Aligned([0.0f32; HO * WO * CMAX]);

    for c in [4usize, 16, 20] {
        for (i, v) in inp[..H * W * c].iter_mut().enumerate() {
            *v = ((i * 37 % 101) as f32 / 50.0) - 1.0;
        }
        for (i, v) in filt[..K * K * c].iter_mut().enumerate() {
            *v = ((i * 17 % 61) as f32 / 30.0) - 1.0;
        }

        naive::pad(
            &inp[..H * W * c], [1, H, W, c],
            &mut padded[..(H + 2) * (W + 2) * c], [1, H + 2, W + 2, c],
            [[0, 0], [1, 1], [1, 1], [0, 0]],
        );
        kernels::depthwise_conv2d(
            &padded[..(H + 2) * (W + 2) * c], [1, H + 2, W + 2, c],
            &filt[..K * K * c], [1, K, K, c],
            None, [S, S], [0, 0, 0, 0],
            &mut a[..HO * WO * c], [1, HO, WO, c],
        );
        kernels::depthwise_conv2d(
            &inp[..H * W * c], [1, H, W, c],
            &filt[..K * K * c], [1, K, K, c],
            None, [S, S], [1, 1, 1, 1],
            &mut b[..HO * WO * c], [1, HO, WO, c],
        );

        if a[..HO * WO * c] != b[..HO * WO * c] {
            return false;
        }
    }
    true
}

/// `pow_const` must agree with `libm::powf`, and must *not* silently produce a
/// plausible number for a negative base.
///
/// `ir::fuse` rewrites `Pow` with a scalar exponent into `PowConst` on the
/// argument that `vlog2` of a negative is NaN, so a non-integer exponent over a
/// possibly-negative base diverges loudly rather than quietly — `libm::powf` is
/// NaN there too. That is a claim about the hardware, and the host mirror of
/// `pow_const` uses `libm::powf`, so only the device runner can actually test
/// it. If this fails on device, the fusion guard in `fuse_pow_const` is unsound.
fn test_pow_const_matches_libm() -> bool {
    const N: usize = 32;
    const C: f32 = 0.2199; // BirdNET's spectrogram compression exponent

    // `pow_const` falls back to the scalar `libm` path unless both buffers are
    // 16-byte aligned, which would make this test pass on device without ever
    // running the instruction it is here to check — so both stay in `Aligned`
    // for the whole check. Copying one out into a bare `[f32; N]` local is
    // enough to lose the alignment and the coverage with it.
    // Non-negative inputs, including the 0 that a squared base reaches.
    let mut inp = Aligned([0.0f32; N]);
    for i in 0..N {
        inp[i] = (i as f32) * 0.37;
    }
    let mut got = Aligned([0.0f32; N]);
    kernels::pow_const(&inp, &mut got, C);
    for i in 0..N {
        let want = libm::powf(inp[i], C);
        let d = if got[i] > want { got[i] - want } else { want - got[i] };
        // vlog2/vexp2 are hardware approximations; scale the bound by the value.
        let tol = 1e-3 * if want > 1.0 { want } else { 1.0 };
        if !(d <= tol) {
            return false;
        }
    }

    // A negative base with a non-integer exponent must be NaN, not a finite
    // value derived from log2(|x|).
    let neg = Aligned([-1.0f32, -2.5, -8.0, -0.5]);
    let mut out = Aligned([0.0f32; 4]);
    kernels::pow_const(&neg, &mut out, C);
    for v in out.iter() {
        if !v.is_nan() {
            return false;
        }
    }
    true
}

device_checks! {
    // Every kernel has a scalar fallback in the same signature, so the whole
    // suite is meaningful on both runners.
    shared: [
        test_relu,
        test_bias_add,
        test_matmul_bt_identity,
        test_matmul_bt_known,
        test_matmul_bt_non_aligned,
        test_matmul_bt_tiled,
        test_matmul_bt_tiled_large,
        test_im2col_simple,
        test_im2col_with_padding,
        test_im2col_padded_vs_im2col,
        test_conv2d_via_im2col_vs_naive,
        test_depthwise_padding_matches_explicit_pad,
        test_pow_const_matches_libm,
    ],
    device: [],
}

