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
        let diff = if a[i] > b[i] {
            a[i] - b[i]
        } else {
            b[i] - a[i]
        };
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
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let bias = [0.1f32, 0.2, 0.3, 0.4];
    let expected = [
        1.1f32, 2.2, 3.3, 4.4, 5.1, 6.2, 7.3, 8.4, 9.1, 10.2, 11.3, 12.4,
    ];
    kernels::bias_add(&mut data, &bias, 3, 4);
    approx_eq(&data, &expected)
}

fn test_matmul_bt_identity() -> bool {
    // A[4,4] @ I[4,4]^T = A
    let a = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let b = [
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
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
        1.0f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0,
    ];
    let b = [
        1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
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
        3.0f32, 0.0, 1.0, 2.0, 7.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 15.0, 8.0, 7.0,
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
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.5, 1.5, 2.5, 3.5, 4.5,
        5.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    ];
    let b: [f32; 18] = [
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
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
    for i in 0..64 {
        a[i] = (i as f32) * 0.1;
    }
    let mut b = Aligned([0.0f32; 32]);
    for i in 0..32 {
        b[i] = (i as f32) * 0.05 + 0.01;
    }

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
    for i in 0..320 {
        a[i] = ((i % 17) as f32) * 0.1 - 0.8;
    }
    let mut b = Aligned([0.0f32; 160]);
    for i in 0..160 {
        b[i] = ((i % 13) as f32) * 0.07 - 0.4;
    }

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
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
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
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
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
    for i in 0..32 {
        input[i] = (i as f32) * 0.1 + 0.05;
    }

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
    for i in 0..72 {
        input[i] = ((i % 11) as f32) * 0.1 - 0.5;
    }

    // Generate deterministic filters: [Co, Kh, Kw, Ci] = [4, 3, 3, 2]
    let mut filter = [0.0f32; 72]; // 4*3*3*2
    for i in 0..72 {
        filter[i] = ((i % 7) as f32) * 0.05 - 0.15;
    }

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
/// formulations have to agree. Adding a zero tap is exact in IEEE 754, but
/// the two runs don't take the same kernel at the border: the padded input
/// has full 3x3 windows everywhere (the register-resident 8-lane kernel,
/// which sums on two accumulator chains), while the kernel-padded run gives
/// its partial edge windows to the generic 16/4-lane kernel (one chain). So
/// the border pixels can differ in the last bit — hence a 1e-5 relative
/// tolerance rather than bit equality.
///
/// Shaped like BirdNET's 4th depthwise ([1,6,16,C] stride 2, 3x3). The channel
/// counts matter: C=4 exercises the quad path only, C=16 the 8-lane and
/// 16-lane paths, C=20 those plus the 4-lane remainder.
fn test_depthwise_padding_matches_explicit_pad() -> bool {
    const H: usize = 6;
    const W: usize = 16;
    const K: usize = 3;
    const S: usize = 2;
    const HO: usize = (H + 2 - K) / S + 1;
    const WO: usize = (W + 2 - K) / S + 1;
    const CMAX: usize = 20;

    // `depthwise_conv2d` only takes its VFPU path on 16-byte-aligned tensors
    // with `c % 4 == 0`; these five bases make every slice it takes aligned.
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
            &inp[..H * W * c],
            [1, H, W, c],
            &mut padded[..(H + 2) * (W + 2) * c],
            [1, H + 2, W + 2, c],
            [[0, 0], [1, 1], [1, 1], [0, 0]],
        );
        kernels::depthwise_conv2d(
            &padded[..(H + 2) * (W + 2) * c],
            [1, H + 2, W + 2, c],
            &filt[..K * K * c],
            [1, K, K, c],
            None,
            [S, S],
            [0, 0, 0, 0],
            &mut a[..HO * WO * c],
            [1, HO, WO, c],
            kernels::Epilogue::None,
        );
        kernels::depthwise_conv2d(
            &inp[..H * W * c],
            [1, H, W, c],
            &filt[..K * K * c],
            [1, K, K, c],
            None,
            [S, S],
            [1, 1, 1, 1],
            &mut b[..HO * WO * c],
            [1, HO, WO, c],
            kernels::Epilogue::None,
        );

        if !approx_eq_tol(&a[..HO * WO * c], &b[..HO * WO * c], 1e-5) {
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
        let d = if got[i] > want {
            got[i] - want
        } else {
            want - got[i]
        };
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

fn test_rfft_strided_matches_dense() -> bool {
    // Windows overlap (hop < n), so the strided kernel reads each sample
    // through several frames — on device the butterfly stages and stage-0
    // `vbfy1.q` run for real, which the host mirror never exercises. The
    // inner stride (the small-FFT frontend's decimated read) is covered by
    // running the same comparison at stride 2.
    test_rfft_strided_at(1) && test_rfft_strided_at(2)
}

fn test_rfft_strided_at(in_stride: usize) -> bool {
    const N: usize = 32;
    const NC: usize = N / 2;
    const HOP: usize = 9;
    const FRAMES: usize = 4;
    const BINS: usize = NC + 1;
    const N_SAMPLES: usize = (FRAMES - 1) * HOP + 2 * (N - 1) + 1;

    // Twiddles in the split layout psp-tc emits (see `lower_rfft`). Both the
    // stage runs and the scratch are loaded with `lv.q`, hence `Aligned`.
    let mut stage = Aligned([0.0f32; kernels::stage_tw_len(N)]);
    let stages = NC.trailing_zeros() as usize;
    let mut off = 0usize;
    for s in 0..stages {
        let half = 1usize << s;
        for j in 0..half {
            let a = -2.0 * core::f64::consts::PI * j as f64 / (2.0 * half as f64);
            stage[off + j] = libm::cos(a) as f32;
            stage[off + half + j] = libm::sin(a) as f32;
        }
        off += kernels::stage_tw_block(s);
    }
    let mut unpack = Aligned([0.0f32; (NC - 1) * 2]);
    for k in 1..NC {
        let a = 2.0 * core::f64::consts::PI * k as f64 / N as f64;
        unpack[k - 1] = libm::cos(a) as f32;
        unpack[NC - 1 + k - 1] = -(libm::sin(a) as f32);
    }

    let mut samples = Aligned([0.0f32; N_SAMPLES]);
    for i in 0..N_SAMPLES {
        samples[i] = libm::sinf(i as f32 * 0.7) + 0.25 * (i as f32 % 5.0);
    }
    let mut window = Aligned([0.0f32; N]);
    for j in 0..N {
        window[j] = 0.5 - 0.5 * libm::cosf(2.0 * core::f32::consts::PI * j as f32 / N as f32);
    }

    // Dense reference: materialise every window, multiply, rfft_batch.
    let mut dense = Aligned([0.0f32; FRAMES * N]);
    for f in 0..FRAMES {
        for j in 0..N {
            dense[f * N + j] = samples[f * HOP + in_stride * j] * window[j];
        }
    }
    let mut scratch = Aligned([7.5f32; N]);
    let mut want = Aligned([0.0f32; FRAMES * BINS]);
    kernels::rfft_batch(&dense, &stage, &unpack, &mut scratch, &mut want, N, FRAMES);

    let mut scratch = Aligned([-3.25f32; N]);
    let mut got = Aligned([0.0f32; FRAMES * BINS]);
    kernels::rfft_strided_batch(
        &samples,
        Some(&window),
        &stage,
        &unpack,
        &mut scratch,
        &mut got,
        N,
        HOP,
        in_stride,
        FRAMES,
    );

    for i in 0..FRAMES * BINS {
        if got[i].to_bits() != want[i].to_bits() {
            return false;
        }
    }
    true
}

fn test_square_pow_matches_mul_then_pow() -> bool {
    // On device the fused kernel is pow_const's vlog2/vexp2 pipeline with one
    // extra vmul.q; squaring via scalar mul first must land on the same bits
    // (an IEEE f32 multiply is exact in either engine). Aligned buffers so the
    // VFPU path actually runs — unaligned would silently take the scalar
    // fallback and prove nothing.
    const N: usize = 32;
    const C: f32 = 0.2199;
    let mut inp = Aligned([0.0f32; N]);
    for i in 0..N {
        inp[i] = (i as f32 - 13.0) * 0.31; // negatives included: x^2 fixes them
    }
    let mut squared = Aligned([0.0f32; N]);
    for i in 0..N {
        squared[i] = inp[i] * inp[i];
    }
    let mut want = Aligned([0.0f32; N]);
    kernels::pow_const(&squared, &mut want, C);
    let mut got = Aligned([f32::NAN; N]);
    kernels::square_pow(&inp, &mut got, C);
    for i in 0..N {
        if got[i].to_bits() != want[i].to_bits() {
            return false;
        }
    }
    true
}

fn test_fc_cb_matches_dense() -> bool {
    // On device this exercises the real VFPU path — 4-row groups through the
    // vtfm4 GEMV tiles (11 rows = 2 groups + a 3-row scalar tail), a 17-long
    // band (two coefficient chunks), and a band whose padded window runs past
    // the row end. The VFPU chunk-tree summation reorders the adds, so the
    // check is tolerance-based, not bitwise; the output is [OUT, ROWS]
    // (transposed relative to the dense matmul).
    const ROWS: usize = 11;
    const IN: usize = 40;
    const OUT: usize = 6;
    let mut input = Aligned([0.0f32; ROWS * IN]);
    for i in 0..ROWS * IN {
        input[i] = libm::sinf(i as f32 * 0.9);
    }
    let band_meta: [i32; OUT * 2] = [0, 3, 2, 1, 5, 17, 10, 2, 20, 4, 36, 4];
    let mut band_data = Aligned([0.0f32; 3 + 1 + 17 + 2 + 4 + 4]);
    for (i, v) in band_data.iter_mut().enumerate() {
        *v = 0.1 + i as f32 * 0.07;
    }

    let mut dense = Aligned([0.0f32; OUT * IN]);
    let mut off = 0usize;
    for b in 0..OUT {
        let (start, len) = (band_meta[2 * b] as usize, band_meta[2 * b + 1] as usize);
        for k in 0..len {
            dense[b * IN + start + k] = band_data[off + k];
        }
        off += len;
    }

    let mut want = Aligned([0.0f32; ROWS * OUT]);
    for m in 0..ROWS {
        naive::fully_connected(
            &input[m * IN..(m + 1) * IN],
            IN,
            &dense,
            None,
            &mut want[m * OUT..(m + 1) * OUT],
            OUT,
        );
    }
    let mut got = Aligned([f32::NAN; ROWS * OUT]);
    kernels::fully_connected_cb(&input, ROWS, IN, &band_meta, &band_data, &mut got, OUT);
    for m in 0..ROWS {
        for b in 0..OUT {
            let (g, w) = (got[b * ROWS + m], want[m * OUT + b]);
            let d = if g > w { g - w } else { w - g };
            if !(d <= EPS) {
                return false;
            }
        }
    }
    true
}

/// The fact `fully_connected_cb`'s asm rests on, measured on hardware:
/// `vtfm4.q rd, Mxxx, rt` dots the register matrix's *columns* with the
/// vector (the same implicit transpose as `vmmul.q`'s first operand), so a
/// matrix loaded row-wise with `lv.q R000..R003` needs the E-form --
/// `vtfm4.q rd, Exxx, rt` -- to get row dots. If a rust-psp upgrade ever
/// changes the encoding, this names the failure that
/// `test_fc_cb_matches_dense` would only report as a wrong result.
#[cfg(target_os = "psp")]
fn test_vtfm4_e_form_is_row_dots() -> bool {
    use psp::vfpu_asm;
    #[repr(align(16))]
    struct A16([f32; 16]);
    #[repr(align(16))]
    struct A4([f32; 4]);
    // A[r][j] = 4r + j -- asymmetric, distinguishes rows from columns.
    let mut a = A16([0.0; 16]);
    for r in 0..4 {
        for j in 0..4 {
            a.0[r * 4 + j] = (r * 4 + j) as f32;
        }
    }
    let v = A4([1.0, 10.0, 100.0, 1000.0]);
    let mut out = A4([0.0; 4]);
    unsafe {
        vfpu_asm!(
            "lv.q R000, 0({a})",
            "lv.q R001, 16({a})",
            "lv.q R002, 32({a})",
            "lv.q R003, 48({a})",
            "lv.q R400, 0({v})",
            "vtfm4.q R500, E000, R400",
            "sv.q R500, 0({o})",
            a = in(reg) (a.0.as_ptr()),
            v = in(reg) (v.0.as_ptr()),
            o = in(reg) (out.0.as_mut_ptr()),
            options(nostack),
        );
    }
    let mut want = [0.0f32; 4];
    for (r, w) in want.iter_mut().enumerate() {
        for (j, vv) in [1.0f32, 10.0, 100.0, 1000.0].iter().enumerate() {
            *w += (r * 4 + j) as f32 * vv;
        }
    }
    out.0 == want
}

fn test_fir_decimate_impulse_and_dc() -> bool {
    // Same references as the host unit test: an impulse lands the taps at
    // the decimated positions, and a constant passes with gain sum(taps).
    let mut x = Aligned([0.0f32; 64]);
    x[10] = 1.0;
    let taps = Aligned([0.25f32, 0.5, 1.0, 0.5, 0.25]);
    let mut y = Aligned([0.0f32; 32]);
    kernels::fir_decimate(&x, &taps, &mut y, 2);
    for n in 0..32usize {
        let mut want = 0.0f32;
        for (t, &h) in taps.iter().enumerate() {
            if 2 * n + t == 12 {
                want += h;
            }
        }
        if (y[n] - want).abs() > 1e-6 {
            return false;
        }
    }
    let x = Aligned([1.0f32; 64]);
    let mut y = Aligned([0.0f32; 32]);
    kernels::fir_decimate(&x, &taps, &mut y, 2);
    let dc: f32 = taps.iter().sum();
    for n in 2..30 {
        if (y[n] - dc).abs() > 1e-5 {
            return false;
        }
    }
    true
}

/// Relative/absolute tolerance for the VFPU transcendental approximations
/// (`vexp2`, `vrcp`): same bound `test_pow_const_matches_libm` uses.
fn approx_eq_tol(a: &[f32], b: &[f32], rel: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        let mag = if b[i].abs() > 1.0 { b[i].abs() } else { 1.0 };
        let d = (a[i] - b[i]).abs();
        if !(d <= rel * mag) {
            return false;
        }
    }
    true
}

fn pseudo(i: usize, salt: usize) -> f32 {
    (((i * 7919 + salt * 104729) % 1009) as f32 / 504.5) - 1.0
}

/// The fused GEMM epilogue must equal the unfused GEMM followed by scalar
/// bias + activation, on both its paths: single k-block (tile goes straight
/// from registers to C through `store_tile_4x8`) and multi k-block (slab
/// accumulation, epilogue applied on the unpack).
///
/// Shapes are chosen for partial tiles on every axis: m not a multiple of 4,
/// n not a multiple of 8 (but of 4, so rows stay quad-aligned for the VFPU
/// store), k not a multiple of 8.
fn test_gemm_fused_epilogue_matches_unfused() -> bool {
    // Direct path: k=36 -> 10 k-tiles, kc=40 covers it in one block.
    const M1: usize = 13;
    const K1: usize = 36;
    const N1: usize = 20;
    let mut a = Aligned([0.0f32; M1 * K1]);
    let mut b = [0.0f32; N1 * K1];
    let mut bias = Aligned([0.0f32; N1]);
    for (i, v) in a.iter_mut().enumerate() {
        *v = pseudo(i, 1);
    }
    for (i, v) in b.iter_mut().enumerate() {
        *v = pseudo(i, 2);
    }
    for (i, v) in bias.iter_mut().enumerate() {
        *v = pseudo(i, 3);
    }
    let mut bp = Aligned([0.0f32; 3 * 10 * 32]);
    kernels::pack_b_panel(&b, &mut bp, N1, K1);
    let mut ap = Aligned([0.0f32; 4 * 10 * 16]);
    let mut cp = Aligned([0.0f32; 4 * 3 * 32]);
    let mut plain = Aligned([0.0f32; M1 * N1]);
    kernels::gemm_bt_packed(
        &a, K1, &bp, &mut plain, &mut ap, &mut cp, M1, K1, N1, 16, 40,
    );
    for act in [
        kernels::Epilogue::None,
        kernels::Epilogue::Relu,
        kernels::Epilogue::Swish,
    ] {
        let mut want = [0.0f32; M1 * N1];
        for r in 0..M1 {
            for c in 0..N1 {
                want[r * N1 + c] = kernels::apply_epilogue(plain[r * N1 + c] + bias[c], act);
            }
        }
        let mut got = Aligned([0.0f32; M1 * N1]);
        kernels::gemm_bt_packed_fused(
            &a,
            K1,
            &bp,
            &mut got,
            &mut ap,
            &mut cp,
            M1,
            K1,
            N1,
            16,
            40,
            Some(&bias),
            act,
        );
        if !approx_eq_tol(&got, &want, 1e-3) {
            return false;
        }
    }

    // Slab path: k=300 -> 76 k-tiles over kc=64 (16 tiles per block).
    const M2: usize = 9;
    const K2: usize = 300;
    const N2: usize = 16;
    let mut a = Aligned([0.0f32; M2 * K2]);
    let mut b = [0.0f32; N2 * K2];
    for (i, v) in a.iter_mut().enumerate() {
        *v = pseudo(i, 4);
    }
    for (i, v) in b.iter_mut().enumerate() {
        *v = pseudo(i, 5);
    }
    let mut bp = Aligned([0.0f32; 2 * 76 * 32]);
    kernels::pack_b_panel(&b, &mut bp, N2, K2);
    let mut ap = Aligned([0.0f32; 2 * 16 * 16]);
    let mut cp = Aligned([0.0f32; 2 * 2 * 32]);
    let mut plain = Aligned([0.0f32; M2 * N2]);
    kernels::gemm_bt_packed(&a, K2, &bp, &mut plain, &mut ap, &mut cp, M2, K2, N2, 8, 64);
    for act in [
        kernels::Epilogue::None,
        kernels::Epilogue::Relu,
        kernels::Epilogue::Swish,
    ] {
        let mut want = [0.0f32; M2 * N2];
        for r in 0..M2 {
            for c in 0..N2 {
                want[r * N2 + c] = kernels::apply_epilogue(plain[r * N2 + c] + bias[c], act);
            }
        }
        let mut got = Aligned([0.0f32; M2 * N2]);
        kernels::gemm_bt_packed_fused(
            &a,
            K2,
            &bp,
            &mut got,
            &mut ap,
            &mut cp,
            M2,
            K2,
            N2,
            8,
            64,
            Some(&bias[..N2]),
            act,
        );
        if !approx_eq_tol(&got, &want, 1e-3) {
            return false;
        }
    }
    true
}

/// A 1x1 conv fed straight from its NHWC input must equal the im2col route:
/// `lda` = channels, no copy. Same check the compiler relies on when it
/// elides the im2col scratch.
fn test_gemm_lda_equals_im2col_for_1x1() -> bool {
    const HW: usize = 10;
    const CI: usize = 12;
    const CO: usize = 8;
    let mut x = Aligned([0.0f32; HW * CI]);
    for (i, v) in x.iter_mut().enumerate() {
        *v = pseudo(i, 6);
    }
    let mut w = [0.0f32; CO * CI];
    for (i, v) in w.iter_mut().enumerate() {
        *v = pseudo(i, 7);
    }
    let mut col = Aligned([0.0f32; HW * CI]);
    kernels::im2col_padded(
        &x,
        [1, 1, HW, CI],
        [1, 1],
        [1, 1],
        [0, 0, 0, 0],
        [1, HW],
        &mut col,
    );
    let mut bp = Aligned([0.0f32; 1 * 4 * 32]);
    kernels::pack_b_panel(&w, &mut bp, CO, CI);
    let mut ap = Aligned([0.0f32; 3 * 4 * 16]);
    let mut cp = Aligned([0.0f32; 32]);
    let mut via_col = Aligned([0.0f32; HW * CO]);
    let mut direct = Aligned([0.0f32; HW * CO]);
    kernels::gemm_bt_packed(
        &col,
        CI,
        &bp,
        &mut via_col,
        &mut ap,
        &mut cp,
        HW,
        CI,
        CO,
        12,
        16,
    );
    kernels::gemm_bt_packed(
        &x,
        CI,
        &bp,
        &mut direct,
        &mut ap,
        &mut cp,
        HW,
        CI,
        CO,
        12,
        16,
    );
    via_col.0 == direct.0
}

/// Fused depthwise (bias + activation in the store) against the reference
/// with the activation applied afterwards. C=20 covers one 16-lane group and
/// one quad group; stride 2 with padding covers edge pixels with fewer taps.
fn test_depthwise_fused_activation_matches_ref() -> bool {
    const H: usize = 5;
    const W: usize = 9;
    const C: usize = 20;
    const K: usize = 3;
    let mut inp = Aligned([0.0f32; H * W * C]);
    let mut filt = Aligned([0.0f32; K * K * C]);
    let mut bias = Aligned([0.0f32; C]);
    for (i, v) in inp.iter_mut().enumerate() {
        *v = pseudo(i, 8);
    }
    for (i, v) in filt.iter_mut().enumerate() {
        *v = pseudo(i, 9);
    }
    for (i, v) in bias.iter_mut().enumerate() {
        *v = pseudo(i, 10);
    }
    for (stride, ho, wo) in [
        (1usize, H, W),
        (2, (H + 2 - K) / 2 + 1, (W + 2 - K) / 2 + 1),
    ] {
        let mut want = Aligned([0.0f32; H * W * C]);
        kernels::depthwise_conv2d_ref(
            &inp,
            [1, H, W, C],
            &filt,
            [1, K, K, C],
            Some(&bias),
            [stride, stride],
            [1, 1, 1, 1],
            &mut want[..ho * wo * C],
            [1, ho, wo, C],
        );
        for act in [
            kernels::Epilogue::None,
            kernels::Epilogue::Relu,
            kernels::Epilogue::Swish,
        ] {
            let mut got = Aligned([0.0f32; H * W * C]);
            kernels::depthwise_conv2d(
                &inp,
                [1, H, W, C],
                &filt,
                [1, K, K, C],
                Some(&bias),
                [stride, stride],
                [1, 1, 1, 1],
                &mut got[..ho * wo * C],
                [1, ho, wo, C],
                act,
            );
            let mut want_act = [0.0f32; H * W * C];
            for i in 0..ho * wo * C {
                want_act[i] = kernels::apply_epilogue(want[i], act);
            }
            if !approx_eq_tol(&got[..ho * wo * C], &want_act[..ho * wo * C], 1e-3) {
                return false;
            }
        }
    }
    true
}

/// A 128-channel shape takes `depthwise_conv2d`'s whole-pixel chunk with a
/// narrow (4-column) strip, so interior runs, strip boundaries and the
/// single-pixel edge columns all get exercised; must match the reference
/// for every activation and both strides, with the padding rows that make
/// some pixels one- or two-row windows.
fn test_depthwise_wide_pixel_strips_match_ref() -> bool {
    extern crate alloc;
    use alloc::vec::Vec;
    const H: usize = 5;
    const W: usize = 16;
    const C: usize = 128;
    const K: usize = 3;
    // Heap buffers (160 KB would be a lot of stack on the device), trimmed
    // to a 16-byte boundary so the VFPU path is the one under test.
    fn aligned(len: usize, seed: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..len + 4).map(|i| pseudo(i, seed)).collect();
        let skip = (16 - (v.as_ptr() as usize % 16)) % 16 / 4;
        v.drain(..skip);
        v.truncate(len);
        v
    }
    let inp = aligned(H * W * C, 12);
    let filt = aligned(K * K * C, 13);
    let bias = aligned(C, 14);
    for (stride, ho, wo) in [
        (1usize, H, W),
        (2, (H + 2 - K) / 2 + 1, (W + 2 - K) / 2 + 1),
    ] {
        let mut want = aligned(ho * wo * C, 0);
        kernels::depthwise_conv2d_ref(
            &inp,
            [1, H, W, C],
            &filt,
            [1, K, K, C],
            Some(&bias),
            [stride, stride],
            [1, 1, 1, 1],
            &mut want,
            [1, ho, wo, C],
        );
        for act in [
            kernels::Epilogue::None,
            kernels::Epilogue::Relu,
            kernels::Epilogue::Swish,
        ] {
            let mut got = aligned(ho * wo * C, 0);
            kernels::depthwise_conv2d(
                &inp,
                [1, H, W, C],
                &filt,
                [1, K, K, C],
                Some(&bias),
                [stride, stride],
                [1, 1, 1, 1],
                &mut got,
                [1, ho, wo, C],
                act,
            );
            let want_act: Vec<f32> = want
                .iter()
                .map(|&v| kernels::apply_epilogue(v, act))
                .collect();
            if !approx_eq_tol(&got, &want_act, 1e-3) {
                return false;
            }
        }
    }
    true
}

/// Row-streaming `reduce_mean_hw` against the direct per-channel mean.
fn test_reduce_mean_hw_matches_direct() -> bool {
    const N: usize = 7;
    const C: usize = 12;
    let mut x = Aligned([0.0f32; N * C]);
    for (i, v) in x.iter_mut().enumerate() {
        *v = pseudo(i, 11);
    }
    let mut got = Aligned([0.0f32; C]);
    kernels::reduce_mean_hw(&x, &mut got);
    let mut want = [0.0f32; C];
    for ch in 0..C {
        let mut sum = 0.0f32;
        for i in 0..N {
            sum += x[i * C + ch];
        }
        want[ch] = sum / N as f32;
    }
    approx_eq(&got, &want)
}

/// `binary_add` / `binary_mul` on every broadcast shape: full, scalar, row.
fn test_binary_broadcast_matches_scalar() -> bool {
    const LEN: usize = 24;
    let mut a = Aligned([0.0f32; LEN]);
    let mut b = Aligned([0.0f32; LEN]);
    for (i, v) in a.iter_mut().enumerate() {
        *v = pseudo(i, 12);
    }
    for (i, v) in b.iter_mut().enumerate() {
        *v = pseudo(i, 13);
    }
    for b_len in [LEN, 1usize, 8, 6] {
        let mut add = Aligned([0.0f32; LEN]);
        let mut mul = Aligned([0.0f32; LEN]);
        kernels::binary_add(&a, &b[..b_len], &mut add, b_len);
        kernels::binary_mul(&a, &b[..b_len], &mut mul, b_len);
        for i in 0..LEN {
            let bi = b[i % b_len];
            if add[i] != a[i] + bi || mul[i] != a[i] * bi {
                return false;
            }
        }
    }
    true
}

/// VFPU FIR against the scalar definition, on aligned and unaligned input,
/// factors 2 and 3, including the edges that stay scalar.
fn test_fir_decimate_matches_scalar() -> bool {
    const LEN: usize = 160;
    const T: usize = 31;
    let mut x = Aligned([0.0f32; LEN]);
    for (i, v) in x.iter_mut().enumerate() {
        *v = pseudo(i, 14);
    }
    let mut taps = [0.0f32; T];
    for (i, v) in taps.iter_mut().enumerate() {
        *v = pseudo(i, 15);
    }
    let center = (T - 1) / 2;
    for skew in [0usize, 1, 3] {
        let input = &x[skew..];
        for factor in [2usize, 3] {
            let n_out = input.len() / factor;
            let mut y = Aligned([0.0f32; LEN]);
            kernels::fir_decimate(input, &taps, &mut y[..n_out], factor);
            for n in 0..n_out {
                let mut want = 0.0f32;
                for (t, &h) in taps.iter().enumerate() {
                    let idx = n * factor + t;
                    if idx >= center && idx - center < input.len() {
                        want += h * input[idx - center];
                    }
                }
                if (y[n] - want).abs() > 1e-4 {
                    return false;
                }
            }
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
        test_rfft_strided_matches_dense,
        test_fir_decimate_impulse_and_dc,
        test_square_pow_matches_mul_then_pow,
        test_fc_cb_matches_dense,
        test_gemm_fused_epilogue_matches_unfused,
        test_gemm_lda_equals_im2col_for_1x1,
        test_depthwise_fused_activation_matches_ref,
        test_depthwise_wide_pixel_strips_match_ref,
        test_reduce_mean_hw_matches_direct,
        test_binary_broadcast_matches_scalar,
        test_fir_decimate_matches_scalar,
    ],
    device: [
        test_vtfm4_e_form_is_row_dots,
    ],
}
