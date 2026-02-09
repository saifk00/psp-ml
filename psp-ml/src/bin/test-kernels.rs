#![cfg_attr(not(feature = "local"), no_std)]
#![cfg_attr(not(feature = "local"), no_main)]

#[cfg(not(feature = "local"))]
use core::ffi::c_void;
#[cfg(not(feature = "local"))]
use psp::dprintln;
#[cfg(not(feature = "local"))]
use psp::sys::{sceIoClose, sceIoOpen, sceIoWrite, IoOpenFlags};

#[cfg(not(feature = "local"))]
psp::module!("kernel_tests", 1, 0);

use psp_ml::kernels;
use psp_ml::kernels::naive;

// ============================================================================
// Test infrastructure
// ============================================================================

const EPS: f32 = 1e-4;

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

// Macro to reduce boilerplate — works in both std and no_std
macro_rules! print_msg {
    ($($arg:tt)*) => {
        #[cfg(feature = "local")]
        println!($($arg)*);
        #[cfg(not(feature = "local"))]
        dprintln!($($arg)*);
    };
}

// ============================================================================
// Test cases
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
    let mut a = [0.0f32; 64];
    for i in 0..64 { a[i] = (i as f32) * 0.1; }
    let mut b = [0.0f32; 32];
    for i in 0..32 { b[i] = (i as f32) * 0.05 + 0.01; }

    let mut c_ref = [0.0f32; 32];
    kernels::matmul_bt(&a, &b, &mut c_ref, 8, 8, 4);

    let mut c_tiled = [0.0f32; 32];
    kernels::matmul_bt_tiled(&a, &b, &mut c_tiled, 2, 2, 1);

    approx_eq(&c_tiled, &c_ref)
}

fn test_matmul_bt_tiled_large() -> bool {
    // A[16,20] @ B[8,20]^T = C[16,8]
    // m_tiles=4, k_tiles=5, n_tiles=2
    let mut a = [0.0f32; 320];
    for i in 0..320 { a[i] = ((i % 17) as f32) * 0.1 - 0.8; }
    let mut b = [0.0f32; 160];
    for i in 0..160 { b[i] = ((i % 13) as f32) * 0.07 - 0.4; }

    let mut c_ref = [0.0f32; 128];
    kernels::matmul_bt(&a, &b, &mut c_ref, 16, 20, 8);

    let mut c_tiled = [0.0f32; 128];
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
        [pad_h, pad_w],
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

    let mut im2col_buf = [0.0f32; 720]; // 36 * 20
    kernels::im2col_padded(
        &input,
        [1, h, w, ci],
        [kh, kw],
        [pad_h, pad_w],
        [ho, wo],
        &mut im2col_buf,
    );

    // Step 2: Pad weights [Co, K] → [Co, K_padded]
    // Filter is [Co, Kh, Kw, Ci] = [4, 18] row-major
    let mut weights_padded = [0.0f32; 80]; // 4 * 20
    for row in 0..co {
        for col in 0..gemm_k {
            weights_padded[row * k_padded + col] = filter[row * gemm_k + col];
        }
    }

    // Step 3: matmul_bt_tiled
    let m_tiles = m_padded / 4;
    let k_tiles = k_padded / 4;
    let n_tiles = n_padded / 4;
    let mut output_vfpu = [0.0f32; 144]; // 36 * 4
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

// ============================================================================
// Test runner
// ============================================================================

type TestFn = fn() -> bool;

const NUM_TESTS: usize = 11;

const TESTS: [(&str, TestFn); NUM_TESTS] = [
    ("relu", test_relu),
    ("bias_add", test_bias_add),
    ("matmul_bt_identity", test_matmul_bt_identity),
    ("matmul_bt_known", test_matmul_bt_known),
    ("matmul_bt_non_aligned", test_matmul_bt_non_aligned),
    ("matmul_bt_tiled", test_matmul_bt_tiled),
    ("matmul_bt_tiled_large", test_matmul_bt_tiled_large),
    ("im2col_simple", test_im2col_simple),
    ("im2col_with_padding", test_im2col_with_padding),
    ("im2col_padded_vs_im2col", test_im2col_padded_vs_im2col),
    ("conv2d_via_im2col_vs_naive", test_conv2d_via_im2col_vs_naive),
];

fn run_all_tests() -> (u32, u32, [bool; NUM_TESTS]) {
    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut results = [false; NUM_TESTS];

    for (i, &(name, test_fn)) in TESTS.iter().enumerate() {
        let ok = test_fn();
        results[i] = ok;
        if ok {
            print_msg!("  PASS  {}", name);
            passed += 1;
        } else {
            print_msg!("  FAIL  {}", name);
            failed += 1;
        }
    }

    (passed, failed, results)
}

// ============================================================================
// JSON output
// ============================================================================

struct JsonBuf {
    buf: [u8; 2048],
    pos: usize,
}

impl JsonBuf {
    fn new() -> Self {
        JsonBuf {
            buf: [0u8; 2048],
            pos: 0,
        }
    }

    fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.pos]
    }

    fn push_str(&mut self, s: &str) {
        for &b in s.as_bytes() {
            if self.pos < self.buf.len() {
                self.buf[self.pos] = b;
                self.pos += 1;
            }
        }
    }

    fn push_u32(&mut self, mut val: u32) {
        if val == 0 {
            self.push_str("0");
            return;
        }
        let start = self.pos;
        while val > 0 {
            if self.pos < self.buf.len() {
                self.buf[self.pos] = b'0' + (val % 10) as u8;
                self.pos += 1;
            }
            val /= 10;
        }
        let end = self.pos;
        let mut i = start;
        let mut j = end - 1;
        while i < j {
            self.buf.swap(i, j);
            i += 1;
            j -= 1;
        }
    }
}

fn format_results(passed: u32, failed: u32, results: &[bool; NUM_TESTS]) -> JsonBuf {
    let mut j = JsonBuf::new();
    j.push_str("{\n");
    j.push_str("  \"passed\": ");
    j.push_u32(passed);
    j.push_str(",\n");
    j.push_str("  \"failed\": ");
    j.push_u32(failed);
    j.push_str(",\n");
    j.push_str("  \"tests\": [\n");
    for (idx, &(name, _)) in TESTS.iter().enumerate() {
        j.push_str("    { \"name\": \"");
        j.push_str(name);
        j.push_str("\", \"passed\": ");
        j.push_str(if results[idx] { "true" } else { "false" });
        j.push_str(" }");
        if idx + 1 < TESTS.len() {
            j.push_str(",");
        }
        j.push_str("\n");
    }
    j.push_str("  ]\n");
    j.push_str("}\n");
    j
}

// ============================================================================
// PSP entry point
// ============================================================================

#[cfg(not(feature = "local"))]
fn psp_main() {
    psp::enable_home_button();

    dprintln!("psp-ml Kernel Tests");
    dprintln!("====================");
    dprintln!("");

    let (passed, failed, results) = run_all_tests();

    dprintln!("");
    dprintln!("Results: {} passed, {} failed", passed, failed);

    // Write JSON to host0:/test-results.json
    let json = format_results(passed, failed, &results);
    let path = b"host0:/test-results.json\0";
    let fd = unsafe {
        sceIoOpen(
            path.as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o644,
        )
    };
    if fd.0 >= 0 {
        unsafe {
            sceIoWrite(fd, json.as_bytes().as_ptr() as *const c_void, json.as_bytes().len());
            sceIoClose(fd);
        }
        dprintln!("Wrote test-results.json to host0:/");
    }

    dprintln!("");
    if failed == 0 {
        dprintln!("All tests passed!");
    } else {
        dprintln!("{} test(s) FAILED", failed);
    }

    let exit_code = if failed > 0 { 1 } else { 0 };
    psp_ml::psp_exit(exit_code);
}

// ============================================================================
// Local (host CPU) entry point
// ============================================================================

#[cfg(feature = "local")]
fn main() {
    println!("psp-ml Kernel Tests");
    println!("====================");
    println!();

    let (passed, failed, results) = run_all_tests();

    println!();
    println!("Results: {} passed, {} failed", passed, failed);

    // Write JSON
    let json = format_results(passed, failed, &results);
    let out_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test-results.json");
    std::fs::write(out_path, json.as_bytes()).expect("failed to write test-results.json");
    println!("Wrote {}", out_path);

    if failed > 0 {
        std::process::exit(1);
    }
}
