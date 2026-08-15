#![cfg_attr(not(feature = "local"), no_std)]
#![cfg_attr(not(feature = "local"), no_main)]

#[cfg(not(feature = "local"))]
use core::ffi::c_void;
#[cfg(not(feature = "local"))]
use psp::sys::{sceIoClose, sceIoOpen, sceIoWrite, IoOpenFlags};

#[cfg(not(feature = "local"))]
psp_rt::module!("kernel_tests", 1, 0);

use psp_rt::kernels;

// ============================================================================
// Test infrastructure
// ============================================================================



macro_rules! print_msg {
    ($($arg:tt)*) => {
        #[cfg(feature = "local")]
        println!($($arg)*);
        #[cfg(not(feature = "local"))]
        psp_rt::dprintln!($($arg)*);
    };
}
// Test cases now live in `psp_rt::kernels::checks`, so the host suite
// (`cargo test -p psp-rt`) and this on-device runner execute the identical
// list — one against the scalar fallbacks, one against the real VFPU code.

const NUM_TESTS: usize = kernels::checks::CHECKS.len();

const TESTS: &[(&str, fn() -> bool)] = kernels::checks::CHECKS;

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
fn app_main() {
    psp_rt::enable_home_button();

    psp_rt::dprintln!("psp-rt Kernel Tests");
    psp_rt::dprintln!("====================");
    psp_rt::dprintln!("");

    let (passed, failed, results) = run_all_tests();

    psp_rt::dprintln!("");
    psp_rt::dprintln!("Results: {} passed, {} failed", passed, failed);

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
        psp_rt::dprintln!("Wrote test-results.json to host0:/");
    }

    psp_rt::dprintln!("");
    if failed == 0 {
        psp_rt::dprintln!("All tests passed!");
    } else {
        psp_rt::dprintln!("{} test(s) FAILED", failed);
    }
}

// ============================================================================
// Local (host CPU) entry point
// ============================================================================

#[cfg(feature = "local")]
fn main() {
    println!("psp-rt Kernel Tests");
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
