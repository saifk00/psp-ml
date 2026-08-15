//! Runs `psp_rt::device_test::SUITES` and reports over the wire protocol.
//!
//! Built for the PSP by `device-tests`' build script and deployed by
//! `cargo test -p device-tests`, which parses what this prints. Results go over
//! stdout, not a file: a check that faults takes psplink down with it and never
//! reaches a `sceIoWrite`, but everything printed before the fault has already
//! crossed the USB link — including the `start` line naming the check.
//!
//! `--features local` builds the same walk as an ordinary host binary, which
//! is a way to eyeball the protocol without hardware. It runs the scalar
//! fallbacks and skips the device-only checks, so `cargo test -p psp-rt` is
//! the real host-side gate.

#![cfg_attr(not(feature = "local"), no_std)]
#![cfg_attr(not(feature = "local"), no_main)]

use psp_rt::device_test::protocol::{emit, Line, Name};
use psp_rt::device_test::{total_checks, SUITES};

#[cfg(not(feature = "local"))]
psp_rt::module!("device_tests", 1, 0);

/// Walk every suite, announcing each check before running it.
fn run_all() -> (u32, u32) {
    let mut passed = 0u32;
    let mut failed = 0u32;

    emit(Line::Plan(total_checks()));
    for suite in SUITES {
        for &(check, run) in suite.checks {
            let name = Name {
                suite: suite.name,
                check,
            };
            // Before, not after: if `run` hangs or faults, this is the only
            // record of which check did it.
            emit(Line::Start(name));
            if run() {
                emit(Line::Ok(name));
                passed += 1;
            } else {
                emit(Line::Fail(name));
                failed += 1;
            }
        }
    }
    emit(Line::Done { passed, failed });

    (passed, failed)
}

#[cfg(not(feature = "local"))]
fn app_main() {
    psp_rt::enable_home_button();
    run_all();
}

#[cfg(feature = "local")]
fn main() {
    let (_, failed) = run_all();
    if failed > 0 {
        std::process::exit(1);
    }
}
