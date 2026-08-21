//! Pure-Rust integer SIMD multiply-accumulate on the PSP's VME.
//!
//! The C `examples/vme-mac` proved the mechanism; this is the pure-Rust
//! version. The privileged ME boot + VME driver live in the `psp_vme_kernel`
//! plugin (install it once, power-cycle); here, Rust:
//!   1. boots the ME via `vme::init`,
//!   2. validates the whole path with the plugin's built-in `self_test` (72),
//!   3. authors a VME datapath program with `vme_asm!` and runs it — a pure
//!      Rust integer dot product cross-checked against a scalar reference.

#![no_std]
#![no_main]

use psp_rt::dprintln;
use psp_rt::vme;

psp_rt::module!("vme-mac-rs", 1, 0);

fn app_main() {
    psp_rt::enable_home_button();
    unsafe { psp::sys::scePowerSetClockFrequency(333, 333, 166) };

    dprintln!("=== pure-Rust VME MAC ===");

    let table = vme::init();
    if table < 0 {
        dprintln!("vme::init failed: {} (no plugin? not power-cycled?)", table);
        return;
    }
    dprintln!("ME booted, image table = {}", table);

    // Stage 1 — validate boot + runner + shared memory with the plugin's own
    // C-built MAC. (Needs the fixed plugin reinstalled; harmless if stale.)
    let st = vme::self_test();
    dprintln!("self_test() = {} (built-in C MAC)", st);

    // Stage 2 — the real goal: a pure-Rust integer dot product. Author the VME
    // datapath program with `vme_asm!`, stage the operands, run, read back.
    mac_rust();
}

/// A pure-Rust VME integer MAC: dot(weights, inputs) with a scalar cross-check.
fn mac_rust() {
    use vme::asm::*;
    use vme::rings;

    const N: u32 = 8;
    const GAP: u32 = 256; // keep inputs clear of the streamed output in BASE_0
    let weights: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let inputs: [u32; 8] = [2, 2, 2, 2, 2, 2, 2, 2];

    // The datapath program: PE0 computes Σ(TOP_0[n] · BASE_0[n]) with k=0.
    let ctx = psp_rt::vme_asm![
        icn(AGU_BASE)         => 0x4440,
        pe_fu_primary(0)      => fu(MAC_INNER_PRODUCT_BIAS, FRONT_TOP_0, BACK_BASE_0, 0),
        pe_fu_primary_b(0)    => 0,                        // bias
        pe_read_top_mode(0)   => def_mode(),
        pe_read_top_count(0)  => step(N),
        pe_read_base_mode(0)  => def_mode() | gap(GAP),
        pe_read_base_count(0) => step(N),
        pe_write_mode(0)      => def_mode() | cycle(9),
        pe_write_count(0)     => step(N),
    ];

    let Some(job) = vme::Job::get() else {
        dprintln!("no shared job (init failed?)");
        return;
    };

    // Operands: weights at data[0..], inputs at data[64..].
    job.set_data(0, &weights);
    job.set_data(64, &inputs);
    // Stage weights → TOP_0, inputs → BASE_0 + GAP.
    job.set_stages(&[
        vme::VmeStage { src_word: 0, ring_woff: rings::top_woff(0), count: N },
        vme::VmeStage { src_word: 64, ring_woff: rings::base_woff(0) + GAP, count: N },
    ]);
    job.set_ctx(&ctx);
    // Result is the last accumulated word of BASE_0.
    job.set_readback(rings::BASE_BUFFERS, N - 1);

    let rc = vme::run();
    if rc < 0 {
        dprintln!("vme::run timed out: {}", rc);
        return;
    }

    let got = job.result();
    let mut reference: u32 = 0;
    for i in 0..N as usize {
        reference += weights[i] * inputs[i];
    }
    dprintln!(
        "VME dot = {}, scalar reference = {} -> {}",
        got,
        reference,
        if got == reference { "PASS" } else { "FAIL" }
    );
}
