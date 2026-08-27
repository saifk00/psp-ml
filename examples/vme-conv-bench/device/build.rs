//! Bakes the VME conv contexts for the bench shapes (psp-tc's generator is
//! host-side; the device just streams the baked words).

use std::fmt::Write as _;

/// (k, co, pixels) — BirdNET's int8 1x1 conv shapes at runtime, from the
/// FLOP census (the K=1728 3x3 exceeds the ring-fit range and never maps).
const SHAPES: [(usize, usize, usize); 4] =
    [(24, 72, 6144), (36, 288, 1536), (72, 864, 384), (108, 1536, 96)];

fn ctx_str(k: usize, p: usize) -> String {
    let ctx = psp_tc::vme_conv::vme_conv1x1_ctx(k, p)
        .unwrap_or_else(|e| panic!("ctx k={k} p={p}: {e}"));
    let mut s = String::from("[");
    for w in ctx {
        write!(s, "{}, ", w as i32).unwrap();
    }
    s.push(']');
    s
}

fn main() {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let mut rs = String::new();

    writeln!(rs, "pub const N_SHAPES: usize = {};", SHAPES.len()).unwrap();
    writeln!(
        rs,
        "pub struct Shape {{ pub k: usize, pub co: usize, pub pixels: usize, \
         pub p_full: usize, pub weights_off: usize, \
         pub ctx_full: &'static [i32], pub ctx_rem: &'static [i32] }}"
    )
    .unwrap();

    let mut entries = String::new();
    for (i, &(k, co, pixels)) in SHAPES.iter().enumerate() {
        let plan = psp_tc::vme_conv::plan_vme_conv1x1(k).unwrap();
        let p_full = plan.pixels_per_job;
        let rem = pixels % p_full;
        writeln!(rs, "static CTX_FULL_{i}: [i32; 106] = {};", ctx_str(k, p_full)).unwrap();
        if rem > 0 {
            writeln!(rs, "static CTX_REM_{i}: [i32; 106] = {};", ctx_str(k, rem)).unwrap();
        } else {
            writeln!(rs, "static CTX_REM_{i}: [i32; 0] = [];").unwrap();
        }
        writeln!(
            entries,
            "    Shape {{ k: {k}, co: {co}, pixels: {pixels}, p_full: {p_full}, \
             weights_off: {}, ctx_full: &CTX_FULL_{i}, ctx_rem: &CTX_REM_{i} }},",
            plan.weights_off
        )
        .unwrap();
    }
    writeln!(rs, "pub static SHAPES: [Shape; N_SHAPES] = [\n{entries}];").unwrap();

    std::fs::write(out_dir.join("shapes.rs"), rs).unwrap();
}
