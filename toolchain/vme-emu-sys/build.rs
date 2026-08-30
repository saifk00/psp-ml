//! Verilates ../vme-emu/rtl into a static library (model + verilated
//! runtime + the C shim, all in one archive via --lib-create) and links it.
//! Needs `verilator` on PATH -- the same prerequisite class as libusb for
//! usbhostfs-sys; this crate is deliberately not a workspace default-member.

use std::path::Path;
use std::process::Command;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let rtl_dir = Path::new(&manifest).join("../vme-emu/rtl");
    let shim = Path::new(&manifest).join("csrc/vme_shim.cpp");
    let obj_dir = Path::new(&out_dir).join("obj");

    let mut rtl: Vec<_> = std::fs::read_dir(&rtl_dir)
        .expect("vme-emu/rtl not found")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "v").then_some(p)
        })
        .collect();
    rtl.sort();

    let mut cmd = Command::new("verilator");
    cmd.args(["--cc", "--build", "-j", "0", "--lib-create", "vme_rtl"])
        .args(["--top-module", "vme_top"])
        .args(["-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC"])
        .arg("-Mdir")
        .arg(&obj_dir)
        .args(&rtl)
        .arg(&shim);
    let status = cmd.status().unwrap_or_else(|e| {
        panic!("failed to run verilator ({e}). Is Verilator installed? apt install verilator")
    });
    assert!(status.success(), "verilator failed to build the VME model");

    println!("cargo:rustc-link-search=native={}", obj_dir.display());
    println!("cargo:rustc-link-lib=static=vme_rtl");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rerun-if-changed={}", rtl_dir.display());
    println!("cargo:rerun-if-changed={}", shim.display());
}
