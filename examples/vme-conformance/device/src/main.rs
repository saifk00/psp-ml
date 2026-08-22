//! Device half of the VME conformance harness: a machine-image server.
//!
//! Loops accepting jobs from the host and running each on the *real* VME
//! (via the `psp_vme_kernel` plugin's v1.1 image mode — install the
//! rebuilt plugin and power-cycle first), then streams all eight ring
//! buffers back over stdout. The host runs the same images against the
//! Verilated RTL and diffs the two.
//!
//! Job protocol (psplink offers no app-stdin channel — the shell channel is
//! command-framed and nothing feeds `sceKernelStdin` — so jobs arrive over
//! the mounted filesystem, the repo's standard host→device path; results
//! go back over stdout as requested):
//!
//!   host writes  host0:/job.bin   (1 MB machine image), then
//!                host0:/job.go    (empty marker — written second, so the
//!                                  image is complete when the marker lands)
//!   device       removes job.go, reads job.bin, runs it, prints
//!                  #vme result begin
//!                  <16384 hex words: TOP_0..TOP_3 then BASE_0..BASE_3>
//!                  #vme result end
//!   host writes  host0:/quit.go   (the exit sentinel) — device exits its
//!                                  loop and the module completes cleanly.
//!
//! Everything must finish inside `psp_rt::module!`'s 240 s budget; the host
//! sends the sentinel as soon as its last test is done.

#![no_std]
#![no_main]

use psp::sys::{
    sceIoClose, sceIoOpen, sceIoRead, sceIoRemove, sceKernelDelayThread, IoOpenFlags,
};
use psp_rt::{dprint, dprintln, vme};

psp_rt::module!("vme-conformance", 1, 0);

const IMAGE_BYTES: usize = 0x100000;
const BUF_WORDS: usize = 2048;
/// Byte offsets of the eight buffers within the image, in wire order
/// TOP_0..TOP_3 then BASE_0..BASE_3.
const WIRE_ORDER: [usize; 8] = [
    0x20000, 0x22000, 0x24000, 0x26000, 0x0000, 0x2000, 0x4000, 0x6000,
];

fn app_main() {
    psp_rt::enable_home_button();
    unsafe { psp::sys::scePowerSetClockFrequency(333, 333, 166) };

    dprintln!("=== vme-conformance device ===");

    let table = vme::init();
    if table < 0 {
        dprintln!("#vme fatal init {} (no plugin? not power-cycled?)", table);
        return;
    }
    dprintln!("ME booted, image table = {}", table);
    let Some(job) = vme::Job::get() else {
        dprintln!("#vme fatal no shared job");
        return;
    };
    if !job.has_image_mode() {
        dprintln!("#vme fatal plugin too old (v1.1 image mode needed -- reinstall + power-cycle)");
        return;
    }

    // The image lives in the user partition; hand the ME (and our own
    // fills/reads) the uncached alias so nothing needs cache maintenance.
    let base = psp_rt::mem::alloc_partition(b"vme_img\0", IMAGE_BYTES + 64, None);
    if base.is_null() {
        dprintln!("#vme fatal image alloc failed");
        return;
    }
    let aligned = ((base as usize + 63) & !63) | 0x4000_0000;
    let img = aligned as *mut u8;

    job.set_image(aligned as u32);
    dprintln!("#vme ready");

    let mut jobs = 0u32;
    loop {
        if try_remove(b"host0:/quit.go\0") {
            break;
        }
        if !try_remove(b"host0:/job.go\0") {
            unsafe { sceKernelDelayThread(20_000) };
            continue;
        }

        if !read_image(b"host0:/job.bin\0", img) {
            dprintln!("#vme error job.bin read failed");
            continue;
        }

        let rc = vme::run();
        if rc < 0 {
            dprintln!("#vme error run {}", rc);
            continue;
        }
        jobs += 1;

        dprintln!("#vme result begin");
        for off in WIRE_ORDER {
            let words = unsafe { img.add(off) } as *const u32;
            for row in 0..(BUF_WORDS / 16) {
                for i in 0..16 {
                    let w = unsafe { core::ptr::read_volatile(words.add(row * 16 + i)) };
                    dprint!("{:08x} ", w);
                }
                dprintln!("");
            }
        }
        dprintln!("#vme result end");
    }

    dprintln!("#vme quit after {} jobs", jobs);
    // alloc_partition's registry frees the image in the module! epilogue.
}

/// Remove `path` if it exists; whether it did is the poll result.
fn try_remove(path: &[u8]) -> bool {
    unsafe { sceIoRemove(path.as_ptr()) >= 0 }
}

/// Read exactly IMAGE_BYTES from `path` into `dst` (uncached), in 64 KiB
/// chunks — hostfs throughput collapses below that.
fn read_image(path: &[u8], dst: *mut u8) -> bool {
    let fd = unsafe { sceIoOpen(path.as_ptr(), IoOpenFlags::RD_ONLY, 0o777) };
    if fd.0 < 0 {
        return false;
    }
    let mut got: usize = 0;
    while got < IMAGE_BYTES {
        let chunk = core::cmp::min(64 * 1024, IMAGE_BYTES - got);
        let n = unsafe { sceIoRead(fd, dst.add(got) as *mut _, chunk as u32) };
        if n <= 0 {
            unsafe { sceIoClose(fd) };
            return false;
        }
        got += n as usize;
    }
    unsafe { sceIoClose(fd) };
    got == IMAGE_BYTES
}
