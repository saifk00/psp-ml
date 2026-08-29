//! Installs the standalone PSPBird app onto the Memory Stick: copies
//! `EBOOT.PBP`, `weights.bin`, `blobs/*.bin` and, when the build staged
//! one, `blobs/birds.img` (the species-image pack) from the mounted host
//! directory (the pspbird build output) to `ms0:/PSP/GAME/PSPBIRD/`, the
//! layout `pspbird-host --pack` produces. No pack is not an error: the
//! app runs text-only, and any stale pack on the stick is removed so it
//! cannot disagree with the blobs just installed. Same idea as profiler-install:
//! device code can write `ms0:` directly, so no USB-storage mode needed.
//!
//! Idempotent: overwrites whatever is there.

#![no_std]
#![no_main]

use core::ffi::c_void;
use psp::sys::{
    sceIoClose, sceIoDclose, sceIoDopen, sceIoDread, sceIoMkdir, sceIoOpen, sceIoRead,
    sceIoRemove, sceIoWrite, IoOpenFlags, SceIoDirent,
};
use psp_rt::dprintln;

psp_rt::module!("pspbird-install", 1, 0);

const DST_DIR: &str = "ms0:/PSP/GAME/PSPBIRD/";
const CHUNK: usize = 64 * 1024;
static mut BUF: [u8; CHUNK] = [0; CHUNK];

fn app_main() {
    psp_rt::enable_home_button();
    dprintln!("=== PSPBird installer ===");

    for dir in ["ms0:/PSP\0", "ms0:/PSP/GAME\0", "ms0:/PSP/GAME/PSPBIRD\0", "ms0:/PSP/GAME/PSPBIRD/blobs\0"] {
        unsafe { sceIoMkdir(dir.as_ptr(), 0o777) };
    }

    let mut total = 0u64;
    let mut ok = copy("pspbird.EBOOT.PBP", "EBOOT.PBP", &mut total);
    ok &= copy("weights.bin", "weights.bin", &mut total);

    // Every blob the build staged, whatever the region list is.
    let dfd = unsafe { sceIoDopen(b"host0:/blobs\0".as_ptr()) };
    if dfd.0 < 0 {
        dprintln!("FATAL: cannot open host0:/blobs");
        return;
    }
    let mut ent: SceIoDirent = unsafe { core::mem::zeroed() };
    loop {
        if unsafe { sceIoDread(dfd, &mut ent) } <= 0 {
            break;
        }
        let name = &ent.d_name[..ent.d_name.iter().position(|&b| b == 0).unwrap_or(0)];
        let Ok(name) = core::str::from_utf8(name) else { continue };
        if !name.ends_with(".bin") {
            continue;
        }
        let mut src = heapless_path("blobs/", name);
        let mut dst = heapless_path("blobs/", name);
        ok &= copy(src.as_str(), dst.as_str(), &mut total);
        src.clear();
        dst.clear();
    }
    unsafe { sceIoDclose(dfd) };

    // Optional: the species-image pack.
    const PACK: &str = "blobs/birds.img";
    if exists(&cpath("host0:/", PACK)) {
        ok &= copy(PACK, PACK, &mut total);
    } else {
        let dst = cpath(DST_DIR, PACK);
        if exists(&dst) {
            let r = unsafe { sceIoRemove(dst.as_ptr()) };
            dprintln!("  removed stale {}{} ({:#x})", DST_DIR, PACK, r);
        }
        dprintln!("  (no species images staged; run examples/birdnet/fetch_images.py to add them)");
    }

    if ok {
        dprintln!("");
        dprintln!("DONE: {} MB in {}", total / 1_000_000, DST_DIR);
        dprintln!("Launch it from the XMB Game menu (Memory Stick).");
    } else {
        dprintln!("");
        dprintln!("FAILED: see above");
    }
}

/// Tiny fixed-capacity string: `prefix + name`.
struct SmallPath {
    buf: [u8; 64],
    len: usize,
}
impl SmallPath {
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
    fn clear(&mut self) {
        self.len = 0;
    }
}
fn heapless_path(prefix: &str, name: &str) -> SmallPath {
    let mut p = SmallPath { buf: [0; 64], len: 0 };
    for b in prefix.bytes().chain(name.bytes()).take(63) {
        p.buf[p.len] = b;
        p.len += 1;
    }
    p
}

/// NUL-terminated `prefix + rel` for the kernel.
fn cpath(prefix: &str, rel: &str) -> [u8; 96] {
    let mut p = [0u8; 96];
    let mut n = 0;
    for b in prefix.bytes().chain(rel.bytes()) {
        p[n] = b;
        n += 1;
    }
    p
}

fn exists(path: &[u8; 96]) -> bool {
    let fd = unsafe { sceIoOpen(path.as_ptr(), IoOpenFlags::RD_ONLY, 0) };
    if fd.0 < 0 {
        return false;
    }
    unsafe { sceIoClose(fd) };
    true
}

/// Copy `host0:/<src>` to `ms0:/PSP/GAME/PSPBIRD/<dst>` in 64 KiB chunks.
fn copy(src: &str, dst: &str, total: &mut u64) -> bool {
    let sp = cpath("host0:/", src);
    let dp = cpath(DST_DIR, dst);
    let sfd = unsafe { sceIoOpen(sp.as_ptr(), IoOpenFlags::RD_ONLY, 0) };
    if sfd.0 < 0 {
        dprintln!("FATAL: cannot open host0:/{} ({:#x})", src, sfd.0);
        return false;
    }
    let dfd = unsafe {
        sceIoOpen(
            dp.as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o777,
        )
    };
    if dfd.0 < 0 {
        dprintln!("FATAL: cannot create {}{} ({:#x}) - Memory Stick inserted?", DST_DIR, dst, dfd.0);
        unsafe { sceIoClose(sfd) };
        return false;
    }
    let buf = unsafe { &mut *core::ptr::addr_of_mut!(BUF) };
    let mut bytes = 0u64;
    let mut ok = true;
    loop {
        let n = unsafe { sceIoRead(sfd, buf.as_mut_ptr() as *mut c_void, CHUNK as u32) };
        if n <= 0 {
            break;
        }
        let w = unsafe { sceIoWrite(dfd, buf.as_ptr() as *const c_void, n as usize) };
        if w != n {
            dprintln!("FATAL: short write to {}{} ({} of {})", DST_DIR, dst, w, n);
            ok = false;
            break;
        }
        bytes += n as u64;
    }
    unsafe {
        sceIoClose(sfd);
        sceIoClose(dfd);
    }
    if ok {
        dprintln!("  {:>9} B  {}", bytes, dst);
        *total += bytes;
    }
    ok
}
