//! One-time installer for the `psp_ml_kernel` profiling plugin.
//!
//! The plugin runs in kernel mode, and psplink's user-mode `ld` cannot start a
//! kernel PRX — attempting it kills the hostfs session. A kernel plugin has to
//! be loaded by the firmware at boot from `ms0:/seplugins/`, which normally
//! means pulling the Memory Stick or rebooting into USB storage mode.
//!
//! It doesn't have to: device code can write `ms0:` directly. This runs as an
//! ordinary user-mode program under psplink, copies the plugin from the
//! mounted host directory to the Memory Stick, and registers it in `game.txt`.
//! Power-cycle once afterwards and the profiler works from then on.
//!
//! Idempotent: re-running overwrites the PRX (so a rebuilt plugin is picked
//! up) but never duplicates the `game.txt` line.

#![no_std]
#![no_main]

use core::ffi::c_void;
use psp::sys::{
    sceIoClose, sceIoLseek, sceIoMkdir, sceIoOpen, sceIoRead, sceIoWrite, IoOpenFlags, IoWhence,
};
use psp_rt::dprintln;

psp_rt::module!("profiler-install", 1, 0);

const SRC: &[u8] = b"host0:/psp_ml_kernel.prx\0";
const DST: &[u8] = b"ms0:/seplugins/psp_ml_kernel.prx\0";
const GAME_TXT: &[u8] = b"ms0:/seplugins/game.txt\0";
const SEPLUGINS: &[u8] = b"ms0:/seplugins\0";
/// The line firmware reads to load the plugin. Trailing `1` = enabled.
const ENTRY: &[u8] = b"ms0:/seplugins/psp_ml_kernel.prx 1\n";
/// Enough for the plugin (~3.5 KB today) with room to grow.
const BUF_BYTES: usize = 64 * 1024;

static mut BUF: [u8; BUF_BYTES] = [0; BUF_BYTES];

fn app_main() {
    psp_rt::enable_home_button();

    dprintln!("=== Profiler plugin installer ===");

    // sceIoMkdir on an existing directory returns an error we deliberately
    // ignore; the open below is the real test of whether the path is usable.
    unsafe { sceIoMkdir(SEPLUGINS.as_ptr(), 0o777) };

    let Some(len) = read_plugin() else { return };
    dprintln!("read {} bytes from host0:/psp_ml_kernel.prx", len);

    if !write_plugin(len) {
        return;
    }
    dprintln!("wrote ms0:/seplugins/psp_ml_kernel.prx");

    match register_in_game_txt() {
        Some(true) => dprintln!("registered in ms0:/seplugins/game.txt"),
        Some(false) => dprintln!("already registered in game.txt; left unchanged"),
        None => return,
    }

    dprintln!("");
    dprintln!("DONE. Power-cycle the PSP once, then profiling is available:");
    dprintln!("  PSP_PROFILE=1 cargo run -p birdnet-host --release");
}

/// Read the plugin from the mounted host directory into `BUF`.
fn read_plugin() -> Option<usize> {
    let fd = unsafe { sceIoOpen(SRC.as_ptr(), IoOpenFlags::RD_ONLY, 0) };
    if fd.0 < 0 {
        dprintln!("FATAL: cannot open host0:/psp_ml_kernel.prx");
        dprintln!("  the host mounts kernel-plugin/ as host0:; run `make -C kernel-plugin` first");
        return None;
    }
    let size = unsafe { sceIoLseek(fd, 0, IoWhence::End) } as usize;
    unsafe { sceIoLseek(fd, 0, IoWhence::Set) };
    if size > BUF_BYTES {
        dprintln!("FATAL: plugin is {} B, buffer is {} B", size, BUF_BYTES);
        unsafe { sceIoClose(fd) };
        return None;
    }

    let mut done = 0usize;
    while done < size {
        let n = unsafe {
            sceIoRead(
                fd,
                (&raw mut BUF as *mut u8).add(done) as *mut c_void,
                (size - done) as u32,
            )
        };
        if n <= 0 {
            break;
        }
        done += n as usize;
    }
    unsafe { sceIoClose(fd) };

    if done != size {
        dprintln!("FATAL: short read: {} of {} bytes", done, size);
        return None;
    }
    Some(size)
}

fn write_plugin(len: usize) -> bool {
    let fd = unsafe {
        sceIoOpen(
            DST.as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o777,
        )
    };
    if fd.0 < 0 {
        dprintln!("FATAL: cannot write ms0:/seplugins/ — is a Memory Stick inserted?");
        return false;
    }
    let n = unsafe { sceIoWrite(fd, &raw const BUF as *const c_void, len) };
    unsafe { sceIoClose(fd) };
    if n < 0 || n as usize != len {
        dprintln!("FATAL: short write to Memory Stick ({} of {})", n, len);
        return false;
    }
    true
}

/// Append the plugin line to `game.txt` unless it is already there.
///
/// Returns `Some(true)` if it appended, `Some(false)` if it was already
/// present, `None` on error.
fn register_in_game_txt() -> Option<bool> {
    // Read what's there (absent file is fine — it just means an empty config).
    let mut existing = 0usize;
    let fd = unsafe { sceIoOpen(GAME_TXT.as_ptr(), IoOpenFlags::RD_ONLY, 0) };
    if fd.0 >= 0 {
        let size = unsafe { sceIoLseek(fd, 0, IoWhence::End) } as usize;
        unsafe { sceIoLseek(fd, 0, IoWhence::Set) };
        if size < BUF_BYTES {
            while existing < size {
                let n = unsafe {
                    sceIoRead(
                        fd,
                        (&raw mut BUF as *mut u8).add(existing) as *mut c_void,
                        (size - existing) as u32,
                    )
                };
                if n <= 0 {
                    break;
                }
                existing += n as usize;
            }
        }
        unsafe { sceIoClose(fd) };
    }

    let text = unsafe { &(&raw const BUF as *const [u8; BUF_BYTES]).as_ref().unwrap()[..existing] };
    if contains(text, b"psp_ml_kernel.prx") {
        return Some(false);
    }

    // Append rather than rewrite: game.txt may hold other people's plugins.
    let fd = unsafe {
        sceIoOpen(
            GAME_TXT.as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::APPEND,
            0o777,
        )
    };
    if fd.0 < 0 {
        dprintln!("FATAL: cannot open ms0:/seplugins/game.txt for append");
        return None;
    }
    // A config whose last line lacks a newline would otherwise be joined to
    // ours, disabling both entries.
    if existing > 0 && text[existing - 1] != b'\n' {
        unsafe { sceIoWrite(fd, b"\n".as_ptr() as *const c_void, 1) };
    }
    let n = unsafe { sceIoWrite(fd, ENTRY.as_ptr() as *const c_void, ENTRY.len()) };
    unsafe { sceIoClose(fd) };
    if n < 0 {
        dprintln!("FATAL: failed to append to game.txt");
        return None;
    }
    Some(true)
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}
