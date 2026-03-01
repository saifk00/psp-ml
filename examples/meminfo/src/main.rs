#![no_std]
#![no_main]

use core::ffi::c_void;
use psp::sys::{
    sceIoClose, sceIoOpen, sceIoWrite, sceKernelMaxFreeMemSize, sceKernelTotalFreeMemSize,
    IoOpenFlags,
};
use psp_ml::dprintln;

psp_ml::module!("meminfo", 1, 0);

fn app_main() {
    psp::enable_home_button();

    let total_free = unsafe { sceKernelTotalFreeMemSize() };
    let max_block = unsafe { sceKernelMaxFreeMemSize() };

    dprintln!("=== PSP Memory Info ===");
    dprintln!("Total free:     {} bytes ({} KB / {} MB)", total_free, total_free / 1024, total_free / (1024 * 1024));
    dprintln!("Max free block: {} bytes ({} KB / {} MB)", max_block, max_block / 1024, max_block / (1024 * 1024));

    // Write to host file
    let mut buf = [0u8; 256];
    let len = fmt_meminfo(&mut buf, total_free, max_block);
    let fd = unsafe {
        sceIoOpen(
            b"host0:/meminfo.txt\0".as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o644,
        )
    };
    if fd.0 >= 0 {
        unsafe {
            sceIoWrite(fd, buf.as_ptr() as *const c_void, len);
            sceIoClose(fd);
        }
        dprintln!("Wrote meminfo.txt");
    }
}

fn fmt_meminfo(buf: &mut [u8], total_free: usize, max_block: usize) -> usize {
    let mut pos = 0;

    fn push(buf: &mut [u8], pos: &mut usize, s: &[u8]) {
        for &b in s {
            if *pos < buf.len() {
                buf[*pos] = b;
                *pos += 1;
            }
        }
    }

    fn push_usize(buf: &mut [u8], pos: &mut usize, mut val: usize) {
        if val == 0 {
            push(buf, pos, b"0");
            return;
        }
        let start = *pos;
        while val > 0 && *pos < buf.len() {
            buf[*pos] = b'0' + (val % 10) as u8;
            val /= 10;
            *pos += 1;
        }
        buf[start..*pos].reverse();
    }

    push(buf, &mut pos, b"total_free_bytes: ");
    push_usize(buf, &mut pos, total_free);
    push(buf, &mut pos, b"\nmax_block_bytes:  ");
    push_usize(buf, &mut pos, max_block);
    push(buf, &mut pos, b"\ntotal_free_mb:    ");
    push_usize(buf, &mut pos, total_free / (1024 * 1024));
    push(buf, &mut pos, b"\nmax_block_mb:     ");
    push_usize(buf, &mut pos, max_block / (1024 * 1024));
    push(buf, &mut pos, b"\n");
    pos
}
