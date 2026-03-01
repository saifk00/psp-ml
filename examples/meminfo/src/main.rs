#![no_std]
#![no_main]

use core::ffi::c_void;
use psp::sys::{
    sceIoClose, sceIoOpen, sceIoWrite, sceKernelAllocPartitionMemory, sceKernelFreePartitionMemory,
    sceKernelGetBlockHeadAddr, sceKernelMaxFreeMemSize, sceKernelTotalFreeMemSize,
    IoOpenFlags, SceSysMemBlockTypes,
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

    // Probe partitions by trying to allocate 1KB from each
    dprintln!("");
    dprintln!("=== Partition Probe ===");
    for pid in [2u32, 5, 6, 8, 10] {
        let uid = unsafe {
            sceKernelAllocPartitionMemory(
                core::mem::transmute(pid as i32),
                b"probe\0".as_ptr(),
                SceSysMemBlockTypes::Low,
                1024,
                core::ptr::null_mut(),
            )
        };
        if uid.0 >= 0 {
            let addr = unsafe { sceKernelGetBlockHeadAddr(uid) };
            dprintln!("  Partition {}: addr=0x{:08X}", pid, addr as usize);
            unsafe { sceKernelFreePartitionMemory(uid) };
        } else {
            dprintln!("  Partition {}: FAILED (0x{:08X})", pid, uid.0 as u32);
        }
    }

    // Try large allocations from partition 2 to find contiguous limit
    dprintln!("");
    dprintln!("=== Max Contiguous Alloc (Partition 2) ===");
    for size_mb in [48u32, 40, 32, 28, 24, 20, 16] {
        let size = size_mb * 1024 * 1024;
        let uid = unsafe {
            sceKernelAllocPartitionMemory(
                core::mem::transmute(2i32),
                b"test\0".as_ptr(),
                SceSysMemBlockTypes::Low,
                size,
                core::ptr::null_mut(),
            )
        };
        if uid.0 >= 0 {
            let addr = unsafe { sceKernelGetBlockHeadAddr(uid) };
            dprintln!("  {}MB: OK at 0x{:08X}", size_mb, addr as usize);
            unsafe { sceKernelFreePartitionMemory(uid) };
        } else {
            dprintln!("  {}MB: FAILED", size_mb);
        }
    }

    // Write all results to host file
    let mut buf = [0u8; 2048];
    let mut pos = 0;

    pos += fmt_line(&mut buf[pos..], b"total_free_bytes: ", total_free);
    pos += fmt_line(&mut buf[pos..], b"max_block_bytes:  ", max_block);
    pos += fmt_line(&mut buf[pos..], b"total_free_mb:    ", total_free / (1024 * 1024));
    pos += fmt_line(&mut buf[pos..], b"max_block_mb:     ", max_block / (1024 * 1024));
    push(&mut buf, &mut pos, b"\n--- Partition Probe ---\n");

    for pid in [2u32, 5, 6, 8, 10] {
        let uid = unsafe {
            sceKernelAllocPartitionMemory(
                core::mem::transmute(pid as i32),
                b"probe\0".as_ptr(),
                SceSysMemBlockTypes::Low,
                1024,
                core::ptr::null_mut(),
            )
        };
        push(&mut buf, &mut pos, b"partition ");
        push_usize(&mut buf, &mut pos, pid as usize);
        if uid.0 >= 0 {
            let addr = unsafe { sceKernelGetBlockHeadAddr(uid) } as usize;
            push(&mut buf, &mut pos, b": addr=0x");
            push_hex(&mut buf, &mut pos, addr);
            push(&mut buf, &mut pos, b"\n");
            unsafe { sceKernelFreePartitionMemory(uid) };
        } else {
            push(&mut buf, &mut pos, b": FAILED (0x");
            push_hex(&mut buf, &mut pos, uid.0 as usize);
            push(&mut buf, &mut pos, b")\n");
        }
    }

    push(&mut buf, &mut pos, b"\n--- Max Contiguous Alloc (Partition 2) ---\n");
    for size_mb in [48u32, 40, 32, 28, 24, 20, 16] {
        let size = size_mb * 1024 * 1024;
        let uid = unsafe {
            sceKernelAllocPartitionMemory(
                core::mem::transmute(2i32),
                b"test\0".as_ptr(),
                SceSysMemBlockTypes::Low,
                size,
                core::ptr::null_mut(),
            )
        };
        push_usize(&mut buf, &mut pos, size_mb as usize);
        push(&mut buf, &mut pos, b"MB: ");
        if uid.0 >= 0 {
            let addr = unsafe { sceKernelGetBlockHeadAddr(uid) } as usize;
            push(&mut buf, &mut pos, b"OK at 0x");
            push_hex(&mut buf, &mut pos, addr);
            push(&mut buf, &mut pos, b"\n");
            unsafe { sceKernelFreePartitionMemory(uid) };
        } else {
            push(&mut buf, &mut pos, b"FAILED\n");
        }
    }

    let fd = unsafe {
        sceIoOpen(
            b"host0:/meminfo.txt\0".as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o644,
        )
    };
    if fd.0 >= 0 {
        unsafe {
            sceIoWrite(fd, buf.as_ptr() as *const c_void, pos);
            sceIoClose(fd);
        }
        dprintln!("Wrote meminfo.txt");
    }
}

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

fn push_hex(buf: &mut [u8], pos: &mut usize, val: usize) {
    for i in (0..8).rev() {
        let nibble = (val >> (i * 4)) & 0xF;
        let ch = if nibble < 10 { b'0' + nibble as u8 } else { b'A' + (nibble - 10) as u8 };
        if *pos < buf.len() {
            buf[*pos] = ch;
            *pos += 1;
        }
    }
}

fn fmt_line(buf: &mut [u8], label: &[u8], val: usize) -> usize {
    let mut pos = 0;
    push(buf, &mut pos, label);
    push_usize(buf, &mut pos, val);
    push(buf, &mut pos, b"\n");
    pos
}
