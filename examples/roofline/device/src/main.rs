//! Roofline microbenchmarks for the PSP's main CPU.
//!
//! Measures, on real hardware:
//!   - memory bandwidth: sequential read/write/copy through the cache, through the
//!     uncached alias (`ptr | 0x4000_0000`, see psp_doc 4.10.3), per-cache-line
//!     stride (line-fill bandwidth), and dependent-load latency (pointer chase)
//!   - compute peaks: VFPU `vmmul.q`/`vadd.q` throughput vs scalar FPU mul+add
//!   - hostfs: `sceIoRead` throughput from `host0:` at several chunk sizes, which
//!     bounds any weight-streaming scheme
//!
//! Results print to stdout and land in `host0:/roofline.json`. The host runner
//! (`roofline-host`) provides `host0:/bwtest.bin` for the hostfs tests.
//!
//! Timing uses `sceRtcGetCurrentTick` (1 us resolution), so every test loops
//! long enough to make the timer error negligible (>= tens of ms).

#![no_std]
#![no_main]
#![feature(asm_experimental_arch)]

use core::ffi::c_void;
use psp::sys::{
    sceIoClose, sceIoOpen, sceIoRead, sceIoWrite, sceKernelDcacheWritebackInvalidateAll,
    scePowerGetBusClockFrequencyInt, scePowerGetCpuClockFrequencyInt, scePowerSetClockFrequency,
    sceRtcGetCurrentTick, sceRtcGetTickResolution, IoOpenFlags,
};
use psp::vfpu_asm;
use psp_rt::dprintln;

psp_rt::module!("roofline", 1, 0);

/// Big buffer for memory tests: far larger than the 32 KB L1 D-cache.
const BUF_BYTES: usize = 16 * 1024 * 1024;
/// Small buffer that fits comfortably in L1, for cache-hit bandwidth.
const L1_BYTES: usize = 16 * 1024;
const CACHE_LINE: usize = 64;

fn tick() -> u64 {
    let mut t = 0u64;
    unsafe { sceRtcGetCurrentTick(&mut t) };
    t
}

struct Json {
    buf: [u8; 8192],
    pos: usize,
}

impl Json {
    fn new() -> Self {
        Json { buf: [0; 8192], pos: 0 }
    }
    fn push(&mut self, s: &[u8]) {
        for &b in s {
            if self.pos < self.buf.len() {
                self.buf[self.pos] = b;
                self.pos += 1;
            }
        }
    }
    fn push_u64(&mut self, mut v: u64) {
        if v == 0 {
            self.push(b"0");
            return;
        }
        let mut tmp = [0u8; 20];
        let mut n = 0;
        while v > 0 {
            tmp[n] = b'0' + (v % 10) as u8;
            v /= 10;
            n += 1;
        }
        for i in (0..n).rev() {
            self.push(&[tmp[i]]);
        }
    }
}

/// Record one measurement: prints it and appends a JSON object.
/// `bytes`/`flops`: whichever is 0 is omitted. Rates are computed host-side.
fn record(json: &mut Json, first: &mut bool, group: &str, name: &str, bytes: u64, flops: u64, us: u64) {
    if us > 0 {
        if bytes > 0 {
            // MB/s = bytes / us (both carry a factor of ~1e6)
            let mbps_x100 = bytes * 100 / us;
            dprintln!(
                "  {:<28} {:>9} us   {:>6}.{:02} MB/s",
                name,
                us,
                mbps_x100 / 100,
                mbps_x100 % 100
            );
        } else {
            let mflops = flops / us;
            dprintln!("  {:<28} {:>9} us   {:>6} MFLOP/s", name, us, mflops);
        }
    }
    if !*first {
        json.push(b",\n");
    }
    *first = false;
    json.push(b"  {\"group\":\"");
    json.push(group.as_bytes());
    json.push(b"\",\"name\":\"");
    json.push(name.as_bytes());
    json.push(b"\",\"bytes\":");
    json.push_u64(bytes);
    json.push(b",\"flops\":");
    json.push_u64(flops);
    json.push(b",\"us\":");
    json.push_u64(us);
    json.push(b"}");
}

/// Allocation goes through `psp_rt::mem`, whose registry the `module!`
/// epilogue drains at exit — partition memory is NOT reclaimed on module
/// unload under psplink, and leaked benchmark buffers once ate 24 MiB of the
/// pool and broke the birdnet weight alloc.
fn alloc_bytes(len: usize) -> *mut u8 {
    let mut err = 0u32;
    let raw = psp_rt::mem::alloc_partition(b"roofline\0", len + CACHE_LINE, Some(&mut err));
    assert!(!raw.is_null(), "alloc failed: 0x{err:08X}");
    (((raw as usize) + CACHE_LINE - 1) & !(CACHE_LINE - 1)) as *mut u8
}

/// Uncached alias of a cached user pointer (KU0 -> KU1 segment).
fn uncached<T>(p: *mut T) -> *mut T {
    ((p as usize) | 0x4000_0000) as *mut T
}

// ============================================================================
// Memory kernels (all #[inline(never)] so the timed region is honest)
// ============================================================================

#[inline(never)]
fn read_seq_u32(p: *const u32, words: usize) -> u32 {
    let (mut a, mut b, mut c, mut d) = (0u32, 0u32, 0u32, 0u32);
    let mut i = 0;
    unsafe {
        while i + 4 <= words {
            a = a.wrapping_add(p.add(i).read_volatile());
            b = b.wrapping_add(p.add(i + 1).read_volatile());
            c = c.wrapping_add(p.add(i + 2).read_volatile());
            d = d.wrapping_add(p.add(i + 3).read_volatile());
            i += 4;
        }
    }
    a.wrapping_add(b).wrapping_add(c).wrapping_add(d)
}

#[inline(never)]
fn write_seq_u32(p: *mut u32, words: usize, val: u32) {
    let mut i = 0;
    unsafe {
        while i + 4 <= words {
            p.add(i).write_volatile(val);
            p.add(i + 1).write_volatile(val);
            p.add(i + 2).write_volatile(val);
            p.add(i + 3).write_volatile(val);
            i += 4;
        }
    }
}

#[inline(never)]
fn copy_u32(dst: *mut u32, src: *const u32, words: usize) {
    unsafe { core::ptr::copy_nonoverlapping(src, dst, words) }
}

/// One u32 per cache line: sequential-miss (line fill) bandwidth.
#[inline(never)]
fn read_stride_line(p: *const u32, words: usize) -> u32 {
    let step = CACHE_LINE / 4;
    let mut acc = 0u32;
    let mut i = 0;
    unsafe {
        while i < words {
            acc = acc.wrapping_add(p.add(i).read_volatile());
            i += step;
        }
    }
    acc
}

/// VFPU quad-load read bandwidth: how fast `lv.q` can pull from memory.
/// This is the load path the matmul kernels actually use.
#[inline(never)]
fn read_lvq(p: *const u8, bytes: usize) {
    let mut ptr = p as usize;
    let end = ptr + bytes;
    unsafe {
        while ptr + 64 <= end {
            vfpu_asm!(
                "lv.q R000,  0({0})",
                "lv.q R001, 16({0})",
                "lv.q R002, 32({0})",
                "lv.q R003, 48({0})",
                in(reg) ptr,
                options(nostack),
            );
            ptr += 64;
        }
    }
}

/// Dependent-load latency: pointer chase over `words` u32 slots, one hop per
/// cache line, order scrambled with a co-prime stride so each hop misses.
#[inline(never)]
fn chase(p: *const u32, hops: usize) -> u32 {
    let mut cur = 0u32;
    for _ in 0..hops {
        cur = unsafe { p.add(cur as usize).read_volatile() };
    }
    cur
}

fn build_chain(p: *mut u32, words: usize) -> usize {
    // Visit lines in a scrambled cyclic order: index_{k+1} = (index_k + STEP) mod n_lines,
    // STEP co-prime to n_lines gives a full cycle that defeats simple prefetch.
    let lines = words / (CACHE_LINE / 4);
    let mut step = lines / 2 + 37;
    while gcd(step, lines) != 1 {
        step += 1;
    }
    let mut cur = 0usize;
    for _ in 0..lines {
        let next = (cur + step) % lines;
        unsafe { p.add(cur * (CACHE_LINE / 4)).write_volatile((next * (CACHE_LINE / 4)) as u32) };
        cur = next;
    }
    lines
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// ============================================================================
// Compute kernels
// ============================================================================

/// VFPU 4x4 matmul throughput, registers only. One vmmul.q = 64 mul + 48 add
/// = 112 FLOPs. Four independent destinations to expose pipelining.
#[inline(never)]
fn vfpu_mmul_regs(iters: usize) {
    unsafe {
        vfpu_asm!("vmidt.q M000", "vmidt.q M100", options(nostack));
    }
    for _ in 0..iters {
        unsafe {
            vfpu_asm!(
                "vmmul.q M400, M000, E100",
                "vmmul.q M500, M000, E100",
                "vmmul.q M600, M000, E100",
                "vmmul.q M700, M000, E100",
                options(nostack),
            );
        }
    }
}

/// VFPU quad add throughput: 4 FLOPs per vadd.q, four independent chains.
#[inline(never)]
fn vfpu_vadd_regs(iters: usize) {
    unsafe {
        vfpu_asm!("vmidt.q M000", "vmidt.q M400", options(nostack));
    }
    for _ in 0..iters {
        unsafe {
            vfpu_asm!(
                "vadd.q R400, R400, R000",
                "vadd.q R401, R401, R001",
                "vadd.q R402, R402, R002",
                "vadd.q R403, R403, R003",
                options(nostack),
            );
        }
    }
}

/// Scalar FPU mul+add with 8 independent accumulator chains (enough to cover
/// the FPU's mul/add latency on a single-issue pipeline).
#[inline(never)]
fn scalar_fma(iters: usize) -> f32 {
    let mut acc = [1.0f32, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7];
    let (x, y) = (
        core::hint::black_box(1.000001f32),
        core::hint::black_box(0.0000001f32),
    );
    for _ in 0..iters {
        acc[0] = acc[0] * x + y;
        acc[1] = acc[1] * x + y;
        acc[2] = acc[2] * x + y;
        acc[3] = acc[3] * x + y;
        acc[4] = acc[4] * x + y;
        acc[5] = acc[5] * x + y;
        acc[6] = acc[6] * x + y;
        acc[7] = acc[7] * x + y;
    }
    acc.iter().sum()
}

// ============================================================================

fn app_main() {
    psp::enable_home_button();

    let tick_res = unsafe { sceRtcGetTickResolution() } as u64;
    assert!(tick_res == 1_000_000, "unexpected tick resolution");

    dprintln!("=== PSP Roofline ===");

    // --- Clock: report, force to max, report again ---
    let (cpu0, bus0) = unsafe {
        (scePowerGetCpuClockFrequencyInt(), scePowerGetBusClockFrequencyInt())
    };
    let set_ret = unsafe { scePowerSetClockFrequency(333, 333, 166) };
    let (cpu1, bus1) = unsafe {
        (scePowerGetCpuClockFrequencyInt(), scePowerGetBusClockFrequencyInt())
    };
    dprintln!(
        "clock: was {}/{} MHz, set(333,333,166) -> {} (now {}/{} MHz)",
        cpu0, bus0, set_ret, cpu1, bus1
    );

    let mut json = Json::new();
    let mut first = true;
    json.push(b"{\n \"clock\":{\"cpu_before\":");
    json.push_u64(cpu0 as u64);
    json.push(b",\"bus_before\":");
    json.push_u64(bus0 as u64);
    json.push(b",\"cpu_after\":");
    json.push_u64(cpu1 as u64);
    json.push(b",\"bus_after\":");
    json.push_u64(bus1 as u64);
    json.push(b"},\n \"results\":[\n");

    let buf = alloc_bytes(BUF_BYTES);
    let buf2 = alloc_bytes(BUF_BYTES / 2);
    let words = BUF_BYTES / 4;

    // Touch everything once so the working set exists and page state is settled.
    write_seq_u32(buf as *mut u32, words, 0x0101_0101);
    write_seq_u32(buf2 as *mut u32, BUF_BYTES / 8, 0x0202_0202);

    dprintln!("--- memory bandwidth ({} MiB buffer) ---", BUF_BYTES / 1024 / 1024);

    // Cached sequential read: buffer >> L1, so this is sustained miss bandwidth.
    let t0 = tick();
    let mut sink = 0u32;
    for _ in 0..4 {
        sink = sink.wrapping_add(read_seq_u32(buf as *const u32, words));
    }
    record(&mut json, &mut first, "membw", "read_seq_cached", 4 * BUF_BYTES as u64, 0, tick() - t0);

    // L1-resident read for contrast (cache-hit bandwidth).
    let l1_words = L1_BYTES / 4;
    read_seq_u32(buf as *const u32, l1_words);
    let t0 = tick();
    for _ in 0..4096 {
        sink = sink.wrapping_add(read_seq_u32(buf as *const u32, l1_words));
    }
    record(&mut json, &mut first, "membw", "read_seq_l1", 4096 * L1_BYTES as u64, 0, tick() - t0);

    // Cached sequential write (write-allocate: each miss also fills the line).
    let t0 = tick();
    for _ in 0..4 {
        write_seq_u32(buf as *mut u32, words, 0xDEAD_BEEF);
    }
    record(&mut json, &mut first, "membw", "write_seq_cached", 4 * BUF_BYTES as u64, 0, tick() - t0);

    // memcpy 8 MiB -> 8 MiB (bytes = amount copied; bus traffic is ~3x).
    let t0 = tick();
    for _ in 0..4 {
        copy_u32(buf2 as *mut u32, buf as *const u32, BUF_BYTES / 8);
    }
    record(&mut json, &mut first, "membw", "memcpy", 4 * (BUF_BYTES / 2) as u64, 0, tick() - t0);

    // One word per line: pure line-fill bandwidth (bytes counted as full lines).
    let t0 = tick();
    for _ in 0..4 {
        sink = sink.wrapping_add(read_stride_line(buf as *const u32, words));
    }
    record(&mut json, &mut first, "membw", "read_stride64_cached", 4 * BUF_BYTES as u64, 0, tick() - t0);

    // VFPU lv.q streaming read (the matmul load path).
    let t0 = tick();
    for _ in 0..4 {
        read_lvq(buf, BUF_BYTES);
    }
    record(&mut json, &mut first, "membw", "read_lvq_cached", 4 * BUF_BYTES as u64, 0, tick() - t0);

    // Uncached alias: every access goes straight to DRAM.
    unsafe { sceKernelDcacheWritebackInvalidateAll() };
    let ubuf = uncached(buf);
    let t0 = tick();
    sink = sink.wrapping_add(read_seq_u32(ubuf as *const u32, words));
    record(&mut json, &mut first, "membw", "read_seq_uncached", BUF_BYTES as u64, 0, tick() - t0);

    let t0 = tick();
    write_seq_u32(ubuf as *mut u32, words, 0xCAFE_F00D);
    record(&mut json, &mut first, "membw", "write_seq_uncached", BUF_BYTES as u64, 0, tick() - t0);

    // Load-to-use latency via pointer chase (report ns/hop host-side: us*1000/hops).
    unsafe { sceKernelDcacheWritebackInvalidateAll() };
    let lines = build_chain(buf as *mut u32, words);
    unsafe { sceKernelDcacheWritebackInvalidateAll() };
    let hops = 4 * lines;
    let t0 = tick();
    sink = sink.wrapping_add(chase(buf as *const u32, hops));
    record(&mut json, &mut first, "latency", "chase_dram", hops as u64, 0, tick() - t0);

    dprintln!("--- compute peaks ---");

    // VFPU matmul: 112 FLOPs per vmmul.q.
    let iters = 400_000usize;
    let t0 = tick();
    vfpu_mmul_regs(iters);
    record(&mut json, &mut first, "compute", "vfpu_vmmul_q", 0, (iters as u64) * 4 * 112, tick() - t0);

    let iters = 2_000_000usize;
    let t0 = tick();
    vfpu_vadd_regs(iters);
    record(&mut json, &mut first, "compute", "vfpu_vadd_q", 0, (iters as u64) * 4 * 4, tick() - t0);

    let iters = 2_000_000usize;
    let t0 = tick();
    let s = scalar_fma(iters);
    record(&mut json, &mut first, "compute", "scalar_mul_add", 0, (iters as u64) * 8 * 2, tick() - t0);
    sink = sink.wrapping_add(s as u32);

    dprintln!("--- hostfs (host0:) throughput ---");

    // Read host0:/bwtest.bin (provided by the host runner) at several chunk sizes.
    for &chunk in &[4096usize, 16384, 65536, 262144] {
        let fd = unsafe {
            sceIoOpen(b"host0:/bwtest.bin\0".as_ptr(), IoOpenFlags::RD_ONLY, 0)
        };
        if fd.0 < 0 {
            dprintln!("  bwtest.bin missing (0x{:08X}), skipping hostfs read", fd.0 as u32);
            break;
        }
        let mut total = 0u64;
        let t0 = tick();
        loop {
            let n = unsafe { sceIoRead(fd, buf as *mut c_void, chunk as u32) };
            if n <= 0 {
                break;
            }
            total += n as u64;
        }
        let us = tick() - t0;
        unsafe { sceIoClose(fd) };
        let name: &str = match chunk {
            4096 => "hostfs_read_4k",
            16384 => "hostfs_read_16k",
            65536 => "hostfs_read_64k",
            _ => "hostfs_read_256k",
        };
        record(&mut json, &mut first, "hostfs", name, total, 0, us);
    }

    // Write throughput at 64 KiB (the generated weight loader's chunk size).
    let fd = unsafe {
        sceIoOpen(
            b"host0:/bwout.bin\0".as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o644,
        )
    };
    if fd.0 >= 0 {
        let total = 4 * 1024 * 1024usize;
        let t0 = tick();
        let mut written = 0usize;
        while written < total {
            let n = unsafe { sceIoWrite(fd, buf as *const c_void, 65536) };
            if n <= 0 {
                break;
            }
            written += n as usize;
        }
        let us = tick() - t0;
        unsafe { sceIoClose(fd) };
        record(&mut json, &mut first, "hostfs", "hostfs_write_64k", written as u64, 0, us);
    }

    json.push(b"\n ],\n \"sink\":");
    json.push_u64(sink as u64);
    json.push(b"\n}\n");

    let fd = unsafe {
        sceIoOpen(
            b"host0:/roofline.json\0".as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o644,
        )
    };
    if fd.0 >= 0 {
        unsafe {
            sceIoWrite(fd, json.buf.as_ptr() as *const c_void, json.pos);
            sceIoClose(fd);
        }
        dprintln!("Wrote roofline.json");
    }
    // Partition blocks are freed by the module! epilogue via psp_rt::mem.
}
