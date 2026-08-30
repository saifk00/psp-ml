//! Device-only checks for the partition-memory registry.
//!
//! There is nothing to run on the host here: `sceKernelAllocPartitionMemory`
//! only exists on the PSP, and the behaviour worth checking — that the
//! registry hands back usable memory and that `free_all` really returns it —
//! is precisely the behaviour the host stub cannot have. So every check lives
//! in `device_checks!`'s `device:` group and only `cargo test -p device-tests`
//! ever runs it.

use crate::device_test::device_checks;

/// One block, big enough that a failure to reclaim it would show up in the
/// next allocation rather than being absorbed by allocator slack.
#[cfg(target_os = "psp")]
const BLOCK: usize = 1 << 20;

/// A block from `alloc_partition` must be usable and must come back after
/// `free_all` — the leak that used to make `reset` a routine chore.
///
/// Allocating the same size twice in a row only succeeds if the first block
/// was genuinely returned to partition 2, so the second allocation is the
/// assertion. It also pins the alignment the VFPU needs: `lv.q` on a
/// misaligned address is a CPU fault, not a Rust panic, and would take psplink
/// down with it.
#[cfg(target_os = "psp")]
fn test_partition_alloc_reclaimed() -> bool {
    // Nothing else has allocated at this point in the run, so `free_all` below
    // is only undoing this check's own work.
    let p = crate::mem::alloc_partition(b"psptest_a\0", BLOCK, None);
    if p.is_null() || (p as usize) % 16 != 0 {
        return false;
    }

    // Prove the whole span is backed, not just the head address.
    let probes = [0usize, BLOCK / 2, BLOCK - 4];
    for (i, &off) in probes.iter().enumerate() {
        unsafe { core::ptr::write_volatile(p.add(off) as *mut u32, 0xA5A5_0000 | i as u32) };
    }
    for (i, &off) in probes.iter().enumerate() {
        let got = unsafe { core::ptr::read_volatile(p.add(off) as *const u32) };
        if got != (0xA5A5_0000 | i as u32) {
            return false;
        }
    }

    crate::mem::free_all();

    let q = crate::mem::alloc_partition(b"psptest_b\0", BLOCK, None);
    let reclaimed = !q.is_null();
    crate::mem::free_all();
    reclaimed
}

/// The registry must refuse an over-full run rather than losing track of a
/// block, and `free_all` must reset its count.
///
/// An untracked block is the failure mode with teeth: it survives the module
/// unload and starves every later run until the PSP reboots.
#[cfg(target_os = "psp")]
fn test_partition_registry_bounds() -> bool {
    use crate::mem::{ERR_REGISTRY_FULL, MAX_BLOCKS};
    const SMALL: usize = 256;

    for _ in 0..MAX_BLOCKS {
        if crate::mem::alloc_partition(b"psptest_f\0", SMALL, None).is_null() {
            crate::mem::free_all();
            return false;
        }
    }

    let mut err = 0u32;
    let overflow = crate::mem::alloc_partition(b"psptest_f\0", SMALL, Some(&mut err));
    crate::mem::free_all();
    if !overflow.is_null() || err != ERR_REGISTRY_FULL {
        return false;
    }

    // `free_all` reset the count, so the registry accepts blocks again.
    let after = crate::mem::alloc_partition(b"psptest_g\0", SMALL, None);
    let ok = !after.is_null();
    crate::mem::free_all();
    ok
}

device_checks! {
    shared: [],
    device: [
        test_partition_alloc_reclaimed,
        test_partition_registry_bounds,
    ],
}
