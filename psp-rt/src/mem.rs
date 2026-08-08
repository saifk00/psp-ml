//! Tracked partition-memory allocation.
//!
//! Partition memory allocated with `sceKernelAllocPartitionMemory` is NOT
//! reclaimed when a module unloads under psplink — a leaked block survives
//! until the PSP reboots, and a few leaked runs exhaust the ~48 MiB pool
//! (empirically: two roofline runs leaked 24 MiB and made birdnet's 16 MiB
//! weight alloc fail with 0x800200D9). Allocate through this registry and the
//! `module!` epilogue frees everything after `app_main` returns *or panics*.
//!
//! Single-threaded by design (the `module!` main thread); not Sync.

#[cfg(target_os = "psp")]
mod imp {
    use psp::sys::{
        sceKernelAllocPartitionMemory, sceKernelFreePartitionMemory, sceKernelGetBlockHeadAddr,
        SceSysMemBlockTypes, SceUid,
    };

    const MAX_BLOCKS: usize = 16;
    static mut UIDS: [SceUid; MAX_BLOCKS] = [SceUid(-1); MAX_BLOCKS];
    static mut COUNT: usize = 0;

    /// Allocate `size` bytes from the primary user partition (partition 2),
    /// registered for automatic cleanup at module exit. `name` must be
    /// NUL-terminated. Returns null on failure (with the kernel error code in
    /// `err` if non-null).
    pub fn alloc_partition(name: &[u8], size: usize, err: Option<&mut u32>) -> *mut u8 {
        unsafe {
            if COUNT >= MAX_BLOCKS {
                if let Some(e) = err {
                    *e = 0xDEAD_0010; // registry full — raise MAX_BLOCKS
                }
                return core::ptr::null_mut();
            }
            let uid = sceKernelAllocPartitionMemory(
                core::mem::transmute(2i32),
                name.as_ptr(),
                SceSysMemBlockTypes::Low,
                size as u32,
                core::ptr::null_mut(),
            );
            if uid.0 < 0 {
                if let Some(e) = err {
                    *e = uid.0 as u32;
                }
                return core::ptr::null_mut();
            }
            UIDS[COUNT] = uid;
            COUNT += 1;
            sceKernelGetBlockHeadAddr(uid) as *mut u8
        }
    }

    /// Free every block allocated through `alloc_partition`. Called by the
    /// `module!` epilogue; safe to call multiple times.
    pub fn free_all() {
        unsafe {
            for i in 0..COUNT {
                sceKernelFreePartitionMemory(UIDS[i]);
            }
            COUNT = 0;
        }
    }
}

#[cfg(target_os = "psp")]
pub use imp::{alloc_partition, free_all};

/// Host stub: partition memory doesn't exist off-PSP.
#[cfg(not(target_os = "psp"))]
pub fn free_all() {}
