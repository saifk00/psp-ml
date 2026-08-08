//! Hardware profiling register snapshot.
//!
//! Requires the `psp_ml_kernel.prx` kernel plugin loaded via
//! `ms0:/seplugins/game.txt`. Import stubs are generated in pure Rust
//! using `#[link_section]` attributes — no external assembler needed.

/// Hardware profiling counters read from the PSP profiler MMIO at `0xBC400000`.
///
/// Each field is a 32-bit counter. The layout matches the kernel plugin's
/// `ProfileRegs` struct and the hardware register order.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ProfileRegs {
    pub enable: u32,
    pub systemck: u32,
    pub cpuck: u32,
    pub internal: u32,
    pub memory: u32,
    pub copz: u32,
    pub vfpu_stalls: u32,
    pub sleep: u32,
    pub bus_access: u32,
    pub uncached_load: u32,
    pub uncached_store: u32,
    pub cached_load: u32,
    pub cached_store: u32,
    pub i_miss: u32,
    pub d_miss: u32,
    pub d_writeback: u32,
    pub cop0_inst: u32,
    pub fpu_inst: u32,
    pub vfpu_inst: u32,
    pub local_bus: u32,
}

/// Per-op hardware profiling accumulators.
///
/// Each counter is widened to u64 to avoid u32 wrapping across multiple
/// inference invocations. Use `zero()` to initialize and `accumulate()`
/// to add a `ProfileRegs` snapshot (taken after clear → enable → kernel → disable → read).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct OpProfileStats {
    pub systemck: u64,
    pub cpuck: u64,
    pub internal: u64,
    pub memory: u64,
    pub copz: u64,
    pub vfpu_stalls: u64,
    pub sleep: u64,
    pub bus_access: u64,
    pub uncached_load: u64,
    pub uncached_store: u64,
    pub cached_load: u64,
    pub cached_store: u64,
    pub i_miss: u64,
    pub d_miss: u64,
    pub d_writeback: u64,
    pub cop0_inst: u64,
    pub fpu_inst: u64,
    pub vfpu_inst: u64,
    pub local_bus: u64,
}

impl OpProfileStats {
    pub const fn zero() -> Self {
        Self {
            systemck: 0, cpuck: 0, internal: 0, memory: 0, copz: 0,
            vfpu_stalls: 0, sleep: 0, bus_access: 0, uncached_load: 0,
            uncached_store: 0, cached_load: 0, cached_store: 0,
            i_miss: 0, d_miss: 0, d_writeback: 0, cop0_inst: 0,
            fpu_inst: 0, vfpu_inst: 0, local_bus: 0,
        }
    }

    /// Accumulate a hardware counter snapshot into this accumulator.
    pub fn accumulate(&mut self, regs: &ProfileRegs) {
        self.systemck += regs.systemck as u64;
        self.cpuck += regs.cpuck as u64;
        self.internal += regs.internal as u64;
        self.memory += regs.memory as u64;
        self.copz += regs.copz as u64;
        self.vfpu_stalls += regs.vfpu_stalls as u64;
        self.sleep += regs.sleep as u64;
        self.bus_access += regs.bus_access as u64;
        self.uncached_load += regs.uncached_load as u64;
        self.uncached_store += regs.uncached_store as u64;
        self.cached_load += regs.cached_load as u64;
        self.cached_store += regs.cached_store as u64;
        self.i_miss += regs.i_miss as u64;
        self.d_miss += regs.d_miss as u64;
        self.d_writeback += regs.d_writeback as u64;
        self.cop0_inst += regs.cop0_inst as u64;
        self.fpu_inst += regs.fpu_inst as u64;
        self.vfpu_inst += regs.vfpu_inst as u64;
        self.local_bus += regs.local_bus as u64;
    }
}

// --- Pure-Rust import stubs for psp_ml_kernel PRX ---
//
// Replicates the psp crate's `psp_extern!` pattern: structured data in
// specific ELF sections that the PSP loader patches into syscall
// trampolines at module load time.

#[cfg(target_os = "psp")]
mod stubs {
    /// 8-byte stub: two pointers the PSP loader overwrites with
    /// `jr $ra; syscall N` at load time.
    #[derive(Copy, Clone)]
    #[repr(C)]
    pub(super) struct Stub {
        pub(super) lib_addr: &'static psp::sys::SceStubLibraryEntry,
        pub(super) nid_addr: &'static u32,
    }

    unsafe impl Sync for Stub {}

    // Library name, null-terminated and padded to a multiple of 4 bytes.
    // "psp_ml_kernel" = 13 chars + 3 padding = 16 bytes.
    #[link_section = ".rodata.sceResident.psp_ml_kernel"]
    static LIB_NAME: [u8; 16] = *b"psp_ml_kernel\0\0\0";

    // Sentinel markers for the start of NID and stub tables.
    #[link_section = ".rodata.sceNid.psp_ml_kernel"]
    static NID_START: () = ();

    #[link_section = ".sceStub.text.psp_ml_kernel"]
    static STUB_START: () = ();

    // The library entry that ties everything together.
    #[link_section = ".lib.stub.entry.psp_ml_kernel"]
    pub(super) static LIB_ENTRY: psp::sys::SceStubLibraryEntry =
        psp::sys::SceStubLibraryEntry {
            name: &LIB_NAME[0],
            version: [0, 0],
            flags: 0x4009,
            len: 5,
            v_stub_count: 0,
            stub_count: 0, // patched by cargo-psp ELF fixup
            nid_table: &NID_START as *const () as *const u32,
            stub_table: &STUB_START as *const () as *const _,
        };

    // --- NIDs ---
    #[link_section = ".rodata.sceNid.psp_ml_kernel.ProfileEnable"]
    pub(super) static PROFILE_ENABLE_NID: u32 = 0x29B089BC;

    #[link_section = ".rodata.sceNid.psp_ml_kernel.ProfileDisable"]
    pub(super) static PROFILE_DISABLE_NID: u32 = 0x739CC27B;

    #[link_section = ".rodata.sceNid.psp_ml_kernel.ProfileClear"]
    pub(super) static PROFILE_CLEAR_NID: u32 = 0x9A983384;

    #[link_section = ".rodata.sceNid.psp_ml_kernel.ProfileGetRegs"]
    pub(super) static PROFILE_GET_REGS_NID: u32 = 0x84ABB8F1;

    // --- Stubs (8 bytes each, patched to syscall trampolines at load) ---
    #[link_section = ".sceStub.text.psp_ml_kernel.ProfileEnable"]
    #[no_mangle]
    pub(super) static __ProfileEnable_stub: Stub = Stub {
        lib_addr: &LIB_ENTRY,
        nid_addr: &PROFILE_ENABLE_NID,
    };

    #[link_section = ".sceStub.text.psp_ml_kernel.ProfileDisable"]
    #[no_mangle]
    pub(super) static __ProfileDisable_stub: Stub = Stub {
        lib_addr: &LIB_ENTRY,
        nid_addr: &PROFILE_DISABLE_NID,
    };

    #[link_section = ".sceStub.text.psp_ml_kernel.ProfileClear"]
    #[no_mangle]
    pub(super) static __ProfileClear_stub: Stub = Stub {
        lib_addr: &LIB_ENTRY,
        nid_addr: &PROFILE_CLEAR_NID,
    };

    #[link_section = ".sceStub.text.psp_ml_kernel.ProfileGetRegs"]
    #[no_mangle]
    pub(super) static __ProfileGetRegs_stub: Stub = Stub {
        lib_addr: &LIB_ENTRY,
        nid_addr: &PROFILE_GET_REGS_NID,
    };
}

// Public wrapper functions. The extern "C" declarations inside each
// function body resolve to the #[no_mangle] stubs above — this is
// how the psp crate's psp_extern! macro works.

#[cfg(target_os = "psp")]
#[allow(non_snake_case)]
pub unsafe fn ProfileEnable() {
    extern "C" {
        fn __ProfileEnable_stub();
    }
    unsafe { __ProfileEnable_stub() }
}

#[cfg(target_os = "psp")]
#[allow(non_snake_case)]
pub unsafe fn ProfileDisable() {
    extern "C" {
        fn __ProfileDisable_stub();
    }
    unsafe { __ProfileDisable_stub() }
}

#[cfg(target_os = "psp")]
#[allow(non_snake_case)]
pub unsafe fn ProfileClear() {
    extern "C" {
        fn __ProfileClear_stub();
    }
    unsafe { __ProfileClear_stub() }
}

#[cfg(target_os = "psp")]
#[allow(non_snake_case)]
pub unsafe fn ProfileGetRegs(regs: *mut ProfileRegs) {
    extern "C" {
        fn __ProfileGetRegs_stub(regs: *mut ProfileRegs);
    }
    unsafe { __ProfileGetRegs_stub(regs) }
}
