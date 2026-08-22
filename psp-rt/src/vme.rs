//! Pure-Rust bindings for the `psp_vme_kernel` plugin.
//!
//! The plugin (`kernel-plugin-vme/`) boots the Media Engine and exposes a small
//! job interface for driving the Virtual Mobile Engine (VME2) — the integer
//! SIMD the VFPU lacks. Install it once with the manual `pspsh` copy (or
//! `vme-install-host`) and power-cycle; then these syscalls are available.
//!
//! Import stubs are generated in pure Rust with `#[link_section]` attributes,
//! exactly like [`crate::profiler`] — no external assembler, no `.a`.
//!
//! This is original Rust, but the VME register layout, opcodes, and magic
//! values it encodes come from mcidclan's reverse engineering (MIT):
//! <https://github.com/mcidclan/psp-media-engine-custom-core> and
//! <https://github.com/mcidclan/psp-virtual-mobile-engine-ext>.
//!
//! Layer split: this crate authors the VME datapath *program* (the 112-word
//! context, via [`vme_asm!`](crate::vme_asm)) and invokes it; the plugin does
//! the privileged ME boot and the ME-side run. See `kernel-plugin-vme/main.cpp`
//! for the `VmeJob` contract mirrored below.

/// One ring-buffer load performed before a run.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct VmeStage {
    /// First word in [`VmeJob::data`] to copy from.
    pub src_word: u32,
    /// Destination ring-buffer word offset (a `VME_*_BUFFn_WOFF` constant).
    pub ring_woff: u32,
    /// Number of words to copy.
    pub count: u32,
}

/// Number of full VME datapath registers in one context.
pub const VME_CTX_WORDS: usize = 112;
/// Maximum ring-buffer loads staged before a run.
pub const VME_MAX_STAGE: usize = 4;
/// Words of input scratch referenced by [`VmeStage::src_word`].
pub const VME_DATA_WORDS: usize = 512;

/// Shared job block — must match `VmeJob` in `kernel-plugin-vme/main.cpp`.
///
/// Lives in uncached user memory (address from [`shared_addr`]); the main CPU
/// fills it, bumps `seq` via [`run`], and the ME writes `result`. Access every
/// field through volatile reads/writes ([`Job`] wraps that).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct VmeJob {
    /// Main CPU bumps this to request a run.
    pub seq: u32,
    /// ME sets this equal to `seq` when a run completes.
    pub ack: u32,
    /// Readback value produced by the last run.
    pub result: u32,
    /// Number of populated [`VmeJob::stage`] entries.
    pub n_stage: u32,
    /// Ring-buffer loads applied before the run.
    pub stage: [VmeStage; VME_MAX_STAGE],
    /// Absolute ring base to read the result from (e.g. `0x4400_0000`).
    pub readback_base: u32,
    /// Word offset within that ring.
    pub readback_woff: u32,
    /// The datapath program (one full context) — [`vme_asm!`] fills this.
    pub ctx: [u32; VME_CTX_WORDS],
    /// Input words referenced by the stages.
    pub data: [u32; VME_DATA_WORDS],
    /// Appended in plugin v1.1: run a full 1 MB machine image (vme-emu /
    /// vme-assembler format) instead of the stage/readback fields. The ME
    /// stages all eight ring buffers from the image, loads the context from
    /// its mapped offset (0xF8000), runs, and reads every buffer back into
    /// the image.
    pub image_mode: u32,
    /// Uncached-user address of the machine image when `image_mode` is set.
    pub image_addr: u32,
}

/// Encoders for the VME datapath — the "assembler" behind [`vme_asm!`].
///
/// A VME "program" is a 112-word datapath context: interconnect wiring, one
/// functional-unit descriptor per processing element, and per-PE address
/// generators. Unlike a sequential ISA, order does not matter — each entry is
/// one register. So [`vme_asm!`](crate::vme_asm) is declarative: `index => value`,
/// where these `const fn`s name the register indices and encode the values.
/// Extending it is additive: a new op is one value encoder; a new register is
/// one index `const fn`. Only what the integer MAC needs is encoded so far;
/// the full opcode set is in `kernel-plugin-vme/vendor/vme-opcodes.h`.
///
/// ```ignore
/// use psp_rt::vme::asm::*;
/// let ctx = psp_rt::vme_asm![
///     icn(AGU_BASE)          => 0x4440,
///     pe_fu_primary(0)       => fu(MAC_INNER_PRODUCT_BIAS, FRONT_TOP_0, BACK_BASE_0, 0),
///     pe_read_top_mode(0)    => def_mode(),
///     pe_read_top_count(0)   => step(n),
///     pe_read_base_mode(0)   => def_mode() | gap(256),
///     pe_read_base_count(0)  => step(n),
///     pe_write_mode(0)       => def_mode() | cycle(9),
///     pe_write_count(0)      => step(n),
/// ];
/// ```
pub mod asm {
    // --- register indices (word offsets into the 112-word context) ---------
    // Layout from kernel-plugin-vme/vendor/vme-lib.h.

    /// Interconnect field selector, for [`icn`].
    #[derive(Clone, Copy)]
    pub enum Icn {
        /// Maps source buffers to PE inputs.
        AguTop = 28,
        /// Inter-PE flow routing / BASE buffer mapping.
        AguBase = 29,
        /// Write routing.
        AguWrite = 30,
    }
    pub use Icn::{AguBase as AGU_BASE, AguTop as AGU_TOP, AguWrite as AGU_WRITE};

    /// Interconnect register index.
    pub const fn icn(field: Icn) -> usize {
        field as usize
    }

    // Per-PE register block: PE0 primary FU is index `pe`; the AGU block starts
    // at 33 and strides 18 words per PE.
    const PE_AGU_BASE: usize = 33;
    const PE_AGU_STRIDE: usize = 18;

    /// PE `pe`'s primary functional-unit descriptor.
    pub const fn pe_fu_primary(pe: usize) -> usize {
        pe
    }
    /// PE `pe`'s primary-FU constant register B (bias).
    pub const fn pe_fu_primary_b(pe: usize) -> usize {
        9 + pe * 2
    }
    /// PE `pe`'s "read TOP" address-generator mode.
    pub const fn pe_read_top_mode(pe: usize) -> usize {
        PE_AGU_BASE + pe * PE_AGU_STRIDE
    }
    /// PE `pe`'s "read TOP" count.
    pub const fn pe_read_top_count(pe: usize) -> usize {
        PE_AGU_BASE + 1 + pe * PE_AGU_STRIDE
    }
    /// PE `pe`'s "read BASE" address-generator mode.
    pub const fn pe_read_base_mode(pe: usize) -> usize {
        PE_AGU_BASE + 6 + pe * PE_AGU_STRIDE
    }
    /// PE `pe`'s "read BASE" count.
    pub const fn pe_read_base_count(pe: usize) -> usize {
        PE_AGU_BASE + 7 + pe * PE_AGU_STRIDE
    }
    /// PE `pe`'s "write" address-generator mode.
    pub const fn pe_write_mode(pe: usize) -> usize {
        PE_AGU_BASE + 12 + pe * PE_AGU_STRIDE
    }
    /// PE `pe`'s "write" count.
    pub const fn pe_write_count(pe: usize) -> usize {
        PE_AGU_BASE + 13 + pe * PE_AGU_STRIDE
    }

    // --- functional-unit opcodes (subset; see vme-opcodes.h) ---------------

    /// `(Σₙ back[n]·front[n] + b) >> k` — the SIMD inner-product MAC.
    pub const MAC_INNER_PRODUCT_BIAS: u32 = 0x0024_0000;
    /// `(back[n] · front[n]) >> k`.
    pub const MUL_RSHIFT: u32 = 0x0020_0000;
    /// `(back[n] + front[n]) >> k`.
    pub const ADD_IBUF_RSHIFT: u32 = 0x0001_0000;

    // --- operand MUX selectors (front = the [n] stream, back = accumuland) --

    /// Front-operand = TOP ring buffer 0.
    pub const FRONT_TOP_0: u32 = 0x0000_0000;
    /// Front-operand = TOP ring buffer 1.
    pub const FRONT_TOP_1: u32 = 0x1000_0000;
    /// Front-operand = TOP ring buffer 2.
    pub const FRONT_TOP_2: u32 = 0x2000_0000;
    /// Front-operand = TOP ring buffer 3.
    pub const FRONT_TOP_3: u32 = 0x3000_0000;
    /// Back-operand = BASE ring buffer 0.
    pub const BACK_BASE_0: u32 = 0x0200_0000;
    /// Back-operand = BASE ring buffer 1.
    pub const BACK_BASE_1: u32 = 0x0280_0000;
    /// Back-operand = TOP ring buffer 0.
    pub const BACK_TOP_0: u32 = 0x0000_0000;

    /// Encode a primary functional-unit descriptor: opcode + operand routing +
    /// right-shift `k`.
    pub const fn fu(op: u32, front: u32, back: u32, shift: u32) -> u32 {
        op | front | back | (shift & 0x3f)
    }

    // --- address-generator value fields ------------------------------------

    /// Default AGU mode word.
    pub const fn def_mode() -> u32 {
        0x8400_0000
    }
    /// Step-1 count word carrying `n` (elements to stream).
    pub const fn step(n: u32) -> u32 {
        0x0001_0000 | n
    }
    /// Buffer word offset to OR into a read/write mode (e.g. an input gap).
    pub const fn gap(words: u32) -> u32 {
        words
    }
    /// Pipeline cycle skew to OR into a write mode.
    pub const fn cycle(n: u32) -> u32 {
        n << 16
    }
}

/// Build a 112-word VME datapath context declaratively: `index => value`,
/// using the [`asm`] encoders. See [`asm`] for the grammar and an example.
///
/// This is the VME analogue of `vfpu_asm!`: where that emits one VFPU
/// instruction per line, this sets one datapath register per line. Order is
/// irrelevant (each line is an independent register); unset words are 0.
#[macro_export]
macro_rules! vme_asm {
    ($($idx:expr => $val:expr),* $(,)?) => {{
        let mut __ctx = [0u32; $crate::vme::VME_CTX_WORDS];
        $( __ctx[$idx] = $val; )*
        __ctx
    }};
}

/// Absolute ring-buffer bases and word offsets (from `vme-ext.h`).
pub mod rings {
    /// BASE ring buffers base address (also where results are read).
    pub const BASE_BUFFERS: u32 = 0x4400_0000;
    /// TOP ring buffers base address.
    pub const TOP_BUFFERS: u32 = 0x4402_0000;

    /// Word offset of TOP ring buffer `n` (n = 0..3).
    pub const fn top_woff(n: u32) -> u32 {
        0x8000 + 0x800 * n
    }
    /// Word offset of BASE ring buffer `n` (n = 0..3).
    pub const fn base_woff(n: u32) -> u32 {
        0x800 * n
    }
}

// --- Pure-Rust import stubs for the psp_vme_kernel PRX ---

#[cfg(target_os = "psp")]
mod stubs {
    #[derive(Copy, Clone)]
    #[repr(C)]
    pub(super) struct Stub {
        pub(super) lib_addr: &'static psp::sys::SceStubLibraryEntry,
        pub(super) nid_addr: &'static u32,
    }
    unsafe impl Sync for Stub {}

    // "psp_vme_kernel" = 14 chars + 2 padding = 16 bytes.
    #[link_section = ".rodata.sceResident.psp_vme_kernel"]
    static LIB_NAME: [u8; 16] = *b"psp_vme_kernel\0\0";

    #[link_section = ".rodata.sceNid.psp_vme_kernel"]
    static NID_START: () = ();

    #[link_section = ".sceStub.text.psp_vme_kernel"]
    static STUB_START: () = ();

    #[link_section = ".lib.stub.entry.psp_vme_kernel"]
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

    // NIDs = SHA-1(name)[..4] little-endian (verified against ProfileEnable).
    #[link_section = ".rodata.sceNid.psp_vme_kernel.VmeInit"]
    pub(super) static VME_INIT_NID: u32 = 0xD040_2895;
    #[link_section = ".rodata.sceNid.psp_vme_kernel.VmeSharedAddr"]
    pub(super) static VME_SHARED_ADDR_NID: u32 = 0x5562_A274;
    #[link_section = ".rodata.sceNid.psp_vme_kernel.VmeRun"]
    pub(super) static VME_RUN_NID: u32 = 0x1E4E_DE74;
    #[link_section = ".rodata.sceNid.psp_vme_kernel.VmeShutdown"]
    pub(super) static VME_SHUTDOWN_NID: u32 = 0xC9B3_15E4;
    #[link_section = ".rodata.sceNid.psp_vme_kernel.VmeSelfTest"]
    pub(super) static VME_SELF_TEST_NID: u32 = 0x6F02_4768;

    #[link_section = ".sceStub.text.psp_vme_kernel.VmeInit"]
    #[no_mangle]
    pub(super) static __VmeInit_stub: Stub = Stub { lib_addr: &LIB_ENTRY, nid_addr: &VME_INIT_NID };
    #[link_section = ".sceStub.text.psp_vme_kernel.VmeSharedAddr"]
    #[no_mangle]
    pub(super) static __VmeSharedAddr_stub: Stub = Stub { lib_addr: &LIB_ENTRY, nid_addr: &VME_SHARED_ADDR_NID };
    #[link_section = ".sceStub.text.psp_vme_kernel.VmeRun"]
    #[no_mangle]
    pub(super) static __VmeRun_stub: Stub = Stub { lib_addr: &LIB_ENTRY, nid_addr: &VME_RUN_NID };
    #[link_section = ".sceStub.text.psp_vme_kernel.VmeShutdown"]
    #[no_mangle]
    pub(super) static __VmeShutdown_stub: Stub = Stub { lib_addr: &LIB_ENTRY, nid_addr: &VME_SHUTDOWN_NID };
    #[link_section = ".sceStub.text.psp_vme_kernel.VmeSelfTest"]
    #[no_mangle]
    pub(super) static __VmeSelfTest_stub: Stub = Stub { lib_addr: &LIB_ENTRY, nid_addr: &VME_SELF_TEST_NID };
}

/// Boot the Media Engine and allocate the shared job. Returns the ME image
/// table id (>=2) on success, or a negative error.
#[cfg(target_os = "psp")]
pub fn init() -> i32 {
    extern "C" {
        fn __VmeInit_stub() -> i32;
    }
    unsafe { __VmeInit_stub() }
}

/// Uncached-user address of the shared [`VmeJob`], or 0 before [`init`].
#[cfg(target_os = "psp")]
pub fn shared_addr() -> u32 {
    extern "C" {
        fn __VmeSharedAddr_stub() -> u32;
    }
    unsafe { __VmeSharedAddr_stub() }
}

/// Kick one run and block until the ME acks. 0 on success, negative on timeout.
#[cfg(target_os = "psp")]
pub fn run() -> i32 {
    extern "C" {
        fn __VmeRun_stub() -> i32;
    }
    unsafe { __VmeRun_stub() }
}

/// Free the shared job.
#[cfg(target_os = "psp")]
pub fn shutdown() -> i32 {
    extern "C" {
        fn __VmeShutdown_stub() -> i32;
    }
    unsafe { __VmeShutdown_stub() }
}

/// Run the plugin's built-in integer MAC self test. Returns the result
/// (expected `72`) or a negative error. Validates boot + runner + shared memory
/// without needing the Rust assembler.
#[cfg(target_os = "psp")]
pub fn self_test() -> i32 {
    extern "C" {
        fn __VmeSelfTest_stub() -> i32;
    }
    unsafe { __VmeSelfTest_stub() }
}

/// Safe volatile accessor over the shared [`VmeJob`] at [`shared_addr`].
#[cfg(target_os = "psp")]
pub struct Job(*mut VmeJob);

#[cfg(target_os = "psp")]
impl Job {
    /// Wrap the shared job. Returns `None` before [`init`] has run.
    pub fn get() -> Option<Job> {
        let a = shared_addr();
        if a == 0 {
            None
        } else {
            Some(Job(a as *mut VmeJob))
        }
    }

    /// Load `n` input words into `data[dst_word..]`.
    pub fn set_data(&self, dst_word: usize, words: &[u32]) {
        let base = unsafe { &raw mut (*self.0).data } as *mut u32;
        for (i, &w) in words.iter().enumerate() {
            unsafe { core::ptr::write_volatile(base.add(dst_word + i), w) };
        }
    }

    /// Write the 112-word datapath context (from [`vme_asm!`](crate::vme_asm)).
    pub fn set_ctx(&self, ctx: &[u32; VME_CTX_WORDS]) {
        let base = unsafe { &raw mut (*self.0).ctx } as *mut u32;
        for (i, &w) in ctx.iter().enumerate() {
            unsafe { core::ptr::write_volatile(base.add(i), w) };
        }
    }

    /// Configure the ring-buffer loads for the next run.
    pub fn set_stages(&self, stages: &[VmeStage]) {
        unsafe {
            core::ptr::write_volatile(&raw mut (*self.0).n_stage, stages.len() as u32);
            let base = unsafe { &raw mut (*self.0).stage } as *mut VmeStage;
            for (i, s) in stages.iter().enumerate() {
                core::ptr::write_volatile(base.add(i), *s);
            }
        }
    }

    /// Set where the run's result is read from.
    pub fn set_readback(&self, ring_base: u32, word_off: u32) {
        unsafe {
            core::ptr::write_volatile(&raw mut (*self.0).readback_base, ring_base);
            core::ptr::write_volatile(&raw mut (*self.0).readback_woff, word_off);
        }
    }

    /// Result of the last run.
    pub fn result(&self) -> u32 {
        unsafe { core::ptr::read_volatile(&raw const (*self.0).result) }
    }

    /// Whether the installed plugin supports machine-image mode: v1.1's
    /// `VmeInit` leaves a capability marker in `image_addr`. **Check this
    /// before [`Job::set_image`]** — a v1.0 plugin allocated the smaller
    /// job block, and writing the appended fields against it lands past
    /// the allocation.
    pub fn has_image_mode(&self) -> bool {
        const IMAGE_CAPS: u32 = 0x564D_4531; // "VME1", matches the plugin
        let caps = unsafe { core::ptr::read_volatile(&raw const (*self.0).image_addr) };
        caps == IMAGE_CAPS
    }

    /// Switch the next runs to machine-image mode: `addr` is the
    /// uncached-user address of a 1 MB image (vme-emu / vme-assembler
    /// format). Needs plugin v1.1 — gate on [`Job::has_image_mode`].
    pub fn set_image(&self, addr: u32) {
        unsafe {
            core::ptr::write_volatile(&raw mut (*self.0).image_addr, addr);
            core::ptr::write_volatile(&raw mut (*self.0).image_mode, 1);
        }
    }

    /// Back to the legacy stage/readback mode.
    pub fn clear_image(&self) {
        unsafe { core::ptr::write_volatile(&raw mut (*self.0).image_mode, 0) };
    }
}
