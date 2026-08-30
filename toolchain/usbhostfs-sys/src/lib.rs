//! Raw FFI bindings to the vendored, refactored `usbhostfs_pc` C sources in
//! `vendor/` (forked from `~/psplinkusb/usbhostfs_pc/main.c`).
//!
//! This is a deliberately thin, unsafe layer — see the `psplink-connection`
//! crate for the safe wrapper (`PSPConnection`) meant for actual use.
//!
//! Stage A note: `UhfsCtx` is effectively a marker today; the vendored C
//! still keeps real connection state (USB handle, drive table, open file
//! table) in file-scope globals, so only one `UhfsCtx` may be connected at
//! a time per process. Stage B moves that state into the C struct itself.

use std::os::raw::{c_char, c_int, c_void};

/// Opaque handle to a `usbhostfs-sys` connection. See the module-level
/// "Stage A note" above for the current single-connection-per-process
/// constraint.
#[repr(C)]
pub struct UhfsCtx {
    _private: [u8; 0],
}

/// Async channel IDs, matching `usbhostfs/usbhostfs.h`'s `USB_ASYNC_CHANNELS`.
pub const ASYNC_SHELL: c_int = 0;
pub const ASYNC_GDB: c_int = 1;
pub const ASYNC_STDOUT: c_int = 2;
pub const ASYNC_STDERR: c_int = 3;

/// Callback invoked (on whatever thread calls `uhfs_pump`) with data the
/// PSP pushed on an async channel. `data` is only valid for the duration
/// of the call.
pub type UhfsAsyncCallback =
    extern "C" fn(user: *mut c_void, channel: c_int, data: *const u8, len: c_int);

// Linked via build.rs (cc::Build::compile emits the cargo:rustc-link-lib
// directive), so no #[link(...)] attribute is needed here.
extern "C" {
    pub fn uhfs_ctx_new() -> *mut UhfsCtx;
    pub fn uhfs_ctx_free(ctx: *mut UhfsCtx);

    /// Configures a host drive (0 or 1) to serve `dir` before connecting.
    /// Returns 0 on success, -1 on error (invalid drive number or
    /// non-existent directory — see stderr for details, matching upstream).
    pub fn uhfs_add_drive(ctx: *mut UhfsCtx, num: c_int, dir: *const c_char) -> c_int;

    /// Blocks until a PSP is found over USB and the HOSTFS_MAGIC handshake
    /// write succeeds. Returns 0 on success, -1 on error.
    pub fn uhfs_connect(ctx: *mut UhfsCtx) -> c_int;

    pub fn uhfs_disconnect(ctx: *mut UhfsCtx);

    /// One read+dispatch cycle. Returns >0 if a command was processed, 0 on
    /// an idle timeout (safe to call again immediately), <0 if the USB
    /// connection is gone (caller should `uhfs_disconnect`).
    pub fn uhfs_pump(ctx: *mut UhfsCtx, timeout_ms: c_int) -> c_int;

    pub fn uhfs_set_async_callback(ctx: *mut UhfsCtx, cb: UhfsAsyncCallback, user: *mut c_void);

    /// Sends `data` on `channel` to the PSP. Single USB packet per call
    /// (~508 byte payload cap, silently truncated if exceeded) — matches
    /// upstream's own unchunked async-write behavior.
    pub fn uhfs_async_write(ctx: *mut UhfsCtx, channel: c_int, data: *const u8, len: c_int)
        -> c_int;
}
