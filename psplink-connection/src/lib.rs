//! Safe wrapper over `usbhostfs-sys`: connect to a PSP running psplink and
//! load/run PRXs directly, without a `usbhostfs_pc` subprocess or TCP
//! bridge in between.

mod events;
mod load;

pub use events::{PspEvent, ShellMarker};
pub use load::LoadOutcome;

use std::os::raw::{c_int, c_void};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use usbhostfs_sys::UhfsCtx;

/// How long a single `uhfs_pump` call blocks before the reader thread
/// re-checks whether it's been asked to stop.
const PUMP_POLL_MS: c_int = 200;

#[derive(Debug)]
pub enum PspError {
    /// Failed to configure a host drive (bad path, or drive number out of
    /// range — see stderr for usbhostfs-sys's own diagnostic).
    AddDriveFailed { drive: i32 },
    /// `uhfs_connect` failed (no PSP found / USB handshake failed).
    ConnectFailed,
    /// A `uhfs_async_write` call failed; the raw (negative) return value
    /// from the underlying `libusb_bulk_transfer`.
    UsbWrite(i32),
    /// The reader thread observed the connection drop.
    Disconnected,
    /// No event arrived within the expected window.
    Timeout,
}

impl std::fmt::Display for PspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PspError::AddDriveFailed { drive } => write!(f, "failed to configure host drive {drive}"),
            PspError::ConnectFailed => write!(f, "failed to connect to PSP over USB"),
            PspError::UsbWrite(code) => write!(f, "USB write failed (libusb error {code})"),
            PspError::Disconnected => write!(f, "PSP connection was lost"),
            PspError::Timeout => write!(f, "timed out waiting for a response from the PSP"),
        }
    }
}

impl std::error::Error for PspError {}

/// Wraps the raw `*mut UhfsCtx`. Safe to move/share across the reader
/// thread and whichever thread owns the `PSPConnection`: usbhostfs-sys's
/// `uhfs_pump` (ep 0x81 reads) and `uhfs_async_write` (ep 0x3 writes) are
/// independent `libusb_bulk_transfer` calls on the same device handle,
/// which libusb documents as safe to issue concurrently from different
/// threads — this mirrors what upstream usbhostfs_pc already did with its
/// own two threads (main thread reading, `async_thread` writing).
struct UsbHandle(NonNull<UhfsCtx>);
unsafe impl Send for UsbHandle {}
unsafe impl Sync for UsbHandle {}

/// State shared with the C callback via a raw pointer. Boxed once at
/// connect time, freed at disconnect time.
struct CallbackState {
    sender: mpsc::Sender<PspEvent>,
}

extern "C" fn async_trampoline(user: *mut c_void, channel: c_int, data: *const u8, len: c_int) {
    if user.is_null() || data.is_null() || len < 0 {
        return;
    }
    let state = unsafe { &*(user as *const CallbackState) };
    let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) }.to_vec();

    let event = match channel {
        usbhostfs_sys::ASYNC_STDOUT => PspEvent::Stdout(bytes),
        usbhostfs_sys::ASYNC_STDERR => PspEvent::Stderr(bytes),
        usbhostfs_sys::ASYNC_SHELL => PspEvent::ShellRaw(bytes),
        _ => return, // ASYNC_GDB or anything else — not something we consume today
    };
    let _ = state.sender.send(event);
}

#[derive(Debug, Clone, Copy)]
pub struct ConnectOptions {
    /// How long the reader thread's single USB read blocks per poll.
    /// Smaller values make `disconnect()` return faster at the cost of
    /// slightly more idle CPU/USB-transfer overhead.
    pub pump_poll_ms: i32,
}

impl Default for ConnectOptions {
    fn default() -> Self {
        Self {
            pump_poll_ms: PUMP_POLL_MS,
        }
    }
}

pub struct PSPConnection {
    ctx: UsbHandle,
    events: mpsc::Receiver<PspEvent>,
    stop: std::sync::Arc<AtomicBool>,
    reader: Option<JoinHandle<()>>,
    // Kept alive for as long as the C side might still call back into it.
    _callback_state: Box<CallbackState>,
}

impl PSPConnection {
    /// Connects to a PSP running psplink over USB, mounting `host0`/`host1`
    /// as the psplink `host0:`/`host1:` drives. Blocks until a device is
    /// found and the HOSTFS_MAGIC handshake succeeds.
    pub fn connect(
        host0: &std::path::Path,
        host1: &std::path::Path,
        opts: ConnectOptions,
    ) -> Result<Self, PspError> {
        let ctx = unsafe { usbhostfs_sys::uhfs_ctx_new() };
        let ctx = NonNull::new(ctx).ok_or(PspError::ConnectFailed)?;

        let host0_c = path_to_cstring(host0);
        let host1_c = path_to_cstring(host1);
        unsafe {
            if usbhostfs_sys::uhfs_add_drive(ctx.as_ptr(), 0, host0_c.as_ptr()) != 0 {
                usbhostfs_sys::uhfs_ctx_free(ctx.as_ptr());
                return Err(PspError::AddDriveFailed { drive: 0 });
            }
            if usbhostfs_sys::uhfs_add_drive(ctx.as_ptr(), 1, host1_c.as_ptr()) != 0 {
                usbhostfs_sys::uhfs_ctx_free(ctx.as_ptr());
                return Err(PspError::AddDriveFailed { drive: 1 });
            }
        }

        let (tx, rx) = mpsc::channel();
        let callback_state = Box::new(CallbackState { sender: tx.clone() });
        let user_ptr = callback_state.as_ref() as *const CallbackState as *mut c_void;
        unsafe {
            usbhostfs_sys::uhfs_set_async_callback(ctx.as_ptr(), async_trampoline, user_ptr);
        }

        let connect_ret = unsafe { usbhostfs_sys::uhfs_connect(ctx.as_ptr()) };
        if connect_ret != 0 {
            unsafe { usbhostfs_sys::uhfs_ctx_free(ctx.as_ptr()) };
            return Err(PspError::ConnectFailed);
        }

        let handle = UsbHandle(ctx);
        let stop = std::sync::Arc::new(AtomicBool::new(false));

        let reader = {
            let raw = handle.0.as_ptr() as usize; // Copy the pointer value into the thread, see UsbHandle's doc comment.
            let stop = stop.clone();
            let tx = tx.clone();
            let pump_timeout = opts.pump_poll_ms as c_int;
            std::thread::spawn(move || {
                let ctx_ptr = raw as *mut UhfsCtx;
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let ret = unsafe { usbhostfs_sys::uhfs_pump(ctx_ptr, pump_timeout) };
                    if ret < 0 {
                        let _ = tx.send(PspEvent::Disconnected);
                        break;
                    }
                }
            })
        };

        Ok(PSPConnection {
            ctx: handle,
            events: rx,
            stop,
            reader: Some(reader),
            _callback_state: callback_state,
        })
    }

    /// Sends a psplink shell command. Wire format: each arg NUL-terminated,
    /// followed by a trailing 0x01 byte — psplink's `usbShellReadInput`
    /// (psplink/usbshell.c) reads the shell channel byte-by-byte, not as
    /// text lines; see `psp_rt::module!`'s doc comments / this session's
    /// findings for why a plain `"cmd arg\n"` string doesn't work.
    pub fn send_shell_command(&self, args: &[&str]) -> Result<(), PspError> {
        let mut buf = Vec::new();
        for arg in args {
            buf.extend_from_slice(arg.as_bytes());
            buf.push(0);
        }
        buf.push(1);

        let ret = unsafe {
            usbhostfs_sys::uhfs_async_write(
                self.ctx.0.as_ptr(),
                usbhostfs_sys::ASYNC_SHELL,
                buf.as_ptr(),
                buf.len() as c_int,
            )
        };
        if ret < 0 {
            return Err(PspError::UsbWrite(ret));
        }
        Ok(())
    }

    /// Receives the next raw event (stdout/stderr chunk, unframed shell
    /// bytes, or disconnect notice) from the reader thread. `load_program`
    /// is built entirely on top of this — most callers want that instead,
    /// but this is here for anything that needs lower-level access (e.g.
    /// sending an arbitrary shell command and inspecting the raw
    /// response).
    pub fn recv_event(&self, timeout: Duration) -> Result<PspEvent, PspError> {
        self.events.recv_timeout(timeout).map_err(|_| PspError::Timeout)
    }

    /// Stops the reader thread and tears down the USB connection.
    pub fn disconnect(self) {
        // All the real work happens in Drop; this just makes "I'm done"
        // explicit at the call site instead of relying on scope exit.
    }
}

impl Drop for PSPConnection {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        unsafe {
            usbhostfs_sys::uhfs_disconnect(self.ctx.0.as_ptr());
            usbhostfs_sys::uhfs_ctx_free(self.ctx.0.as_ptr());
        }
    }
}

fn path_to_cstring(p: &std::path::Path) -> std::ffi::CString {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::CString::new(p.as_os_str().as_bytes()).expect("path contains a NUL byte")
}
