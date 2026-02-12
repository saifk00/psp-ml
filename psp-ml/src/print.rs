//! Dual-output print macros for PSP programs.
//!
//! On PSP: writes to both the PSP screen (via psp::debug) and to psplink's
//! USB stdout (so `cargo psp-ml run` captures the output on the host).
//!
//! On host (`--features local`): delegates to `std::print!`.

/// Max bytes per print call. Messages longer than this are truncated.
/// Avoids heap allocation — uses a stack buffer with `core::fmt::Write`.
const BUF_SIZE: usize = 512;

/// Stack buffer that implements `core::fmt::Write`.
pub struct WriteBuf {
    buf: [u8; BUF_SIZE],
    pos: usize,
}

impl WriteBuf {
    pub fn new() -> Self {
        Self {
            buf: [0u8; BUF_SIZE],
            pos: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.pos]
    }
}

impl core::fmt::Write for WriteBuf {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let remaining = BUF_SIZE - self.pos;
        let len = bytes.len().min(remaining);
        self.buf[self.pos..self.pos + len].copy_from_slice(&bytes[..len]);
        self.pos += len;
        if bytes.len() > remaining {
            Err(core::fmt::Error)
        } else {
            Ok(())
        }
    }
}

/// Write formatted bytes to psplink's stdout (routed to host via USB).
#[cfg(target_os = "psp")]
pub fn write_stdout(bytes: &[u8]) {
    unsafe {
        let fd = psp::sys::sceKernelStdout();
        psp::sys::sceIoWrite(fd, bytes.as_ptr() as *const _, bytes.len());
    }
}

/// Write pre-formatted bytes to the PSP debug screen.
#[cfg(target_os = "psp")]
pub fn write_screen(bytes: &[u8]) {
    let s = core::str::from_utf8(bytes).unwrap_or("");
    psp::debug::print_args(core::format_args!("{}", s));
}

#[macro_export]
macro_rules! dprint {
    ($($arg:tt)*) => {{
        #[cfg(target_os = "psp")]
        {
            use core::fmt::Write;
            let mut buf = $crate::print::WriteBuf::new();
            let _ = write!(buf, $($arg)*);
            $crate::print::write_stdout(buf.as_bytes());
            $crate::print::write_screen(buf.as_bytes());
        }
        #[cfg(not(target_os = "psp"))]
        {
            extern crate std;
            std::print!($($arg)*);
        }
    }};
}

#[macro_export]
macro_rules! dprintln {
    () => { $crate::dprint!("\n") };
    ($($arg:tt)*) => {{
        #[cfg(target_os = "psp")]
        {
            use core::fmt::Write;
            let mut buf = $crate::print::WriteBuf::new();
            let _ = write!(buf, $($arg)*);
            let _ = buf.write_str("\n");
            $crate::print::write_stdout(buf.as_bytes());
            $crate::print::write_screen(buf.as_bytes());
        }
        #[cfg(not(target_os = "psp"))]
        {
            extern crate std;
            std::println!($($arg)*);
        }
    }};
}
