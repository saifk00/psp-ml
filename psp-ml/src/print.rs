//! Host-visible print macros for PSP programs.
//!
//! On PSP (via psplink): writes to `sceKernelStdout()` which psplink routes
//! through USB to the host. Output appears in `psp-runner`'s stdout.
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


#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {{
        #[cfg(target_os = "psp")]
        {
            use core::fmt::Write;
            let mut buf = $crate::print::WriteBuf::new();
            let _ = write!(buf, $($arg)*);
            $crate::print::write_stdout(buf.as_bytes());
        }
        #[cfg(not(target_os = "psp"))]
        {
            extern crate std;
            std::print!($($arg)*);
        }
    }};
}

#[macro_export]
macro_rules! println {
    () => { $crate::print!("\n") };
    ($($arg:tt)*) => {{
        #[cfg(target_os = "psp")]
        {
            use core::fmt::Write;
            let mut buf = $crate::print::WriteBuf::new();
            let _ = write!(buf, $($arg)*);
            let _ = buf.write_str("\n");
            $crate::print::write_stdout(buf.as_bytes());
        }
        #[cfg(not(target_os = "psp"))]
        {
            extern crate std;
            std::println!($($arg)*);
        }
    }};
}
