use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// PSP device not found on USB bus
    DeviceNotFound,
    /// USB open/claim error
    UsbOpen(rusb::Error),
    /// Handshake with PSP/psplink failed
    HandshakeFailed,
    /// USB I/O error during operation
    UsbIo(rusb::Error),
    /// Timeout waiting for result
    Timeout,
    /// Protocol error (unexpected packet format)
    Protocol(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DeviceNotFound => write!(f, "PSP device not found (VID:054C PID:01C9)"),
            Error::UsbOpen(e) => write!(f, "USB open error: {e}"),
            Error::HandshakeFailed => write!(f, "psplink handshake failed"),
            Error::UsbIo(e) => write!(f, "USB I/O error: {e}"),
            Error::Timeout => write!(f, "timed out waiting for PSP"),
            Error::Protocol(msg) => write!(f, "protocol error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
