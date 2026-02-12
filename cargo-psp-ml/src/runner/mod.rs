//! PSP USB runner — HostFS server, shell client, and output capture.
//!
//! Speaks the psplink protocol to a Sony PSP over USB.

pub mod async_io;
pub mod error;
pub mod exec;
pub mod hostfs;
pub mod protocol;
pub mod shell;
pub mod usb;

pub use error::Error;
pub use exec::{ExitReason, Execution, PspRunner};
