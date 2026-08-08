//! `PSPConnection::load_program` — send `ld <path>`, stream stdout/stderr,
//! and block until psplink's shell-channel completion marker arrives.

use crate::events::{PspEvent, ShellFramer, ShellMarker};
use crate::{PSPConnection, PspError};
use std::time::Duration;

/// `main_thread`'s exit-status convention chosen for `psp_rt::module!`
/// (see that macro's doc comment for the full trace through uofw/psplink
/// justifying these specific values). `main_thread` itself returns `1` on
/// a clean exit, but the kernel's ModuleManager collapses
/// `SCE_KERNEL_NO_RESIDENT` (1) down to `SCE_ERROR_OK` (0) before it
/// reaches the shell channel — so by the time it gets here, `0` is what a
/// clean exit looks like, and this sentinel (passed through raw, since
/// it's neither 0 nor 1) is a caught panic. Kept here (not psp-rt, which
/// is a `no_std` on-device crate this host-side crate can't depend on)
/// since both sides need to agree on the same bit pattern.
pub const PANIC_SENTINEL: u32 = 0xFFFF_FFFF;

/// How long `load_program` waits for a single event (stdout chunk or the
/// completion marker) before giving up. Loading a large debug PRX over
/// hostfs can legitimately take tens of seconds — see this session's own
/// measurements — so this is deliberately generous.
const EVENT_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOutcome {
    /// The module ran to completion without panicking.
    Success,
    /// `app_main` panicked (caught by `catch_unwind` in `psp_rt::module!`).
    Panicked,
    /// psplink's shell couldn't dispatch the `ld` command at all.
    ShellError(u32),
    /// `sceKernelLoadModule`/`sceKernelStartModule` failed before the
    /// module's own code ever ran (bad PRX, bad path, etc.), or the
    /// module returned a value outside the `{1, PANIC_SENTINEL}`
    /// convention.
    KernelError(u32),
}

impl PSPConnection {
    /// Loads and runs `prx_path` (e.g. `"host1:hello-psp.prx"`), streaming
    /// its stdout/stderr to `on_stdout` as it arrives, and blocking until
    /// psplink reports the module's thread has finished — see
    /// `psp_rt::module!`'s doc comment for why that's a truthful signal
    /// only once its wait-on-thread-end fix has landed on the device side.
    pub fn load_program(
        &self,
        prx_path: &str,
        mut on_stdout: impl FnMut(&[u8]),
    ) -> Result<LoadOutcome, PspError> {
        self.send_shell_command(&["ld", prx_path])?;

        let mut framer = ShellFramer::new();
        loop {
            match self.recv_event(EVENT_TIMEOUT)? {
                PspEvent::Stdout(bytes) | PspEvent::Stderr(bytes) => on_stdout(&bytes),
                PspEvent::ShellRaw(bytes) => {
                    for marker in framer.push(&bytes) {
                        return Ok(outcome_for(marker));
                    }
                }
                PspEvent::Disconnected => return Err(PspError::Disconnected),
            }
        }
    }
}

fn outcome_for(marker: ShellMarker) -> LoadOutcome {
    match marker {
        // modulemgr.c's CMD_START_MODULE handler collapses module_start
        // returning SCE_KERNEL_NO_RESIDENT (1) down to SCE_ERROR_OK (0)
        // before it reaches userland/the shell channel — so 0, not 1, is
        // what a clean exit actually looks like from here. See
        // psp_rt::module!'s doc comment for the full chain.
        ShellMarker::Success(0) => LoadOutcome::Success,
        ShellMarker::Success(PANIC_SENTINEL) => LoadOutcome::Panicked,
        ShellMarker::Success(other) => LoadOutcome::KernelError(other),
        ShellMarker::Error(v) => LoadOutcome::ShellError(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_success_value() {
        assert_eq!(outcome_for(ShellMarker::Success(0)), LoadOutcome::Success);
    }

    #[test]
    fn maps_panic_sentinel() {
        assert_eq!(
            outcome_for(ShellMarker::Success(PANIC_SENTINEL)),
            LoadOutcome::Panicked
        );
    }

    #[test]
    fn maps_other_success_value_to_kernel_error() {
        assert_eq!(
            outcome_for(ShellMarker::Success(1)),
            LoadOutcome::KernelError(1)
        );
    }

    #[test]
    fn maps_shell_error() {
        assert_eq!(
            outcome_for(ShellMarker::Error(0x8002013C)),
            LoadOutcome::ShellError(0x8002013C)
        );
    }
}
