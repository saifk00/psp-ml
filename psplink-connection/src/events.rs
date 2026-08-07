//! Event types produced by a `PSPConnection`'s reader thread, plus the
//! (pure, host-testable) parser that turns raw bytes off the psplink shell
//! channel (ASYNC_SHELL) into framed `ShellMarker`s.
//!
//! Wire format, from psplink/psplink.h:
//!   SHELL_PRINT(fmt, ...)          -> "\xff" fmt "\xfe"
//!   SHELL_PRINT_CMD(cmd, fmt, ...) -> "\xff" cmd fmt "\xfe"
//! where `cmd` is SHELL_CMD_SUCCESS (0xFD) or SHELL_CMD_ERROR (0xFC), and
//! `fmt` for the post-command marker is always "0x%08X" (10 ASCII bytes).
//! Plain SHELL_PRINT frames (human-readable text, no cmd byte) are also
//! sent — e.g. the "Load/Start ... UID: ..." line right before the marker
//! — so the framer must tell those apart from the one that matters.

/// One event from the PSP, demultiplexed by channel. Shell-channel bytes
/// arrive raw/unframed here (a single write on the PSP side can span
/// multiple of these) — `load_program` owns a `ShellFramer` to turn them
/// into `ShellMarker`s, rather than the reader thread's callback doing it,
/// so framing state lives in one place on one thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PspEvent {
    Stdout(Vec<u8>),
    Stderr(Vec<u8>),
    ShellRaw(Vec<u8>),
    /// The reader thread observed the USB connection go away.
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellMarker {
    /// SHELL_CMD_SUCCESS (0xFD) — dispatch succeeded; see the
    /// psp_ml::module! convention for what the value itself means for `ld`.
    Success(u32),
    /// SHELL_CMD_ERROR (0xFC) — shellExecute couldn't dispatch the command
    /// at all (e.g. unknown command name).
    Error(u32),
}

const FRAME_START: u8 = 0xFF;
const FRAME_END: u8 = 0xFE;
const CMD_SUCCESS: u8 = 0xFD;
const CMD_ERROR: u8 = 0xFC;

/// Accumulates raw bytes from the ASYNC_SHELL channel (which can arrive
/// split across multiple USB packets/callback invocations) and yields
/// `ShellMarker`s as complete `0xFF <cmd> 0x%08X 0xFE` frames appear.
/// Plain-text SHELL_PRINT frames (no recognized cmd byte right after
/// `0xFF`) are parsed too, but discarded — they're human-readable log
/// lines, not something `load_program` needs to act on.
#[derive(Debug, Default)]
pub struct ShellFramer {
    /// Bytes seen since the last unmatched 0xFF, or empty if idle.
    buf: Vec<u8>,
    in_frame: bool,
}

impl ShellFramer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed newly-arrived shell-channel bytes; returns any markers that
    /// completed as a result.
    pub fn push(&mut self, data: &[u8]) -> Vec<ShellMarker> {
        let mut markers = Vec::new();

        for &b in data {
            if !self.in_frame {
                if b == FRAME_START {
                    self.in_frame = true;
                    self.buf.clear();
                }
                // else: stray byte outside any frame — ignore.
                continue;
            }

            if b == FRAME_END {
                if let Some(marker) = Self::parse_frame(&self.buf) {
                    markers.push(marker);
                }
                self.in_frame = false;
                self.buf.clear();
                continue;
            }

            self.buf.push(b);
        }

        markers
    }

    /// `body` is everything between the 0xFF and 0xFE, cmd byte included.
    fn parse_frame(body: &[u8]) -> Option<ShellMarker> {
        let (&cmd, rest) = body.split_first()?;
        let value = parse_hex_u32(rest)?;
        match cmd {
            CMD_SUCCESS => Some(ShellMarker::Success(value)),
            CMD_ERROR => Some(ShellMarker::Error(value)),
            _ => None, // plain SHELL_PRINT text frame, not a completion marker
        }
    }
}

/// Parses a "0x%08X"-style hex literal (case-insensitive, "0x" required —
/// that's what psplink/shell.c's SHELL_PRINT_CMD always emits).
fn parse_hex_u32(bytes: &[u8]) -> Option<u32> {
    let s = std::str::from_utf8(bytes).ok()?;
    let hex = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    u32::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_success_marker_in_one_push() {
        let mut f = ShellFramer::new();
        let markers = f.push(b"\xff\xfd0x00000000\xfe");
        assert_eq!(markers, vec![ShellMarker::Success(0)]);
    }

    #[test]
    fn parses_error_marker() {
        let mut f = ShellFramer::new();
        let markers = f.push(b"\xff\xfc0xFFFFFFFF\xfe");
        assert_eq!(markers, vec![ShellMarker::Error(0xFFFFFFFF)]);
    }

    #[test]
    fn parses_panic_sentinel_value() {
        let mut f = ShellFramer::new();
        let markers = f.push(b"\xff\xfd0xFFFFFFFF\xfe");
        assert_eq!(markers, vec![ShellMarker::Success(0xFFFFFFFF)]);
    }

    #[test]
    fn frame_split_across_multiple_pushes() {
        let mut f = ShellFramer::new();
        assert_eq!(f.push(b"\xff\xfd0x0000"), vec![]);
        assert_eq!(f.push(b"0000\xfe"), vec![ShellMarker::Success(0)]);
    }

    #[test]
    fn frame_split_byte_by_byte() {
        let mut f = ShellFramer::new();
        let mut all = Vec::new();
        for &b in b"\xff\xfd0x00000042\xfe" {
            all.extend(f.push(&[b]));
        }
        assert_eq!(all, vec![ShellMarker::Success(0x42)]);
    }

    #[test]
    fn plain_text_frame_is_ignored() {
        let mut f = ShellFramer::new();
        let markers = f.push(b"\xffLoad/Start host1:hello-psp.prx UID: 0x00CA7663\r\n\xfe");
        assert_eq!(markers, vec![]);
    }

    #[test]
    fn text_frame_then_marker_frame() {
        let mut f = ShellFramer::new();
        let mut all = Vec::new();
        all.extend(f.push(b"\xffLoad/Start foo.prx UID: 0x00000001\r\n\xfe"));
        all.extend(f.push(b"\xff\xfd0x00000001\xfe"));
        assert_eq!(all, vec![ShellMarker::Success(1)]);
    }

    #[test]
    fn multiple_markers_in_one_push() {
        let mut f = ShellFramer::new();
        let markers = f.push(b"\xff\xfd0x00000000\xfe\xff\xfc0x00000001\xfe");
        assert_eq!(
            markers,
            vec![ShellMarker::Success(0), ShellMarker::Error(1)]
        );
    }

    #[test]
    fn garbage_before_first_frame_is_ignored() {
        let mut f = ShellFramer::new();
        let markers = f.push(b"garbage\xff\xfd0x00000000\xfe");
        assert_eq!(markers, vec![ShellMarker::Success(0)]);
    }

    #[test]
    fn malformed_frame_yields_no_marker_and_recovers() {
        let mut f = ShellFramer::new();
        // Not enough hex digits to be a real 0x%08X value — parse_hex_u32
        // still succeeds on a short hex string (u32::from_str_radix
        // doesn't require exactly 8 digits), so use genuinely non-hex
        // content to exercise the "give up on this frame" path.
        let markers = f.push(b"\xff\xfdnot-hex\xfe\xff\xfd0x00000005\xfe");
        assert_eq!(markers, vec![ShellMarker::Success(5)]);
    }
}
