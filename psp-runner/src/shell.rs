use crate::protocol::{SHELL_BEGIN, SHELL_END};

/// Shell response type codes (from shellcmd.h).
pub const SHELL_TYPE_SUCCESS: u8 = 0xFD;
pub const SHELL_TYPE_ERROR: u8 = 0xFC;
pub const SHELL_TYPE_CWD: u8 = 0xFB;
pub const SHELL_TYPE_TAB: u8 = 0xFA;
pub const SHELL_TYPE_DISASM: u8 = 0xF8;
pub const SHELL_TYPE_SYMLOAD: u8 = 0xF7;

/// Build a simple shell command with no arguments (e.g. "reset").
pub fn build_simple_command(cmd: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(cmd.as_bytes());
    buf.push(0x00); // NUL-terminate command
    buf.push(0x01); // command terminator
    buf
}

/// Build a "load and start module" shell command.
///
/// psplink shell protocol: NUL-separated args terminated by 0x01.
/// Wire format: `ld\0host0:/path/to/module.prx\0\x01`
pub fn build_load_command(prx_path: &str) -> Vec<u8> {
    let mut cmd = Vec::new();
    cmd.extend_from_slice(b"ld\0");
    cmd.extend_from_slice(prx_path.as_bytes());
    cmd.push(0x00); // NUL-terminate path arg
    cmd.push(0x01); // command terminator
    cmd
}

/// A parsed shell response frame.
pub struct ShellFrame {
    pub frame_type: u8,
    pub payload: Vec<u8>,
}

/// Parse shell response frames from raw async channel 0 data.
///
/// Shell output uses framing: `0xFF <type> <payload...> 0xFE`.
/// Returns parsed frames and any remaining (incomplete) data.
pub fn parse_shell_frames(data: &[u8]) -> (Vec<ShellFrame>, Vec<u8>) {
    let mut frames = Vec::new();
    let mut remainder = Vec::new();
    let mut i = 0;

    while i < data.len() {
        if data[i] == SHELL_BEGIN {
            // Find the matching END
            if let Some(end_pos) = data[i + 1..].iter().position(|&b| b == SHELL_END) {
                let end_pos = i + 1 + end_pos;
                if end_pos > i + 1 {
                    frames.push(ShellFrame {
                        frame_type: data[i + 1],
                        payload: data[i + 2..end_pos].to_vec(),
                    });
                }
                i = end_pos + 1;
            } else {
                // Incomplete frame — save as remainder
                remainder.extend_from_slice(&data[i..]);
                break;
            }
        } else {
            // Unframed data (plain text output)
            let next_begin = data[i..].iter().position(|&b| b == SHELL_BEGIN);
            let end = next_begin.map_or(data.len(), |pos| i + pos);
            if end > i {
                frames.push(ShellFrame {
                    frame_type: 0, // plain text
                    payload: data[i..end].to_vec(),
                });
            }
            i = end;
        }
    }

    (frames, remainder)
}
