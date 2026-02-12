use super::protocol::{self, ASYNC_MAGIC, ASYNC_SHELL, ASYNC_STDERR, ASYNC_STDOUT};

/// A parsed async message received from the PSP on EP1.
pub struct AsyncMessage {
    pub channel: u32,
    pub payload: Vec<u8>,
}

impl AsyncMessage {
    /// Parse an async packet. Returns `None` if the buffer is too short or has wrong magic.
    pub fn parse(packet: &[u8]) -> Option<Self> {
        if packet.len() < 8 {
            return None;
        }
        let magic = protocol::read_u32_le(packet, 0)?;
        if magic != ASYNC_MAGIC {
            return None;
        }
        let channel = protocol::read_u32_le(packet, 4)?;
        let payload = packet[8..].to_vec();
        Some(AsyncMessage { channel, payload })
    }
}

/// Build an async packet to send on EP3.
pub fn build_async_packet(channel: u32, payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + payload.len());
    buf.extend_from_slice(&ASYNC_MAGIC.to_le_bytes());
    buf.extend_from_slice(&channel.to_le_bytes());
    buf.extend_from_slice(payload);
    buf
}

/// Accumulates output from async channels during a PRX execution.
pub struct OutputCapture {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub shell: Vec<u8>,
}

impl OutputCapture {
    pub fn new() -> Self {
        OutputCapture {
            stdout: Vec::new(),
            stderr: Vec::new(),
            shell: Vec::new(),
        }
    }

    /// Feed an async message into the appropriate buffer.
    pub fn feed(&mut self, msg: &AsyncMessage) {
        match msg.channel {
            ASYNC_SHELL => self.shell.extend_from_slice(&msg.payload),
            ASYNC_STDOUT => self.stdout.extend_from_slice(&msg.payload),
            ASYNC_STDERR => self.stderr.extend_from_slice(&msg.payload),
            _ => log::debug!("async channel {}: {} bytes", msg.channel, msg.payload.len()),
        }
    }

    pub fn stdout_str(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    pub fn stderr_str(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}
