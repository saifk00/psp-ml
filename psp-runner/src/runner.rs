use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::async_io::{self, AsyncMessage, OutputCapture};
use crate::error::Error;
use crate::hostfs::FdTable;
use crate::protocol::*;
use crate::shell;
use crate::usb::PspUsb;

/// Result of executing a PRX on the PSP.
pub struct Execution {
    pub stdout: String,
    pub stderr: String,
    pub files_written: Vec<PathBuf>,
    pub exit_reason: ExitReason,
}

pub enum ExitReason {
    /// The --wait-for file was written and closed by the PSP.
    FileReceived(PathBuf),
    /// Timed out waiting.
    Timeout,
    /// PSP disconnected or USB error.
    Disconnected(String),
}

/// High-level PSP runner: connect over USB, serve HostFS, execute PRX, collect results.
pub struct PspRunner {
    usb: Arc<PspUsb>,
    root_dir: PathBuf,
}

impl PspRunner {
    /// Open USB connection to a PSP running psplink.
    pub fn connect(root_dir: PathBuf) -> Result<Self, Error> {
        let usb = PspUsb::open()?;
        Ok(PspRunner {
            usb: Arc::new(usb),
            root_dir,
        })
    }

    /// Perform the psplink handshake.
    ///
    /// 1. Write HOSTFS_MAGIC to EP2
    /// 2. Read HELLO from EP1
    /// 3. Write HELLO response back on EP2
    fn handshake(&self) -> Result<(), Error> {
        // Step 1: Send magic word
        self.usb.write_ep2(&HOSTFS_MAGIC.to_le_bytes())?;

        // Step 2: Read HELLO command from PSP
        let mut buf = vec![0u8; 512];
        let n = self
            .usb
            .read_ep1(&mut buf, Duration::from_secs(5))
            .map_err(|_| Error::HandshakeFailed)?;
        if n < 12 {
            return Err(Error::HandshakeFailed);
        }
        let magic = read_u32_le(&buf, 0);
        let cmd = read_u32_le(&buf, 4);
        if magic != HOSTFS_MAGIC || cmd != HELLO_CMD {
            return Err(Error::Protocol(format!(
                "expected HELLO, got magic={magic:#010X} cmd={cmd:#010X}"
            )));
        }

        // Step 3: Respond with HELLO
        let resp = hostfs_header(HELLO_CMD, 0);
        self.usb.write_ep2(&resp)?;

        log::info!("handshake complete");
        Ok(())
    }

    /// Execute a PRX on the PSP and collect output.
    ///
    /// - `prx_path`: the host0:/ path sent to psplink (e.g. `"host0:/target/.../foo.prx"`)
    /// - `wait_for`: if set, return once a file with this name is written to host0:/
    /// - `timeout`: maximum time to wait
    pub fn execute(
        &self,
        prx_path: &str,
        wait_for: Option<&str>,
        timeout: Duration,
    ) -> Result<Execution, Error> {
        self.handshake()?;

        let shutdown = Arc::new(AtomicBool::new(false));

        // Channels
        let (hostfs_tx, hostfs_rx) = mpsc::channel::<HostFsPacket>();
        let (async_tx, async_rx) = mpsc::channel::<Vec<u8>>();
        let (file_tx, file_rx) = mpsc::channel::<PathBuf>();

        // --- Reader thread: reads EP1, classifies packets, dispatches ---
        let usb_r = Arc::clone(&self.usb);
        let shutdown_r = Arc::clone(&shutdown);
        let reader_handle = thread::spawn(move || {
            let mut buf = vec![0u8; 65536 + 512];
            while !shutdown_r.load(Ordering::Relaxed) {
                match usb_r.read_ep1(&mut buf, Duration::from_millis(100)) {
                    Ok(n) if n >= 4 => {
                        let data = buf[..n].to_vec();
                        match classify_packet(&data) {
                            PacketKind::HostFs => {
                                // Parse the command header to get extra_len
                                let extra_len = if n >= 12 {
                                    read_u32_le(&data, 8) as usize
                                } else {
                                    0
                                };
                                let _ = hostfs_tx.send(HostFsPacket {
                                    data,
                                    expected_extra: extra_len,
                                });
                            }
                            PacketKind::Async => {
                                let _ = async_tx.send(data);
                            }
                            PacketKind::Bulk => {
                                log::debug!("bulk packet ({} bytes, ignored)", n);
                            }
                            PacketKind::Unknown(m) => {
                                log::warn!("unknown magic {:#010X} ({} bytes)", m, n);
                            }
                        }
                    }
                    Ok(_) => {} // short read
                    Err(Error::UsbIo(rusb::Error::Timeout)) => continue,
                    Err(e) => {
                        log::error!("EP1 read error: {}", e);
                        break;
                    }
                }
            }
        });

        // --- HostFS thread: processes filesystem commands, writes responses on EP2 ---
        let usb_h = Arc::clone(&self.usb);
        let shutdown_h = Arc::clone(&shutdown);
        let usb_extra = Arc::clone(&self.usb);
        let root = self.root_dir.clone();
        let hostfs_handle = thread::spawn(move || {
            let mut fd_table = FdTable::new(root, Some(file_tx));
            while !shutdown_h.load(Ordering::Relaxed) {
                match hostfs_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(pkt) => {
                        // Read any extra data that the command expects
                        let extra_data = if pkt.expected_extra > 0 {
                            // The extra data may already be appended after the command struct
                            // in the same USB transfer, or it may need a separate read.
                            let cmd_struct_size = guess_cmd_struct_size(&pkt.data);
                            if pkt.data.len() >= cmd_struct_size + pkt.expected_extra {
                                // All data arrived in one transfer
                                pkt.data[cmd_struct_size..cmd_struct_size + pkt.expected_extra]
                                    .to_vec()
                            } else {
                                // Need to read remaining bytes from EP1
                                // (This shouldn't normally happen since the PSP sends cmd+extra
                                // in sequence, and the reader sees them in the same bulk read.
                                // But handle it just in case.)
                                let already_got = if pkt.data.len() > cmd_struct_size {
                                    pkt.data.len() - cmd_struct_size
                                } else {
                                    0
                                };
                                let mut extra = if already_got > 0 {
                                    pkt.data[cmd_struct_size..].to_vec()
                                } else {
                                    Vec::new()
                                };
                                let remaining = pkt.expected_extra - already_got;
                                if remaining > 0 {
                                    let mut tmp = vec![0u8; remaining];
                                    match usb_extra.read_ep1(&mut tmp, Duration::from_secs(5)) {
                                        Ok(n) => extra.extend_from_slice(&tmp[..n]),
                                        Err(e) => log::error!("extra read error: {}", e),
                                    }
                                }
                                extra
                            }
                        } else {
                            Vec::new()
                        };

                        let cmd_struct_size = guess_cmd_struct_size(&pkt.data);
                        let cmd_buf = &pkt.data[..cmd_struct_size.min(pkt.data.len())];
                        let response = fd_table.handle_command(cmd_buf, &extra_data);

                        // Send response on EP2 (may include header + extra data)
                        if let Err(e) = usb_h.write_ep2(&response) {
                            log::error!("EP2 write error: {}", e);
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        });

        // --- Send load command on EP3 (async channel 0 = shell) ---
        let load_cmd = shell::build_load_command(prx_path);
        let packet = async_io::build_async_packet(ASYNC_SHELL, &load_cmd);
        self.usb.write_ep3(&packet)?;
        log::info!("sent load command: {}", prx_path);

        // --- Main loop: collect output, watch for file-write events ---
        let mut output = OutputCapture::new();
        let mut files_written = Vec::new();
        let start = Instant::now();
        let wait_for_owned = wait_for.map(|s| s.to_owned());

        let exit_reason = loop {
            if start.elapsed() > timeout {
                break ExitReason::Timeout;
            }

            // Drain async messages
            while let Ok(packet) = async_rx.try_recv() {
                if let Some(msg) = AsyncMessage::parse(&packet) {
                    // Forward PSP stdout to host stderr in real-time
                    if msg.channel == ASYNC_STDOUT || msg.channel == ASYNC_SHELL {
                        let text = String::from_utf8_lossy(&msg.payload);
                        eprint!("{}", text);
                    }
                    output.feed(&msg);
                }
            }

            // Check file notifications
            while let Ok(path) = file_rx.try_recv() {
                log::info!("file written: {}", path.display());
                files_written.push(path.clone());

                if let Some(ref wait_name) = wait_for_owned {
                    if let Some(fname) = path.file_name() {
                        if fname.to_string_lossy() == *wait_name {
                            break;
                        }
                    }
                }
            }

            // Check if the wait-for file was received (after draining)
            if let Some(ref wait_name) = wait_for_owned {
                let found = files_written.iter().any(|p| {
                    p.file_name()
                        .map(|f| f.to_string_lossy() == *wait_name)
                        .unwrap_or(false)
                });
                if found {
                    break ExitReason::FileReceived(
                        files_written
                            .iter()
                            .find(|p| {
                                p.file_name()
                                    .map(|f| f.to_string_lossy() == *wait_name)
                                    .unwrap_or(false)
                            })
                            .cloned()
                            .unwrap(),
                    );
                }
            }

            thread::sleep(Duration::from_millis(10));
        };

        // Signal threads to stop
        shutdown.store(true, Ordering::Relaxed);
        drop(async_rx); // unblock reader if it's sending
        let _ = reader_handle.join();
        let _ = hostfs_handle.join();

        Ok(Execution {
            stdout: output.stdout_str(),
            stderr: output.stderr_str(),
            files_written,
            exit_reason,
        })
    }
}

/// Internal: a HostFS packet from the reader thread to the HostFS thread.
struct HostFsPacket {
    data: Vec<u8>,
    expected_extra: usize,
}

/// Guess the fixed command struct size based on the command ID.
/// This determines where extra data begins in a combined USB transfer.
fn guess_cmd_struct_size(data: &[u8]) -> usize {
    if data.len() < 12 {
        return data.len();
    }
    let cmd = read_u32_le(data, 4);
    match cmd {
        HELLO_CMD | BYE_CMD => 12,
        OPEN_CMD => 24,  // header(12) + mode(4) + mask(4) + fsnum(4)
        CLOSE_CMD => 16, // header(12) + fd(4)
        READ_CMD => 20,  // header(12) + fd(4) + len(4)
        WRITE_CMD => 16, // header(12) + fd(4)
        LSEEK_CMD => 28, // header(12) + fd(4) + offset(8) + whence(4)
        REMOVE_CMD | RMDIR_CMD | DOPEN_CMD | GETSTAT_CMD | CHDIR_CMD => 16,
        MKDIR_CMD => 20, // header(12) + mode(4) + fsnum(4)
        DREAD_CMD | DCLOSE_CMD => 16,
        IOCTL_CMD | DEVCTL_CMD => 24,
        _ => 16, // fallback: header + one i32
    }
}
