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

/// Sentinel filename that PSP programs write via HostFS to signal exit.
const EXIT_SENTINEL: &str = "__psp_exit";

/// Result of executing a PRX on the PSP.
pub struct Execution {
    pub stdout: String,
    pub stderr: String,
    pub exit_reason: ExitReason,
}

pub enum ExitReason {
    /// The PSP program called psp_ml::exit().
    ModuleExited,
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
        // Drain any stale data on EP1 before starting
        let mut drain_buf = vec![0u8; 65536];
        loop {
            match self.usb.read_ep1(&mut drain_buf, Duration::from_millis(100)) {
                Ok(n) => log::debug!("drained {} stale bytes from EP1", n),
                Err(_) => break,
            }
        }

        // Step 1: Send magic word
        log::debug!("handshake: sending magic to EP2");
        self.usb.write_ep2(&HOSTFS_MAGIC.to_le_bytes())?;

        // Step 2: Read HELLO command from PSP
        log::debug!("handshake: waiting for HELLO on EP1");
        let mut buf = vec![0u8; 512];
        let n = match self.usb.read_ep1(&mut buf, Duration::from_secs(5)) {
            Ok(n) => n,
            Err(e) => {
                log::error!("handshake: EP1 read failed: {}", e);
                return Err(Error::HandshakeFailed);
            }
        };
        log::debug!(
            "handshake: got {} bytes: {:02X?}",
            n,
            &buf[..n.min(32)]
        );
        if n < 12 {
            log::error!("handshake: response too short ({} bytes)", n);
            return Err(Error::HandshakeFailed);
        }
        let magic = read_u32_le(&buf, 0).unwrap_or(0);
        let cmd = read_u32_le(&buf, 4).unwrap_or(0);
        if magic != HOSTFS_MAGIC || cmd != HELLO_CMD {
            return Err(Error::Protocol(format!(
                "expected HELLO, got magic={magic:#010X} cmd={cmd:#010X}"
            )));
        }

        // Step 3: Send HELLO response back on EP2.
        // The PSP's send_hello_cmd() uses command_xchg which waits for this
        // response before calling set_ayncreq() to enable EP3.
        log::debug!("handshake: sending HELLO response on EP2");
        let resp = hostfs_header(HELLO_CMD, 0);
        self.usb.write_ep2(&resp)?;

        log::info!("handshake complete");
        Ok(())
    }

    /// Execute a PRX on the PSP and collect output.
    ///
    /// Returns when the PSP program calls `psp_ml::exit()`, or on timeout.
    pub fn execute(
        &self,
        prx_path: &str,
        timeout: Duration,
    ) -> Result<Execution, Error> {
        self.handshake()?;

        let shutdown = Arc::new(AtomicBool::new(false));

        // Channels
        let (hostfs_tx, hostfs_rx) = mpsc::channel::<HostFsPacket>();
        let (async_tx, async_rx) = mpsc::channel::<Vec<u8>>();
        let (file_tx, file_rx) = mpsc::channel::<PathBuf>();

        // --- Reader thread: reads EP1, classifies packets, dispatches ---
        // IMPORTANT: Only this thread reads from EP1. When a HostFS command
        // has extra data (e.g. a filename), we read it here before dispatching,
        // so the HostFS thread never races with us on EP1.
        let usb_r = Arc::clone(&self.usb);
        let shutdown_r = Arc::clone(&shutdown);
        let reader_handle = thread::spawn(move || {
            let mut buf = vec![0u8; 65536 + 512];
            while !shutdown_r.load(Ordering::Relaxed) {
                match usb_r.read_ep1(&mut buf, Duration::from_millis(100)) {
                    Ok(n) if n >= 4 => {
                        let mut data = buf[..n].to_vec();
                        log::debug!(
                            "EP1 read {} bytes: {:02X?}{}",
                            n,
                            &data[..n.min(64)],
                            if n > 64 { "..." } else { "" }
                        );
                        match classify_packet(&data) {
                            PacketKind::HostFs => {
                                // Parse the command header to get extra_len
                                let extra_len = read_u32_le(&data, 8)
                                    .map(|v| v as usize)
                                    .unwrap_or(0);

                                // If extra data didn't arrive in the same transfer,
                                // read it now before dispatching.
                                if extra_len > 0 {
                                    let cmd_size = guess_cmd_struct_size(&data);
                                    let have = data.len().saturating_sub(cmd_size);
                                    if have < extra_len {
                                        let remaining = extra_len - have;
                                        let mut tmp = vec![0u8; remaining];
                                        match usb_r
                                            .read_ep1(&mut tmp, Duration::from_secs(5))
                                        {
                                            Ok(m) => {
                                                log::debug!(
                                                    "extra read {} of {} bytes",
                                                    m,
                                                    remaining
                                                );
                                                data.extend_from_slice(&tmp[..m]);
                                            }
                                            Err(e) => {
                                                log::error!("extra read error: {}", e);
                                            }
                                        }
                                    }
                                }

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
        // This thread never reads from EP1 — the reader thread bundles extra data
        // into HostFsPacket before dispatching here.
        let usb_h = Arc::clone(&self.usb);
        let shutdown_h = Arc::clone(&shutdown);
        let root = self.root_dir.clone();
        let hostfs_handle = thread::spawn(move || {
            let mut fd_table = FdTable::new(root, Some(file_tx));
            while !shutdown_h.load(Ordering::Relaxed) {
                match hostfs_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(pkt) => {
                        let cmd_struct_size = guess_cmd_struct_size(&pkt.data);
                        let cmd_buf = &pkt.data[..cmd_struct_size.min(pkt.data.len())];

                        // Extra data (e.g. filename) follows the command struct
                        let extra_data = if pkt.expected_extra > 0
                            && pkt.data.len() > cmd_struct_size
                        {
                            let end =
                                (cmd_struct_size + pkt.expected_extra).min(pkt.data.len());
                            pkt.data[cmd_struct_size..end].to_vec()
                        } else {
                            Vec::new()
                        };

                        log::debug!(
                            "HostFS {} (extra {} bytes)",
                            cmd_name(read_u32_le(cmd_buf, 4).unwrap_or(0)),
                            extra_data.len()
                        );

                        let (resp_hdr, resp_extra) =
                            fd_table.handle_command(cmd_buf, &extra_data);

                        // Send header and extra as SEPARATE USB bulk writes.
                        // The PSP reads them as distinct transfers via sceUsbbdReqRecv.
                        if let Err(e) = usb_h.write_ep2(&resp_hdr) {
                            log::error!("EP2 write error (header): {}", e);
                            break;
                        }
                        if !resp_extra.is_empty() {
                            if let Err(e) = usb_h.write_ep2(&resp_extra) {
                                log::error!("EP2 write error (extra): {}", e);
                                break;
                            }
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

        // --- Main loop: collect output, wait for exit sentinel ---
        let mut output = OutputCapture::new();
        let mut exited = false;
        let start = Instant::now();

        let exit_reason = loop {
            if start.elapsed() > timeout {
                break ExitReason::Timeout;
            }

            // Drain async messages
            while let Ok(packet) = async_rx.try_recv() {
                if let Some(msg) = AsyncMessage::parse(&packet) {
                    if msg.channel == ASYNC_STDOUT {
                        let text = String::from_utf8_lossy(&msg.payload);
                        eprint!("{}", text);
                    } else if msg.channel == ASYNC_SHELL {
                        log::debug!("shell: {:?}", String::from_utf8_lossy(&msg.payload));
                    }
                    output.feed(&msg);
                }
            }

            // Check for exit sentinel from psp_ml::exit()
            while let Ok(path) = file_rx.try_recv() {
                log::info!("file event: {}", path.display());
                if path == PathBuf::from(EXIT_SENTINEL) {
                    exited = true;
                }
            }

            if exited {
                break ExitReason::ModuleExited;
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
    let cmd = match read_u32_le(data, 4) {
        Some(c) => c,
        None => return data.len(),
    };
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
