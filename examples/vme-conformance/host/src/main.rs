//! Host half of the VME conformance harness.
//!
//! Each test builds one `VmeConfig`, assembles it, and runs the *same*
//! machine image twice: on the real VME (shipped to the device server over
//! the mounted filesystem, results streamed back over stdout) and on the
//! Verilated RTL in-process (`vme_emu_sys::vme_emu`). The two `VmeResult`s
//! are diffed over each test's declared regions — regions outside a write
//! port's valid window carry pipeline junk on both sides and are expected
//! to differ.
//!
//! As of 2026-08-27 all three tests pass bit-exact: the RTL and
//! `vme_assembler::timing` are calibrated to silicon (write skew 6,
//! staging hop +3, base-bank affinity, shared read halt, data-rotation
//! alignment -- see vme-emu/README.md).  A future mismatch means the model
//! drifted or a new mechanism was touched; the probes in
//! `src/bin/probe.rs` are the measurement tools that closed the gap.

use psplink_connection::{PSPConnection, PspError, PspEvent, ShellFramer, ShellMarker};
use std::path::{Path, PathBuf};
use std::time::Duration;
use vme_assembler::*;

const EVENT_TIMEOUT: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------
// the tests: a config plus the buffer regions whose contents must agree
// ---------------------------------------------------------------------

struct Check {
    buffer: Buffer,
    range: std::ops::Range<usize>,
}

fn test_vmul() -> (VmeConfig, Vec<Check>) {
    const N: usize = 16;
    let mut vme = VmeConfig::new();
    vme.set_stream_len(N as u32);
    vme.buffer_mut(Buffer::Top0).set_callback(|b| {
        for i in 0..N {
            b[i] = i as i32 + 1;
        }
    });
    vme.buffer_mut(Buffer::Base0).set_callback(|b| {
        for i in 0..N {
            b[256 + i] = 3 * i as i32 - 20;
        }
    });
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_front(Source::Buf(Buffer::Top0));
    pe0.fu0().set_back(Source::Buf(Buffer::Base0));
    pe0.fu0().set_op(Operation::new(Opcode::VMul).k(4).round());
    pe0.read_base.offset = 256;
    pe0.allow_write_clobber = true;
    (vme, vec![Check { buffer: Buffer::Base0, range: 0..N }])
}

fn test_staging_pipeline() -> (VmeConfig, Vec<Check>) {
    const N: usize = 16;
    let mut vme = VmeConfig::new();
    vme.set_stream_len(N as u32);
    vme.buffer_mut(Buffer::Top0).set_callback(|b| {
        for i in 0..N {
            b[i] = i as i32 + 1;
        }
    });
    vme.buffer_mut(Buffer::Top1).set_callback(|b| {
        for i in 0..N {
            b[i] = 2 * i as i32 - 9;
        }
    });
    vme.buffer_mut(Buffer::Top2).set_callback(|b| {
        for i in 0..N {
            b[i] = 100 - 7 * i as i32;
        }
    });
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_back(Source::Buf(Buffer::Top0));
    pe0.fu0().set_front(Source::Buf(Buffer::Top1));
    pe0.fu0().set_op(Operation::new(Opcode::VMul));
    pe0.write_disabled = true;
    let pe1 = vme.pe_mut(Pe::Pe1);
    pe1.fu0().set_back(Source::Primary(Pe::Pe0));
    pe1.fu0().set_front(Source::Buf(Buffer::Top2));
    pe1.fu0().set_op(Operation::new(Opcode::Add));
    (vme, vec![Check { buffer: Buffer::Base1, range: 0..N }])
}

fn test_segment_replay() -> (VmeConfig, Vec<Check>) {
    const N: usize = 16;
    let mut vme = VmeConfig::new();
    vme.set_stream_len(N as u32);
    vme.buffer_mut(Buffer::Top0).set_callback(|b| {
        for i in 0..N {
            b[i] = i as i32 + 2;
        }
    });
    vme.buffer_mut(Buffer::Base0).set_callback(|b| {
        b[256..260].copy_from_slice(&[-15, -5, 5, 15]);
    });
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_front(Source::Buf(Buffer::Top0));
    pe0.fu0().set_back(Source::Buf(Buffer::Base0));
    pe0.fu0().set_op(Operation::new(Opcode::VMul));
    pe0.read_base.offset = 256;
    pe0.read_base.replay = Some(Replay { seg_len: 4, stride: 0 });
    pe0.allow_write_clobber = true;
    (vme, vec![Check { buffer: Buffer::Base0, range: 0..N }])
}

// ---------------------------------------------------------------------
// device session: ld the server, feed it jobs, parse framed results
// ---------------------------------------------------------------------

struct DeviceSession {
    conn: PSPConnection,
    jobs_dir: PathBuf,
    stdout_tail: String,
    framer: ShellFramer,
    exited: Option<ShellMarker>,
}

impl DeviceSession {
    /// Pump events until `stop` says the accumulated stdout is enough.
    /// Module exit while waiting is an error surfaced to the caller.
    fn pump_until(
        &mut self,
        mut stop: impl FnMut(&str) -> bool,
    ) -> Result<(), String> {
        loop {
            if stop(&self.stdout_tail) {
                return Ok(());
            }
            if let Some(m) = self.exited {
                return Err(format!("device exited early: {m:?}"));
            }
            match self.conn.recv_event(EVENT_TIMEOUT) {
                Ok(PspEvent::Stdout(b)) | Ok(PspEvent::Stderr(b)) => {
                    print!("{}", summarize(&b));
                    self.stdout_tail.push_str(&String::from_utf8_lossy(&b));
                }
                Ok(PspEvent::ShellRaw(b)) => {
                    if let Some(m) = self.framer.push(&b).into_iter().next() {
                        self.exited = Some(m);
                    }
                }
                Ok(PspEvent::Disconnected) => return Err("PSP disconnected".into()),
                Err(PspError::Timeout) => return Err("timed out waiting for device".into()),
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    /// Ship one image to the device server and parse the result frame,
    /// retrying when the frame arrives damaged (a dropped stdout chunk).
    fn run(&mut self, image: &MachineImage) -> Result<VmeResult, String> {
        let mut last = String::new();
        for _ in 0..3 {
            match self.run_once(image) {
                Ok(r) => return Ok(r),
                Err(e) if e.contains("expected") || e.contains("bad hex") => last = e,
                Err(e) => return Err(e),
            }
        }
        Err(format!("{last} (after retries)"))
    }

    fn run_once(&mut self, image: &MachineImage) -> Result<VmeResult, String> {
        self.stdout_tail.clear();
        image
            .write_to(self.jobs_dir.join("job.bin"))
            .map_err(|e| format!("writing job.bin: {e}"))?;
        std::fs::write(self.jobs_dir.join("job.go"), b"")
            .map_err(|e| format!("writing job.go: {e}"))?;

        self.pump_until(|s| s.contains("#vme result end") || s.contains("#vme error"))?;
        if let Some(err) = self.stdout_tail.lines().find(|l| l.starts_with("#vme error")) {
            return Err(err.to_string());
        }

        let body = self
            .stdout_tail
            .split("#vme result begin")
            .nth(1)
            .and_then(|s| s.split("#vme result end").next())
            .ok_or("malformed result frame")?;
        let words: Vec<i32> = body
            .split_whitespace()
            .map(|t| u32::from_str_radix(t, 16).map(|w| w as i32))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("bad hex word in result frame: {e}"))?;
        VmeResult::from_words(&words)
    }

    /// Send the exit sentinel and wait for the module to complete.
    fn quit(mut self) -> Result<(), String> {
        let _ = std::fs::write(self.jobs_dir.join("quit.go"), b"");
        let r = self.pump_until(|_| false);
        match self.exited {
            Some(ShellMarker::Success(0)) => Ok(()),
            Some(m) => Err(format!("device exit: {m:?}")),
            None => r.map(|_| ()),
        }
    }
}

/// Keep the raw device stream visible without drowning the report: pass
/// protocol/log lines through, count the hex-dump lines silently.
fn summarize(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter(|l| !l.trim_start().chars().next().map_or(true, |c| c.is_ascii_hexdigit()))
        .map(|l| format!("    [psp] {l}\n"))
        .collect()
}

// ---------------------------------------------------------------------

fn compare(name: &str, checks: &[Check], device: &VmeResult, sim: &VmeResult) -> bool {
    let mut ok = true;
    for c in checks {
        let d = &device.buffer(c.buffer)[c.range.clone()];
        let s = &sim.buffer(c.buffer)[c.range.clone()];
        let diffs: Vec<usize> = (0..d.len()).filter(|i| d[*i] != s[*i]).collect();
        if diffs.is_empty() {
            println!("    {:?}[{:?}]: device == RTL", c.buffer, c.range);
        } else {
            ok = false;
            println!(
                "    {:?}[{:?}]: {} of {} words differ",
                c.buffer,
                c.range,
                diffs.len(),
                d.len()
            );
            for i in diffs.iter().take(4) {
                println!(
                    "      [{}] device {:08x}  rtl {:08x}",
                    c.range.start + i,
                    d[*i] as u32,
                    s[*i] as u32
                );
            }
        }
    }
    println!("==> {name}: {}", if ok { "PASS" } else { "FAIL" });
    ok
}

fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().expect("PRX_PATH has no parent");
    let prx_name = prx_path.file_name().unwrap().to_str().unwrap();

    let jobs_dir = std::env::temp_dir().join(format!("vme-conformance-{}", std::process::id()));
    std::fs::create_dir_all(&jobs_dir).expect("creating jobs dir");

    eprintln!("==> Connecting...");
    let conn = PSPConnection::connect(&jobs_dir, prx_dir, Default::default())
        .unwrap_or_else(|e| {
            eprintln!("error: failed to connect to PSP: {e}");
            std::process::exit(1);
        });

    eprintln!("==> Starting device server: ld host1:{prx_name}");
    // Flush psplink's shell parser first: residual bytes from a previous
    // session otherwise eat the first command's leading args (the harmless
    // "unknown command" line in the log is this terminator landing).
    let _ = conn.send_shell_command(&[]);
    std::thread::sleep(Duration::from_millis(300));
    conn.send_shell_command(&["ld", &format!("host1:{prx_name}")])
        .expect("sending ld");

    let mut session = DeviceSession {
        conn,
        jobs_dir: jobs_dir.clone(),
        stdout_tail: String::new(),
        framer: ShellFramer::new(),
        exited: None,
    };
    if let Err(e) = session.pump_until(|s| s.contains("#vme ready") || s.contains("#vme fatal")) {
        eprintln!("error: device server never came up: {e}");
        std::process::exit(1);
    }
    if session.stdout_tail.contains("#vme fatal") {
        eprintln!("error: device reports the VME plugin is unavailable — build it");
        eprintln!("  (make -C kernel-plugin-vme), install psp_vme_kernel.prx to");
        eprintln!("  ms0:/seplugins (cargo run -p vme-install-host --release), and");
        eprintln!("  power-cycle the PSP.");
        std::process::exit(1);
    }

    let tests: [(&str, fn() -> (VmeConfig, Vec<Check>)); 3] = [
        ("vmul-round", test_vmul),
        ("staging-pipeline", test_staging_pipeline),
        ("segment-replay", test_segment_replay),
    ];

    let mut failures = 0;
    for (name, build) in tests {
        println!("==> {name}");
        let (cfg, checks) = build();
        let image = match generate_config(&cfg) {
            Ok(img) => img,
            Err(errs) => {
                for e in errs {
                    println!("    assemble error: {e}");
                }
                failures += 1;
                continue;
            }
        };
        let sim = match vme_emu_sys::vme_emu(&image) {
            Ok(r) => r,
            Err(e) => {
                println!("    RTL simulation failed: {e}");
                failures += 1;
                continue;
            }
        };
        match session.run(&image) {
            Ok(device) => {
                if !compare(name, &checks, &device, &sim) {
                    failures += 1;
                }
            }
            Err(e) => {
                println!("==> {name}: DEVICE ERROR: {e}");
                failures += 1;
            }
        }
    }

    eprintln!("==> Stopping device server");
    if let Err(e) = session.quit() {
        eprintln!("warning: {e}");
    }
    let _ = std::fs::remove_dir_all(&jobs_dir);

    if failures > 0 {
        eprintln!("==> {failures} test(s) FAILED (a real-vs-RTL gap is this tool's yield -- see vme-emu/README.md)");
        std::process::exit(1);
    }
    eprintln!("==> All tests passed");
}
