#![allow(dead_code)] // a lab notebook of hardware experiments; superseded probes stay
//! Skew probe: measure the real VME's pipeline depth.
//!
//! Runs the conformance VMUL config on the device repeatedly, sweeping the
//! write-port skew, and reports for each skew how the device's BASE_0
//! compares against the mathematically expected products -- including the
//! shift at which they best align, which *is* the real machine's
//! address-issue-to-write-capture latency. Also checks that the staged
//! TOP_0 round-trips, separating "staging broken" from "timing wrong".
//!
//!     cargo run -p vme-conformance-host --release --bin vme-skew-probe

use psplink_connection::{PSPConnection, PspError, PspEvent, ShellFramer, ShellMarker};
use std::path::Path;
use std::time::Duration;
use vme_assembler::*;

const EVENT_TIMEOUT: Duration = Duration::from_secs(60);
const N: usize = 16;

fn build2(skew: u8, srcmap: Option<u32>) -> MachineImage {
    // Hardware-discipline variant: front = TOP bank, back = BASE bank
    // (mcidclan's proven arrangement), optionally with SRCMAP 0x4440.
    let mut vme = VmeConfig::new();
    vme.set_stream_len(N as u32);
    vme.buffer_mut(Buffer::Top0).set_callback(|b| {
        for i in 0..N {
            b[i] = i as i32 + 1;
        }
    });
    vme.buffer_mut(Buffer::Base1).set_callback(|b| {
        for i in 0..N {
            b[i] = 3 * i as i32 - 20;
        }
    });
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_front(Source::Buf(Buffer::Top0));
    pe0.fu0().set_back(Source::Buf(Buffer::Base1));
    pe0.fu0().set_op(Operation::new(Opcode::VMul));
    pe0.write.skew = Some(skew);
    let mut img = generate_config(&vme).unwrap();
    if let Some(m) = srcmap {
        // patch ICN_SRCMAP (context word 29) in place
        let mut bytes = img.bytes().to_vec();
        bytes[0xF8000 + 4 * 29..0xF8000 + 4 * 29 + 4].copy_from_slice(&m.to_le_bytes());
        img = MachineImage::from_bytes(bytes).unwrap();
    }
    img
}

fn build(skew: u8) -> MachineImage {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(N as u32);
    vme.buffer_mut(Buffer::Top0).set_callback(|b| {
        for i in 0..N {
            b[i] = i as i32 + 1;
        }
    });
    vme.buffer_mut(Buffer::Base1).set_callback(|b| {
        for i in 0..N {
            b[i] = 3 * i as i32 - 20;
        }
    });
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_back(Source::Buf(Buffer::Top0));
    pe0.fu0().set_front(Source::Buf(Buffer::Base1));
    pe0.fu0().set_op(Operation::new(Opcode::VMul)); // k = 0: raw products
    pe0.write.skew = Some(skew);
    generate_config(&vme).unwrap()
}


/// MAC-demo shape, but VMUL: front = TOP_0 (data n+1), back = BASE_0 read
/// at word offset 256 where the coefficients are staged (clear of the
/// write region at 0..N).
fn build3(skew: u8, srcmap: Option<u32>) -> MachineImage {
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
    pe0.fu0().set_op(Operation::new(Opcode::VMul));
    pe0.read_base.offset = 256;
    pe0.write.skew = Some(skew);
    pe0.allow_write_clobber = true;   // disjoint offsets: reads 256.., writes 0..
    patch_srcmap(generate_config(&vme).unwrap(), srcmap)
}

/// The hardware-proven MAC demo verbatim, expressed through the assembler:
/// MACI of TOP_0 (weights 1..16) x BASE_0[256..] (all 2s), write skew 9.
fn build_maci(skew: u8, srcmap: Option<u32>) -> MachineImage {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(N as u32);
    vme.buffer_mut(Buffer::Top0).set_callback(|b| {
        for i in 0..N {
            b[i] = i as i32 + 1;
        }
    });
    vme.buffer_mut(Buffer::Base0).set_callback(|b| {
        for i in 0..N {
            b[256 + i] = 2;
        }
    });
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_front(Source::Buf(Buffer::Top0));
    pe0.fu0().set_back(Source::Buf(Buffer::Base0));
    pe0.fu0().set_op(Operation::new(Opcode::MacI));
    pe0.read_base.offset = 256;
    pe0.write.skew = Some(skew);
    pe0.allow_write_clobber = true;   // disjoint offsets: reads 256.., writes 0..
    patch_srcmap(generate_config(&vme).unwrap(), srcmap)
}

fn patch_srcmap(img: MachineImage, srcmap: Option<u32>) -> MachineImage {
    let Some(m) = srcmap else { return img };
    let mut bytes = img.bytes().to_vec();
    bytes[0xF8000 + 4 * 29..0xF8000 + 4 * 29 + 4].copy_from_slice(&m.to_le_bytes());
    MachineImage::from_bytes(bytes).unwrap()
}

/// Run one image and print BASE_0's interesting windows plus any buffer
/// words that changed from what was staged.
fn report(s: &mut Session, name: &str, img: &MachineImage) {
    match s.run(img) {
        Ok(r) => {
            println!("{name}: base0[0..16]={:?}", &r.base[0][..N]);
            let staged = img.result();
            for b in Buffer::ALL {
                let now = r.buffer(b);
                let was = staged.buffer(b);
                let diffs: Vec<usize> = (0..2048).filter(|i| now[*i] != was[*i]).collect();
                if !diffs.is_empty() {
                    let show: Vec<String> = diffs
                        .iter()
                        .take(6)
                        .map(|i| format!("[{}]{}", i, now[*i]))
                        .collect();
                    println!("    {b:?}: {} words changed: {}", diffs.len(), show.join(" "));
                }
            }
        }
        Err(e) => println!("{name}: {e}"),
    }
}


fn build_staging(rtop1: u8, wr1: u8) -> MachineImage {
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
    // assemble with derived skews, then patch PE1's ports to the sweep
    // values -- deliberately mis-skewed contexts are the probe's business,
    // and the solver rightly refuses to derive them.
    let img = generate_config(&vme).unwrap();
    let img = patch_skew(img, 51, rtop1); // PE1 RTOP MODE
    patch_skew(img, 63, wr1) // PE1 WR MODE
}

fn patch_skew(img: MachineImage, ctx_word: usize, skew: u8) -> MachineImage {
    let mut bytes = img.bytes().to_vec();
    let off = 0xF8000 + 4 * ctx_word;
    let mut w = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    w = (w & !0x00FF_0000) | ((skew as u32) << 16);
    bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
    MachineImage::from_bytes(bytes).unwrap()
}

/// Fit device[m] = product[m+a] + top2[m+b] over small integer offsets.
fn fit(dev: &[i32]) -> (i32, i32, usize) {
    let product = |i: i64| (i + 1) * (2 * i - 9);
    let top2 = |i: i64| 100 - 7 * i;
    let mut best = (0, 0, 0usize);
    for a in -6i32..=8 {
        for b in -6i32..=8 {
            let hits = (0..N as i32)
                .filter(|m| {
                    let (pa, pb) = ((m + a) as i64, (m + b) as i64);
                    (0..N as i64).contains(&pa)
                        && (0..N as i64).contains(&pb)
                        && dev[*m as usize] as i64 == product(pa) + top2(pb)
                })
                .count();
            if hits > best.2 {
                best = (a, b, hits);
            }
        }
    }
    best
}


/// VMUL in the verified shape, but with an explicit TOP-port start offset.
/// TOP_0[p] = 1000 + p so the consumed position is directly readable.
fn build_topoff(top_offset: u16, extra: u16) -> MachineImage {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(16 + extra as u32);
    vme.buffer_mut(Buffer::Top0).set_callback(|b| {
        for p in 0..64 {
            b[p] = 1000 + p as i32;
        }
        // and mirror-side values so a wrapped read is recognizable
        for p in 2040..2048 {
            b[p] = -(p as i32);
        }
    });
    vme.buffer_mut(Buffer::Base0).set_callback(|b| {
        for i in 0..24 {
            b[256 + i] = 1; // coeff 1: result = consumed TOP_0 value
        }
    });
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_front(Source::Buf(Buffer::Top0));
    pe0.fu0().set_back(Source::Buf(Buffer::Base0));
    pe0.fu0().set_op(Operation::new(Opcode::VMul));
    pe0.read_top.offset = top_offset;
    pe0.read_base.offset = 256;
    pe0.allow_write_clobber = true;
    generate_config(&vme).unwrap()
}

/// The staging pipeline, but aligned by displacing TOP_2's *data* three
/// positions instead of shifting the AGU offset: value[e] staged at e+3.
fn build_staging_displaced(extra_count: u16, extra_wr: u16, extra_pe0: u16) -> MachineImage {
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
            b[i + 3] = 100 - 7 * i as i32;
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
    let img = generate_config(&vme).unwrap();
    // undo the assembler's own -3 offset shift on PE1's top port: emit the
    // plain user offset (0) so ONLY the data displacement aligns
    let mut bytes = img.bytes().to_vec();
    let off = 0xF8000 + 4 * 51; // PE1 RTOP MODE
    let w = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) & 0xFFFF_0000;
    bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
    let off = 0xF8000 + 4 * 52; // PE1 RTOP COUNT: optionally lengthen
    let w = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) + extra_count as u32;
    bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
    let off = 0xF8000 + 4 * 64; // PE1 WR COUNT: optionally lengthen
    let w = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) + extra_wr as u32;
    bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
    let off = 0xF8000 + 4 * 34; // PE0 RTOP COUNT: optionally lengthen
    let w = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) + extra_pe0 as u32;
    bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
    MachineImage::from_bytes(bytes).unwrap()
}

fn report_words(name: &str, img: &MachineImage, n: usize) {
    // uses the global session via a thread-local? no -- kept simple: this
    // helper is inlined at the call site instead.
    let _ = (name, img, n);
}


fn product_a(i: i64) -> i64 {
    (i + 1) * (3 * i - 20)
}

fn stage_ab(vme: &mut VmeConfig) {
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
}

/// FU0 multiplies, FU1 (driving the write port) adds 1000 to the tap.
fn build_fu1_tap(wr_skew: u8) -> MachineImage {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(N as u32);
    stage_ab(&mut vme);
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_front(Source::Buf(Buffer::Top0));
    pe0.fu0().set_back(Source::Buf(Buffer::Base0));
    pe0.fu0().set_op(Operation::new(Opcode::VMul));
    pe0.fu1().set_back(Source::Primary(Pe::Pe0));
    pe0.fu1().set_op(Operation::new(Opcode::AddI).b(1000));
    pe0.read_base.offset = 256;
    pe0.allow_write_clobber = true;
    patch_skew(generate_config(&vme).unwrap(), 45, wr_skew) // PE0 WR MODE
}

/// FU1 alone, fed straight from the buffers; FU0 left unconfigured.
fn build_fu1_direct(wr_skew: u8) -> MachineImage {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(N as u32);
    stage_ab(&mut vme);
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu1().set_front(Source::Buf(Buffer::Top0));
    pe0.fu1().set_back(Source::Buf(Buffer::Base0));
    pe0.fu1().set_op(Operation::new(Opcode::VMul));
    pe0.read_base.offset = 256;
    pe0.allow_write_clobber = true;
    patch_skew(generate_config(&vme).unwrap(), 45, wr_skew)
}

/// Best element shift d fitting dev[m] == f(m + d).
fn fit1(dev: &[i32], f: impl Fn(i64) -> i64) -> (i32, usize) {
    let mut best = (0i32, 0usize);
    for d in -8i32..=8 {
        let hits = (0..N as i32)
            .filter(|m| {
                let i = (*m + d) as i64;
                (0..N as i64).contains(&i) && dev[*m as usize] as i64 == f(i)
            })
            .count();
        if hits > best.1 {
            best = (d, hits);
        }
    }
    best
}

fn expected(i: i64) -> i32 {
    let p = (i + 1) * (3 * i - 20);
    (((p & 0xFF_FFFF) ^ 0x80_0000) - 0x80_0000) as i32
}

struct Session {
    conn: PSPConnection,
    jobs_dir: std::path::PathBuf,
    tail: String,
    framer: ShellFramer,
    exited: Option<ShellMarker>,
}

impl Session {
    fn pump_until(&mut self, mut stop: impl FnMut(&str) -> bool) -> Result<(), String> {
        loop {
            if stop(&self.tail) {
                return Ok(());
            }
            if let Some(m) = self.exited {
                return Err(format!("device exited early: {m:?}"));
            }
            match self.conn.recv_event(EVENT_TIMEOUT) {
                Ok(PspEvent::Stdout(b)) | Ok(PspEvent::Stderr(b)) => {
                    self.tail.push_str(&String::from_utf8_lossy(&b));
                }
                Ok(PspEvent::ShellRaw(b)) => {
                    if let Some(m) = self.framer.push(&b).into_iter().next() {
                        self.exited = Some(m);
                    }
                }
                Ok(PspEvent::Disconnected) => return Err("PSP disconnected".into()),
                Err(PspError::Timeout) => return Err("timed out".into()),
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    fn run(&mut self, image: &MachineImage) -> Result<VmeResult, String> {
        self.tail.clear();
        image.write_to(self.jobs_dir.join("job.bin")).map_err(|e| e.to_string())?;
        std::fs::write(self.jobs_dir.join("job.go"), b"").map_err(|e| e.to_string())?;
        self.pump_until(|s| s.contains("#vme result end") || s.contains("#vme error"))?;
        if let Some(err) = self.tail.lines().find(|l| l.starts_with("#vme error")) {
            return Err(err.to_string());
        }
        let body = self
            .tail
            .split("#vme result begin")
            .nth(1)
            .and_then(|s| s.split("#vme result end").next())
            .ok_or("malformed frame")?;
        let words: Vec<i32> = body
            .split_whitespace()
            .map(|t| u32::from_str_radix(t, 16).map(|w| w as i32))
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        VmeResult::from_words(&words)
    }
}

fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().unwrap();
    let prx_name = prx_path.file_name().unwrap().to_str().unwrap();

    let jobs_dir = std::env::temp_dir().join(format!("vme-probe-{}", std::process::id()));
    std::fs::create_dir_all(&jobs_dir).unwrap();

    eprintln!("==> Connecting...");
    let conn = PSPConnection::connect(&jobs_dir, prx_dir, Default::default()).unwrap();
    let _ = conn.send_shell_command(&[]); // flush the shell parser
    std::thread::sleep(Duration::from_millis(300));
    conn.send_shell_command(&["ld", &format!("host1:{prx_name}")]).unwrap();

    let mut s = Session {
        conn,
        jobs_dir: jobs_dir.clone(),
        tail: String::new(),
        framer: ShellFramer::new(),
        exited: None,
    };
    s.pump_until(|t| t.contains("#vme ready") || t.contains("#vme fatal")).unwrap();
    if s.tail.contains("#vme fatal") {
        eprintln!("plugin unavailable; see conformance harness");
        std::process::exit(1);
    }

    // FU1 timing.  (A) FU1 driven by its own PE's FU0 tap (AddI +1000 on
    // the product stream): sweep the write skew; the model predicts
    // alignment at 9 (= FU0's 6 + one staging hop).  (B) FU1 fed directly
    // from the buffers with FU0 idle: unmeasured; a plain FU would align
    // at 6.  d is the element shift that best fits the output.
    println!("-- FU1 via own FU0 tap: expect alignment (d=0) at wr skew 9 --");
    for wr in 6..=13u8 {
        match s.run(&build_fu1_tap(wr)) {
            Ok(r) => {
                let (d, hits) = fit1(&r.base[0][..N], |i| product_a(i) + 1000);
                println!("wr {wr:2}: d={d} ({hits}/16) base0={:?}", &r.base[0][..8]);
            }
            Err(e) => println!("wr {wr:2}: {e}"),
        }
    }
    println!("-- FU1 direct from buffers (FU0 idle): alignment at wr 6? --");
    for wr in 3..=10u8 {
        match s.run(&build_fu1_direct(wr)) {
            Ok(r) => {
                let (d, hits) = fit1(&r.base[0][..N], product_a);
                println!("wr {wr:2}: d={d} ({hits}/16) base0={:?}", &r.base[0][..8]);
            }
            Err(e) => println!("wr {wr:2}: {e}"),
        }
    }

    let _ = std::fs::write(jobs_dir.join("quit.go"), b"");
    let _ = s.pump_until(|t| t.contains("#vme quit"));
    let _ = std::fs::remove_dir_all(&jobs_dir);
}
