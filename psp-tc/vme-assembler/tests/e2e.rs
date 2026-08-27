//! End-to-end: assemble a machine image, execute it on the Verilated RTL
//! via `vme-emu`, and check the result buffers.  This is the guard that
//! keeps the assembler's timing model honest against the RTL -- if the
//! pipeline in `vme-emu/rtl/` changes, this fails until `timing.rs` is
//! re-tuned.
//!
//! Skipped (with a note) when the simulator binary is absent; build it with
//! `make -C vme-emu build/vme-emu`, or point `VME_EMU` at it.

use std::path::PathBuf;
use std::process::Command;
use vme_assembler::*;

fn vme_emu() -> Option<PathBuf> {
    let path = match std::env::var_os("VME_EMU") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vme-emu/build/vme-emu"),
    };
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("skipping: vme-emu not built ({}); make -C vme-emu build/vme-emu", path.display());
        None
    }
}

fn run(img: &MachineImage, tag: &str) -> MachineImage {
    let emu = vme_emu().unwrap();
    let dir = std::env::temp_dir();
    let inp = dir.join(format!("vme-e2e-{}-{}.bin", tag, std::process::id()));
    let outp = dir.join(format!("vme-e2e-{}-{}.out.bin", tag, std::process::id()));
    img.write_to(&inp).unwrap();
    let status = Command::new(&emu).arg(&inp).arg(&outp).status().unwrap();
    assert!(status.success(), "vme-emu failed");
    let out = MachineImage::from_file(&outp).unwrap();
    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(outp.with_extension("bin.vcd"));
    let _ = std::fs::remove_file(&outp);
    out
}

fn sext24(x: i64) -> i32 {
    (((x & 0xFF_FFFF) ^ 0x80_0000) - 0x80_0000) as i32
}

/// Rounded rescaled multiply, entirely derived-skew, in the
/// silicon-verified shape: front = TOP_0 data, back = own BASE_0 at a
/// disjoint offset.
#[test]
fn e2e_vmul_round() {
    if vme_emu().is_none() {
        return;
    }
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

    let out = run(&generate_config(&vme).unwrap(), "vmul");
    let base0 = out.read_buffer(Buffer::Base0);
    for i in 0..N {
        let exp = sext24(((i as i64 + 1) * (3 * i as i64 - 20) + 8) >> 4);
        assert_eq!(base0[i], exp, "BASE_0[{i}]");
    }
}

/// Two-element pipeline over the staging bus with derived skew ladder,
/// plus an FU1 clamp stage: PE0 multiplies, PE1 adds a buffer stream to
/// the product and clamps the sum in its secondary unit.
#[test]
fn e2e_staging_pipeline_with_fu1() {
    if vme_emu().is_none() {
        return;
    }
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
    pe0.write_disabled = true; // product travels by staging only
    let pe1 = vme.pe_mut(Pe::Pe1);
    pe1.fu0().set_back(Source::Primary(Pe::Pe0));
    pe1.fu0().set_front(Source::Buf(Buffer::Top2));
    pe1.fu0().set_op(Operation::new(Opcode::Add));
    pe1.fu1().set_back(Source::Primary(Pe::Pe1));
    pe1.fu1().set_op(Operation::new(Opcode::Clamp).a(120).b(-50));

    let out = run(&generate_config(&vme).unwrap(), "staging");
    let base1 = out.read_buffer(Buffer::Base1);
    for i in 0..N {
        let sum = (i as i64 + 1) * (2 * i as i64 - 9) + (100 - 7 * i as i64);
        let exp = sext24(sum.clamp(-50, 120));
        assert_eq!(base1[i], exp, "BASE_1[{i}]");
    }
}

/// Segment replay: a 4-coefficient vector against a 16-element stream.
#[test]
fn e2e_segment_replay() {
    if vme_emu().is_none() {
        return;
    }
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

    let out = run(&generate_config(&vme).unwrap(), "replay");
    let base0 = out.read_buffer(Buffer::Base0);
    for i in 0..N {
        let coeff = [-15i64, -5, 5, 15][i % 4];
        assert_eq!(base0[i], sext24((i as i64 + 2) * coeff), "BASE_0[{i}]");
    }
}
