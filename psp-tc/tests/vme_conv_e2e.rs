//! End-to-end check of the VME int8-conv job on the Verilated RTL: build
//! the machine image exactly the way the device kernel does (context from
//! `vme_conv1x1_ctx`, buffers written directly into the image), execute it
//! with `vme-emu`, and recover the dot products by differencing the
//! cumulative MacI stream.
//!
//! This is the guard for the three assumptions the whole offload rests on:
//! the per-pixel write-replay collapse (step 0, seg K, stride 1), the
//! weight replay against a linear activation stream, and 24-bit *wrapping*
//! of the cumulative sum (differences then recover exact dots).
//!
//! Skipped when the simulator binary is absent; build it with
//! `make -C vme-emu build/vme-emu`, or point `VME_EMU` at it.

use std::path::PathBuf;
use std::process::Command;

use psp_tc::vme_conv::{plan_vme_conv1x1, reference_dots, LANES};
use vme_assembler::assemble::{CTX_OFFSET, IMAGE_SIZE};
use vme_assembler::{Buffer, MachineImage};

fn vme_emu() -> Option<PathBuf> {
    let path = match std::env::var_os("VME_EMU") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vme-emu/build/vme-emu"),
    };
    if path.is_file() {
        Some(path)
    } else {
        eprintln!(
            "skipping: vme-emu not built ({}); make -C vme-emu build/vme-emu",
            path.display()
        );
        None
    }
}

fn run(img: &MachineImage, tag: &str) -> MachineImage {
    let emu = vme_emu().unwrap();
    let dir = std::env::temp_dir();
    let inp = dir.join(format!("vme-conv-{}-{}.bin", tag, std::process::id()));
    let outp = dir.join(format!("vme-conv-{}-{}.out.bin", tag, std::process::id()));
    img.write_to(&inp).unwrap();
    let status = Command::new(&emu).arg(&inp).arg(&outp).status().unwrap();
    assert!(status.success(), "vme-emu failed");
    let out = MachineImage::from_file(&outp).unwrap();
    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(outp.with_extension("bin.vcd"));
    let _ = std::fs::remove_file(&outp);
    out
}

/// Build the job image the way `psp_rt::kernels::vme_conv1x1` does on
/// device: zeros, activations into TOP_0, one weight row per BASE ring at
/// the plan's offset, context at its mapped offset.
fn build_image(
    k: usize,
    p: usize,
    acts: &[i32],
    weights: &[&[i32]; LANES],
) -> MachineImage {
    let plan = plan_vme_conv1x1(k).unwrap();
    assert!(p <= plan.pixels_per_job);
    let ctx = psp_tc::vme_conv::vme_conv1x1_ctx(k, p).unwrap();

    let mut bytes = vec![0u8; IMAGE_SIZE];
    let mut put_words = |byte_off: usize, words: &[i32]| {
        for (i, w) in words.iter().enumerate() {
            bytes[byte_off + 4 * i..byte_off + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
    };
    put_words(Buffer::Top0.image_offset(), &acts[..p * k]);
    for (lane, row) in weights.iter().enumerate() {
        let buf = [Buffer::Base0, Buffer::Base1, Buffer::Base2, Buffer::Base3][lane];
        put_words(buf.image_offset() + 4 * plan.weights_off, row);
    }
    let ctx_i32: Vec<i32> = ctx.iter().map(|w| *w as i32).collect();
    put_words(CTX_OFFSET, &ctx_i32);
    MachineImage::from_bytes(bytes).unwrap()
}

/// Difference the cumulative per-pixel stream back into dots, in wrapping
/// 24-bit arithmetic — the device kernel's readback pass.
fn dots_from_cumulative(cum: &[i32], p: usize) -> Vec<i32> {
    let mut prev = 0i64;
    (0..p)
        .map(|px| {
            let d = (cum[px] as i64 - prev) & 0xFF_FFFF;
            prev = cum[px] as i64;
            (((d ^ 0x80_0000) - 0x80_0000) as i32)
        })
        .collect()
}

fn lcg(seed: &mut u32) -> i32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 16) as i32 & 0xFF) - 128 // int8 range
}

fn check_shape(k: usize, p: usize, tag: &str) {
    let mut s = 0xC0FFEEu32 ^ (k as u32) << 4 ^ p as u32;
    let acts: Vec<i32> = (0..p * k).map(|_| lcg(&mut s)).collect();
    let rows: Vec<Vec<i32>> = (0..LANES).map(|_| (0..k).map(|_| lcg(&mut s)).collect()).collect();
    let weights: [&[i32]; LANES] = [&rows[0], &rows[1], &rows[2], &rows[3]];

    let out = run(&build_image(k, p, &acts, &weights), tag);
    let want = reference_dots(&acts, &weights, k, p);

    for (lane, buf) in [Buffer::Base0, Buffer::Base1, Buffer::Base2, Buffer::Base3]
        .iter()
        .enumerate()
    {
        let cum = out.read_buffer(*buf);
        let got = dots_from_cumulative(&cum, p);
        for px in 0..p {
            assert_eq!(
                got[px], want[px][lane],
                "{tag}: lane {lane} pixel {px} (cum stream: {:?})",
                &cum[..p.min(8)]
            );
        }
    }
}

#[test]
fn conv_job_small() {
    if vme_emu().is_none() {
        return;
    }
    check_shape(24, 4, "small");
}

#[test]
fn conv_job_birdnet_shapes() {
    if vme_emu().is_none() {
        return;
    }
    // The real dot lengths, at their full per-job pixel batches.
    check_shape(24, 85, "k24");
    check_shape(72, 28, "k72");
    check_shape(108, 18, "k108");
}

#[test]
fn conv_job_partial_batch() {
    if vme_emu().is_none() {
        return;
    }
    // Remainder batches use a smaller count; same context generator.
    check_shape(72, 5, "rem");
}

/// The wrap guard: drive the cumulative sum far past 2^23 and prove the
/// differences still recover exact dots (i.e. the datapath wraps rather
/// than saturates). Max-magnitude int8 data: each dot is 108·(-128·127)
/// ≈ -1.76 M, so 18 pixels sweep the cumulative past -2^24.
#[test]
fn conv_job_cumulative_wrap() {
    if vme_emu().is_none() {
        return;
    }
    let (k, p) = (108usize, 18usize);
    let acts: Vec<i32> = vec![-128; p * k];
    let rows: Vec<Vec<i32>> = (0..LANES).map(|_| vec![127; k]).collect();
    let weights: [&[i32]; LANES] = [&rows[0], &rows[1], &rows[2], &rows[3]];

    let out = run(&build_image(k, p, &acts, &weights), "wrap");
    let want = reference_dots(&acts, &weights, k, p);
    let cum = out.read_buffer(Buffer::Base0);
    let got = dots_from_cumulative(&cum, p);
    for px in 0..p {
        assert_eq!(got[px], want[px][0], "pixel {px}");
    }
}
