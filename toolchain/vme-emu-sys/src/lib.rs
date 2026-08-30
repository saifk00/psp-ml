//! vme-emu-sys -- the VME RTL simulation, in-process.
//!
//! Links the Verilated model of `vme-emu/rtl/` as a static library (built by
//! build.rs; needs Verilator installed) and exposes one call:
//! [`vme_emu`] takes a machine image, performs the same bring-up +
//! TRIGGER + wait-for-VD sequence as the standalone `vme-emu` binary, and
//! returns the unpacked post-run buffers.  No subprocess, no temp files, no
//! VCD -- for waveforms, use the standalone binary.

use vme_assembler::{assemble::IMAGE_SIZE, MachineImage, VmeResult};

/// Cycle cap per run, matching the standalone driver's `VME_EMU_MAX_CYCLES`.
pub const MAX_CYCLES: u64 = 1 << 20;

extern "C" {
    fn vme_emu_run(
        image_in: *const u8,
        image_out: *mut u8,
        max_cycles: u64,
        cycles_out: *mut u64,
    ) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmeEmuError {
    /// DMA_STAT.VD never set within [`MAX_CYCLES`] -- a context that never
    /// completes (with the RTL as built, only possible via an RTL bug, since
    /// disabled AGUs report done immediately).
    Timeout,
    /// A context word read back wrong through the live window.
    ContextVerify,
    /// An unknown status from the shim.
    Unknown(i32),
}

impl std::fmt::Display for VmeEmuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VmeEmuError::Timeout => write!(f, "array never reported done within {MAX_CYCLES} cycles"),
            VmeEmuError::ContextVerify => write!(f, "context readback verification failed"),
            VmeEmuError::Unknown(c) => write!(f, "unknown simulator status {c}"),
        }
    }
}

impl std::error::Error for VmeEmuError {}

/// Execute one machine image on the RTL and return the unpacked buffers.
pub fn vme_emu(image: &MachineImage) -> Result<VmeResult, VmeEmuError> {
    Ok(run_image(image)?.0.result())
}

/// Like [`vme_emu`], but returns the whole post-run image (buffers, context,
/// DMA_STAT) plus the cycle count from trigger to done.
pub fn run_image(image: &MachineImage) -> Result<(MachineImage, u64), VmeEmuError> {
    let mut out = vec![0u8; IMAGE_SIZE];
    let mut cycles = 0u64;
    let rc = unsafe {
        vme_emu_run(image.bytes().as_ptr(), out.as_mut_ptr(), MAX_CYCLES, &mut cycles)
    };
    match rc {
        0 => Ok((MachineImage::from_bytes(out).expect("shim wrote a full image"), cycles)),
        1 => Err(VmeEmuError::Timeout),
        2 => Err(VmeEmuError::ContextVerify),
        c => Err(VmeEmuError::Unknown(c)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vme_assembler::*;

    /// The vme-assembler e2e VMUL case, run in-process.
    #[test]
    fn in_process_vmul() {
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

        let img = generate_config(&vme).unwrap();
        let (out, cycles) = run_image(&img).unwrap();
        assert!(cycles > 0 && cycles < 1000, "unexpected cycle count {cycles}");
        let result = out.result();
        for i in 0..N {
            let prod = (((i as i64 + 1) * (3 * i as i64 - 20)) + 8) >> 4;
            let exp = (((prod & 0xFF_FFFF) ^ 0x80_0000) - 0x80_0000) as i32;
            assert_eq!(result.base[0][i], exp, "BASE_0[{i}]");
        }
        // inputs come back untouched
        assert_eq!(result.top[0][..N], (1..=N as i32).collect::<Vec<_>>()[..]);
    }
}
