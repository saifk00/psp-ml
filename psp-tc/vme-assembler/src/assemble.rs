//! Validation and machine-image assembly: `VmeConfig` in, 1 MB image out.

use crate::config::{
    AguParams, Buffer, BufferInit, Fu, Pe, Source, Transform, VmeConfig, BUFFER_WORDS,
};
use crate::timing::{self, TimingPlan};
use crate::VmeError;

/// Image geometry: the VME address space 0x4400_0000-0x440F_FFFF, byte
/// offset = address - 0x4400_0000, words little-endian.
pub const IMAGE_SIZE: usize = 0x100000;
pub const CTX_OFFSET: usize = 0xF8000;
pub const CTX_WORDS: usize = 106;
pub const DMA_STAT_OFFSET: usize = 0xFF000;

/// The write-port drain delay line depth in the RTL (`vme_pe`'s MAX_DRAIN).
pub const MAX_DRAIN: u16 = 64;

/// A 1 MB VME machine image -- `vme-emu`'s input, and (with the post-run
/// buffers) its output.
pub struct MachineImage {
    bytes: Vec<u8>,
}

impl MachineImage {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn write_to<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        std::fs::write(path, &self.bytes)
    }

    /// Read an image back -- typically `vme-emu`'s output image, to get at
    /// the result buffers.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() != IMAGE_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("image is {} bytes, expected {}", bytes.len(), IMAGE_SIZE),
            ));
        }
        Ok(MachineImage { bytes })
    }

    pub fn word(&self, byte_offset: usize) -> u32 {
        u32::from_le_bytes(self.bytes[byte_offset..byte_offset + 4].try_into().unwrap())
    }

    fn set_word(&mut self, byte_offset: usize, v: u32) {
        self.bytes[byte_offset..byte_offset + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// One buffer's 2048 samples, as signed words.
    pub fn read_buffer(&self, b: Buffer) -> Vec<i32> {
        let off = b.image_offset();
        (0..BUFFER_WORDS).map(|w| self.word(off + 4 * w) as i32).collect()
    }

    /// The 106 context words.
    pub fn context_words(&self) -> [u32; CTX_WORDS] {
        std::array::from_fn(|i| self.word(CTX_OFFSET + 4 * i))
    }
}

/// Check a configuration for assembly: source/operation completeness, field
/// ranges, the write-clobber hazard, and a solvable skew schedule.  Returns
/// the derived [`TimingPlan`] so callers can inspect the schedule.
pub fn validate(cfg: &VmeConfig) -> Result<TimingPlan, Vec<VmeError>> {
    let mut errors = Vec::new();

    let mut any = false;
    for pe in Pe::ALL {
        let elem = cfg.pe(pe);
        if !elem.is_configured() {
            continue;
        }
        any = true;

        for (fu, unit) in [(Fu::Fu0, &elem.fu0), (Fu::Fu1, &elem.fu1)] {
            let Some(op) = unit.op else { continue };
            let name = format!("PE{}.{}", pe.index(), if fu == Fu::Fu0 { "FU0" } else { "FU1" });
            if unit.back.is_none() {
                errors.push(VmeError::MissingBack { fu: name.clone() });
            }
            if op.opcode.uses_front_stream() && unit.front.is_none() {
                errors.push(VmeError::MissingFront { fu: name.clone(), opcode: op.opcode });
            }
            if op.k > 63 {
                errors.push(VmeError::FieldRange { fu: name.clone(), field: "k", value: op.k as u32, max: 63 });
            }
            if op.sat > 31 {
                errors.push(VmeError::FieldRange { fu: name.clone(), field: "sat", value: op.sat as u32, max: 31 });
            }
            // staging sources must name a configured unit
            for src in [unit.back, unit.front].into_iter().flatten() {
                let producer = match src {
                    Source::Primary(p) => Some((p, Fu::Fu0)),
                    Source::Secondary(p) => Some((p, Fu::Fu1)),
                    Source::Buf(_) => None,
                };
                if let Some((p, pf)) = producer {
                    let punit = match pf {
                        Fu::Fu0 => &cfg.pe(p).fu0,
                        Fu::Fu1 => &cfg.pe(p).fu1,
                    };
                    if !punit.is_configured() {
                        errors.push(VmeError::UnconfiguredProducer {
                            fu: name.clone(),
                            producer: format!("PE{}.{}", p.index(), if pf == Fu::Fu0 { "FU0" } else { "FU1" }),
                        });
                    }
                }
            }
        }

        match cfg.count_for(pe) {
            None => errors.push(VmeError::MissingCount { pe: pe.index() }),
            Some(0) => errors.push(VmeError::FieldRange {
                fu: format!("PE{}", pe.index()), field: "count", value: 0, max: 0x10000,
            }),
            Some(n) if n > 0x10000 => errors.push(VmeError::FieldRange {
                fu: format!("PE{}", pe.index()), field: "count", value: n, max: 0x10000,
            }),
            _ => {}
        }

        if let Some(d) = elem.write.drain {
            if d > MAX_DRAIN {
                errors.push(VmeError::FieldRange {
                    fu: format!("PE{}", pe.index()), field: "drain", value: d as u32, max: MAX_DRAIN as u32,
                });
            }
        }
    }

    if !any {
        errors.push(VmeError::NothingConfigured);
    }

    // write-clobber hazard: PE p writes BASE_p unconditionally
    for pe in Pe::ALL {
        let elem = cfg.pe(pe);
        if !elem.is_configured() || elem.write_disabled {
            continue;
        }
        let victim = pe.write_buffer();
        for reader in Pe::ALL {
            for unit in [&cfg.pe(reader).fu0, &cfg.pe(reader).fu1] {
                if !unit.is_configured() {
                    continue;
                }
                for src in [unit.back, unit.front].into_iter().flatten() {
                    if src == Source::Buf(victim) {
                        errors.push(VmeError::WriteClobber {
                            writer: pe.index(),
                            buffer: victim,
                            reader: reader.index(),
                        });
                    }
                }
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    timing::solve(cfg)
}

/// Build the 106-word context register-file image for a validated
/// configuration and its solved timing.  This is the piece a device runtime
/// can reuse verbatim: these words go straight into the window at
/// 0x440F_8000 (or a 112-word upload image).
pub fn context_words(cfg: &VmeConfig, plan: &TimingPlan) -> [u32; CTX_WORDS] {
    let mut ctx = [0u32; CTX_WORDS];

    for pe in Pe::ALL {
        let p = pe.index();
        let elem = cfg.pe(pe);

        for (fu, unit) in [(Fu::Fu0, &elem.fu0), (Fu::Fu1, &elem.fu1)] {
            let (desc_idx, const_idx) = match fu {
                Fu::Fu0 => (p, 8 + 2 * p),
                Fu::Fu1 => (4 + p, 16 + 2 * p),
            };
            let Some(op) = unit.op else {
                ctx[desc_idx] = 0x0000_4000; // reset value: MOV
                continue;
            };
            let fsel = unit.front.map_or(0, |s| s.selector());
            let bsel = unit.back.map_or(0, |s| s.selector());
            ctx[desc_idx] = (fsel << 27)
                | (bsel << 22)
                | op.opcode.op_word(op.acc)
                | ((op.sat as u32) << 7)
                | ((op.round as u32) << 6)
                | (op.k as u32);
            ctx[const_idx] = op.a as u32;
            ctx[const_idx + 1] = op.b as u32;
            if fu == Fu::Fu1 {
                ctx[27] |= 0x8000_0000 >> p; // FU1EN, bit 31-p
            }
        }

        let count = cfg.count_for(pe).unwrap_or(1);
        let block = 33 + 18 * p;
        emit_agu(&mut ctx, block, &elem.read_top, plan.top_skew[p], count, false);
        emit_agu(&mut ctx, block + 6, &elem.read_base, plan.base_skew[p], count, false);
        emit_agu(&mut ctx, block + 12, &elem.write, plan.write_skew[p], count, true);
    }

    ctx[29] = 0x0000_3210; // ICN_SRCMAP identity
    ctx[30] = 0x0000_3210; // ICN_CFGMAP identity
    ctx[105] = 0x0000_0018; // CTX_END
    ctx
}

/// Emit one six-word AGU group.  `skew = None` leaves the port disabled
/// (E = 0, all words zero).
fn emit_agu(ctx: &mut [u32], at: usize, agu: &AguParams, skew: Option<u32>, count: u32, is_write: bool) {
    let Some(skew) = skew else { return };
    let mode = if agu.replay.is_some() { 0x02u32 } else { 0x04 };
    ctx[at] = 0x8000_0000 | (mode << 24) | (skew << 16) | agu.offset as u32;
    ctx[at + 1] = ((agu.step as u32) << 16) | (count - 1);
    let mut fmt0 = 0u32;
    let mut fmt1 = 0u32;
    if let Some(r) = agu.replay {
        ctx[at + 2] = (0x0001 << 16) | (r.seg_len as u32 - 1); // INNER0: CFG 1
        ctx[at + 3] = r.stride as u32;
        fmt0 |= 0x0002_0000; // RNG
    }
    match agu.transform {
        Transform::Linear => {}
        Transform::Reverse => fmt0 |= 0x1000_0000,
        Transform::Replicate => fmt0 |= 0x0021_0000,
        Transform::BitReversed { width } => fmt1 |= 0xA400_0000 | width as u32,
    }
    if is_write {
        if let Some(d) = agu.drain {
            fmt0 |= d as u32; // DRAIN
            fmt1 |= 0x0020_0000; // END token, required for DRAIN to act
        }
    }
    ctx[at + 4] = fmt0;
    ctx[at + 5] = fmt1;
}

/// Validate, solve the timing, run the buffer initialisers, and assemble
/// the machine image.
pub fn generate_config(cfg: &VmeConfig) -> Result<MachineImage, Vec<VmeError>> {
    let plan = validate(cfg)?;
    let mut img = MachineImage { bytes: vec![0u8; IMAGE_SIZE] };

    // buffers: run the initialisers, store in the 24-bit sample format
    for b in Buffer::ALL {
        let mut words = [0i32; BUFFER_WORDS];
        match &cfg.buffer(b).init {
            BufferInit::Zero => {}
            BufferInit::Data(d) => {
                for (i, v) in d.iter().take(BUFFER_WORDS).enumerate() {
                    words[i] = *v;
                }
            }
            BufferInit::Callback(f) => f(&mut words),
        }
        let off = b.image_offset();
        for (w, v) in words.iter().enumerate() {
            let sample = ((v & 0xFF_FFFF) ^ 0x80_0000) - 0x80_0000; // sign-extend 24-bit
            img.set_word(off + 4 * w, sample as u32);
        }
    }

    let ctx = context_words(cfg, &plan);
    for (i, w) in ctx.iter().enumerate() {
        img.set_word(CTX_OFFSET + 4 * i, *w);
    }
    Ok(img)
}
