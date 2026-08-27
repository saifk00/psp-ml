//! vme-assembler -- configure the PSP's Virtual Mobile Engine as an object,
//! then assemble it into a machine image.
//!
//! The VME (see `docs/vme-reference.html`) is not instruction-driven: a
//! 106-word context wires four processing elements, their address
//! generators and eight ring buffers into a dataflow graph.  This crate
//! provides the friendly half of that contract:
//!
//! * [`VmeConfig`] -- a builder for the array: per-PE functional units and
//!   operand routing, per-port addressing patterns, buffer initialisers.
//! * [`generate_config`] -- validates the configuration ([`validate`] runs
//!   the same checks standalone), derives every cycle skew from the
//!   dataflow graph and the RTL latency model in [`timing`], and emits the
//!   1 MB machine image that `vme-emu` (in `vme-emu/`) executes.
//! * [`assemble::context_words`] -- just the 106 context words, the piece a
//!   device runtime loads into the real block at `0x440F_8000`.
//!
//! The skew derivation is deliberately centralised: the RTL's pipeline
//! depths live in [`timing`] as two constants, so the numbers can be
//! re-tuned there as the RTL (or measurements of the real hardware) evolve.

pub mod assemble;
pub mod config;
pub mod opcode;
pub mod timing;

#[cfg(test)]
mod tests;

pub use assemble::{context_words, generate_config, validate, MachineImage, VmeResult};
pub use config::{
    AguParams, Buffer, BufferInit, BufferSlot, Fu, FunctionalUnit, Operation, Pe,
    ProcessingElement, Replay, Source, Transform, VmeConfig, BUFFER_WORDS,
};
pub use opcode::{AccMode, Opcode};
pub use timing::TimingPlan;

use std::fmt;

/// Everything [`validate`] / [`generate_config`] can reject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VmeError {
    /// No functional unit on any element has an operation configured.
    NothingConfigured,
    /// A unit has an operation but no back source.  The back operand paces
    /// the unit -- even `MovI`/`Ramp` need a streaming back source -- and
    /// the hardware default (TOP_0) computes the right function on the
    /// wrong data, so it is never filled in silently.
    MissingBack { fu: String },
    /// The operation reads the front stream but no front source is set
    /// (same silent-TOP_0 hazard as above).
    MissingFront { fu: String, opcode: Opcode },
    /// A staging source names a functional unit with no operation.
    UnconfiguredProducer { fu: String, producer: String },
    /// No stream length: set it per element or via
    /// [`VmeConfig::set_stream_len`].
    MissingCount { pe: usize },
    /// A descriptor or AGU field is out of range.
    FieldRange { fu: String, field: &'static str, value: u32, max: u32 },
    /// A base-bank read names a buffer that is not the element's own:
    /// measured on silicon, PE n's base-bank port reads BASE_n regardless
    /// of the selector index (the manual's full-crossbar claim is wrong for
    /// the base bank).  Stage that input in a TOP buffer, or in this PE's
    /// own BASE_n at a disjoint offset.
    BaseAffinity { fu: String, buffer: Buffer, pe: usize },
    /// PE `writer` writes BASE_n unconditionally, and something reads that
    /// buffer in the same pass -- the read races the write.  Stage the
    /// input in a different buffer.
    WriteClobber { writer: usize, buffer: Buffer, reader: usize },
    /// The staging graph has a cycle; feed a loop through a buffer across
    /// two passes instead.
    StagingCycle,
    /// Two streams into one unit cannot be aligned (e.g. one port needed
    /// at two different offset shifts).  Restage one leg, or route one
    /// operand through a buffer.
    SkewConflict { fu: String, a: String, a_cycle: u32, b: String, b_cycle: u32 },
    /// A staging source is wired to the front operand where it cannot be
    /// aligned: staging must be the back (pacing) operand, since a buffer
    /// partner can be advanced by an offset shift but a tap cannot.
    FrontStaging { fu: String, producer: String },
    /// FU1's back operand names a buffer.  Measured on silicon: a
    /// buffer-fed FU1 (FU0 idle) produces nothing at any skew -- the
    /// secondary unit consumes the staging bus, not the read ports
    /// (a buffer *front* on FU1 is untested; the back is what paces).
    Fu1BufferBack { pe: usize, buffer: Buffer },
    /// A staging consumer's buffer leg needs its data rotated for
    /// alignment, which requires the port's start offset to be 0.
    ShiftedOffset { pe: usize, port: &'static str },
    /// A read-port skew was requested, but read AGUs ignore the skew field
    /// on real hardware -- alignment happens via start-offset shifts, which
    /// the assembler derives automatically.
    ReadSkewUnsupported { pe: usize, port: &'static str },
    /// A derived or manual skew exceeds the 8-bit MODE field.
    SkewRange { fu: String, skew: u32 },
}

impl fmt::Display for VmeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmeError::NothingConfigured => {
                write!(f, "no functional unit is configured; nothing to run")
            }
            VmeError::MissingBack { fu } => {
                write!(f, "{fu}: no back source; the back operand paces the unit, set one explicitly")
            }
            VmeError::MissingFront { fu, opcode } => {
                write!(f, "{fu}: {opcode:?} reads the front stream but no front source is set")
            }
            VmeError::UnconfiguredProducer { fu, producer } => {
                write!(f, "{fu}: staging source {producer} has no operation configured")
            }
            VmeError::MissingCount { pe } => {
                write!(f, "PE{pe}: no stream length; set_count() or VmeConfig::set_stream_len()")
            }
            VmeError::FieldRange { fu, field, value, max } => {
                write!(f, "{fu}: {field} = {value} exceeds {max}")
            }
            VmeError::BaseAffinity { fu, buffer, pe } => write!(
                f,
                "{fu}: {buffer:?} is not PE{pe}'s own base buffer; the base-bank read \
                 port has own-lane affinity on real hardware -- stage this input in a \
                 TOP buffer or in BASE_{pe} at a disjoint offset"
            ),
            VmeError::WriteClobber { writer, buffer, reader } => write!(
                f,
                "PE{writer} writes {buffer:?} while PE{reader} reads it; stage that input elsewhere"
            ),
            VmeError::StagingCycle => write!(
                f,
                "staging sources form a cycle; run feedback through a buffer across two passes"
            ),
            VmeError::SkewConflict { fu, a, a_cycle, b, b_cycle } => write!(
                f,
                "{fu}: cannot align {a} (cycle {a_cycle}) with {b} (cycle {b_cycle}); \
                 restage one leg or route it through a buffer"
            ),
            VmeError::SkewRange { fu, skew } => {
                write!(f, "{fu}: derived write skew {skew} exceeds the 8-bit MODE field")
            }
            VmeError::FrontStaging { fu, producer } => write!(
                f,
                "{fu}: staging source {producer} must be the back operand (staging                  fronts cannot be aligned; swap the operands or restage via a buffer)"
            ),
            VmeError::Fu1BufferBack { pe, buffer } => write!(
                f,
                "PE{pe}.FU1: back operand {buffer:?} is a buffer, but the secondary                  unit only consumes the staging bus (measured); route the buffer                  through FU0 and feed FU1 its tap"
            ),
            VmeError::ShiftedOffset { pe, port } => write!(
                f,
                "PE{pe}.{port}: a staging-aligned read must start at offset 0 (its                  staged data is rotated for alignment instead)"
            ),
            VmeError::ReadSkewUnsupported { pe, port } => write!(
                f,
                "PE{pe}.{port}: read AGUs ignore the skew field on real hardware;                  remove the override (alignment is derived via offset shifts)"
            ),
        }
    }
}

impl std::error::Error for VmeError {}
