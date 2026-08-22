//! The user-facing configuration model: manipulate the VME as an object,
//! then commit it to a machine image with [`crate::generate_config`].
//!
//! ```no_run
//! use vme_assembler::*;
//!
//! let mut vme = VmeConfig::new();
//! vme.set_stream_len(16);
//! vme.buffer_mut(Buffer::Top0).set_callback(|buf| {
//!     for (i, w) in buf.iter_mut().enumerate() { *w = i as i32; }
//! });
//! let pe0 = vme.pe_mut(Pe::Pe0);
//! pe0.fu0().set_back(Source::Buf(Buffer::Top0));
//! pe0.fu0().set_front(Source::Buf(Buffer::Base1));
//! pe0.fu0().set_op(Operation::new(Opcode::VMul).k(23).round());
//! let image = generate_config(&vme).unwrap();
//! image.write_to("dot.bin").unwrap();
//! ```

use crate::opcode::{AccMode, Opcode};

pub const BUFFER_WORDS: usize = 2048;

/// One of the eight 8 KB ring buffers (Table 2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buffer {
    Top0,
    Top1,
    Top2,
    Top3,
    Base0,
    Base1,
    Base2,
    Base3,
}

impl Buffer {
    pub const ALL: [Buffer; 8] = [
        Buffer::Top0,
        Buffer::Top1,
        Buffer::Top2,
        Buffer::Top3,
        Buffer::Base0,
        Buffer::Base1,
        Buffer::Base2,
        Buffer::Base3,
    ];

    pub fn is_top(self) -> bool {
        matches!(self, Buffer::Top0 | Buffer::Top1 | Buffer::Top2 | Buffer::Top3)
    }

    /// Index within its bank, 0-3.
    pub fn lane(self) -> usize {
        match self {
            Buffer::Top0 | Buffer::Base0 => 0,
            Buffer::Top1 | Buffer::Base1 => 1,
            Buffer::Top2 | Buffer::Base2 => 2,
            Buffer::Top3 | Buffer::Base3 => 3,
        }
    }

    /// The 5-bit FSEL/BSEL selector value (section 5.2): TOP at 0x00-0x06,
    /// BASE at 0x08-0x0E, bit 0 unused for buffers.
    pub fn selector(self) -> u32 {
        let idx = if self.is_top() { self.lane() } else { 4 + self.lane() };
        (idx as u32) << 1
    }

    /// Byte offset of this buffer's storage within the 1 MB machine image.
    pub fn image_offset(self) -> usize {
        let bank = if self.is_top() { 0x20000 } else { 0x00000 };
        bank + self.lane() * 0x2000
    }
}

/// One of the four processing elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pe {
    Pe0,
    Pe1,
    Pe2,
    Pe3,
}

impl Pe {
    pub const ALL: [Pe; 4] = [Pe::Pe0, Pe::Pe1, Pe::Pe2, Pe::Pe3];

    pub fn index(self) -> usize {
        match self {
            Pe::Pe0 => 0,
            Pe::Pe1 => 1,
            Pe::Pe2 => 2,
            Pe::Pe3 => 3,
        }
    }

    /// The buffer this element's write port lands in -- PEn writes BASE_n,
    /// there is no destination selector (section 4.2).
    pub fn write_buffer(self) -> Buffer {
        match self {
            Pe::Pe0 => Buffer::Base0,
            Pe::Pe1 => Buffer::Base1,
            Pe::Pe2 => Buffer::Base2,
            Pe::Pe3 => Buffer::Base3,
        }
    }
}

/// Which of a PE's two functional units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Fu {
    Fu0,
    Fu1,
}

/// An operand source: a ring buffer, or another functional unit's result
/// off the staging bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Buffer memory, addressed by this PE's bank read AGU.
    Buf(Buffer),
    /// The named PE's primary (FU0) result -- staging taps 0-3.
    Primary(Pe),
    /// The named PE's secondary (FU1) result -- staging taps 4-7.
    Secondary(Pe),
}

impl Source {
    /// The 5-bit FSEL/BSEL selector encoding.
    pub fn selector(self) -> u32 {
        match self {
            Source::Buf(b) => b.selector(),
            Source::Primary(p) => 0x10 | p.index() as u32,
            Source::Secondary(p) => 0x14 | p.index() as u32,
        }
    }
}

/// A fully specified functional-unit operation: the opcode plus the
/// descriptor's post-processing fields and the two constant registers.
#[derive(Debug, Clone, Copy)]
pub struct Operation {
    pub opcode: Opcode,
    pub acc: AccMode,
    /// Post-operation shift, 0-63 (K).  For `Qf x Qf -> Qf` set `k = f`.
    pub k: u8,
    /// Round the shifted result instead of truncating (R).
    pub round: bool,
    /// Saturation width in bits, 0 = disabled (SAT).
    pub sat: u8,
    /// Constant register `a` (bias, mask, clamp ceiling, ramp slope...).
    pub a: i32,
    /// Constant register `b` (immediate co-operand, bias, clamp floor...).
    pub b: i32,
}

impl Operation {
    pub fn new(opcode: Opcode) -> Self {
        Operation { opcode, acc: AccMode::None, k: 0, round: false, sat: 0, a: 0, b: 0 }
    }

    pub fn k(mut self, k: u8) -> Self {
        self.k = k;
        self
    }

    pub fn round(mut self) -> Self {
        self.round = true;
        self
    }

    pub fn sat(mut self, bits: u8) -> Self {
        self.sat = bits;
        self
    }

    pub fn a(mut self, a: i32) -> Self {
        self.a = a;
        self
    }

    pub fn b(mut self, b: i32) -> Self {
        self.b = b;
        self
    }

    pub fn acc(mut self, acc: AccMode) -> Self {
        self.acc = acc;
        self
    }
}

/// One functional unit's configuration: what it computes and where its
/// operands come from.  A unit with no operation set is inert.
#[derive(Debug, Clone, Copy, Default)]
pub struct FunctionalUnit {
    pub(crate) op: Option<Operation>,
    pub(crate) back: Option<Source>,
    pub(crate) front: Option<Source>,
}

impl FunctionalUnit {
    /// The primary operand.  Every unit needs one: unary operations act on
    /// it, and it paces the unit -- even `MovI`/`Ramp` advance only while
    /// their back source streams.
    pub fn set_back(&mut self, src: Source) -> &mut Self {
        self.back = Some(src);
        self
    }

    /// The co-operand stream.  Only meaningful for operations whose
    /// mnemonic reads the front stream; immediate-mode operations take
    /// constant `b` instead.
    pub fn set_front(&mut self, src: Source) -> &mut Self {
        self.front = Some(src);
        self
    }

    pub fn set_op(&mut self, op: Operation) -> &mut Self {
        self.op = Some(op);
        self
    }

    pub fn is_configured(&self) -> bool {
        self.op.is_some()
    }
}

/// AGU element-order transform (FMT0/FMT1, section 7.5/7.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Transform {
    /// Linear walk (counter A only).
    #[default]
    Linear,
    /// Element order reversed across the range (FMT0.REV).
    Reverse,
    /// The first element broadcast across the whole range (FMT0.RP1|RP0).
    Replicate,
    /// Bit-reversed order over `2^width` elements (FMT1.BRV + BRVW).
    BitReversed { width: u8 },
}

/// Segment replay: counter B walks `0..seg_len` repeatedly while the outer
/// window advances by `stride` per reload -- the array's loop construct
/// (a coefficient vector against a longer data stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Replay {
    pub seg_len: u16,
    pub stride: u16,
}

/// Per-port address-generation parameters.  Everything defaults to a linear
/// walk from offset 0 with step 1; the skew is derived by the assembler
/// unless overridden.
#[derive(Debug, Clone, Copy)]
pub struct AguParams {
    /// Start offset in words.  Wraps at 16 bits, so `0x1_0000 - x` is a
    /// negative offset (the drain-cancel idiom).
    pub offset: u16,
    /// Words advanced per element.
    pub step: u16,
    pub replay: Option<Replay>,
    pub transform: Transform,
    /// Manual cycle-skew override.  `None`: the assembler derives it from
    /// the dataflow graph and the RTL latency model ([`crate::timing`]).
    pub skew: Option<u8>,
    /// Write port only: pipeline drain in elements.  Emits FMT0.DRAIN and
    /// the FMT1 END token; valid element `j` then lands at sequence
    /// position `j + drain`.
    pub drain: Option<u16>,
}

impl Default for AguParams {
    fn default() -> Self {
        AguParams { offset: 0, step: 1, replay: None, transform: Transform::Linear, skew: None, drain: None }
    }
}

/// One processing element: two functional units, three address generators.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessingElement {
    pub(crate) fu0: FunctionalUnit,
    pub(crate) fu1: FunctionalUnit,
    pub(crate) count: Option<u32>,
    pub read_top: AguParams,
    pub read_base: AguParams,
    pub write: AguParams,
    /// Set to suppress this element's write port (its results are then
    /// visible only on the staging bus).  Default: writes whenever a
    /// functional unit is configured.
    pub write_disabled: bool,
}

impl ProcessingElement {
    /// The primary functional unit.
    pub fn fu0(&mut self) -> &mut FunctionalUnit {
        &mut self.fu0
    }

    /// The secondary functional unit.  Configuring it sets its FU1EN bit
    /// and routes it to the element's write port (section 4.5).
    pub fn fu1(&mut self) -> &mut FunctionalUnit {
        &mut self.fu1
    }

    /// Stream length for this element (all three of its AGUs).  Falls back
    /// to [`VmeConfig::set_stream_len`].
    pub fn set_count(&mut self, count: u32) -> &mut Self {
        self.count = Some(count);
        self
    }

    pub fn is_configured(&self) -> bool {
        self.fu0.is_configured() || self.fu1.is_configured()
    }
}

/// Initial contents of one ring buffer.
pub enum BufferInit {
    Zero,
    Data(Vec<i32>),
    Callback(Box<dyn Fn(&mut [i32; BUFFER_WORDS])>),
}

/// Handle for configuring one buffer's initial contents.
pub struct BufferSlot {
    pub(crate) init: BufferInit,
}

impl BufferSlot {
    /// Fill the buffer through a callback at assembly time.
    pub fn set_callback<F: Fn(&mut [i32; BUFFER_WORDS]) + 'static>(&mut self, f: F) {
        self.init = BufferInit::Callback(Box::new(f));
    }

    /// Fill the buffer from a slice (shorter slices are zero-padded).
    pub fn set_data(&mut self, data: &[i32]) {
        self.init = BufferInit::Data(data.to_vec());
    }
}

/// The whole array: four processing elements, eight buffers, one stream
/// length.  Build it up with the accessors, then commit it with
/// [`crate::generate_config`].
pub struct VmeConfig {
    pub(crate) pes: [ProcessingElement; 4],
    pub(crate) buffers: [BufferSlot; 8],
    pub(crate) default_count: Option<u32>,
}

impl VmeConfig {
    pub fn new() -> Self {
        VmeConfig {
            pes: [ProcessingElement::default(); 4],
            buffers: [(); 8].map(|_| BufferSlot { init: BufferInit::Zero }),
            default_count: None,
        }
    }

    pub fn pe_mut(&mut self, pe: Pe) -> &mut ProcessingElement {
        &mut self.pes[pe.index()]
    }

    pub fn pe(&self, pe: Pe) -> &ProcessingElement {
        &self.pes[pe.index()]
    }

    /// Buffer index order: TOP_0..3 then BASE_0..3 is *not* assumed
    /// anywhere -- address by name.
    pub fn buffer_mut(&mut self, b: Buffer) -> &mut BufferSlot {
        let idx = if b.is_top() { 4 + b.lane() } else { b.lane() };
        &mut self.buffers[idx]
    }

    pub(crate) fn buffer(&self, b: Buffer) -> &BufferSlot {
        let idx = if b.is_top() { 4 + b.lane() } else { b.lane() };
        &self.buffers[idx]
    }

    /// Default stream length for every configured element.
    pub fn set_stream_len(&mut self, count: u32) {
        self.default_count = Some(count);
    }

    pub(crate) fn count_for(&self, pe: Pe) -> Option<u32> {
        self.pes[pe.index()].count.or(self.default_count)
    }
}

impl Default for VmeConfig {
    fn default() -> Self {
        Self::new()
    }
}
