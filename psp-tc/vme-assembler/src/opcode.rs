//! The VME operation set: every defined operation of the reference manual's
//! Appendix C (Table C.1), one enum variant per distinct operation.
//!
//! Each variant fixes the descriptor's CLASS, FN *and* OPM fields -- the
//! operand mode is part of the mnemonic, exactly as in the manual's tables
//! (`Add` reads both streams, `AddI` reads the back stream and constant `b`).
//! Aliased encodings (the manual lists several) collapse to one canonical
//! variant here.
//!
//! Notation in the summaries: `back[n]`/`front[n]` are the two operand
//! streams, `a`/`b` the unit's constant registers, `k` the descriptor's
//! post-operation shift, `out[n-1]` the unit's own previous result, `acc`
//! the 64-bit accumulator.

/// Accumulator control -- the ACC sub-field of the OP field (Table 6.3a).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccMode {
    /// Accumulator untouched; the result is the operation output.
    #[default]
    None,
    /// Each result accumulates into the unit's accumulator across the
    /// stream, and the accumulated value is the output.
    Hold,
    /// The accumulator is preloaded from constant `a` at trigger.
    Load,
    /// The accumulator is zeroed at trigger.
    Zero,
}

impl AccMode {
    pub fn bits(self) -> u32 {
        match self {
            AccMode::None => 0b00,
            AccMode::Hold => 0b01,
            AccMode::Load => 0b10,
            AccMode::Zero => 0b11,
        }
    }
}

macro_rules! opcodes {
    ($( $(#[$doc:meta])* $name:ident = ($class:literal, $fnf:literal, $opm:literal, front: $front:literal), )*) => {
        /// One VME functional-unit operation.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Opcode {
            $( $(#[$doc])* $name, )*
        }

        impl Opcode {
            /// (CLASS, FN, OPM) descriptor sub-fields.
            pub fn fields(self) -> (u32, u32, u32) {
                match self {
                    $( Opcode::$name => ($class, $fnf, $opm), )*
                }
            }

            /// Whether the operation consumes the front *stream* (as opposed
            /// to constant `b` or nothing).  Streams that are consumed must
            /// be routed and cycle-aligned; the assembler uses this to know
            /// which ports a functional unit needs.
            pub fn uses_front_stream(self) -> bool {
                match self {
                    $( Opcode::$name => $front, )*
                }
            }
        }
    };
}

opcodes! {
    // ---- class 0: ALU, logic, shift and select -----------------------
    /// `back[n]` -- pass the back stream through unchanged.
    Mov = (0, 0b0000, 0b01, front: false),
    /// `b` -- emit constant `b` every element.
    MovI = (0, 0b0000, 0b11, front: false),
    /// `(back[n] + front[n]) >> k` -- elementwise add.
    Add = (0, 0b0001, 0b00, front: true),
    /// `back[n] + front[n] + a` -- add with constant bias.
    AddA = (0, 0b0011, 0b00, front: true),
    /// `(back[n] + b) >> k` -- add an immediate.
    AddI = (0, 0b0010, 0b01, front: false),
    /// `back[n] + (front[n] >> b)` -- add a downscaled stream.
    AddSf = (0, 0b0100, 0b00, front: true),
    /// `back[n] - (front[n] >> b)` -- subtract a downscaled stream.
    SubSf = (0, 0b0101, 0b00, front: true),
    /// `(back[n] - b) + a` -- subtract immediate, add bias.
    SubA = (0, 0b0011, 0b01, front: false),
    /// `(back[n] - b) << k` -- subtract immediate, upscale.
    SubIl = (0, 0b0111, 0b01, front: false),
    /// `(front[n] - back[n]) >> k` -- reverse subtract.
    Rsb = (0, 0b0001, 0b10, front: true),
    /// `(front[n] - back[n]) + b` -- reverse subtract with bias.
    RsbI = (0, 0b0100, 0b10, front: true),
    /// `b - back[n]` -- reverse subtract from an immediate, no rescale.
    RsbU = (0, 0b0011, 0b11, front: false),
    /// `front[n] + back[n]` -- add, no rescale ("unscaled").
    AddU = (0, 0b0011, 0b10, front: true),
    /// `back[n] >> k` -- arithmetic shift right by the descriptor shift.
    Asr = (0, 0b0001, 0b01, front: false),
    /// `(back[n] >> b) + a` -- shift right by `b`, add bias.
    AsrA = (0, 0b0100, 0b01, front: false),
    /// `(back[n] >> b) - a` -- shift right by `b`, subtract bias.
    AsrS = (0, 0b0101, 0b01, front: false),
    /// `back[n] >> b` -- arithmetic shift right by an immediate.
    AsrI = (0, 0b1010, 0b11, front: false),
    /// `back[n] >> front[n]` -- shift right by the front stream.
    AsrF = (0, 0b1011, 0b10, front: true),
    /// `b >> back[n]` -- shift an immediate right by the back stream.
    AsrKb = (0, 0b1001, 0b01, front: false),
    /// `-(back[n] >> k) + b` -- negated shift, plus immediate.
    NasrI = (0, 0b0010, 0b11, front: false),
    /// `-(back[n] >> b) + a` -- negated shift by `b`, plus bias.
    NasrA = (0, 0b0100, 0b11, front: false),
    /// `(front[n] & a) ? -back[n] : back[n]` -- negate where the mask hits.
    NegF = (0, 0b0110, 0b00, front: true),
    /// `(back[n] & a) ? back[n] : -back[n]` -- negate where the mask misses.
    NegB = (0, 0b0110, 0b01, front: false),
    /// `(front[n] & a) ? b : back[n]` -- predicated select of an immediate.
    Sel = (0, 0b0110, 0b10, front: true),
    /// `((front[n] & a) ? back[n] : 0) + b` -- predicated zeroing, plus bias.
    SelZ = (0, 0b0110, 0b11, front: false),
    /// `(back[n] & a) ? b : 0` -- bit-test against mask `a`.
    TstI = (0, 0b1000, 0b11, front: false),
    /// `back[n] != 0` -- non-zero test, emits 0 or 1.
    Tst = (0, 0b1111, 0b01, front: false),
    /// `(back[n] << k) + b` -- upscale, plus immediate.
    LslB = (0, 0b0111, 0b00, front: false),
    /// `b << back[n]` -- shift an immediate left by the back stream.
    LslKb = (0, 0b1001, 0b00, front: false),
    /// `back[n] << front[n]` -- shift left by the front stream.
    LslF = (0, 0b1010, 0b00, front: true),
    /// `back[n] << b` -- shift left by an immediate.
    LslI = (0, 0b1010, 0b01, front: false),
    /// `back[n] ROR64 front[n]` -- rotate the 64-bit accumulator right by
    /// the front stream (pair with `AccMode::Load` to seed it from `a`).
    Ror64 = (0, 0b1010, 0b10, front: true),
    /// `min(back[n], front[n])` -- elementwise minimum.
    Min = (0, 0b0111, 0b10, front: true),
    /// `max(b, min(a, back[n]))` -- clamp: `a` is the ceiling, `b` the floor.
    Clamp = (0, 0b0111, 0b11, front: false),
    /// `(back[n] * front[n]) * 1[-2,2](front[n])` -- windowed multiply:
    /// the product where the front operand lies in [-2, 2], else 0.
    MulW = (0, 0b1000, 0b00, front: true),
    /// `-(back[n] * b) * 1[-2,2](b)` -- negated windowed multiply by `b`.
    MulWn = (0, 0b1000, 0b01, front: false),
    /// `back[n] & front[n]` -- bitwise AND.
    And = (0, 0b1100, 0b00, front: true),
    /// `back[n] & b` -- bitwise AND with a mask.
    AndI = (0, 0b1100, 0b01, front: false),
    /// `~(front[n] & back[n])` -- NAND.
    Nand = (0, 0b1100, 0b10, front: true),
    /// `~(back[n] & b)` -- NAND with a mask.
    NandI = (0, 0b1100, 0b11, front: false),
    /// `back[n] | front[n]` -- bitwise OR.
    Orr = (0, 0b1101, 0b00, front: true),
    /// `back[n] | b` -- bitwise OR with a mask.
    OrrI = (0, 0b1101, 0b01, front: false),
    /// `~(front[n] | back[n])` -- NOR.
    Nor = (0, 0b1101, 0b10, front: true),
    /// `~back[n] & ~b` -- NOR with a mask.
    NorI = (0, 0b1101, 0b11, front: false),
    /// `back[n] ^ front[n]` -- bitwise exclusive OR.
    Eor = (0, 0b1110, 0b00, front: true),
    /// `back[n] ^ b` -- exclusive OR with a mask.
    EorI = (0, 0b1110, 0b01, front: false),
    /// `~|front[n] - back[n]|` -- negated absolute difference (bitwise NOT).
    NabsD = (0, 0b1110, 0b10, front: true),
    /// `~back[n] ^ b` -- exclusive NOR with a mask.
    XnorI = (0, 0b1110, 0b11, front: false),
    /// `~back[n]` -- bitwise NOT.
    Not = (0, 0b1111, 0b00, front: false),
    /// `back[0] ^ back[1] ^ ... ^ back[n]` -- running parity/XOR reduction.
    Parity = (0, 0b1111, 0b10, front: false),
    /// `(((0xFF00 & front[n]) + (0xFF00 & back[n])) >> 8) << k` -- add the
    /// middle packed byte channels.
    AddP1 = (0, 0b0101, 0b10, front: true),
    /// `((0xFF & back[n]) + (0xFF & b)) << k` -- add the low packed byte
    /// channel of the back stream and `b`; `k` in [0, 1].
    AddP0 = (0, 0b0101, 0b11, front: false),

    // ---- class 1: extrema and absolute -------------------------------
    /// `max(out[n-1], back[n])` -- running maximum (peak detector).
    RMax = (1, 0b0000, 0b00, front: false),
    /// `min(out[n-1], back[n] - front[n])` -- running minimum of a difference.
    RMinD = (1, 0b0001, 0b00, front: true),
    /// `min(back[n], max(back[0..n-1]))` -- clamp to the running peak
    /// (envelope follower); `out[0] = back[0]`.
    PClamp = (1, 0b0010, 0b00, front: false),
    /// `max(out[n-1], back[n] - front[n])` -- running maximum of a difference.
    RMaxD = (1, 0b0011, 0b00, front: true),
    /// `max(back[n], front[n])` -- elementwise maximum.
    Max = (1, 0b0100, 0b00, front: true),
    /// `|back[n] - b|` -- absolute distance from an immediate.
    AbsI = (1, 0b0111, 0b00, front: false),
    /// `|back[n] + front[n]|` -- absolute value of a sum.
    AbsS = (1, 0b1100, 0b00, front: true),
    /// `|back[n] - front[n]| + a` -- absolute difference, plus bias `a`.
    AbsDA = (1, 0b1110, 0b00, front: true),
    /// `|front[n] - back[n]| + b` -- absolute difference, plus bias `b`.
    AbsDI = (1, 0b1111, 0b00, front: true),

    // ---- class 2: multiply, MAC, IIR and SAD --------------------------
    /// `(back[n] * front[n]) >> k` -- elementwise multiply.
    Mul = (2, 0b0000, 0b00, front: true),
    /// `(back[n] * b) >> k` -- multiply by a constant.
    MulI = (2, 0b0000, 0b01, front: false),
    /// `-(back[n] * front[n]) >> k` -- negated multiply.
    MulN = (2, 0b0000, 0b10, front: true),
    /// `-(back[n] * b) >> k` -- negated multiply by a constant.
    MulNI = (2, 0b0000, 0b11, front: false),
    /// `(back[n] * back[n-1]) >> k` -- multiply by the previous element;
    /// `back[-1] = b`.
    MulD = (2, 0b0001, 0b00, front: false),
    /// `(back[n] * front[n]) >> k` -- vector multiply (the manual's
    /// worked-example op; identical arithmetic to `Mul`).
    VMul = (2, 0b0010, 0b00, front: true),
    /// `-((back[n] * front[n]) >> k)` -- negated vector multiply.
    VMulN = (2, 0b0011, 0b00, front: true),
    /// `(sum(back * front) + b) >> k` -- multiply-accumulate: each element
    /// emits the running inner product (a dot product's last element is the
    /// full sum).
    MacI = (2, 0b0100, 0b00, front: true),
    /// `n == 0 ? b >> k : (out[n-1] + back[n] + b) >> k` -- running sum
    /// with per-element bias.
    AccI = (2, 0b0101, 0b00, front: false),
    /// `((a * n) + b) >> k` -- ramp generator: index times `a` plus `b`.
    Ramp = (2, 0b0110, 0b00, front: false),
    /// `((back[n] * front[n]) + b) >> k` -- multiply, plus bias.
    VMulI = (2, 0b0111, 0b00, front: true),
    /// `n < 2 ? 0 : ((out[n-1] + front[n-2] + back[n-2]) >> k) + a` --
    /// two-tap IIR recurrence.
    Iir = (2, 0b1000, 0b00, front: true),
    /// `sum(|back[m] - front[m]|, m < n) + b` -- sum of absolute
    /// differences over the elements before this one (block matching).
    Sad = (2, 0b1001, 0b00, front: true),
}

impl Opcode {
    /// The ten-bit OP field with the given accumulator mode, positioned at
    /// descriptor bits [21:12].
    pub fn op_word(self, acc: AccMode) -> u32 {
        let (class, f, opm) = self.fields();
        (class << 20) | (f << 16) | (opm << 14) | (acc.bits() << 12)
    }
}
