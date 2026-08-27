use crate::*;

/// The canonical silicon-verified shape: front = TOP_0 (data), back = the
/// element's own BASE_0 at a disjoint offset (coefficients), results
/// streaming into BASE_0[0..].
fn minimal_vmul() -> VmeConfig {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(16);
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_front(Source::Buf(Buffer::Top0));
    pe0.fu0().set_back(Source::Buf(Buffer::Base0));
    pe0.fu0().set_op(Operation::new(Opcode::VMul).k(20).round());
    pe0.read_base.offset = 256;
    pe0.allow_write_clobber = true; // reads 256.., writes 0..
    vme
}

/// The manual's worked example (section 6.8): FSEL BASE_1, BSEL TOP_0,
/// VMUL, R, K=20 assembles to 0x5022_0054.  Placed on PE1 so BASE_1 is the
/// element's own base buffer (base-bank affinity).
#[test]
fn descriptor_matches_manual_worked_example() {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(16);
    let pe1 = vme.pe_mut(Pe::Pe1);
    pe1.fu0().set_front(Source::Buf(Buffer::Base1));
    pe1.fu0().set_back(Source::Buf(Buffer::Top0));
    pe1.fu0().set_op(Operation::new(Opcode::VMul).k(20).round());
    pe1.allow_write_clobber = true; // front names its own BASE_1
    let plan = validate(&vme).unwrap();
    let ctx = context_words(&vme, &plan);
    assert_eq!(ctx[1], 0x5022_0054);
}

/// Opcode words against Appendix C.
#[test]
fn opcode_words_match_appendix_c() {
    use AccMode::None as N;
    assert_eq!(Opcode::Add.op_word(N), 0x0001_0000);
    assert_eq!(Opcode::AddI.op_word(N), 0x0002_4000);
    assert_eq!(Opcode::Clamp.op_word(N), 0x0007_C000);
    assert_eq!(Opcode::MovI.op_word(N), 0x0000_C000);
    assert_eq!(Opcode::Rsb.op_word(N), 0x0001_8000);
    assert_eq!(Opcode::MulI.op_word(N), 0x0020_4000);
    assert_eq!(Opcode::MacI.op_word(N), 0x0024_0000);
    assert_eq!(Opcode::Sad.op_word(N), 0x0029_0000);
    // ACC adds 0x1000/0x2000/0x3000 (Appendix C note)
    assert_eq!(Opcode::MacI.op_word(AccMode::Hold), 0x0024_1000);
    assert_eq!(Opcode::MacI.op_word(AccMode::Zero), 0x0024_3000);
}

/// Source selectors against Table 5.1 (as FSEL constants, << 27).
#[test]
fn source_selectors_match_table() {
    assert_eq!(Source::Buf(Buffer::Top0).selector() << 27, 0x0000_0000);
    assert_eq!(Source::Buf(Buffer::Top1).selector() << 27, 0x1000_0000);
    assert_eq!(Source::Buf(Buffer::Base0).selector() << 27, 0x4000_0000);
    assert_eq!(Source::Buf(Buffer::Base1).selector() << 27, 0x5000_0000);
    assert_eq!(Source::Primary(Pe::Pe0).selector() << 27, 0x8000_0000);
    assert_eq!(Source::Secondary(Pe::Pe3).selector() << 27, 0xB800_0000u32);
}

/// FU0 reading buffers: write skew = BUFFER_LATENCY = 6
/// (silicon-calibrated, 2026-08-27 skew probes); no offset shifts.
#[test]
fn skew_simple_pipeline() {
    let plan = validate(&minimal_vmul()).unwrap();
    assert_eq!(plan.top_shift[0], Some(0));
    assert_eq!(plan.base_shift[0], Some(0));
    assert_eq!(plan.write_skew[0], Some(6));
}

/// FU1 conditioning FU0 via its own staging tap: one STAGING_HOP on top.
#[test]
fn skew_fu1_chain() {
    let mut vme = minimal_vmul();
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu1().set_back(Source::Primary(Pe::Pe0));
    pe0.fu1().set_op(Operation::new(Opcode::Clamp).a(100).b(-100));
    let plan = validate(&vme).unwrap();
    assert_eq!(plan.write_skew[0], Some(9));
}

/// Cross-PE staging: PE1 adds PE0's product to a buffer stream.  Reads
/// cannot be skewed, so PE1's buffer read is shifted 3 positions and its
/// write skewed to the tap's availability + one hop (9).
#[test]
fn skew_cross_pe_ladder() {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(16);
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_back(Source::Buf(Buffer::Top0));
    pe0.fu0().set_front(Source::Buf(Buffer::Top1));
    pe0.fu0().set_op(Operation::new(Opcode::VMul));
    pe0.write_disabled = true;
    let pe1 = vme.pe_mut(Pe::Pe1);
    pe1.fu0().set_back(Source::Primary(Pe::Pe0));
    pe1.fu0().set_front(Source::Buf(Buffer::Top2));
    pe1.fu0().set_op(Operation::new(Opcode::Add));
    let plan = validate(&vme).unwrap();
    assert_eq!(plan.top_shift[0], Some(0));
    assert_eq!(plan.top_shift[1], Some(3));
    assert_eq!(plan.write_skew[1], Some(9));
    assert_eq!(plan.write_skew[0], None); // suppressed
}

/// A staging source on the front operand cannot be aligned -- rejected.
#[test]
fn rejects_front_staging() {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(16);
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_back(Source::Buf(Buffer::Top0));
    pe0.fu0().set_front(Source::Buf(Buffer::Top1));
    pe0.fu0().set_op(Operation::new(Opcode::VMul));
    pe0.write_disabled = true;
    let pe1 = vme.pe_mut(Pe::Pe1);
    pe1.fu0().set_back(Source::Buf(Buffer::Top2));
    pe1.fu0().set_front(Source::Primary(Pe::Pe0));
    pe1.fu0().set_op(Operation::new(Opcode::Add));
    let errs = validate(&vme).unwrap_err();
    assert!(errs.iter().any(|e| matches!(e, VmeError::FrontStaging { .. })));
}

/// Read-port skew overrides are rejected: the hardware ignores them.
#[test]
fn rejects_read_skew_override() {
    let mut vme = minimal_vmul();
    vme.pe_mut(Pe::Pe0).read_top.skew = Some(3);
    let errs = validate(&vme).unwrap_err();
    assert!(errs.iter().any(|e| matches!(e, VmeError::ReadSkewUnsupported { .. })));
}

/// FU1 has no buffer path (measured): a buffer back on FU1 is rejected.
#[test]
fn rejects_fu1_buffer_back() {
    let mut vme = minimal_vmul();
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu1().set_back(Source::Buf(Buffer::Top1));
    pe0.fu1().set_op(Operation::new(Opcode::Mov));
    let errs = validate(&vme).unwrap_err();
    assert!(errs.iter().any(|e| matches!(e, VmeError::Fu1BufferBack { .. })));
}

#[test]
fn rejects_missing_back() {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(4);
    vme.pe_mut(Pe::Pe0).fu0().set_op(Operation::new(Opcode::Mov));
    let errs = validate(&vme).unwrap_err();
    assert!(errs.iter().any(|e| matches!(e, VmeError::MissingBack { .. })));
}

#[test]
fn rejects_missing_front_for_stream_op() {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(4);
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_back(Source::Buf(Buffer::Top0));
    pe0.fu0().set_op(Operation::new(Opcode::Add)); // needs a front stream
    let errs = validate(&vme).unwrap_err();
    assert!(errs.iter().any(|e| matches!(e, VmeError::MissingFront { .. })));
}

/// PE0 writes BASE_0; reading BASE_0 in the same pass needs the explicit
/// allow_write_clobber opt-in (legitimate only with disjoint offsets).
#[test]
fn rejects_write_clobber() {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(4);
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_back(Source::Buf(Buffer::Base0));
    pe0.fu0().set_op(Operation::new(Opcode::Mov));
    let errs = validate(&vme).unwrap_err();
    assert!(errs.iter().any(|e| matches!(e, VmeError::WriteClobber { .. })));
}

/// Base-bank reads have own-lane affinity: PE0 naming BASE_1 is rejected.
#[test]
fn rejects_foreign_base_buffer() {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(4);
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_back(Source::Buf(Buffer::Base1));
    pe0.fu0().set_op(Operation::new(Opcode::Mov));
    let errs = validate(&vme).unwrap_err();
    assert!(errs.iter().any(|e| matches!(e, VmeError::BaseAffinity { .. })));
}

#[test]
fn rejects_staging_cycle() {
    let mut vme = VmeConfig::new();
    vme.set_stream_len(4);
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_back(Source::Primary(Pe::Pe1));
    pe0.fu0().set_op(Operation::new(Opcode::Mov));
    let pe1 = vme.pe_mut(Pe::Pe1);
    pe1.fu0().set_back(Source::Primary(Pe::Pe0));
    pe1.fu0().set_op(Operation::new(Opcode::Mov));
    let errs = validate(&vme).unwrap_err();
    assert!(errs.iter().any(|e| matches!(e, VmeError::StagingCycle)));
}

#[test]
fn rejects_missing_count() {
    let mut vme = VmeConfig::new();
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.fu0().set_back(Source::Buf(Buffer::Top0));
    pe0.fu0().set_op(Operation::new(Opcode::Mov));
    let errs = validate(&vme).unwrap_err();
    assert!(errs.iter().any(|e| matches!(e, VmeError::MissingCount { .. })));
}

/// Image geometry: buffers and context land at their mapped offsets, in
/// the 24-bit sign-extended sample format.
#[test]
fn image_layout() {
    let mut vme = minimal_vmul();
    vme.buffer_mut(Buffer::Top0).set_data(&[7, -1]);
    vme.buffer_mut(Buffer::Base0).set_callback(|buf| buf[256] = 0x40_0000);
    let img = generate_config(&vme).unwrap();
    assert_eq!(img.bytes().len(), assemble::IMAGE_SIZE);
    assert_eq!(img.word(0x20000), 7);
    assert_eq!(img.word(0x20004), 0xFFFF_FFFF); // -1 sign-extended
    assert_eq!(img.word(4 * 256), 0x40_0000); // BASE_0[256]
    // ctx[0]: FSEL TOP_0 | BSEL BASE_0 | VMUL | R | K=20
    assert_eq!(img.word(0xF8000), 0x0222_0054);
    assert_eq!(img.word(0xF8000 + 4 * 105), 0x18); // CTX_END
    // AGU groups: PE0 base read at offset 256 (no read skew -- the field
    // is ignored by hardware and never emitted), write skewed 6
    assert_eq!(img.word(0xF8000 + 4 * 39), 0x8400_0100);
    assert_eq!(img.word(0xF8000 + 4 * 45), 0x8406_0000);
    assert_eq!(img.word(0xF8000 + 4 * 46), 0x0001_000F);
}

/// Replay and drain emission: INNER0/FMT0/FMT1 fields.
#[test]
fn replay_and_drain_fields() {
    let mut vme = minimal_vmul();
    let pe0 = vme.pe_mut(Pe::Pe0);
    pe0.read_base.replay = Some(Replay { seg_len: 4, stride: 0 });
    pe0.write.drain = Some(16);
    let plan = validate(&vme).unwrap();
    let ctx = context_words(&vme, &plan);
    assert_eq!(ctx[39] >> 24, 0x82); // RBASE: E + segmented mode
    assert_eq!(ctx[39] & 0xFFFF, 256); // start offset kept
    assert_eq!(ctx[41], 0x0001_0003); // INNER0: CFG 1, SEGMENT 3
    assert_eq!(ctx[43] & 0x0002_0000, 0x0002_0000); // FMT0.RNG
    assert_eq!(ctx[49] & 0xFFFF, 16); // WR FMT0.DRAIN
    assert_eq!(ctx[50] & 0x0020_0000, 0x0020_0000); // WR FMT1.END
}
