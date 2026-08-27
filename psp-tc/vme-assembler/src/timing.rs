//! The silicon-calibrated latency model and the alignment solver.
//!
//! What the 2026-08-27 hardware probes established (and the RTL in
//! `vme-emu/rtl/` now reproduces):
//!
//! * **Read AGUs ignore their skew field.**  Every read port starts at
//!   trigger; only the *write* AGU's MODE[23:16] skew works.  (Consistent
//!   with every proven mcidclan context, which only ever skews writes.)
//! * The address-issue-to-write-capture path for a buffer-fed unit is
//!   **6 cycles** (read 3 + FU 3), and each staging hop adds the FU half
//!   (**3 cycles**).
//!
//! * All active read ports advance on a **shared enable that halts when
//!   the shortest read stream ends** (probed: a lengthened port still froze
//!   at the array-wide minimum count).  So every read port is emitted with
//!   the same extended length.
//! * A staging consumer's buffer leg pairs tap element `m` with buffer
//!   *position* `m + 3·hops`, and the port's start-offset field proved
//!   unreliable for shifting that pairing.  Alignment is therefore done by
//!   **rotating the staged data** forward by the shift (the assembler does
//!   this in the image) rather than via offsets.
//!
//! Consequences the solver enforces:
//! * a staging source must be the **back** operand (it paces the unit; a
//!   staging *front* would lag its buffer-fed partner by 3 elements with
//!   no mechanism to advance it);
//! * two staging operands into one unit must have equal-depth producers;
//! * a shifted reader's buffer must have start offset 0 (its data is
//!   rotated instead) and no second reader at a different shift;
//! * write skew = the driving unit's `avail`.

use crate::config::{Fu, Pe, Source, VmeConfig};
use crate::VmeError;

/// Cycles from a read AGU's address issue to a buffer-fed functional
/// unit's result being capturable by its write port (read path 3 + FU 3).
pub const BUFFER_LATENCY: u32 = 6;

/// Extra result-availability cycles per staging hop (the FU half of the
/// path: a consumer of a tap produces 3 cycles after the producer).
pub const STAGING_HOP: u32 = 3;

/// Largest value the write MODE[23:16] skew field can hold.
pub const MAX_SKEW: u32 = 255;

/// The solved schedule.  Read ports carry no skew (the hardware ignores
/// it); `Some(shift)` marks a port in use whose staged data the assembler
/// rotates forward by `shift` words.  `read_len_extra` is the array-wide
/// count extension every read port is emitted with (the shared read enable
/// halts at the minimum active count, so all reads must cover the longest
/// shifted stream).
#[derive(Debug, Clone, Default)]
pub struct TimingPlan {
    pub top_shift: [Option<u16>; 4],
    pub base_shift: [Option<u16>; 4],
    pub write_skew: [Option<u32>; 4],
    pub read_len_extra: u16,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Node {
    pe: Pe,
    fu: Fu,
}

fn node_of(src: Source) -> Option<Node> {
    match src {
        Source::Primary(p) => Some(Node { pe: p, fu: Fu::Fu0 }),
        Source::Secondary(p) => Some(Node { pe: p, fu: Fu::Fu1 }),
        Source::Buf(_) => None,
    }
}

fn fu_name(n: Node) -> String {
    format!("PE{}.{}", n.pe.index(), if n.fu == Fu::Fu0 { "FU0" } else { "FU1" })
}

/// Derive write skews and read-port offset shifts for the configured graph.
pub fn solve(cfg: &VmeConfig) -> Result<TimingPlan, Vec<VmeError>> {
    let mut errors = Vec::new();

    // collect configured functional units and their operands
    let mut nodes: Vec<(Node, Option<Source>, Option<Source>)> = Vec::new();
    for pe in Pe::ALL {
        for fu in [Fu::Fu0, Fu::Fu1] {
            let unit = match fu {
                Fu::Fu0 => &cfg.pe(pe).fu0,
                Fu::Fu1 => &cfg.pe(pe).fu1,
            };
            let Some(op) = unit.op else { continue };
            let front = if op.opcode.uses_front_stream() { unit.front } else { None };
            nodes.push((Node { pe, fu }, unit.back, front));
        }
        // read-port skews do not exist on hardware
        for (port, agu) in [("read_top", &cfg.pe(pe).read_top), ("read_base", &cfg.pe(pe).read_base)] {
            if agu.skew.is_some() {
                errors.push(VmeError::ReadSkewUnsupported { pe: pe.index(), port });
            }
        }
    }

    // topological order over back-edges (staging chains)
    let dep_idx = |src: Option<Source>| -> Option<usize> {
        src.and_then(node_of).and_then(|n| nodes.iter().position(|(m, _, _)| *m == n))
    };
    let mut avail: Vec<Option<u32>> = vec![None; nodes.len()];
    for _ in 0..nodes.len() {
        for i in 0..nodes.len() {
            if avail[i].is_some() {
                continue;
            }
            let (_, back, _) = nodes[i];
            avail[i] = match back.and_then(node_of) {
                None => Some(BUFFER_LATENCY),
                Some(_) => match dep_idx(back) {
                    Some(d) => avail[d].map(|a| a + STAGING_HOP),
                    None => Some(BUFFER_LATENCY), // producer missing: caught by validate
                },
            };
        }
    }
    if avail.iter().any(|a| a.is_none()) {
        errors.push(VmeError::StagingCycle);
        return Err(errors);
    }

    let mut plan = TimingPlan::default();
    let pin_shift = |errors: &mut Vec<VmeError>,
                         plan: &mut TimingPlan,
                         node: Node,
                         b: crate::config::Buffer,
                         shift: u16| {
        let p = node.pe.index();
        let slot = if b.is_top() { &mut plan.top_shift[p] } else { &mut plan.base_shift[p] };
        match slot {
            None => *slot = Some(shift),
            Some(s) if *s != shift => errors.push(VmeError::SkewConflict {
                fu: fu_name(node),
                a: format!("port already shifted by {s}"),
                a_cycle: *s as u32,
                b: format!("read of {b:?}"),
                b_cycle: shift as u32,
            }),
            _ => {}
        }
    };

    for (i, (node, back, front)) in nodes.iter().enumerate() {
        let a = avail[i].unwrap();
        let shift = (a - BUFFER_LATENCY) as u16;

        if let Some(Source::Buf(b)) = back {
            pin_shift(&mut errors, &mut plan, *node, *b, shift);
        }
        match front {
            Some(Source::Buf(b)) => pin_shift(&mut errors, &mut plan, *node, *b, shift),
            Some(src) => {
                // a staging front cannot be advanced to match: it must pair
                // with a back tap of equal producer depth
                let fd = dep_idx(Some(*src)).unwrap();
                let back_is_tap = matches!(back, Some(Source::Primary(_)) | Some(Source::Secondary(_)));
                if !back_is_tap || avail[fd] != avail[dep_idx(*back).unwrap()] {
                    errors.push(VmeError::FrontStaging {
                        fu: fu_name(*node),
                        producer: fu_name(nodes[fd].0),
                    });
                }
            }
            None => {}
        }
    }

    // write ports: skew = driving unit's availability
    for pe in Pe::ALL {
        let p = pe.index();
        let elem = cfg.pe(pe);
        if !elem.is_configured() || elem.write_disabled {
            continue;
        }
        let driver = if elem.fu1.is_configured() { Fu::Fu1 } else { Fu::Fu0 };
        let d = nodes
            .iter()
            .position(|(n, _, _)| *n == Node { pe, fu: driver })
            .expect("configured driver is a node");
        let skew = match elem.write.skew {
            Some(s) => s as u32,
            None => avail[d].unwrap(),
        };
        if skew > MAX_SKEW {
            errors.push(VmeError::SkewRange { fu: fu_name(nodes[d].0), skew });
        }
        plan.write_skew[p] = Some(skew);
    }

    // a buffer's staged data can be rotated by only one shift: two readers
    // at different depths cannot share it
    for b in crate::config::Buffer::ALL {
        let mut seen: Option<(u16, Node)> = None;
        for (i, (node, back, front)) in nodes.iter().enumerate() {
            let reads = [back, front]
                .into_iter()
                .flatten()
                .any(|s| *s == Source::Buf(b));
            if !reads {
                continue;
            }
            let shift = (avail[i].unwrap() - BUFFER_LATENCY) as u16;
            match seen {
                None => seen = Some((shift, *node)),
                Some((s0, n0)) if s0 != shift => errors.push(VmeError::SkewConflict {
                    fu: fu_name(*node),
                    a: format!("{} reads {b:?} at shift {s0}", fu_name(n0)),
                    a_cycle: s0 as u32,
                    b: format!("read of {b:?}"),
                    b_cycle: shift as u32,
                }),
                _ => {}
            }
        }
    }

    plan.read_len_extra = plan
        .top_shift
        .iter()
        .chain(plan.base_shift.iter())
        .filter_map(|s| *s)
        .max()
        .unwrap_or(0);

    // rotation replaces the offset mechanism for shifted readers, so their
    // buffers must start at 0
    for pe in Pe::ALL {
        let p = pe.index();
        if plan.top_shift[p].unwrap_or(0) != 0 && cfg.pe(pe).read_top.offset != 0 {
            errors.push(VmeError::ShiftedOffset { pe: p, port: "read_top" });
        }
        if plan.base_shift[p].unwrap_or(0) != 0 && cfg.pe(pe).read_base.offset != 0 {
            errors.push(VmeError::ShiftedOffset { pe: p, port: "read_base" });
        }
    }

    if errors.is_empty() {
        Ok(plan)
    } else {
        Err(errors)
    }
}
