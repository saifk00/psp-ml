//! The RTL latency model and the skew solver.
//!
//! The VME has no interlocks: synchronisation is per-port cycle skew, and a
//! mis-skewed context silently reads stale data.  This module is the single
//! place that knows the pipeline depths of the RTL in `vme-emu/rtl/` (see
//! its README, "Timing of this model") and derives every port's skew from
//! the dataflow graph.  If the RTL's pipeline changes, change the two
//! constants below and every assembled context follows.
//!
//! The model, in cycles relative to trigger, for a port with skew `s`:
//!
//! ```text
//! address issue for element m:        s + m
//! buffer read data visible:           s + m + READ_LATENCY
//! FU result visible (staging tap):    operand-visible + FU_LATENCY
//! write port capture for element m:   write-skew + m   (captures whatever
//!                                     the driving FU's result register
//!                                     holds that cycle)
//! ```
//!
//! Constraints solved here: every functional unit's two stream operands
//! must become visible on the same cycle, and each write port's skew must
//! equal its driving unit's result-visible offset.  Streams are element-
//! synchronous (one element per cycle per active port), so alignment at
//! element 0 aligns the whole stream.

use crate::config::{Fu, Pe, Source, VmeConfig};
use crate::VmeError;

/// Cycles from a read AGU's address issue to that data being visible at the
/// functional units (`vme_ringbuf`'s registered read port).
pub const READ_LATENCY: u32 = 1;

/// Cycles from a functional unit sampling its operands to its result being
/// visible on the staging bus / at the write port (`vme_fu`'s result
/// register).
pub const FU_LATENCY: u32 = 1;

/// Largest value the MODE[23:16] skew field can hold.
pub const MAX_SKEW: u32 = 255;

/// The solved schedule: one skew per used port.  `None` = port unused
/// (its AGU is left disabled).
#[derive(Debug, Clone, Default)]
pub struct TimingPlan {
    pub top_skew: [Option<u32>; 4],
    pub base_skew: [Option<u32>; 4],
    pub write_skew: [Option<u32>; 4],
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

/// Derive every port skew for the configured dataflow graph.
pub fn solve(cfg: &VmeConfig) -> Result<TimingPlan, Vec<VmeError>> {
    let mut errors = Vec::new();

    // collect configured functional units and their stream inputs
    let mut nodes: Vec<(Node, Vec<Source>)> = Vec::new();
    for pe in Pe::ALL {
        for fu in [Fu::Fu0, Fu::Fu1] {
            let unit = match fu {
                Fu::Fu0 => &cfg.pe(pe).fu0,
                Fu::Fu1 => &cfg.pe(pe).fu1,
            };
            let Some(op) = unit.op else { continue };
            let mut inputs = Vec::new();
            if let Some(b) = unit.back {
                inputs.push(b);
            }
            if op.opcode.uses_front_stream() {
                if let Some(f) = unit.front {
                    inputs.push(f);
                }
            }
            nodes.push((Node { pe, fu }, inputs));
        }
    }

    // topological order over staging edges (Kahn)
    let dep_idx = |src: Source| -> Option<usize> {
        node_of(src).and_then(|n| nodes.iter().position(|(m, _)| *m == n))
    };
    let mut indeg: Vec<usize> = nodes
        .iter()
        .map(|(_, ins)| ins.iter().filter_map(|s| dep_idx(*s)).count())
        .collect();
    let mut order: Vec<usize> = Vec::new();
    let mut ready: Vec<usize> =
        (0..nodes.len()).filter(|i| indeg[*i] == 0).collect();
    while let Some(i) = ready.pop() {
        order.push(i);
        for (j, (_, ins)) in nodes.iter().enumerate() {
            if ins.iter().filter_map(|s| dep_idx(*s)).any(|d| d == i) {
                indeg[j] -= 1;
                if indeg[j] == 0 {
                    ready.push(j);
                }
            }
        }
    }
    if order.len() != nodes.len() {
        errors.push(VmeError::StagingCycle);
        return Err(errors);
    }

    // per-PE port skews: None = unconstrained so far
    let mut top: [Option<u32>; 4] = [None; 4];
    let mut base: [Option<u32>; 4] = [None; 4];
    for pe in Pe::ALL {
        let p = pe.index();
        if let Some(s) = cfg.pe(pe).read_top.skew {
            top[p] = Some(s as u32);
        }
        if let Some(s) = cfg.pe(pe).read_base.skew {
            base[p] = Some(s as u32);
        }
    }

    let mut out_visible: Vec<Option<u32>> = vec![None; nodes.len()];
    let mut top_used = [false; 4];
    let mut base_used = [false; 4];

    for &i in &order {
        let (node, ref inputs) = nodes[i];
        let p = node.pe.index();

        // gather fixed visible-times: staging producers and already-pinned ports
        let mut fixed: Option<(u32, String)> = None;
        let conflict = |errors: &mut Vec<VmeError>,
                            fixed: &mut Option<(u32, String)>,
                            v: u32,
                            what: String| {
            match fixed {
                None => *fixed = Some((v, what)),
                Some((fv, fwhat)) if *fv != v => {
                    errors.push(VmeError::SkewConflict {
                        fu: fu_name(node),
                        a: fwhat.clone(),
                        a_cycle: *fv,
                        b: what,
                        b_cycle: v,
                    });
                }
                _ => {}
            }
        };

        for src in inputs {
            match src {
                Source::Buf(b) => {
                    let port = if b.is_top() { &top[p] } else { &base[p] };
                    if let Some(s) = port {
                        conflict(&mut errors, &mut fixed, s + READ_LATENCY,
                                 format!("{} read of {:?}", fu_name(node), b));
                    }
                }
                _ => {
                    let d = dep_idx(*src).unwrap();
                    if let Some(v) = out_visible[d] {
                        conflict(&mut errors, &mut fixed, v,
                                 format!("staging tap of {}", fu_name(nodes[d].0)));
                    }
                }
            }
        }

        // element-0 operand-visible time: pinned value, else the minimum
        let v = fixed.map(|(v, _)| v).unwrap_or(READ_LATENCY);

        // pin the buffer ports this unit reads
        for src in inputs {
            if let Source::Buf(b) = src {
                let (port, used) = if b.is_top() {
                    (&mut top[p], &mut top_used[p])
                } else {
                    (&mut base[p], &mut base_used[p])
                };
                *used = true;
                match port {
                    None => *port = Some(v - READ_LATENCY),
                    Some(s) if *s + READ_LATENCY != v => {
                        errors.push(VmeError::SkewConflict {
                            fu: fu_name(node),
                            a: format!("port skew {} already pinned", s),
                            a_cycle: *s + READ_LATENCY,
                            b: format!("read of {:?}", b),
                            b_cycle: v,
                        });
                    }
                    _ => {}
                }
            }
        }

        out_visible[i] = Some(v + FU_LATENCY);
    }

    // write ports: skew = driving unit's result-visible offset
    let mut plan = TimingPlan::default();
    for pe in Pe::ALL {
        let p = pe.index();
        let elem = cfg.pe(pe);
        if !elem.is_configured() || elem.write_disabled {
            continue;
        }
        let driver = if elem.fu1.is_configured() { Fu::Fu1 } else { Fu::Fu0 };
        let d = nodes
            .iter()
            .position(|(n, _)| *n == Node { pe, fu: driver })
            .expect("configured driver is a node");
        let skew = match elem.write.skew {
            Some(s) => s as u32,
            None => out_visible[d].unwrap_or(READ_LATENCY + FU_LATENCY),
        };
        if skew > MAX_SKEW {
            errors.push(VmeError::SkewRange { fu: fu_name(nodes[d].0), skew });
        }
        plan.write_skew[p] = Some(skew);
    }

    for p in 0..4 {
        if top_used[p] {
            plan.top_skew[p] = Some(top[p].unwrap_or(0));
        }
        if base_used[p] {
            plan.base_skew[p] = Some(base[p].unwrap_or(0));
        }
    }

    if errors.is_empty() {
        Ok(plan)
    } else {
        Err(errors)
    }
}
