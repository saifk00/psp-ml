//! Turns a device run's stdout into a verdict.
//!
//! The device runner (`psp-rt`'s `device-tests` binary) streams its results as
//! protocol lines; [`Feed`] reassembles them from arbitrary USB chunks, tracks
//! what ran, and decides whether the run passed. Every way a run can go wrong
//! is decided here rather than in the driver, so all of them are testable from
//! a canned transcript with no PSP attached — including the ones that are
//! expensive to reproduce on hardware, like a check that hangs.

use psp_rt::device_test::protocol::{self, Line};
use psplink_connection::LoadOutcome;
use std::fmt;

/// Something the device said, ready for the driver to print.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// How many checks the device is about to run.
    Plan(u32),
    /// A check is about to run. Nothing more will be printed if it hangs.
    Start(String),
    Ok(String),
    Fail(String),
    Done { passed: u32, failed: u32 },
    /// Not a protocol line — pass it through so `dprintln!` debugging inside a
    /// check still reaches the terminal.
    Echo(String),
}

/// A run that finished with every check accounted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    pub passed: u32,
    pub failed: u32,
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "test result: ok. {} passed; {} failed",
            self.passed, self.failed
        )
    }
}

/// Why a run is not a pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// Not one line came back. Its own variant because the cause is different
    /// from a truncated run, and so is the fix.
    NoOutput { outcome: LoadOutcome },
    /// Checks ran and returned `false`.
    ChecksFailed(Vec<String>),
    /// The module didn't finish: it panicked, or `module!` terminated it.
    Interrupted {
        outcome: LoadOutcome,
        last_started: Option<String>,
    },
    /// The module exited cleanly but never said `done` — it stopped partway
    /// through without faulting, so either a check exited the thread or the
    /// output stream was cut short.
    NoVerdict { last_started: Option<String> },
    /// The device's own tally disagrees with the lines it sent, so one of the
    /// two is lying and neither can be trusted.
    CountMismatch {
        reported: Summary,
        observed: Summary,
    },
    /// The suite list is empty — a green run that proves nothing.
    NothingRan,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Failure::NoOutput { outcome } => write!(
                f,
                "test result: FAILED. the device sent nothing at all ({outcome:?}).\n    \
                 psplink's first `ld` after a `reset` reliably behaves this way — \
                 run the command again before looking further."
            ),
            Failure::ChecksFailed(names) => {
                writeln!(f, "test result: FAILED. {} check(s) failed:", names.len())?;
                for name in names {
                    writeln!(f, "    {name}")?;
                }
                Ok(())
            }
            Failure::Interrupted {
                outcome,
                last_started,
            } => {
                let what = match outcome {
                    LoadOutcome::Panicked => "the module panicked".to_string(),
                    LoadOutcome::TimedOut => {
                        "the module ran past its budget and was terminated".to_string()
                    }
                    LoadOutcome::ShellError(v) => format!("psplink shell error 0x{v:08X}"),
                    LoadOutcome::KernelError(v) => format!("kernel error 0x{v:08X}"),
                    LoadOutcome::Success => "the module exited".to_string(),
                };
                write!(f, "test result: FAILED. {what}, {}", blame(last_started))
            }
            Failure::NoVerdict { last_started } => write!(
                f,
                "test result: FAILED. the run ended without a verdict, {}",
                blame(last_started)
            ),
            Failure::CountMismatch { reported, observed } => write!(
                f,
                "test result: FAILED. device reported {}/{} passed/failed but sent {}/{}",
                reported.passed, reported.failed, observed.passed, observed.failed
            ),
            Failure::NothingRan => write!(
                f,
                "test result: FAILED. nothing ran — is the suite list empty?"
            ),
        }
    }
}

fn blame(last_started: &Option<String>) -> String {
    match last_started {
        Some(name) => format!("during {name}"),
        None => "after the last check reported".to_string(),
    }
}

/// Consumes a run's output and decides the verdict.
///
/// Feed it every chunk `load_program` hands out, then call [`Feed::finish`].
#[derive(Debug, Default)]
pub struct Feed {
    /// Bytes since the last newline. USB chunks split lines anywhere, so the
    /// buffer is bytes, not text — splitting a UTF-8 sequence would otherwise
    /// corrupt it.
    pending: Vec<u8>,
    heard_anything: bool,
    /// The check that started but hasn't reported — the culprit if the device
    /// stops here.
    in_flight: Option<String>,
    passed: u32,
    failed: Vec<String>,
    reported: Option<Summary>,
}

impl Feed {
    pub fn new() -> Self {
        Self::default()
    }

    /// Consume one chunk of device output, returning whatever it completed.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<Event> {
        self.pending.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(nl) = self.pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=nl).collect();
            let text = String::from_utf8_lossy(&line);
            let text = text.trim_end_matches(['\r', '\n']);
            self.heard_anything = true;
            events.push(match protocol::parse(text) {
                Some(line) => self.record(line),
                None => Event::Echo(text.to_string()),
            });
        }
        events
    }

    fn record(&mut self, line: Line<'_>) -> Event {
        match line {
            Line::Plan(n) => Event::Plan(n),
            Line::Start(name) => {
                let name = name.to_string();
                self.in_flight = Some(name.clone());
                Event::Start(name)
            }
            Line::Ok(name) => {
                self.in_flight = None;
                self.passed += 1;
                Event::Ok(name.to_string())
            }
            Line::Fail(name) => {
                self.in_flight = None;
                let name = name.to_string();
                self.failed.push(name.clone());
                Event::Fail(name)
            }
            Line::Done { passed, failed } => {
                self.reported = Some(Summary { passed, failed });
                Event::Done { passed, failed }
            }
        }
    }

    /// Decide the run, given how the module itself exited.
    ///
    /// A trailing partial line is dropped: the device always ends a protocol
    /// line with a newline, so anything left over is a truncated print.
    pub fn finish(self, outcome: LoadOutcome) -> Result<Summary, Failure> {
        // Total silence first: however the module exited, the advice is the
        // same, and "no verdict, after the last check reported" is a confusing
        // way to say "there were no checks".
        if !self.heard_anything {
            return Err(Failure::NoOutput { outcome });
        }
        // An interrupted run comes next: its counts are meaningless, and the
        // in-flight name is the whole reason the protocol exists.
        if outcome != LoadOutcome::Success {
            return Err(Failure::Interrupted {
                outcome,
                last_started: self.in_flight,
            });
        }
        let Some(reported) = self.reported else {
            return Err(Failure::NoVerdict {
                last_started: self.in_flight,
            });
        };
        if !self.failed.is_empty() {
            return Err(Failure::ChecksFailed(self.failed));
        }
        let observed = Summary {
            passed: self.passed,
            failed: self.failed.len() as u32,
        };
        if reported != observed {
            return Err(Failure::CountMismatch { reported, observed });
        }
        if observed.passed + observed.failed == 0 {
            return Err(Failure::NothingRan);
        }
        Ok(observed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a whole transcript one byte at a time, which is the worst case the
    /// USB chunking can produce.
    fn feed(text: &str) -> Feed {
        let mut f = Feed::new();
        for b in text.as_bytes() {
            f.push(&[*b]);
        }
        f
    }

    const GOOD: &str = "\
#psptest plan 2
#psptest start kernels::test_relu
#psptest ok kernels::test_relu
#psptest start mem::test_partition_alloc_reclaimed
#psptest ok mem::test_partition_alloc_reclaimed
#psptest done 2 0
";

    #[test]
    fn clean_run_passes() {
        let summary = feed(GOOD).finish(LoadOutcome::Success).unwrap();
        assert_eq!(
            summary,
            Summary {
                passed: 2,
                failed: 0
            }
        );
    }

    #[test]
    fn chunk_boundaries_do_not_matter() {
        let mut whole = Feed::new();
        whole.push(GOOD.as_bytes());
        assert_eq!(
            whole.finish(LoadOutcome::Success),
            feed(GOOD).finish(LoadOutcome::Success)
        );
    }

    #[test]
    fn failing_checks_are_named() {
        let text = "\
#psptest plan 2
#psptest start kernels::test_relu
#psptest fail kernels::test_relu
#psptest start kernels::test_bias_add
#psptest ok kernels::test_bias_add
#psptest done 1 1
";
        assert_eq!(
            feed(text).finish(LoadOutcome::Success),
            Err(Failure::ChecksFailed(vec![
                "kernels::test_relu".to_string()
            ]))
        );
    }

    /// The case the file-based reporting could never handle: the device stops
    /// mid-check, so the only evidence is the `start` line.
    #[test]
    fn a_hang_is_blamed_on_the_check_that_was_running() {
        let text = "\
#psptest plan 2
#psptest start kernels::test_relu
#psptest ok kernels::test_relu
#psptest start kernels::test_matmul_bt_tiled_large
";
        assert_eq!(
            feed(text).finish(LoadOutcome::TimedOut),
            Err(Failure::Interrupted {
                outcome: LoadOutcome::TimedOut,
                last_started: Some("kernels::test_matmul_bt_tiled_large".to_string()),
            })
        );
    }

    /// Output cut off mid-line is still attributed, because the `start` line
    /// before it was complete.
    #[test]
    fn truncated_output_is_attributed() {
        let text = "\
#psptest plan 2
#psptest start kernels::test_relu
#psptest ok kernels::test_relu
#psptest start kernels::test_pow_const_matches_libm
gemm: partial outp";
        assert_eq!(
            feed(text).finish(LoadOutcome::Success),
            Err(Failure::NoVerdict {
                last_started: Some("kernels::test_pow_const_matches_libm".to_string()),
            })
        );
    }

    #[test]
    fn a_panic_between_checks_is_still_a_failure() {
        let text = "\
#psptest plan 2
#psptest start kernels::test_relu
#psptest ok kernels::test_relu
";
        assert_eq!(
            feed(text).finish(LoadOutcome::Panicked),
            Err(Failure::Interrupted {
                outcome: LoadOutcome::Panicked,
                last_started: None,
            })
        );
    }

    #[test]
    fn a_missing_result_line_is_caught() {
        // `done` claims three passes; only two arrived.
        let text = "\
#psptest plan 3
#psptest ok kernels::test_relu
#psptest ok kernels::test_bias_add
#psptest done 3 0
";
        let err = feed(text).finish(LoadOutcome::Success).unwrap_err();
        assert!(matches!(err, Failure::CountMismatch { .. }), "{err:?}");
    }

    /// Observed on hardware: the first `ld` after a psplink `reset` runs but
    /// sends no output. Say so, rather than blaming a check.
    #[test]
    fn total_silence_is_called_out_separately() {
        assert_eq!(
            feed("").finish(LoadOutcome::Success),
            Err(Failure::NoOutput {
                outcome: LoadOutcome::Success
            })
        );
        assert_eq!(
            feed("").finish(LoadOutcome::TimedOut),
            Err(Failure::NoOutput {
                outcome: LoadOutcome::TimedOut
            })
        );
    }

    #[test]
    fn an_empty_suite_is_not_a_pass() {
        let text = "\
#psptest plan 0
#psptest done 0 0
";
        assert_eq!(
            feed(text).finish(LoadOutcome::Success),
            Err(Failure::NothingRan)
        );
    }

    #[test]
    fn check_output_passes_through() {
        let mut f = Feed::new();
        let events = f.push(b"gemm: 1992 MFLOP/s\n#psptest plan 1\n");
        assert_eq!(
            events,
            vec![
                Event::Echo("gemm: 1992 MFLOP/s".to_string()),
                Event::Plan(1),
            ]
        );
    }
}
