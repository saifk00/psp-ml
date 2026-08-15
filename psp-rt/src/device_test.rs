//! The device test registry and the wire protocol its runner speaks.
//!
//! One list, two runners: `cargo test -p psp-rt` runs the checks on the host
//! against the scalar fallbacks, and the `device-tests` binary runs the same
//! list on hardware against the real VFPU assembly (`cargo test -p
//! device-tests` builds it, deploys it, and reports the results). Keeping one
//! source of truth is the point — a check that only exists in the device
//! binary never runs in CI, and one that only exists as a `#[test]` never sees
//! the assembly it is meant to validate.
//!
//! Not every check has a host counterpart, though: partition memory, the
//! hardware profiler and anything else reaching into `psp::sys` only mean
//! something on device. Those go in `device_checks!`'s `device:` group, which
//! emits no `#[test]` and vanishes from `CHECKS` on host builds.
//!
//! Results come back over stdout rather than a result file, because a check
//! that faults or hangs never gets to write a file — and that is exactly the
//! run whose last-started check you need to know. See [`protocol`].

/// A check: a nullary predicate over the runtime, `true` for pass.
///
/// No assertion machinery, no message, no panic: a device-side panic has been
/// observed to lock psplink up the same way a fault does, so a failure is just
/// `false` and the runner names it.
pub type Check = (&'static str, fn() -> bool);

/// A named group of checks, one per area of the runtime.
pub struct Suite {
    pub name: &'static str,
    pub checks: &'static [Check],
}

/// Every suite the device runner walks.
///
/// Adding an area is one line here plus a `device_checks!` invocation in that
/// module; adding a check to an existing area is one identifier.
pub const SUITES: &[Suite] = &[
    Suite {
        name: "kernels",
        checks: crate::kernels::checks::CHECKS,
    },
    Suite {
        name: "mem",
        checks: crate::mem::checks::CHECKS,
    },
];

/// Total number of checks across all suites, for the runner's `plan` line.
pub fn total_checks() -> u32 {
    SUITES.iter().map(|s| s.checks.len() as u32).sum()
}

/// Declare a module's checks, building the runner's table and a `#[test]` per
/// shared check from one list, so adding a check cannot silently skip a runner.
///
/// ```ignore
/// device_checks! {
///     shared: [ test_relu, test_bias_add ],   // both runners
///     device: [ test_partition_alloc ],       // hardware only
/// }
/// ```
///
/// Both groups are always written out, even when empty, so a reader can always
/// tell which kind a check is without counting commas.
macro_rules! device_checks {
    (
        shared: [ $($shared:ident),* $(,)? ],
        device: [ $($device:ident),* $(,)? ] $(,)?
    ) => {
        /// Every check in this suite, as `(name, predicate)`.
        ///
        /// The `device:` group is only present on device builds; its checks
        /// have no host counterpart to run.
        #[cfg(target_os = "psp")]
        pub const CHECKS: &[$crate::device_test::Check] = &[
            $((stringify!($shared), $shared),)*
            $((stringify!($device), $device),)*
        ];

        /// Every check in this suite, as `(name, predicate)`.
        #[cfg(not(target_os = "psp"))]
        pub const CHECKS: &[$crate::device_test::Check] = &[
            $((stringify!($shared), $shared),)*
        ];

        #[cfg(test)]
        mod tests {
            $(
                #[test]
                fn $shared() {
                    assert!(super::$shared(), concat!(stringify!($shared), " failed"));
                }
            )*
        }
    };
}
pub(crate) use device_checks;

/// The line protocol the device runner speaks over stdout.
///
/// The device emits with [`emit`] and the host parses with [`parse`], through
/// this one module, so the two cannot drift. Every line is prefixed so that
/// ordinary `dprintln!` output from a check passes through the host untouched:
///
/// ```text
/// #psptest plan 14
/// #psptest start kernels::test_relu
/// #psptest ok kernels::test_relu
/// #psptest fail kernels::test_matmul_bt_known
/// #psptest done 13 1
/// ```
///
/// `start` is written *before* the check runs. A device fault or hang leaves
/// no further output — and locks psplink until a power-cycle — so that line is
/// the only thing that will name the culprit.
pub mod protocol {
    use core::fmt;

    /// Marks a line as protocol rather than check output.
    pub const PREFIX: &str = "#psptest";

    /// A check's fully-qualified name, `suite::check`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Name<'a> {
        pub suite: &'a str,
        pub check: &'a str,
    }

    impl fmt::Display for Name<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}::{}", self.suite, self.check)
        }
    }

    /// One event in a run.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Line<'a> {
        /// How many checks are about to run.
        Plan(u32),
        /// This check is about to run.
        Start(Name<'a>),
        Ok(Name<'a>),
        Fail(Name<'a>),
        /// The run reached the end. Its absence is itself a failure.
        Done { passed: u32, failed: u32 },
    }

    impl fmt::Display for Line<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Line::Plan(n) => write!(f, "{PREFIX} plan {n}"),
                Line::Start(n) => write!(f, "{PREFIX} start {n}"),
                Line::Ok(n) => write!(f, "{PREFIX} ok {n}"),
                Line::Fail(n) => write!(f, "{PREFIX} fail {n}"),
                Line::Done { passed, failed } => write!(f, "{PREFIX} done {passed} {failed}"),
            }
        }
    }

    /// Write one protocol line to stdout (psplink's, or the host's under
    /// `--features local`).
    pub fn emit(line: Line<'_>) {
        crate::dprintln!("{}", line);
    }

    /// Parse one line of the device's output. `None` for anything that isn't a
    /// protocol line, including a malformed one — the caller echoes those
    /// through as ordinary output.
    pub fn parse(line: &str) -> Option<Line<'_>> {
        let rest = line.trim().strip_prefix(PREFIX)?;
        let mut words = rest.split_whitespace();
        match words.next()? {
            "plan" => Some(Line::Plan(num(words.next()?)?)),
            "start" => Some(Line::Start(name(words.next()?)?)),
            "ok" => Some(Line::Ok(name(words.next()?)?)),
            "fail" => Some(Line::Fail(name(words.next()?)?)),
            "done" => Some(Line::Done {
                passed: num(words.next()?)?,
                failed: num(words.next()?)?,
            }),
            _ => None,
        }
    }

    fn name(s: &str) -> Option<Name<'_>> {
        let (suite, check) = s.split_once("::")?;
        if suite.is_empty() || check.is_empty() {
            return None;
        }
        Some(Name { suite, check })
    }

    fn num(s: &str) -> Option<u32> {
        s.parse().ok()
    }

    #[cfg(test)]
    mod tests {
        extern crate std;
        use super::*;
        use std::format;

        const NAME: Name<'static> = Name {
            suite: "kernels",
            check: "test_relu",
        };

        /// Every variant must survive Display -> parse unchanged; that round
        /// trip is what keeps the emitter and the parser honest.
        #[test]
        fn round_trips() {
            for line in [
                Line::Plan(14),
                Line::Start(NAME),
                Line::Ok(NAME),
                Line::Fail(NAME),
                Line::Done {
                    passed: 13,
                    failed: 1,
                },
            ] {
                let text = format!("{line}");
                assert_eq!(parse(&text), Some(line), "round trip failed for {text}");
            }
        }

        #[test]
        fn tolerates_line_endings() {
            assert_eq!(
                parse("#psptest ok kernels::test_relu\r\n"),
                Some(Line::Ok(NAME))
            );
        }

        #[test]
        fn ignores_check_output() {
            assert_eq!(parse("gemm: 1992 MFLOP/s"), None);
            assert_eq!(parse(""), None);
            // A check that happens to mention the prefix mid-line is not a
            // protocol line.
            assert_eq!(parse("note: #psptest ok kernels::test_relu"), None);
        }

        #[test]
        fn rejects_malformed_lines() {
            assert_eq!(parse("#psptest ok"), None);
            assert_eq!(parse("#psptest ok bare_name"), None);
            assert_eq!(parse("#psptest ok ::test_relu"), None);
            assert_eq!(parse("#psptest plan lots"), None);
            assert_eq!(parse("#psptest done 3"), None);
            assert_eq!(parse("#psptest shrug"), None);
        }
    }
}
