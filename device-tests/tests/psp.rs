//! `cargo test -p device-tests` — runs psp-rt's checks on a USB-connected PSP.
//!
//! `cargo test` runs the same `device_test::SUITES` list on the host, but
//! there it compiles to the scalar fallbacks — every `#[cfg(target_os = "psp")]`
//! VFPU block is excluded, and the `device:` group of each suite is excluded
//! too. So the host suite validates the *algorithms* and this validates the
//! *assembly* and the hardware, which is where the bugs have actually been.
//!
//! Needs a PSP running psplink over USB. The package is not a workspace
//! default member, so a plain `cargo test` never reaches it.

use device_tests::{Event, Feed};
use psplink_connection::PSPConnection;
use std::io::Write;
use std::path::Path;

fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().expect("PRX_PATH has no parent directory");
    let prx_name = prx_path
        .file_name()
        .expect("PRX_PATH has no file name")
        .to_str()
        .expect("PRX_PATH is not valid UTF-8");

    // Results come back over stdout, so neither drive needs to point anywhere
    // in particular; host1 is where the PRX is loaded from.
    eprintln!("   Waiting for a PSP running psplink over USB...");
    let conn = PSPConnection::connect(prx_dir, prx_dir, Default::default()).unwrap_or_else(|e| {
        eprintln!("error: failed to connect to PSP: {e}");
        std::process::exit(1);
    });

    eprintln!("   Running host1:{prx_name}\n");
    let mut feed = Feed::new();
    let mut printer = Printer::default();
    let outcome = conn
        .load_program(&format!("host1:{prx_name}"), |bytes| {
            for event in feed.push(bytes) {
                printer.show(event);
            }
        })
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    conn.disconnect();
    printer.close();

    match feed.finish(outcome) {
        Ok(summary) => println!("\n{summary}\n"),
        Err(failure) => {
            println!("\n{failure}");
            std::process::exit(1);
        }
    }
}

/// Prints progress in cargo's shape, opening each line at `start` so a check
/// that never returns still leaves its name on screen.
#[derive(Default)]
struct Printer {
    open: bool,
}

impl Printer {
    fn show(&mut self, event: Event) {
        match event {
            Event::Plan(n) => println!("running {n} device checks"),
            Event::Start(name) => {
                print!("test {name} ... ");
                let _ = std::io::stdout().flush();
                self.open = true;
            }
            Event::Ok(_) => {
                self.open = false;
                println!("ok");
            }
            Event::Fail(_) => {
                self.open = false;
                println!("FAILED");
            }
            Event::Done { .. } => self.close(),
            Event::Echo(text) => {
                // Don't let a check's own output land in the middle of a
                // half-written result line.
                self.close();
                println!("{text}");
            }
        }
    }

    fn close(&mut self) {
        if self.open {
            println!();
            self.open = false;
        }
    }
}
