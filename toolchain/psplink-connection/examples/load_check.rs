//! Manual hardware checkpoint for the psplink-connection crate: connect,
//! load a PRX, stream its stdout, and print the completion outcome. Not
//! wired into any example's `host/` crate — this is a standalone smoke test.
//!
//! Usage: cargo run --example load_check -p psplink-connection -- <host0_dir> <host1_dir> <prx_path>

use psplink_connection::PSPConnection;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let host0 = args.get(1).cloned().unwrap_or_else(|| ".".to_string());
    let host1 = args.get(2).cloned().unwrap_or_else(|| ".".to_string());
    let prx_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "host1:hello-psp.prx".to_string());

    eprintln!("connecting (host0={host0}, host1={host1})...");
    let conn = PSPConnection::connect(Path::new(&host0), Path::new(&host1), Default::default())
        .expect("connect failed");
    eprintln!("connected. loading {prx_path}...");

    let t0 = Instant::now();
    let mut first_stdout_at = None;
    let outcome = conn
        .load_program(&prx_path, |bytes| {
            if first_stdout_at.is_none() {
                first_stdout_at = Some(t0.elapsed());
            }
            std::io::stdout().write_all(bytes).ok();
        })
        .expect("load_program failed");
    let marker_at = t0.elapsed();

    eprintln!(
        "\n==> first stdout at {:?}, completion marker at {:?}",
        first_stdout_at, marker_at
    );
    eprintln!("==> outcome: {outcome:?}");
    conn.disconnect();
}
