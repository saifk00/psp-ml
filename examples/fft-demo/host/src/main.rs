//! Host runner for `fft-demo`. Deploys the PRX and summarises the per-variant
//! results, sorted fastest-first per shape, with the speedup over the scalar
//! baseline that codegen currently emits.

use psplink_connection::{LoadOutcome, PSPConnection};
use std::io::Write;
use std::path::Path;

#[allow(dead_code)]
fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().expect("PRX_PATH has no parent directory");
    let prx_name = prx_path.file_name().unwrap().to_str().unwrap();

    eprintln!("==> Connecting (host0/host1:{})...", prx_dir.display());
    let conn = PSPConnection::connect(prx_dir, prx_dir, Default::default()).unwrap_or_else(|e| {
        eprintln!("error: failed to connect to PSP: {e}");
        std::process::exit(1);
    });

    eprintln!("==> Loading host1:{prx_name}");
    let outcome = conn
        .load_program(&format!("host1:{prx_name}"), |bytes| {
            std::io::stdout().write_all(bytes).ok();
        })
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    conn.disconnect();

    if !matches!(outcome, LoadOutcome::Success) {
        eprintln!("==> Run failed: {outcome:?}");
        std::process::exit(1);
    }

}

