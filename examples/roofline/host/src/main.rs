//! Host runner for `roofline`. Provides `bwtest.bin` in the mounted dir for
//! the device's hostfs throughput tests, deploys the PRX, then summarizes
//! `roofline.json` (rates are computed here — the device only records raw
//! bytes/flops/us).

use psplink_connection::{LoadOutcome, PSPConnection};
use std::io::Write;
use std::path::Path;

fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().expect("PRX_PATH has no parent directory");
    let prx_name = prx_path.file_name().unwrap().to_str().unwrap();

    // 8 MiB test file for the device's sceIoRead throughput sweep.
    let bwtest = prx_dir.join("bwtest.bin");
    if !bwtest.exists() || bwtest.metadata().map(|m| m.len()).unwrap_or(0) != 8 * 1024 * 1024 {
        std::fs::write(&bwtest, vec![0xA5u8; 8 * 1024 * 1024]).expect("write bwtest.bin");
    }
    let json_path = prx_dir.join("roofline.json");
    let _ = std::fs::remove_file(&json_path);

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

    match outcome {
        LoadOutcome::Success => {}
        other => {
            eprintln!("==> Run failed: {other:?}");
            std::process::exit(1);
        }
    }

    let json = std::fs::read_to_string(&json_path).unwrap_or_else(|e| {
        eprintln!("error: device did not produce roofline.json: {e}");
        std::process::exit(1);
    });
    println!("\n=== Summary ({}) ===", json_path.display());
    summarize(&json);
}

/// Minimal parse of the known result shape; avoids a serde dependency.
fn summarize(json: &str) {
    for obj in json.split('{').skip(2) {
        let field = |k: &str| -> Option<&str> {
            let pat = format!("\"{k}\":");
            let rest = &obj[obj.find(&pat)? + pat.len()..];
            let end = rest.find([',', '}']).unwrap_or(rest.len());
            Some(rest[..end].trim().trim_matches('"'))
        };
        let (Some(group), Some(name)) = (field("group"), field("name")) else {
            continue;
        };
        let bytes: u64 = field("bytes").and_then(|v| v.parse().ok()).unwrap_or(0);
        let flops: u64 = field("flops").and_then(|v| v.parse().ok()).unwrap_or(0);
        let us: u64 = field("us").and_then(|v| v.parse().ok()).unwrap_or(0);
        if us == 0 {
            continue;
        }
        match group {
            "latency" => {
                // bytes field carries hop count for the pointer chase
                println!("  {name:<28} {:>10.1} ns/hop", us as f64 * 1000.0 / bytes as f64);
            }
            "compute" => {
                println!("  {name:<28} {:>10.1} MFLOP/s", flops as f64 / us as f64);
            }
            _ => {
                println!("  {name:<28} {:>10.2} MB/s", bytes as f64 / us as f64);
            }
        }
    }
}
