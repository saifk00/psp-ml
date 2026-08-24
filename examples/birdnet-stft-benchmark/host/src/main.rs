//! Host runner for the STFT frontend benchmark. Deploys one PRX per mode
//! (dense_gather, strided_view — see the sibling build.rs for why they are
//! separate), streams each run's stdout, then verifies and compares:
//!
//!   - each mode's outputs against the TFLite golden (rel-RMS per frame)
//!   - dense vs strided bit-exact (they run the same f32 ops in the same order)
//!   - the memory story: compile-time arena/blob per mode, and the on-device
//!     free-memory measurement each PRX printed after loading
//!
//! Exits nonzero on any mismatch. `--mode dense_gather|strided_view|both`
//! (default both).

use psplink_connection::{LoadOutcome, PSPConnection};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

const OUT_2048: usize = 511 * 1025;
const OUT_1024: usize = 511 * 513;

fn main() {
    let mode = parse_mode();
    let deploy_dir = PathBuf::from(env!("PRX_DEPLOY_DIR"));
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mount_dir = manifest_dir.join("..");
    let golden_dir = manifest_dir.join("../../../models/birdnet/stft");

    let runs: Vec<(&str, &str)> = match mode.as_str() {
        "dense_gather" => vec![("dense_gather", "stft_dense.prx")],
        "strided_view" => vec![("strided_view", "stft_strided.prx")],
        _ => vec![
            ("dense_gather", "stft_dense.prx"),
            ("strided_view", "stft_strided.prx"),
        ],
    };

    for stale in [
        "out_dense_2048.bin",
        "out_dense_1024.bin",
        "out_strided_2048.bin",
        "out_strided_1024.bin",
    ] {
        let _ = std::fs::remove_file(mount_dir.join(stale));
    }

    eprintln!(
        "==> Connecting (host0:{}, host1:{})...",
        mount_dir.display(),
        deploy_dir.display()
    );
    let conn = PSPConnection::connect(&mount_dir, &deploy_dir, Default::default())
        .unwrap_or_else(|e| {
            eprintln!("error: failed to connect to PSP: {e}");
            std::process::exit(1);
        });

    let mut stats: Vec<(String, HashMap<String, u64>)> = Vec::new();
    for (mode_name, prx) in &runs {
        eprintln!("==> Loading host1:{prx} ({mode_name})");
        let mut captured = String::new();
        let outcome = conn
            .load_program(&format!("host1:{prx}"), |bytes| {
                std::io::stdout().write_all(bytes).ok();
                captured.push_str(&String::from_utf8_lossy(bytes));
            })
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
        match outcome {
            LoadOutcome::Success => {}
            LoadOutcome::TimedOut => {
                eprintln!("==> Timed out: app_main exceeded its budget and was terminated");
                std::process::exit(1);
            }
            other => {
                eprintln!("==> Device run failed: {other:?}");
                std::process::exit(1);
            }
        }
        match parse_bench_line(&captured) {
            Some(kv) => stats.push((mode_name.to_string(), kv)),
            None => {
                eprintln!("error: no #stftbench line in {mode_name}'s output");
                std::process::exit(1);
            }
        }
    }

    // Correctness: each mode against the TFLite golden, then across modes.
    let golden_2048 = read_f32(&golden_dir.join("golden_2048.bin"), OUT_2048);
    let golden_1024 = read_f32(&golden_dir.join("golden_1024.bin"), OUT_1024);
    let mut ok = true;
    let mut outs: HashMap<&str, (Vec<f32>, Vec<f32>)> = HashMap::new();
    for (mode_name, _) in &runs {
        let tag = if *mode_name == "dense_gather" { "dense" } else { "strided" };
        let o0 = read_f32(&mount_dir.join(format!("out_{tag}_2048.bin")), OUT_2048);
        let o1 = read_f32(&mount_dir.join(format!("out_{tag}_1024.bin")), OUT_1024);
        ok &= check_golden(&o0, &golden_2048, 1025, &format!("{mode_name} L=2048"));
        ok &= check_golden(&o1, &golden_1024, 513, &format!("{mode_name} L=1024"));
        outs.insert(mode_name, (o0, o1));
    }
    if let (Some((d0, d1)), Some((s0, s1))) =
        (outs.get("dense_gather"), outs.get("strided_view"))
    {
        let exact = d0
            .iter()
            .zip(s0.iter())
            .chain(d1.iter().zip(s1.iter()))
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "dense vs strided outputs: {}",
            if exact { "bit-identical" } else { "MISMATCH" }
        );
        ok &= exact;
    }

    // The comparison table.
    println!();
    println!("mode           arena         blob     free after load    max block      frontend");
    for (mode_name, kv) in &stats {
        println!(
            "{:<13} {:>8} KiB {:>8} KiB {:>13} KiB {:>10} KiB {:>10} ms",
            mode_name,
            kv["arena_bytes"] / 1024,
            kv["blob_bytes"] / 1024,
            kv["free_bytes"] / 1024,
            kv["max_block_bytes"] / 1024,
            kv["avg_us"] / 1000,
        );
    }
    if stats.len() == 2 {
        let dense = &stats[0].1;
        let strided = &stats[1].1;
        let static_delta = (dense["arena_bytes"] + dense["blob_bytes"]) as i64
            - (strided["arena_bytes"] + strided["blob_bytes"]) as i64;
        let free_delta = strided["free_bytes"] as i64 - dense["free_bytes"] as i64;
        println!();
        println!(
            "strided_view saves {} KiB at compile time (arena+blob); the device \
             measured {} KiB more free memory after load",
            static_delta / 1024,
            free_delta / 1024
        );
        let (d_us, s_us) = (dense["avg_us"], strided["avg_us"]);
        println!(
            "frontend time: {} ms -> {} ms ({:+.1}%)",
            d_us / 1000,
            s_us / 1000,
            (s_us as f64 - d_us as f64) * 100.0 / d_us as f64
        );
    }

    if !ok {
        eprintln!("FAIL: output mismatch");
        std::process::exit(1);
    }
    println!("PASS");
}

fn parse_mode() -> String {
    let args: Vec<String> = std::env::args().collect();
    let mode = match args.iter().position(|a| a == "--mode") {
        Some(i) => args
            .get(i + 1)
            .unwrap_or_else(|| {
                eprintln!("--mode needs a value");
                std::process::exit(2);
            })
            .clone(),
        None => "both".to_string(),
    };
    if !["dense_gather", "strided_view", "both"].contains(&mode.as_str()) {
        eprintln!("--mode must be dense_gather, strided_view or both");
        std::process::exit(2);
    }
    mode
}

fn parse_bench_line(output: &str) -> Option<HashMap<String, u64>> {
    let line = output
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with("#stftbench "))?;
    let mut kv = HashMap::new();
    for part in line.trim().split_whitespace().skip(1) {
        let (k, v) = part.split_once('=')?;
        if let Ok(n) = v.parse::<u64>() {
            kv.insert(k.to_string(), n);
        }
    }
    Some(kv)
}

fn read_f32(path: &Path, len: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("error: failed to read {}: {e}", path.display());
        std::process::exit(1);
    });
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if vals.len() != len {
        eprintln!("error: {} has {} floats, expected {len}", path.display(), vals.len());
        std::process::exit(1);
    }
    vals
}

/// Max per-frame error normalised by that frame's RMS — the same yardstick
/// as the device crate's local check.
fn check_golden(got: &[f32], golden: &[f32], bins: usize, label: &str) -> bool {
    let mut worst = 0.0f64;
    for (g_row, w_row) in got.chunks_exact(bins).zip(golden.chunks_exact(bins)) {
        let rms = (w_row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / bins as f64)
            .sqrt()
            .max(1e-9);
        for (g, w) in g_row.iter().zip(w_row.iter()) {
            worst = worst.max(((g - w).abs() as f64) / rms);
        }
    }
    let ok = worst < 1e-3;
    println!(
        "{label}: max err/frame-RMS vs TFLite golden = {worst:.2e} {}",
        if ok { "(ok)" } else { "EXCEEDS 1e-3" }
    );
    ok
}
