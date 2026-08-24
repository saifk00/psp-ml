//! Host runner for the whole-frontend benchmark. Deploys one PRX per mode
//! (dense_tflite, custom_ops), streams each run's stdout, verifies both
//! modes' outputs against the TFLite goldens (indexing custom_ops's
//! bank-major layout), and prints the comparison table. Exits nonzero on
//! any mismatch. `--mode dense_tflite|custom_ops|both` (default both).

use psplink_connection::{LoadOutcome, PSPConnection};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

const N_WINDOWS: usize = 511;
const N_BANKS: usize = 96;
const OUT_LEN: usize = N_WINDOWS * N_BANKS;

fn main() {
    let mode = parse_mode();
    let deploy_dir = PathBuf::from(env!("PRX_DEPLOY_DIR"));
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mount_dir = manifest_dir.join("..");
    let golden_dir = manifest_dir.join("../../../models/birdnet/stft");

    let runs: Vec<(&str, &str, &str)> = match mode.as_str() {
        "dense_tflite" => vec![("dense_tflite", "frontend_dense.prx", "dense")],
        "custom_ops" => vec![("custom_ops", "frontend_custom.prx", "custom")],
        _ => vec![
            ("dense_tflite", "frontend_dense.prx", "dense"),
            ("custom_ops", "frontend_custom.prx", "custom"),
        ],
    };

    for stale in [
        "out_dense_front_2048.bin",
        "out_dense_front_1024.bin",
        "out_custom_front_2048.bin",
        "out_custom_front_1024.bin",
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
    for (mode_name, prx, _) in &runs {
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
                eprintln!("error: no #frontbench line in {mode_name}'s output");
                std::process::exit(1);
            }
        }
    }

    let golden_2048 = read_f32(&golden_dir.join("golden_mel_2048.bin"), OUT_LEN);
    let golden_1024 = read_f32(&golden_dir.join("golden_mel_1024.bin"), OUT_LEN);
    let mut ok = true;
    for (mode_name, _, tag) in &runs {
        // Both gate at 1e-2 for the whole frontend: FFT last-bit differences
        // vs TFLite get amplified by the x^~0.17-0.22 compression on
        // near-zero mel values (measured: dense 5.4e-3, custom 4.4e-3).
        // custom is bank-major.
        let custom = *mode_name == "custom_ops";
        let tol = 1e-2;
        let o0 = read_f32(&mount_dir.join(format!("out_{tag}_front_2048.bin")), OUT_LEN);
        let o1 = read_f32(&mount_dir.join(format!("out_{tag}_front_1024.bin")), OUT_LEN);
        ok &= check_golden(&o0, &golden_2048, &format!("{mode_name} L=2048"), tol, custom);
        ok &= check_golden(&o1, &golden_1024, &format!("{mode_name} L=1024"), tol, custom);
    }

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
        let custom = &stats[1].1;
        let static_delta = (dense["arena_bytes"] + dense["blob_bytes"]) as i64
            - (custom["arena_bytes"] + custom["blob_bytes"]) as i64;
        let free_delta = custom["free_bytes"] as i64 - dense["free_bytes"] as i64;
        println!();
        println!(
            "custom_ops saves {} KiB at compile time (arena+blob); the device \
             measured {} KiB more free memory after load",
            static_delta / 1024,
            free_delta / 1024
        );
        let (d_us, c_us) = (dense["avg_us"], custom["avg_us"]);
        println!(
            "frontend time: {} ms -> {} ms ({:+.1}%)",
            d_us / 1000,
            c_us / 1000,
            (c_us as f64 - d_us as f64) * 100.0 / d_us as f64
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
    if !["dense_tflite", "custom_ops", "both"].contains(&mode.as_str()) {
        eprintln!("--mode must be dense_tflite, custom_ops or both");
        std::process::exit(2);
    }
    mode
}

fn parse_bench_line(output: &str) -> Option<HashMap<String, u64>> {
    let line = output
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with("#frontbench "))?;
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

/// Max per-frame error normalised by that frame's RMS. The golden is
/// `[N_WINDOWS, N_BANKS]` row-major; `transposed` says `got` is bank-major.
fn check_golden(got: &[f32], golden: &[f32], label: &str, tol: f64, transposed: bool) -> bool {
    let mut worst = 0.0f64;
    for (m, w_row) in golden.chunks_exact(N_BANKS).enumerate() {
        let rms = (w_row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
            / N_BANKS as f64)
            .sqrt()
            .max(1e-9);
        for (b, w) in w_row.iter().enumerate() {
            let g = if transposed {
                got[b * N_WINDOWS + m]
            } else {
                got[m * N_BANKS + b]
            };
            worst = worst.max(((g - w).abs() as f64) / rms);
        }
    }
    let ok = worst < tol;
    println!(
        "{label}: max err/frame-RMS vs TFLite golden = {worst:.2e} {}",
        if ok { "(ok)" } else { "EXCEEDS TOLERANCE" }
    );
    ok
}
