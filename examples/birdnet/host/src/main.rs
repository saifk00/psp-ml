//! Host runner for `birdnet`. `build.rs` cross-compiled the sibling
//! `device/` crate into a `.prx` and handed us its path via `PRX_PATH`.
//! `host0:` is mounted to the birdnet/ example root, which holds
//! `weights.bin` (copied here at deploy time from beside the .prx, so blob
//! and code always agree) — the device loads its resident weights and
//! streams the classifier from there — and receives
//! the device's `results.txt`/`benchmarks.json`. After the run, the device
//! scores are verified against the Python/TFLite golden (`golden.json`).

use psplink_connection::{LoadOutcome, PSPConnection};
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

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mount_dir = Path::new(manifest_dir).join("..");

    // Publish the weight blob staged beside this exact .prx. Doing the copy
    // here, at deploy time, is what guarantees blob and .prx agree: the device
    // reads exactly WEIGHT_BYTES from host0:/weights.bin without validating the
    // file, so a blob left behind by a *different* build of this crate (an
    // unpruned 42.8 MB one against a TOPK .prx wanting 17.8 MB) is read
    // happily and silently computed on at wrong offsets. Both come from
    // prx_dir, so they cannot disagree.
    let staged_weights = prx_dir.join("weights.bin");
    if !staged_weights.exists() {
        eprintln!(
            "error: {} missing — the device build should have staged it",
            staged_weights.display()
        );
        std::process::exit(1);
    }
    std::fs::copy(&staged_weights, mount_dir.join("weights.bin")).unwrap_or_else(|e| {
        eprintln!("error: failed to publish weights.bin to the host0: mount: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "==> Published weights.bin ({} bytes) from {}",
        std::fs::metadata(&staged_weights).map(|m| m.len()).unwrap_or(0),
        prx_dir.display()
    );

    // Same for the pruned-class index map: copy it when this build produced
    // one, and clear any stale copy when it did not, so the golden projection
    // can never be applied to an unpruned run.
    let staged_kept = prx_dir.join("kept_indices.txt");
    if staged_kept.exists() {
        std::fs::copy(&staged_kept, mount_dir.join("kept_indices.txt"))
            .expect("failed to publish kept_indices.txt");
    } else {
        let _ = std::fs::remove_file(mount_dir.join("kept_indices.txt"));
    }

    for stale in ["benchmarks.json", "results.txt"] {
        let _ = std::fs::remove_file(mount_dir.join(stale));
    }

    eprintln!(
        "==> Connecting (host0:{}, host1:{})...",
        mount_dir.display(),
        prx_dir.display()
    );
    let conn = PSPConnection::connect(&mount_dir, prx_dir, Default::default()).unwrap_or_else(|e| {
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

    match outcome {
        LoadOutcome::Success => eprintln!("==> Done"),
        LoadOutcome::TimedOut => {
            eprintln!("==> Timed out: app_main exceeded its budget and was terminated");
            std::process::exit(1);
        }
        LoadOutcome::Panicked => {
            eprintln!("==> Program panicked");
            std::process::exit(1);
        }
        LoadOutcome::ShellError(v) => {
            eprintln!("==> Shell error loading PRX: 0x{v:08X}");
            std::process::exit(1);
        }
        LoadOutcome::KernelError(v) => {
            eprintln!("==> Kernel error: 0x{v:08X}");
            std::process::exit(1);
        }
    }

    conn.disconnect();

    verify_against_golden(&mount_dir);
}

/// Compare the device's raw scores against the TFLite golden run.
///
/// Pass criteria: identical top-1 class, and top-3 classes present in the
/// golden top-5 (the fake-quant pipeline is expected to differ by ~1 output
/// quantum, so exact tie order deep in the tail is not meaningful).
fn verify_against_golden(mount_dir: &Path) {
    let results = match std::fs::read_to_string(mount_dir.join("results.txt")) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("==> Warning: no results.txt from device: {e}");
            return;
        }
    };
    let device: Vec<f32> = results
        .lines()
        .map(|l| l.trim().parse::<f32>().unwrap_or(f32::NAN))
        .collect();

    let golden_path = mount_dir.join("golden.json");
    let golden_raw = match std::fs::read_to_string(&golden_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("==> Warning: no golden.json ({e}); run models/birdnet_reference.py");
            return;
        }
    };
    // Crude but dependency-free: pull the "output_raw" float array.
    let arr = golden_raw
        .split("\"output_raw\":")
        .nth(1)
        .and_then(|s| s.split('[').nth(1))
        .and_then(|s| s.split(']').next())
        .expect("golden.json missing output_raw");
    let golden: Vec<f32> = arr
        .split(',')
        .map(|v| v.trim().parse::<f32>().unwrap())
        .collect();

    // With a pruned classifier (TOPK) the device emits one score per surviving
    // species, while golden.json still holds all 6522. Project golden onto the
    // kept classes so the check stays meaningful instead of being skipped.
    // Reported indices are then pruned-model indices, matching the device.
    let golden = match std::fs::read_to_string(mount_dir.join("kept_indices.txt")) {
        Ok(s) => {
            let kept: Vec<usize> = s
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().parse::<usize>().expect("malformed kept_indices.txt"))
                .collect();
            println!(
                "    pruned model: projecting golden onto {} kept classes",
                kept.len()
            );
            kept.iter().map(|&i| golden[i]).collect::<Vec<f32>>()
        }
        Err(_) => golden,
    };

    assert_eq!(device.len(), golden.len(), "score count mismatch");

    let top = |v: &[f32], k: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_unstable_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.truncate(k);
        idx
    };
    let d_top = top(&device, 5);
    let g_top = top(&golden, 5);

    let max_diff = device
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\n==> Verification vs TFLite golden:");
    println!("    max |Δraw| = {max_diff:.4} (output quantum ≈ 0.1656)");
    println!("    device top-5: {d_top:?}");
    println!("    golden top-5: {g_top:?}");

    let top1_ok = d_top[0] == g_top[0];
    let top3_ok = d_top[..3].iter().all(|i| g_top.contains(i));
    if top1_ok && top3_ok {
        println!("    PASS: top-1 exact, device top-3 ⊆ golden top-5");
    } else {
        println!("    FAIL: top-1 match={top1_ok}, top-3⊆top-5={top3_ok}");
        std::process::exit(1);
    }
}
