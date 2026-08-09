//! Host runner for `fc-bench`. Deploys the PRX and summarises the per-variant
//! results, sorted fastest-first per shape, with the speedup over the scalar
//! baseline that codegen currently emits.

use psplink_connection::{LoadOutcome, PSPConnection};
use std::io::Write;
use std::path::Path;

struct Row {
    shape: String,
    variant: String,
    us: u64,
    mflops: u64,
    pct: u64,
    ppm: u64,
}

fn main() {
    let prx_path = Path::new(env!("PRX_PATH"));
    let prx_dir = prx_path.parent().expect("PRX_PATH has no parent directory");
    let prx_name = prx_path.file_name().unwrap().to_str().unwrap();

    let json_path = prx_dir.join("fc-bench.json");
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

    if !matches!(outcome, LoadOutcome::Success) {
        eprintln!("==> Run failed: {outcome:?}");
        std::process::exit(1);
    }

    let json = std::fs::read_to_string(&json_path).unwrap_or_else(|e| {
        eprintln!("error: device did not produce fc-bench.json: {e}");
        std::process::exit(1);
    });
    summarize(&json);
}

fn summarize(json: &str) {
    let mut rows = Vec::new();
    for obj in json.split('{').skip(2) {
        let field = |k: &str| -> Option<String> {
            let pat = format!("\"{k}\":");
            let rest = &obj[obj.find(&pat)? + pat.len()..];
            let end = rest.find([',', '}']).unwrap_or(rest.len());
            Some(rest[..end].trim().trim_matches('"').to_string())
        };
        let (Some(shape), Some(variant)) = (field("shape"), field("variant")) else {
            continue;
        };
        let num = |k: &str| field(k).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        rows.push(Row {
            shape,
            variant,
            us: num("us"),
            mflops: num("mflops"),
            pct: num("pct_ceiling"),
            ppm: num("err_ppm"),
        });
    }

    let mut shapes: Vec<&str> = Vec::new();
    for r in &rows {
        if !shapes.contains(&r.shape.as_str()) {
            shapes.push(&r.shape);
        }
    }

    println!("\n=== Summary (ceiling 1992 MFLOP/s) ===");
    let mut total_best = 0.0f64;
    let mut total_base = 0.0f64;
    for shape in shapes {
        let mut group: Vec<&Row> = rows.iter().filter(|r| r.shape == shape).collect();
        let baseline = group
            .iter()
            .find(|r| r.variant.starts_with("v0"))
            .map(|r| r.us)
            .unwrap_or(0);
        group.sort_by_key(|r| r.us);
        println!("\n{shape}:");
        for r in &group {
            let speedup = if r.us > 0 {
                baseline as f64 / r.us as f64
            } else {
                0.0
            };
            let flag = if r.ppm > 1000 { "  <-- WRONG" } else { "" };
            println!(
                "  {:<22} {:>8.1} ms  {:>6} MFLOP/s  {:>3}% ceil  {:>6.1}x  err {:>5} ppm{}",
                r.variant,
                r.us as f64 / 1000.0,
                r.mflops,
                r.pct,
                speedup,
                r.ppm,
                flag
            );
        }
        if let Some(best) = group.iter().find(|r| r.ppm <= 1000 && !r.variant.starts_with("v0")) {
            total_best += best.us as f64;
            total_base += baseline as f64;
        }
    }
    if total_base > 0.0 {
        println!(
            "\nBoth mel FCs: {:.0} ms -> {:.0} ms ({:.1}x). \
             BirdNET total would go 19967 ms -> {:.0} ms.",
            total_base / 1000.0,
            total_best / 1000.0,
            total_base / total_best,
            19967.0 - (total_base - total_best) / 1000.0
        );
    }
}
