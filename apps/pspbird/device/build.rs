//! Build pipeline for the birdnet device crate. Each step feeds the next:
//!
//!   1. pick the model      full .tflite, or a TOPK-pruned copy in OUT_DIR
//!   2. psp-tc compile      -> generated.rs + weights.bin
//!   3. emit classes.rs     -> OUTPUT_CLASSES, so main.rs matches the model
//!   4. embed the audio     cardinal_3s.wav -> audio_f32.bin
//!   5. stage for deploy    weights.bin next to the .prx
//!
//! Steps 1 and 3 exist only because pruning changes the model's output width;
//! everything downstream (labels, class count, blob) is derived from whichever
//! model step 1 selected, so the three can never disagree.
//!
//! Knobs: TOPK, BIRDNET_REGION, BIRDNET_BBOX, BIRDNET_PYTHON,
//! BIRDNET_GENERATED_OVERRIDE. See examples/birdnet/README.md.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Overridable via BIRDNET_MODEL (path relative to the repo root). The
/// default is the Zenodo FP32 build: no QUANTIZE/DEQUANTIZE ops, f32
/// weights, so none of the fake-quant tax (measured 4296 vs 5160 ms at
/// TOPK=500). `audio-model-int8.tflite` is the same graph with int8 weights
/// and is the only one that fits *unpruned* (40.4 vs 51.7 MB) — pass it
/// with TOPK=0. The pruner and the frontend severing are dtype-agnostic.
const FULL_MODEL: &str = "models/birdnet/audio-model-fp32.tflite";
/// Default pruning. `TOPK=0` disables it (the stock 6522-class model).
const DEFAULT_TOPK: u32 = 500;
const FULL_LABELS: &str = "models/birdnet/labels/en_us.txt";
const PRUNER: &str = "apps/pspbird/prune_classifier.py";
const SLICER: &str = "examples/birdnet-stft-benchmark/slice_stft.py";
const STFT_ASSETS: &str = "models/birdnet/stft";
const INPUT_SAMPLES: usize = 144_000;
const N_WINDOWS: usize = 511;
const N_BANKS: usize = 96;
/// (fft_length, fmin, fmax) — the reverse-engineered filterbank ranges,
/// verified against the stored matrices by psp-tc's mel tests.
const BRANCHES: [(usize, f64, f64); 2] = [(2048, 0.0, 3000.0), (1024, 500.0, 15000.0)];
const SAMPLING_RATE: f64 = 48_000.0;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let repo_root = manifest_dir.join("../../..");

    for var in [
        "TOPK",
        "BIRDNET_REGION",
        "BIRDNET_BBOX",
        "BIRDNET_PYTHON",
        "BIRDNET_GENERATED_OVERRIDE",
        "PSP_TC_FORCE_VME",
        "BIRDNET_MODEL",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed={}", repo_root.join(FULL_MODEL).display());
    println!("cargo:rerun-if-changed={}", repo_root.join(FULL_LABELS).display());
    println!("cargo:rerun-if-changed={}", repo_root.join(PRUNER).display());
    println!("cargo:rerun-if-changed={}", repo_root.join(SLICER).display());
    println!("cargo:rerun-if-changed=../cardinal_3s.wav");

    // A library-only build for `imfile` (pspbird-imview's device crate)
    // needs no generated code and no model. Only skip codegen when the
    // binaries cannot be compiled at all (`local` off, device-only lib):
    // in a workspace build, feature unification hands the `birdnet` bin
    // target `imfile` too (pspbird-host enables imfile-pack), and its
    // include! of generated.rs must still be satisfiable.
    if std::env::var("CARGO_FEATURE_IMFILE").is_ok()
        && std::env::var("CARGO_FEATURE_APP").is_err()
        && std::env::var("CARGO_FEATURE_LOCAL").is_err()
    {
        return;
    }

    // The app (`pspbird` binary, `app` feature) is a different compile: a
    // headless backbone plus runtime-loaded classifier blobs. Nothing the
    // benchmark binary includes is built for it.
    if std::env::var("CARGO_FEATURE_APP").is_ok() {
        build_app(&repo_root, &out_dir);
        return;
    }

    // 1. Model selection. Also writes labels.txt (and, when pruned,
    //    kept_indices.txt) into OUT_DIR alongside it.
    let model = select_model(&repo_root, &out_dir);

    // 2. Codegen. Produces generated.rs and weights.bin in OUT_DIR — from
    //    the whole model normally, or (custom frontend) from the backbone
    //    severed at the branch-merge CONCAT, plus a builder-generated
    //    frontend module in OUT_DIR/frontend/.
    if std::env::var("CARGO_FEATURE_CUSTOM_FRONTEND").is_ok() {
        // BIRDNET_SMALL_FFT=1: additionally apply the FFT-pruning pass.
        println!("cargo:rerun-if-env-changed=BIRDNET_SMALL_FFT");
        let small_fft = std::env::var("BIRDNET_SMALL_FFT").is_ok_and(|v| v != "0");
        build_custom_frontend(&repo_root, &out_dir, &model, false, small_fft);
    } else {
        psp_tc::compile_tflite(&model, &out_dir, None)
            .unwrap_or_else(|e| panic!("psp-tc codegen failed: {e}"));
    }
    apply_generated_override(&out_dir);

    // 3. Class count, read back from the labels the selected model implies.
    let num_classes = read_lines(&out_dir.join("labels.txt")).len();
    write(
        &out_dir.join("classes.rs"),
        format!("pub const OUTPUT_CLASSES: usize = {num_classes};\n").as_bytes(),
    );
    println!("cargo:warning=birdnet: {num_classes} output classes");

    // 4. Model input, pre-converted so the device does not need a WAV parser.
    embed_audio(&manifest_dir, &out_dir);

    // 5. Hand the blob to the host runner.
    stage_for_deploy(&out_dir);
}

/// Returns the .tflite psp-tc should compile, and writes the labels that go
/// with it (plus `kept_indices.txt` when pruned) into `out_dir`.
///
/// TOPK=0 means the stock 6522-class model and the full labels file.
fn select_model(repo_root: &Path, out_dir: &Path) -> PathBuf {
    let model_rel = std::env::var("BIRDNET_MODEL").unwrap_or_else(|_| FULL_MODEL.to_string());
    let full_model = repo_root.join(&model_rel);
    assert!(
        full_model.exists(),
        "BIRDNET_MODEL {} not found",
        full_model.display()
    );
    let full_labels = repo_root.join(FULL_LABELS);

    let topk = topk_from_env();
    if topk == 0 {
        // Unpruned: labels are the full list, and no index map -- deleting
        // it matters, or a stale map from a previous pruned build would be
        // applied to full-width output.
        std::fs::copy(&full_labels, out_dir.join("labels.txt"))
            .expect("failed to read models/birdnet/labels/en_us.txt");
        let _ = std::fs::remove_file(out_dir.join("kept_indices.txt"));
        return full_model;
    }

    let pruned = out_dir.join("audio-model-pruned.tflite");
    let mut cmd = pruner_command(&repo_root.join(PRUNER));
    cmd.arg(&full_model)
        .arg(&full_labels)
        .arg("--top-n")
        .arg(topk.to_string())
        .arg("-o")
        .arg(&pruned)
        .arg("--write-labels")
        .arg(out_dir.join("labels.txt"))
        .arg("--write-indices")
        .arg(out_dir.join("kept_indices.txt"));
    // An explicit bbox wins over the named region.
    match std::env::var("BIRDNET_BBOX") {
        Ok(s) if !s.trim().is_empty() => {
            let corners: Vec<&str> = s.split([',', ' ']).filter(|p| !p.is_empty()).collect();
            assert_eq!(
                corners.len(),
                4,
                "BIRDNET_BBOX must be 'lat0,lat1,lon0,lon1', got {s:?}"
            );
            cmd.arg("--bbox").args(corners);
        }
        _ => {
            let region = std::env::var("BIRDNET_REGION").unwrap_or_else(|_| "eastern-na".into());
            cmd.arg("--region").arg(region);
        }
    }

    let status = cmd
        .current_dir(repo_root)
        .status()
        .unwrap_or_else(|e| panic!("failed to launch the pruner ({e}). {PYTHON_HELP}"));
    assert!(status.success(), "prune_classifier.py failed. {PYTHON_HELP}");
    pruned
}

/// TOPK, defaulting to `DEFAULT_TOPK`; 0 means no pruning.
fn topk_from_env() -> u32 {
    match std::env::var("TOPK") {
        Ok(v) if !v.trim().is_empty() => v
            .trim()
            .parse()
            .unwrap_or_else(|e| panic!("TOPK must be an integer (0 = unpruned), got {v:?} ({e})")),
        _ => DEFAULT_TOPK,
    }
}

const PYTHON_HELP: &str = "Install uv, or set BIRDNET_PYTHON to a python that \
                           already has the pruner's deps (ai-edge-litert or \
                           tensorflow, plus numpy).";

/// `uv run` resolves the pruner's PEP 723 inline deps, matching how the
/// reference scripts in models/ are run. BIRDNET_PYTHON bypasses uv with an
/// interpreter that already has them.
fn pruner_command(script: &Path) -> Command {
    match std::env::var("BIRDNET_PYTHON") {
        Ok(py) if !py.trim().is_empty() => {
            let mut c = Command::new(py);
            c.arg(script);
            c
        }
        _ => {
            let mut c = Command::new("uv");
            c.arg("run").arg(script);
            c
        }
    }
}

/// The custom-frontend build: sever the (possibly pruned) model at the two
/// branches' 4D CONCAT and compile only the conv backbone from it (its input
/// is the assembled `[1, 96, 511, 2]` spectrogram pair, which main.rs builds
/// from the frontend's bank-major outputs); then generate the frontend
/// module — normalisation, strided-view STFTs, banded mel projections —
/// with the builder.
///
/// `headless` also cuts the classifier off: the backbone then ends at the
/// pooled `[1, 1024]` embedding (the app supplies its own classifier).
/// `small_fft` applies the FFT-pruning pass — shrink each branch's FFT to
/// the columns its mel banks read (anti-alias filtered, same bin grid);
/// prunes the L=2048 branch to a 512-point transform, leaves L=1024 alone.
fn build_custom_frontend(
    repo_root: &Path,
    out_dir: &Path,
    model: &Path,
    headless: bool,
    small_fft: bool,
) {
    let assets = repo_root.join(STFT_ASSETS);
    let needed = ["window_2048.bin", "window_1024.bin", "manifest.json"];
    if !needed.iter().all(|f| assets.join(f).exists()) {
        let status = pruner_command(&repo_root.join(SLICER))
            .current_dir(repo_root)
            .status()
            .unwrap_or_else(|e| panic!("failed to run the slicer ({e}). {PYTHON_HELP}"));
        assert!(status.success(), "slice_stft.py failed. {PYTHON_HELP}");
    }

    // Backbone: severed from whichever model select_model() picked, so TOPK
    // pruning composes with the custom frontend.
    let backbone = out_dir.join("backbone.tflite");
    let mut cmd = pruner_command(&repo_root.join(SLICER));
    cmd.arg(if headless { "sever-backbone-headless" } else { "sever-backbone" })
        .arg(model)
        .arg(&backbone);
    let status = cmd
        .current_dir(repo_root)
        .status()
        .unwrap_or_else(|e| panic!("failed to run the slicer ({e}). {PYTHON_HELP}"));
    assert!(status.success(), "sever-backbone failed. {PYTHON_HELP}");
    psp_tc::compile_tflite(&backbone, out_dir, None)
        .unwrap_or_else(|e| panic!("psp-tc codegen (backbone) failed: {e}"));

    // Frontend module: raw signal -> normalize -> 2x (strided STFT -> mel).
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(assets.join("manifest.json")).unwrap(),
    )
    .unwrap();
    let pow_exponent = |fft_length: usize| -> f32 {
        manifest["branches"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["fft_length"].as_u64() == Some(fft_length as u64))
            .unwrap_or_else(|| panic!("no manifest entry for L={fft_length}"))["pow_exponent"]
            .as_f64()
            .unwrap() as f32
    };
    let branches: Vec<psp_tc::mel::FrontendBranch> = BRANCHES
        .iter()
        .map(|&(l, fmin, fmax)| {
            let bytes = std::fs::read(assets.join(format!("window_{l}.bin"))).unwrap();
            psp_tc::mel::FrontendBranch {
                fft_length: l,
                window: bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                fmin,
                fmax,
                pow_exponent: pow_exponent(l),
            }
        })
        .collect();

    let mut b = psp_tc::PspModelBuilder::new();
    let raw = b.input(vec![1, INPUT_SAMPLES]);
    let norm = psp_tc::mel::birdnet_normalize(&mut b, raw);
    let outs = if small_fft {
        println!("cargo:warning=birdnet: small-FFT pruning pass enabled");
        psp_tc::stft_mel_frontend_small_fft(
            &mut b,
            norm,
            N_WINDOWS,
            SAMPLING_RATE,
            N_BANKS,
            &branches,
        )
    } else {
        psp_tc::mel::stft_mel_frontend(
            &mut b,
            norm,
            N_WINDOWS,
            SAMPLING_RATE,
            N_BANKS,
            &branches,
        )
    };
    for out in outs {
        b.output(out);
    }
    let mut fe_model = b.finish();
    let fe_dir = out_dir.join("frontend");
    std::fs::create_dir_all(&fe_dir).unwrap();
    psp_tc::compile_graph(&mut fe_model, &fe_dir, "custom_frontend")
        .unwrap_or_else(|e| panic!("psp-tc codegen (custom frontend) failed: {e}"));

    println!("cargo:warning=birdnet: custom frontend enabled (backbone severed at the branch concat)");
}

/// PBRD classifier blob header (see `write_blob` in prune_classifier.py).
const BLOB_MAGIC: &[u8; 4] = b"PBRD";
const BLOB_HEADER: usize = 32;
const EMBEDDING: usize = 1024;

/// The `pspbird` app build (`app` feature). Everything lands in
/// `OUT_DIR/app/`:
///
///   frontend/custom_frontend.rs   builder frontend, small-FFT variant
///   generated.rs + weights.bin    headless backbone -> [1, 1024] embedding
///   classifier.rs                 external-weight FC [N, 1024], N frozen
///   blobs/<region>.bin            one PBRD blob per pruner region
///   regions.rs                    REGIONS table, OUTPUT_CLASSES, APP_TOPK
///
/// Same model/TOPK defaults as the benchmark (FP32, 500; the classifier
/// blobs are f32 either way). TOPK fixes the classifier width for *every*
/// region: the slot is sized once, and each blob must match it — checked
/// here at build time and again on load.
fn build_app(repo_root: &Path, out_dir: &Path) {
    let app_dir = out_dir.join("app");
    std::fs::create_dir_all(&app_dir).unwrap();
    let model_rel = std::env::var("BIRDNET_MODEL").unwrap_or_else(|_| FULL_MODEL.to_string());
    let model = repo_root.join(&model_rel);
    assert!(model.exists(), "BIRDNET_MODEL {} not found", model.display());
    let topk = topk_from_env();
    assert!(topk > 0, "the app needs a pruned classifier; TOPK=0 is the benchmark's unpruned mode");

    // 1. Frontend + headless backbone.
    build_custom_frontend(repo_root, &app_dir, &model, true, true);

    // 2. One classifier blob per region the pruner knows.
    let regions = {
        let out = pruner_command(&repo_root.join(PRUNER))
            .arg("--list-regions")
            .current_dir(repo_root)
            .output()
            .unwrap_or_else(|e| panic!("failed to launch the pruner ({e}). {PYTHON_HELP}"));
        assert!(out.status.success(), "prune_classifier.py --list-regions failed. {PYTHON_HELP}");
        String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(|l| l.trim().to_owned())
            .filter(|l| !l.is_empty())
            .collect::<Vec<String>>()
    };
    assert!(!regions.is_empty(), "the pruner listed no regions");
    let blob_dir = app_dir.join("blobs");
    std::fs::create_dir_all(&blob_dir).unwrap();
    let mut n_classes: Option<usize> = None;
    for region in &regions {
        let blob = blob_dir.join(format!("{region}.bin"));
        let status = pruner_command(&repo_root.join(PRUNER))
            .arg(&model)
            .arg(repo_root.join(FULL_LABELS))
            .arg("--top-n")
            .arg(topk.to_string())
            .arg("--region")
            .arg(region)
            .arg("--no-tflite")
            .arg("--write-blob")
            .arg(&blob)
            .current_dir(repo_root)
            .status()
            .unwrap_or_else(|e| panic!("failed to launch the pruner ({e}). {PYTHON_HELP}"));
        assert!(status.success(), "prune_classifier.py failed for {region}. {PYTHON_HELP}");
        let (n, k) = read_blob_header(&blob);
        assert_eq!(k, EMBEDDING, "{region}: blob in_features {k} != {EMBEDDING}");
        match n_classes {
            None => n_classes = Some(n),
            Some(prev) => assert_eq!(prev, n, "{region}: blob has {n} classes, others {prev}"),
        }
    }
    let n_classes = n_classes.unwrap();
    println!("cargo:warning=pspbird: {} regions x {n_classes} classes (TOPK={topk})", regions.len());

    // 3. The classifier: an FC whose weights and bias are runtime slots.
    let mut b = psp_tc::PspModelBuilder::new();
    let emb = b.input(vec![1, EMBEDDING]);
    let w = b.external_f32(vec![n_classes, EMBEDDING], "weights");
    let bias = b.external_f32(vec![n_classes], "bias");
    let logits = b.fully_connected(emb, w, Some(bias));
    b.output(logits);
    let mut cls = b.finish();
    psp_tc::compile_graph(&mut cls, &app_dir, "classifier")
        .unwrap_or_else(|e| panic!("psp-tc codegen (classifier) failed: {e}"));

    // 4. The table the app selects from.
    let mut rs = String::new();
    rs.push_str(&format!("pub const OUTPUT_CLASSES: usize = {n_classes};
"));
    rs.push_str(&format!("pub const APP_TOPK: u32 = {topk};
"));
    rs.push_str(&format!(
        "/// (display name, blob file name) per region, pruner order.
pub const REGIONS: [(&str, &str); {}] = [
",
        regions.len()
    ));
    for r in &regions {
        rs.push_str(&format!("    ({r:?}, {:?}),
", format!("{r}.bin")));
    }
    rs.push_str("];
");
    write(&app_dir.join("regions.rs"), rs.as_bytes());

    // 5. Stage the blobs — and the backbone's weights.bin, which at 24.7 MB
    //    (f32 convs, prepacked) is past psp-tc's include_bytes! threshold and
    //    is read from host0:/weights.bin by init() — beside the .prx for the
    //    host runner to publish. Same reasoning as stage_for_deploy().
    let profile = std::env::var("PROFILE").unwrap();
    let stage_dir = out_dir
        .ancestors()
        .find(|p| p.file_name().is_some_and(|n| n == profile.as_str()))
        .unwrap_or_else(|| panic!("no `{profile}` component in OUT_DIR {}", out_dir.display()));
    std::fs::copy(app_dir.join("weights.bin"), stage_dir.join("weights.bin"))
        .expect("failed to stage weights.bin beside the prx");
    let stage_blobs = stage_dir.join("blobs");
    let _ = std::fs::remove_dir_all(&stage_blobs);
    std::fs::create_dir_all(&stage_blobs).unwrap();
    for r in &regions {
        let name = format!("{r}.bin");
        std::fs::copy(blob_dir.join(&name), stage_blobs.join(&name)).expect("failed to stage blob");
    }
}

/// `(n_classes, in_features)` from a PBRD blob, validating the header.
fn read_blob_header(path: &Path) -> (usize, usize) {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut h = [0u8; BLOB_HEADER];
    std::io::Read::read_exact(&mut f, &mut h).expect("blob shorter than its header");
    assert_eq!(&h[0..4], BLOB_MAGIC, "{}: bad magic", path.display());
    let u = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap()) as usize;
    assert_eq!(u(4), 1, "{}: unsupported blob version", path.display());
    (u(8), u(12))
}

/// Prototyping hook: swap in a hand-edited `generated.rs` to measure a codegen
/// optimisation before implementing it in psp-tc. The model is still compiled,
/// so weights.bin and the blob layout stay consistent — only the emitted code
/// is replaced, and it must match that same blob.
fn apply_generated_override(out_dir: &Path) {
    if let Ok(over) = std::env::var("BIRDNET_GENERATED_OVERRIDE") {
        std::fs::copy(&over, out_dir.join("generated.rs"))
            .unwrap_or_else(|e| panic!("failed to apply {over}: {e}"));
        println!("cargo:warning=using hand-edited generated.rs from {over}");
        println!("cargo:rerun-if-changed={over}");
    }
}

/// Convert the 3 s window to f32 exactly as models/birdnet_reference.py does
/// (i16 / 32768.0), which is what makes the device's input bit-identical to the
/// Python golden run.
fn embed_audio(manifest_dir: &Path, out_dir: &Path) {
    let wav_path = manifest_dir.join("../cardinal_3s.wav");
    let wav = std::fs::read(&wav_path).unwrap_or_else(|e| {
        panic!(
            "{} not found ({e}); run `uv run models/fetch.py` first",
            wav_path.display()
        )
    });

    let pcm = &wav[44..]; // skip WAV header
    let mut floats = vec![0.0f32; INPUT_SAMPLES];
    for (i, slot) in floats.iter_mut().enumerate().take(pcm.len() / 2) {
        *slot = i16::from_le_bytes([pcm[i * 2], pcm[i * 2 + 1]]) as f32 / 32768.0;
    }

    let bytes: Vec<u8> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
    write(&out_dir.join("audio_f32.bin"), &bytes);
}

/// Stage weights.bin (and the index map) beside the .prx this build produces.
///
/// Deliberately *not* copied into the host0: mount here. Several build trees
/// compile this crate — the workspace `cargo build -p birdnet` and cargo-psp's
/// own `--target-dir` — each with its own OUT_DIR and its own blob. If they all
/// wrote to the shared mount, the last one to run would win and could leave a
/// blob that does not match the deployed .prx (an unpruned 42.8 MB blob against
/// a TOPK .prx expecting 17.8 MB). `init()` reads exactly WEIGHT_BYTES without
/// validating the file, so that mismatch is silent: the run computes on wrong
/// offsets and produces garbage. The host runner copies from here at deploy
/// time instead, taking blob and .prx from the same directory.
fn stage_for_deploy(out_dir: &Path) {
    // Locate the profile dir by name, not by depth: plain cargo nests OUT_DIR as
    // <profile>/build/<pkg>-<hash>/out, while cargo-psp splits package and hash
    // into separate components, so a fixed ancestor count fits only one of them.
    let profile = std::env::var("PROFILE").unwrap();
    let stage_dir = out_dir
        .ancestors()
        .find(|p| p.file_name().is_some_and(|n| n == profile.as_str()))
        .unwrap_or_else(|| panic!("no `{profile}` component in OUT_DIR {}", out_dir.display()));

    std::fs::copy(out_dir.join("weights.bin"), stage_dir.join("weights.bin"))
        .expect("failed to stage weights.bin beside the prx");

    let kept = out_dir.join("kept_indices.txt");
    if kept.exists() {
        std::fs::copy(&kept, stage_dir.join("kept_indices.txt"))
            .expect("failed to stage kept_indices.txt");
    } else {
        let _ = std::fs::remove_file(stage_dir.join("kept_indices.txt"));
    }
}

fn read_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
        .lines()
        .map(str::to_owned)
        .collect()
}

fn write(path: &Path, bytes: &[u8]) {
    let mut f = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
    f.write_all(bytes)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}
