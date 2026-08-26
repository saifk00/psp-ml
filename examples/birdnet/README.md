# birdnet

BirdNET v2.4 (int8) running on the PSP: 3 s of 48 kHz audio in, one logit per
species out. Optionally pruned to the species that can actually occur where the
device is deployed, which is what takes it from 41 MB to 16 MB.

```bash
TOPK=400 cargo run -p birdnet-host --release
```

## Files

| path | what it is |
|---|---|
| `device/` | the PSP crate. `build.rs` runs the pipeline below; `src/main.rs` is glue around the generated inference — top-k, benchmark JSON, PSP entry point. |
| `host/` | the runner. `build.rs` cross-compiles `device/` via `cargo psp`; `src/main.rs` deploys, runs, and verifies against the golden. |
| `prune_classifier.py` | the pruner. Lives here, not in `models/`, because only this example uses it — `models/` holds runners shared across examples. |
| `cardinal_3s.wav` | the 3 s input, cut from a longer recording by `models/birdnet_reference.py`. Committed so the build is reproducible. |
| `golden.json` | TFLite's answer for that input. What `host/` checks against. |
| `AUDIO_CREDITS.txt` | provenance and licence for the audio. Required: `cardinal_3s.wav` is CC BY-NC-SA. |
| `weights.bin` | **generated.** Written at deploy time by `host/`. Not in git. |
| `kept_indices.txt` | **generated.** Present only when pruned. Not in git. |

Model assets (`audio-model-int8.tflite`, `meta-model.tflite`, `labels/`) live in
`models/birdnet/` and are gitignored — downloaded separately, shared with the
reference runners.

## Pipeline

```
models/birdnet/audio-model-int8.tflite   41 MB, 6522 classes
        |
        |  (1) prune_classifier.py, only when TOPK is set
        |      meta-model.tflite: (lat,lon,week) -> per-species likelihood
        v
OUT_DIR/audio-model-pruned.tflite        16 MB, 413 classes
        |  + labels.txt, kept_indices.txt
        |
        |  (2) psp_tc::compile_tflite
        v
OUT_DIR/generated.rs + weights.bin
        |
        |  (3) classes.rs  <- OUTPUT_CLASSES, so main.rs matches the model
        |  (4) audio_f32.bin <- cardinal_3s.wav as f32
        |
        |  (5) staged beside birdnet.prx
        v
host/src/main.rs  copies weights.bin -> host0:  at deploy time
        |
        v
PSP: reads host0:/weights.bin, runs, writes host0:/results.txt
        |
        v
host/src/main.rs  compares results.txt against golden.json
```

Two details in there are load-bearing:

**Why the class count is generated (step 3).** Pruning changes the model's
output width. `OUTPUT_CLASSES` and the labels are both derived from whichever
model step 1 selected, so they cannot drift from it.

**Why the host publishes `weights.bin`, not the build (step 5).** Several build
trees compile the device crate — the workspace `cargo build -p birdnet` and
cargo-psp's own `--target-dir` — each with its own `OUT_DIR` and its own blob.
When they all wrote into the shared `host0:` mount, the last one to run won, and
could leave a blob that did not match the deployed `.prx`. The device's
`init()` reads exactly `WEIGHT_BYTES` without checking the file, so a mismatch
is silent: it runs on wrong offsets and returns garbage. Staging beside the
`.prx` and copying at deploy time makes blob and code come from one place.

## Knobs

| env var | default | effect |
|---|---|---|
| `TOPK` | unset (no pruning) | keep the N most likely species. `TOPK=400` → 413 classes (400 + 13 non-bird), 17.8 MB blob. |
| `BIRDNET_REGION` | `eastern-na` | named bounding box: `eastern-na`, `eastern-na-wide`, `western-na`, `north-america`, `europe`. |
| `BIRDNET_BBOX` | unset | `"lat0,lat1,lon0,lon1"`, overrides `BIRDNET_REGION`. |
| `BIRDNET_PYTHON` | unset (uses `uv run`) | interpreter for the pruner, when uv can't resolve its deps. |
| `BIRDNET_WAV` | the XC468176 fixture | input for `models/birdnet_reference.py` when regenerating the golden. That script lives under gitignored `models/`, so this knob is not carried by this branch. |
| `BIRDNET_GENERATED_OVERRIDE` | unset | hand-edited `generated.rs`, for prototyping codegen changes. |
| `PSP_PROFILE` | unset | build with hardware counters (`hwprofile`). |
| `BIRDNET_TAP` | unset | local runs dump every op's output tensor to `device/tap/`. Not supported with the custom frontend. |
| `BIRDNET_CUSTOM_FRONTEND` | unset | replace the model's spectrogram frontend with the custom-op one (`StridedViewStft` + banded mel): `build.rs` severs the .tflite at the branch-merge CONCAT and compiles only the conv backbone, plus a builder-generated frontend module; `main.rs` runs the two forwards in sequence. Composes with `TOPK`. Measured (TOPK=500, cardinal fixture): 5617 → 4865 ms (−13.4%), golden PASS. |
| `BIRDNET_SMALL_FFT` | unset | with the custom frontend, also apply the FFT-pruning pass (`psp_tc::plan_small_fft`): the L=2048 branch computes a 512-point FFT over the anti-alias-decimated signal instead of 2048 (same 23.44 Hz bins, only the 128 columns mel reads). Measured (TOPK=500): 4865 → 4668 ms, golden PASS, identical max Δraw. |

Picking `TOPK`: ask the pruner. Only ~405 species clear 0.01 likelihood
anywhere in `eastern-na`, so a much larger `TOPK` buys rows that can never fire.

```bash
uv run examples/birdnet/prune_classifier.py \
  models/birdnet/audio-model-int8.tflite models/birdnet/labels/en_us.txt --report
```

`--min-score 0.01` selects by likelihood instead of by rank, and an explicit
labels subset can be passed as a third positional argument in place of
`--top-n`.

## Why pruning is safe

The classifier is `Dense(6522, linear)` and the sigmoid is applied *outside* the
model, so each logit is an independent dot product `W[j]·x + b[j]`. Dropping
rows of `W` cannot change any surviving logit — verified bit-identical against
the unpruned model on every window of the test audio. This holds only because
the activation is sigmoid; under softmax the shared denominator would make every
surviving score change.

It buys memory, not time. The classifier is ~62% of the blob's bytes but under
1% of its MACs — a fully-connected layer uses each weight exactly once, while
the convolutions reuse theirs across the whole feature map. Expect the blob to
shrink by more than half and the runtime by tens of milliseconds.

## Regenerating the golden

```bash
uv run models/birdnet_reference.py
```

Scores every 3 s window of `models/birdnet/cardinal_xc468176.wav`, picks the
most confident, and rewrites `golden.json` + `cardinal_3s.wav`.

The fixture is a **verified** Northern Cardinal (XC468176, recorded in Central
Park, bird seen). The previous `cardinal.wav` was not: BirdNET scores it 0.0096
for Northern Cardinal and calls it Pyrrhuloxia at 0.66, while scoring three
independent ground-truth cardinal recordings 0.38–0.84. Every stage agreed with
every other stage on that file — they were faithfully reproducing a wrong
answer. A golden that only proves `device == TFLite` proves nothing about
correctness; it has to be anchored to an input whose right answer is known.
