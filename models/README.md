# models

Scripts for fetching model assets and verifying device output against TFLite.
The assets themselves land in `models/birdnet/` (gitignored).

```bash
uv run models/fetch.py            # download everything below (idempotent; --check, --force)
```

| asset | size | source |
|---|---|---|
| `birdnet/audio-model-fp32.tflite` | 51.7 MB | Zenodo 15050749, `BirdNET_v2.4_tflite.zip` |
| `birdnet/audio-model-int8.tflite` | 41.1 MB | Zenodo 15050749, `BirdNET_v2.4_tflite_int8.zip` |
| `birdnet/meta-model.tflite` | 8.1 MB | species-by-location prior, same record |
| `birdnet/labels/*.txt` | | label files, all languages |
| `birdnet/cardinal_xc468176.wav` | 3.5 MB | XC468176 (xeno-canto), decoded to 48 kHz mono |

BirdNET models are CC BY-NC-SA 4.0 (Cornell Lab of Ornithology / Chemnitz
University of Technology). The recording is CC BY-NC-SA 4.0; provenance in
`apps/pspbird/AUDIO_CREDITS.txt`.

## Scripts

- `birdnet_reference.py` — scores every 3 s window of the fixture wav with
  TFLite and writes `apps/pspbird/golden.json` + `cardinal_3s.wav` (the
  committed fixtures the benchmark verifies against). `BIRDNET_MODEL` and
  `BIRDNET_WAV` override the model and input. Note: the committed fixtures
  are the reference; xeno-canto currently serves a shorter file than the one
  they were cut from, so only regenerate them deliberately.
- `compare_taps.py` — layer-by-layer diff of a `BIRDNET_TAP=1` run against
  TFLite with `experimental_preserve_all_tensors`. The first row with large
  relative error localizes the first divergent op.
- `op_breakdown.py` — groups the per-op timings a device run writes to
  `benchmarks.json`; `--each` lists the slowest individual calls.
- `run_inference.py` — MNIST reference (run from `examples/mnist-bench/`).
