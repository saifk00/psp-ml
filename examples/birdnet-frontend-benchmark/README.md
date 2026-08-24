# BirdNET Whole-Frontend Benchmark

The STFT and mel pieces tied together: BirdNET's spectrogram pipeline from
the normalised signal to both branches' compressed mel spectrograms.

- `dense_tflite` — `models/birdnet/stft/frontend_mel.tflite` (the frontend
  sliced whole out of the real model by `slice_stft.py`): dense index
  gathers, materialised `[511, L]` window matrices, dense mel FCs, MUL+POW.
- `custom_ops` — `psp_tc::stft_mel_frontend`: per branch a `StridedViewStft`
  (windows read straight out of the signal, hann folded into the FFT pack)
  into `FullyConnectedCB` (the `make_mel`-regenerated banded filterbank,
  vtfm4-tiled GEMVs) plus the fused `SquarePow`. Outputs are `[96, 511]`
  bank-major — the CB kernel's transposed layout, which is also what the
  full model's downstream TRANSPOSE wants.

## Show the difference

```bash
cargo run -p birdnet-frontend-benchmark-host --release     # hardware, both modes
cargo run -p birdnet-frontend-benchmark-host --release -- --mode dense_tflite
cargo run -p birdnet-frontend-benchmark --release          # no hardware
```

## Measured (real PSP, 333 MHz, 2026-08-23)

| mode         | arena    | weight blob | free after load | frontend |
|--------------|----------|-------------|-----------------|----------|
| dense_tflite | 9301 KiB | 2917 KiB    | 42,779 KiB      | 1459 ms  |
| custom_ops   | 2245 KiB | 40 KiB      | 52,956 KiB      | 826 ms   |

custom_ops saves 9932 KiB at compile time (arena+blob); the device measured
9938 KiB more free memory after load. Frontend time −43%. Both modes gate
at 1e-2 err/frame-RMS vs the TFLite goldens (measured: dense 5.3e-3, custom
4.4e-3 — FFT last-bit differences vs TFLite's FFT amplified by the
`x^~0.17–0.22` compression on near-zero mel values; the custom mode adds the
regenerated filterbank's ~2e-5 weight deltas).

## The real payoff: `BIRDNET_CUSTOM_FRONTEND=1`

The same frontend graph (plus the model's min-max normalisation, replicated
op-for-op by `psp_tc::mel::birdnet_normalize`) can replace the frontend in
the actual birdnet example — `build.rs` severs the .tflite at the two
branches' merge CONCAT and compiles only the conv backbone, and `main.rs`
runs the two forwards in sequence with a reverse+interleave shuffle between
them (the model's REVERSE_V2 + TRANSPOSE + CONCAT, done in plain code —
folding it away is future work):

```bash
TOPK=500 cargo run -p birdnet-host --release                            # baseline: 5617 ms
BIRDNET_CUSTOM_FRONTEND=1 TOPK=500 cargo run -p birdnet-host --release  # 4865 ms (-13.4%)
```

Measured 2026-08-23, cardinal fixture: total 5617 → 4865 ms, frontend
(incl. assembly) 891 ms, golden verification PASS — top-1 exact and the
custom run's top-5 matches the golden top-5 exactly.
