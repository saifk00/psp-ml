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
- `small_fft` — `custom_ops` plus the FFT-pruning pass
  (`psp_tc::plan_small_fft`): the mel banks read only bins 0–127 of the
  L=2048 branch's 1025, so that branch's needed columns round up to a
  power-of-two-plus-one (128 → 129 → N=256; a Nyquist guard doubles it to
  512 because at N=256 the top bank sits at 99% of the pruned Nyquist and no
  anti-alias filter could protect it). The signal is Kaiser-lowpassed
  (60 dB, transition 2977→9023 Hz — everything the transition band aliases
  lands in bins mel never reads) and stored decimated by 2 (hop 278 → 139
  stays integer); the STFT reads it at inner stride 2 with the window
  subsampled and scaled by the decimation, so the same 23.44 Hz bin grid
  comes out of a 4× smaller transform. The L=1024 branch needs 320 of its
  513 columns, rounds back to 513 — unchanged.

## Show the difference

```bash
cargo run -p birdnet-frontend-benchmark-host --release     # hardware, all three modes
cargo run -p birdnet-frontend-benchmark-host --release -- --mode small_fft
cargo run -p birdnet-frontend-benchmark --release          # no hardware
```

## Measured (real PSP, 333 MHz, 2026-08-26)

| mode         | arena    | weight blob | free after load | frontend | worst err (2048/1024) |
|--------------|----------|-------------|-----------------|----------|-----------------------|
| dense_tflite | 9301 KiB | 2917 KiB    | 41,776 KiB      | 1475 ms  | 4.2e-5 / 5.3e-3       |
| custom_ops   | 2245 KiB | 40 KiB      | 51,715 KiB      | 846 ms   | 2.5e-3 / 4.4e-3       |
| small_fft    | 1820 KiB | 22 KiB      | 52,158 KiB      | 603 ms   | 2.3e-3 / 4.4e-3       |

small_fft: −59% vs dense, −29% vs custom_ops; the pruned branch's STFT fell
481 → 95 ms and its accuracy *improved* slightly (the 60 dB filter is
effectively transparent below 3 kHz and the smaller transform accumulates
less rounding). The 168 ms scalar `fir_decimate` is the new visible cost —
a VFPU FIR is the obvious next trim. Errors are max err/frame-RMS vs the
TFLite goldens (gates: 1e-2; 3e-2 for small_fft's pruned branch): FFT
last-bit differences vs TFLite's FFT amplified by the `x^~0.17–0.22`
compression on near-zero mel values, plus the regenerated filterbank's
~2e-5 weight deltas in the custom modes.

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
BIRDNET_CUSTOM_FRONTEND=1 BIRDNET_SMALL_FFT=1 TOPK=500 \
  cargo run -p birdnet-host --release                                   # 4668 ms (-16.9%)
```

Measured on the cardinal fixture: dense 5617 ms → custom 4865 ms → custom +
small-FFT 4668 ms (frontend incl. assembly 891 → 692 ms). All golden
verifications PASS with the identical max |Δraw| = 0.6622 — the frontend
rework contributes no measurable classifier error.
