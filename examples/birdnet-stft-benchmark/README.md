# BirdNET STFT Frontend Benchmark

BirdNET's spectrogram frontend is a 2-branch STFT (`L=2048, hop=278` and
`L=1024, hop=280`, 511 windows each over the 144000-sample signal). TFLite
expresses the windowing as a dense gather: a constant `[511, L/g]` index
matrix (g = gcd(hop, L), the "GCD trick" already in the graph) and a
materialised `[511, L]` window matrix per branch — >1.5 M floats of
duplicated signal from a 144 K input, plus 2.3 MiB of index constants.
That gather chain is what sets the *whole model's* arena watermark.

`PspOp::StridedViewStft` replaces it: every window is a strided view
`signal[f*hop .. f*hop+L]` read directly by the FFT, with the hann window
multiplied in during the bit-reversal pack. Same f32 operations in the same
order, so the outputs are **bit-identical** to the dense path.

## Modes

- `dense_gather` — `models/birdnet/stft/frontend.tflite` (sliced out of the
  real BirdNET model by `slice_stft.py`) through the standard psp-tc pipeline.
- `strided_view` — the same two branches built with `psp_tc::PspModelBuilder`
  and compiled by `psp_tc::compile_graph`.

Each device PRX contains exactly one mode (its own arena, its own weight
blob), so the on-device free-memory number attributes cleanly to the mode.

## Show the difference

```bash
# On hardware (USB-connected PSP running psplink): runs both PRXes, verifies
# outputs against the TFLite golden and across modes, prints the table:
cargo run -p birdnet-stft-benchmark-host --release

# One mode at a time:
cargo run -p birdnet-stft-benchmark-host --release -- --mode dense_gather
cargo run -p birdnet-stft-benchmark-host --release -- --mode strided_view

# No hardware (scalar kernels, same generated code, same checks):
cargo run -p birdnet-stft-benchmark --release
cargo run -p birdnet-stft-benchmark --release -- --mode dense_gather
```

## Measured (real PSP, 333 MHz, 2026-08-23)

| mode         | arena     | weight blob | free after load | frontend time |
|--------------|-----------|-------------|-----------------|---------------|
| dense_gather | 9301 KiB  | 2335 KiB    | 29,646 KiB      | 1195 ms       |
| strided_view | 8 KiB     | 35 KiB      | 41,245 KiB      | 790 ms        |

strided_view saves 11,592 KiB at compile time (arena + blob); the device
measured 11,598 KiB more free partition memory after module load — the same
number, observed on hardware. Frontend time drops 33.8%. Outputs are
bit-identical between modes and match the TFLite golden at ~4e-6
err/frame-RMS.

The arena that remains in strided mode is the single 2048-float FFT scratch.
The blob that remains is the two hann windows plus FFT twiddles.

## Files

- `slice_stft.py` — slices the frontend out of
  `models/birdnet/audio-model-int8.tflite` into `models/birdnet/stft/`
  (gitignored, regenerated on demand — `device/build.rs` invokes it when the
  outputs are missing): `frontend.tflite`, the hann `window_*.bin`s, the
  normalised `samples.bin` for the cardinal fixture, and TFLite-run
  `golden_*.bin`s. The slice's input is the *normalised* signal so every
  consumer sees bit-identical input bytes, and the script asserts the sliced
  model reproduces the full model's taps exactly before writing anything.
- `device/` — one crate, three personalities: `local` (default; both modes,
  `--mode` flag, golden + cross-mode checks), `mode-dense`, `mode-strided`.
- `host/` — builds both PRXes (one `cargo psp` per mode, separate target
  dirs), deploys over psplink, parses each run's `#stftbench` line, verifies
  outputs, prints the comparison, exits nonzero on mismatch.
