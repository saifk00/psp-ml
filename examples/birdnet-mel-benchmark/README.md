# BirdNET Mel Projection Benchmark

After the STFT, BirdNET projects each branch's spectrum onto 96 mel banks as
a dense FC — `[511, 1025] @ [1025, 96]` and `[511, 513] @ [513, 96]` — then
compresses with `MUL(x,x)` + `POW(·, p)`. The mel matrices are triangles in
mel space and are **0.3% / 1.2% nonzero** (one contiguous band of 1–17 bins
per bank), so >99% of the dense MACs multiply by zero.

`banded_cb` replaces that with:

- `psp_tc::make_mel` — regenerates the filterbank from the
  reverse-engineered parameters (HTK mel `2595·log10(1+f/700)`;
  `fmin,fmax = 0,3000` for L=2048 and `500,15000` for L=1024; `B+2` points at
  `Δm = (mel(fmax)−mel(fmin))/(B+1)`), stored as a `CBMatrix` — one
  `[start, len]` band per bank. Verified against the stored matrices: exact
  sparsity pattern, values within 5e-5 (psp-tc test
  `make_mel_matches_birdnet_stored_matrices`).
- `PspOp::FullyConnectedCB` → `psp_rt::kernels::fully_connected_cb`. **The
  VFPU kernel is deliberately not designed yet** — the current body is the
  scalar reference on both targets, pinned by a device check to produce
  results identical to the dense matmul. The numbers below are a floor.
- `PspOp::SquarePow` → `psp_rt::kernels::square_pow`, the fused `(x²)^p`
  (one extra `vmul.q` on `pow_const`'s `vlog2`/`vexp2` pipeline;
  bit-identical to `MUL` + `pow_const`, checked on hardware).

`make_mel` takes the sampling rate as a parameter because the PSP records at
44.1 kHz while the frontend assumes 48 kHz — regenerate the banks at the true
rate instead of resampling.

## Show the difference

```bash
# On hardware: one PRX per mode, verifies vs the TFLite golden, prints the table:
cargo run -p birdnet-mel-benchmark-host --release
cargo run -p birdnet-mel-benchmark-host --release -- --mode dense_fc     # or banded_cb

# No hardware (scalar kernels, same generated code, same checks):
cargo run -p birdnet-mel-benchmark --release
```

Inputs are the STFT goldens (`golden_<L>.bin`) — bit-exactly what feeds this
subgraph in the full model; goldens are TFLite's own post-POW outputs.

## Measured (real PSP, 333 MHz, 2026-08-23 — CB kernel still scalar)

| mode      | arena    | weight blob | free after load | mel 2048 | mel 1024 |
|-----------|----------|-------------|-----------------|----------|----------|
| dense_fc  | 3493 KiB | 582 KiB     | 47,419 KiB      | 227 ms   | 126 ms   |
| banded_cb | 383 KiB  | 4 KiB       | 51,112 KiB      | 23 ms    | 36 ms    |

banded_cb saves 3687 KiB at compile time; the device measured 3693 KiB more
free memory after load. Mel time drops 83% *with the scalar reference
kernel* — `gemm_vfpu`'s 152 ms + 80 ms (plus 90 ms of reshape copies) become
16 ms + 28 ms of `fc_cb`, and the fused `square_pow` (2.6 ms/branch) replaces
`binary_mul` + `pow_const` (6.5 ms/branch).

## Tolerances

dense_fc gates at 1e-3 err/frame-RMS vs the golden (lands at ~1e-5).
banded_cb gates at 5e-3 (lands at ~2.5e-3): its filterbank is *regenerated*,
and the ~2e-5 weight deltas are amplified by the `x^~0.22` compression.
Build the device crate with `MEL_CB_USE_STORED=1` to compress the stored
matrix instead — error then collapses to dense levels (~1e-5), which is how
the CB kernel itself was shown to be exact.

## Files

- `device/` — `local` (default; both modes, `--mode` flag, golden checks),
  `mode-dense`, `mode-cb`; one mode per PRX so on-device memory attributes
  cleanly. Branch inputs are embedded 16-byte-aligned and used in place.
- `host/` — builds both PRXes, deploys over psplink, parses `#melbench`
  lines, verifies outputs, prints the comparison, exits nonzero on mismatch.
- Fixtures come from `examples/birdnet-stft-benchmark/slice_stft.py`
  (auto-run when missing): the per-branch `mel_<L>.tflite` slices,
  `mel_dense_<L>.bin` stored matrices, `golden_mel_<L>.bin` outputs, and the
  per-branch pow exponents in `manifest.json`.
