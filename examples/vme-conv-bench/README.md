# vme-conv-bench

Hardware bench for the VME int8 1x1 conv offload (`psp_tc::vme_conv` +
`psp_rt::kernels::vme_conv1x1_i8`): verifies the array path bit-for-bit
against the scalar integer reference at BirdNET's real conv shapes, and
measures the per-job invocation cost the lowering heuristic consumes.

```bash
cargo run -p vme-conv-bench-host --release   # needs the psp_vme_kernel v1.1 plugin
```

## Measured (retail silicon, 2026-08-27)

| k   | co   | px   | jobs | total   | per job | MACs/µs | verify |
|-----|------|------|------|---------|---------|---------|--------|
| 24  | 72   | 6144 | 1314 | 3.29 s  | 2503 µs | 3       | PASS (bit-exact) |
| 36  | 288  | 1536 | 2016 | 3.48 s  | 1725 µs | 4       | PASS |
| 72  | 864  | 384  | 3024 | 3.79 s  | 1252 µs | 6       | PASS |
| 108 | 1536 | 96   | 2304 | 3.43 s  | 1488 µs | 4       | PASS |

The array itself computes a full job (2048-element stream, 4 lanes) in
~2065 cycles ≈ 12 µs at 166 MHz — the RTL and silicon agree. Everything
else is invocation: the v1.1 image-mode job restages all eight 8 KB rings
from the 1 MB image and reads them all back per run, and the CPU touches
the image uncached. That fixed ~1.25–2.5 ms per job versus a hard
`4 lanes x 2048 words = 8192` MACs-per-job ring-capacity ceiling puts the
offload at **3–6 MACs/µs against the VFPU fake-quant path's 217** — the
break-even needs ~400 K MACs per job, 50x the capacity.

So the heuristic (`psp_tc::vme_conv::vme_conv1x1_profitable`) selects the
VFPU for every real shape; `PSP_TC_FORCE_VME=1` forces the offload for
end-to-end validation (a full birdnet run with all 33 int8 1x1 convs on
the Media Engine passes the golden gate — top-5 exactly — in 87.9 s, of
which 84.6 s is `vme_conv`).

What would flip it: a plugin job mode that stages only deltas (weights
persist ring-side across jobs; activations/results move as 8 KB, not
128 KB) would cut the fixed cost toward the ~12 µs compute floor — parity
needs ~19 µs/job at current capacity. A two-level AGU loop (or packed int8
samples) would raise MACs/job instead. Until one of those exists, the
VFPU wins every conv this model has.

Also documented here: a k = 8 probe job never signalled completion (the
plugin's run timeout fired). k >= 24 is the silicon-verified floor
(`vme_conv::MIN_K`); the boundary between is unmapped.
