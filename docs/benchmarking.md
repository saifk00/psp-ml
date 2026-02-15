# Benchmarking

## Overview

The benchmarking system has two parts:

1. **PSP-side** (`psp_ml::bench`) — format and write self-describing JSON
2. **Host-side** (`parse_bench.sh`) — enrich with timestamp/notes, append to history

The compiler generates `forward_timed()`, `OP_NAMES`, and `NUM_OPS`. The user
writes their own benchmark loop — it's ~20 lines. Host-side post-processing is
a separate concern handled by whatever tool the user wants (we include a shell
script).

## Generated code

`cargo psp-ml compile` produces `generated.rs` with:

```rust
pub fn forward(input: &[f32; 784]) -> [f32; 10];

pub fn forward_timed(
    input: &[f32; 784],
    op_ticks: &mut [u64; NUM_OPS],
    get_tick: fn() -> u64,
) -> [f32; 10];

pub const NUM_OPS: usize = 11;
pub const OP_NAMES: [&str; NUM_OPS] = ["im2col", "matmul", ...];
```

`forward_timed` accumulates (`+=`) per-op tick deltas into the caller's array.
The `get_tick` parameter lets the caller provide the clock source — on PSP
it wraps `sceRtcGetCurrentTick`, on host it wraps `std::time::Instant`.

## `psp_ml::bench`

Three building blocks:

- **`BenchmarkResult`** — struct with `id`, timing, accuracy, `op_names`,
  `op_ticks`. Everything needed for a self-describing JSON record.
- **`format_results(result) -> JsonBuf`** — pure JSON formatting into a 4KB
  stack buffer. No heap, works in `no_std`.
- **`write_hostfs(path, data)`** — write bytes via psplink HostFS. No-op on
  non-PSP targets.
- **`argmax(output) -> usize`** — index of maximum element.

## Example usage (PSP)

```rust
let (total_ticks, correct, op_ticks) = run_benchmark(get_tick);

let result = BenchmarkResult {
    id: "vfpu",
    num_samples: 100,
    tick_resolution: tick_res,
    total_ticks,
    correct,
    op_names: &generated::OP_NAMES,
    op_ticks: &op_ticks,
};

let json = bench::format_results(&result);
bench::write_hostfs("host0:/benchmarks.json", json.as_bytes());
```

## JSON output

```json
{
  "id": "vfpu",
  "num_samples": 100,
  "tick_resolution": 1000000,
  "total_ticks": 12345678,
  "total_us": 12345678,
  "per_sample_us": 123456,
  "correct": 97,
  "ops": [
    { "name": "im2col", "ticks": 1234567, "us": 1234567 },
    { "name": "matmul", "ticks": 2345678, "us": 2345678 },
    ...
  ]
}
```

Self-describing: op names, timing, and accuracy are all in the output.
Host tools need no knowledge of the compiler or model.

## Host-side workflow

```bash
# Deploy and run on PSP
cargo psp-ml run -p mnist-bench --release

# Enrich and record
cd examples/mnist-bench
./parse_bench.sh /path/to/benchmarks.json "VFPU conv2d"

# Result: data/benchmarks.jsonl gets a new line:
# {"id":"vfpu","num_samples":100,...,"_timestamp":"2025-07-12T14:30:00Z","_notes":"VFPU conv2d"}
```

The JSONL history file is append-only, one JSON object per line. Each entry
is the original PSP output plus `_timestamp` and `_notes`.

## Design decisions

**No generated `benchmark()` function.** The benchmark loop is ~20 lines and
fully transparent in user code. Generating it saves nothing and hides what's
happening.

**Host-side is a shell script, not Rust.** It's just `jq` enrichment + append.
If users want fancier analysis (diffing, charting), they write their own tool
in whatever language they prefer. We don't prescribe.

**`write_hostfs` is separate from `format_results`.** Formatting is pure and
works everywhere. I/O is platform-specific. Keeping them separate lets the
host `local` build use `std::fs::write` while PSP uses HostFS.
