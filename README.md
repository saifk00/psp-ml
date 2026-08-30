# psp-ml

Inference for the Sony PSP: a TFLite -> Rust AOT compiler, a `no_std`
device runtime with VFPU kernels, a native USB link to psplink — and PSPBird,
an app built on all three.

## PSPBird

<!-- TODO: demo video/GIF -->

Live bird-sound identification on a PSP. BirdNET v2.4, compiled to native
VFPU code, listens on the microphone and classifies every 3 seconds of audio
against a region-pruned species list (~3.9 s per window on hardware). Species
photos are drawn for the leading detections.

**[Download the latest release](https://github.com/saifk00/psp-ml/releases)** —
unzip, copy `PSPBIRD/` to `ms0:/PSP/GAME/`, launch from the XMB. Requires
custom firmware.

To build and run it from source with a USB-connected PSP running psplink:

```bash
uv run models/fetch.py                        # model weights + fixtures (one time)
cargo run -p pspbird-host --release           # build, deploy, stream the log
cargo run -p pspbird-host --release -- --pack DIR    # assemble a standalone install
cargo run -p pspbird-install-host --release   # copy it onto the Memory Stick over USB
```

The app lives in [`apps/pspbird/`](apps/pspbird/), including the benchmark
harness (`pspbird-bench-host`) that verifies device output against a TFLite
golden run.

## The toolchain

- **`toolchain/psp-tc`** — TFLite → Rust compiler. Parses the model, lowers it
  through an IR (quantization rewrite, fusion, shape inference, constant
  folding, static memory planning), and emits Rust inference code with no
  interpreter and no heap. Library API for `build.rs` use plus a CLI
  (`psp-tc compile`, `psp-tc info`). Also includes `PspModelBuilder` for
  hand-constructed graphs (PSPBird's custom STFT frontend) and runtime
  weight slots (its swappable classifier).
- **`toolchain/psp-rt`** — the `no_std` device runtime: VFPU kernels
  (GEMM, conv2d, FFT, pooling, transcendentals), the `psp_rt::module!` entry
  macro, print macros, tracked partition allocation, hardware-profiler
  bindings.
- **`toolchain/psplink-connection`** / **`toolchain/usbhostfs-sys`** — talk to
  a PSP running psplink from ordinary Rust: mount host directories, load a
  PRX, stream its stdout, get a real exit status. No `usbhostfs_pc`
  subprocess, no TCP bridge.
- **`toolchain/vme-emu`** / **`vme-emu-sys`** / **`psp-tc/vme-assembler`** —
  a Verilog model of the Media Engine's VME array (silicon-calibrated), plus
  an assembler that emits machine images for it.
- **`plugins/`** — kernel-mode C PRXes: hardware performance counters
  (`kernel-plugin`) and the VME job server (`kernel-plugin-vme`). Installed
  to the Memory Stick once; see each Makefile.

### Using the compiler

```rust
// device/build.rs
psp_tc::compile_tflite(Path::new("model.tflite"), &out_dir, None)?;
```

```rust
// device/src/main.rs
mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated.rs"));
}

generated::init();
generated::forward(&input, &mut output);
```

The generated code is plain Rust: it also compiles for the host (every crate
here has a `local` feature for that), so models can be debugged natively
before touching hardware.

### Using the USB connection

```rust
let conn = PSPConnection::connect(prx_dir, prx_dir, Default::default())?;

let outcome = conn.load_program(&format!("host1:{prx_name}"), |bytes| {
    std::io::stdout().write_all(bytes).ok();
})?;

match outcome {
    LoadOutcome::Success => println!("done"),
    LoadOutcome::Panicked => println!("panicked"),
    LoadOutcome::ShellError(v) | LoadOutcome::KernelError(v) => println!("error: {v:#x}"),
}
```

`load_program` streams stdout live and only returns once psplink's shell
channel reports the module's thread has exited.

## Structure

```
psp-ml/
├── apps/
│   └── pspbird/            # the app: device crate, host runner, benchmark, installer
├── toolchain/
│   ├── psp-tc/             # TFLite -> Rust compiler (+ vme-assembler)
│   ├── psp-rt/             # no_std device runtime and kernels
│   ├── psplink-connection/ # USB link to psplink
│   ├── usbhostfs-sys/      # vendored hostfs protocol FFI
│   ├── vme-emu/ vme-emu-sys/  # Media Engine RTL model
│   └── device-tests/       # on-hardware test runner (cargo test -p device-tests)
├── plugins/                # kernel-mode PRXes (profiler, VME server)
├── examples/               # host/device pairs: benchmarks, demos, diagnostics
├── models/                 # fetch.py + reference/verification scripts
└── docs/                   # engineering log and retrospective
```

Every example (and the app) is a `host/` + `device/` pair. `device/` is an
ordinary PSP crate; `host/` is a native binary whose `build.rs` cross-compiles
the sibling via the unmodified `cargo psp` and whose `main()` deploys it over
USB. `cargo run -p <name>-host --release` is the whole loop.

## Setup

- Install the [PSP SDK](https://pspdev.github.io/installation) — installing to
  the repo root and sourcing `.envrc` works well
- `cargo install cargo-psp`
- libusb-1.0 dev headers (via pkg-config)
- `uv run models/fetch.py` for the model assets (gitignored; ~100 MB)

`cargo test` runs everything that needs neither hardware nor downloaded
assets. `cargo test -p device-tests` runs the kernel checks on a connected
PSP.

## Examples

`examples/` holds the benchmarks and diagnostics the toolchain was built
against: `mnist-bench` (the first end-to-end model), `roofline` (memory/VFPU
peaks), `fc-bench`/`gemv-bench` (kernel tuning), `fft-demo` (VFPU instruction
behaviour), the `birdnet-*-benchmark` trio (frontend development), the VME
suite, `profiler-*` (hardware counters), and `hello-psp`/`meminfo`/
`audio-recorder`.
