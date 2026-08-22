# Inference for the PlayStation Portable

An AOT tflite compiler and psplink-compatibly connection library in rust. Two pieces:

- **A native Rust connection to a live PSP.** Run psplink on the device, and launch programs from ordinary rust and read the PSP's stdout
- **A TFLite → Rust AOT compiler.** `psp-tc` reads a `.tflite` model and generates Rust inference
  code — VFPU-accelerated conv2d/pooling/fully-connected kernels, static memory planning, no heap,
  no interpreter — that you link into your own PSP crate from a normal `build.rs`. Currently runs
  MNIST at 99% accuracy in ~15ms/image and BirdNET (hybrid) quantized int8/fp16 in about 5.6sec / audio clip

## Quick start

```bash
cargo run -p mnist-bench-host --release
```

One command: cross-compiles the device crate, deploys it to a PSP connected over USB, streams its
stdout back, reports accuracy and per-op timing, and leaves nothing running on the device
afterward. No separate build/install/run steps.

## The USB connection, on its own

Instead of having to launch `usbhostfs_pc` in a separate shell, compile your PRX, then launch with `pspsh -e`, set up a `PSPConnection` in rust:

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

`load_program` streams stdout live and only returns once psplink's shell channel reports the
module's thread has actually exited (a real completion signal, not a load acknowledgement).

## The compiler, on its own

`psp-tc` turns a `.tflite` file into a Rust module — no runtime interpreter, no dynamic graph
traversal, just generated code that calls straight into VFPU-accelerated kernels with all
intermediate tensor storage planned statically at compile time. It's a normal `build.rs`
dependency:

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

The generated code is just Rust — profile it, feature-gate it, run it on your dev machine under
`#[cfg(feature = "local")]` for fast iteration before ever touching hardware (every example here
does exactly that), or link it into a larger PSP application alongside handwritten code.

## Structure

```
psp-ml/
├── psp-rt/                 # device-side no_std runtime: kernels, profiler, module! macro
├── psp-tc/                 # TFLite -> Rust compiler (library + standalone `psp-tc` CLI)
├── psplink-connection/     # safe API for talking to a PSP running psplink over USB
├── usbhostfs-sys/          # raw FFI to the vendored USB/hostfs protocol implementation
├── audio-recorder/         # PSP audio recording app
├── examples/
│   └── mnist-bench/
│       ├── host/            # native binary: builds device/, deploys + runs it over USB
│       └── device/          # the actual PSP crate (#![no_std], links against psp-rt)
└── models/                  # training scripts
```

Every example is a `host/` + `device/` pair. `device/` is an ordinary PSP crate — if it needs
a TFLite model, its own `build.rs` calls `psp-tc` to generate the inference code at build time.
`host/` is a normal native binary: its `build.rs` cross-compiles the sibling `device/` crate
into a `.prx` via the standard `cargo psp`, and its `main()` connects over USB
(`psplink-connection`) and deploys it.

## Setup
- Install the PSP dev [SDK](https://pspdev.github.io/installation)
    - I like to install it directly to the repo root and then source `.envrc`
- Install `cargo-psp`: https://crates.io/crates/psp

## Components

### psp-rt

The `no_std` runtime every device crate links against: VFPU-accelerated kernel
implementations (conv2d, pooling, fully connected, etc.), the hardware profiler bindings, and
the `psp_rt::module!` macro that PSP binaries use as their entry point. Most of this crate
(`module!`, `print`, `profiler`) isn't ML-specific at all — it's the general PSP homebrew
runtime; `kernels` is the one ML-specific module, used only by examples with a TFLite model.

### psp-tc

The TFLite -> Rust compiler. Parses TFLite models (via FlatBuffers), lowers them to an IR, and
generates Rust inference code that calls into `psp-rt`'s kernels. Exposes both a library API
(`psp_tc::compile_tflite`, what `device/build.rs` scripts call) and a standalone CLI
(`psp-tc compile model.tflite -o src/`, `psp-tc info model.tflite`) for manual/diagnostic use.

### psplink-connection / usbhostfs-sys

Talks directly to a PSP running psplink over USB — no `usbhostfs_pc` subprocess, no TCP
bridge. `usbhostfs-sys` is a vendored, refactored fork of `usbhostfs_pc`'s device-lifecycle and
hostfs-protocol C code; `psplink-connection` is the safe Rust API on top
(`PSPConnection::connect`/`load_program`/`disconnect`) that every `host/` crate's `main()` uses.

### audio-recorder

A standalone PSP application for recording audio via the microphone.

### examples/mnist-bench

Benchmark that runs MNIST inference on the PSP, measuring accuracy and throughput.
