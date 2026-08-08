# psp-ml

A TFLite-to-Rust AOT compiler and runtime for the PSP.

## Status: Experimental

Currently runs MNIST at 99% accuracy. Working toward real-time BirdNET inference.

## Setup
- Install the PSP dev [SDK](https://pspdev.github.io/installation)
    - I like to install it directly to the repo root and then source `.envrc`
- Install `cargo-psp`: https://crates.io/crates/psp

## Structure

```
psp-ml/
├── psp-rt/                  # device-side no_std runtime: kernels, profiler, module! macro
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

## Quick Start

```bash
cargo run -p mnist-bench-host --release
```

That's it — no separate compile step, no subprocess to keep running. `build.rs` handles
codegen and cross-compilation; `main()` handles deploying to a PSP connected over USB and
running psplink.

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
