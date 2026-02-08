# PSP-ML

Machine learning inference on the PlayStation Portable.

## Structure

```
psp-ml/
├── cargo-psp-ml/       # TFLite compiler + runtime kernels
├── audio-recorder/     # PSP audio recording app
├── examples/
│   └── mnist-bench/    # MNIST inference benchmark
└── models/             # Training scripts
```

## Quick Start

1. Install the cargo subcommand:
   ```bash
   cargo install --path cargo-psp-ml --features compiler
   ```

2. Compile a TFLite model:
   ```bash
   cargo psp-ml model.tflite --out examples/mnist-bench/src
   ```

3. Build for PSP:
   ```bash
   cd examples/mnist-bench
   cargo psp --release
   ```

## Components

### cargo-psp-ml

A library and CLI tool that:
- Parses TFLite models (via FlatBuffers)
- Generates Rust inference code
- Provides `no_std` kernel implementations (conv2d, pooling, fully connected, etc.)

### audio-recorder

A standalone PSP application for recording audio via the microphone.

### examples/mnist-bench

Benchmark that runs MNIST inference on the PSP, measuring accuracy and throughput.
