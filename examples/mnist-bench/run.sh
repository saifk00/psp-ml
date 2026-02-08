#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Step 1: Generate inference code from TFLite model
echo "==> Generating inference code..."
cd "$ROOT_DIR"
cargo psp-ml "$SCRIPT_DIR/mnist_cnn.tflite" -o "$SCRIPT_DIR/src/"

# Step 2: Build for PSP
echo "==> Building for PSP..."
cargo +nightly psp --release

# Step 3: Run on device via PSPLINK
echo "==> Running on PSP..."
pspsh -e "host0:/target/mipsel-sony-psp/release/mnist-bench.prx"
