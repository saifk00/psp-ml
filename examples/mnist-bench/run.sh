#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_JSON="$SCRIPT_DIR/benchmarks.json"

usage() {
    echo "Usage: $0 [--local | --psp] [--config <kernel_type>]"
    echo ""
    echo "Modes:"
    echo "  --local   Build and run on host CPU (default)"
    echo "  --psp     Build for PSP, deploy via cargo psp-ml run, collect results"
    echo ""
    echo "Options:"
    echo "  --config  Kernel configuration tag (default: naive)"
    echo "            Recorded in benchmarks.json for comparison"
    exit 1
}

MODE="local"
CONFIG="naive"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) MODE="local"; shift ;;
        --psp)   MODE="psp"; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --help|-h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Step 1: Generate inference code from TFLite model
echo "==> Generating inference code..."
cd "$ROOT_DIR"
cargo psp-ml compile "$SCRIPT_DIR/mnist_cnn.tflite" -o "$SCRIPT_DIR/src/"

if [ "$MODE" = "local" ]; then
    # -------------------------------------------------------------------------
    # Local mode: build and run on host CPU
    # -------------------------------------------------------------------------
    echo "==> Building for local host (release)..."
    cargo run -p mnist-bench --features local --release
    echo ""
    echo "==> benchmarks.json:"
    cat "$BENCH_JSON"

elif [ "$MODE" = "psp" ]; then
    # -------------------------------------------------------------------------
    # PSP mode: build, deploy via cargo psp-ml run, collect JSON
    # -------------------------------------------------------------------------

    # Remove stale results
    rm -f "$ROOT_DIR/benchmarks.json"

    echo "==> Building and running on PSP..."
    cargo psp-ml run -p mnist-bench --release

    # benchmarks.json is written to workspace root via HostFS
    if [ -f "$ROOT_DIR/benchmarks.json" ]; then
        mv "$ROOT_DIR/benchmarks.json" "$BENCH_JSON"
        echo ""
        echo "==> benchmarks.json:"
        cat "$BENCH_JSON"
    else
        echo "==> Error: benchmarks.json not found after PSP execution"
        exit 1
    fi
fi
