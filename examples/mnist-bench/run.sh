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
    echo "  --psp     Build for PSP, deploy via usbhostfs_pc, collect results"
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
    # PSP mode: build, deploy via usbhostfs_pc + pspsh, collect JSON
    # -------------------------------------------------------------------------
    echo "==> Building for PSP (release)..."
    cargo +nightly psp --release

    PRX="target/mipsel-sony-psp/release/mnist-bench.prx"

    # Ensure usbhostfs_pc is running (start it if not)
    if ! pgrep -x usbhostfs_pc > /dev/null 2>&1; then
        echo "==> Starting usbhostfs_pc in background..."
        usbhostfs_pc &
        USBHOST_PID=$!
        sleep 2  # Give it time to connect
        echo "    PID: $USBHOST_PID"
    else
        USBHOST_PID=""
        echo "==> usbhostfs_pc already running"
    fi

    # Remove stale results
    rm -f benchmarks.json

    # Deploy and run on PSP
    echo "==> Deploying to PSP via psplink..."
    pspsh -e "host0:/$PRX"

    # Wait for the PSP to write results (poll for benchmarks.json)
    echo "==> Waiting for benchmark results..."
    TIMEOUT=60
    ELAPSED=0
    while [ ! -f benchmarks.json ] && [ "$ELAPSED" -lt "$TIMEOUT" ]; do
        sleep 1
        ELAPSED=$((ELAPSED + 1))
    done

    if [ -f benchmarks.json ]; then
        echo "==> Benchmark results collected!"
        mv benchmarks.json "$BENCH_JSON"
        echo ""
        cat "$BENCH_JSON"
    else
        echo "==> Timed out waiting for benchmarks.json after ${TIMEOUT}s"
        exit 1
    fi

    # Clean up usbhostfs_pc if we started it
    if [ -n "$USBHOST_PID" ]; then
        echo "==> Stopping usbhostfs_pc (PID $USBHOST_PID)..."
        kill "$USBHOST_PID" 2>/dev/null || true
    fi
fi
