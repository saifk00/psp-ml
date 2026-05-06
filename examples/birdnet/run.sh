#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_JSON="$SCRIPT_DIR/benchmarks.json"

usage() {
    echo "Usage: $0 [--local | --psp]"
    echo ""
    echo "Modes:"
    echo "  --local   Build and run on host CPU (default)"
    echo "  --psp     Build for PSP, deploy via cargo psp-ml run, collect results"
    exit 1
}

MODE="local"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) MODE="local"; shift ;;
        --psp)   MODE="psp"; shift ;;
        --help|-h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Step 1: Generate inference code from TFLite model
# --stream-batch 27:279 processes 511 frames one-at-a-time through the main conv
# block, reducing arena from ~161 MiB to ~9.6 MiB.
echo "==> Generating inference code..."
cd "$ROOT_DIR"
cargo psp-ml compile "$ROOT_DIR/models/BirdNET_v2.4_tflite/audio-model.tflite" \
    --stream-batch 27:279 \
    -o "$SCRIPT_DIR/src/"

if [ "$MODE" = "local" ]; then
    # -------------------------------------------------------------------------
    # Local mode: build and run on host CPU
    # -------------------------------------------------------------------------
    echo "==> Building for local host (release)..."
    cargo run -p birdnet --features local --release
    echo ""
    echo "==> benchmarks.json:"
    cat "$BENCH_JSON"

elif [ "$MODE" = "psp" ]; then
    # -------------------------------------------------------------------------
    # PSP mode: build, deploy via cargo psp-ml run, wait for results
    # -------------------------------------------------------------------------

    rm -f "$BENCH_JSON"

    # cd to script dir so host0:/ maps here (benchmarks.json lands in place)
    cd "$SCRIPT_DIR"
    echo "==> Building and deploying to PSP..."
    cargo psp-ml run -p birdnet --release

    # Wait for PSP to write benchmarks.json via HostFS (host0:/)
    echo "==> Waiting for benchmarks.json..."
    TIMEOUT=600
    ELAPSED=0
    while [ ! -f "$BENCH_JSON" ] && [ "$ELAPSED" -lt "$TIMEOUT" ]; do
        sleep 1
        ELAPSED=$((ELAPSED + 1))
    done

    if [ -f "$BENCH_JSON" ]; then
        echo ""
        echo "==> benchmarks.json:"
        cat "$BENCH_JSON"
    else
        echo "==> Timed out waiting for benchmarks.json"
        exit 1
    fi
fi
