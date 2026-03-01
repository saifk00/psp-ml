#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_JSON="$SCRIPT_DIR/benchmarks.json"

HISTORY_JSONL="$SCRIPT_DIR/benchmarks/history.jsonl"

usage() {
    echo "Usage: $0 [--local | --psp] [--config <kernel_type>] [--notes \"description\"]"
    echo ""
    echo "Modes:"
    echo "  --local   Build and run on host CPU (default)"
    echo "  --psp     Build for PSP, deploy via cargo psp-ml run, collect results"
    echo ""
    echo "Options:"
    echo "  --config  Kernel configuration tag (default: naive)"
    echo "            Recorded in benchmarks.json for comparison"
    echo "  --notes   When provided, appends this run to benchmarks/history.jsonl"
    echo "            with git hash, dirty status, and timestamp metadata"
    exit 1
}

MODE="local"
CONFIG="naive"
NOTES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) MODE="local"; shift ;;
        --psp)   MODE="psp"; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --notes)  NOTES="$2"; shift 2 ;;
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
    # PSP mode: build, deploy via cargo psp-ml run, wait for results
    # -------------------------------------------------------------------------

    rm -f "$BENCH_JSON"

    # cd to script dir so host0:/ maps here (benchmarks.json lands in place)
    cd "$SCRIPT_DIR"
    echo "==> Building and deploying to PSP..."
    cargo psp-ml run -p mnist-bench --release

    # Wait for PSP to write benchmarks.json via HostFS (host0:/)
    echo "==> Waiting for benchmarks.json..."
    TIMEOUT=120
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

# Append to JSONL history if --notes was provided
if [ -n "$NOTES" ] && [ -f "$BENCH_JSON" ]; then
    GIT_HASH="$(git -C "$ROOT_DIR" rev-parse --short HEAD)"
    GIT_DIRTY="$(git -C "$ROOT_DIR" diff --quiet && echo false || echo true)"
    TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    jq -c --arg hash "$GIT_HASH" \
           --argjson dirty "$GIT_DIRTY" \
           --arg ts "$TIMESTAMP" \
           --arg notes "$NOTES" \
      '{meta: {git_hash: $hash, git_dirty: $dirty, timestamp: $ts, notes: $notes}} + .' \
      "$BENCH_JSON" >> "$HISTORY_JSONL"

    echo ""
    echo "==> Appended to benchmarks/history.jsonl"
fi
