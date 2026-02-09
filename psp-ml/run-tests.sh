#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_JSON="$SCRIPT_DIR/test-results.json"

usage() {
    echo "Usage: $0 [--local | --psp]"
    echo ""
    echo "Modes:"
    echo "  --local   Build and run on host CPU (default)"
    echo "  --psp     Build for PSP, deploy via usbhostfs_pc, collect results"
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

if [ "$MODE" = "local" ]; then
    echo "==> Building kernel tests for local host (release)..."
    cd "$ROOT_DIR"
    cargo run -p psp-ml --bin test-kernels --features psp-ml/test-kernels,psp-ml/local --release
    echo ""
    echo "==> test-results.json:"
    cat "$RESULTS_JSON"

elif [ "$MODE" = "psp" ]; then
    echo "==> Building kernel tests for PSP (release)..."
    cd "$ROOT_DIR"
    cargo +nightly psp -p psp-ml --bin test-kernels --features psp-ml/test-kernels --release

    PRX="target/mipsel-sony-psp/release/test-kernels.prx"

    # Ensure usbhostfs_pc is running
    if ! pgrep -x usbhostfs_pc > /dev/null 2>&1; then
        echo "==> Starting usbhostfs_pc in background..."
        usbhostfs_pc &
        USBHOST_PID=$!
        sleep 2
        echo "    PID: $USBHOST_PID"
    else
        USBHOST_PID=""
        echo "==> usbhostfs_pc already running"
    fi

    # Remove stale results
    rm -f test-results.json

    # Deploy and run on PSP
    echo "==> Deploying to PSP via psplink..."
    pspsh -e "host0:/$PRX"

    # Wait for results
    echo "==> Waiting for test results..."
    TIMEOUT=30
    ELAPSED=0
    while [ ! -f test-results.json ] && [ "$ELAPSED" -lt "$TIMEOUT" ]; do
        sleep 1
        ELAPSED=$((ELAPSED + 1))
    done

    if [ -f test-results.json ]; then
        echo "==> Test results collected!"
        mv test-results.json "$RESULTS_JSON"
        echo ""
        cat "$RESULTS_JSON"

        # Check for failures
        FAILED=$(python3 -c "import json; print(json.load(open('$RESULTS_JSON'))['failed'])" 2>/dev/null || echo "?")
        if [ "$FAILED" = "0" ]; then
            echo ""
            echo "==> All tests passed!"
        else
            echo ""
            echo "==> $FAILED test(s) FAILED"
            exit 1
        fi
    else
        echo "==> Timed out waiting for test-results.json after ${TIMEOUT}s"
        exit 1
    fi

    # Clean up usbhostfs_pc if we started it
    if [ -n "$USBHOST_PID" ]; then
        echo "==> Stopping usbhostfs_pc (PID $USBHOST_PID)..."
        kill "$USBHOST_PID" 2>/dev/null || true
    fi
fi
