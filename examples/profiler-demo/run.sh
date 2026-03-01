#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# cd to script dir so host0:/ maps here for psplink output
cd "$SCRIPT_DIR"

echo "==> Building and deploying profiler-demo to PSP..."
cargo psp-ml run -p profiler-demo --release
