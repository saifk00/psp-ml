#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PRX="$SCRIPT_DIR/psp_ml_kernel.prx"

if [ ! -f "$PRX" ]; then
    echo "==> PRX not found, building..."
    make -C "$SCRIPT_DIR"
fi

# usbhostfs_pc maps host1: to the workspace root
PRX_REL="${PRX#$WORKSPACE_ROOT/}"
HOST1_PATH="host1:/$PRX_REL"

echo "==> Loading kernel plugin: $HOST1_PATH"

# Send psplink 'ld' command over TCP (same protocol as cargo psp-ml run)
# Format: "ld\0<path>\0\x01"
printf 'ld\x00%s\x00\x01' "$HOST1_PATH" | nc -w 2 127.0.0.1 10000

echo "==> Done"
