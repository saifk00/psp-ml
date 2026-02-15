#!/bin/bash
# Enrich a PSP benchmark result with timestamp and optional notes,
# then append it to the history file (JSONL, one run per line).
#
# Usage:
#   ./parse_bench.sh benchmarks.json              # no notes
#   ./parse_bench.sh benchmarks.json "VFPU conv2d" # with notes

set -euo pipefail

BENCH_FILE="${1:?usage: parse_bench.sh <benchmarks.json> [notes]}"
NOTES="${2:-}"
HISTORY="data/benchmarks.jsonl"

if [ ! -f "$BENCH_FILE" ]; then
    echo "error: $BENCH_FILE not found" >&2
    exit 1
fi

mkdir -p "$(dirname "$HISTORY")"

jq -c ". + {\"_timestamp\": (now | todate), \"_notes\": \"$NOTES\"}" "$BENCH_FILE" >> "$HISTORY"

echo "Appended to $HISTORY ($(wc -l < "$HISTORY") entries)"
