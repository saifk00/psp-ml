#!/usr/bin/env python3
"""Summarise a device run's per-op timings.

Every `cargo run -p <example>-host --release` writes a benchmarks.json next to
the example with one entry per *sub-op*. That's 393 entries for BirdNET, which
is too granular to read — this aggregates them by kernel so the hot spots are
obvious.

    python3 models/op_breakdown.py                      # BirdNET, grouped
    python3 models/op_breakdown.py --each               # slowest individual calls
    python3 models/op_breakdown.py path/to/benchmarks.json

Stdlib only, so no `uv run` needed.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT = REPO / "apps/pspbird/benchmarks.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT, type=Path)
    ap.add_argument("--each", action="store_true", help="list slowest individual calls")
    ap.add_argument("-n", type=int, default=15, help="rows to show (default 15)")
    args = ap.parse_args()

    if not args.path.exists():
        sys.exit(
            f"{args.path} not found — run the example first, e.g.\n"
            f"  cargo run -p pspbird-bench-host --release"
        )

    d = json.loads(args.path.read_text())
    ops = d["ops"]
    total = d["inference"]["total_us"]
    print(f"{d.get('model', '?')}: {total / 1000:.0f} ms over {len(ops)} sub-ops\n")

    if args.each:
        print(f"{'idx':>5} {'kernel':<28}{'ms':>9}{'%':>7}")
        for o in sorted(ops, key=lambda o: -o["total_us"])[: args.n]:
            print(
                f"{o['index']:>5} {o['name']:<28}"
                f"{o['total_us'] / 1000:9.1f}{100 * o['total_us'] / total:7.1f}"
            )
        return

    agg = collections.defaultdict(lambda: [0, 0])
    for o in ops:
        a = agg[o["name"]]
        a[0] += o["total_us"]
        a[1] += 1

    print(f"{'kernel':<28}{'ms':>9}{'%':>7}{'calls':>7}{'ms/call':>9}{'cum%':>7}")
    cum = 0.0
    for name, (us, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[: args.n]:
        pct = 100 * us / total
        cum += pct
        print(
            f"{name:<28}{us / 1000:9.1f}{pct:7.1f}{n:7d}"
            f"{us / n / 1000:9.2f}{cum:7.0f}"
        )

    accounted = sum(us for us, _ in agg.values())
    print(f"\naccounted: {100 * accounted / total:.0f}% of wall time")


if __name__ == "__main__":
    main()
