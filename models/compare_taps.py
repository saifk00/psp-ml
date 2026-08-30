# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tensorflow>=2.16",
#   "numpy",
# ]
# ///
"""Diff psp-tc's per-op tensor taps against a TFLite reference run.

Run `BIRDNET_TAP=1 cargo run -p birdnet --features local --release` first to
produce apps/pspbird/device/tap/t<ID>.bin, then this script. psp-tc tensor
ids equal TFLite tensor indices, so each tap is compared elementwise with the
(dequantized) TFLite tensor of the same index, in graph-op order. The first
tensors with large error localize the first divergent op.
"""

import wave
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO = Path(__file__).resolve().parent.parent
TAP_DIR = REPO / "apps/pspbird/device/tap"
MODEL = REPO / "models/birdnet/audio-model-int8.tflite"
WAV = REPO / "apps/pspbird/cardinal_3s.wav"

with wave.open(str(WAV)) as w:
    samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
x = (samples.astype(np.float32) / 32768.0)[None, :]

interp = tf.lite.Interpreter(
    model_path=str(MODEL), experimental_preserve_all_tensors=True
)
interp.allocate_tensors()
(inp,) = interp.get_input_details()
interp.set_tensor(inp["index"], x)
interp.invoke()

details = {d["index"]: d for d in interp.get_tensor_details()}

manifest = [
    line.split() for line in (TAP_DIR / "manifest.txt").read_text().splitlines()
]

print(f"{'op':>4} {'tensor':>6} {'n':>9} {'max_abs_diff':>12} {'rel':>9}  note")
worst = []
for op_idx, tid, n in ((int(a), int(b), int(c)) for a, b, c in manifest):
    ours = np.fromfile(TAP_DIR / f"t{tid}.bin", dtype="<f4")
    if tid not in details:
        print(f"{op_idx:>4} {tid:>6} {n:>9}  (no TFLite tensor)")
        continue
    d = details[tid]
    try:
        ref = interp.get_tensor(tid)
    except Exception as e:
        print(f"{op_idx:>4} {tid:>6} {n:>9}  (unreadable: {e})")
        continue
    if ref.dtype == np.int8:
        q = d["quantization_parameters"]
        scale = q["scales"][0] if len(q["scales"]) else 1.0
        zp = q["zero_points"][0] if len(q["zero_points"]) else 0
        ref = (ref.astype(np.float32) - zp) * scale
    elif ref.dtype != np.float32:
        continue
    ref = ref.flatten()
    if len(ref) != len(ours):
        print(f"{op_idx:>4} {tid:>6} {n:>9}  SIZE MISMATCH ours={len(ours)} ref={len(ref)}")
        continue
    diff = np.abs(ours - ref)
    denom = np.abs(ref).max() + 1e-9
    rel = diff.max() / denom
    flag = "  <-- DIVERGES" if rel > 0.05 else ""
    print(f"{op_idx:>4} {tid:>6} {n:>9} {diff.max():>12.5f} {rel:>9.4f}{flag}")
    worst.append((rel, op_idx, tid))

worst.sort(reverse=True)
print("\nworst 5:", [(f"op{o}", f"t{t}", f"{r:.3f}") for r, o, t in worst[:5]])
