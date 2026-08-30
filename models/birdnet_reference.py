# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tensorflow>=2.16",
#   "numpy",
# ]
# ///
"""Host-side golden reference for the int8 BirdNET model.

Scores every 3 s window of a wav with tf.lite.Interpreter, picks the most
confident window, and writes:
  - apps/pspbird/golden.json    (chosen window, top-10 labels, full raw output)
  - apps/pspbird/cardinal_3s.wav (the chosen window's exact i16 samples)

The device build converts wav samples as i16 / 32768.0 -> f32; this script does
the identical conversion so host and device see bit-identical model inputs.

Run from the repo root:  uv run models/birdnet_reference.py
"""

import json
import os
import wave
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO = Path(__file__).resolve().parent.parent
# BIRDNET_MODEL (repo-relative) overrides; the default matches the device
# build's default so golden.json and the PSP score the same weights.
MODEL = REPO / os.environ.get("BIRDNET_MODEL", "models/birdnet/audio-model-fp32.tflite")
LABELS = REPO / "models/birdnet/labels/en_us.txt"
# BIRDNET_WAV overrides the fixture. The default is a verified Northern
# Cardinal (XC468176); the old cardinal.wav is NOT one -- BirdNET scores it
# 0.0096 for Northern Cardinal and calls it Pyrrhuloxia, while it scores three
# independent ground-truth cardinal recordings 0.38-0.84. See models/README.md.
WAV = Path(os.environ.get("BIRDNET_WAV", REPO / "models/birdnet/cardinal_xc468176.wav"))
OUT_DIR = REPO / "apps/pspbird"

SAMPLE_RATE = 48000
WINDOW = 3 * SAMPLE_RATE  # 144000
HOP = SAMPLE_RATE  # 1 s


def read_wav_i16(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 1 and w.getframerate() == SAMPLE_RATE and w.getsampwidth() == 2, (
            w.getparams()
        )
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write_wav_i16(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())


def main() -> None:
    samples = read_wav_i16(WAV)
    labels = LABELS.read_text().splitlines()

    interp = tf.lite.Interpreter(model_path=str(MODEL))
    interp.allocate_tensors()
    (inp,) = interp.get_input_details()
    (out,) = interp.get_output_details()
    assert inp["dtype"] == np.float32 and tuple(inp["shape"]) == (1, WINDOW), inp
    print(f"model: in {inp['shape']} {inp['dtype'].__name__}, out {out['shape']} {out['dtype'].__name__}")

    offsets = list(range(0, len(samples) - WINDOW + 1, HOP))
    if offsets[-1] != len(samples) - WINDOW:
        offsets.append(len(samples) - WINDOW)

    results = []
    for off in offsets:
        x = (samples[off : off + WINDOW].astype(np.float32) / 32768.0)[None, :]
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        raw = interp.get_tensor(out["index"])[0].copy()
        best = int(np.argmax(raw))
        results.append((off, raw))
        print(f"  window @{off:7d} ({off / SAMPLE_RATE:5.2f}s): best [{best}] {labels[best]}  raw={raw[best]:.4f}")

    best_off, best_raw = max(results, key=lambda r: r[1].max())
    top10 = np.argsort(best_raw)[::-1][:10]
    golden = {
        "model": MODEL.name,
        "wav": WAV.name,
        "window_offset_samples": int(best_off),
        "window_samples": WINDOW,
        "top10": [
            {"index": int(i), "label": labels[i], "raw": float(best_raw[i])} for i in top10
        ],
        "output_raw": [float(v) for v in best_raw],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "golden.json").write_text(json.dumps(golden, indent=1))
    write_wav_i16(OUT_DIR / "cardinal_3s.wav", samples[best_off : best_off + WINDOW])

    print(f"\nchosen window @ {best_off} ({best_off / SAMPLE_RATE:.2f}s). top-10:")
    for i in top10:
        print(f"  [{i:4d}] {best_raw[i]:9.4f}  {labels[i]}")
    print(f"\nwrote {OUT_DIR / 'golden.json'} and {OUT_DIR / 'cardinal_3s.wav'}")


if __name__ == "__main__":
    main()
