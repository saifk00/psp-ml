# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "soundfile", "soxr"]
# ///
"""Download the gitignored model assets into models/birdnet/.

Fetches from Zenodo record 15050749 (BirdNET Model V2.4) and xeno-canto:

  models/birdnet/audio-model-fp32.tflite   51.7 MB  BirdNET_v2.4_tflite.zip
  models/birdnet/audio-model-int8.tflite   41.1 MB  BirdNET_v2.4_tflite_int8.zip
  models/birdnet/meta-model.tflite          8.1 MB  species-by-location prior
  models/birdnet/labels/*.txt                       label files, all languages
  models/birdnet/cardinal_xc468176.wav      3.5 MB  XC468176, decoded to 48 kHz mono

Everything is idempotent: present files are skipped. --check reports what is
missing without downloading. --force re-downloads.

The committed fixtures (apps/pspbird/cardinal_3s.wav, golden.json) were cut
from this recording; regenerate them with `uv run models/birdnet_reference.py`
after fetching. Two caveats about the wav: it is decoded from xeno-canto's
mp3 by this script, and as of 2026-08-30 xeno-canto serves a 26 s file where
the page (and the original fixture) say 36 s -- so a golden regenerated from
a fresh fetch can pick a different window than the committed one. The
committed fixtures are the reference; only regenerate them deliberately.

Licences: BirdNET models CC BY-NC-SA 4.0 (Cornell Lab of Ornithology /
Chemnitz University of Technology). XC468176 (Northern Cardinal, J. Tinsley)
CC BY-NC-SA 4.0 -- see apps/pspbird/AUDIO_CREDITS.txt.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "birdnet"

ZENODO = "https://zenodo.org/api/records/15050749/files/{}/content"
XC_WAV = "https://xeno-canto.org/468176/download"

# (target under models/birdnet/, source zip, member name)
FROM_ZENODO = [
    ("audio-model-fp32.tflite", "BirdNET_v2.4_tflite.zip", "audio-model.tflite"),
    ("meta-model.tflite", "BirdNET_v2.4_tflite.zip", "meta-model.tflite"),
    ("audio-model-int8.tflite", "BirdNET_v2.4_tflite_int8.zip", "audio-model-int8.tflite"),
]


def fetch(url: str, what: str) -> bytes:
    print(f"downloading {what} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "psp-ml-fetch/1.0"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def missing(force: bool) -> tuple[list, bool, bool]:
    from_zips = [t for t in FROM_ZENODO if force or not (DEST / t[0]).exists()]
    labels = force or not (DEST / "labels/en_us.txt").exists()
    wav = force or not (DEST / "cardinal_xc468176.wav").exists()
    return from_zips, labels, wav


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="report missing files and exit")
    p.add_argument("--force", action="store_true", help="re-download everything")
    args = p.parse_args()

    from_zips, labels, wav = missing(args.force)
    if args.check:
        for target, _, _ in from_zips:
            print(f"missing: models/birdnet/{target}")
        if labels:
            print("missing: models/birdnet/labels/")
        if wav:
            print("missing: models/birdnet/cardinal_xc468176.wav")
        if not (from_zips or labels or wav):
            print("all model assets present")
        sys.exit(1 if (from_zips or labels or wav) else 0)

    DEST.mkdir(parents=True, exist_ok=True)

    # One download per zip, however many members it supplies.
    needed_zips = {z for _, z, _ in from_zips}
    if labels:
        needed_zips.add("BirdNET_v2.4_tflite.zip")
    for zname in sorted(needed_zips):
        zf = zipfile.ZipFile(io.BytesIO(fetch(ZENODO.format(zname), zname)))
        for target, src, member in from_zips:
            if src == zname:
                (DEST / target).write_bytes(zf.read(member))
                print(f"  wrote models/birdnet/{target}")
        if labels and zname == "BirdNET_v2.4_tflite.zip":
            (DEST / "labels").mkdir(exist_ok=True)
            for member in zf.namelist():
                if member.startswith("labels/") and member.endswith(".txt"):
                    (DEST / member).write_bytes(zf.read(member))
            print("  wrote models/birdnet/labels/ (all languages)")

    if wav:
        import numpy as np
        import soundfile as sf
        import soxr

        data, rate = sf.read(io.BytesIO(fetch(XC_WAV, "XC468176 (Northern Cardinal)")))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if rate != 48_000:
            data = soxr.resample(data, rate, 48_000)
        pcm = np.clip(data * 32768.0, -32768, 32767).astype(np.int16)
        sf.write(DEST / "cardinal_xc468176.wav", pcm, 48_000, subtype="PCM_16")
        print("  wrote models/birdnet/cardinal_xc468176.wav (48 kHz mono)")

    print("done. next steps as needed:")
    print("  uv run models/birdnet_reference.py     # regenerate golden.json + cardinal_3s.wav")
    print("  uv run apps/pspbird/fetch_images.py    # species photos for the app")


if __name__ == "__main__":
    main()
