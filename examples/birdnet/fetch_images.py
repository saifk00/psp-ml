# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests",
# ]
# ///
"""Fetch one representative photo per BirdNET species from iNaturalist.

BirdNET labels are ``Scientific name_Common name``; the scientific name is
the join key into iNaturalist's taxonomy (``GET /v1/taxa?q=<name>``). Each
taxon carries a ``default_photo`` with a machine-readable licence and
attribution string, which is what makes the result shippable: photos whose
licence is not in ``--licenses`` are skipped in favour of the first
acceptable one in the taxon's ``taxon_photos`` list.

Everything lands under ``--out`` (default ``examples/birdnet/images/``,
gitignored -- only this script is committed):

    raw/<Genus_species>.jpg   the photo, iNat "medium" size (500 px long edge)
    manifest.toml             ``"<label>" = "raw/<file>"`` -- what
                              ``birdnet::imfile::pack_images`` consumes
    manifest.json             provenance: taxon id, photo id, licence, url
    IMAGE_CREDITS.txt         attribution lines, one per photo

The run is resumable: a label already in manifest.json with its file on
disk is not fetched again, so the ~1150-species union of every region blob
can be built up across runs at iNat's polite rate (one request per second,
which is what ``--delay`` defaults to).

Run from the repo root. Inputs are labels.txt files or PBRD region blobs
(their labels are read out of the blob), and the union is fetched -- so
every region the app ships is one command::

    uv run examples/birdnet/fetch_images.py examples/birdnet/device/target/mipsel-sony-psp/release/blobs/*.bin
    uv run examples/birdnet/fetch_images.py a.labels.txt b.labels.txt --report

Taxonomy mismatches (BirdNET follows eBird/Clements; iNat lags some splits)
show up under ``--report`` as misses; pin those by hand in
``<out>/overrides.json`` as ``{"<label>": <inat taxon id>}``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import requests

API = "https://api.inaturalist.org/v1"
USER_AGENT = "psp-ml pspbird image fetcher (github.com/saifkhattak/psp-ml)"

# BirdNET's non-bird classes (Engine, Siren, ...) look like "Engine_Engine"
# and have no taxon; they get a placeholder on device, not a photo.
NON_BIRD_RE = re.compile(r"^(?P<sci>[^_]+)_(?P<common>.+)$")

DEFAULT_LICENSES = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa")


def labels_of(path: str) -> list[str]:
  """Labels from a labels.txt, or from a PBRD classifier blob (the labels
  ride at the end of the file; see prune_classifier.py --write-blob)."""
  with open(path, "rb") as f:
    data = f.read()
  if data[:4] == b"PBRD":
    import struct
    n, k, labels_len = struct.unpack_from("<III", data, 8)
    start = 32 + 4 * (n * k + n)
    text = data[start:start + labels_len].decode("utf-8")
  else:
    text = data.decode("utf-8")
  return [l.rstrip("\r").strip() for l in text.splitlines()]


def read_labels(paths: list[str]) -> list[str]:
  """Union of the label sources, first-seen order, duplicates dropped."""
  seen: dict[str, None] = {}
  for path in paths:
    for line in labels_of(path):
      if line:
        seen.setdefault(line, None)
  return list(seen)


def split_label(label: str) -> tuple[str, str] | None:
  """``(scientific, common)`` for a bird label, None for a non-bird class."""
  m = NON_BIRD_RE.match(label)
  if m is None:
    return None
  sci, common = m.group("sci"), m.group("common")
  # Non-bird classes repeat the name on both sides ("Engine_Engine"), and
  # a real binomial has a space in it.
  if sci == common or " " not in sci:
    return None
  return sci, common


def file_name_for(sci: str) -> str:
  return re.sub(r"[^A-Za-z0-9]+", "_", sci).strip("_") + ".jpg"


class Client:
  def __init__(self, delay: float):
    self.delay = delay
    self.session = requests.Session()
    self.session.headers["User-Agent"] = USER_AGENT
    self.last = 0.0

  def get(self, url: str, **params):
    wait = self.last + self.delay - time.monotonic()
    if wait > 0:
      time.sleep(wait)
    for attempt in range(4):
      r = self.session.get(url, params=params or None, timeout=30)
      self.last = time.monotonic()
      if r.status_code == 429 or r.status_code >= 500:
        time.sleep(5 * (attempt + 1))
        continue
      r.raise_for_status()
      return r
    r.raise_for_status()
    return r


def pick_photo(photos: list[dict], licenses: tuple[str, ...]) -> dict | None:
  for p in photos:
    if p and p.get("license_code") in licenses and p.get("medium_url"):
      return p
  return None


def lookup(client: Client, sci: str, taxon_id: int | None,
           licenses: tuple[str, ...]) -> tuple[dict, dict] | None:
  """``(taxon, photo)`` for a scientific name, or None if nothing usable."""
  if taxon_id is None:
    r = client.get(f"{API}/taxa", q=sci, rank="species", per_page=5)
    results = r.json().get("results", [])
    # Prefer an exact name match; iNat's search is fuzzy.
    exact = [t for t in results if t.get("name", "").lower() == sci.lower()]
    candidates = exact or results[:1]
    if not candidates:
      return None
    taxon = candidates[0]
  else:
    r = client.get(f"{API}/taxa/{taxon_id}")
    results = r.json().get("results", [])
    if not results:
      return None
    taxon = results[0]

  photo = pick_photo([taxon.get("default_photo")], licenses)
  if photo is None:
    # The summary record only carries default_photo; the full one lists
    # every curated photo, in the order the taxon page shows them.
    r = client.get(f"{API}/taxa/{taxon['id']}")
    full = r.json().get("results", [{}])[0]
    photo = pick_photo([tp.get("photo") for tp in full.get("taxon_photos", [])], licenses)
  if photo is None:
    return None
  return taxon, photo


def write_outputs(out: str, manifest: dict) -> None:
  with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
  # TOML: keys are BirdNET labels (spaces, apostrophes), so always quoted.
  with open(os.path.join(out, "manifest.toml"), "w", encoding="utf-8") as f:
    f.write("# Generated by examples/birdnet/fetch_images.py -- do not edit.\n")
    f.write("# label = path, relative to this file. Read by birdnet::imfile::pack_images.\n\n")
    for label in sorted(manifest):
      entry = manifest[label]
      if entry.get("file"):
        f.write(f'{json.dumps(label)} = {json.dumps(entry["file"])}\n')
  with open(os.path.join(out, "IMAGE_CREDITS.txt"), "w", encoding="utf-8") as f:
    f.write("Species photos, fetched from iNaturalist by examples/birdnet/fetch_images.py.\n"
            "Each line: label | iNat photo id | licence | attribution | source\n\n")
    for label in sorted(manifest):
      e = manifest[label]
      if e.get("file"):
        f.write(f"{label} | photo {e['photo_id']} | {e['license']} | "
                f"{e['attribution']} | {e['url']}\n")


def main() -> int:
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("labels", nargs="+",
                 help="labels.txt file(s) and/or PBRD blobs; their union is fetched")
  p.add_argument("-o", "--out", default=os.path.join(os.path.dirname(__file__), "images"),
                 help="output directory (default: examples/birdnet/images)")
  p.add_argument("--licenses", default=",".join(DEFAULT_LICENSES),
                 help="comma-separated iNat licence codes to accept "
                      f"(default: {','.join(DEFAULT_LICENSES)})")
  p.add_argument("--delay", type=float, default=1.0,
                 help="seconds between requests (iNat asks for <=1 req/s)")
  p.add_argument("--report", action="store_true",
                 help="after fetching, list every label without a photo")
  p.add_argument("--dry-run", action="store_true",
                 help="resolve taxa and photos but download nothing")
  args = p.parse_args()

  licenses = tuple(s.strip() for s in args.licenses.split(",") if s.strip())
  out = args.out
  raw = os.path.join(out, "raw")
  os.makedirs(raw, exist_ok=True)

  manifest_path = os.path.join(out, "manifest.json")
  manifest: dict = {}
  if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
      manifest = json.load(f)
  overrides: dict[str, int] = {}
  overrides_path = os.path.join(out, "overrides.json")
  if os.path.exists(overrides_path):
    with open(overrides_path, "r", encoding="utf-8") as f:
      overrides = json.load(f)

  labels = read_labels(args.labels)
  client = Client(args.delay)
  fetched = skipped = missed = nonbird = 0
  try:
    for i, label in enumerate(labels, 1):
      parts = split_label(label)
      if parts is None:
        nonbird += 1
        continue
      sci, _common = parts
      entry = manifest.get(label)
      if entry and entry.get("file") and os.path.exists(os.path.join(out, entry["file"])):
        skipped += 1
        continue
      if entry is not None and entry.get("file") is None and label not in overrides:
        # A recorded miss; only an override changes the answer.
        missed += 1
        continue

      print(f"[{i}/{len(labels)}] {label} ...", end=" ", flush=True)
      try:
        found = lookup(client, sci, overrides.get(label), licenses)
      except requests.RequestException as e:
        print(f"error: {e}")
        continue
      if found is None:
        print("no usable photo")
        manifest[label] = {"file": None, "reason": "no photo under accepted licences"}
        missed += 1
        continue
      taxon, photo = found
      fname = file_name_for(sci)
      rel = os.path.join("raw", fname)
      if args.dry_run:
        print(f"taxon {taxon['id']} photo {photo['id']} ({photo.get('license_code')}) [dry run]")
        fetched += 1
        continue
      r = client.get(photo["medium_url"])
      # iNat occasionally serves a PNG under a .jpg URL (fine, the packer
      # sniffs), but an HTML error page is not a photo: do not record it.
      if r.content[:2] != b"\xff\xd8" and r.content[:8] != b"\x89PNG\r\n\x1a\n":
        print(f"not an image ({len(r.content)} B, {r.headers.get('content-type')})")
        continue
      with open(os.path.join(out, rel), "wb") as f:
        f.write(r.content)
      manifest[label] = {
          "file": rel,
          "scientific_name": sci,
          "inat_name": taxon.get("name"),
          "taxon_id": taxon["id"],
          "photo_id": photo["id"],
          "license": photo.get("license_code"),
          "attribution": photo.get("attribution"),
          "url": photo["medium_url"],
      }
      fetched += 1
      print(f"taxon {taxon['id']} photo {photo['id']} ({photo.get('license_code')})")
      # Checkpoint: a crash or Ctrl-C loses nothing already fetched.
      write_outputs(out, manifest)
  finally:
    write_outputs(out, manifest)

  print(f"\n{fetched} fetched, {skipped} already present, {missed} without a photo, "
        f"{nonbird} non-bird classes skipped; {len(labels)} labels in.")
  print(f"manifest: {os.path.join(out, 'manifest.toml')}")
  if args.report:
    misses = [l for l in labels if split_label(l) and not manifest.get(l, {}).get("file")]
    if misses:
      print(f"\n{len(misses)} label(s) without a photo (pin in {overrides_path}):")
      for l in misses:
        print(f"  {l}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
