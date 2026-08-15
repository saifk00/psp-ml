# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "ai-edge-litert",
#   "numpy",
#   "flatbuffers",
# ]
# # ai-edge-litert (~20 MB) supplies both the TFLite schema and the interpreter
# # this needs. Deliberately NOT tensorflow: a build script should not pull half
# # a gigabyte. The code still falls back to tensorflow's copies if that is all
# # that is installed, which is what BIRDNET_PYTHON generally points at.
# ///
"""Prune the BirdNET classifier layer down to a chosen subset of species.

The BirdNET audio model ends in::

    GLOBAL_AVG_POOL -> FULLY_CONNECTED [6522, 1024] -> QUANTIZE -> DEQUANTIZE

Each logit is an independent dot product ``W[j] . x + b[j]`` and the sigmoid is
applied *outside* the model, so dropping rows of W (and the matching entries of
b) leaves every surviving logit bit-for-bit unchanged. No retraining, no
recalibration. Dropping the ~6000 species that cannot occur in the deployment
region takes the int8 model from 41 MB to ~16 MB.

The species list can be given explicitly, or derived from BirdNET's own
meta-model, which maps (lat, lon, week) -> per-species likelihood.

Run from the repo root::

    # explicit list
    uv run examples/birdnet/prune_classifier.py MODEL.tflite labels.txt filtered.txt -o pruned.tflite

    # top 500 species for eastern North America
    uv run examples/birdnet/prune_classifier.py MODEL.tflite labels.txt --top-n 500

    # what threshold should I use?
    uv run examples/birdnet/prune_classifier.py MODEL.tflite labels.txt --report

`examples/birdnet/device/build.rs` invokes this automatically when TOPK is set.

Output classes are emitted in label-file order, and the matching labels file is
written alongside the pruned model.
"""

from __future__ import annotations

import argparse
import sys

import flatbuffers
import numpy as np

try:
  from ai_edge_litert import schema_py_generated as schema
except ImportError:  # pragma: no cover - fallback for tensorflow-only installs
  try:
    from tensorflow.lite.python import schema_py_generated as schema
  except ImportError:
    sys.exit(
        "Need a TFLite schema. Install one of:\n"
        "  pip install ai-edge-litert\n"
        "  pip install tensorflow"
    )

TFLITE_FILE_IDENTIFIER = b"TFL3"

# TFLite stores FULLY_CONNECTED weights as [out_features, in_features], so the
# class axis is axis 0 -- we slice rows.
CLASS_AXIS = 0


def read_lines(path: str) -> list[str]:
  """Read a labels file, preserving line order (index == class id)."""
  with open(path, "r", encoding="utf-8") as f:
    text = f.read()
  # BirdNET's labels files have no trailing newline, so `wc -l` under-reports by
  # one. splitlines() gets the true count and tolerates CRLF.
  return [line.rstrip("\r") for line in text.splitlines()]


def build_keep_indices(
    all_labels: list[str], wanted: list[str]
) -> tuple[list[int], list[str]]:
  """Map each wanted label to its row index in the full labels file."""
  exact: dict[str, int] = {}
  duplicates: set[str] = set()
  for i, label in enumerate(all_labels):
    if label in exact:
      duplicates.add(label)
    else:
      exact[label] = i
  if duplicates:
    raise ValueError(
        f"labels file has {len(duplicates)} duplicate entries, "
        f"e.g. {sorted(duplicates)[:3]}"
    )

  # Fallback index for case differences (e.g. a capitalised species epithet).
  folded: dict[str, list[int]] = {}
  for i, label in enumerate(all_labels):
    folded.setdefault(label.casefold(), []).append(i)

  keep: list[int] = []
  resolved: list[str] = []
  missing: list[str] = []
  seen: set[int] = set()

  for label in wanted:
    if not label.strip():
      continue
    idx = exact.get(label)
    if idx is None:
      candidates = folded.get(label.casefold(), [])
      if len(candidates) == 1:
        idx = candidates[0]
        print(
            f"  note: '{label}' matched '{all_labels[idx]}' "
            "(case-insensitive)",
            file=sys.stderr,
        )
      elif len(candidates) > 1:
        raise ValueError(f"'{label}' is ambiguous when matched case-insensitively")
    if idx is None:
      missing.append(label)
      continue
    if idx in seen:
      print(f"  note: '{label}' listed more than once, keeping first",
            file=sys.stderr)
      continue
    seen.add(idx)
    keep.append(idx)
    resolved.append(all_labels[idx])

  if missing:
    raise ValueError(
        f"{len(missing)} label(s) not found in the labels file: {missing[:5]}"
    )
  if not keep:
    raise ValueError("filtered labels file selected zero classes")
  return keep, resolved


# Bounding boxes as (lat_min, lat_max, lon_min, lon_max).
#
# `eastern-na` is deliberately tight: Great Lakes / Northeast / Mid-Atlantic
# down to Georgia. An earlier, wider box (lon -100) reached south Texas and let
# southwestern species through with high scores -- Pyrrhuloxia at 0.74, which is
# a near neighbour of Northern Cardinal and duly showed up as a false top-1.
# Under this box it scores 0.0003. Widen only if you mean it.
REGIONS = {
    "eastern-na": (32.0, 50.0, -88.0, -72.0),
    "eastern-na-wide": (25.0, 50.0, -100.0, -60.0),
    "western-na": (25.0, 50.0, -125.0, -100.0),
    "north-america": (15.0, 60.0, -130.0, -60.0),
    "europe": (35.0, 65.0, -10.0, 30.0),
}


def is_non_bird(label: str) -> bool:
  """BirdNET's ~10 non-species classes repeat the name on both sides of '_'.

  e.g. 'Dog_Dog', 'Power tools_Power tools'. Real species never do, since the
  left side is a binomial and the right a common name.
  """
  sci, _, common = label.partition("_")
  return bool(sci) and sci == common


def rank_species_by_meta(meta_path: str, num_classes: int, bbox, weeks,
                         step: float, reduce: str) -> np.ndarray:
  """Score every class over a lat/lon/week grid, returning per-class scores.

  The meta-model maps (lat, lon, week) -> per-species likelihood already in
  [0, 1]. Reducing with 'max' asks "could this species occur anywhere in the
  region at any time of year", which is what you want for a device that may be
  deployed anywhere inside the box. 'mean' favours widespread residents.
  """
  try:
    from ai_edge_litert.interpreter import Interpreter
  except ImportError:  # pragma: no cover
    from tensorflow.lite.python.interpreter import Interpreter

  interp = Interpreter(model_path=meta_path)
  interp.allocate_tensors()
  d_in = interp.get_input_details()[0]
  d_out = interp.get_output_details()[0]
  if tuple(d_out["shape"]) != (1, num_classes):
    raise ValueError(
        f"meta-model outputs {list(d_out['shape'])}, expected [1, {num_classes}]"
        " -- meta-model and labels file disagree"
    )

  lat0, lat1, lon0, lon1 = bbox
  lats = np.arange(lat0, lat1 + 1e-9, step)
  lons = np.arange(lon0, lon1 + 1e-9, step)
  acc = np.zeros(num_classes, dtype=np.float64)
  best = np.zeros(num_classes, dtype=np.float32)
  n = 0
  for lat in lats:
    for lon in lons:
      for wk in weeks:
        interp.set_tensor(
            d_in["index"], np.array([[lat, lon, wk]], dtype=np.float32)
        )
        interp.invoke()
        v = interp.get_tensor(d_out["index"])[0]
        np.maximum(best, v, out=best)
        acc += v
        n += 1
  print(f"  swept {len(lats)}x{len(lons)} grid @ {step} deg x {len(weeks)} "
        f"weeks = {n} evaluations")
  return best if reduce == "max" else (acc / n).astype(np.float32)


def select_top_n(all_labels: list[str], scores: np.ndarray,
                 top_n: int | None, min_score: float,
                 keep_non_bird: bool) -> tuple[list[int], list[str]]:
  """Pick classes by rank and/or likelihood, always retaining non-bird classes."""
  non_bird = set(i for i, l in enumerate(all_labels) if is_non_bird(l))
  order = np.argsort(scores)[::-1]
  limit = top_n if top_n is not None else len(all_labels)
  chosen: list[int] = []
  for i in order:
    i = int(i)
    if keep_non_bird and i in non_bird:
      continue  # added separately below so they never consume a slot
    if scores[i] <= min_score:
      break
    chosen.append(i)
    if len(chosen) >= limit:
      break

  if top_n is not None and len(chosen) < top_n:
    print(f"  note: only {len(chosen)} classes score above {min_score} in this "
          f"region (asked for {top_n})")
  print(f"  selected {len(chosen)} species "
        f"(score range {scores[chosen].min():.4f}..{scores[chosen].max():.4f})")

  if keep_non_bird:
    print(f"  force-keeping {len(non_bird)} non-bird classes "
          "(they score 0.0 geographically)")
    chosen.extend(non_bird)

  # Emit in original label-file order so the mapping is easy to eyeball.
  chosen = sorted(set(chosen))
  return chosen, [all_labels[i] for i in chosen]


def _buffer_array(model, tensor, dtype):
  buf = model.buffers[tensor.buffer]
  if buf.data is None or len(buf.data) == 0:
    raise ValueError(f"tensor '{tensor.name.decode()}' has no constant data")
  return np.frombuffer(bytes(bytearray(buf.data)), dtype=dtype).reshape(
      list(tensor.shape)
  )


def _tflite_dtype(tensor):
  mapping = {
      schema.TensorType.FLOAT32: np.float32,
      schema.TensorType.FLOAT16: np.float16,
      schema.TensorType.INT8: np.int8,
      schema.TensorType.UINT8: np.uint8,
      schema.TensorType.INT32: np.int32,
  }
  dt = mapping.get(tensor.type)
  if dt is None:
    raise ValueError(f"unsupported tensor dtype {tensor.type}")
  return dt


def find_classifier(model, subgraph, num_classes: int, tensor_name: str | None):
  """Locate the FULLY_CONNECTED op that produces the per-class logits."""
  fc_code = schema.BuiltinOperator.FULLY_CONNECTED
  candidates = []
  for op_idx, op in enumerate(subgraph.operators):
    code = model.operatorCodes[op.opcodeIndex]
    builtin = max(code.builtinCode, code.deprecatedBuiltinCode)
    if builtin != fc_code:
      continue
    if len(op.inputs) < 2 or op.inputs[1] < 0:
      continue
    w = subgraph.tensors[op.inputs[1]]
    if len(w.shape) != 2 or int(w.shape[CLASS_AXIS]) != num_classes:
      continue
    candidates.append((op_idx, op))

  if tensor_name:
    # Accept either the weight tensor's name or the FC output's name -- the
    # latter is what shows up in most netron/graph dumps.
    for op_idx, op in candidates:
      names = [subgraph.tensors[op.inputs[1]].name.decode()]
      names += [subgraph.tensors[o].name.decode() for o in op.outputs if o >= 0]
      if any(tensor_name in n for n in names):
        return op_idx, op
    raise ValueError(
        f"no classifier matched --tensor-name '{tensor_name}'. Candidates: "
        + ", ".join(
            subgraph.tensors[op.inputs[1]].name.decode() for _, op in candidates
        )
    )

  if not candidates:
    raise ValueError(
        f"found no FULLY_CONNECTED with a [{num_classes}, N] weight tensor; "
        "does the labels file match this model?"
    )
  if len(candidates) > 1:
    raise ValueError(
        f"found {len(candidates)} candidate classifiers; disambiguate with "
        "--tensor-name"
    )
  return candidates[0]


def resize_downstream(subgraph, start_tensor: int, num_classes: int,
                      new_count: int) -> list[int]:
  """Shrink the class axis of every tensor fed by the classifier output."""
  resized = []
  frontier = [start_tensor]
  visited = set()
  while frontier:
    t_idx = frontier.pop()
    if t_idx in visited or t_idx < 0:
      continue
    visited.add(t_idx)
    tensor = subgraph.tensors[t_idx]
    shape = list(tensor.shape)
    if not shape or int(shape[-1]) != num_classes:
      continue
    shape[-1] = new_count
    tensor.shape = np.array(shape, dtype=np.int32)
    resized.append(t_idx)
    for op in subgraph.operators:
      if t_idx in list(op.inputs):
        frontier.extend(int(o) for o in op.outputs)
  return resized


def prune(model_path: str, all_labels: list[str], keep: list[int],
          resolved: list[str], out_path: str,
          tensor_name: str | None) -> None:
  print(f"keeping {len(keep)} classes at indices {keep[:8]}"
        + (" ..." if len(keep) > 8 else ""))

  with open(model_path, "rb") as f:
    buf = bytearray(f.read())
  model = schema.ModelT.InitFromObj(schema.Model.GetRootAs(buf, 0))
  if len(model.subgraphs) != 1:
    raise ValueError(f"expected 1 subgraph, got {len(model.subgraphs)}")
  sg = model.subgraphs[0]

  num_classes = len(all_labels)
  op_idx, fc_op = find_classifier(model, sg, num_classes, tensor_name)
  w_tensor = sg.tensors[fc_op.inputs[1]]
  print(f"classifier: op[{op_idx}] weights '{w_tensor.name.decode()[:70]}' "
        f"{list(w_tensor.shape)}")

  keep_np = np.asarray(keep, dtype=np.int64)

  # --- weights -------------------------------------------------------------
  w = _buffer_array(model, w_tensor, _tflite_dtype(w_tensor))
  new_w = np.ascontiguousarray(w[keep_np, :])
  model.buffers[w_tensor.buffer].data = np.frombuffer(
      new_w.tobytes(), dtype=np.uint8
  )
  w_tensor.shape = np.array(new_w.shape, dtype=np.int32)
  print(f"  weights {tuple(w.shape)} -> {tuple(new_w.shape)} "
        f"({w.nbytes/1e6:.2f} MB -> {new_w.nbytes/1e6:.2f} MB)")

  # Per-channel weight quantization carries one scale per class.
  wq = w_tensor.quantization
  if wq is not None and wq.scale is not None and len(wq.scale) == num_classes:
    wq.scale = np.asarray(wq.scale, dtype=np.float32)[keep_np]
    if wq.zeroPoint is not None and len(wq.zeroPoint) == num_classes:
      wq.zeroPoint = np.asarray(wq.zeroPoint, dtype=np.int64)[keep_np]
    print(f"  sliced {len(wq.scale)} per-channel weight scales")

  # --- bias ----------------------------------------------------------------
  if len(fc_op.inputs) > 2 and fc_op.inputs[2] >= 0:
    b_tensor = sg.tensors[fc_op.inputs[2]]
    b = _buffer_array(model, b_tensor, _tflite_dtype(b_tensor))
    new_b = np.ascontiguousarray(b[keep_np])
    model.buffers[b_tensor.buffer].data = np.frombuffer(
        new_b.tobytes(), dtype=np.uint8
    )
    b_tensor.shape = np.array(new_b.shape, dtype=np.int32)
    print(f"  bias    {tuple(b.shape)} -> {tuple(new_b.shape)}")
    bq = b_tensor.quantization
    if bq is not None and bq.scale is not None and len(bq.scale) == num_classes:
      bq.scale = np.asarray(bq.scale, dtype=np.float32)[keep_np]
      if bq.zeroPoint is not None and len(bq.zeroPoint) == num_classes:
        bq.zeroPoint = np.asarray(bq.zeroPoint, dtype=np.int64)[keep_np]

  # --- activations downstream of the classifier ----------------------------
  resized = resize_downstream(sg, int(fc_op.outputs[0]), num_classes, len(keep))
  print(f"  resized {len(resized)} downstream tensor(s): "
        + ", ".join(sg.tensors[i].name.decode()[:28] for i in resized))

  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), TFLITE_FILE_IDENTIFIER)
  with open(out_path, "wb") as f:
    f.write(builder.Output())

  import os
  print(f"\nwrote {out_path} "
        f"({os.path.getsize(model_path)/1e6:.2f} MB -> "
        f"{os.path.getsize(out_path)/1e6:.2f} MB)")
  print("output class order:")
  for i, (src, name) in enumerate(zip(keep, resolved)):
    if i >= 10:
      print(f"  ... and {len(keep) - 10} more")
      break
    print(f"  [{i}] <- original index {src}: {name}")


def main() -> None:
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  p.add_argument("model", help="input .tflite")
  p.add_argument("labels", help="full labels.txt (one label per line)")
  p.add_argument("filtered", nargs="?", default=None,
                 help="labels.txt subset to keep (omit if using --top-n)")
  p.add_argument("-o", "--output", default=None,
                 help="output .tflite (default: <model>.pruned.tflite)")
  p.add_argument("--tensor-name", default=None,
                 help="substring of the classifier weight or output tensor name")

  g = p.add_argument_group("meta-model species selection")
  g.add_argument("-n", "--top-n", type=int, default=None,
                 help="keep the N most likely species in the region")
  g.add_argument("--min-score", type=float, default=0.0,
                 help="drop species scoring at or below this (default: 0.0). "
                      "Often more principled than a fixed N -- see --report")
  g.add_argument("--report", action="store_true",
                 help="print the score distribution and exit without pruning")
  g.add_argument("--meta-model", default=None,
                 help="meta-model.tflite (default: alongside the audio model)")
  g.add_argument("--region", default="eastern-na", choices=sorted(REGIONS),
                 help="preset bounding box (default: eastern-na)")
  g.add_argument("--bbox", type=float, nargs=4, default=None,
                 metavar=("LAT0", "LAT1", "LON0", "LON1"),
                 help="explicit bounding box, overrides --region")
  g.add_argument("--grid-step", type=float, default=1.0,
                 help="grid resolution in degrees (default: 1.0)")
  g.add_argument("--week-step", type=int, default=4,
                 help="sample every Nth week of 1..48 (default: 4)")
  g.add_argument("--reduce", choices=("max", "mean"), default="max",
                 help="reduce scores across the grid (default: max)")
  g.add_argument("--no-keep-nonbird", action="store_true",
                 help="drop Dog/Engine/Noise/etc (kept by default)")
  g.add_argument("--write-labels", default=None,
                 help="write the selected labels (default: <output>.labels.txt)")
  p.add_argument("--write-indices", default=None,
                 help="write the kept classes' ORIGINAL indices, one per line. "
                      "Lets a consumer remap full-model reference output onto "
                      "the pruned model's outputs.")

  args = p.parse_args()
  by_meta = args.top_n is not None or args.min_score > 0.0 or args.report
  if (args.filtered is None) == (not by_meta):
    p.error(
        "provide exactly one of: a filtered labels file, or --top-n / "
        "--min-score / --report"
    )

  all_labels = read_lines(args.labels)
  print(f"labels file:   {len(all_labels)} classes")
  out = args.output or args.model.replace(".tflite", "") + ".pruned.tflite"

  if args.filtered is not None:
    wanted = read_lines(args.filtered)
    print(f"filtered file: {len(wanted)} requested")
    keep, resolved = build_keep_indices(all_labels, wanted)
    labels_out = args.write_labels
  else:
    import os
    meta = args.meta_model or os.path.join(
        os.path.dirname(os.path.abspath(args.model)), "meta-model.tflite"
    )
    if not os.path.exists(meta):
      p.error(f"meta-model not found at {meta}; pass --meta-model")
    bbox = tuple(args.bbox) if args.bbox else REGIONS[args.region]
    weeks = list(range(1, 49, args.week_step))
    print(f"meta-model:    {meta}")
    print(f"region:        lat {bbox[0]}..{bbox[1]}, lon {bbox[2]}..{bbox[3]} "
          f"({'custom' if args.bbox else args.region})")
    scores = rank_species_by_meta(meta, len(all_labels), bbox, weeks,
                                  args.grid_step, args.reduce)
    if args.report:
      print(f"\nscore distribution ({args.reduce} over the region):")
      for th in (0.9, 0.7, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001):
        print(f"  species scoring > {th:<6}: {int((scores > th).sum()):5d}")
      order = np.argsort(scores)[::-1]
      print("\ntop 15:")
      for i in order[:15]:
        print(f"  {scores[i]:.4f}  {all_labels[int(i)][:52]}")
      return
    keep, resolved = select_top_n(all_labels, scores, args.top_n,
                                  args.min_score, not args.no_keep_nonbird)
    labels_out = args.write_labels or out + ".labels.txt"

  prune(args.model, all_labels, keep, resolved, out, args.tensor_name)

  if labels_out:
    with open(labels_out, "w", encoding="utf-8") as f:
      f.write("\n".join(resolved))
    print(f"wrote {labels_out} ({len(resolved)} labels, matching output order)")

  if args.write_indices:
    with open(args.write_indices, "w", encoding="utf-8") as f:
      f.write("\n".join(str(i) for i in keep))
    print(f"wrote {args.write_indices} ({len(keep)} original indices)")


if __name__ == "__main__":
  main()
