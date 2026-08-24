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
"""Slice BirdNET's 2-branch STFT frontend into its own .tflite, plus fixtures.

The frontend is everything from the normalised signal to the two
RFFT2D -> SQUEEZE -> CAST(complex->f32) outputs::

    norm [1,144000] --(framing arithmetic + GATHER + window MUL + RFFT2D)-->
        [511, 1025]   (L=2048 branch)
        [511,  513]   (L=1024 branch)

The slice's input is the *normalised* signal (the output of the leading
REDUCE_MIN/SUB/REDUCE_MAX/ADD/DIV/SUB/MUL chain), so the benchmark measures
the STFT alone and every consumer — dense-gather compile, strided-view
compile, and the TFLite golden — sees bit-identical input bytes.

Also slices the mel projection that follows each branch (FC by the mel
filterbank matrix, then MUL(x,x) + POW) into per-branch models whose input
is that branch's STFT output — golden_<L>.bin is bit-exactly their input.

Writes into models/birdnet/stft/ (gitignored, regenerated on demand):
  frontend.tflite              the sliced STFT model (dense-gather semantics)
  window_2048.bin/1024.bin     the hann window constants, f32
  samples.bin                  normalised cardinal_3s.wav samples, f32[144000]
  golden_2048.bin/1024.bin     TFLite's own frontend outputs for samples.bin
  mel_2048.tflite/1024.tflite  per-branch mel slices (dense-FC semantics)
  mel_dense_2048.bin/1024.bin  the stored [96, bins] mel matrices, row-major
  golden_mel_2048.bin/1024.bin TFLite's own mel outputs ([511, 96], post-POW)
  manifest.json                shapes + the pow exponent, read by build.rs

Run from the repo root::

    uv run examples/birdnet-stft-benchmark/slice_stft.py

`examples/birdnet-stft-benchmark/device/build.rs` invokes this automatically
when the outputs are missing.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

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

try:
  from ai_edge_litert.interpreter import Interpreter
except ImportError:  # pragma: no cover
  from tensorflow.lite.python.interpreter import Interpreter

TFLITE_FILE_IDENTIFIER = b"TFL3"

REPO = Path(__file__).resolve().parent.parent.parent
MODEL = REPO / "models/birdnet/audio-model-int8.tflite"
WAV = REPO / "examples/birdnet/cardinal_3s.wav"
OUT_DIR = REPO / "models/birdnet/stft"

BO = schema.BuiltinOperator


def opcode(model, op) -> int:
  """Builtin code of an operator, tolerating the pre-2.4 schema split."""
  oc = model.operatorCodes[op.opcodeIndex]
  return max(oc.builtinCode, oc.deprecatedBuiltinCode)


def buffer_data(model, sg, tensor_idx) -> np.ndarray | None:
  t = sg.tensors[tensor_idx]
  buf = model.buffers[t.buffer]
  if buf.data is None or len(buf.data) == 0:
    return None
  return np.frombuffer(bytes(buf.data), dtype=np.float32)


def find_frontend(model, sg):
  """Locate both STFT branches and the shared normalised-signal tensor.

  Returns (norm_tensor, branches) with branches ordered L=2048 first; each
  branch is (fft_length, window_tensor_idx, cast_output_tensor_idx).
  """
  producer = {}
  consumers = {}
  for op_idx, op in enumerate(sg.operators):
    for t in op.outputs:
      if t >= 0:
        producer[t] = op_idx
    for t in op.inputs:
      if t >= 0:
        consumers.setdefault(t, []).append(op_idx)

  branches = []
  for op_idx, op in enumerate(sg.operators):
    if opcode(model, op) != BO.RFFT2D:
      continue
    # fft_length is the second input, a constant [2] of (1, L)
    fft_len_const = buffer_data(model, sg, op.inputs[1])
    fft_length = int(
        np.frombuffer(
            bytes(model.buffers[sg.tensors[op.inputs[1]].buffer].data),
            dtype=np.int32,
        )[-1]
    )

    # Forward: RFFT2D -> (SQUEEZE) -> CAST(c64->f32). The CAST output is the
    # branch's frontend output.
    t = op.outputs[0]
    cast_out = None
    while cast_out is None:
      nxt = [sg.operators[c] for c in consumers.get(t, [])]
      if not nxt:
        raise ValueError(f"RFFT2D at op {op_idx}: no CAST downstream")
      follow = nxt[0]
      code = opcode(model, follow)
      if code == BO.CAST:
        cast_out = follow.outputs[0]
      elif code in (BO.SQUEEZE, BO.RESHAPE):
        t = follow.outputs[0]
      else:
        raise ValueError(
            f"RFFT2D at op {op_idx}: unexpected op {code} downstream"
        )

    # Backward: RFFT2D <- EXPAND_DIMS <- MUL(x, window[L]). The constant
    # operand of that MUL is the hann window.
    t = op.inputs[0]
    window_idx = None
    while window_idx is None:
      prev = sg.operators[producer[t]]
      code = opcode(model, prev)
      if code == BO.MUL:
        consts = [i for i in prev.inputs if buffer_data(model, sg, i) is not None]
        if len(consts) != 1 or buffer_data(model, sg, consts[0]).size != fft_length:
          raise ValueError(f"MUL feeding RFFT2D at op {op_idx} is not the window")
        window_idx = consts[0]
      elif code in (BO.EXPAND_DIMS, BO.RESHAPE, BO.SQUEEZE):
        t = prev.inputs[0]
      else:
        raise ValueError(
            f"RFFT2D at op {op_idx}: unexpected op {code} upstream of window"
        )

    branches.append((fft_length, window_idx, cast_out))

  if len(branches) != 2:
    raise ValueError(f"expected 2 RFFT2D branches, found {len(branches)}")
  branches.sort(key=lambda b: -b[0])  # L=2048 first

  # The normalised signal: both branches' framing starts with a STRIDED_SLICE
  # over the same [1, 144000] f32 tensor.
  slice_inputs = {
      op.inputs[0]
      for op in sg.operators
      if opcode(model, op) == BO.STRIDED_SLICE
      and sg.tensors[op.inputs[0]].type == schema.TensorType.FLOAT32
  }
  if len(slice_inputs) != 1:
    raise ValueError(
        f"expected one shared STRIDED_SLICE source, found {sorted(slice_inputs)}"
    )
  (norm,) = slice_inputs
  return norm, branches


def backward_slice(model, sg, outputs, stop_tensor):
  """Op indices (original order) needed for `outputs`, halting at `stop_tensor`."""
  producer = {}
  for op_idx, op in enumerate(sg.operators):
    for t in op.outputs:
      if t >= 0:
        producer[t] = op_idx

  keep_ops = set()
  visited = set()
  stack = list(outputs)
  while stack:
    t = stack.pop()
    if t in visited or t == stop_tensor:
      continue
    visited.add(t)
    op_idx = producer.get(t)
    if op_idx is None:
      continue  # constant or graph input
    keep_ops.add(op_idx)
    stack.extend(i for i in sg.operators[op_idx].inputs if i >= 0)
  return sorted(keep_ops)


def slice_subgraph(buf: bytes, input_tensor: int, outputs: list[int],
                   input_shape: list[int] | None = None) -> bytes:
  """Cut the subgraph feeding `outputs` out of the model, with `input_tensor`
  as its sole input.

  `input_shape` overrides the input tensor's stored shape — needed when the
  stored static shape is the single-frame lie (e.g. the CAST outputs say
  [1, 1, 1025] but run as [1, 511, 1025]); downstream SHAPE arithmetic and
  psp-tc's inference both derive everything from the input's shape, so the
  slice must declare the runtime truth.

  Parses its own ModelT: slicing mutates ops/tensors in place, so each slice
  needs a fresh object tree.
  """
  model = schema.ModelT.InitFromObj(schema.Model.GetRootAs(buf, 0))
  if len(model.subgraphs) != 1:
    raise ValueError(f"expected 1 subgraph, got {len(model.subgraphs)}")
  sg = model.subgraphs[0]

  keep_ops = backward_slice(model, sg, outputs, input_tensor)

  # The framing arithmetic reads the shape of the *graph input* via SHAPE ops;
  # after the slice the new input tensor plays that role, so redirect any
  # remaining reads of the old graph input.
  old_input = sg.inputs[0]
  for op_idx in keep_ops:
    op = sg.operators[op_idx]
    op.inputs = [input_tensor if i == old_input else i for i in op.inputs]

  if input_shape is not None:
    sg.tensors[input_tensor].shape = list(input_shape)

  # Tensor + buffer remap. Buffer 0 stays the canonical empty buffer.
  kept_tensors = [input_tensor]
  seen = {input_tensor}
  for op_idx in keep_ops:
    op = sg.operators[op_idx]
    for t in list(op.inputs) + list(op.outputs):
      if t >= 0 and t not in seen:
        seen.add(t)
        kept_tensors.append(t)

  tensor_map = {old: new for new, old in enumerate(kept_tensors)}
  new_tensors = []
  new_buffers = [model.buffers[0]]
  for old in kept_tensors:
    t = sg.tensors[old]
    buf_obj = model.buffers[t.buffer]
    if t.buffer != 0 and buf_obj.data is not None and len(buf_obj.data) > 0:
      t.buffer = len(new_buffers)
      new_buffers.append(buf_obj)
    else:
      t.buffer = 0
    new_tensors.append(t)

  new_ops = []
  for op_idx in keep_ops:
    op = sg.operators[op_idx]
    op.inputs = [tensor_map[i] if i >= 0 else i for i in op.inputs]
    op.outputs = [tensor_map[i] if i >= 0 else i for i in op.outputs]
    new_ops.append(op)

  sg.tensors = new_tensors
  sg.operators = new_ops
  sg.inputs = [tensor_map[input_tensor]]
  sg.outputs = [tensor_map[t] for t in outputs]
  model.buffers = new_buffers
  # Metadata and signatures index into the old tensor/buffer tables.
  model.metadata = []
  model.metadataBuffer = []
  model.signatureDefs = []

  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), TFLITE_FILE_IDENTIFIER)
  return bytes(builder.Output())


def find_mel(model, sg, cast_out: int):
  """From a branch's CAST output, locate the mel projection downstream:
  CAST -> (shape arithmetic + RESHAPE) -> FULLY_CONNECTED [96, bins]
       -> RESHAPE -> MUL(x, x) -> POW(x, const).

  Returns (pow_output_tensor, fc_weights_tensor, pow_exponent).
  """
  consumers = {}
  for op_idx, op in enumerate(sg.operators):
    for t in op.inputs:
      if t >= 0:
        consumers.setdefault(t, []).append(op_idx)

  # Follow the data path (not the SHAPE arithmetic) to the FC.
  t = cast_out
  fc = None
  while fc is None:
    data_ops = [
        sg.operators[c]
        for c in consumers.get(t, [])
        if opcode(model, sg.operators[c]) in (BO.RESHAPE, BO.FULLY_CONNECTED)
    ]
    if not data_ops:
      raise ValueError(f"no FC downstream of t{cast_out}")
    follow = data_ops[0]
    if opcode(model, follow) == BO.FULLY_CONNECTED:
      fc = follow
    else:
      t = follow.outputs[0]

  weights_idx = fc.inputs[1]
  # FC -> RESHAPE -> MUL -> POW
  t = fc.outputs[0]
  pow_op = None
  while pow_op is None:
    nxt = sg.operators[consumers[t][0]]
    code = opcode(model, nxt)
    if code == BO.POW:
      pow_op = nxt
    elif code in (BO.RESHAPE, BO.MUL):
      t = nxt.outputs[0]
    else:
      raise ValueError(f"unexpected op {code} between FC and POW")
  exponent = float(buffer_data(model, sg, pow_op.inputs[1])[0])
  return pow_op.outputs[0], weights_idx, exponent


def load_wav(path: Path) -> np.ndarray:
  with wave.open(str(path), "rb") as w:
    assert w.getnchannels() == 1 and w.getsampwidth() == 2
    raw = w.readframes(w.getnframes())
  samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
  assert samples.size == 144000, f"expected 144000 samples, got {samples.size}"
  return samples


def main() -> None:
  buf = MODEL.read_bytes()
  OUT_DIR.mkdir(parents=True, exist_ok=True)

  # Window constants come from the *original* model (tensor indices there).
  model = schema.ModelT.InitFromObj(schema.Model.GetRootAs(buf, 0))
  sg = model.subgraphs[0]
  norm, branches = find_frontend(model, sg)
  for fft_length, window_idx, _ in branches:
    window = buffer_data(model, sg, window_idx)
    (OUT_DIR / f"window_{fft_length}.bin").write_bytes(window.tobytes())
    print(f"window_{fft_length}.bin: {window.size} floats")

  # Golden: run the FULL model, tapping the normalised input and both branch
  # outputs. Anchored to the verified cardinal fixture.
  interp = Interpreter(
      model_content=buf, experimental_preserve_all_tensors=True
  )
  interp.allocate_tensors()
  (inp,) = interp.get_input_details()
  samples = load_wav(WAV)
  interp.set_tensor(inp["index"], samples[np.newaxis, :])
  interp.invoke()

  norm_vals = interp.get_tensor(norm).reshape(-1).astype(np.float32)
  (OUT_DIR / "samples.bin").write_bytes(norm_vals.tobytes())
  goldens = {}
  n_windows = None
  for fft_length, _, cast_out in branches:
    vals = interp.get_tensor(cast_out)
    vals = vals.reshape(vals.shape[-2], vals.shape[-1]).astype(np.float32)
    n_windows = vals.shape[0]
    assert vals.shape[1] == fft_length // 2 + 1, vals.shape
    (OUT_DIR / f"golden_{fft_length}.bin").write_bytes(vals.tobytes())
    goldens[fft_length] = vals
    print(f"golden_{fft_length}.bin: {vals.shape}")

  # The STFT slice itself.
  sliced = slice_subgraph(buf, norm, [cast_out for _, _, cast_out in branches])
  (OUT_DIR / "frontend.tflite").write_bytes(sliced)
  print(f"frontend.tflite: {len(sliced)} bytes (from {len(buf)})")

  # Sanity: the sliced model on samples.bin must reproduce the full model's
  # taps exactly — same ops, same order, nothing lost in the slice.
  check = Interpreter(model_content=sliced)
  check.allocate_tensors()
  (cin,) = check.get_input_details()
  check.set_tensor(cin["index"], norm_vals.reshape(1, -1))
  check.invoke()
  outs = check.get_output_details()
  assert len(outs) == 2, [o["shape"] for o in outs]
  for o in outs:
    vals = check.get_tensor(o["index"])
    vals = vals.reshape(vals.shape[-2], vals.shape[-1])
    fft_length = (vals.shape[1] - 1) * 2
    if not np.array_equal(vals, goldens[fft_length]):
      raise AssertionError(f"sliced model diverges from full model (L={fft_length})")
  print("sliced model matches full-model taps exactly")

  # ── Mel projection slices, one per branch ────────────────────────────
  # CAST(f32 bins) -> FC [96, bins] (the mel filterbank as a dense matmul)
  # -> MUL(x,x) -> POW. The slice's input is the branch's STFT output with
  # its runtime shape declared, so golden_<L>.bin is exactly its input.
  pow_exponents = {}
  for fft_length, _, cast_out in branches:
    bins = fft_length // 2 + 1
    pow_out, weights_idx, exponent = find_mel(model, sg, cast_out)
    pow_exponents[fft_length] = exponent

    weights = buffer_data(model, sg, weights_idx)
    assert weights.size == 96 * bins
    (OUT_DIR / f"mel_dense_{fft_length}.bin").write_bytes(weights.tobytes())

    golden_mel = interp.get_tensor(pow_out)
    golden_mel = golden_mel.reshape(golden_mel.shape[-2], -1).astype(np.float32)
    assert golden_mel.shape == (n_windows, 96), golden_mel.shape
    (OUT_DIR / f"golden_mel_{fft_length}.bin").write_bytes(golden_mel.tobytes())

    mel_sliced = slice_subgraph(
        buf, cast_out, [pow_out], input_shape=[1, int(n_windows), bins]
    )
    (OUT_DIR / f"mel_{fft_length}.tflite").write_bytes(mel_sliced)
    print(f"mel_{fft_length}.tflite: {len(mel_sliced)} bytes, pow={exponent}")

    check = Interpreter(model_content=mel_sliced)
    check.allocate_tensors()
    (cin,) = check.get_input_details()
    check.set_tensor(cin["index"], goldens[fft_length].reshape(1, n_windows, bins))
    check.invoke()
    (out,) = check.get_output_details()
    vals = check.get_tensor(out["index"]).reshape(n_windows, -1)
    if not np.array_equal(vals, golden_mel):
      raise AssertionError(f"mel slice diverges from full model (L={fft_length})")
    print(f"mel_{fft_length} slice matches full-model taps exactly")

  manifest = {
      "n_samples": 144000,
      "n_windows": int(n_windows),
      "branches": [
          {
              "fft_length": int(l),
              "bins": int(l) // 2 + 1,
              "pow_exponent": pow_exponents[l],
          }
          for l, _, _ in branches
      ],
  }
  (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"wrote {OUT_DIR}/manifest.json")


if __name__ == "__main__":
  main()
