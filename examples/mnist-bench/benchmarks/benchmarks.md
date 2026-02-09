# VFPU Kernel
benchmarks-vfpu-0 shows the benchmark before we did compile-time padding; doing it gives us a ~65% improvement! (from 121ms to 44ms per image).

44ms is now the time to beat!

# MNIST CNN Benchmark Results (PSP)

Model: `mnist_cnn.tflite` — 2x Conv2d+ReLU, 2x MaxPool2d, Reshape, 2x FullyConnected
Device: Sony PSP (MIPS R4000 @ 333MHz + VFPU)
Dataset: 100 images from MNIST t10k
Accuracy: 99/100 (both configurations)

## Per-Op Comparison

| Op | Name | Shape | Naive (ms) | VFPU im2col+matmul (ms) | Delta |
|----|------|-------|------------|-------------------------|-------|
| 0 | conv2d_relu | in:[1,28,28,1] f:[8,5,5,1] → [1,28,28,8] | 4,689 | 3,078 | **-34%** |
| 1 | max_pool2d | [1,28,28,8] → [1,14,14,8] | 105 | 106 | 0% |
| 2 | conv2d_relu | in:[1,14,14,8] f:[16,5,5,8] → [1,14,14,16] | 6,655 | 9,449 | **+42%** |
| 3 | max_pool2d | [1,14,14,16] → [1,7,7,16] | 48 | 47 | 0% |
| 4 | reshape | [1,7,7,16] → [784] | 12 | 12 | 0% |
| 5 | fc_relu | 784 → 64 | 450 | 456 | 0% |
| 6 | fc | 64 → 10 | 6 | 6 | 0% |
| | **Total** | | **12,148** | **13,340** | **+10%** |
| | **Per image** | | **121 ms** | **133 ms** | |

## Analysis

- **Conv #1 (K=25)**: VFPU wins (-34%). im2col output is 784x25, matmul is 784x25x8. Small K = only 7 tile iterations per output tile. VFPU `vmmul.q` pays off.
- **Conv #2 (K=200)**: Naive wins by a lot (+42% regression). im2col produces 39,200-float scratch buffer (196x200), matmul is 196x200x16. K=200 = 50 tile iterations per output tile with heavy scalar overhead per tile.
- **Non-conv ops**: Identical between configurations (same kernels used).

## Disassembly Analysis

Full disassembly in [`disassembled-prx.txt`](disassembled-prx.txt). Built with `cargo +nightly psp --release` (optimized + debuginfo), disassembled with `psp-objdump -d -S -C -m mips:allegrex`.

### Symbol Map

| Function | Address | Size | Notes |
|----------|---------|------|-------|
| `psp_ml::kernels::relu` | 0x3780 | 340B | VFPU-accelerated (vzero.q + vmax.q) |
| `psp_ml::kernels::im2col` | 0x38d4 | 1032B | Pure scalar, 6-nested loops |
| `psp_ml::kernels::matmul_bt` | 0x3d44 | 1628B | Tiled GEMM, VFPU inner kernel |
| `psp_ml::kernels::naive::max_pool2d` | 0x4440 | 1048B | Naive scalar |
| `psp_ml::kernels::naive::fully_connected_relu` | 0x4858 | 836B | Naive scalar |

### VFPU Inner Kernel: `vfpu_mul_acc_bt` (inlined at 0x4098)

The actual VFPU block is 21 instructions — elegant and fast:

```asm
4098: lv.q  R000.q,  0(at)     ;  ─┐
409c: lv.q  R001.q, 16(at)     ;   │ Load A tile → M000
40a0: lv.q  R002.q, 32(at)     ;   │ (4 row vectors)
40a4: lv.q  R003.q, 48(at)     ;  ─┘
40a8: lv.q  R100.q,  0(v0)     ;  ─┐
40ac: lv.q  R101.q, 16(v0)     ;   │ Load B tile → M100
40b0: lv.q  R102.q, 32(v0)     ;   │ (4 row vectors)
40b4: lv.q  R103.q, 48(v0)     ;  ─┘
40b8: lv.q  R200.q,  0(v1)     ;  ─┐
40bc: lv.q  R201.q, 16(v1)     ;   │ Load accumulator → M200
40c0: lv.q  R202.q, 32(v1)     ;   │
40c4: lv.q  R203.q, 48(v1)     ;  ─┘
40c8: vmmul.q M300, M000, E100  ;  M300 = M000 @ M100^T  (E100 = transposed view)
40cc: vadd.q R200, R200, R300   ;  ─┐
40d0: vadd.q R201, R201, R301   ;   │ Accumulate: M200 += M300
40d4: vadd.q R202, R202, R302   ;   │
40d8: vadd.q R203, R203, R303   ;  ─┘
40dc: sv.q  R200.q,  0(v1)     ;  ─┐
40e0: sv.q  R201.q, 16(v1)     ;   │ Store accumulator
40e4: sv.q  R202.q, 32(v1)     ;   │
40e8: sv.q  R203.q, 48(v1)     ;  ─┘
```

### VFPU ReLU: `vfpu_relu4` (inlined at 0x380c)

```asm
380c: vzero.q R100.q            ; R100 = {0, 0, 0, 0}
3810: lv.q   R000.q, 0(s3)     ; Load 4 floats
3814: vmax.q R000, R000, R100   ; R000 = max(R000, 0)
3818: sv.q   R000.q, 0(s3)     ; Store back
```

### Why im2col+matmul_bt Is Slow for Conv #2

The disassembly reveals **massive scalar overhead** around the VFPU core:

**1. Tile zeroing via `memset` (3 calls per tile iteration)**

Each inner iteration of `matmul_bt` calls `memset` to zero three 64-byte tiles:
```asm
4020: lw    a0, 100(sp)         ; &acc
4028: li    a2, 64
402c: jal   memset              ; Zero 64B accumulator tile
...
4118: lw    a0, 92(sp)          ; &b_tile
4120: li    a2, 64
4128: jal   memset              ; Zero 64B B tile
...
4134: li    a2, 64
413c: jal   memset              ; Zero 64B A tile
4140: move  a0, s7
```

For Conv #2: `49 × 4 × 50 = 9,800` tile iterations → **29,400 `memset` calls**

**2. Element-by-element `load_tile` with bounds checks**

Each tile load is a scalar loop with per-element bounds checking:
```asm
; load_tile inner loop (address 0x41a8):
41a8: sltu  at, a0, s2          ; Bounds check: row+r < max_rows?
41ac: beqz  at, 43d4            ; Branch if out of bounds
41b4: sltiu at, v0, 16          ; Bounds check: r*4+c < 16?
41b8: beqz  at, 43e4            ; Branch if out of bounds
41c0: lwc1  $f0, 0(t3)          ; Load one float
41d0: swc1  $f0, 0(t4)          ; Store to tile buffer
41d4: [loop control]
```

Two branches per float loaded. For a full 4×4 tile: **32 branches, 16 scalar loads**.

**3. Element-by-element `store_tile` (same pattern)**

```asm
; store_tile inner loop (address 0x3f04):
3f04: sltiu at, a0, 16          ; Bounds check tile index
3f08: beqz  at, 441c            ; Branch on overflow
3f10: sltu  at, v0, s0          ; Bounds check output index
3f14: beqz  at, 442c            ; Branch on overflow
3f1c: lwc1  $f0, 0(t2)          ; Load from tile
3f34: swc1  $f0, 0(t3)          ; Store to output
```

**4. im2col: Pure scalar with expensive index math**

Each element in the im2col output requires:
- Two `mult` instructions for multi-dimensional index computation (30+ cycle latency each on R4000)
- Conditional branches for padding checks on every element
- For Conv #2: 196 × 200 = **39,200 elements**, each with address arithmetic

```asm
; im2col inner loop (0x3b70):
3b70: sltu  at, t0, a1         ; Bounds check input index
3b74: beqz  at, 3d20           ; → panic_bounds_check
3b7c: sltu  at, a3, v0         ; Bounds check output index
3b80: beqz  at, 3d30           ; → panic_bounds_check
3b88: lwc1  $f0, 0(t7)         ; Load input element
3b9c: swc1  $f0, 0(t8)         ; Store to column matrix
```

### Cost Breakdown Estimate for Conv #2 (per image)

| Phase | Work | Est. Cycles | Notes |
|-------|------|-------------|-------|
| im2col | 39,200 elements × ~20 cyc | ~784K | Index math + branches |
| memset | 29,400 × ~64 cyc | ~1,882K | 3 tiles/iteration × 9,800 iterations |
| load_tile | 19,600 × ~40 cyc | ~784K | 2 tiles × 16 elements × bounds checks |
| **vmmul.q** | **9,800 × ~40 cyc** | **~392K** | **The actual VFPU work** |
| store_tile | 9,800 × ~40 cyc | ~392K | 16 elements × bounds checks |
| **Total** | | **~4,234K** | ~12.7ms at 333MHz |

The VFPU multiply (`vmmul.q` + `vadd.q`) is only **~9%** of the estimated matmul_bt cycle count. The scalar tile management dominates.

### Naive Conv2d for Comparison

The naive `conv2d_relu` at 0x3874 is a compact 172-byte function that directly computes the convolution with 7 nested loops and no intermediate buffer. It processes each output element with a tight FPU inner loop:
```asm
; naive conv2d inner accumulation:
38e0: lwc1  $f2, 0(v0)         ; Load weight
38e4: lwc1  $f0, 0(at)         ; Load input
38e8: mul.s $f0, $f2, $f0      ; Multiply
38ec: add.s $f4, $f4, $f0      ; Accumulate
38f0: [loop back]
```

No tile overhead, no `memset`, no bounds checks on tiles. The scalar FPU multiply-accumulate is 2 cycles/element in the inner loop.

### Optimization Opportunities

1. **Eliminate `memset` calls**: Pre-zero the accumulator tile with `vmzero.q` in the VFPU block instead of calling `memset`. Use `vmzero.q M200` before the tile loop.
2. **VFPU-accelerated `load_tile`/`store_tile`**: Use `lv.q`/`sv.q` for aligned 4-element loads instead of scalar loops. For boundary tiles, use a pre-zeroed buffer and conditionally copy.
3. **Remove per-element bounds checks**: The tile dimensions are computed from known matrix dimensions — the bounds are provable at codegen time. Use unchecked indexing.
4. **Fuse im2col into matmul**: Instead of materializing the full im2col buffer, compute patches on-the-fly into tiles within the matmul loop ("implicit GEMM"). This eliminates the 39,200-float scratch buffer and the separate im2col pass.
5. **VFPU-accelerate im2col**: For the non-padding case, use `lv.q` to load 4 contiguous input channels at once instead of scalar element-by-element copy.
6. **Fuse bias_add + relu into matmul**: Avoid a second pass over the output buffer by applying bias and ReLU as each output tile is stored.

## Raw Data

### Naive conv2d

```json
{
  "model": "mnist_cnn",
  "config": {
    "kernel_type": "naive"
  },
  "inference": {
    "num_images": 100,
    "total_us": 12148111,
    "per_image_us": 121481,
    "correct": 99,
    "total": 100
  },
  "ops": [
    { "index": 0, "name": "conv2d_relu", "total_us": 4689476, "calls": 100 },
    { "index": 1, "name": "max_pool2d", "total_us": 105078, "calls": 100 },
    { "index": 2, "name": "conv2d_relu", "total_us": 6654608, "calls": 100 },
    { "index": 3, "name": "max_pool2d", "total_us": 47973, "calls": 100 },
    { "index": 4, "name": "reshape", "total_us": 11647, "calls": 100 },
    { "index": 5, "name": "fully_connected_relu", "total_us": 449506, "calls": 100 },
    { "index": 6, "name": "fully_connected", "total_us": 5644, "calls": 100 }
  ]
}
```

### VFPU im2col + matmul_bt (V0)

```json
{
  "model": "mnist_cnn",
  "config": {
    "kernel_type": "vfpu"
  },
  "inference": {
    "num_images": 100,
    "total_us": 13339871,
    "per_image_us": 133398,
    "correct": 99,
    "total": 100
  },
  "ops": [
    { "index": 0, "name": "conv2d_relu", "total_us": 3078070, "calls": 100 },
    { "index": 1, "name": "max_pool2d", "total_us": 106173, "calls": 100 },
    { "index": 2, "name": "conv2d_relu", "total_us": 9448667, "calls": 100 },
    { "index": 3, "name": "max_pool2d", "total_us": 47369, "calls": 100 },
    { "index": 4, "name": "reshape", "total_us": 11716, "calls": 100 },
    { "index": 5, "name": "fully_connected_relu", "total_us": 455537, "calls": 100 },
    { "index": 6, "name": "fully_connected", "total_us": 5822, "calls": 100 }
  ]
}
```

---

# V1: Compile-Time Padding + `matmul_bt_tiled`

## What Changed

Replaced `im2col` + `matmul_bt` with `im2col_padded` + `matmul_bt_tiled`:

1. **Compile-time K-padding**: Weight rows with K not a multiple of VFPU_Q (4) are zero-padded at codegen time via `static mut` buffers + copy loops. `im2col_padded` outputs K-padded columns. Weight blob stays unmodified.
   - Conv #1: K=25 → K_padded=28 (padded weight buffer: 8×28 = 224 floats)
   - Conv #2: K=200 (already aligned, no padding needed)

2. **Tile-count-based API**: `matmul_bt_tiled(a, b, c, m_tiles, k_tiles, n_tiles)` takes pre-computed tile counts instead of raw dimensions. All dimensions guaranteed multiples of VFPU_Q.

3. **`load_tile_direct` / `store_tile_direct`**: Row-wise `copy_from_slice` instead of element-by-element with per-element bounds checks. No conditional zeroing — all 16 elements always written.

4. **One acc zeroing per output tile**: Accumulator `memset` moved outside the K-tile loop. Zeroed once per (ti, tj) pair, not once per (ti, tj, tk) iteration.

5. **`im2col_padded`**: Stride-1 only (stride support removed), K-padded output columns. Codegen asserts `stride == [1,1]`.

## Per-Op Comparison (3-Way)

| Op | Name | Naive (ms) | V0: matmul_bt (ms) | V1: matmul_bt_tiled (ms) | V1 vs Naive | V1 vs V0 |
|----|------|------------|---------------------|--------------------------|-------------|----------|
| 0 | conv2d_relu | 4,689 | 3,078 | 1,182 | **-75%** | **-62%** |
| 1 | max_pool2d | 105 | 106 | 105 | 0% | 0% |
| 2 | conv2d_relu | 6,655 | 9,449 | 2,469 | **-63%** | **-74%** |
| 3 | max_pool2d | 48 | 47 | 48 | 0% | 0% |
| 4 | reshape | 12 | 12 | 12 | 0% | 0% |
| 5 | fc_relu | 450 | 456 | 450 | 0% | 0% |
| 6 | fc | 6 | 6 | 6 | 0% | 0% |
| | **Total** | **12,148** | **13,340** | **4,456** | **-63%** | **-67%** |
| | **Per image** | **121 ms** | **133 ms** | **44.6 ms** | | |

Conv #2 went from **+42% regression** (V0 vs naive) to **-63% improvement** (V1 vs naive). The VFPU path now decisively beats naive for both convolutions.

## V1 Disassembly Analysis

Full disassembly in [`disassembled-prx-tiled.txt`](disassembled-prx-tiled.txt).

### Symbol Map

| Function | Address | Size | Size Delta vs V0 |
|----------|---------|------|-------------------|
| `psp_ml::kernels::relu` | 0x4518 | 340B | (unchanged) |
| `psp_ml::kernels::im2col_padded` | 0x37d4 | 1124B | +9% (was 1032B) |
| `psp_ml::kernels::matmul_bt_tiled` | 0x3c38 | 2272B | +39% (was 1628B) |

Code size increase (+736B total) is a good trade — the extra code eliminates per-element branches in the hot inner loop.

### Inner Loop Structure: `matmul_bt_tiled`

The K-tile inner loop runs from 0x3e18 to 0x4160 (~210 instructions per iteration):

**1. `load_tile_direct` for A tile (4 rows)**

Each row: one `sltu + beqz` bounds check (from `copy_from_slice`), then `lw×4` + `sw×4` to copy 4 floats to the aligned stack buffer.

```asm
; Row 0 of A tile (representative pattern):
3e24: sltu  at, a0+3, s2      ; bounds check: start+4 <= a.len()?
3e28: beqz  at, panic          ; branch on failure
3e34: lw    a0, 0(at)          ; ─┐
3e38: lw    a2, 4(at)          ;  │ Load 4 floats from source
3e3c: lw    a3, 8(at)          ;  │ (scalar lw, not lv.q)
3e40: lw    at, 12(at)         ; ─┘
3e4c: sw    a0, 368(sp)        ; ─┐
3e64: sw    a2, 372(sp)        ;  │ Store to aligned stack slot
3e50: sw    a3, 376(sp)        ;  │
3e44: sw    at, 380(sp)        ; ─┘
```

4 rows × (1 bounds check branch) = **4 branches per A tile load**.

**2. `load_tile_direct` for B tile (same pattern)**

**4 branches per B tile load**.

**3. Register shuffling**

~60 instructions copying data between stack slots to prepare aligned buffers for VFPU loads. This is compiler-generated spill/reload due to heavy register pressure (480-byte stack frame, all 10 s-registers used).

**4. VFPU block (0x410c–0x415c)**

Same 21 instructions as V0 — the VFPU core is unchanged:

```asm
410c: lv.q  R000.q, 0(at)      ; ─┐ Load A tile → M000
4110: lv.q  R001.q, 16(at)     ;  │
4114: lv.q  R002.q, 32(at)     ;  │
4118: lv.q  R003.q, 48(at)     ; ─┘
411c: lv.q  R100.q, 0(a0)      ; ─┐ Load B tile → M100
4120: lv.q  R101.q, 16(a0)     ;  │
4124: lv.q  R102.q, 32(a0)     ;  │
4128: lv.q  R103.q, 48(a0)     ; ─┘
412c: lv.q  R200.q, 0(s4)      ; ─┐ Load acc → M200
4130: lv.q  R201.q, 16(s4)     ;  │
4134: lv.q  R202.q, 32(s4)     ;  │
4138: lv.q  R203.q, 48(s4)     ; ─┘
413c: vmmul.q M300, M000, E100  ; M300 = A @ B^T
4140: vadd.q R200, R200, R300   ; ─┐
4144: vadd.q R201, R201, R301   ;  │ acc += result
4148: vadd.q R202, R202, R302   ;  │
414c: vadd.q R203, R203, R303   ; ─┘
4150: sv.q  R200.q, 0(s4)      ; ─┐ Store acc
4154: sv.q  R201.q, 16(s4)     ;  │
4158: sv.q  R202.q, 32(s4)     ;  │
415c: sv.q  R203.q, 48(s4)     ; ─┘
```

**5. Loop back**

```asm
4160: bnez  v1, 3e18            ; 1 branch
```

**6. Accumulator zeroing (once per output tile, outside K loop)**

```asm
3df8: jal   memset              ; Zero 64B acc tile
3dfc: move  a0, s4              ; (called ONCE per output tile)
```

### Overhead Comparison: V0 vs V1

#### Per K-tile iteration (Conv #2: 9,800 iterations)

| Metric | V0: `matmul_bt` | V1: `matmul_bt_tiled` | Reduction |
|--------|-----------------|----------------------|-----------|
| `memset` calls | 3 (a_tile + b_tile + acc) | 0 | eliminated |
| Branches | ~68 (2 per element × 16 × 2 tiles + loop) | 9 (1 per row × 4 × 2 tiles + loop) | **7.5×** |
| Tile load style | Element-by-element scalar | Row-wise `lw×4 + sw×4` | 4× fewer loads |

#### Per output tile (Conv #2: 196 tiles)

| Metric | V0 | V1 | Reduction |
|--------|----|----|-----------|
| `memset` calls | 0 (was in K loop) | 1 (acc only) | n/a |
| Store branches | ~32 (2 per element × 16) | 4 (1 per row) | **8×** |

#### Totals for Conv #2 (per image)

| Metric | V0 | V1 | Reduction |
|--------|----|----|-----------|
| `memset` calls | 29,400 | 196 | **150×** |
| Branches | ~673K | ~89K | **7.5×** |
| VFPU blocks | 9,800 | 9,800 | (same) |

### Remaining Overhead

The VFPU block (21 instructions) is still only ~10% of the ~210 instructions per K-tile iteration. The remaining 90% is:

1. **`copy_from_slice` bounds checks** (8 branches/iter): Unnecessary since all dimensions are guaranteed VFPU_Q-aligned by codegen. Could eliminate with `unsafe` `get_unchecked` or `ptr::copy_nonoverlapping`.

2. **Scalar tile data path**: Data flows memory → `lw×4` → stack → `lv.q` → VFPU. Two indirections because the compiler doesn't know the source data is 16-byte aligned. If source pointers were guaranteed aligned, `lv.q` could load directly from the matrix buffers.

3. **Register pressure / spills**: 480-byte stack frame with all 10 s-registers used. The compiler spills heavily between inner loop iterations (~60 `lw`/`sw` for register shuffling per iteration).

### Optimization Opportunities (updated from V0)

1. ~~Eliminate `memset` calls~~ — **Done** (moved outside K-loop, 150× reduction)
2. ~~Remove per-element bounds checks~~ — **Mostly done** (element→row granularity, 7.5× reduction). Remaining: use `get_unchecked` to eliminate row-level bounds checks entirely.
3. **Direct VFPU loads from matrix buffers**: Ensure 16-byte alignment on im2col output and weight buffers so `lv.q` can load directly, bypassing the stack staging area. This would eliminate ~120 `lw`/`sw` per K-tile iteration.
4. **Fuse im2col into matmul** (implicit GEMM): Still applicable — eliminates scratch buffer and separate im2col pass.
5. **Fuse bias_add + relu into matmul**: Still applicable — eliminates output buffer re-traversal.

## V1 Raw Data

### VFPU im2col_padded + matmul_bt_tiled (V1)

```json
{
  "model": "mnist_cnn",
  "config": {
    "kernel_type": "vfpu"
  },
  "inference": {
    "num_images": 100,
    "total_us": 4456244,
    "per_image_us": 44562,
    "correct": 99,
    "total": 100
  },
  "ops": [
    { "index": 0, "name": "conv2d_relu", "total_us": 1181787, "calls": 100 },
    { "index": 1, "name": "max_pool2d", "total_us": 104590, "calls": 100 },
    { "index": 2, "name": "conv2d_relu", "total_us": 2469376, "calls": 100 },
    { "index": 3, "name": "max_pool2d", "total_us": 47873, "calls": 100 },
    { "index": 4, "name": "reshape", "total_us": 11905, "calls": 100 },
    { "index": 5, "name": "fully_connected_relu", "total_us": 450345, "calls": 100 },
    { "index": 6, "name": "fully_connected", "total_us": 5959, "calls": 100 }
  ]
}
```
