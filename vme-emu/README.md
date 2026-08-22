# vme-emu — Verilog RTL model of the PSP's Virtual Mobile Engine

A register-transfer-level implementation of the VME array described by
`docs/vme-reference.html`: four processing elements (each with two functional
units and three address generators), the eight 8 KB ring buffers in their TOP
and BASE banks, the staging bus, and the interconnect registers — driven by
the same 106-word context image the real hardware consumes. Initialise the
context window through the memory-mapped host port, write `TRIGGER` to
`DMA_CTRL`, and data streams through the configured graph into the BASE
buffers, one element per clock per port, with no interlocks.

(The context register file is 106 mapped words, per Table 3.1/3.2 of the
manual — words 0–104 plus `CTX_END`; the 112-word figure is the padded
memory-resident image.)

## Layout

| File | Contents |
| --- | --- |
| `rtl/vme_top.v` | Context register file, interconnect (CFGMAP/SRCMAP/SKEW/FU1EN), 8 ring buffers, trigger/done, host port on the documented memory map |
| `rtl/vme_pe.v` | One PE: 3 AGUs, 4 operand muxes, FU0/FU1, write port with the DRAIN delay |
| `rtl/vme_fu.v` | One functional unit: the complete Appendix C opcode table, ACC/SAT/R/K post-processing, stream state |
| `rtl/vme_agu.v` | One AGU: skew hold, counters A/B, REV/replicate/bit-reverse transforms |
| `rtl/vme_operand.v` | One operand path: the 16-way FSEL/BSEL mux + ICN_SKEW delay |
| `rtl/vme_ringbuf.v` | 2048×32 buffer RAM (mirror = mod-2048 addressing) |
| `tb/tb_vme.v` | Self-checking testbench, 7 contexts / 112 checks |
| `driver/main.cpp` | The `vme-emu` host driver: a Verilated `vme_top` executing machine images |

## Running

The testbench needs Icarus Verilog (`apt install iverilog`); the driver
needs Verilator:

```bash
make test              # iverilog + vvp; prints ALL n CHECKS PASSED
make build/vme-emu     # verilate the RTL and build the host driver
build/vme-emu in.bin out.bin
```

## Machine images and the driver

`vme-emu <input-image> <output-image>` executes one machine image: a file of
exactly 1 MB mirroring the VME address space `0x4400_0000–0x440F_FFFF`
(byte offset = address − `0x4400_0000`, words little-endian):

| Offset | Region |
| --- | --- |
| `0x00000` | `BASE_0..3`, 8 KB each |
| `0x20000` | `TOP_0..3` |
| `0xF8000` | the 106 context words |
| `0xFF000` | `DMA_STAT` (meaningful in the *output* image: final status) |

The driver performs the library bring-up sequence, minus what has no
meaning in simulation: the bus-clock enables live at `0xBC10_0050` outside
this address space (symbolic), buffers arrive pre-staged in the image (no
`LOAD` DMA), and the context goes in through the live context window with a
readback verify (the section 11.3 fine path) rather than `CTXLOAD`. It then
writes TRIGGER, clocks until `DMA_STAT[11]` (VD), and writes the post-run
buffers + context + status to the output image. A full-hierarchy VCD always
lands at `<output-image>.vcd`. If VD never sets within `VME_EMU_MAX_CYCLES`
(a compile-time cap in `driver/main.cpp`), the driver exits nonzero — the
VCD is still written.

Images are built by hand or, much more comfortably, by the
`vme-assembler` crate in `psp-tc/vme-assembler/`, which also derives the
cycle skews from the timing model below.

The testbench drives the array exactly the way `me-core-lib` drives the real
one: it stores a full context word-by-word through the live context window
(the section 11.3 fine path), stages inputs in buffer memory, writes `0x18`
to `DMA_CTRL`, and polls `DMA_STAT[11]`. The seven contexts cover: VMUL with
R/K rounding, MACI accumulation, an FU0→FU1 clamp chain over the staging bus
with `FU1EN`, bit-reversed addressing (`FMT1.BRV`), the section 7.7 drain
construction, segment replay (counter B against a linear stream), and a
two-PE staging pipeline with a hand-timed skew ladder.

## Host port

`vme_top` exposes one synchronous write / combinational read port whose
address space is byte offsets from `0x4400_0000`:

| Offset | Region |
| --- | --- |
| `0x00000–0x07FFF` | `BASE_0..3`, `0x2000` bytes each |
| `0x20000–0x27FFF` | `TOP_0..3` |
| `0xF8000–0xF81A4` | context words 0–105 (live: single-word stores rewire one node) |
| `0xFF000` | `DMA_STAT` — `[11]` VD array done, `[9]` TD (always set) |
| `0xFF008` | `DMA_CTRL` — writing `0x18` (TRIGGER) starts the array |

The local DMA controller's block-move/fill/context-upload machinery is not
modelled — the host port *is* the data path in simulation, so `LOAD`/`STORE`/
`FILL`/`CTXLOAD` reduce to plain writes. Only `TRIGGER` and the two status
bits carry over.

## Timing of this model

The manual is explicit that the array has no interlock, scoreboard or stall:
synchronisation is per-port cycle skew, and a mis-skewed context reads stale
data and produces plausible-but-wrong output. This model reproduces that
contract with a short, *known* pipeline (the real depth is "approximately
ten stages"; here it is documented and exact):

```
cycle 0: AGU emits address        (after MODE[23:16] skew cycles)
cycle 1: buffer read data valid
cycle 2: FU0 result registered    -> staging taps 0-3
cycle 3: FU1 result registered    -> staging taps 4-7
```

The write port stores whatever the selected FU's result register holds on
the cycle its WR AGU emits an address. Hence the skew ladder every context
needs:

- **WR skew = read skew + 2** when FU0 drives the write port, **+ 3** for FU1.
  (`vme-assembler`'s `timing.rs` encodes exactly this model and derives the
  skews automatically; its e2e tests run assembled images through `vme-emu`
  when it is built, so a pipeline change here fails there until re-tuned.)
- A consumer mixing a buffer stream with a staging stream leads the staging
  stream by 1 cycle per FU stage; cancel it with the buffer-side AGU skew or
  an `ICN_SKEW` bank code (000/100/101/110/111 → 0/1/2/3/4 cycles), which is
  exactly what that register exists for.
- `FMT0.DRAIN` on a write port (only with `FMT1.END` set, as documented)
  adds DRAIN cycles of delay to the write *data* path: the first DRAIN
  offsets of the address sequence receive junk and valid element j lands at
  offset j+DRAIN, which the final stage cancels with a negative start offset
  (`0x1_0000 − prologue`) — the section 7.7 construction, verified by test E.

## Interpretation decisions

The reference manual is a reconstruction, and in a few places it
under-determines or contradicts itself. Where a choice had to be made:

- **Shift vs saturation order.** Section 5.1 says K applies before
  saturation; section 5.3 says the reverse. The op tables embed `>>k` inside
  each equation, so this model computes the equation (including its shift,
  with R rounding) and saturates the result.
- **Operations are implemented per-cell from Table C.1**, not by orthogonal
  OPM transforms — the class-0 mode tables are irregular (`SUBA` at an
  immediate mode still names `front`, the bitwise group inverts rather than
  negates), and Appendix C is the authoritative enumeration. Where an
  immediate-mode equation says `front`, constant `b` is substituted. OPM bit
  1 (negate back) is applied generically to classes 1 and 2 only.
- **Pacing.** A unit advances its stream state when its *back* source
  produces ("back is the primary operand"); the front stream is sampled
  un-interlocked. Ops with no natural back stream (MOVI, RAMP) still pace on
  the configured back source.
- **Segment length.** `INNER0.CFG = 0x0003` ("segment length taken from the
  low half") doubles SEGMENT+1 — the only reading under which Table 7.5's
  `INNER0 = 0x0003_0001 → 0 1 2 3 …` example comes out. `0x0000`/`0x0001`
  use SEGMENT+1 directly. The `INNER1` CFG codes 0x8500/0x8C00 ("insert one
  step") are not modelled; STRIDE applies per counter-B reload.
- **ROR64** rotates the 64-bit accumulator (its documented "rotate
  capability") by the front operand and yields the low sample.
- **MULW's** `1[−2,2](front)` gate passes the product when the front
  operand's integer value lies in [−2, 2], else 0.
- **ACC.** LOAD/ZERO initialise the accumulator at trigger; HOLD accumulates
  each result and outputs the running value; MACI/SAD/ROR64 own the
  accumulator internally.
- **Sample width.** Arithmetic is 64-bit internally; every registered result
  is truncated to the architectural sample: 24-bit two's complement
  sign-extended into 32 bits. SAT clamps to `[−2^(n−1), 2^(n−1)−1]` first.
- **ICN_SRCMAP** re-routes a PE's read *address streams* to another
  element's AGUs (out-of-range = unrouted, reads never validate);
  **ICN_CFGMAP** substitutes another element's whole 18-word AGU block
  (out-of-range = own). Together they reproduce the documented four-lanes /
  paired-lanes / PE0-only patterns of Table 8.2.
- **Stored but inert:** `ICN_INMAP` (the descriptor's own selectors always
  win, section 8.2), `ICN_XPCR` sub-word packing, and the `MOD_A/B/C` flow
  modifiers. `CTX_END` is storage only.
- Every PE gets an independent read view of every buffer — a superset of
  the real port structure that is indistinguishable for contexts which
  respect the two-read-ports-per-element model.

## Not modelled

MTVME/MFVME coprocessor moves and their access patterns (chapter 10 — a
separate control path the manual itself notes nothing depends on), the DMA
controller's inner address generators, clock-enable/bus-clock behaviour, and
the uncached-alias distinction (no caches here). The `audio-recorder` /
`examples/vme-mac` crates in this repo talk to the real block; this model is
for understanding and experimenting with contexts before committing them to
hardware.
