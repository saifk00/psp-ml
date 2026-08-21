# vme-mac-rs — pure-Rust integer SIMD MAC on the VME

The pure-Rust counterpart to the C `examples/vme-mac`. A Rust program authors a
VME datapath program with **`vme_asm!`**, stages the operands, invokes the
engine, and reads back an integer dot product from the Media Engine's VME —
with no C in the invocation path.

**Hardware-verified:**

```
=== pure-Rust VME MAC ===
ME booted, image table = 2
VME dot = 72, scalar reference = 72 -> PASS
```

`weights = {1..8}`, `inputs = {2,…}` → `Σ = 2·(1+…+8) = 72`, computed by one VME
processing element via the `MAC_INNER_PRODUCT_BIAS` opcode, cross-checked
against a scalar reference.

## The layering

- **`kernel-plugin-vme`** (C, installed once) — boots the Media Engine and runs
  a generic ME-side job runner. The privileged, power-cycle-prone parts.
- **`psp-rt::vme`** (Rust) — syscall stubs for the plugin, the `VmeJob` shared
  struct, and `vme_asm!` + `vme::asm` encoders that build the 112-word datapath
  context. This is the "pure Rust" surface.
- **this example** — `vme::init()` boots the ME; `vme_asm![…]` builds the
  program; `Job` fills the shared buffer; `vme::run()` executes; `job.result()`
  reads back.

`vme_asm!` is the VME analogue of `vfpu_asm!`: where that emits one VFPU
instruction per line, this sets one datapath register per line (`index =>
value`, order-independent). It is deliberately non-exhaustive — only what the
MAC needs is encoded. Extending it is additive: a new op is one `const` value
encoder in `vme::asm`; a new register is one index `const fn`.

## Prerequisites

Install the kernel plugin once and power-cycle (see `kernel-plugin-vme/README.md`):

```bash
make -C kernel-plugin-vme
# copy psp_vme_kernel.prx to ms0:/seplugins + add to game.txt (installer or manual pspsh cp)
# power-cycle the PSP, relaunch psplink
```

## Run

```bash
cargo run -p vme-mac-rs-host --release
```

Requires the plugin resident (CFW/kernel; psplink provides it). If `vme::init`
returns negative, the plugin isn't loaded — check `pspsh -e modlist` for
`psp_vme_kernel`.

## Notes

- `image table = 0` on a re-run just means the ME was already booted by a prior
  load (the plugin is resident); it's returned by `VmeInit` as "already up".
- The built-in `self_test()` line exercises the plugin's *own* C-built MAC and
  is only a convenience; the Rust path builds its own context and is what this
  example proves.
