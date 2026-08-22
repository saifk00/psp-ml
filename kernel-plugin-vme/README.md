# psp_vme_kernel — VME bootstrap plugin

A kernel-mode PRX that boots the PSP's Media Engine and exposes a tiny job
interface for driving the **Virtual Mobile Engine (VME2)** — the integer-SIMD
CGRA the VFPU lacks — from user mode. This is the C bootstrap half of the
pure-Rust VME effort: the privileged, power-cycle-prone machinery lives here
(exactly like `kernel-plugin/main.c` is C while `psp-rt` drives it in Rust);
next milestone, pure-Rust `psp-rt` stubs + a `vme_asm!` macro author and invoke
VME programs through the exports below.

Verified: **builds clean** against our `./pspdev` (psp-gcc 15.2.0) into
`psp_vme_kernel.prx`; the ME boot handler is present in `_me_section`.
**Not yet run on hardware** — that needs the install + power-cycle below.

## What it does

- Reuses mcidclan's proven Media-Engine boot verbatim (vendored, MIT — see
  `vendor/`): selects the ME firmware table from the witness word, copies an ME
  handler to `0xbfc00000`, resets the ME into our routine.
- Runs a **generic, data-driven ME-side job runner** (`meLibOnProcess`): stage
  ring buffers → DMA-load a caller-supplied 112-word datapath context → trigger
  → read one word back. Its shape is the hardware-passing `examples/vme-mac`
  routine, generalised to read everything from a shared user-partition buffer.

## Exports (syscalls)

| export | purpose |
|---|---|
| `VmeInit() -> i32` | allocate the shared job (user partition) + boot the ME. Returns the ME image table id (≥2) or negative error. |
| `VmeSharedAddr() -> u32` | uncached-user address of the shared `VmeJob` for the caller to fill/read. |
| `VmeRun() -> i32` | kick one run, block until the ME acks. 0 = ok. |
| `VmeShutdown() -> i32` | free the shared job. |
| `VmeSelfTest() -> i32` | fill the job with the proven int MAC (weights `1..8`, inputs `2`), run it, return the result. **Expected: 72.** Lets us validate the whole path on hardware before the Rust assembler exists. |

The `VmeJob` struct at the top of `main.cpp` is the contract the Rust side
(`psp_rt::vme`) mirrors. `buildMacContextInto` is the C prototype of what
`vme_asm!` emits.

**v1.1 adds machine-image mode**: two fields appended to `VmeJob`
(`image_mode`, `image_addr`) switch the runner to executing a full 1 MB
machine image (vme-emu / vme-assembler format) — the ME stages all eight ring
buffers from the image, loads the context from its mapped offset (0xF8000),
runs, and reads every buffer back into the image. `VmeInit` leaves a
capability marker (`0x564D4531`, "VME1") in `image_addr` so user code can
detect an old plugin before touching the appended fields
(`psp_rt::vme::Job::has_image_mode`). Used by `examples/vme-conformance` to
diff real silicon against the RTL in `vme-emu/`.

## Build

```bash
source ../.envrc
make                      # -> psp_vme_kernel.prx
./build.sh --refresh-vendor   # only when updating the pinned mcidclan sources
```

## Install (one-time, then power-cycle)

A kernel PRX can't be `ld`-started by psplink; firmware loads it at boot from
`ms0:/seplugins`. The installer writes it there from device code:

```bash
make -C kernel-plugin-vme
cargo run -p vme-install-host --release
# then power-cycle the PSP and relaunch psplink
```

Requires CFW / kernel access (psplink already provides it). After the
power-cycle, `VmeInit`/`VmeSelfTest`/… are callable as syscalls.

## Risk notes (unvalidated on hardware)

- `VmeInit` boots the ME on the export call, **not** at `module_start`, so a bad
  boot fails at run time rather than hanging the PSP at power-on.
- The novel VME/ME code runs on the **ME core**; a bad datapath context should
  yield a wrong `result` or a `VmeRun` timeout (`-3`), not a main-CPU fault. But
  the ME boot + reset sequence is privileged — if it disrupts the USB/hostfs
  link, psplink may drop and need a power-cycle.
- The cross-core channel is a **module global** (`g_job`) holding an
  uncached-user pointer to a user-partition buffer, written back before the ME
  boots. This is the main untested assumption; if the ME reads a stale pointer,
  `VmeSelfTest` will time out cleanly.

## Credit

ME boot + VME driver: mcidclan
([custom-core](https://github.com/mcidclan/psp-media-engine-custom-core),
[vme-ext](https://github.com/mcidclan/psp-virtual-mobile-engine-ext), MIT),
vendored under `vendor/`.
