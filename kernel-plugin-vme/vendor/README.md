# Vendored ME/VME sources (third-party, MIT)

These files are copied **unmodified** from mcidclan's MIT-licensed projects and
are the proven Media-Engine boot + VME driver the `psp_vme_kernel` plugin builds
on. They are refreshed by `../build.sh --refresh-vendor` at pinned commits.

## Provenance

From **psp-media-engine-custom-core** (commit `287ac5f1744cc68c845221613541704bf8c93625`)
— license: [`LICENSE-custom-core`](LICENSE-custom-core), © 2025 m-c/d:

```
me-core.h  me-core-custom.{c,h}  me-core-mapper.h  me-core-mapping.{c,h}
me-core-mapping.def.h  me-lib.h  vme-lib.{c,h}  vme-opcodes.h
vme-fu-opcodes.h  hw-registers.h  context.S  kernel/kcall.h
```

From **psp-virtual-mobile-engine-ext** (commit `c4691f76e0433bcf8e31b22c40cf9d7ea59097de`)
— license: [`LICENSE-vme-ext`](LICENSE-vme-ext), © 2026 m-c/d, mcidclan:

```
vme-ext.h
```

## Notes

- The files here are upstream and unmodified; the repo-specific glue (the ME job
  runner, the `VmeJob` contract, the exports) lives in `../main.cpp`, which
  credits mcidclan in its header.
- Both licenses are MIT. Keep both `LICENSE-*` files alongside these sources when
  redistributing (that is the MIT notice-inclusion requirement).
