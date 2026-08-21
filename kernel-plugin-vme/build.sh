#!/usr/bin/env bash
#
# Fetch the mcidclan ME/VME sources this kernel PRX vendors, then `make`.
#
# The vendored copies under vendor/ are what the module actually builds against
# and are committed; this script refreshes them from the pinned upstream commits
# (run it only when deliberately updating the dependency).
#
# Prereqs: `source ../.envrc` first so $PSPDEV and psp-* tools are on PATH.
# Produces: psp_vme_kernel.prx
#
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PSPDEV:?source ../.envrc first}"

CORE_REPO="https://github.com/mcidclan/psp-media-engine-custom-core"
CORE_SHA="287ac5f1744cc68c845221613541704bf8c93625"
EXT_REPO="https://github.com/mcidclan/psp-virtual-mobile-engine-ext"
EXT_SHA="c4691f76e0433bcf8e31b22c40cf9d7ea59097de"

if [ "${1:-}" = "--refresh-vendor" ]; then
  work="$here/.deps"; mkdir -p "$work"
  fetch() {  # repo sha dst
    if [ ! -d "$3/.git" ]; then git clone "$1" "$3"; fi
    git -C "$3" fetch --depth 1 origin "$2" 2>/dev/null || git -C "$3" fetch origin
    git -C "$3" checkout -q "$2"
  }
  fetch "$CORE_REPO" "$CORE_SHA" "$work/core"
  fetch "$EXT_REPO"  "$EXT_SHA"  "$work/ext"
  mkdir -p "$here/vendor/kernel"
  for f in vme-lib.c me-core-custom.c me-core-mapping.c context.S \
           hw-registers.h vme-lib.h vme-opcodes.h vme-fu-opcodes.h me-lib.h \
           me-core.h me-core-custom.h me-core-mapper.h me-core-mapping.h me-core-mapping.def.h; do
    cp "$work/core/$f" "$here/vendor/"
  done
  cp "$work/core/kernel/kcall.h" "$here/vendor/kernel/"
  cp "$work/ext/vme-ext.h" "$here/vendor/"
  # MIT requires the copyright + permission notice travel with the sources.
  cp "$work/core/LICENSE.md" "$here/vendor/LICENSE-custom-core"
  cp "$work/ext/LICENSE.md"  "$here/vendor/LICENSE-vme-ext"
  echo ">> vendor/ refreshed to pinned commits"
fi

make -C "$here"
echo ">> built $here/psp_vme_kernel.prx"
