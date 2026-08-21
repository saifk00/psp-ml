# psp_vme_kernel — kernel-mode PRX that bootstraps the VME.
#
# Standalone C/C++ kernel PRX (like kernel-plugin/), built with the classic
# pspsdk build_prx.mak. `source ../.envrc` first so PSPDEV/psp-* are on PATH.
# `make` produces psp_vme_kernel.prx; install it with profiler-install-style
# tooling (see README) then relaunch psplink.

TARGET = psp_vme_kernel
OBJS = main.o \
       vendor/vme-lib.o \
       vendor/me-core-custom.o \
       vendor/me-core-mapping.o \
       vendor/context.o \
       exports.o

INCDIR   = vendor
CFLAGS   = -O2 -G0 -Wall -fno-pic
CXXFLAGS = $(CFLAGS) -fno-rtti -fno-exceptions -std=gnu++11
ASFLAGS  = $(CFLAGS)

BUILD_PRX = 1
USE_KERNEL_LIBC = 1
USE_KERNEL_LIBS = 1
PRX_EXPORTS = exports.exp

PSPSDK = $(shell psp-config --pspsdk-path)
include $(PSPSDK)/lib/build_prx.mak

# Generate the user-mode stub library (libpsp_vme_kernel.a) from exports.exp so
# a host/device caller can link against the syscalls. The .S name is chosen by
# psp-build-exports (libname_XXXX.S).
stubs:
	psp-build-exports -s exports.exp
	psp-gcc -I$(PSPSDK)/include -c psp_vme_kernel.S -o psp_vme_kernel_stub.o
	psp-ar rcs libpsp_vme_kernel.a psp_vme_kernel_stub.o

clean-stubs:
	rm -f psp_vme_kernel.S psp_vme_kernel_stub.o libpsp_vme_kernel.a
