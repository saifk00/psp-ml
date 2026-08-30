TARGET = psp_ml_kernel
OBJS = main.o exports.o

CFLAGS = -O2 -G0 -Wall
BUILD_PRX = 1
USE_KERNEL_LIBC = 1
USE_KERNEL_LIBS = 1
PRX_EXPORTS = exports.exp

PSPSDK = $(shell psp-config --pspsdk-path)
include $(PSPSDK)/lib/build_prx.mak

# Generate user-mode stub library from exports.exp.
# The .S filename is determined by psp-build-exports (libname_XXXX.S).
stubs:
	psp-build-exports -s exports.exp
	psp-gcc -I$(PSPSDK)/include -c psp_ml_kernel.S -o psp_ml_kernel_stub.o
	psp-ar rcs libpsp_ml_kernel.a psp_ml_kernel_stub.o

clean-stubs:
	rm -f psp_ml_kernel.S psp_ml_kernel_stub.o libpsp_ml_kernel.a
