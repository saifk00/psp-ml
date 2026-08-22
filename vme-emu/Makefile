IVERILOG  ?= iverilog
VVP       ?= vvp
VERILATOR ?= verilator

RTL := $(wildcard rtl/*.v)
TB  := tb/tb_vme.v

all: build/tb_vme build/vme-emu

build/tb_vme: $(TB) $(RTL) | build
	$(IVERILOG) -g2012 -Wall -o $@ $(TB) $(RTL)

# The host driver: Verilated vme_top linked with driver/main.cpp.
build/vme-emu: driver/main.cpp $(RTL) | build
	$(VERILATOR) --cc --exe --build -j 0 --trace \
		-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
		--top-module vme_top -Mdir build/obj \
		-o ../vme-emu $(RTL) $(CURDIR)/driver/main.cpp

build:
	mkdir -p build

.PHONY: all test clean
test: build/tb_vme
	$(VVP) build/tb_vme

clean:
	rm -rf build
