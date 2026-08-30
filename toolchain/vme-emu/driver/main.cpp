// vme-emu -- host driver wrapping a Verilated simulation of the VME RTL.
//
//   vme-emu <input-image> <output-image>
//
// The image is exactly 1 MB and mirrors the VME address space
// 0x4400_0000-0x440F_FFFF (docs/vme-reference.html Table 2.1), byte offset =
// address - 0x4400_0000, words little-endian:
//   0x00000  BASE_0..3   four 8 KB ring buffers at 0x2000-byte strides
//   0x20000  TOP_0..3
//   0xF8000  context     106 words
//   0xFF000  DMA_STAT    (output image only: final status is stored here)
//
// The driver performs the same bring-up as the initialisation library, minus
// what has no meaning in simulation: the bus-clock enables live at
// 0xBC10_0050 on the Media Engine side, outside this address space, so that
// step is symbolic; buffers arrive pre-staged in the image, so no LOAD
// operations are issued; and the context goes in through the live context
// window (the section 11.3 fine path, store-per-word with readback verify)
// rather than a CTXLOAD DMA the RTL does not model.  Then TRIGGER is written
// to DMA_CTRL and the clock runs until DMA_STAT[11] (VD) sets.
//
// On completion the post-run state -- all eight buffers, the context, and
// DMA_STAT -- is written to the output image.  A full-hierarchy VCD is
// always dumped alongside it (<output-image>.vcd).  If VD never sets within
// VME_EMU_MAX_CYCLES the driver still writes the VCD and exits nonzero.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Vvme_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#define VME_EMU_MAX_CYCLES (1u << 20)

namespace {

constexpr size_t kImageSize = 0x100000;
constexpr uint32_t kBaseBuf = 0x00000;   // BASE_0..3
constexpr uint32_t kTopBuf  = 0x20000;   // TOP_0..3
constexpr uint32_t kCtx     = 0xF8000;   // context register file
constexpr uint32_t kDmaStat = 0xFF000;
constexpr uint32_t kDmaCtrl = 0xFF008;
constexpr uint32_t kTrigger = 0x18;
constexpr int kCtxWords    = 106;
constexpr int kBufWords    = 2048;

Vvme_top* dut;
VerilatedVcdC* tfp;
uint64_t half_ticks = 0;
uint64_t cycles = 0;

void tick() {
    dut->clk = 1; dut->eval(); tfp->dump(half_ticks++);
    dut->clk = 0; dut->eval(); tfp->dump(half_ticks++);
    cycles++;
}

void host_write(uint32_t addr, uint32_t data) {
    dut->host_we = 1;
    dut->host_addr = addr;
    dut->host_wdata = data;
    tick();
    dut->host_we = 0;
}

uint32_t host_read(uint32_t addr) {   // combinational, consumes no cycle
    dut->host_we = 0;
    dut->host_addr = addr;
    dut->eval();
    return dut->host_rdata;
}

uint32_t img_word(const std::vector<uint8_t>& img, uint32_t off) {
    return (uint32_t)img[off] | ((uint32_t)img[off + 1] << 8) |
           ((uint32_t)img[off + 2] << 16) | ((uint32_t)img[off + 3] << 24);
}

void img_store(std::vector<uint8_t>& img, uint32_t off, uint32_t v) {
    img[off] = v & 0xFF;
    img[off + 1] = (v >> 8) & 0xFF;
    img[off + 2] = (v >> 16) & 0xFF;
    img[off + 3] = (v >> 24) & 0xFF;
}

// buffer index 0-3 BASE_0..3, 4-7 TOP_0..3 (Table 2.2 order)
uint32_t buf_off(int b, int word) {
    uint32_t bank = (b < 4) ? kBaseBuf : kTopBuf;
    return bank + (uint32_t)(b % 4) * 0x2000 + (uint32_t)word * 4;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: vme-emu <input-image> <output-image>\n");
        return 2;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];
    const std::string vcd_path = out_path + ".vcd";

    // ---- read the machine image -------------------------------------
    std::vector<uint8_t> img(kImageSize);
    {
        FILE* f = std::fopen(in_path.c_str(), "rb");
        if (!f) { std::perror(in_path.c_str()); return 2; }
        size_t n = std::fread(img.data(), 1, kImageSize + 1, f);
        std::fclose(f);
        if (n != kImageSize) {
            std::fprintf(stderr, "vme-emu: %s is %zu bytes, expected exactly %zu\n",
                         in_path.c_str(), n, kImageSize);
            return 2;
        }
    }

    // ---- bring the array up -----------------------------------------
    Verilated::traceEverOn(true);
    dut = new Vvme_top;
    tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open(vcd_path.c_str());

    dut->clk = 0; dut->rst = 1; dut->host_we = 0;
    dut->host_addr = 0; dut->host_wdata = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;
    for (int i = 0; i < 2; i++) tick();

    std::printf("vme-emu: bus clock enables (0xBC10_0050) outside VME space -- not modelled\n");

    // ---- stage the buffers ------------------------------------------
    for (int b = 0; b < 8; b++)
        for (int w = 0; w < kBufWords; w++)
            host_write(buf_off(b, w), img_word(img, buf_off(b, w)));

    // ---- context handshake: store per word, then verify by readback --
    for (int i = 0; i < kCtxWords; i++)
        host_write(kCtx + 4 * i, img_word(img, kCtx + 4 * i));
    for (int i = 0; i < kCtxWords; i++) {
        uint32_t want = img_word(img, kCtx + 4 * i);
        uint32_t got = host_read(kCtx + 4 * i);
        if (got != want) {
            std::fprintf(stderr,
                         "vme-emu: context word %d readback %08x != %08x\n",
                         i, got, want);
            tfp->close();
            return 1;
        }
    }

    // ---- trigger, then run to completion ----------------------------
    uint64_t start = cycles;
    host_write(kDmaCtrl, kTrigger);
    bool vd = false;
    while (cycles - start < VME_EMU_MAX_CYCLES) {
        if (host_read(kDmaStat) & (1u << 11)) { vd = true; break; }
        tick();
    }
    uint64_t ran = cycles - start;

    if (!vd) {
        std::fprintf(stderr,
                     "vme-emu: DMA_STAT.VD never set within %u cycles -- see %s\n",
                     VME_EMU_MAX_CYCLES, vcd_path.c_str());
        tfp->close();
        return 1;
    }

    // ---- read the post-run state into the output image --------------
    for (int b = 0; b < 8; b++)
        for (int w = 0; w < kBufWords; w++)
            img_store(img, buf_off(b, w), host_read(buf_off(b, w)));
    for (int i = 0; i < kCtxWords; i++)
        img_store(img, kCtx + 4 * i, host_read(kCtx + 4 * i));
    img_store(img, kDmaStat, host_read(kDmaStat));

    tfp->close();

    {
        FILE* f = std::fopen(out_path.c_str(), "wb");
        if (!f) { std::perror(out_path.c_str()); return 2; }
        size_t n = std::fwrite(img.data(), 1, kImageSize, f);
        std::fclose(f);
        if (n != kImageSize) {
            std::fprintf(stderr, "vme-emu: short write to %s\n", out_path.c_str());
            return 2;
        }
    }

    std::printf("vme-emu: array done in %llu cycles; %s written, trace in %s\n",
                (unsigned long long)ran, out_path.c_str(), vcd_path.c_str());
    delete dut;
    return 0;
}
