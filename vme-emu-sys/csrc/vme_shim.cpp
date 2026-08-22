// C ABI over the Verilated VME model, for Rust to link.
//
// Same bring-up sequence as vme-emu/driver/main.cpp, minus the file I/O and
// VCD trace: stage buffers from the input image, store + verify the context
// through the live window, TRIGGER, clock until DMA_STAT[11] (VD), read the
// post-run state into the output image.  Each call builds and tears down its
// own model instance, so concurrent calls from different threads are fine.
//
// Returns 0 on success, 1 if VD never set within max_cycles, 2 if a context
// word read back wrong.  cycles_out (optional) receives the cycle count
// from trigger to VD.

#include <cstdint>

#include "Vvme_top.h"
#include "verilated.h"

namespace {

constexpr uint32_t kBaseBuf = 0x00000;
constexpr uint32_t kTopBuf = 0x20000;
constexpr uint32_t kCtx = 0xF8000;
constexpr uint32_t kDmaStat = 0xFF000;
constexpr uint32_t kDmaCtrl = 0xFF008;
constexpr uint32_t kTrigger = 0x18;
constexpr int kCtxWords = 106;
constexpr int kBufWords = 2048;

struct Sim {
    VerilatedContext ctx;
    Vvme_top top;
    Sim() : top(&ctx) {}

    void tick() {
        top.clk = 1;
        top.eval();
        top.clk = 0;
        top.eval();
    }

    void write(uint32_t addr, uint32_t data) {
        top.host_we = 1;
        top.host_addr = addr;
        top.host_wdata = data;
        tick();
        top.host_we = 0;
    }

    uint32_t read(uint32_t addr) {
        top.host_we = 0;
        top.host_addr = addr;
        top.eval();
        return top.host_rdata;
    }
};

uint32_t img_word(const uint8_t* img, uint32_t off) {
    return (uint32_t)img[off] | ((uint32_t)img[off + 1] << 8) |
           ((uint32_t)img[off + 2] << 16) | ((uint32_t)img[off + 3] << 24);
}

void img_store(uint8_t* img, uint32_t off, uint32_t v) {
    img[off] = v & 0xFF;
    img[off + 1] = (v >> 8) & 0xFF;
    img[off + 2] = (v >> 16) & 0xFF;
    img[off + 3] = (v >> 24) & 0xFF;
}

uint32_t buf_off(int b, int word) {
    uint32_t bank = (b < 4) ? kBaseBuf : kTopBuf;
    return bank + (uint32_t)(b % 4) * 0x2000 + (uint32_t)word * 4;
}

}  // namespace

extern "C" int vme_emu_run(const uint8_t* image_in, uint8_t* image_out,
                           uint64_t max_cycles, uint64_t* cycles_out) {
    Sim sim;
    sim.top.clk = 0;
    sim.top.rst = 1;
    sim.top.host_we = 0;
    sim.top.host_addr = 0;
    sim.top.host_wdata = 0;
    for (int i = 0; i < 4; i++) sim.tick();
    sim.top.rst = 0;
    for (int i = 0; i < 2; i++) sim.tick();

    for (int b = 0; b < 8; b++)
        for (int w = 0; w < kBufWords; w++)
            sim.write(buf_off(b, w), img_word(image_in, buf_off(b, w)));

    for (int i = 0; i < kCtxWords; i++)
        sim.write(kCtx + 4 * i, img_word(image_in, kCtx + 4 * i));
    for (int i = 0; i < kCtxWords; i++)
        if (sim.read(kCtx + 4 * i) != img_word(image_in, kCtx + 4 * i))
            return 2;

    sim.write(kDmaCtrl, kTrigger);
    uint64_t cycles = 0;
    bool vd = false;
    while (cycles < max_cycles) {
        if (sim.read(kDmaStat) & (1u << 11)) {
            vd = true;
            break;
        }
        sim.tick();
        cycles++;
    }
    if (cycles_out) *cycles_out = cycles;
    if (!vd) return 1;

    for (uint32_t off = 0; off < 0x100000; off += 4)
        img_store(image_out, off, img_word(image_in, off));
    for (int b = 0; b < 8; b++)
        for (int w = 0; w < kBufWords; w++)
            img_store(image_out, buf_off(b, w), sim.read(buf_off(b, w)));
    for (int i = 0; i < kCtxWords; i++)
        img_store(image_out, kCtx + 4 * i, sim.read(kCtx + 4 * i));
    img_store(image_out, kDmaStat, sim.read(kDmaStat));
    return 0;
}
