#include <pspsdk.h>
#include <pspkernel.h>
#include <pspdebug.h>

PSP_MODULE_INFO("psp_ml_kernel", 0x1000, 1, 0);
PSP_NO_CREATE_MAIN_THREAD();

#define PROFILER_REG_BASE  ((volatile u32 *)0xBC400000)
/* Must equal the field count of ProfileRegs (and the Rust mirror in
 * psp-rt/src/profiler.rs) — GetRegs copies exactly this many words into the
 * caller's struct, so an extra register here overruns the caller's stack. */
#define PROFILER_REG_COUNT (sizeof(ProfileRegs) / sizeof(u32))

typedef struct {
    u32 enable;
    u32 systemck, cpuck, internal, memory, copz, vfpu, sleep;
    u32 bus_access;
    u32 uncached_load, uncached_store, cached_load, cached_store;
    u32 i_miss, d_miss, d_writeback;
    u32 cop0_inst, fpu_inst, vfpu_inst;
    u32 local_bus;
} ProfileRegs;

void ProfileEnable(void)  { PROFILER_REG_BASE[0] = 1; }
void ProfileDisable(void) { PROFILER_REG_BASE[0] = 0; asm("sync"); }
void ProfileClear(void)   { for (int i = 1; i < PROFILER_REG_COUNT; i++) PROFILER_REG_BASE[i] = 0; }
void ProfileGetRegs(ProfileRegs *regs) {
    u32 *dst = (u32 *)regs;
    for (int i = 0; i < PROFILER_REG_COUNT; i++) dst[i] = PROFILER_REG_BASE[i];
}

int module_start(SceSize args, void *argp) {
    Kprintf("psp_ml_kernel: loaded\n");
    return 0;
}
int module_stop(SceSize args, void *argp) { return 0; }
