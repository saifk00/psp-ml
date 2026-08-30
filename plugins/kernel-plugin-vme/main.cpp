/*
 * psp_vme_kernel — kernel-mode PRX that bootstraps the PSP Virtual Mobile
 * Engine (VME2) and exposes a tiny job interface to user mode.
 *
 * Layering (mirrors kernel-plugin/main.c ↔ psp-rt): the privileged,
 * power-cycle-prone parts — booting the Media Engine, the ME-side firmware
 * calls, the VME MMIO — live here in C.
 *
 * The ME-side routine `meLibOnProcess` is a generic, data-driven job runner: it
 * stages ring buffers, DMA-loads a caller-supplied context, triggers, and reads
 * one word back. Its shape is my hardware-verified vme-mac example generalised
 * to read everything from a shared user-partition buffer.
 *
 * Reused under MIT from mcidclan:
 *   https://github.com/mcidclan/psp-media-engine-custom-core
 */
#include <pspsdk.h>
#include <pspkernel.h>
#include <string.h>

extern "C" {
#include "vendor/me-core.h"      /* pulls in me-lib.h, vme-lib.h, mapper; defines
                                  * meLibHandler (_me_section) + meLibOnPreProcess */
#include "vendor/vme-ext.h"      /* VME_TOP_BUFF0_WOFF / VME_BASE_BUFFER_0 / ... */
}

PSP_MODULE_INFO("psp_vme_kernel", 0x1006, 1, 1);
PSP_NO_CREATE_MAIN_THREAD();

/* ---- shared job contract (must match the Rust side, next milestone) ------ */

#define VME_CTX_WORDS   112     /* one full VME datapath context             */
#define VME_MAX_STAGE   4       /* ring-buffer loads before a run            */
#define VME_DATA_WORDS  512     /* scratch input area referenced by stages   */

typedef struct {
  u32 src_word;    /* first word in data[] to copy                          */
  u32 ring_woff;   /* destination ring-buffer word offset (VME_*_BUFFn_WOFF)*/
  u32 count;       /* number of words                                       */
} VmeStage;

typedef struct {
  volatile u32 seq;              /* main CPU bumps to request a run          */
  volatile u32 ack;              /* ME sets == seq when the run is complete  */
  volatile u32 result;           /* readback value                          */
  u32 n_stage;
  VmeStage stage[VME_MAX_STAGE];
  u32 readback_base;             /* absolute ring base, e.g. VME_BASE_BUFFER_0 */
  u32 readback_woff;             /* word offset within that ring            */
  u32 ctx[VME_CTX_WORDS];        /* the datapath program (vme_asm! output)  */
  u32 data[VME_DATA_WORDS];      /* input words referenced by stage[]       */
  /* --- appended in v1.1 (offsets of everything above are unchanged) ----- */
  u32 image_mode;                /* 1 = run a full machine image instead    */
  u32 image_addr;                /* uncached-user address of the 1 MB image */
} VmeJob;

/* Machine-image geometry (matches vme-emu / vme-assembler): byte offset =
 * VME address - 0x4400_0000. BASE buffers at 0x0, TOP at 0x20000, context
 * at 0xF8000. One buffer = 0x2000 bytes = 2048 words. */
#define IMG_BASE_OFF   0x00000u
#define IMG_TOP_OFF    0x20000u
#define IMG_CTX_OFF    0xF8000u
#define IMG_BUF_WORDS  2048u

/* Capability marker: VmeInit leaves this in image_addr so user code can tell
 * a v1.1 plugin (image mode present, job block big enough) from a v1.0 one
 * before touching the appended fields. Must match psp-rt's vme::IMAGE_CAPS. */
#define VME_JOB_CAPS_IMAGE 0x564D4531u   /* "VME1" */

/* Uncached-user pointer to the job, in the USER partition so user mode can
 * touch it. Set on the main CPU before the ME boots; read by the ME from this
 * same module global (same address space — no SRAM channel needed). */
static volatile VmeJob* g_job = 0;
static SceUID           g_job_uid = -1;

/* ---- ME-side cache ops needed by me-core.h's meLibOnPreProcess ----------- */
/* (copied from mcidclan me-lib.c so we don't drag its user-mode helpers). */

extern "C" void meLibDcacheInvalidateRange(const u32 addr, const u32 size) {
  asm volatile("sync");
  for (volatile u32 i = addr; i < addr + size; i += 64) {
    asm volatile("cache 0x19, 0(%0)\n cache 0x19, 0(%0)\n" :: "r"(i));
  }
  asm volatile("sync");
}
extern "C" void meLibIcacheInvalidateRange(const u32 addr, const u32 size) {
  asm volatile("sync");
  for (volatile u32 i = addr; i < addr + size; i += 64) {
    asm volatile("cache 0x08, 0(%0)\n cache 0x08, 0(%0)\n" :: "r"(i));
  }
  asm volatile("sync");
}

/* me-core-custom.c references these user-mode symbols from code paths we never
 * call (its meLibDefaultInit). Provide inert definitions so the link resolves
 * without pulling in me-lib.c / kcall.prx. */
extern "C" int  meLibLoadPrx(void) { return 0; }
extern "C" int  kinit(const void* const) { return 0; }
extern "C" int  kcall_2(FCall, unsigned int) { return 0; }
extern "C" int  kcall_3(FPCall, unsigned int, void* const) { return 0; }
unsigned char embedded_kcall[1] = {0};
unsigned int  embedded_kcall_len = 0;

/* ---- the ME-side job runner (runs on the Media Engine) ------------------- */

extern "C" void meLibOnProcess(void) {
  meCoreDcacheWritebackInvalidateAll();
  meLibExceptionHandlerInit(0);

  volatile VmeJob* const job = g_job;      /* module global, valid pre-boot */
  if (!job) { while (1) { asm volatile("nop"); } }

  /* A cached staging copy of the context, written back before DMA — this
   * reproduces exactly what a VME_LIB_CONTEXT_BUILDER produced in the proven
   * vme-mac path (the DMA source must be a cached, dcache-flushed array). */
  static u32 ctxbuf[VME_CTX_WORDS] __attribute__((aligned(64)));

  u32 last = job->seq;
  job->ack = last;
  meLibSync();

  while (1) {
    if (job->seq != last) {
      vmeLibEnable();
      vmeLibWipe();

      if (job->image_mode) {
        /* Full machine image: stage all eight buffers, load the context
         * from its mapped offset, run, read every buffer back into the
         * image. The image lives in uncached user memory, same DMA-source
         * class as job->data. */
        u8* const img = (u8*)job->image_addr;
        for (u32 b = 0; b < 4; b++) {
          vmeLibMemoryToRingBuffer(img + IMG_TOP_OFF + b * 0x2000,
                                   VME_TOP_BUFF0_WOFF + 0x800 * b, IMG_BUF_WORDS);
          vmeLibMemoryToRingBuffer(img + IMG_BASE_OFF + b * 0x2000,
                                   VME_BASE_BUFF0_WOFF + 0x800 * b, IMG_BUF_WORDS);
        }

        meCoreMemcpy(ctxbuf, img + IMG_CTX_OFF, VME_CTX_WORDS * 4);
        meCoreDcacheWritebackRange(ctxbuf, VME_CTX_WORDS * 4);

        vmeLibStart();
        vmeLibLoadCustomContext(ctxbuf);
        vmeLibFinish();

        for (u32 b = 0; b < 4; b++) {
          vmeLibRingBufferToMemory(VME_TOP_BUFF0_WOFF + 0x800 * b,
                                   img + IMG_TOP_OFF + b * 0x2000, IMG_BUF_WORDS);
          vmeLibRingBufferToMemory(VME_BASE_BUFF0_WOFF + 0x800 * b,
                                   img + IMG_BASE_OFF + b * 0x2000, IMG_BUF_WORDS);
        }
        job->result = 0;
      } else {
        for (u32 i = 0; i < job->n_stage; i++) {
          vmeLibMemoryToRingBuffer((void*)&job->data[job->stage[i].src_word],
                                   job->stage[i].ring_woff,
                                   job->stage[i].count);
        }

        meCoreMemcpy(ctxbuf, (void*)job->ctx, VME_CTX_WORDS * 4);
        meCoreDcacheWritebackRange(ctxbuf, VME_CTX_WORDS * 4);

        vmeLibStart();
        vmeLibLoadCustomContext(ctxbuf);
        vmeLibFinish();

        job->result = hw(job->readback_base + job->readback_woff * 4);
      }
      vmeLibDisable();

      last = job->seq;
      job->ack = last;
      meLibSync();
    }
  }
}

/* ---- exports (called from user mode as syscalls) ------------------------- */

/* Allocate the shared job in the user partition and boot the Media Engine.
 * Returns the ME firmware image table id (>=2) or a negative error. */
extern "C" int VmeInit(void) {
  if (g_job) return 0;   /* already up */

  g_job_uid = sceKernelAllocPartitionMemory(
      PSP_MEMORY_PARTITION_USER, "vme_job", PSP_SMEM_Low,
      sizeof(VmeJob) + 64, NULL);
  if (g_job_uid < 0) return -2;

  u32 base = (u32)sceKernelGetBlockHeadAddr(g_job_uid);
  u32 aligned = (base + 63) & ~63u;
  g_job = (volatile VmeJob*)(UNCACHED_USER_MASK | aligned);
  memset((void*)g_job, 0, sizeof(VmeJob));
  g_job->image_addr = VME_JOB_CAPS_IMAGE;   /* capability marker, see above */

  int tid = meCoreGetTableIdFromWitnessWord();
  if (tid < 2) { return -6; }         /* only t2img/og validated upstream */
  meCoreSelectSystemTable(tid);

  /* Publish g_job to the ME before it boots. */
  sceKernelDcacheWritebackInvalidateAll();
  sceKernelIcacheInvalidateAll();

  meLibReset();                       /* copies _me_section -> 0xbfc00000, resets ME */
  return tid;
}

/* Uncached-user address of the shared job (for user code to fill/read). */
extern "C" unsigned int VmeSharedAddr(void) {
  return (unsigned int)g_job;
}

/* Kick one run and block until the ME acks (bounded). 0 on success. */
extern "C" int VmeRun(void) {
  if (!g_job) return -1;
  u32 s = g_job->seq + 1;
  g_job->seq = s;
  asm volatile("sync");
  for (u32 spins = 0; spins < 20000000u; spins++) {
    if (g_job->ack == s) return 0;
  }
  return -3;   /* ME did not ack — likely a bad context or a boot problem */
}

/* Free the shared job. (The ME keeps spinning; a real teardown would halt it,
 * left for later — a bench never needs it.) */
extern "C" int VmeShutdown(void) {
  if (g_job_uid >= 0) { sceKernelFreePartitionMemory(g_job_uid); g_job_uid = -1; }
  g_job = 0;
  return 0;
}

/* ---- built-in self test: the proven int MAC, no assembler needed --------- */
/*
 * Fills the job with the exact weights/inputs/context of the hardware-verified
 * vme-mac example, runs it, and returns the result. This lets us validate the
 * whole kernel path (boot + runner + shared memory) on hardware BEFORE the Rust
 * `vme_asm!` exists. Expected result for the built-in vectors: 72.
 *
 * `buildMacContextInto` is the C prototype of what `vme_asm!` will emit: it
 * writes the 112-word datapath context by pointing the vme-lib register macros
 * at a plain array instead of the live datapath.
 */
static void buildMacContextInto(volatile u32* ctx, int n) {
  volatile u32 VME_BASE_ADDR = (u32)ctx;      /* vme_set() writes land here */
  memset((void*)ctx, 0, VME_CTX_WORDS * 4);

  vme_icn(AGU_TOP,   0);
  vme_icn(AGU_BASE,  0x4440);
  vme_icn(AGU_WRITE, 0);

  const u32 VMAC = fu_op(MAC, INNER_PRODUCT_BIAS);   /* 0x00240000 */
  const int count = n;
  const int gap   = 256;

  const u32 mux = vme_mux(TOP_0, BASE_0);            /* front=TOP_0, back=BASE_0 */
  vme_pe0(vme_fu(PRIMARY), VMAC, mux, 0);            /* k = 0 */
  vme_pe0(fu_reg(PRIMARY, B), 0);                    /* bias = 0 */

  vme_pe0(agu_top(MODE),  VME_DEF_MODE);
  vme_pe0(agu_top(COUNT), VME_DEF_STEP, count);
  vme_pe0(agu_base(MODE),  VME_DEF_MODE, gap);
  vme_pe0(agu_base(COUNT), VME_DEF_STEP, count);
  vme_pe0(agu_write(MODE),  VME_DEF_MODE, VME_CYCLE_9);
  vme_pe0(agu_write(COUNT), VME_DEF_STEP, count);
}

extern "C" int VmeSelfTest(void) {
  if (!g_job) return -1;
  volatile VmeJob* j = g_job;

  const int N = 8;
  const int gap = 256;
  static const int w[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
  for (int i = 0; i < N; i++) { j->data[i] = (u32)w[i]; j->data[64 + i] = 2; }

  j->n_stage = 2;
  j->stage[0].src_word = 0;   j->stage[0].ring_woff = VME_TOP_BUFF0_WOFF;         j->stage[0].count = N;
  j->stage[1].src_word = 64;  j->stage[1].ring_woff = VME_BASE_BUFF0_WOFF + gap;  j->stage[1].count = N;
  j->readback_base = VME_BASE_BUFFER_0;
  j->readback_woff = N - 1;

  buildMacContextInto(j->ctx, N);
  asm volatile("sync");

  int rc = VmeRun();
  if (rc < 0) return rc;
  return (int)j->result;      /* expected 72 = 2*(1+..+8) */
}

extern "C" int module_start(SceSize args, void* argp) { return 0; }
extern "C" int module_stop(SceSize args, void* argp)  { return 0; }
