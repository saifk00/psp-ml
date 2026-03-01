#![no_std]
#![no_main]

use psp_ml::dprintln;
use psp_ml::profiler::{ProfileClear, ProfileDisable, ProfileEnable, ProfileGetRegs, ProfileRegs};

psp_ml::module!("profiler_demo", 1, 0);

fn app_main() {
    psp::enable_home_button();
    dprintln!("=== psp-ml profiler demo ===");

    // Clear and enable hardware profiler
    unsafe {
        ProfileClear();
        ProfileEnable();
    }

    // Workload: doubly nested loop with float multiply-accumulate
    let mut acc = 0.0f32;
    for i in 0..256 {
        for j in 0..256 {
            acc += (i as f32) * (j as f32) * 0.001;
        }
    }

    // Read counters
    let regs = unsafe {
        ProfileDisable();
        let mut regs = core::mem::zeroed::<ProfileRegs>();
        ProfileGetRegs(&mut regs);
        regs
    };

    // Prevent the compiler from optimizing the loop away
    dprintln!("result: {}", acc as u32);

    dprintln!("--- hardware counters ---");
    dprintln!("enable:         {:>10}", regs.enable);
    dprintln!("system clk:     {:>10}", regs.systemck);
    dprintln!("cpu clk:        {:>10}", regs.cpuck);
    dprintln!("internal:       {:>10}", regs.internal);
    dprintln!("memory stalls:  {:>10}", regs.memory);
    dprintln!("cop0 stalls:    {:>10}", regs.copz);
    dprintln!("vfpu stalls:    {:>10}", regs.vfpu);
    dprintln!("sleep:          {:>10}", regs.sleep);
    dprintln!("bus access:     {:>10}", regs.bus_access);
    dprintln!("uncached load:  {:>10}", regs.uncached_load);
    dprintln!("uncached store: {:>10}", regs.uncached_store);
    dprintln!("cached load:    {:>10}", regs.cached_load);
    dprintln!("cached store:   {:>10}", regs.cached_store);
    dprintln!("i-cache miss:   {:>10}", regs.i_miss);
    dprintln!("d-cache miss:   {:>10}", regs.d_miss);
    dprintln!("d-cache wb:     {:>10}", regs.d_writeback);
    dprintln!("cop0 insn:      {:>10}", regs.cop0_inst);
    dprintln!("fpu insn:       {:>10}", regs.fpu_inst);
    dprintln!("vfpu insn:      {:>10}", regs.vfpu_inst);
    dprintln!("local bus:      {:>10}", regs.local_bus);
}
