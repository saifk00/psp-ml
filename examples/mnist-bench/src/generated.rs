//! Generated inference module
#[allow(unused_imports)]
use psp_ml::kernels::naive::*;
#[allow(unused_imports)]
use psp_ml::kernels::*;
static mut ARENA: Aligned16<67648usize> = Aligned16([0.0f32; 67648usize]);
pub const OUTPUT_SIZE: usize = 10usize;
pub fn forward(input: &[f32; 784usize], output: &mut [f32; 10usize]) {
    let t_10 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(21952usize),
            6272usize,
        )
    };
    let t_11 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            1568usize,
        )
    };
    let t_12 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(4768usize),
            3136usize,
        )
    };
    let t_13 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            784usize,
        )
    };
    let t_14 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(784usize),
            784usize,
        )
    };
    let t_15 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            64usize,
        )
    };
    static mut T_16_BUF: Aligned16<10usize> = Aligned16([0.0f32; 10usize]);
    let t_16 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_16_BUF) as *mut f32,
            10usize,
        )
    };
    let tensor_data = tensor_data_f32();
    let t_1 = &tensor_data[T_1_OFFSET..T_1_OFFSET + T_1_LEN];
    let t_2 = &tensor_data[T_2_OFFSET..T_2_OFFSET + T_2_LEN];
    let t_3 = &tensor_data[T_3_OFFSET..T_3_OFFSET + T_3_LEN];
    let t_4 = &tensor_data[T_4_OFFSET..T_4_OFFSET + T_4_LEN];
    let t_5 = &tensor_data[T_5_OFFSET..T_5_OFFSET + T_5_LEN];
    let t_6 = &tensor_data[T_6_OFFSET..T_6_OFFSET + T_6_LEN];
    let t_8 = &tensor_data[T_8_OFFSET..T_8_OFFSET + T_8_LEN];
    let t_9 = &tensor_data[T_9_OFFSET..T_9_OFFSET + T_9_LEN];
    let scratch_0_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            21952usize,
        )
    };
    let scratch_0_1 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(28224usize),
            224usize,
        )
    };
    for row in 0..8usize {
        scratch_0_1[row * 28usize..row * 28usize + 25usize]
            .copy_from_slice(&t_8[row * 25usize..(row + 1) * 25usize]);
    }
    im2col_padded(
        input,
        [1usize, 28usize, 28usize, 1usize],
        [5usize, 5usize],
        [1usize, 1usize],
        [2usize, 2usize, 2usize, 2usize],
        [28usize, 28usize],
        scratch_0_0,
    );
    matmul_bt_tiled(scratch_0_0, scratch_0_1, t_10, 196usize, 7usize, 2usize);
    bias_add(t_10, t_9, 784usize, 8usize);
    relu(t_10);
    max_pool2d(
        t_10,
        [1usize, 28usize, 28usize, 8usize],
        [2usize, 2usize],
        [2usize, 2usize],
        [0usize, 0usize, 0usize, 0usize],
        t_11,
        [1usize, 14usize, 14usize, 8usize],
    );
    let scratch_2_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(28448usize),
            39200usize,
        )
    };
    let scratch_2_1 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(1568usize),
            3200usize,
        )
    };
    scratch_2_1.copy_from_slice(t_4);
    im2col_padded(
        t_11,
        [1usize, 14usize, 14usize, 8usize],
        [5usize, 5usize],
        [1usize, 1usize],
        [2usize, 2usize, 2usize, 2usize],
        [14usize, 14usize],
        scratch_2_0,
    );
    matmul_bt_tiled(scratch_2_0, scratch_2_1, t_12, 49usize, 50usize, 4usize);
    bias_add(t_12, t_2, 196usize, 16usize);
    relu(t_12);
    max_pool2d(
        t_12,
        [1usize, 14usize, 14usize, 16usize],
        [2usize, 2usize],
        [2usize, 2usize],
        [0usize, 0usize, 0usize, 0usize],
        t_13,
        [1usize, 7usize, 7usize, 16usize],
    );
    reshape(t_13, t_14);
    fully_connected_relu(t_14, 784usize, t_6, Some(t_1), t_15, 64usize);
    fully_connected(t_15, 64usize, t_5, Some(t_3), t_16, 10usize);
    output.copy_from_slice(&t_16);
}
/// Instrumented inference: accumulates per-op tick deltas into `op_ticks`.
pub fn forward_timed(
    input: &[f32; 784usize],
    output: &mut [f32; 10usize],
    op_ticks: &mut [u64; NUM_OPS],
    get_tick: fn() -> u64,
) {
    let t_10 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(21952usize),
            6272usize,
        )
    };
    let t_11 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            1568usize,
        )
    };
    let t_12 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(4768usize),
            3136usize,
        )
    };
    let t_13 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            784usize,
        )
    };
    let t_14 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(784usize),
            784usize,
        )
    };
    let t_15 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            64usize,
        )
    };
    static mut T_16_BUF: Aligned16<10usize> = Aligned16([0.0f32; 10usize]);
    let t_16 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_16_BUF) as *mut f32,
            10usize,
        )
    };
    let tensor_data = tensor_data_f32();
    let t_1 = &tensor_data[T_1_OFFSET..T_1_OFFSET + T_1_LEN];
    let t_2 = &tensor_data[T_2_OFFSET..T_2_OFFSET + T_2_LEN];
    let t_3 = &tensor_data[T_3_OFFSET..T_3_OFFSET + T_3_LEN];
    let t_4 = &tensor_data[T_4_OFFSET..T_4_OFFSET + T_4_LEN];
    let t_5 = &tensor_data[T_5_OFFSET..T_5_OFFSET + T_5_LEN];
    let t_6 = &tensor_data[T_6_OFFSET..T_6_OFFSET + T_6_LEN];
    let t_8 = &tensor_data[T_8_OFFSET..T_8_OFFSET + T_8_LEN];
    let t_9 = &tensor_data[T_9_OFFSET..T_9_OFFSET + T_9_LEN];
    let scratch_0_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            21952usize,
        )
    };
    let scratch_0_1 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(28224usize),
            224usize,
        )
    };
    for row in 0..8usize {
        scratch_0_1[row * 28usize..row * 28usize + 25usize]
            .copy_from_slice(&t_8[row * 25usize..(row + 1) * 25usize]);
    }
    let __t0 = get_tick();
    im2col_padded(
        input,
        [1usize, 28usize, 28usize, 1usize],
        [5usize, 5usize],
        [1usize, 1usize],
        [2usize, 2usize, 2usize, 2usize],
        [28usize, 28usize],
        scratch_0_0,
    );
    op_ticks[0usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_0_0, scratch_0_1, t_10, 196usize, 7usize, 2usize);
    op_ticks[1usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_10, t_9, 784usize, 8usize);
    relu(t_10);
    op_ticks[2usize] += get_tick() - __t0;
    let __t0 = get_tick();
    max_pool2d(
        t_10,
        [1usize, 28usize, 28usize, 8usize],
        [2usize, 2usize],
        [2usize, 2usize],
        [0usize, 0usize, 0usize, 0usize],
        t_11,
        [1usize, 14usize, 14usize, 8usize],
    );
    op_ticks[3usize] += get_tick() - __t0;
    let scratch_2_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(28448usize),
            39200usize,
        )
    };
    let scratch_2_1 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(1568usize),
            3200usize,
        )
    };
    scratch_2_1.copy_from_slice(t_4);
    let __t0 = get_tick();
    im2col_padded(
        t_11,
        [1usize, 14usize, 14usize, 8usize],
        [5usize, 5usize],
        [1usize, 1usize],
        [2usize, 2usize, 2usize, 2usize],
        [14usize, 14usize],
        scratch_2_0,
    );
    op_ticks[4usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_2_0, scratch_2_1, t_12, 49usize, 50usize, 4usize);
    op_ticks[5usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_12, t_2, 196usize, 16usize);
    relu(t_12);
    op_ticks[6usize] += get_tick() - __t0;
    let __t0 = get_tick();
    max_pool2d(
        t_12,
        [1usize, 14usize, 14usize, 16usize],
        [2usize, 2usize],
        [2usize, 2usize],
        [0usize, 0usize, 0usize, 0usize],
        t_13,
        [1usize, 7usize, 7usize, 16usize],
    );
    op_ticks[7usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_13, t_14);
    op_ticks[8usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fully_connected_relu(t_14, 784usize, t_6, Some(t_1), t_15, 64usize);
    op_ticks[9usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fully_connected(t_15, 64usize, t_5, Some(t_3), t_16, 10usize);
    op_ticks[10usize] += get_tick() - __t0;
    output.copy_from_slice(&t_16);
}
/// Instrumented inference with per-op hardware profiling counters.
pub fn forward_profiled(
    input: &[f32; 784usize],
    output: &mut [f32; 10usize],
    op_ticks: &mut [u64; NUM_OPS],
    #[allow(unused)]
    op_profile: &mut [psp_ml::profiler::OpProfileStats; NUM_OPS],
    get_tick: fn() -> u64,
) {
    let t_10 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(21952usize),
            6272usize,
        )
    };
    let t_11 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            1568usize,
        )
    };
    let t_12 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(4768usize),
            3136usize,
        )
    };
    let t_13 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            784usize,
        )
    };
    let t_14 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(784usize),
            784usize,
        )
    };
    let t_15 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            64usize,
        )
    };
    static mut T_16_BUF: Aligned16<10usize> = Aligned16([0.0f32; 10usize]);
    let t_16 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_16_BUF) as *mut f32,
            10usize,
        )
    };
    let tensor_data = tensor_data_f32();
    let t_1 = &tensor_data[T_1_OFFSET..T_1_OFFSET + T_1_LEN];
    let t_2 = &tensor_data[T_2_OFFSET..T_2_OFFSET + T_2_LEN];
    let t_3 = &tensor_data[T_3_OFFSET..T_3_OFFSET + T_3_LEN];
    let t_4 = &tensor_data[T_4_OFFSET..T_4_OFFSET + T_4_LEN];
    let t_5 = &tensor_data[T_5_OFFSET..T_5_OFFSET + T_5_LEN];
    let t_6 = &tensor_data[T_6_OFFSET..T_6_OFFSET + T_6_LEN];
    let t_8 = &tensor_data[T_8_OFFSET..T_8_OFFSET + T_8_LEN];
    let t_9 = &tensor_data[T_9_OFFSET..T_9_OFFSET + T_9_LEN];
    let scratch_0_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            21952usize,
        )
    };
    let scratch_0_1 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(28224usize),
            224usize,
        )
    };
    for row in 0..8usize {
        scratch_0_1[row * 28usize..row * 28usize + 25usize]
            .copy_from_slice(&t_8[row * 25usize..(row + 1) * 25usize]);
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    im2col_padded(
        input,
        [1usize, 28usize, 28usize, 1usize],
        [5usize, 5usize],
        [1usize, 1usize],
        [2usize, 2usize, 2usize, 2usize],
        [28usize, 28usize],
        scratch_0_0,
    );
    op_ticks[0usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[0usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_0_0, scratch_0_1, t_10, 196usize, 7usize, 2usize);
    op_ticks[1usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[1usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    bias_add(t_10, t_9, 784usize, 8usize);
    relu(t_10);
    op_ticks[2usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[2usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    max_pool2d(
        t_10,
        [1usize, 28usize, 28usize, 8usize],
        [2usize, 2usize],
        [2usize, 2usize],
        [0usize, 0usize, 0usize, 0usize],
        t_11,
        [1usize, 14usize, 14usize, 8usize],
    );
    op_ticks[3usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[3usize].accumulate(__regs.assume_init_ref());
    }
    let scratch_2_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(28448usize),
            39200usize,
        )
    };
    let scratch_2_1 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(1568usize),
            3200usize,
        )
    };
    scratch_2_1.copy_from_slice(t_4);
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    im2col_padded(
        t_11,
        [1usize, 14usize, 14usize, 8usize],
        [5usize, 5usize],
        [1usize, 1usize],
        [2usize, 2usize, 2usize, 2usize],
        [14usize, 14usize],
        scratch_2_0,
    );
    op_ticks[4usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[4usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_2_0, scratch_2_1, t_12, 49usize, 50usize, 4usize);
    op_ticks[5usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[5usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    bias_add(t_12, t_2, 196usize, 16usize);
    relu(t_12);
    op_ticks[6usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[6usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    max_pool2d(
        t_12,
        [1usize, 14usize, 14usize, 16usize],
        [2usize, 2usize],
        [2usize, 2usize],
        [0usize, 0usize, 0usize, 0usize],
        t_13,
        [1usize, 7usize, 7usize, 16usize],
    );
    op_ticks[7usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[7usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reshape(t_13, t_14);
    op_ticks[8usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[8usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fully_connected_relu(t_14, 784usize, t_6, Some(t_1), t_15, 64usize);
    op_ticks[9usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[9usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fully_connected(t_15, 64usize, t_5, Some(t_3), t_16, 10usize);
    op_ticks[10usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[10usize].accumulate(__regs.assume_init_ref());
    }
    output.copy_from_slice(&t_16);
}
pub const NUM_OPS: usize = 11usize;
pub const OP_NAMES: [&str; NUM_OPS] = [
    "im2col",
    "matmul",
    "bias_add_relu",
    "max_pool2d",
    "im2col",
    "matmul",
    "bias_add_relu",
    "max_pool2d",
    "reshape",
    "fully_connected_relu",
    "fully_connected",
];
#[allow(dead_code)]
#[repr(align(16))]
struct AlignedBytes<const N: usize>([u8; N]);
/// 16-byte aligned f32 array for VFPU `lv.q`/`sv.q`.
#[repr(C, align(16))]
struct Aligned16<const N: usize>([f32; N]);
static TENSOR_DATA_BYTES: AlignedBytes<217264usize> = AlignedBytes(
    *include_bytes!("weights.bin"),
);
const TENSOR_DATA_FLOATS: usize = 54316usize;
const T_1_OFFSET: usize = 0usize;
const T_1_LEN: usize = 64usize;
const T_2_OFFSET: usize = 64usize;
const T_2_LEN: usize = 16usize;
const T_3_OFFSET: usize = 80usize;
const T_3_LEN: usize = 10usize;
const T_4_OFFSET: usize = 92usize;
const T_4_LEN: usize = 3200usize;
const T_5_OFFSET: usize = 3292usize;
const T_5_LEN: usize = 640usize;
const T_6_OFFSET: usize = 3932usize;
const T_6_LEN: usize = 50176usize;
const T_8_OFFSET: usize = 54108usize;
const T_8_LEN: usize = 200usize;
const T_9_OFFSET: usize = 54308usize;
const T_9_LEN: usize = 8usize;
fn tensor_data_f32() -> &'static [f32] {
    unsafe {
        core::slice::from_raw_parts(
            TENSOR_DATA_BYTES.0.as_ptr() as *const f32,
            TENSOR_DATA_FLOATS,
        )
    }
}
/// No-op for embedded weights. Provided so callers can
/// unconditionally call `init()` regardless of model size.
pub fn init() {}
