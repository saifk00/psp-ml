//! Generated inference module
#[allow(unused_imports)]
use psp_ml::kernels::naive::{
    conv2d, conv2d_relu, max_pool2d, reshape, fully_connected, fully_connected_relu,
};
#[allow(unused_imports)]
use psp_ml::kernels::{im2col, im2col_padded, matmul_bt, matmul_bt_tiled, bias_add, relu};
pub fn forward(input: &[f32; 784usize]) -> [f32; 10usize] {
    let mut t_10 = [0.0f32; 6272usize];
    let mut t_11 = [0.0f32; 1568usize];
    let mut t_12 = [0.0f32; 3136usize];
    let mut t_13 = [0.0f32; 784usize];
    let mut t_14 = [0.0f32; 784usize];
    let mut t_15 = [0.0f32; 64usize];
    let mut t_16 = [0.0f32; 10usize];
    let tensor_data = tensor_data_f32();
    let t_1 = &tensor_data[T_1_OFFSET..T_1_OFFSET + T_1_LEN];
    let t_2 = &tensor_data[T_2_OFFSET..T_2_OFFSET + T_2_LEN];
    let t_3 = &tensor_data[T_3_OFFSET..T_3_OFFSET + T_3_LEN];
    let t_4 = &tensor_data[T_4_OFFSET..T_4_OFFSET + T_4_LEN];
    let t_5 = &tensor_data[T_5_OFFSET..T_5_OFFSET + T_5_LEN];
    let t_6 = &tensor_data[T_6_OFFSET..T_6_OFFSET + T_6_LEN];
    let t_7 = &tensor_data[T_7_OFFSET..T_7_OFFSET + T_7_LEN];
    let t_8 = &tensor_data[T_8_OFFSET..T_8_OFFSET + T_8_LEN];
    let t_9 = &tensor_data[T_9_OFFSET..T_9_OFFSET + T_9_LEN];
    static mut CONV_SCRATCH_0: [f32; 21952usize] = [0.0f32; 21952usize];
    let conv_scratch_0 = unsafe { &mut *::core::ptr::addr_of_mut!(CONV_SCRATCH_0) };
    static mut PADDED_W_0: [f32; 224usize] = [0.0f32; 224usize];
    let padded_w_0 = unsafe { &mut *::core::ptr::addr_of_mut!(PADDED_W_0) };
    for row in 0..8usize {
        padded_w_0[row * 28usize..row * 28usize + 25usize]
            .copy_from_slice(&t_8[row * 25usize..(row + 1) * 25usize]);
    }
    im2col_padded(
        input,
        [1usize, 28usize, 28usize, 1usize],
        [5usize, 5usize],
        [2usize, 2usize],
        [28usize, 28usize],
        conv_scratch_0,
    );
    matmul_bt_tiled(conv_scratch_0, padded_w_0, &mut t_10, 196usize, 7usize, 2usize);
    bias_add(&mut t_10, t_9, 784usize, 8usize);
    relu(&mut t_10);
    max_pool2d(
        &t_10,
        [1usize, 28usize, 28usize, 8usize],
        [2, 2],
        [2, 2],
        &mut t_11,
        [1usize, 14usize, 14usize, 8usize],
    );
    static mut CONV_SCRATCH_2: [f32; 39200usize] = [0.0f32; 39200usize];
    let conv_scratch_2 = unsafe { &mut *::core::ptr::addr_of_mut!(CONV_SCRATCH_2) };
    im2col_padded(
        &t_11,
        [1usize, 14usize, 14usize, 8usize],
        [5usize, 5usize],
        [2usize, 2usize],
        [14usize, 14usize],
        conv_scratch_2,
    );
    matmul_bt_tiled(conv_scratch_2, t_4, &mut t_12, 49usize, 50usize, 4usize);
    bias_add(&mut t_12, t_2, 196usize, 16usize);
    relu(&mut t_12);
    max_pool2d(
        &t_12,
        [1usize, 14usize, 14usize, 16usize],
        [2, 2],
        [2, 2],
        &mut t_13,
        [1usize, 7usize, 7usize, 16usize],
    );
    reshape(&t_13, &mut t_14);
    fully_connected_relu(&t_14, 784usize, t_6, t_1, &mut t_15, 64usize);
    fully_connected(&t_15, 64usize, t_5, t_3, &mut t_16, 10usize);
    t_16
}
/// Instrumented inference: accumulates per-op tick deltas into `op_ticks`.
pub fn forward_timed(
    input: &[f32; 784usize],
    op_ticks: &mut [u64; NUM_OPS],
    get_tick: fn() -> u64,
) -> [f32; 10usize] {
    let mut t_10 = [0.0f32; 6272usize];
    let mut t_11 = [0.0f32; 1568usize];
    let mut t_12 = [0.0f32; 3136usize];
    let mut t_13 = [0.0f32; 784usize];
    let mut t_14 = [0.0f32; 784usize];
    let mut t_15 = [0.0f32; 64usize];
    let mut t_16 = [0.0f32; 10usize];
    let tensor_data = tensor_data_f32();
    let t_1 = &tensor_data[T_1_OFFSET..T_1_OFFSET + T_1_LEN];
    let t_2 = &tensor_data[T_2_OFFSET..T_2_OFFSET + T_2_LEN];
    let t_3 = &tensor_data[T_3_OFFSET..T_3_OFFSET + T_3_LEN];
    let t_4 = &tensor_data[T_4_OFFSET..T_4_OFFSET + T_4_LEN];
    let t_5 = &tensor_data[T_5_OFFSET..T_5_OFFSET + T_5_LEN];
    let t_6 = &tensor_data[T_6_OFFSET..T_6_OFFSET + T_6_LEN];
    let t_7 = &tensor_data[T_7_OFFSET..T_7_OFFSET + T_7_LEN];
    let t_8 = &tensor_data[T_8_OFFSET..T_8_OFFSET + T_8_LEN];
    let t_9 = &tensor_data[T_9_OFFSET..T_9_OFFSET + T_9_LEN];
    static mut CONV_SCRATCH_0: [f32; 21952usize] = [0.0f32; 21952usize];
    let conv_scratch_0 = unsafe { &mut *::core::ptr::addr_of_mut!(CONV_SCRATCH_0) };
    static mut PADDED_W_0: [f32; 224usize] = [0.0f32; 224usize];
    let padded_w_0 = unsafe { &mut *::core::ptr::addr_of_mut!(PADDED_W_0) };
    for row in 0..8usize {
        padded_w_0[row * 28usize..row * 28usize + 25usize]
            .copy_from_slice(&t_8[row * 25usize..(row + 1) * 25usize]);
    }
    let __t0 = get_tick();
    im2col_padded(
        input,
        [1usize, 28usize, 28usize, 1usize],
        [5usize, 5usize],
        [2usize, 2usize],
        [28usize, 28usize],
        conv_scratch_0,
    );
    op_ticks[0usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(conv_scratch_0, padded_w_0, &mut t_10, 196usize, 7usize, 2usize);
    op_ticks[1usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(&mut t_10, t_9, 784usize, 8usize);
    relu(&mut t_10);
    op_ticks[2usize] += get_tick() - __t0;
    let __t0 = get_tick();
    max_pool2d(
        &t_10,
        [1usize, 28usize, 28usize, 8usize],
        [2, 2],
        [2, 2],
        &mut t_11,
        [1usize, 14usize, 14usize, 8usize],
    );
    op_ticks[3usize] += get_tick() - __t0;
    static mut CONV_SCRATCH_2: [f32; 39200usize] = [0.0f32; 39200usize];
    let conv_scratch_2 = unsafe { &mut *::core::ptr::addr_of_mut!(CONV_SCRATCH_2) };
    let __t0 = get_tick();
    im2col_padded(
        &t_11,
        [1usize, 14usize, 14usize, 8usize],
        [5usize, 5usize],
        [2usize, 2usize],
        [14usize, 14usize],
        conv_scratch_2,
    );
    op_ticks[4usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(conv_scratch_2, t_4, &mut t_12, 49usize, 50usize, 4usize);
    op_ticks[5usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(&mut t_12, t_2, 196usize, 16usize);
    relu(&mut t_12);
    op_ticks[6usize] += get_tick() - __t0;
    let __t0 = get_tick();
    max_pool2d(
        &t_12,
        [1usize, 14usize, 14usize, 16usize],
        [2, 2],
        [2, 2],
        &mut t_13,
        [1usize, 7usize, 7usize, 16usize],
    );
    op_ticks[7usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(&t_13, &mut t_14);
    op_ticks[8usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fully_connected_relu(&t_14, 784usize, t_6, t_1, &mut t_15, 64usize);
    op_ticks[9usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fully_connected(&t_15, 64usize, t_5, t_3, &mut t_16, 10usize);
    op_ticks[10usize] += get_tick() - __t0;
    t_16
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
#[repr(align(4))]
struct AlignedBytes<const N: usize>([u8; N]);
static TENSOR_DATA_BYTES: AlignedBytes<220240usize> = AlignedBytes(
    *include_bytes!("weights.bin"),
);
const TENSOR_DATA_FLOATS: usize = 55060usize;
const T_1_OFFSET: usize = 54410usize;
const T_1_LEN: usize = 64usize;
const T_2_OFFSET: usize = 54389usize;
const T_2_LEN: usize = 16usize;
const T_3_OFFSET: usize = 54376usize;
const T_3_LEN: usize = 10usize;
const T_4_OFFSET: usize = 51173usize;
const T_4_LEN: usize = 3200usize;
const T_5_OFFSET: usize = 50530usize;
const T_5_LEN: usize = 640usize;
const T_6_OFFSET: usize = 351usize;
const T_6_LEN: usize = 50176usize;
const T_7_OFFSET: usize = 346usize;
const T_7_LEN: usize = 2usize;
const T_8_OFFSET: usize = 143usize;
const T_8_LEN: usize = 200usize;
const T_9_OFFSET: usize = 132usize;
const T_9_LEN: usize = 8usize;
fn tensor_data_f32() -> &'static [f32] {
    unsafe {
        core::slice::from_raw_parts(
            TENSOR_DATA_BYTES.0.as_ptr() as *const f32,
            TENSOR_DATA_FLOATS,
        )
    }
}
