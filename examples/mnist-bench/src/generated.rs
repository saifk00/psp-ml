//! Generated inference module
#[allow(unused_imports)]
use psp_ml::kernels::naive::{
    conv2d, conv2d_relu, max_pool2d, reshape, fully_connected, fully_connected_relu,
};
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
    conv2d_relu(
        input,
        [1usize, 28usize, 28usize, 1usize],
        t_8,
        [8usize, 5usize, 5usize, 1usize],
        Some(t_9),
        [1usize, 1usize],
        [2usize, 2usize],
        &mut t_10,
        [1usize, 28usize, 28usize, 8usize],
    );
    max_pool2d(
        &t_10,
        [1usize, 28usize, 28usize, 8usize],
        [2, 2],
        [2, 2],
        &mut t_11,
        [1usize, 14usize, 14usize, 8usize],
    );
    conv2d_relu(
        &t_11,
        [1usize, 14usize, 14usize, 8usize],
        t_4,
        [16usize, 5usize, 5usize, 8usize],
        Some(t_2),
        [1usize, 1usize],
        [2usize, 2usize],
        &mut t_12,
        [1usize, 14usize, 14usize, 16usize],
    );
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
