//! Naive reference implementations - correct but slow.
//! Use these for testing and as baseline benchmarks.

use core::option::Option;

/// 2D Convolution (NHWC, naive) with padding support
///
/// - `input`:  [N, H, W, Ci]
/// - `filter`: [Co, Kh, Kw, Ci]
/// - `bias`:   [Co]
/// - `padding`: [pad_h, pad_w] - zero padding on each side
/// - `output`: [N, Ho, Wo, Co]
pub fn conv2d(
    input: &[f32],
    input_shape: [usize; 4],
    filter: &[f32],
    filter_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 2],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    let [n, h, w, ci] = input_shape;
    let [co, kh, kw, _] = filter_shape;
    let [_, ho, wo, _] = output_shape;
    let [sh, sw] = stride;
    let [pad_h, pad_w] = padding;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                for oc in 0..co {
                    let mut sum = bias.map_or(0.0, |b| b[oc]);
                    for ky in 0..kh {
                        for kx in 0..kw {
                            // Calculate input position with padding offset
                            let iy_padded = oy * sh + ky;
                            let ix_padded = ox * sw + kx;

                            // Check if within padding region (treat as 0)
                            if iy_padded < pad_h || ix_padded < pad_w {
                                continue;
                            }
                            let iy = iy_padded - pad_h;
                            let ix = ix_padded - pad_w;

                            // Check bounds
                            if iy >= h || ix >= w {
                                continue;
                            }

                            for ic in 0..ci {
                                let in_idx = batch * (h * w * ci) + iy * (w * ci) + ix * ci + ic;
                                let f_idx = oc * (kh * kw * ci) + ky * (kw * ci) + kx * ci + ic;
                                sum += input[in_idx] * filter[f_idx];
                            }
                        }
                    }
                    let out_idx = batch * (ho * wo * co) + oy * (wo * co) + ox * co + oc;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

/// 2D Convolution with ReLU (NHWC, naive)
///
/// - `input`:  [N, H, W, Ci]
/// - `filter`: [Co, Kh, Kw, Ci]
/// - `bias`:   [Co]
/// - `padding`: [pad_h, pad_w] - zero padding on each side
/// - `output`: [N, Ho, Wo, Co]
pub fn conv2d_relu(
    input: &[f32],
    input_shape: [usize; 4],
    filter: &[f32],
    filter_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 2],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    conv2d(input, input_shape, filter, filter_shape, bias, stride, padding, output, output_shape);
    for val in output.iter_mut() {
        if *val < 0.0 {
            *val = 0.0;
        }
    }
}

/// 2D Max Pooling (NHWC, naive)
///
/// - `input`:  [N, H, W, C]
/// - `output`: [N, Ho, Wo, C]
pub fn max_pool2d(
    input: &[f32],
    input_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    let [n, h, w, c] = input_shape;
    let [kh, kw] = kernel;
    let [sh, sw] = stride;
    let [_, ho, wo, _] = output_shape;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                for ch in 0..c {
                    let mut max_val = f32::NEG_INFINITY;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = oy * sh + ky;
                            let ix = ox * sw + kx;
                            let in_idx = batch * (h * w * c) + iy * (w * c) + ix * c + ch;
                            if input[in_idx] > max_val {
                                max_val = input[in_idx];
                            }
                        }
                    }
                    let out_idx = batch * (ho * wo * c) + oy * (wo * c) + ox * c + ch;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

/// Reshape (copy)
pub fn reshape(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i];
    }
}

/// Fully Connected (naive)
///
/// - `input`:   [in_features]
/// - `weights`: [out_features, in_features]
/// - `bias`:    [out_features]
/// - `output`:  [out_features]
pub fn fully_connected(
    input: &[f32],
    in_features: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    out_features: usize,
) {
    for o in 0..out_features {
        let mut sum = bias[o];
        for i in 0..in_features {
            sum += input[i] * weights[o * in_features + i];
        }
        output[o] = sum;
    }
}

// ─── Element-wise binary ops ────────────────────────────────────

macro_rules! binary_op_kernel {
    ($name:ident, $op:expr) => {
        pub fn $name(a: &[f32], b: &[f32], out: &mut [f32], b_len: usize) {
            let op: fn(f32, f32) -> f32 = $op;
            if b_len == out.len() {
                for i in 0..out.len() {
                    out[i] = op(a[i], b[i]);
                }
            } else if b_len == 1 {
                let s = b[0];
                for i in 0..out.len() {
                    out[i] = op(a[i], s);
                }
            } else {
                for i in 0..out.len() {
                    out[i] = op(a[i], b[i % b_len]);
                }
            }
        }
    };
}

binary_op_kernel!(binary_add, |a: f32, b: f32| a + b);
binary_op_kernel!(binary_mul, |a: f32, b: f32| a * b);
binary_op_kernel!(binary_sub, |a: f32, b: f32| a - b);
binary_op_kernel!(binary_div, |a: f32, b: f32| a / b);
binary_op_kernel!(binary_max, |a: f32, b: f32| if a > b { a } else { b });
binary_op_kernel!(binary_pow, |a: f32, b: f32| libm::powf(a, b));

// ─── Element-wise unary ops ─────────────────────────────────────

macro_rules! unary_op_kernel {
    ($name:ident, $op:expr) => {
        pub fn $name(input: &[f32], output: &mut [f32]) {
            let op: fn(f32) -> f32 = $op;
            for i in 0..input.len() {
                output[i] = op(input[i]);
            }
        }
    };
}

unary_op_kernel!(unary_logistic, |x: f32| 1.0 / (1.0 + libm::expf(-x)));

/// Fully Connected with ReLU (naive)
///
/// - `input`:   [in_features]
/// - `weights`: [out_features, in_features]
/// - `bias`:    [out_features]
/// - `output`:  [out_features]
pub fn fully_connected_relu(
    input: &[f32],
    in_features: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    out_features: usize,
) {
    fully_connected(input, in_features, weights, bias, output, out_features);
    for o in 0..out_features {
        if output[o] < 0.0 {
            output[o] = 0.0;
        }
    }
}

// ─── Reduction ops ─────────────────────────────────────────────

/// Reduce max: output[0] = max(input)
pub fn reduce_max(input: &[f32], output: &mut [f32]) {
    let mut val = f32::NEG_INFINITY;
    for &x in input {
        if x > val {
            val = x;
        }
    }
    output[0] = val;
}

/// Reduce min: output[0] = min(input)
pub fn reduce_min(input: &[f32], output: &mut [f32]) {
    let mut val = f32::INFINITY;
    for &x in input {
        if x < val {
            val = x;
        }
    }
    output[0] = val;
}

/// Reduce mean over all dims except the last (channel dim).
///
/// Input has N*C elements (NHWC flattened), output has C elements.
/// Each output[c] = mean of input[c, c+C, c+2C, ...].
pub fn reduce_mean_hw(input: &[f32], output: &mut [f32]) {
    let c = output.len();
    let n = input.len() / c;
    for ch in 0..c {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += input[i * c + ch];
        }
        output[ch] = sum / n as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add_same_shape() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        binary_add(&a, &b, &mut out, 4);
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_binary_mul_scalar_broadcast() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0];
        let mut out = [0.0f32; 3];
        binary_mul(&a, &b, &mut out, 1);
        assert_eq!(out, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_binary_sub_inner_broadcast() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0, 1.0, 1.0];
        let mut out = [0.0f32; 6];
        binary_sub(&a, &b, &mut out, 3);
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_binary_div() {
        let a = [10.0, 20.0];
        let b = [2.0, 5.0];
        let mut out = [0.0f32; 2];
        binary_div(&a, &b, &mut out, 2);
        assert_eq!(out, [5.0, 4.0]);
    }

    #[test]
    fn test_binary_max() {
        let a = [1.0, 5.0, 3.0];
        let b = [2.0, 4.0, 6.0];
        let mut out = [0.0f32; 3];
        binary_max(&a, &b, &mut out, 3);
        assert_eq!(out, [2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_binary_mul_spatial_broadcast() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [10.0, 10.0, 10.0];
        let mut out = [0.0f32; 6];
        binary_mul(&a, &b, &mut out, 3);
        assert_eq!(out, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn test_unary_logistic() {
        let input = [0.0, 1.0, -1.0, 10.0];
        let mut out = [0.0f32; 4];
        unary_logistic(&input, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.7310586).abs() < 1e-5);
        assert!((out[2] - 0.2689414).abs() < 1e-5);
        assert!((out[3] - 1.0).abs() < 1e-4);
    }
}
