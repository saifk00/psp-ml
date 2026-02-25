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

/// 2D Average Pooling (NHWC, naive)
///
/// - `input`:  [N, H, W, C]
/// - `output`: [N, Ho, Wo, C]
pub fn average_pool2d(
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
    let pool_size = (kh * kw) as f32;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = oy * sh + ky;
                            let ix = ox * sw + kx;
                            let in_idx = batch * (h * w * c) + iy * (w * c) + ix * c + ch;
                            sum += input[in_idx];
                        }
                    }
                    let out_idx = batch * (ho * wo * c) + oy * (wo * c) + ox * c + ch;
                    output[out_idx] = sum / pool_size;
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

/// Reverse elements along a specified axis (up to 4D).
pub fn reverse_v2(input: &[f32], input_shape: &[usize], output: &mut [f32], axis: usize) {
    let ndim = input_shape.len();
    let pad = 4 - ndim;
    let mut s = [1usize; 4];
    for i in 0..ndim {
        s[pad + i] = input_shape[i];
    }
    let axis_4d = pad + axis;

    for i0 in 0..s[0] {
        for i1 in 0..s[1] {
            for i2 in 0..s[2] {
                for i3 in 0..s[3] {
                    let mut d = [i0, i1, i2, i3];
                    d[axis_4d] = s[axis_4d] - 1 - d[axis_4d];
                    let in_idx =
                        i0 * s[1] * s[2] * s[3] + i1 * s[2] * s[3] + i2 * s[3] + i3;
                    let out_idx =
                        d[0] * s[1] * s[2] * s[3] + d[1] * s[2] * s[3] + d[2] * s[3] + d[3];
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

/// Permute tensor dimensions (up to 4D).
///
/// Pads shapes/perm to 4D internally, then iterates all elements
/// mapping source indices through the permutation.
pub fn transpose(
    input: &[f32],
    input_shape: &[usize],
    output: &mut [f32],
    output_shape: &[usize],
    perm: &[usize],
) {
    let ndim = input_shape.len();
    let pad = 4 - ndim;
    let mut is = [1usize; 4];
    let mut os = [1usize; 4];
    let mut p = [0usize; 4];
    for i in 0..ndim {
        is[pad + i] = input_shape[i];
        os[pad + i] = output_shape[i];
    }
    for i in 0..pad {
        p[i] = i;
    }
    for i in 0..ndim {
        p[pad + i] = pad + perm[i];
    }

    for i0 in 0..is[0] {
        for i1 in 0..is[1] {
            for i2 in 0..is[2] {
                for i3 in 0..is[3] {
                    let src = [i0, i1, i2, i3];
                    let mut dst = [0usize; 4];
                    dst[p[0]] = src[0];
                    dst[p[1]] = src[1];
                    dst[p[2]] = src[2];
                    dst[p[3]] = src[3];
                    let in_idx =
                        i0 * is[1] * is[2] * is[3] + i1 * is[2] * is[3] + i2 * is[3] + i3;
                    let out_idx = dst[0] * os[1] * os[2] * os[3]
                        + dst[1] * os[2] * os[3]
                        + dst[2] * os[3]
                        + dst[3];
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

/// Depthwise 2D Convolution (NHWC, naive)
///
/// Each output channel depends only on the corresponding input channel.
/// - `input`:  [N, H, W, C]
/// - `filter`: [1, Kh, Kw, C]
/// - `bias`:   [C]
/// - `padding`: [pad_h, pad_w] - zero padding on each side
/// - `output`: [N, Ho, Wo, C]
pub fn depthwise_conv2d(
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
    let [n, h, w, c] = input_shape;
    let [_, kh, kw, _] = filter_shape;
    let [_, ho, wo, _] = output_shape;
    let [sh, sw] = stride;
    let [pad_h, pad_w] = padding;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                for ch in 0..c {
                    let mut sum = bias.map_or(0.0, |b| b[ch]);
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy_padded = oy * sh + ky;
                            let ix_padded = ox * sw + kx;
                            if iy_padded < pad_h || ix_padded < pad_w {
                                continue;
                            }
                            let iy = iy_padded - pad_h;
                            let ix = ix_padded - pad_w;
                            if iy >= h || ix >= w {
                                continue;
                            }
                            let in_idx = batch * (h * w * c) + iy * (w * c) + ix * c + ch;
                            let f_idx = ky * (kw * c) + kx * c + ch;
                            sum += input[in_idx] * filter[f_idx];
                        }
                    }
                    let out_idx = batch * (ho * wo * c) + oy * (wo * c) + ox * c + ch;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

/// Zero-pad an NHWC tensor.
///
/// - `input`:  [N, H, W, C]
/// - `output`: [N+pN, H+pH, W+pW, C+pC]
/// - `padding`: [[before, after]; 4] per NHWC dim
pub fn pad(
    input: &[f32],
    input_shape: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
    padding: [[usize; 2]; 4],
) {
    for v in output.iter_mut() {
        *v = 0.0;
    }
    let [n, h, w, c] = input_shape;
    let [_, o_h, o_w, o_c] = output_shape;
    for batch in 0..n {
        for iy in 0..h {
            for ix in 0..w {
                for ic in 0..c {
                    let ob = batch + padding[0][0];
                    let oy = iy + padding[1][0];
                    let ox = ix + padding[2][0];
                    let oc = ic + padding[3][0];
                    let in_idx = batch * (h * w * c) + iy * (w * c) + ix * c + ic;
                    let out_idx = ob * (o_h * o_w * o_c) + oy * (o_w * o_c) + ox * o_c + oc;
                    output[out_idx] = input[in_idx];
                }
            }
        }
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
