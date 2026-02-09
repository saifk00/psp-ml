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
