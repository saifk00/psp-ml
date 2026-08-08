//! Naive reference implementations - correct but slow.
//! Use these for testing and as baseline benchmarks.

use core::option::Option;

/// 2D Convolution (NHWC, naive) with padding support
///
/// - `input`:  [N, H, W, Ci]
/// - `filter`: [Co, Kh, Kw, Ci]
/// - `bias`:   [Co]
/// - `padding`: [pad_top, pad_bottom, pad_left, pad_right]
/// - `output`: [N, Ho, Wo, Co]
pub fn conv2d(
    input: &[f32],
    input_shape: [usize; 4],
    filter: &[f32],
    filter_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    let [n, h, w, ci] = input_shape;
    let [co, kh, kw, _] = filter_shape;
    let [_, ho, wo, _] = output_shape;
    let [sh, sw] = stride;
    let [pad_top, _pad_bottom, pad_left, _pad_right] = padding;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                for oc in 0..co {
                    let mut sum = bias.map_or(0.0, |b| b[oc]);
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy_padded = oy * sh + ky;
                            let ix_padded = ox * sw + kx;

                            if iy_padded < pad_top || ix_padded < pad_left {
                                continue;
                            }
                            let iy = iy_padded - pad_top;
                            let ix = ix_padded - pad_left;

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
/// - `padding`: [pad_top, pad_bottom, pad_left, pad_right]
/// - `output`: [N, Ho, Wo, Co]
pub fn conv2d_relu(
    input: &[f32],
    input_shape: [usize; 4],
    filter: &[f32],
    filter_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 4],
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

/// 2D Max Pooling (NHWC, naive) with padding support
///
/// - `input`:  [N, H, W, C]
/// - `padding`: [pad_top, pad_bottom, pad_left, pad_right]
/// - `output`: [N, Ho, Wo, C]
pub fn max_pool2d(
    input: &[f32],
    input_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    let [n, h, w, c] = input_shape;
    let [kh, kw] = kernel;
    let [sh, sw] = stride;
    let [_, ho, wo, _] = output_shape;
    let [pad_top, _, pad_left, _] = padding;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                for ch in 0..c {
                    let mut max_val = f32::NEG_INFINITY;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oy * sh + ky) as isize - pad_top as isize;
                            let ix = (ox * sw + kx) as isize - pad_left as isize;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                let in_idx = batch * (h * w * c) + (iy as usize) * (w * c) + (ix as usize) * c + ch;
                                if input[in_idx] > max_val {
                                    max_val = input[in_idx];
                                }
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

/// 2D Average Pooling (NHWC, naive) with padding support
///
/// - `input`:  [N, H, W, C]
/// - `padding`: [pad_top, pad_bottom, pad_left, pad_right]
/// - `output`: [N, Ho, Wo, C]
pub fn average_pool2d(
    input: &[f32],
    input_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    let [n, h, w, c] = input_shape;
    let [kh, kw] = kernel;
    let [sh, sw] = stride;
    let [_, ho, wo, _] = output_shape;
    let [pad_top, _, pad_left, _] = padding;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oy * sh + ky) as isize - pad_top as isize;
                            let ix = (ox * sw + kx) as isize - pad_left as isize;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                let in_idx = batch * (h * w * c) + (iy as usize) * (w * c) + (ix as usize) * c + ch;
                                sum += input[in_idx];
                                count += 1;
                            }
                        }
                    }
                    let out_idx = batch * (ho * wo * c) + oy * (wo * c) + ox * c + ch;
                    output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
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
    bias: Option<&[f32]>,
    output: &mut [f32],
    out_features: usize,
) {
    for o in 0..out_features {
        let mut sum = bias.map_or(0.0, |b| b[o]);
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
unary_op_kernel!(unary_relu, |x: f32| if x < 0.0 { 0.0 } else { x });

// ─── Quantization ops ──────────────────────────────────────────

/// TFLite QUANTIZE simulated in f32: snap each value onto the int8 grid
/// `out = (clamp(round(x/s) + z, -128, 127) - z) * s`. `libm::roundf` rounds
/// half away from zero, matching TfLiteRound.
pub fn fake_quant(input: &[f32], output: &mut [f32], scale: f32, zero_point: i32) {
    let inv = 1.0 / scale;
    let zp = zero_point as f32;
    for i in 0..input.len() {
        let mut q = libm::roundf(input[i] * inv) + zp;
        if q < -128.0 {
            q = -128.0;
        }
        if q > 127.0 {
            q = 127.0;
        }
        output[i] = (q - zp) * scale;
    }
}

/// 2D Convolution with int8 weights dequantized on the fly (NHWC, naive).
///
/// The integer weight products are accumulated in f32 and scaled once per
/// output channel: `out = bias + scales[oc] * sum(in * wq)`. Weight
/// zero-points are 0 (TFLite per-channel int8 spec).
///
/// - `input`:  [N, H, W, Ci] (f32, dequantized real values)
/// - `filter`: [Co, Kh, Kw, Ci] (int8)
/// - `scales`: [Co]
pub fn conv2d_q8(
    input: &[f32],
    input_shape: [usize; 4],
    filter: &[i8],
    filter_shape: [usize; 4],
    scales: &[f32],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    let [n, h, w, ci] = input_shape;
    let [co, kh, kw, _] = filter_shape;
    let [_, ho, wo, _] = output_shape;
    let [sh, sw] = stride;
    let [pad_top, _pad_bottom, pad_left, _pad_right] = padding;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                for oc in 0..co {
                    let mut sum = 0.0f32;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy_padded = oy * sh + ky;
                            let ix_padded = ox * sw + kx;
                            if iy_padded < pad_top || ix_padded < pad_left {
                                continue;
                            }
                            let iy = iy_padded - pad_top;
                            let ix = ix_padded - pad_left;
                            if iy >= h || ix >= w {
                                continue;
                            }
                            for ic in 0..ci {
                                let in_idx = batch * (h * w * ci) + iy * (w * ci) + ix * ci + ic;
                                let f_idx = oc * (kh * kw * ci) + ky * (kw * ci) + kx * ci + ic;
                                sum += input[in_idx] * filter[f_idx] as f32;
                            }
                        }
                    }
                    let out_idx = batch * (ho * wo * co) + oy * (wo * co) + ox * co + oc;
                    output[out_idx] = bias.map_or(0.0, |b| b[oc]) + sum * scales[oc];
                }
            }
        }
    }
}

/// `conv2d_q8` followed by ReLU.
pub fn conv2d_relu_q8(
    input: &[f32],
    input_shape: [usize; 4],
    filter: &[i8],
    filter_shape: [usize; 4],
    scales: &[f32],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    conv2d_q8(
        input, input_shape, filter, filter_shape, scales, bias, stride, padding, output,
        output_shape,
    );
    for v in output.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
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
    bias: Option<&[f32]>,
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
    padding: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
) {
    let [n, h, w, c] = input_shape;
    let [_, kh, kw, _] = filter_shape;
    let [_, ho, wo, _] = output_shape;
    let [sh, sw] = stride;
    let [pad_top, _pad_bottom, pad_left, _pad_right] = padding;

    for batch in 0..n {
        for oy in 0..ho {
            for ox in 0..wo {
                for ch in 0..c {
                    let mut sum = bias.map_or(0.0, |b| b[ch]);
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy_padded = oy * sh + ky;
                            let ix_padded = ox * sw + kx;
                            if iy_padded < pad_top || ix_padded < pad_left {
                                continue;
                            }
                            let iy = iy_padded - pad_top;
                            let ix = ix_padded - pad_left;
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

// ─── StridedSlice / Gather ────────────────────────────────────

/// N-dimensional strided slice.
///
/// `begin`, `end`, `strides` are per-dimension (same length as `input_shape`).
/// Masks follow TFLite semantics: bit `i` set means dimension `i` is auto-filled.
/// `shrink_axis_mask` bit `i` set means dimension `i` is removed from output.
pub fn strided_slice(
    input: &[f32],
    input_shape: &[usize],
    output: &mut [f32],
    begin: &[i32],
    end: &[i32],
    strides: &[i32],
    begin_mask: i32,
    end_mask: i32,
    shrink_axis_mask: i32,
) {
    let ndim = input_shape.len();

    // Resolve begin/end for each dimension
    let mut resolved_begin = [0i32; 8];
    let mut resolved_end = [0i32; 8];
    let mut resolved_stride = [1i32; 8];
    for d in 0..ndim {
        let len = input_shape[d] as i32;
        let stride = strides[d];
        resolved_stride[d] = stride;

        resolved_begin[d] = if begin_mask & (1 << d) != 0 {
            if stride > 0 { 0 } else { len - 1 }
        } else {
            let mut v = begin[d];
            if v < 0 { v += len; }
            v
        };

        resolved_end[d] = if end_mask & (1 << d) != 0 {
            if stride > 0 { len } else { -1 }
        } else {
            let mut v = end[d];
            if v < 0 { v += len; }
            v
        };
    }

    // Compute input strides (row-major)
    let mut in_strides = [0usize; 8];
    in_strides[ndim - 1] = 1;
    for d in (0..ndim - 1).rev() {
        in_strides[d] = in_strides[d + 1] * input_shape[d + 1];
    }

    // Iterate output positions using a coordinate vector
    let mut coord = [0i32; 8];
    for d in 0..ndim {
        coord[d] = resolved_begin[d];
    }

    let mut out_idx = 0;
    loop {
        // Compute flat input index
        let mut in_idx = 0;
        for d in 0..ndim {
            in_idx += coord[d] as usize * in_strides[d];
        }
        output[out_idx] = input[in_idx];
        out_idx += 1;

        // Advance coordinate (innermost first)
        let mut d = ndim - 1;
        loop {
            // Skip shrunk dimensions
            if shrink_axis_mask & (1 << d) != 0 {
                if d == 0 { return; }
                d -= 1;
                continue;
            }
            coord[d] += resolved_stride[d];
            let done = if resolved_stride[d] > 0 {
                coord[d] < resolved_end[d]
            } else {
                coord[d] > resolved_end[d]
            };
            if done {
                break;
            }
            // Reset this dimension and carry to next
            coord[d] = resolved_begin[d];
            if d == 0 { return; }
            d -= 1;
        }
    }
}

/// Gather elements along an axis using integer indices.
///
/// Standard TFLite Gather semantics:
/// `output_shape = input_shape[:axis] + indices_shape + input_shape[axis+1:]`
pub fn gather(
    input: &[f32],
    input_shape: &[usize],
    indices: &[i32],
    output: &mut [f32],
    output_shape: &[usize],
    axis: usize,
) {
    let ndim_in = input_shape.len();

    // Compute flat size of prefix (dims before axis), inner (axis dim), suffix (dims after axis)
    let mut prefix_size = 1usize;
    for d in 0..axis {
        prefix_size *= input_shape[d];
    }
    let axis_size = input_shape[axis];
    let mut suffix_size = 1usize;
    for d in (axis + 1)..ndim_in {
        suffix_size *= input_shape[d];
    }
    let _ = axis_size; // used implicitly through indices

    let _num_indices = indices.len();

    // output layout: [prefix_size, num_indices, suffix_size] flattened
    let mut out_idx = 0;
    for p in 0..prefix_size {
        for &idx in indices {
            let src_base = p * (input_shape[axis] * suffix_size) + (idx as usize) * suffix_size;
            for s in 0..suffix_size {
                output[out_idx] = input[src_base + s];
                out_idx += 1;
            }
        }
    }
    let _ = output_shape; // shape is used by caller for allocation
}

// ─── RFFT kernels ─────────────────────────────────────────────

/// Bit-reverse an index within `bits` bits.
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Pack N real values into N/2 interleaved complex pairs in bit-reversed order.
///
/// `output` has N floats: N/2 complex pairs as [re0, im0, re1, im1, ...].
/// Complex pair k (in bit-reversed order) = (input[2*br(k)], input[2*br(k)+1]).
/// Batched real FFT over `frames` contiguous length-`n` frames.
///
/// `input` is `[frames, n]`, `output` is `[frames, n/2 + 1]` (real parts of
/// the frequency bins, matching TFLite RFFT2D + CAST(complex→f32)). `scratch`
/// holds one frame's packed complex data (`n` floats). `stage_twiddles` is
/// the per-stage twiddle tables concatenated in stage order — stage `s` has
/// `2^s` complex entries at float offset `(2^s - 1) * 2`.
pub fn rfft_batch(
    input: &[f32],
    stage_twiddles: &[f32],
    unpack_twiddles: &[f32],
    scratch: &mut [f32],
    output: &mut [f32],
    n: usize,
    frames: usize,
) {
    let n_complex = n / 2;
    let out_bins = n_complex + 1;
    let num_stages = n_complex.trailing_zeros() as usize;
    for f in 0..frames {
        let data = &mut scratch[..n];
        rfft_pack(&input[f * n..(f + 1) * n], data, n);
        for stage in 0..num_stages {
            let half_size = 1usize << stage;
            let tw_off = (half_size - 1) * 2;
            fft_butterfly_stage(
                data,
                &stage_twiddles[tw_off..tw_off + half_size * 2],
                n_complex,
                half_size,
            );
        }
        rfft_unpack(
            data,
            unpack_twiddles,
            &mut output[f * out_bins..(f + 1) * out_bins],
            n,
        );
    }
}

pub fn rfft_pack(input: &[f32], output: &mut [f32], n: usize) {
    let n_complex = n / 2;
    let bits = {
        let mut b = 0;
        let mut v = n_complex;
        while v > 1 {
            v >>= 1;
            b += 1;
        }
        b
    };
    for k in 0..n_complex {
        let br = bit_reverse(k, bits);
        output[2 * k] = input[2 * br];
        output[2 * k + 1] = input[2 * br + 1];
    }
}

/// One radix-2 DIT butterfly stage.
///
/// - `data`: N interleaved complex values [re, im, re, im, ...]
/// - `twiddles`: `half_size` interleaved [cos, -sin] pairs
/// - `n_complex`: total number of complex elements
/// - `half_size`: butterfly half-size for this stage (1, 2, 4, ..., n_complex/2)
pub fn fft_butterfly_stage(
    data: &mut [f32],
    twiddles: &[f32],
    n_complex: usize,
    half_size: usize,
) {
    let full_size = half_size * 2;
    let num_groups = n_complex / full_size;
    for group in 0..num_groups {
        let base = group * full_size;
        for j in 0..half_size {
            let tw_re = twiddles[2 * j];
            let tw_im = twiddles[2 * j + 1];

            let top = base + j;
            let bot = top + half_size;

            let top_re = data[2 * top];
            let top_im = data[2 * top + 1];
            let bot_re = data[2 * bot];
            let bot_im = data[2 * bot + 1];

            // twiddle * bottom: (tw_re + tw_im*i) * (bot_re + bot_im*i)
            let t_re = tw_re * bot_re - tw_im * bot_im;
            let t_im = tw_re * bot_im + tw_im * bot_re;

            data[2 * top] = top_re + t_re;
            data[2 * top + 1] = top_im + t_im;
            data[2 * bot] = top_re - t_re;
            data[2 * bot + 1] = top_im - t_im;
        }
    }
}

/// Unpack N/2-point complex FFT result to N/2+1 real-part frequency bins.
///
/// Given X[k] = FFT of the N/2-point complex sequence formed by packing N real values,
/// recovers the real parts of the N-point RFFT: Re(F[k]) for k = 0..N/2.
///
/// - `data`: N/2 interleaved complex values (FFT output)
/// - `twiddles`: N/4 interleaved [cos, -sin] pairs for the unpack stage
/// - `output`: N/2+1 real-part values
/// - `n`: original real sequence length
pub fn rfft_unpack(data: &[f32], twiddles: &[f32], output: &mut [f32], n: usize) {
    let nc = n / 2; // number of complex FFT points

    // F[0] = (X[0].re + X[0].im, 0) — both are purely real
    output[0] = data[0] + data[1];

    // F[N/2] = (X[0].re - X[0].im, 0) — Nyquist, also purely real
    output[nc] = data[0] - data[1];

    // F[k] for k = 1..N/2-1:
    // Using the identity:
    //   A[k] = 0.5 * (X[k] + X*[N/2-k])
    //   B[k] = -0.5j * (X[k] - X*[N/2-k])
    //   F[k] = A[k] + W_N^k * B[k]
    // We only need Re(F[k]).
    for k in 1..nc {
        let conj = nc - k;

        let xk_re = data[2 * k];
        let xk_im = data[2 * k + 1];
        let xc_re = data[2 * conj];
        let xc_im = data[2 * conj + 1];

        // A[k] = 0.5 * (X[k] + X*[conj])
        let a_re = 0.5 * (xk_re + xc_re);
        let _a_im = 0.5 * (xk_im - xc_im);

        // B[k] = -0.5j * (X[k] - X*[conj])
        //       = 0.5 * (xk_im + xc_im) + 0.5j * (xk_re - xc_re)  [NOT the version below]
        // Actually: -j * (a+bi) = b - ai
        // So B = -0.5j * ((xk_re - xc_re) + (xk_im + xc_im)i)
        //      = 0.5 * (xk_im + xc_im) + 0.5 * (-(xk_re - xc_re))i
        //      = 0.5 * (xk_im + xc_im) - 0.5 * (xk_re - xc_re)i
        let b_re = 0.5 * (xk_im + xc_im);
        let b_im = -0.5 * (xk_re - xc_re);

        // W_N^k = cos(2πk/N) - j*sin(2πk/N)
        // Twiddle for k: stored as [cos, -sin] pairs, but indexed by k-1 since k starts at 1
        let tw_re = twiddles[2 * (k - 1)];
        let tw_im = twiddles[2 * (k - 1) + 1]; // this is -sin

        // F[k] = A[k] + W * B[k]
        // Re(F[k]) = a_re + (tw_re * b_re - tw_im * b_im)
        output[k] = a_re + (tw_re * b_re - tw_im * b_im);
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::{vec, vec::Vec};
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

    /// Compute a naive DFT for reference, return real parts of first N/2+1 bins.
    fn naive_rfft_real_parts(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let mut result = Vec::with_capacity(n / 2 + 1);
        for k in 0..=n / 2 {
            let mut re = 0.0f64;
            for t in 0..n {
                let angle = -2.0 * core::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
                re += (input[t] as f64) * angle.cos();
            }
            result.push(re as f32);
        }
        result
    }

    #[test]
    fn test_rfft_n8() {
        // 8-point real FFT
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = 8;
        let nc = n / 2; // 4 complex points

        // Step 1: pack
        let mut packed = [0.0f32; 8]; // 4 complex pairs
        rfft_pack(&input, &mut packed, n);

        // Step 2: butterfly stages (log2(4) = 2 stages)
        // Stage 0: half_size=1
        let twiddles_s0 = [1.0f32, 0.0]; // W_1^0 = (1, 0)
        fft_butterfly_stage(&mut packed, &twiddles_s0, nc, 1);

        // Stage 1: half_size=2
        // W_4^0 = (1, 0), W_4^1 = (0, -1)
        let twiddles_s1 = [1.0f32, 0.0, 0.0, -1.0];
        fft_butterfly_stage(&mut packed, &twiddles_s1, nc, 2);

        // Step 3: unpack
        // Unpack twiddles: W_N^k for k=1..nc-1, stored as [cos, -sin]
        // W_8^1 = cos(π/4) - j*sin(π/4) = (0.7071, -0.7071)
        // W_8^2 = cos(π/2) - j*sin(π/2) = (0, -1)
        // W_8^3 = cos(3π/4) - j*sin(3π/4) = (-0.7071, -0.7071)
        use core::f32::consts::PI;
        let mut unpack_tw = [0.0f32; 6]; // 3 pairs for k=1,2,3
        for k in 1..nc {
            let angle = 2.0 * PI * (k as f32) / (n as f32);
            unpack_tw[2 * (k - 1)] = libm::cosf(angle);
            unpack_tw[2 * (k - 1) + 1] = -libm::sinf(angle);
        }

        let mut output = [0.0f32; 5]; // N/2+1 = 5 bins
        rfft_unpack(&packed, &unpack_tw, &mut output, n);

        // Compare against naive DFT
        let expected = naive_rfft_real_parts(&input);
        for k in 0..=nc {
            assert!(
                (output[k] - expected[k]).abs() < 1e-3,
                "bin {k}: got {}, expected {}",
                output[k],
                expected[k]
            );
        }
    }

    #[test]
    fn test_rfft_n16() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let n = 16;
        let nc = n / 2;

        let mut packed = vec![0.0f32; n];
        rfft_pack(&input, &mut packed, n);

        use core::f32::consts::PI;

        // 3 butterfly stages for nc=8
        for stage in 0..3 {
            let half_size = 1 << stage;
            let mut twiddles = vec![0.0f32; half_size * 2];
            for j in 0..half_size {
                let angle = -2.0 * PI * (j as f32) / (2.0 * half_size as f32);
                twiddles[2 * j] = libm::cosf(angle);
                twiddles[2 * j + 1] = libm::sinf(angle);
            }
            fft_butterfly_stage(&mut packed, &twiddles, nc, half_size);
        }

        // Unpack twiddles
        let mut unpack_tw = vec![0.0f32; (nc - 1) * 2];
        for k in 1..nc {
            let angle = 2.0 * PI * (k as f32) / (n as f32);
            unpack_tw[2 * (k - 1)] = libm::cosf(angle);
            unpack_tw[2 * (k - 1) + 1] = -libm::sinf(angle);
        }

        let mut output = vec![0.0f32; nc + 1];
        rfft_unpack(&packed, &unpack_tw, &mut output, n);

        let expected = naive_rfft_real_parts(&input);
        for k in 0..=nc {
            assert!(
                (output[k] - expected[k]).abs() < 1e-3,
                "bin {k}: got {}, expected {}",
                output[k],
                expected[k]
            );
        }
    }
}
