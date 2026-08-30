//! Rewrite a hybrid-quantized graph (TFLite int8 QUANTIZE/DEQUANTIZE blocks)
//! into all-f32 "fake quant" semantics.
//!
//! The PSP's VFPU has no integer SIMD, so real int8 arithmetic would run on
//! the scalar pipeline and lose to the f32 VFPU kernels. Instead, every int8
//! *activation* tensor is reinterpreted as an f32 tensor holding dequantized
//! real values, snapped to the quantization grid at the points TFLite would
//! quantize (QAT-style simulation):
//!
//! - `QUANTIZE` (`PspOp::FakeQuant`) stays as an elementwise grid-snap:
//!   `out = (clamp(round(x/s) + z, -128, 127) - z) * s`.
//! - `DEQUANTIZE` becomes an identity and is aliased away (kept as a copy
//!   only when its output is the graph output).
//! - `CONV_2D` with int8 weights keeps the weights int8 in the blob (4x
//!   memory saving) and dequantizes them on the fly at the kernel boundary;
//!   this pass attaches the per-output-channel scale vector as a new f32
//!   constant and dequantizes the i32 bias into a new f32 constant.
//! - Int8 constants consumed by any other op are dequantized to f32
//!   constants outright.
//!
//! This matches TFLite's int8 reference kernels to within ~1 output quantum
//! per layer (f32 accumulate vs i32 accumulate + fixed-point requant), which
//! the QUANTIZE ops re-snap at every block boundary — but it is not
//! bit-exact by design.

use crate::ir::graph::{DType, TensorId, TensorKind};
use crate::ir::psp::{PspModel, PspOp};

pub fn rewrite(model: &mut PspModel) -> Result<(), String> {
    rewrite_int8_convs(model)?;
    dequantize_stray_int8_consts(model)?;
    eliminate_dequantize_ops(model);
    flip_activation_dtypes(model);
    validate(model)
}

/// Append raw f32 data to `model_data` as a new constant tensor.
fn append_f32_const(model: &mut PspModel, shape: Vec<usize>, data: &[f32]) -> TensorId {
    // Keep the blob offset 16-byte aligned so compact_blob's float math holds.
    while model.model_data.len() % 16 != 0 {
        model.model_data.push(0);
    }
    let offset = model.model_data.len();
    for v in data {
        model.model_data.extend_from_slice(&v.to_le_bytes());
    }
    model.graph.add_tensor(
        shape,
        DType::F32,
        TensorKind::Constant {
            offset,
            len: data.len() * 4,
        },
    )
}

fn read_const_bytes(model: &PspModel, id: TensorId) -> Result<(usize, usize), String> {
    match model.graph.tensor(id).kind {
        TensorKind::Constant { offset, len } => Ok((offset, len)),
        _ => Err(format!("tensor t{id} is not a constant")),
    }
}

/// Attach weight-scale constants and dequantized f32 biases to int8 convs.
fn rewrite_int8_convs(model: &mut PspModel) -> Result<(), String> {
    for op_idx in 0..model.graph.ops.len() {
        let PspOp::Conv2d {
            input,
            weights,
            bias,
            ..
        } = model.graph.ops[op_idx]
        else {
            continue;
        };
        if model.graph.tensor(weights).dtype != DType::I8 {
            continue;
        }

        let co = model.graph.tensor(weights).shape[0];
        let wq = model
            .graph
            .tensor(weights)
            .quant
            .clone()
            .ok_or_else(|| format!("int8 conv weights t{weights} missing quantization"))?;
        if wq.zero_point.iter().any(|&z| z != 0) {
            return Err(format!(
                "int8 conv weights t{weights} have non-zero zero_point (unsupported)"
            ));
        }
        // Broadcast a per-tensor scale to per-channel for a uniform kernel ABI.
        let scales: Vec<f32> = if wq.scale.len() == co {
            wq.scale.clone()
        } else if wq.scale.len() == 1 {
            vec![wq.scale[0]; co]
        } else {
            return Err(format!(
                "int8 conv weights t{weights}: {} scales for {} channels",
                wq.scale.len(),
                co
            ));
        };
        let scales_id = append_f32_const(model, vec![co], &scales);

        // Dequantize the i32 bias using its own scale (TFLite sets it to
        // s_input * s_weight[c], but trust the stored value).
        let new_bias = if let Some(b) = bias {
            let bt = model.graph.tensor(b);
            if bt.dtype != DType::I32 {
                return Err(format!(
                    "int8 conv t{weights}: expected i32 bias, got {:?}",
                    bt.dtype
                ));
            }
            let bq = bt
                .quant
                .clone()
                .ok_or_else(|| format!("int8 conv bias t{b} missing quantization"))?;
            let (offset, len) = read_const_bytes(model, b)?;
            let q: Vec<i32> = model.model_data[offset..offset + len]
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            if q.len() != co {
                return Err(format!("bias t{b}: {} values for {} channels", q.len(), co));
            }
            let scale_at = |i: usize| -> f32 {
                if bq.scale.len() == 1 {
                    bq.scale[0]
                } else {
                    bq.scale[i]
                }
            };
            let f: Vec<f32> = q
                .iter()
                .enumerate()
                .map(|(i, &v)| v as f32 * scale_at(i))
                .collect();
            Some(append_f32_const(model, vec![co], &f))
        } else {
            None
        };

        // The conv input must be a per-tensor quantized activation; verify so
        // failures surface at compile time, not as silent garbage.
        model
            .graph
            .tensor(input)
            .quant
            .as_ref()
            .ok_or_else(|| format!("int8 conv input t{input} missing quantization"))?
            .scalar()?;

        if let PspOp::Conv2d {
            bias,
            weight_scales,
            ..
        } = &mut model.graph.ops[op_idx]
        {
            *weight_scales = Some(scales_id);
            *bias = new_bias;
        }
    }
    Ok(())
}

/// Any int8 constant consumed by an op other than an int8-conv weight slot
/// gets dequantized to a plain f32 constant at compile time.
fn dequantize_stray_int8_consts(model: &mut PspModel) -> Result<(), String> {
    // Tensors legitimately staying i8: conv weight operands.
    let mut conv_weights = std::collections::HashSet::new();
    for op in &model.graph.ops {
        if let PspOp::Conv2d { weights, .. } = op {
            if model.graph.tensor(*weights).dtype == DType::I8 {
                conv_weights.insert(*weights);
            }
        }
    }

    let mut replacements: Vec<(TensorId, TensorId)> = Vec::new();
    for tid in 0..model.graph.tensors.len() {
        let t = &model.graph.tensors[tid];
        if t.dtype != DType::I8
            || conv_weights.contains(&tid)
            || !matches!(t.kind, TensorKind::Constant { .. })
        {
            continue;
        }
        let referenced = model.graph.ops.iter().any(|op| op.inputs().contains(&tid));
        if !referenced {
            continue;
        }
        let q = t
            .quant
            .clone()
            .ok_or_else(|| format!("int8 constant t{tid} missing quantization"))?;
        let (scale, zp) = q.scalar()?;
        let shape = t.shape.clone();
        let (offset, len) = read_const_bytes(model, tid)?;
        let f: Vec<f32> = model.model_data[offset..offset + len]
            .iter()
            .map(|&b| (b as i8 as i32 - zp) as f32 * scale)
            .collect();
        let new_id = append_f32_const(model, shape, &f);
        replacements.push((tid, new_id));
    }
    for (old, new) in replacements {
        for op in &mut model.graph.ops {
            op.replace_tensor_id(old, new);
        }
    }
    Ok(())
}

/// Alias every Dequantize away; keep a copy op only when the output is the
/// graph output (the buffer must exist for the caller).
fn eliminate_dequantize_ops(model: &mut PspModel) {
    let mut idx = 0;
    while idx < model.graph.ops.len() {
        let PspOp::Dequantize { input, output } = model.graph.ops[idx] else {
            idx += 1;
            continue;
        };
        let is_graph_output = matches!(model.graph.tensor(output).kind, TensorKind::Output)
            || model.graph.outputs.contains(&output);
        if is_graph_output {
            model.graph.ops[idx] = PspOp::Reshape {
                input,
                output,
                shape_tensor: None,
                builtin_shape: None,
            };
            idx += 1;
        } else {
            model.graph.ops.remove(idx);
            for op in &mut model.graph.ops {
                op.replace_tensor_id(output, input);
            }
        }
    }
}

/// Every int8 activation now semantically holds f32 dequantized values.
fn flip_activation_dtypes(model: &mut PspModel) {
    for t in &mut model.graph.tensors {
        if matches!(t.dtype, DType::I8 | DType::U8)
            && !matches!(t.kind, TensorKind::Constant { .. })
        {
            t.dtype = DType::F32;
        }
    }
}

fn validate(model: &PspModel) -> Result<(), String> {
    for op in &model.graph.ops {
        for id in op.inputs().into_iter().chain(op.all_outputs()) {
            let t = model.graph.tensor(id);
            let is_conv_weight = matches!(op, PspOp::Conv2d { weights, .. } if *weights == id);
            if matches!(t.dtype, DType::I8 | DType::U8) && !is_conv_weight {
                return Err(format!(
                    "tensor t{id} is still {:?} after quant rewrite (op: {op})",
                    t.dtype
                ));
            }
        }
        if let PspOp::FakeQuant { output, .. } = op {
            let t = model.graph.tensor(*output);
            t.quant
                .as_ref()
                .ok_or_else(|| format!("FakeQuant output t{output} missing quantization"))?
                .scalar()?;
        }
    }
    Ok(())
}
