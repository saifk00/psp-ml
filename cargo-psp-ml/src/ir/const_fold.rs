//! Compile-time constant folding on PspModel graph.
//!
//! Evaluates ops whose inputs are all `TensorKind::Constant` (e.g. INT32 shape
//! computation subgraphs), promotes their outputs to constants, and removes the ops.

use super::graph::{DType, Graph, TensorKind};
use super::psp::{BinaryOp, PspModel, PspOp, ReduceOp};

pub fn fold(model: &mut PspModel) {
    // Clone ops so we can iterate without borrowing model
    let ops: Vec<PspOp> = model.graph.ops.clone();
    let mut to_remove: Vec<usize> = Vec::new();

    for (op_idx, op) in ops.iter().enumerate() {
        let input_ids = op.inputs();

        // Shape op: input doesn't need to be constant (we read the shape metadata)
        let all_const = match op {
            PspOp::Shape { .. } => true,
            _ => input_ids
                .iter()
                .all(|id| matches!(model.graph.tensor(*id).kind, TensorKind::Constant { .. })),
        };
        if !all_const {
            continue;
        }

        match op {
            PspOp::Shape { input, output } => {
                let vals: Vec<i32> = model
                    .graph
                    .tensor(*input)
                    .shape
                    .iter()
                    .map(|&s| s as i32)
                    .collect();
                store_i32(&mut model.model_data, &mut model.graph, *output, &vals);
                to_remove.push(op_idx);
            }
            PspOp::Pack {
                inputs,
                output,
                axis,
            } => {
                if *axis == 0 {
                    let mut result = Vec::new();
                    for id in inputs {
                        result.extend(read_i32(&model.model_data, &model.graph, *id));
                    }
                    store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                    to_remove.push(op_idx);
                }
            }
            PspOp::StridedSlice {
                input,
                begin,
                end,
                strides,
                output,
                begin_mask,
                end_mask,
                shrink_axis_mask,
            } => {
                let data = read_i32(&model.model_data, &model.graph, *input);
                let begins = read_i32(&model.model_data, &model.graph, *begin);
                let ends = read_i32(&model.model_data, &model.graph, *end);
                let strs = read_i32(&model.model_data, &model.graph, *strides);

                if let Some(result) =
                    eval_strided_slice(&data, &begins, &ends, &strs, *begin_mask, *end_mask, *shrink_axis_mask)
                {
                    store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                    to_remove.push(op_idx);
                }
            }
            PspOp::Concatenation { inputs, output, .. } => {
                let mut result = Vec::new();
                for id in inputs {
                    result.extend(read_i32(&model.model_data, &model.graph, *id));
                }
                store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                to_remove.push(op_idx);
            }
            PspOp::Gather {
                input,
                indices,
                output,
                axis,
            } => {
                if *axis == 0 {
                    let data = read_i32(&model.model_data, &model.graph, *input);
                    let idxs = read_i32(&model.model_data, &model.graph, *indices);
                    let result: Vec<i32> = idxs.iter().map(|&i| data[i as usize]).collect();
                    store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                    to_remove.push(op_idx);
                }
            }
            PspOp::Reduce {
                op: reduce_op,
                input,
                output,
                ..
            } => {
                if model.graph.tensor(*input).dtype != DType::I32 {
                    continue;
                }
                let data = read_i32(&model.model_data, &model.graph, *input);
                let result = match reduce_op {
                    ReduceOp::Prod => data.iter().copied().product::<i32>(),
                    ReduceOp::Max => data.iter().copied().max().unwrap_or(0),
                    ReduceOp::Min => data.iter().copied().min().unwrap_or(0),
                    ReduceOp::Mean => data.iter().copied().sum::<i32>() / data.len() as i32,
                };
                store_i32(&mut model.model_data, &mut model.graph, *output, &[result]);
                to_remove.push(op_idx);
            }
            PspOp::Range {
                start,
                limit,
                delta,
                output,
            } => {
                let s = read_i32(&model.model_data, &model.graph, *start)[0];
                let l = read_i32(&model.model_data, &model.graph, *limit)[0];
                let d = read_i32(&model.model_data, &model.graph, *delta)[0];
                let mut result = Vec::new();
                let mut v = s;
                while (d > 0 && v < l) || (d < 0 && v > l) {
                    result.push(v);
                    v += d;
                }
                store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                to_remove.push(op_idx);
            }
            PspOp::SplitV {
                input,
                size_splits,
                outputs,
                ..
            } => {
                let data = read_i32(&model.model_data, &model.graph, *input);
                let splits = read_i32(&model.model_data, &model.graph, *size_splits);
                let mut offset = 0usize;
                for (i, &sz) in splits.iter().enumerate() {
                    let sz = sz as usize;
                    let chunk: Vec<i32> = data[offset..offset + sz].to_vec();
                    store_i32(&mut model.model_data, &mut model.graph, outputs[i], &chunk);
                    offset += sz;
                }
                to_remove.push(op_idx);
            }
            PspOp::Cast { input, output, .. } => {
                let t = model.graph.tensor(*input);
                if let TensorKind::Constant { offset, len } = t.kind {
                    let bytes = model.model_data[offset..offset + len].to_vec();
                    store_bytes(&mut model.model_data, &mut model.graph, *output, &bytes);
                    to_remove.push(op_idx);
                }
            }
            PspOp::ElementWise {
                op: binary_op,
                input_a,
                input_b,
                output,
            } => {
                if model.graph.tensor(*input_a).dtype != DType::I32 {
                    continue;
                }
                let a = read_i32(&model.model_data, &model.graph, *input_a);
                let b = read_i32(&model.model_data, &model.graph, *input_b);
                let result = eval_binary_i32(&a, &b, *binary_op);
                store_i32(&mut model.model_data, &mut model.graph, *output, &result);
                to_remove.push(op_idx);
            }
            PspOp::Reshape { input, output } => {
                if model.graph.tensor(*input).dtype != DType::I32 {
                    continue;
                }
                let t = model.graph.tensor(*input);
                if let TensorKind::Constant { offset, len } = t.kind {
                    let bytes = model.model_data[offset..offset + len].to_vec();
                    store_bytes(&mut model.model_data, &mut model.graph, *output, &bytes);
                    to_remove.push(op_idx);
                }
            }
            _ => {}
        }
    }

    let removed = to_remove.len();
    for idx in to_remove.into_iter().rev() {
        model.graph.ops.remove(idx);
    }

    if removed > 0 {
        eprintln!(
            "const_fold: folded {} ops, {} remain",
            removed,
            model.graph.ops.len()
        );
    }
}

fn store_i32(data: &mut Vec<u8>, graph: &mut Graph<PspOp>, id: usize, vals: &[i32]) {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    store_bytes(data, graph, id, &bytes);
}

fn store_bytes(data: &mut Vec<u8>, graph: &mut Graph<PspOp>, id: usize, bytes: &[u8]) {
    // 4-byte align
    while data.len() % 4 != 0 {
        data.push(0);
    }
    let offset = data.len();
    data.extend_from_slice(bytes);
    graph.tensor_mut(id).kind = TensorKind::Constant {
        offset,
        len: bytes.len(),
    };
}

fn read_i32(data: &[u8], graph: &Graph<PspOp>, id: usize) -> Vec<i32> {
    let t = graph.tensor(id);
    if let TensorKind::Constant { offset, len } = t.kind {
        data[offset..offset + len]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    } else {
        panic!("expected constant tensor {}", id)
    }
}

fn eval_binary_i32(a: &[i32], b: &[i32], op: BinaryOp) -> Vec<i32> {
    let op_fn: fn(i32, i32) -> i32 = match op {
        BinaryOp::Add => |a, b| a + b,
        BinaryOp::Mul => |a, b| a * b,
        BinaryOp::Sub => |a, b| a - b,
        BinaryOp::Div => |a, b| a / b,
        BinaryOp::FloorDiv => |a, b| a.div_euclid(b),
        BinaryOp::Max => |a, b| a.max(b),
        BinaryOp::Pow => |a, b| a.pow(b as u32),
    };
    if b.len() == a.len() {
        a.iter().zip(b.iter()).map(|(&x, &y)| op_fn(x, y)).collect()
    } else if b.len() == 1 {
        a.iter().map(|&x| op_fn(x, b[0])).collect()
    } else {
        a.iter()
            .enumerate()
            .map(|(i, &x)| op_fn(x, b[i % b.len()]))
            .collect()
    }
}

fn eval_strided_slice(
    data: &[i32],
    begins: &[i32],
    ends: &[i32],
    strides: &[i32],
    begin_mask: i32,
    end_mask: i32,
    shrink_axis_mask: i32,
) -> Option<Vec<i32>> {
    // 1-D case only (sufficient for BirdNET shape computation)
    if begins.len() != 1 {
        return None;
    }
    let len = data.len() as i32;
    let stride = strides[0];

    let mut b = if begin_mask & 1 != 0 {
        if stride > 0 { 0 } else { len - 1 }
    } else {
        let mut v = begins[0];
        if v < 0 { v += len; }
        v
    };

    let e = if end_mask & 1 != 0 {
        if stride > 0 { len } else { -1 }
    } else {
        let mut v = ends[0];
        if v < 0 { v += len; }
        v
    };

    if shrink_axis_mask & 1 != 0 {
        // Shrink: return single element as scalar
        return Some(vec![data[b as usize]]);
    }

    let mut result = Vec::new();
    while (stride > 0 && b < e) || (stride < 0 && b > e) {
        result.push(data[b as usize]);
        b += stride;
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::{DType, Graph, TensorKind};

    fn make_model() -> PspModel {
        PspModel {
            graph: Graph::new(),
            model_data: Vec::new(),
        }
    }

    fn add_const_i32(model: &mut PspModel, shape: Vec<usize>, vals: &[i32]) -> usize {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        while model.model_data.len() % 4 != 0 {
            model.model_data.push(0);
        }
        let offset = model.model_data.len();
        let len = bytes.len();
        model.model_data.extend_from_slice(&bytes);
        model.graph.add_tensor(shape, DType::I32, TensorKind::Constant { offset, len })
    }

    fn add_intermediate(model: &mut PspModel, shape: Vec<usize>, dtype: DType) -> usize {
        model.graph.add_tensor(shape, dtype, TensorKind::Intermediate)
    }

    fn read_result(model: &PspModel, id: usize) -> Vec<i32> {
        read_i32(&model.model_data, &model.graph, id)
    }

    #[test]
    fn fold_shape() {
        let mut model = make_model();
        let input = add_intermediate(&mut model, vec![1, 28, 28, 3], DType::F32);
        let output = add_intermediate(&mut model, vec![4], DType::I32);
        model.graph.ops.push(PspOp::Shape { input, output });

        fold(&mut model);
        assert!(model.graph.ops.is_empty());
        assert_eq!(read_result(&model, output), vec![1, 28, 28, 3]);
    }

    #[test]
    fn fold_pack_scalars() {
        let mut model = make_model();
        let a = add_const_i32(&mut model, vec![1], &[4]);
        let b = add_const_i32(&mut model, vec![1], &[5]);
        let c = add_const_i32(&mut model, vec![1], &[6]);
        let d = add_const_i32(&mut model, vec![1], &[7]);
        let output = add_intermediate(&mut model, vec![4], DType::I32);
        model.graph.ops.push(PspOp::Pack {
            inputs: vec![a, b, c, d],
            output,
            axis: 0,
        });

        fold(&mut model);
        assert!(model.graph.ops.is_empty());
        assert_eq!(read_result(&model, output), vec![4, 5, 6, 7]);
    }

    #[test]
    fn fold_strided_slice() {
        let mut model = make_model();
        let input = add_const_i32(&mut model, vec![4], &[10, 20, 30, 40]);
        let begin = add_const_i32(&mut model, vec![1], &[1]);
        let end = add_const_i32(&mut model, vec![1], &[3]);
        let strides = add_const_i32(&mut model, vec![1], &[1]);
        let output = add_intermediate(&mut model, vec![2], DType::I32);
        model.graph.ops.push(PspOp::StridedSlice {
            input,
            begin,
            end,
            strides,
            output,
            begin_mask: 0,
            end_mask: 0,
            shrink_axis_mask: 0,
        });

        fold(&mut model);
        assert!(model.graph.ops.is_empty());
        assert_eq!(read_result(&model, output), vec![20, 30]);
    }

    #[test]
    fn fold_strided_slice_shrink() {
        let mut model = make_model();
        let input = add_const_i32(&mut model, vec![4], &[10, 20, 30, 40]);
        let begin = add_const_i32(&mut model, vec![1], &[2]);
        let end = add_const_i32(&mut model, vec![1], &[3]);
        let strides = add_const_i32(&mut model, vec![1], &[1]);
        let output = add_intermediate(&mut model, vec![], DType::I32);
        model.graph.ops.push(PspOp::StridedSlice {
            input,
            begin,
            end,
            strides,
            output,
            begin_mask: 0,
            end_mask: 0,
            shrink_axis_mask: 1,
        });

        fold(&mut model);
        assert!(model.graph.ops.is_empty());
        assert_eq!(read_result(&model, output), vec![30]);
    }

    #[test]
    fn fold_binary_i32() {
        let mut model = make_model();
        let a = add_const_i32(&mut model, vec![3], &[10, 20, 30]);
        let b = add_const_i32(&mut model, vec![3], &[1, 2, 3]);
        let output = add_intermediate(&mut model, vec![3], DType::I32);
        model.graph.ops.push(PspOp::ElementWise {
            op: BinaryOp::Add,
            input_a: a,
            input_b: b,
            output,
        });

        fold(&mut model);
        assert!(model.graph.ops.is_empty());
        assert_eq!(read_result(&model, output), vec![11, 22, 33]);
    }

    #[test]
    fn fold_floor_div() {
        let mut model = make_model();
        let a = add_const_i32(&mut model, vec![3], &[7, 10, -7]);
        let b = add_const_i32(&mut model, vec![1], &[3]);
        let output = add_intermediate(&mut model, vec![3], DType::I32);
        model.graph.ops.push(PspOp::ElementWise {
            op: BinaryOp::FloorDiv,
            input_a: a,
            input_b: b,
            output,
        });

        fold(&mut model);
        assert!(model.graph.ops.is_empty());
        assert_eq!(read_result(&model, output), vec![2, 3, -3]);
    }

    #[test]
    fn fold_reduce_prod() {
        let mut model = make_model();
        let input = add_const_i32(&mut model, vec![3], &[2, 3, 4]);
        let axes = add_const_i32(&mut model, vec![1], &[0]);
        let output = add_intermediate(&mut model, vec![1], DType::I32);
        model.graph.ops.push(PspOp::Reduce { op: ReduceOp::Prod, input, axes, output });

        fold(&mut model);
        assert!(model.graph.ops.is_empty());
        assert_eq!(read_result(&model, output), vec![24]);
    }

    #[test]
    fn fold_range() {
        let mut model = make_model();
        let start = add_const_i32(&mut model, vec![], &[0]);
        let limit = add_const_i32(&mut model, vec![], &[5]);
        let delta = add_const_i32(&mut model, vec![], &[2]);
        let output = add_intermediate(&mut model, vec![3], DType::I32);
        model.graph.ops.push(PspOp::Range { start, limit, delta, output });

        fold(&mut model);
        assert!(model.graph.ops.is_empty());
        assert_eq!(read_result(&model, output), vec![0, 2, 4]);
    }

    #[test]
    fn fold_gather() {
        let mut model = make_model();
        let input = add_const_i32(&mut model, vec![4], &[10, 20, 30, 40]);
        let indices = add_const_i32(&mut model, vec![2], &[3, 1]);
        let output = add_intermediate(&mut model, vec![2], DType::I32);
        model.graph.ops.push(PspOp::Gather { input, indices, output, axis: 0 });

        fold(&mut model);
        assert!(model.graph.ops.is_empty());
        assert_eq!(read_result(&model, output), vec![40, 20]);
    }

    #[test]
    fn fold_chain() {
        // SHAPE → STRIDED_SLICE → PACK chain
        let mut model = make_model();
        let t0 = add_intermediate(&mut model, vec![1, 28, 28, 3], DType::F32);
        let shape_out = add_intermediate(&mut model, vec![4], DType::I32);
        let begin = add_const_i32(&mut model, vec![1], &[1]);
        let end = add_const_i32(&mut model, vec![1], &[2]);
        let strides = add_const_i32(&mut model, vec![1], &[1]);
        let slice_out = add_intermediate(&mut model, vec![], DType::I32);
        let scalar2 = add_const_i32(&mut model, vec![1], &[100]);
        let pack_out = add_intermediate(&mut model, vec![2], DType::I32);

        model.graph.ops.push(PspOp::Shape { input: t0, output: shape_out });
        model.graph.ops.push(PspOp::StridedSlice {
            input: shape_out,
            begin,
            end,
            strides,
            output: slice_out,
            begin_mask: 0,
            end_mask: 0,
            shrink_axis_mask: 1,
        });
        model.graph.ops.push(PspOp::Pack {
            inputs: vec![slice_out, scalar2],
            output: pack_out,
            axis: 0,
        });

        fold(&mut model);
        assert!(model.graph.ops.is_empty());
        assert_eq!(read_result(&model, pack_out), vec![28, 100]);
    }

    #[test]
    fn fold_skips_float32() {
        let mut model = make_model();
        // F32 constants should not be folded by ElementWise
        let bytes_a: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let bytes_b: Vec<u8> = [3.0f32, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let offset_a = 0;
        model.model_data.extend_from_slice(&bytes_a);
        let offset_b = model.model_data.len();
        model.model_data.extend_from_slice(&bytes_b);
        let a = model.graph.add_tensor(vec![2], DType::F32, TensorKind::Constant { offset: offset_a, len: 8 });
        let b = model.graph.add_tensor(vec![2], DType::F32, TensorKind::Constant { offset: offset_b, len: 8 });
        let output = add_intermediate(&mut model, vec![2], DType::F32);
        model.graph.ops.push(PspOp::ElementWise {
            op: BinaryOp::Add,
            input_a: a,
            input_b: b,
            output,
        });

        fold(&mut model);
        assert_eq!(model.graph.ops.len(), 1); // not folded
    }
}
