//! Weight-footprint tallying.
//!
//! Residency planning (which constants stay in RAM) moved to
//! [`crate::ir::residency`].

use crate::ir::PspModel;

/// Walks the graph in topological order and returns a vector parallel-indexed
/// to `model.graph.ops`, where entry `i` is the cumulative weight bytes
/// (sum of all `const_tensors().size_bytes()`) for ops `0..=i`.
///
/// The last entry is the total weight footprint of the model. Empty graphs
/// return an empty vector.
pub fn swap_analysis(model: &PspModel) -> Vec<usize> {
    let mut cumulative = Vec::with_capacity(model.graph.ops.len());
    let mut running = 0usize;
    for op in &model.graph.ops {
        running += op
            .const_tensors()
            .iter()
            .map(|&tid| {
                let sz = model.graph.tensor(tid).size_bytes();
                if sz > 0 {
                    println!("constant tensor with size: {} (op {})", sz, op)
                }
                sz
            })
            .sum::<usize>();
        cumulative.push(running);
    }
    cumulative
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::{DType, Graph, TensorKind};
    use crate::ir::psp::{Conv2dParams, FullyConnectedParams, PspOp};

    fn model_from(graph: Graph<PspOp>) -> PspModel {
        PspModel { graph, model_data: Vec::new() }
    }

    #[test]
    fn empty_graph_yields_empty() {
        let model = model_from(Graph::<PspOp>::new());
        assert!(swap_analysis(&model).is_empty());
    }

    #[test]
    fn single_fc_tallies_weight_bytes() {
        let mut g = Graph::<PspOp>::new();
        let input = g.add_tensor(vec![1, 4], DType::F32, TensorKind::Input);
        let w = g.add_tensor(
            vec![8, 4],
            DType::F32,
            TensorKind::Constant { offset: 0, len: 128 },
        );
        let out = g.add_tensor(vec![1, 8], DType::F32, TensorKind::Output);
        g.ops.push(PspOp::FullyConnected {
            input,
            weights: w,
            bias: None,
            output: out,
            fused_activation: FullyConnectedParams { fused_activation: None },
        });
        assert_eq!(swap_analysis(&model_from(g)), vec![128]);
    }

    #[test]
    fn two_weight_bearing_ops_accumulate() {
        let mut g = Graph::<PspOp>::new();
        let t_in = g.add_tensor(vec![1, 1, 1, 4], DType::F32, TensorKind::Input);
        let conv_w = g.add_tensor(
            vec![8, 1, 1, 4],
            DType::F32,
            TensorKind::Constant { offset: 0, len: 128 },
        );
        let conv_out = g.add_tensor(vec![1, 1, 1, 8], DType::F32, TensorKind::Intermediate);
        let fc_w = g.add_tensor(
            vec![16, 8],
            DType::F32,
            TensorKind::Constant { offset: 128, len: 512 },
        );
        let fc_out = g.add_tensor(vec![1, 16], DType::F32, TensorKind::Output);
        g.ops.push(PspOp::Conv2d {
            input: t_in,
            weights: conv_w,
            bias: None,
            output: conv_out,
            weight_scales: None,
            params: Conv2dParams {
                kernel_h: 1,
                kernel_w: 1,
                stride_h: 1,
                stride_w: 1,
                pad_top: 0,
                pad_bottom: 0,
                pad_left: 0,
                pad_right: 0,
                fused_activation: None,
            },
        });
        g.ops.push(PspOp::FullyConnected {
            input: conv_out,
            weights: fc_w,
            bias: None,
            output: fc_out,
            fused_activation: FullyConnectedParams { fused_activation: None },
        });
        assert_eq!(swap_analysis(&model_from(g)), vec![128, 640]);
    }

    #[test]
    fn non_weight_bearing_op_does_not_increase_total() {
        let mut g = Graph::<PspOp>::new();
        let t_in = g.add_tensor(vec![1, 1, 1, 4], DType::F32, TensorKind::Input);
        let conv_w = g.add_tensor(
            vec![8, 1, 1, 4],
            DType::F32,
            TensorKind::Constant { offset: 0, len: 128 },
        );
        let conv_out = g.add_tensor(vec![1, 1, 1, 8], DType::F32, TensorKind::Intermediate);
        let reshape_out = g.add_tensor(vec![1, 8], DType::F32, TensorKind::Output);
        g.ops.push(PspOp::Conv2d {
            input: t_in,
            weights: conv_w,
            bias: None,
            output: conv_out,
            weight_scales: None,
            params: Conv2dParams {
                kernel_h: 1,
                kernel_w: 1,
                stride_h: 1,
                stride_w: 1,
                pad_top: 0,
                pad_bottom: 0,
                pad_left: 0,
                pad_right: 0,
                fused_activation: None,
            },
        });
        g.ops.push(PspOp::Reshape {
            input: conv_out,
            output: reshape_out,
            shape_tensor: None,
            builtin_shape: Some(vec![1, 8]),
        });
        assert_eq!(swap_analysis(&model_from(g)), vec![128, 128]);
    }

}
