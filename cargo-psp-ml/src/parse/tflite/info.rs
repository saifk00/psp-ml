use std::collections::BTreeMap;

use super::{root_as_model, BuiltinOperator, Operator, Tensor, TensorType};

const SUPPORTED_OPS: &[BuiltinOperator] = &[
    BuiltinOperator::CONV_2D,
    BuiltinOperator::FULLY_CONNECTED,
    BuiltinOperator::MAX_POOL_2D,
    BuiltinOperator::RESHAPE,
    BuiltinOperator::SQUEEZE,
    BuiltinOperator::EXPAND_DIMS,
    BuiltinOperator::ADD,
    BuiltinOperator::MUL,
    BuiltinOperator::SUB,
    BuiltinOperator::DIV,
    BuiltinOperator::FLOOR_DIV,
    BuiltinOperator::MAXIMUM,
    BuiltinOperator::LOGISTIC,
    BuiltinOperator::SHAPE,
    BuiltinOperator::PACK,
    BuiltinOperator::STRIDED_SLICE,
    BuiltinOperator::CONCATENATION,
    BuiltinOperator::GATHER,
    BuiltinOperator::REDUCE_PROD,
    BuiltinOperator::RANGE,
    BuiltinOperator::SPLIT_V,
    BuiltinOperator::CAST,
    BuiltinOperator::REDUCE_MAX,
    BuiltinOperator::REDUCE_MIN,
    BuiltinOperator::POW,
    BuiltinOperator::MEAN,
    BuiltinOperator::PAD,
    BuiltinOperator::AVERAGE_POOL_2D,
    BuiltinOperator::TRANSPOSE,
    BuiltinOperator::REVERSE_V2,
    BuiltinOperator::DEPTHWISE_CONV_2D,
    BuiltinOperator::RFFT2D,
];

fn type_name(t: TensorType) -> &'static str {
    t.variant_name().unwrap_or("UNKNOWN")
}

fn op_name(code: BuiltinOperator) -> &'static str {
    code.variant_name().unwrap_or("UNKNOWN")
}

fn shape_str(tensor: &Tensor<'_>) -> String {
    match tensor.shape() {
        Some(s) => {
            let dims: Vec<String> = s.iter().map(|d| d.to_string()).collect();
            format!("[{}]", dims.join(", "))
        }
        None => "[]".to_string(),
    }
}

fn format_op_params(op: &Operator<'_>, builtin: BuiltinOperator) -> String {
    match builtin {
        BuiltinOperator::CONV_2D => {
            if let Some(opts) = op.builtin_options_as_conv_2_doptions() {
                format!(
                    "stride=[{},{}] pad={} act={} dilation=[{},{}]",
                    opts.stride_h(),
                    opts.stride_w(),
                    opts.padding().variant_name().unwrap_or("?"),
                    opts.fused_activation_function()
                        .variant_name()
                        .unwrap_or("?"),
                    opts.dilation_h_factor(),
                    opts.dilation_w_factor(),
                )
            } else {
                String::new()
            }
        }
        BuiltinOperator::DEPTHWISE_CONV_2D => {
            if let Some(opts) = op.builtin_options_as_depthwise_conv_2_doptions() {
                format!(
                    "stride=[{},{}] pad={} act={} depth_mul={} dilation=[{},{}]",
                    opts.stride_h(),
                    opts.stride_w(),
                    opts.padding().variant_name().unwrap_or("?"),
                    opts.fused_activation_function()
                        .variant_name()
                        .unwrap_or("?"),
                    opts.depth_multiplier(),
                    opts.dilation_h_factor(),
                    opts.dilation_w_factor(),
                )
            } else {
                String::new()
            }
        }
        BuiltinOperator::FULLY_CONNECTED => {
            if let Some(opts) = op.builtin_options_as_fully_connected_options() {
                format!(
                    "act={}",
                    opts.fused_activation_function()
                        .variant_name()
                        .unwrap_or("?"),
                )
            } else {
                String::new()
            }
        }
        BuiltinOperator::MAX_POOL_2D | BuiltinOperator::AVERAGE_POOL_2D => {
            if let Some(opts) = op.builtin_options_as_pool_2_doptions() {
                format!(
                    "filter=[{},{}] stride=[{},{}] pad={} act={}",
                    opts.filter_height(),
                    opts.filter_width(),
                    opts.stride_h(),
                    opts.stride_w(),
                    opts.padding().variant_name().unwrap_or("?"),
                    opts.fused_activation_function()
                        .variant_name()
                        .unwrap_or("?"),
                )
            } else {
                String::new()
            }
        }
        BuiltinOperator::CONCATENATION => {
            if let Some(opts) = op.builtin_options_as_concatenation_options() {
                format!("axis={}", opts.axis())
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}

pub fn dump_model_info(data: &[u8]) {
    let model = match root_as_model(data) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to parse TFLite model: {e}");
            return;
        }
    };

    println!("=== TFLite Model Info ===");
    println!("Version: {}", model.version());

    let opcodes = match model.operator_codes() {
        Some(o) => o,
        None => {
            println!("No operator codes found.");
            return;
        }
    };

    // Build opcode index -> BuiltinOperator mapping
    let mut opcode_names: Vec<BuiltinOperator> = Vec::new();
    println!("\n--- Operator Codes ({}) ---", opcodes.len());
    for i in 0..opcodes.len() {
        let oc = opcodes.get(i);
        let builtin = oc.builtin_code();
        let name = op_name(builtin);
        println!("  [{i:2}] {name}");
        opcode_names.push(builtin);
    }

    let subgraphs = match model.subgraphs() {
        Some(s) => s,
        None => {
            println!("No subgraphs found.");
            return;
        }
    };

    println!("\nSubgraphs: {}", subgraphs.len());

    for sg_idx in 0..subgraphs.len() {
        let sg = subgraphs.get(sg_idx);
        println!("\n=== Subgraph {} ===", sg_idx);

        let tensors = match sg.tensors() {
            Some(t) => t,
            None => {
                println!("  No tensors.");
                continue;
            }
        };

        let operators = match sg.operators() {
            Some(o) => o,
            None => {
                println!("  No operators.");
                continue;
            }
        };

        // Tensor dtype histogram
        let mut dtype_counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut ndim_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..tensors.len() {
            let t = tensors.get(i);
            *dtype_counts.entry(type_name(t.type_())).or_default() += 1;
            let ndim = t.shape().map(|s| s.len()).unwrap_or(0);
            *ndim_counts.entry(ndim).or_default() += 1;
        }

        println!("\n--- Tensors ({}) ---", tensors.len());
        print!("  Dtypes: ");
        for (name, count) in &dtype_counts {
            print!("{name}={count} ");
        }
        println!();
        print!("  Dims:   ");
        for (ndim, count) in &ndim_counts {
            print!("{ndim}D={count} ");
        }
        println!();

        // Input/output tensors
        if let Some(inputs) = sg.inputs() {
            println!("  Inputs:");
            for idx in inputs.iter() {
                if idx >= 0 && (idx as usize) < tensors.len() {
                    let t = tensors.get(idx as usize);
                    println!("    [{idx}] {} {}", type_name(t.type_()), shape_str(&t),);
                }
            }
        }
        if let Some(outputs) = sg.outputs() {
            println!("  Outputs:");
            for idx in outputs.iter() {
                if idx >= 0 && (idx as usize) < tensors.len() {
                    let t = tensors.get(idx as usize);
                    println!("    [{idx}] {} {}", type_name(t.type_()), shape_str(&t),);
                }
            }
        }

        // Operator walk
        println!("\n--- Operators ({}) ---", operators.len());
        let mut op_histogram: BTreeMap<&str, usize> = BTreeMap::new();

        for i in 0..operators.len() {
            let op = operators.get(i);
            let opcode_idx = op.opcode_index() as usize;
            let builtin = if opcode_idx < opcode_names.len() {
                opcode_names[opcode_idx]
            } else {
                BuiltinOperator::ADD
            };
            let name = op_name(builtin);
            *op_histogram.entry(name).or_default() += 1;

            // Input shapes
            let in_shapes: Vec<String> = op
                .inputs()
                .map(|v| {
                    v.iter()
                        .map(|idx| {
                            if idx >= 0 && (idx as usize) < tensors.len() {
                                let t = tensors.get(idx as usize);
                                format!("{}:{}", type_name(t.type_()), shape_str(&t))
                            } else if idx == -1 {
                                "none".to_string()
                            } else {
                                format!("t{idx}")
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();

            let out_shapes: Vec<String> = op
                .outputs()
                .map(|v| {
                    v.iter()
                        .map(|idx| {
                            if idx >= 0 && (idx as usize) < tensors.len() {
                                let t = tensors.get(idx as usize);
                                format!("{}:{}", type_name(t.type_()), shape_str(&t))
                            } else {
                                format!("t{idx}")
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();

            let params = format_op_params(&op, builtin);
            let params_str = if params.is_empty() {
                String::new()
            } else {
                format!("  {params}")
            };

            println!(
                "  [{i:3}] {name:25} ({}) -> ({}){params_str}",
                in_shapes.join(", "),
                out_shapes.join(", "),
            );
        }

        // Summary
        let supported_set: std::collections::HashSet<&str> =
            SUPPORTED_OPS.iter().map(|o| op_name(*o)).collect();

        println!("\n--- Op Histogram ---");
        let mut sorted: Vec<_> = op_histogram.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        let mut ok_count = 0usize;
        let mut missing_count = 0usize;
        for (name, count) in &sorted {
            let status = if supported_set.contains(*name) {
                ok_count += *count;
                "OK"
            } else {
                missing_count += *count;
                "UNSUPPORTED"
            };
            println!("  {name:30} {count:5}  {status}");
        }
        let total = ok_count + missing_count;
        println!();
        println!(
            "Supported: {ok_count}/{total} ops ({:.0}%)",
            if total > 0 {
                100.0 * ok_count as f64 / total as f64
            } else {
                0.0
            }
        );
        println!(
            "Unsupported:   {missing_count}/{total} ops ({:.0}%)",
            if total > 0 {
                100.0 * missing_count as f64 / total as f64
            } else {
                0.0
            }
        );

        let missing_ops: Vec<&str> = sorted
            .iter()
            .filter(|(name, _)| !supported_set.contains(**name))
            .map(|(name, _)| **name)
            .collect();
        if !missing_ops.is_empty() {
            println!("Unsupported op types ({}):", missing_ops.len());
            for name in missing_ops {
                println!("  - {name}");
            }
        }
    }

    // Model size info
    let size_mb = data.len() as f64 / (1024.0 * 1024.0);
    println!("\n--- Model Size ---");
    println!("  {:.1} MB ({} bytes)", size_mb, data.len());
}
