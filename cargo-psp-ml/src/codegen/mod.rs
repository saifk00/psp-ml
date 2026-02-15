mod lower;
mod plan;
mod render;
mod tensor_expr;

use crate::ir::PspModel;
use proc_macro2::TokenStream;

pub type GenResult<T> = Result<T, String>;

pub struct Generated {
    pub tokens: TokenStream,
    pub data_bytes: Vec<u8>,
    pub data_path: String,
}

pub fn generate_code(model: &PspModel) -> GenResult<Generated> {
    let plan = lower::lower(model)?;
    let tokens = render::render(&plan, &model.graph);

    Ok(Generated {
        tokens,
        data_bytes: model.model_data.clone(),
        data_path: "weights.bin".to_string(),
    })
}
