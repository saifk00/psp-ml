//! TFLite model compiler — generates Rust code targeting the `psp-ml` runtime library.

pub mod codegen;
pub mod ir;
pub mod memory_planner;
pub mod parse;

