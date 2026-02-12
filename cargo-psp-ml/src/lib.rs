//! TFLite model compiler — generates Rust code targeting the `psp-ml` runtime library.

pub mod codegen;
pub mod ir;
pub mod parse;

pub mod runner;
