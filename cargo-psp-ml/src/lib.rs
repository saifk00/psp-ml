#![no_std]

//! TFLite Compiler for PSP
//!
//! ```text
//! tflite_compiler::kernels::naive  - Reference implementations (slow, correct)
//! tflite_compiler::kernels         - Optimized VFPU kernels (fast)
//! ```

pub mod kernels;
