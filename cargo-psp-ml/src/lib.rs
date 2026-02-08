//! # cargo-psp-ml
//!
//! Libraries required by generated code. These have to be `no_std` since cargo-psp
//! doesn't support the std library (yet).
//!
//! ## Features
//!
//! - `compiler` - Enables compiler-time code generation tools (requires std)

#![cfg_attr(not(feature = "compiler"), no_std)]

/// Runtime kernels for ML operations on PSP.
pub mod kernels;

/// Intermediate representation for ML graphs.
///
/// Only available with the `compiler` feature enabled.
#[cfg(feature = "compiler")]
pub mod ir;

/// Compiler libraries for TFLite model parsing and code generation.
///
/// Only available with the `compiler` feature enabled.
#[cfg(feature = "compiler")]
pub mod codegen;

/// Auto-generated FlatBuffers schema for TFLite model parsing.
///
/// Only available with the `compiler` feature enabled.
#[cfg(feature = "compiler")]
pub mod parse;
