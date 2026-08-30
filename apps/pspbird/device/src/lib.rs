//! The birdnet crate's library half: `psp_bird`, the model pipeline the
//! `pspbird` app runs (design/2026-08-24_pspbird-app.md). Only built with
//! the `app` feature; the benchmark binary (`src/main.rs`) still carries its
//! own full-model compile. `imfile` (the species-image pack format) is
//! independent of both and builds with no model at all.

#![cfg_attr(not(feature = "local"), no_std)]

#[cfg(feature = "app")]
pub mod psp_bird;

#[cfg(feature = "imfile")]
pub mod imfile;
