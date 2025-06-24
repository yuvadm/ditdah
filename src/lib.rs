// src/lib.rs
// Library interface for ditdah

pub mod decoder;
pub mod generator;

pub use decoder::MorseDecoder;
pub use generator::MorseGenerator;