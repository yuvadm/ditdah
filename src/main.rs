use anyhow::{bail, Result};
use clap::Parser;
use hound::{SampleFormat, WavReader};
use std::path::PathBuf;

mod decoder;
use decoder::MorseDecoder;

const TARGET_SAMPLE_RATE: u32 = 12000; // Same as ggmorse's kBaseSampleRate
const CHUNK_SIZE: usize = 4096;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the input WAV file
    #[arg(value_name = "WAV_FILE")]
    wav_file: PathBuf,
}

fn main() -> Result<()> {
    // Set up logging. Use `RUST_LOG=info` or `RUST_LOG=debug` to see output.
    env_logger::init();
    let cli = Cli::parse();

    log::info!("Opening WAV file: {:?}", cli.wav_file);
    let mut reader = WavReader::open(cli.wav_file)?;
    let spec = reader.spec();
    log::info!("WAV spec: {:?}", spec);

    if spec.sample_format != SampleFormat::Int && spec.sample_format != SampleFormat::Float {
        bail!(
            "Unsupported sample format: {:?}. Only 16-bit Int and 32-bit Float are supported.",
            spec.sample_format
        );
    }

    // --- Create the Morse Decoder ---
    let mut decoder = MorseDecoder::new(spec.sample_rate, TARGET_SAMPLE_RATE)?;

    // --- Read and process the audio in chunks ---
    let samples_f32: Vec<f32> = if spec.sample_format == SampleFormat::Int {
        reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    } else {
        reader.samples::<f32>().map(|s| s.unwrap()).collect()
    };

    // Convert to mono by averaging channels if necessary
    let mono_samples: Vec<f32> = if spec.channels > 1 {
        samples_f32
            .chunks_exact(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    } else {
        samples_f32
    };

    // Process the entire audio buffer
    for chunk in mono_samples.chunks(CHUNK_SIZE) {
        decoder.process(chunk)?;
    }
    
    // Finalize decoding after all audio is processed
    let decoded_text = decoder.finalize()?;

    println!("\n--- Decoded Text ---");
    println!("{}", decoded_text);

    Ok(())
}