# ditdah - Morse Code Decoder

A Rust implementation of a Morse code decoder that can process WAV audio files and decode them into text.

## Features

- **Audio Processing**: Supports WAV files with various sample rates and formats
- **Signal Processing**: Uses FFT-based pitch detection, Goertzel filtering, and adaptive threshold detection
- **Automatic Parameter Detection**: Automatically determines WPM (words per minute) and optimal thresholds
- **Test Suite**: Comprehensive test suite with Morse code generator for validation

## Prerequisites

- Rust 1.70+ (2021 edition)
- Cargo package manager

## Installation

Clone the repository and build:

```bash
git clone <repository-url>
cd ditdah
cargo build --release
```

## Usage

### Command Line Interface

Decode a WAV file containing Morse code:

```bash
cargo run -- input.wav
```

With debug output:
```bash
RUST_LOG=info cargo run -- input.wav
```

With detailed signal tracing:
```bash
RUST_LOG=trace cargo run -- input.wav
```

### Library Usage

```rust
use ditdah::{MorseDecoder, MorseGenerator};

// Create a decoder
let mut decoder = MorseDecoder::new(44100, 12000)?; // source_rate, target_rate

// Process audio chunks
for chunk in audio_chunks {
    decoder.process(&chunk)?;
}

// Get decoded text
let decoded_text = decoder.finalize()?;
println!("Decoded: {}", decoded_text);
```

## Testing

### Run All Tests

```bash
cargo test
```

### Run Comprehensive Integration Tests

The project includes a comprehensive test suite that generates various Morse code signals and tests the decoder:

```bash
# Run the full test suite (generates WAV files and tests decoder)
cargo test run_comprehensive_test_suite -- --nocapture

# Run just the accuracy calculation unit test
cargo test test_accuracy_calculation
```

### Test Categories

The integration tests cover:

- **Basic signals**: Simple characters like "SOS", "HELLO WORLD"
- **Alphabet test**: All 26 letters  
- **Different frequencies**: 300Hz, 600Hz, 1000Hz
- **Different speeds**: 10 WPM (slow) to 30 WPM (fast)
- **Numbers**: "12345"
- **Mixed content**: "CQ DE W1AW" 
- **Different sample rates**: 12kHz and 44.1kHz

### Understanding Test Output

When tests run, they create:
- `test_outputs/`: Directory with generated WAV files and test reports
- `test_outputs/test_report.txt`: Detailed analysis of test results
- `signal_trace.txt`: Visual representation of signal processing (with RUST_LOG=trace)

### Test Results Interpretation

Tests measure accuracy by comparing expected vs actual decoded text:
- **Pass criteria**: Varies by test complexity (60-80% accuracy required)
- **Current status**: Library is under development, tests help identify issues
- **Common issues**: Timing problems, threshold detection, signal generation

### Generate Test WAV Files

You can also use the built-in generator to create test files:

```rust
use ditdah::MorseGenerator;

let generator = MorseGenerator::new(12000, 600.0, 20.0); // sample_rate, freq, wpm
generator.generate_wav_file("SOS", "test_sos.wav")?;
```

## Configuration

### Decoder Parameters

Key constants that can be adjusted in `src/decoder.rs`:

```rust
const FREQ_MIN_HZ: f32 = 200.0;        // Minimum frequency to detect
const FREQ_MAX_HZ: f32 = 1200.0;       // Maximum frequency to detect  
const DIT_DAH_BOUNDARY: f32 = 2.0;     // Threshold between dots and dashes
const WORD_SPACE_BOUNDARY: f32 = 5.0;  // Threshold between letters and words
```

### Logging Levels

- `RUST_LOG=error`: Only show errors
- `RUST_LOG=info`: Show pitch detection and parameter estimation  
- `RUST_LOG=debug`: Detailed processing information
- `RUST_LOG=trace`: Include signal trace generation

## Project Structure

```
ditdah/
├── src/
│   ├── main.rs          # CLI application
│   ├── lib.rs           # Library interface  
│   ├── decoder.rs       # Core Morse decoder logic
│   └── generator.rs     # Morse code generator (for testing)
├── tests/
│   └── integration_tests.rs  # Comprehensive test suite
├── Cargo.toml           # Project configuration
└── README.md           # This file
```

## Algorithm Overview

1. **Audio Preprocessing**: 
   - Resampling to target sample rate (12kHz)
   - High-pass and low-pass filtering (200Hz - 1200Hz)

2. **Pitch Detection**:
   - STFT analysis to find dominant frequency
   - Automatic frequency detection within valid range

3. **Signal Extraction**:
   - Goertzel filter tuned to detected frequency
   - Power signal generation with decimation

4. **Parameter Optimization**:
   - Automatic WPM detection (5-40 WPM range)
   - Adaptive threshold detection using signal statistics

5. **Decoding**:
   - Element timing analysis (dots vs dashes)
   - Character assembly and text output

## Known Issues

- Signal generation timing needs improvement
- Buffer size handling for different sample rates
- Accuracy varies significantly with signal quality
- Some edge cases in parameter detection

See the test suite results for current decoder performance metrics.

## Contributing

1. Run the test suite to understand current status
2. Focus on improving test pass rates
3. Signal generation and timing are key areas for improvement
4. Add tests for edge cases and new features

## License

[Add your license here]