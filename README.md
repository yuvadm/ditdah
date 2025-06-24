# ditdah - Morse Code Decoder

A high-performance Rust implementation of a Morse code decoder that can process WAV audio files and decode them into text with **100% accuracy** on the comprehensive test suite.

## Features

- **High Accuracy**: Achieves 100% pass rate on comprehensive test suite
- **Audio Processing**: Supports WAV files with various sample rates (12kHz, 44.1kHz) and formats
- **Signal Processing**: Uses FFT-based pitch detection, Goertzel filtering, and adaptive threshold detection
- **Self-Calibrating**: Automatically determines timing, WPM, and optimal thresholds
- **Robust Decoding**: Handles uniform dot/dash sequences and complex multi-letter words
- **Comprehensive Testing**: Built-in test suite with Morse code generator for validation

## Performance

✅ **All tests passing with 100% accuracy:**
- Basic signals (SOS, HELLO WORLD)
- Full alphabet (A-Z)
- Numbers (0-9) 
- Different frequencies (300Hz - 1000Hz)
- Variable speeds (10-30 WPM)
- Different sample rates (12kHz, 44.1kHz)
- Complex content (CQ DE W1AW)

## Installation

```bash
git clone https://github.com/yuvadm/ditdah
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

### Baseline Tests (Quick Verification)

```bash
cargo test baseline_decoder_test -- --nocapture
```

### Comprehensive Test Suite

```bash
cargo test run_comprehensive_test_suite -- --nocapture
```

The test suite automatically:
- Generates test WAV files with known Morse content
- Decodes them using the library
- Measures accuracy and reports results
- Cleans up temporary files automatically

## Algorithm

The decoder uses a sophisticated multi-stage approach:

1. **Audio Preprocessing**: Resampling, bandpass filtering (200Hz-1200Hz)
2. **Pitch Detection**: STFT-based frequency analysis
3. **Signal Extraction**: Goertzel filtering tuned to detected frequency  
4. **Self-Calibration**: Intelligent timing analysis for dots vs dashes
5. **Letter Boundary Detection**: Proper gap analysis for multi-letter words
6. **Character Assembly**: Morse pattern to text conversion

### Key Innovations

- **Self-calibrating timing**: Handles both uniform sequences (EEEE, TTTT) and mixed patterns
- **Adaptive gap detection**: Distinguishes element gaps, letter gaps, and word gaps
- **Robust parameter estimation**: Works across different speeds and frequencies

## Configuration

Key constants in `src/decoder.rs`:

```rust
const FREQ_MIN_HZ: f32 = 200.0;             // Minimum frequency to detect
const FREQ_MAX_HZ: f32 = 1200.0;            // Maximum frequency to detect  
const DIT_DAH_BOUNDARY: f32 = 2.0;          // Threshold between dots and dashes
const LETTER_SPACE_BOUNDARY: f32 = 2.0;     // Threshold to end current letter
const WORD_SPACE_BOUNDARY: f32 = 5.0;       // Threshold to add word space
```

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
├── LICENSE              # MIT License
└── README.md           # This file
```

## Attribution

This implementation is based on the excellent work from [ggmorse](https://github.com/ggerganov/ggmorse) by Georgi Gerganov, which provided inspiration for the signal processing pipeline. The Rust implementation includes significant enhancements for robustness and accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Run the test suite to verify functionality: `cargo test`
2. All tests should pass with 100% accuracy
3. Add tests for new features or edge cases
4. Ensure code is properly formatted: `cargo fmt`
5. Run clippy for additional checks: `cargo clippy`