// A Rust implementation of the ggmorse signal processing pipeline.
// This includes a resampler, band-pass filter, STFT for pitch detection,
// a Goertzel filter for tone extraction, and the core decoding logic.

use anyhow::{Result, bail};
use collections::VecDeque;
use rubato::{InterpolationParameters, InterpolationType, Resampler, SincFixedIn, WindowFunction};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

// --- DSP Constants (ported from ggmorse) ---
const FREQ_MIN_HZ: f32 = 200.0;
const FREQ_MAX_HZ: f32 = 1200.0;
const HISTORY_S: f32 = 3.0; // How many seconds of audio to keep for analysis

// --- Biquad Filter (Corrected Implementation) ---
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    HighPass,
    LowPass,
}

pub struct BiquadFilter {
    a0: f32,
    a1: f32,
    a2: f32,
    b1: f32,
    b2: f32,
    x1: f32,
    x2: f32, // Delayed inputs
    y1: f32,
    y2: f32, // Delayed outputs
}

impl BiquadFilter {
    pub fn new(filter_type: FilterType, cutoff_hz: f32, sample_rate: u32) -> Self {
        let mut filter = Self {
            a0: 1.0,
            a1: 0.0,
            a2: 0.0,
            b1: 0.0,
            b2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        };
        let c = (std::f32::consts::PI * cutoff_hz / sample_rate as f32).tan();
        let sqrt2 = 2.0f32.sqrt();

        match filter_type {
            FilterType::LowPass => {
                let d = 1.0 / (1.0 + sqrt2 * c + c * c);
                filter.a0 = c * c * d;
                filter.a1 = 2.0 * filter.a0;
                filter.a2 = filter.a0;
                filter.b1 = 2.0 * (c * c - 1.0) * d;
                filter.b2 = (1.0 - sqrt2 * c + c * c) * d;
            }
            FilterType::HighPass => {
                let d = 1.0 / (1.0 + sqrt2 * c + c * c);
                filter.a0 = d;
                filter.a1 = -2.0 * d;
                filter.a2 = d;
                filter.b1 = 2.0 * (c * c - 1.0) * d;
                filter.b2 = (1.0 - sqrt2 * c + c * c) * d;
            }
        }
        filter
    }

    pub fn process(&mut self, input: &mut [f32]) {
        for sample in input.iter_mut() {
            let x0 = *sample;
            let y0 = self.a0 * x0 + self.a1 * self.x1 + self.a2 * self.x2
                - self.b1 * self.y1
                - self.b2 * self.y2;

            self.x2 = self.x1;
            self.x1 = x0;
            self.y2 = self.y1;
            self.y1 = y0;

            *sample = y0;
        }
    }
}

// --- Goertzel Filter ---
struct Goertzel {
    coeff: f32,
    history: VecDeque<f32>,
    window: Vec<f32>,
}

impl Goertzel {
    fn new(target_freq: f32, sample_rate: u32, window_size: usize) -> Self {
        let k = (0.5 + (window_size as f32 * target_freq) / sample_rate as f32) as usize;
        let omega = (2.0 * std::f32::consts::PI * k as f32) / window_size as f32;
        let coeff = 2.0 * omega.cos();

        let window = (0..window_size)
            .map(|i| {
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / window_size as f32).cos()
            })
            .collect();

        Self {
            coeff,
            history: VecDeque::with_capacity(window_size),
            window,
        }
    }

    fn run(&self, samples: &[f32]) -> f32 {
        let mut q0;
        let mut q1 = 0.0;
        let mut q2 = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            q0 = self.coeff * q1 - q2 + sample * self.window[i];
            q2 = q1;
            q1 = q0;
        }

        q1 * q1 + q2 * q2 - self.coeff * q1 * q2
    }

    fn process_stream(&mut self, samples: &[f32]) -> Vec<f32> {
        self.history.extend(samples.iter());
        let mut power = Vec::new();
        while self.history.len() >= self.window.len() {
            let chunk: Vec<f32> = self
                .history
                .iter()
                .take(self.window.len())
                .copied()
                .collect();
            power.push(self.run(&chunk));
            self.history.pop_front();
        }
        power
    }
}

// --- Main Decoder ---
pub struct MorseDecoder {
    resampler: SincFixedIn<f32>,
    filter_hp: BiquadFilter,
    filter_lp: BiquadFilter,
    audio_buffer: Vec<f32>,
    target_sample_rate: u32,
    estimated_pitch: Option<f32>,
}

impl MorseDecoder {
    pub fn new(source_sample_rate: u32, target_sample_rate: u32) -> Result<Self> {
        let resampler = if source_sample_rate != target_sample_rate {
            let params = InterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: InterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris,
            };
            SincFixedIn::new(
                target_sample_rate as f64 / source_sample_rate as f64,
                2.0,
                params,
                1024, // chunk size
                1,    // channels
            )?
        } else {
            // Create a dummy resampler if not needed
            SincFixedIn::new(
                1.0,
                1.0,
                InterpolationParameters {
                    sinc_len: 2,
                    ..Default::default()
                },
                1024,
                1,
            )?
        };

        Ok(Self {
            resampler,
            filter_hp: BiquadFilter::new(FilterType::HighPass, FREQ_MIN_HZ, target_sample_rate),
            filter_lp: BiquadFilter::new(FilterType::LowPass, FREQ_MAX_HZ, target_sample_rate),
            audio_buffer: Vec::new(),
            target_sample_rate,
            estimated_pitch: None,
        })
    }

    pub fn process(&mut self, chunk: &[f32]) -> Result<()> {
        let waves_in = vec![chunk.to_vec()];
        let mut resampled = self.resampler.process(&waves_in, None)?;

        let mut audio_chunk = resampled.remove(0);
        self.filter_hp.process(&mut audio_chunk);
        self.filter_lp.process(&mut audio_chunk);

        self.audio_buffer.extend(audio_chunk);
        Ok(())
    }

    pub fn finalize(&mut self) -> Result<String> {
        if self.audio_buffer.is_empty() {
            return Ok(String::new());
        }

        // 1. Detect Pitch using STFT
        let pitch = self.detect_pitch_stft()?;
        log::info!("Estimated pitch: {:.2} Hz", pitch);
        self.estimated_pitch = Some(pitch);

        // 2. Extract signal power using Goertzel filter
        let goertzel_window_size = (self.target_sample_rate / 50) as usize; // ~20ms window
        let mut goertzel_filter =
            Goertzel::new(pitch, self.target_sample_rate, goertzel_window_size);
        let power_signal = goertzel_filter.process_stream(&self.audio_buffer);

        // 3. Find optimal WPM and Threshold
        let (best_wpm, best_threshold) = self.find_best_params(&power_signal)?;
        log::info!(
            "Best fit: WPM = {}, Threshold = {:.4}",
            best_wpm,
            best_threshold
        );

        // 4. Decode with optimal parameters
        let text = self.decode_with_params(&power_signal, best_wpm, best_threshold);

        Ok(text)
    }

    fn detect_pitch_stft(&self) -> Result<f32> {
        let fft_size = 2048;
        let step_size = fft_size / 4;

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos())
            .collect();

        let mut spectrum_sum = vec![0.0; fft_size / 2];
        let mut count = 0;

        for chunk in self.audio_buffer.windows(fft_size).step_by(step_size) {
            let mut buffer: Vec<Complex<f32>> = chunk
                .iter()
                .zip(window.iter())
                .map(|(s, w)| Complex::new(s * w, 0.0))
                .collect();

            fft.process(&mut buffer);

            for (i, v) in buffer.iter().take(fft_size / 2).enumerate() {
                spectrum_sum[i] += v.norm_sqr();
            }
            count += 1;
        }

        if count == 0 {
            bail!("Not enough audio data to detect pitch.");
        }

        let df = self.target_sample_rate as f32 / fft_size as f32;
        let mut max_power = 0.0;
        let mut best_freq = 0.0;

        for i in 0..fft_size / 2 {
            let freq = i as f32 * df;
            if freq >= FREQ_MIN_HZ && freq <= FREQ_MAX_HZ {
                let power = spectrum_sum[i];
                if power > max_power {
                    max_power = power;
                    best_freq = freq;
                }
            }
        }

        Ok(best_freq)
    }

    fn find_best_params(&self, power_signal: &[f32]) -> Result<(f32, f32)> {
        let mut best_cost = f32::MAX;
        let mut best_wpm = 20.0;
        let mut best_threshold = 0.0;

        let mean_power = power_signal.iter().sum::<f32>() / power_signal.len() as f32;

        for wpm_int in 5..=40 {
            let wpm = wpm_int as f32;
            for l in (10..=90).step_by(5) {
                let threshold = mean_power * (l as f32 / 100.0);
                let cost = self.calculate_cost(power_signal, wpm, threshold);
                if cost < best_cost {
                    best_cost = cost;
                    best_wpm = wpm;
                    best_threshold = threshold;
                }
            }
        }
        Ok((best_wpm, best_threshold))
    }

    fn calculate_cost(&self, power_signal: &[f32], wpm: f32, threshold: f32) -> f32 {
        let dot_len_ms = 1200.0 / wpm;
        let dot_len_samples = (dot_len_ms / 1000.0)
            * (self.target_sample_rate as f32 / ((self.target_sample_rate / 50) as f32));

        let mut on_intervals = Vec::new();
        let mut off_intervals = Vec::new();
        let mut current_len = 0;
        let mut is_on = power_signal[0] > threshold;

        for &p in power_signal {
            if (p > threshold) == is_on {
                current_len += 1;
            } else {
                if is_on {
                    on_intervals.push(current_len);
                } else {
                    off_intervals.push(current_len);
                }
                is_on = !is_on;
                current_len = 1;
            }
        }

        if on_intervals.is_empty() {
            return f32::MAX;
        }

        let cost_on: f32 = on_intervals
            .iter()
            .map(|&len| {
                let cost_dot = (len as f32 / dot_len_samples - 1.0).powi(2);
                let cost_dash = (len as f32 / dot_len_samples - 3.0).powi(2);
                cost_dot.min(cost_dash)
            })
            .sum();

        cost_on / on_intervals.len() as f32
    }

    fn decode_with_params(&self, power_signal: &[f32], wpm: f32, threshold: f32) -> String {
        let dot_len_ms = 1200.0 / wpm;
        // The power signal has a lower sample rate because of the Goertzel windowing
        let power_signal_rate =
            self.target_sample_rate as f32 / (self.target_sample_rate as f32 / 50.0);
        let dot_len_samples = (dot_len_ms / 1000.0) * power_signal_rate;

        let mut result = String::new();
        let mut current_letter = String::new();

        let mut current_len = 0;
        let mut is_on = power_signal[0] > threshold;

        for &p in power_signal.iter().chain(std::iter::once(&0.0)) {
            // Add sentinel
            if (p > threshold) == is_on {
                current_len += 1;
            } else {
                let len_norm = current_len as f32 / dot_len_samples;
                if is_on {
                    // end of a tone
                    if (len_norm - 1.0).abs() < (len_norm - 3.0).abs() {
                        current_letter.push('0'); // dot
                    } else {
                        current_letter.push('1'); // dash
                    }
                } else {
                    // end of a space
                    if len_norm > 2.0 {
                        // inter-letter space
                        if let Some(c) = morse_to_char(&current_letter) {
                            result.push(c);
                        } else if !current_letter.is_empty() {
                            result.push('?'); // Unknown character
                        }
                        current_letter.clear();
                        if len_norm > 5.0 {
                            // word space
                            result.push(' ');
                        }
                    }
                    // else, it's an inter-element space, do nothing
                }
                is_on = !is_on;
                current_len = 1;
            }
        }
        result
    }
}

fn morse_to_char(s: &str) -> Option<char> {
    match s {
        "01" => Some('A'),
        "1000" => Some('B'),
        "1010" => Some('C'),
        "100" => Some('D'),
        "0" => Some('E'),
        "0010" => Some('F'),
        "110" => Some('G'),
        "0000" => Some('H'),
        "00" => Some('I'),
        "0111" => Some('J'),
        "101" => Some('K'),
        "0100" => Some('L'),
        "11" => Some('M'),
        "10" => Some('N'),
        "111" => Some('O'),
        "0110" => Some('P'),
        "1101" => Some('Q'),
        "010" => Some('R'),
        "000" => Some('S'),
        "1" => Some('T'),
        "001" => Some('U'),
        "0001" => Some('V'),
        "011" => Some('W'),
        "1001" => Some('X'),
        "1011" => Some('Y'),
        "1100" => Some('Z'),
        "01111" => Some('1'),
        "00111" => Some('2'),
        "00011" => Some('3'),
        "00001" => Some('4'),
        "00000" => Some('5'),
        "10000" => Some('6'),
        "11000" => Some('7'),
        "11100" => Some('8'),
        "11110" => Some('9'),
        "11111" => Some('0'),
        _ => None,
    }
}
