use anyhow::{bail, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::VecDeque;
use std::io::Write;

// --- DSP Constants ---
const FREQ_MIN_HZ: f32 = 200.0;
const FREQ_MAX_HZ: f32 = 1200.0;

// --- Decoding Constants ---
// A dit/dah is classified by its length relative to the dot length. The ideal
// ratio is 1:3. The midpoint 2.0 is a robust boundary.
const DIT_DAH_BOUNDARY: f32 = 2.0;
// An inter-word space is distinguished from an inter-letter space. The ideal
// lengths are 3 dots (inter-letter) and 7 dots (inter-word). The midpoint 5.0 is a good boundary.
const WORD_SPACE_BOUNDARY: f32 = 5.0;

// --- BiquadFilter (Unchanged) ---
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
    x2: f32,
    y1: f32,
    y2: f32,
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

// --- Goertzel Filter (Unchanged) ---
struct Goertzel {
    coeff: f32,
    window: Vec<f32>,
}
impl Goertzel {
    fn new(target_freq: f32, sample_rate: u32, window_size: usize) -> Self {
        let k = (0.5 + (window_size as f32 * target_freq) / sample_rate as f32) as f32;
        let omega = (2.0 * std::f32::consts::PI * k) / window_size as f32;
        let coeff = 2.0 * omega.cos();
        let window = (0..window_size)
            .map(|i| {
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / window_size as f32).cos()
            })
            .collect();
        Self { coeff, window }
    }
    fn run(&self, samples: &[f32]) -> f32 {
        let mut q1 = 0.0;
        let mut q2 = 0.0;
        for (i, &sample) in samples.iter().enumerate() {
            let q0 = self.coeff * q1 - q2 + sample * self.window[i];
            q2 = q1;
            q1 = q0;
        }
        q1 * q1 + q2 * q2 - self.coeff * q1 * q2
    }
    fn process_decimated(&self, samples: &[f32], step_size: usize) -> Vec<f32> {
        if samples.len() < self.window.len() {
            return Vec::new();
        }
        samples
            .windows(self.window.len())
            .step_by(step_size)
            .map(|chunk| self.run(chunk))
            .collect()
    }
}

// --- Main Decoder ---
pub struct MorseDecoder {
    resampler: Option<SincFixedIn<f32>>,
    filter_hp: BiquadFilter,
    filter_lp: BiquadFilter,
    audio_buffer: Vec<f32>,
    target_sample_rate: u32,
    // source_sample_rate and resampler_chunk_size are only needed during construction
}

impl MorseDecoder {
    pub fn new(source_sample_rate: u32, target_sample_rate: u32) -> Result<Self> {
        let resampler = if source_sample_rate != target_sample_rate {
            let resampler_chunk_size = 1024;
            Some(SincFixedIn::new(
                target_sample_rate as f64 / source_sample_rate as f64,
                2.0,
                SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Linear,
                    oversampling_factor: 256,
                    window: WindowFunction::BlackmanHarris,
                },
                resampler_chunk_size,
                1,
            )?)
        } else {
            None
        };

        Ok(Self {
            resampler,
            filter_hp: BiquadFilter::new(FilterType::HighPass, FREQ_MIN_HZ, target_sample_rate),
            filter_lp: BiquadFilter::new(FilterType::LowPass, FREQ_MAX_HZ, target_sample_rate),
            audio_buffer: Vec::new(),
            target_sample_rate,
        })
    }

    /// Processes a chunk of audio samples, resampling and filtering them into an internal buffer.
    pub fn process(&mut self, chunk: &[f32]) -> Result<()> {
        let mut processed_chunk = if let Some(resampler) = &mut self.resampler {
            // Pass a slice of slices to avoid allocation
            let waves_in = &[chunk];
            resampler.process(waves_in, None)?.remove(0)
        } else {
            // If no resampling is needed, just copy the chunk
            chunk.to_vec()
        };

        self.filter_hp.process(&mut processed_chunk);
        self.filter_lp.process(&mut processed_chunk);
        self.audio_buffer.extend(processed_chunk);
        Ok(())
    }

    /// Finalizes the decoding process after all audio has been processed.
    pub fn finalize(&mut self) -> Result<String> {
        if self.audio_buffer.is_empty() {
            bail!("Audio buffer is empty, cannot process.");
        }

        // 1. Detect Pitch using STFT on the whole signal
        let pitch = self.detect_pitch_stft()?;
        log::info!("Estimated pitch: {:.2} Hz", pitch);

        // 2. Extract Power Signal using a Goertzel filter tuned to the detected pitch
        let goertzel_window_size = (self.target_sample_rate as f32 * 0.025) as usize; // 25ms window
        let step_size = (goertzel_window_size / 4).max(1);
        let goertzel_filter = Goertzel::new(pitch, self.target_sample_rate, goertzel_window_size);
        let raw_power = goertzel_filter.process_decimated(&self.audio_buffer, step_size);
        let power_signal_rate = self.target_sample_rate as f32 / step_size as f32;

        // 3. Smooth Power Signal with a moving average
        let smooth_window = (power_signal_rate * 0.02).round() as usize; // 20ms smoothing
        let smoothed_power = moving_average(&raw_power, smooth_window.max(1));
        if smoothed_power.is_empty() {
            bail!("No power signal after processing");
        }

        // 4. Find optimal WPM and Threshold by searching for the best fit
        let (best_wpm, best_threshold) =
            self.find_best_params(&smoothed_power, power_signal_rate)?;
        log::info!(
            "Best fit: WPM = {:.1}, Threshold = {:.4e}",
            best_wpm,
            best_threshold
        );

        // 5. DEBUG: Visualize the power signal and threshold
        if log::log_enabled!(log::Level::Trace) {
            trace_signal(&smoothed_power, best_threshold, best_wpm)?;
            log::trace!("Wrote signal trace to signal_trace.txt");
        }

        // 6. Decode the signal using the optimal parameters
        let text =
            self.decode_with_params(&smoothed_power, best_wpm, best_threshold, power_signal_rate);
        Ok(text)
    }

    fn detect_pitch_stft(&self) -> Result<f32> {
        let fft_size = 4096;
        let step_size = fft_size / 4;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos()) // Hamming window
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
            bail!("Not enough audio data for pitch detection");
        }

        let df = self.target_sample_rate as f32 / fft_size as f32;
        let (max_idx, max_power) =
            spectrum_sum
                .iter()
                .enumerate()
                .fold((0, 0.0), |(max_i, max_p), (i, &p)| {
                    let freq = i as f32 * df;
                    if freq >= FREQ_MIN_HZ && freq <= FREQ_MAX_HZ && p > max_p {
                        (i, p)
                    } else {
                        (max_i, max_p)
                    }
                });

        if max_power == 0.0 {
            bail!("Could not find a dominant frequency in the specified range.");
        }
        Ok(max_idx as f32 * df)
    }

    /// Searches for the best WPM and threshold combination by testing a range of thresholds
    /// derived from the signal's power distribution and finding the WPM that yields the lowest cost for each.
    fn find_best_params(&self, power_signal: &[f32], power_signal_rate: f32) -> Result<(f32, f32)> {
        if power_signal.is_empty() {
            bail!("Power signal is empty");
        }

        let mut sorted_power: Vec<f32> =
            power_signal.iter().cloned().filter(|&p| p > 0.0).collect();
        if sorted_power.len() < 10 {
            bail!("Not enough signal to determine parameters");
        }
        sorted_power.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let p25 = sorted_power[(sorted_power.len() as f32 * 0.25) as usize];
        let p75 = sorted_power[(sorted_power.len() as f32 * 0.75) as usize];
        let iqr = p75 - p25;

        // Test a few threshold candidates within the interquartile range (IQR) of the signal power.
        // This is more robust than relying on a single, fixed calculation.
        let threshold_candidates = [
            p25 + iqr * 0.25, // Lower-biased threshold
            p25 + iqr * 0.50, // Midpoint threshold (original method)
            p25 + iqr * 0.75, // Upper-biased threshold
        ];

        let mut best_cost = f32::MAX;
        let mut best_wpm = 20.0;
        let mut best_threshold = threshold_candidates[1]; // Default to midpoint

        for &threshold in &threshold_candidates {
            for wpm_int in 5..=40 {
                let wpm = wpm_int as f32;
                let cost = self.calculate_cost(power_signal, wpm, threshold, power_signal_rate);
                if cost < best_cost {
                    best_cost = cost;
                    best_wpm = wpm;
                    best_threshold = threshold;
                }
            }
        }
        Ok((best_wpm, best_threshold))
    }

    /// Calculates a "cost" for a given set of parameters (wpm, threshold).
    /// A lower cost indicates a better fit. The cost is the mean squared error
    /// of element lengths from their ideal ratios (1, 3, 7), normalized by a
    /// self-calibrated dot length.
    fn calculate_cost(
        &self,
        power_signal: &[f32],
        wpm: f32,
        threshold: f32,
        power_signal_rate: f32,
    ) -> f32 {
        let (on_intervals, off_intervals) = get_raw_intervals(power_signal, threshold);
        if on_intervals.len() < 3 || off_intervals.len() < 3 {
            return f32::MAX;
        }

        let dot_len_samples = (1200.0 / wpm / 1000.0) * power_signal_rate;
        if dot_len_samples < 1.0 {
            return f32::MAX;
        }

        let on_norm: Vec<f32> = on_intervals
            .iter()
            .map(|&s| s as f32 / dot_len_samples)
            .collect();
        let off_norm: Vec<f32> = off_intervals
            .iter()
            .map(|&s| s as f32 / dot_len_samples)
            .collect();

        // Estimate the "real" dot length by finding the median of all short elements.
        // This self-calibrates to the sender's actual timing.
        let mut short_elements: Vec<f32> = on_norm
            .iter()
            .chain(off_norm.iter())
            .cloned()
            .filter(|&l| l < 2.0)
            .collect();
        if short_elements.is_empty() {
            return f32::MAX;
        }
        short_elements.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let median_dot_len = short_elements[short_elements.len() / 2];
        if median_dot_len < 0.25 {
            return f32::MAX;
        } // Unrealistic

        // Final cost is the deviation from ideal ratios, normalized by our measured median dot length.
        let cost_on: f32 = on_norm
            .iter()
            .map(|&len| {
                (len / median_dot_len - 1.0)
                    .powi(2)
                    .min((len / median_dot_len - 3.0).powi(2))
            })
            .sum();
        let cost_off: f32 = off_norm
            .iter()
            .map(|&len| {
                (len / median_dot_len - 1.0)
                    .powi(2)
                    .min((len / median_dot_len - 3.0).powi(2))
                    .min((len / median_dot_len - 7.0).powi(2))
            })
            .sum();

        (cost_on / on_intervals.len() as f32) + (cost_off / off_intervals.len() as f32)
    }

    /// Decodes the power signal into text using the provided parameters.
    fn decode_with_params(
        &self,
        power_signal: &[f32],
        wpm: f32,
        threshold: f32,
        power_signal_rate: f32,
    ) -> String {
        let dot_len_samples = (1200.0 / wpm / 1000.0) * power_signal_rate;
        let mut result = String::new();
        let mut current_letter = String::new();
        if power_signal.is_empty() {
            return result;
        }

        let mut current_len = 0;
        let mut is_on = power_signal[0] > threshold;
        // Debouncing prevents short noise spikes from being registered as valid elements.
        let debounce_samples = (dot_len_samples * 0.3).round() as usize;

        // Chain a zero to the end to ensure the last element is always processed.
        for &p in power_signal.iter().chain(std::iter::once(&0.0)) {
            if (p > threshold) == is_on {
                current_len += 1;
            } else {
                if current_len > debounce_samples {
                    let len_norm = current_len as f32 / dot_len_samples;
                    if is_on {
                        // End of a tone
                        if len_norm < DIT_DAH_BOUNDARY {
                            current_letter.push('.');
                        } else {
                            current_letter.push('-');
                        }
                    } else {
                        // End of a space
                        if !current_letter.is_empty() {
                            if let Some(c) = morse_to_char(&current_letter) {
                                result.push(c);
                            } else {
                                result.push('?'); // Unknown character
                            }
                            current_letter.clear();
                        }
                        if len_norm > WORD_SPACE_BOUNDARY {
                            if !result.ends_with(' ') {
                                result.push(' ');
                            }
                        }
                    }
                }
                is_on = !is_on;
                current_len = 1;
            }
        }
        result.trim().to_string()
    }
}

// --- Helper Functions ---
fn get_raw_intervals(power_signal: &[f32], threshold: f32) -> (Vec<usize>, Vec<usize>) {
    let mut on = Vec::new();
    let mut off = Vec::new();
    if power_signal.is_empty() {
        return (on, off);
    }

    let mut current_len = 0;
    let mut is_on = power_signal[0] > threshold;
    for &p in power_signal {
        if (p > threshold) == is_on {
            current_len += 1;
        } else {
            if is_on {
                on.push(current_len);
            } else {
                off.push(current_len);
            }
            is_on = !is_on;
            current_len = 1;
        }
    }
    if is_on {
        on.push(current_len);
    } else {
        off.push(current_len);
    }
    (on, off)
}

fn moving_average(data: &[f32], window_size: usize) -> Vec<f32> {
    if window_size <= 1 {
        return data.to_vec();
    }
    let mut smoothed = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    let mut window = VecDeque::with_capacity(window_size);
    for &x in data {
        if window.len() == window_size {
            sum -= window.pop_front().unwrap();
        }
        sum += x;
        window.push_back(x);
        smoothed.push(sum / window.len() as f32);
    }
    smoothed
}

fn trace_signal(signal: &[f32], threshold: f32, wpm: f32) -> std::io::Result<()> {
    let mut file = std::fs::File::create("signal_trace.txt")?;
    writeln!(file, "# WPM: {:.1}, Threshold: {:.4e}", wpm, threshold)?;
    let max_val = signal.iter().cloned().fold(f32::MIN, f32::max);
    if max_val <= 0.0 {
        return Ok(());
    }

    for &val in signal {
        let bar_len = (val / max_val * 100.0).round() as usize;
        let thresh_pos = (threshold / max_val * 100.0).round() as usize;
        let mut line = vec![' '; 101];
        for i in 0..bar_len.min(100) {
            line[i] = '#';
        }
        if thresh_pos <= 100 {
            line[thresh_pos] = '|';
        }
        writeln!(file, "{}", line.into_iter().collect::<String>())?;
    }
    Ok(())
}

fn morse_to_char(s: &str) -> Option<char> {
    match s {
        ".-" => Some('A'),
        "-..." => Some('B'),
        "-.-." => Some('C'),
        "-.." => Some('D'),
        "." => Some('E'),
        "..-." => Some('F'),
        "--." => Some('G'),
        "...." => Some('H'),
        ".." => Some('I'),
        ".---" => Some('J'),
        "-.-" => Some('K'),
        ".-.." => Some('L'),
        "--" => Some('M'),
        "-." => Some('N'),
        "---" => Some('O'),
        ".--." => Some('P'),
        "--.-" => Some('Q'),
        ".-." => Some('R'),
        "..." => Some('S'),
        "-" => Some('T'),
        "..-" => Some('U'),
        "...-" => Some('V'),
        ".--" => Some('W'),
        "-..-" => Some('X'),
        "-.--" => Some('Y'),
        "--.." => Some('Z'),
        ".----" => Some('1'),
        "..---" => Some('2'),
        "...--" => Some('3'),
        "....-" => Some('4'),
        "....." => Some('5'),
        "-...." => Some('6'),
        "--..." => Some('7'),
        "---.." => Some('8'),
        "----." => Some('9'),
        "-----" => Some('0'),
        _ => None,
    }
}
