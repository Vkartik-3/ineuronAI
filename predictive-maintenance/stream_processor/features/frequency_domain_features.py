"""
Frequency-Domain Feature Engineering
Extracts FFT-based features for vibration and pressure analysis
"""

import numpy as np
from typing import Dict, List
from scipy import signal
from scipy.fft import rfft, rfftfreq
import logging

logger = logging.getLogger(__name__)


class FrequencyDomainFeatures:
    """Extracts frequency-domain features using FFT"""

    def __init__(self, config: Dict):
        """
        Initialize FFT feature extractor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        fe_config = config.get("feature_engineering", {})
        fft_config = fe_config.get("frequency_domain", {})

        self.sampling_rate = fft_config.get("sampling_rate", 1.0)
        self.window_size = fft_config.get("window_size", 64)
        self.overlap = fft_config.get("overlap", 0.5)

        # Sensors to apply FFT
        self.target_sensors = fft_config.get(
            "target_sensors", ["vibration_x", "vibration_y", "vibration_z", "pressure"]
        )

        # Frequency bands
        bands_config = fft_config.get("frequency_bands", {})
        self.frequency_bands = {
            "low": bands_config.get("low", [0.0, 0.1]),
            "medium": bands_config.get("medium", [0.1, 0.3]),
            "high": bands_config.get("high", [0.3, 0.5]),
        }

        # Buffer for time series data (per equipment/sensor)
        self.signal_buffers: Dict[str, Dict[str, List[float]]] = {}

        logger.info(
            f"FFT feature extractor initialized. "
            f"Sampling rate: {self.sampling_rate}Hz, Window: {self.window_size}"
        )

    def extract_features(self, equipment_id: str, sensor_data: Dict) -> Dict:
        """
        Extract frequency-domain features

        Args:
            equipment_id: Unique equipment identifier
            sensor_data: Dictionary of sensor readings

        Returns:
            Dictionary of FFT-based features
        """
        features = {}

        # Initialize buffers if needed
        if equipment_id not in self.signal_buffers:
            self.signal_buffers[equipment_id] = {}

        equipment_buffers = self.signal_buffers[equipment_id]

        # Process each target sensor
        for sensor_name in self.target_sensors:
            if sensor_name not in sensor_data:
                continue

            value = sensor_data[sensor_name]
            if not isinstance(value, (int, float)):
                continue

            # Initialize buffer for this sensor
            if sensor_name not in equipment_buffers:
                equipment_buffers[sensor_name] = []

            # Add value to buffer
            equipment_buffers[sensor_name].append(value)

            # Keep only last window_size * 2 values (for overlap)
            max_buffer_size = int(self.window_size * 2)
            if len(equipment_buffers[sensor_name]) > max_buffer_size:
                equipment_buffers[sensor_name] = equipment_buffers[sensor_name][
                    -max_buffer_size:
                ]

            # Extract FFT features when buffer has enough data
            if len(equipment_buffers[sensor_name]) >= self.window_size:
                sensor_features = self._extract_fft_features(
                    equipment_buffers[sensor_name][-self.window_size :], sensor_name
                )
                features.update(sensor_features)

        return features

    def _extract_fft_features(self, signal_data: List[float], sensor_name: str) -> Dict:
        """
        Extract FFT features from signal

        Args:
            signal_data: Time series signal
            sensor_name: Name of the sensor

        Returns:
            Dictionary of FFT features
        """
        features = {}

        try:
            # Apply window function (Hann window)
            windowed_signal = signal_data * np.hanning(len(signal_data))

            # Compute FFT
            fft_values = rfft(windowed_signal)
            fft_freqs = rfftfreq(len(signal_data), 1 / self.sampling_rate)

            # Compute magnitude spectrum
            magnitude = np.abs(fft_values)

            # Skip DC component
            if len(magnitude) > 1:
                magnitude = magnitude[1:]
                fft_freqs = fft_freqs[1:]

            if len(magnitude) == 0:
                return features

            # Dominant frequency
            dominant_idx = np.argmax(magnitude)
            features[f"{sensor_name}_fft_dominant_freq"] = float(
                fft_freqs[dominant_idx]
            )
            features[f"{sensor_name}_fft_dominant_magnitude"] = float(
                magnitude[dominant_idx]
            )

            # Total spectral energy
            spectral_energy = np.sum(magnitude**2)
            features[f"{sensor_name}_fft_spectral_energy"] = float(spectral_energy)

            # Spectral entropy
            if spectral_energy > 0:
                normalized_spectrum = (magnitude**2) / spectral_energy
                # Filter out zeros to avoid log(0)
                normalized_spectrum = normalized_spectrum[normalized_spectrum > 0]
                spectral_entropy = -np.sum(
                    normalized_spectrum * np.log2(normalized_spectrum)
                )
                features[f"{sensor_name}_fft_spectral_entropy"] = float(
                    spectral_entropy
                )

            # Energy in frequency bands
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                band_mask = (fft_freqs >= low_freq) & (fft_freqs < high_freq)
                band_energy = np.sum(magnitude[band_mask] ** 2)
                features[f"{sensor_name}_fft_{band_name}_band_energy"] = float(
                    band_energy
                )

                # Relative band energy
                if spectral_energy > 0:
                    rel_energy = band_energy / spectral_energy
                    features[f"{sensor_name}_fft_{band_name}_band_ratio"] = float(
                        rel_energy
                    )

            # Spectral centroid (center of mass of spectrum)
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(fft_freqs * magnitude) / np.sum(magnitude)
                features[f"{sensor_name}_fft_spectral_centroid"] = float(
                    spectral_centroid
                )

            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumsum_energy = np.cumsum(magnitude**2)
            if spectral_energy > 0:
                rolloff_threshold = 0.85 * spectral_energy
                rolloff_idx = np.where(cumsum_energy >= rolloff_threshold)[0]
                if len(rolloff_idx) > 0:
                    spectral_rolloff = fft_freqs[rolloff_idx[0]]
                    features[f"{sensor_name}_fft_spectral_rolloff"] = float(
                        spectral_rolloff
                    )

        except Exception as e:
            logger.warning(f"Error extracting FFT features for {sensor_name}: {e}")

        return features

    def compute_spectrogram(
        self, equipment_id: str, sensor_name: str, nperseg: int = None
    ) -> Dict:
        """
        Compute spectrogram for time-frequency analysis

        Args:
            equipment_id: Equipment identifier
            sensor_name: Sensor name
            nperseg: Length of each segment (default: window_size)

        Returns:
            Dictionary with frequencies, times, and spectrogram
        """
        if equipment_id not in self.signal_buffers:
            return {}

        equipment_buffers = self.signal_buffers[equipment_id]
        if sensor_name not in equipment_buffers:
            return {}

        signal_data = equipment_buffers[sensor_name]
        if len(signal_data) < self.window_size:
            return {}

        try:
            if nperseg is None:
                nperseg = self.window_size

            noverlap = int(nperseg * self.overlap)

            frequencies, times, Sxx = signal.spectrogram(
                signal_data,
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
            )

            return {
                "frequencies": frequencies.tolist(),
                "times": times.tolist(),
                "spectrogram": Sxx.tolist(),
            }

        except Exception as e:
            logger.error(f"Error computing spectrogram: {e}")
            return {}

    def clear_buffer(self, equipment_id: str):
        """Clear buffers for specific equipment"""
        if equipment_id in self.signal_buffers:
            del self.signal_buffers[equipment_id]
            logger.debug(f"Cleared FFT buffers for equipment {equipment_id}")

    def get_buffer_status(self, equipment_id: str) -> Dict:
        """Get status of FFT buffers"""
        if equipment_id not in self.signal_buffers:
            return {"status": "not_initialized"}

        equipment_buffers = self.signal_buffers[equipment_id]
        buffer_status = {}

        for sensor_name, buffer in equipment_buffers.items():
            buffer_status[sensor_name] = {
                "size": len(buffer),
                "min_required": self.window_size,
                "ready": len(buffer) >= self.window_size,
            }

        return buffer_status


class AdvancedFrequencyFeatures:
    """Advanced frequency analysis features"""

    @staticmethod
    def compute_harmonics(
        fft_freqs: np.ndarray,
        fft_magnitudes: np.ndarray,
        fundamental_freq: float,
        num_harmonics: int = 5,
    ) -> Dict:
        """
        Detect and analyze harmonics

        Args:
            fft_freqs: Frequency bins
            fft_magnitudes: FFT magnitude values
            fundamental_freq: Fundamental frequency
            num_harmonics: Number of harmonics to analyze

        Returns:
            Dictionary of harmonic features
        """
        features = {}

        tolerance = 0.05  # 5% tolerance for harmonic detection

        for h in range(1, num_harmonics + 1):
            harmonic_freq = fundamental_freq * h

            # Find magnitude near harmonic frequency
            freq_diff = np.abs(fft_freqs - harmonic_freq)
            min_idx = np.argmin(freq_diff)

            if freq_diff[min_idx] <= harmonic_freq * tolerance:
                magnitude = fft_magnitudes[min_idx]
                features[f"harmonic_{h}_magnitude"] = float(magnitude)
                features[f"harmonic_{h}_freq"] = float(fft_freqs[min_idx])

        return features

    @staticmethod
    def compute_cepstrum(signal_data: np.ndarray) -> Dict:
        """
        Compute cepstral features (useful for detecting periodic patterns)

        Args:
            signal_data: Time series signal

        Returns:
            Dictionary of cepstral features
        """
        features = {}

        try:
            # Compute power spectrum
            fft_values = np.fft.fft(signal_data)
            power_spectrum = np.abs(fft_values) ** 2

            # Compute cepstrum (inverse FFT of log power spectrum)
            # Add small epsilon to avoid log(0)
            log_power = np.log(power_spectrum + 1e-10)
            cepstrum = np.fft.ifft(log_power).real

            # Take first half (quefrency domain)
            cepstrum = cepstrum[: len(cepstrum) // 2]

            features["cepstrum_peak"] = float(np.max(np.abs(cepstrum)))
            features["cepstrum_mean"] = float(np.mean(np.abs(cepstrum)))

        except Exception as e:
            logger.warning(f"Error computing cepstrum: {e}")

        return features
