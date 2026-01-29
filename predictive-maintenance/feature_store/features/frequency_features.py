"""
Frequency-Domain Feature Engineering
FFT peaks, spectral analysis, wavelet transforms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import signal
from scipy.fft import rfft, rfftfreq
import logging

logger = logging.getLogger(__name__)


class FrequencyFeatureExtractor:
    """Extracts frequency-domain features"""

    def __init__(self, config: Dict):
        """
        Initialize frequency feature extractor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        freq_config = config.get("feature_engineering", {}).get("frequency_domain", {})

        self.enabled = freq_config.get("enabled", True)

        # FFT configuration
        fft_config = freq_config.get("fft", {})
        self.window_size = fft_config.get("window_size", 128)
        self.sampling_rate = fft_config.get("sampling_rate", 1.0)
        self.overlap = fft_config.get("overlap", 0.5)
        self.target_sensors = fft_config.get("target_sensors", [])
        self.fft_features = fft_config.get("features", [])
        self.frequency_bands = fft_config.get("frequency_bands", {})

        logger.info(
            f"Frequency feature extractor initialized. "
            f"Window: {self.window_size}, Sampling: {self.sampling_rate}Hz"
        )

    def extract_features(
        self, df: pd.DataFrame, equipment_id: str = None
    ) -> pd.DataFrame:
        """
        Extract frequency-domain features

        Args:
            df: DataFrame with sensor readings
            equipment_id: Optional equipment identifier

        Returns:
            DataFrame with frequency features
        """
        if not self.enabled or df.empty:
            return df

        # Filter by equipment if specified
        if equipment_id and "equipment_id" in df.columns:
            df = df[df["equipment_id"] == equipment_id].copy()

        result_df = df.copy()

        # Extract FFT features for each target sensor
        for sensor in self.target_sensors:
            if sensor not in df.columns:
                logger.warning(f"Sensor {sensor} not found in dataframe")
                continue

            # Get sensor data
            sensor_data = df[sensor].values

            # Skip if not enough data
            if len(sensor_data) < self.window_size:
                logger.warning(f"Insufficient data for FFT on {sensor}")
                continue

            # Extract FFT features
            fft_features = self._extract_fft_features(sensor_data, sensor)

            # Add to dataframe (broadcast to all rows)
            for feature_name, feature_value in fft_features.items():
                result_df[feature_name] = feature_value

        logger.debug(
            f"Extracted {len(result_df.columns) - len(df.columns)} frequency features"
        )

        return result_df

    def _extract_fft_features(self, signal_data: np.ndarray, sensor_name: str) -> Dict:
        """
        Extract FFT features from signal

        Args:
            signal_data: Time-domain signal
            sensor_name: Sensor name

        Returns:
            Dictionary of FFT features
        """
        features = {}

        try:
            # Use the last window_size samples
            if len(signal_data) > self.window_size:
                signal_data = signal_data[-self.window_size :]

            # Apply Hann window
            windowed_signal = signal_data * np.hanning(len(signal_data))

            # Compute FFT
            fft_values = rfft(windowed_signal)
            fft_freqs = rfftfreq(len(signal_data), 1 / self.sampling_rate)

            # Magnitude spectrum
            magnitude = np.abs(fft_values)

            # Skip DC component
            if len(magnitude) > 1:
                magnitude = magnitude[1:]
                fft_freqs = fft_freqs[1:]

            if len(magnitude) == 0:
                return features

            # Extract requested features
            if "dominant_frequency" in self.fft_features:
                dominant_idx = np.argmax(magnitude)
                features[f"{sensor_name}_fft_dominant_freq"] = float(
                    fft_freqs[dominant_idx]
                )

            if "dominant_magnitude" in self.fft_features:
                features[f"{sensor_name}_fft_dominant_mag"] = float(np.max(magnitude))

            if "spectral_energy" in self.fft_features:
                spectral_energy = np.sum(magnitude**2)
                features[f"{sensor_name}_fft_energy"] = float(spectral_energy)

            if "spectral_entropy" in self.fft_features:
                spectral_energy = np.sum(magnitude**2)
                if spectral_energy > 0:
                    normalized_spectrum = (magnitude**2) / spectral_energy
                    normalized_spectrum = normalized_spectrum[normalized_spectrum > 0]
                    spectral_entropy = -np.sum(
                        normalized_spectrum * np.log2(normalized_spectrum)
                    )
                    features[f"{sensor_name}_fft_entropy"] = float(spectral_entropy)

            if "spectral_centroid" in self.fft_features:
                if np.sum(magnitude) > 0:
                    centroid = np.sum(fft_freqs * magnitude) / np.sum(magnitude)
                    features[f"{sensor_name}_fft_centroid"] = float(centroid)

            if "spectral_rolloff" in self.fft_features:
                spectral_energy = np.sum(magnitude**2)
                if spectral_energy > 0:
                    cumsum_energy = np.cumsum(magnitude**2)
                    rolloff_threshold = 0.85 * spectral_energy
                    rolloff_idx = np.where(cumsum_energy >= rolloff_threshold)[0]
                    if len(rolloff_idx) > 0:
                        features[f"{sensor_name}_fft_rolloff"] = float(
                            fft_freqs[rolloff_idx[0]]
                        )

            if "peak_frequencies" in self.fft_features:
                # Find top 3 peaks
                peak_indices = signal.find_peaks(magnitude)[0]
                if len(peak_indices) > 0:
                    # Sort by magnitude
                    sorted_peaks = peak_indices[
                        np.argsort(magnitude[peak_indices])[::-1]
                    ]

                    # Top 3 peaks
                    for i, peak_idx in enumerate(sorted_peaks[:3]):
                        features[f"{sensor_name}_fft_peak{i + 1}_freq"] = float(
                            fft_freqs[peak_idx]
                        )
                        features[f"{sensor_name}_fft_peak{i + 1}_mag"] = float(
                            magnitude[peak_idx]
                        )

            if "frequency_band_energy" in self.fft_features:
                # Energy in frequency bands
                spectral_energy = np.sum(magnitude**2)

                for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                    band_mask = (fft_freqs >= low_freq) & (fft_freqs < high_freq)
                    band_energy = np.sum(magnitude[band_mask] ** 2)

                    features[f"{sensor_name}_fft_{band_name}_energy"] = float(
                        band_energy
                    )

                    # Relative band energy
                    if spectral_energy > 0:
                        rel_energy = band_energy / spectral_energy
                        features[f"{sensor_name}_fft_{band_name}_ratio"] = float(
                            rel_energy
                        )

        except Exception as e:
            logger.error(f"Error extracting FFT features for {sensor_name}: {e}")

        return features

    def extract_spectrogram_features(
        self, signal_data: np.ndarray, sensor_name: str
    ) -> Dict:
        """
        Extract features from spectrogram

        Args:
            signal_data: Time-domain signal
            sensor_name: Sensor name

        Returns:
            Dictionary of spectrogram features
        """
        features = {}

        try:
            nperseg = self.window_size
            noverlap = int(nperseg * self.overlap)

            # Compute spectrogram
            frequencies, times, Sxx = signal.spectrogram(
                signal_data,
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
            )

            # Statistical features of spectrogram
            features[f"{sensor_name}_spectrogram_mean"] = float(np.mean(Sxx))
            features[f"{sensor_name}_spectrogram_std"] = float(np.std(Sxx))
            features[f"{sensor_name}_spectrogram_max"] = float(np.max(Sxx))

            # Temporal energy variation
            temporal_energy = np.sum(Sxx, axis=0)  # Energy at each time
            features[f"{sensor_name}_temporal_energy_std"] = float(
                np.std(temporal_energy)
            )

            # Spectral energy variation
            spectral_energy = np.sum(Sxx, axis=1)  # Energy at each frequency
            features[f"{sensor_name}_spectral_energy_std"] = float(
                np.std(spectral_energy)
            )

        except Exception as e:
            logger.error(
                f"Error extracting spectrogram features for {sensor_name}: {e}"
            )

        return features


class VibrationAnalyzer:
    """Specialized vibration analysis features"""

    @staticmethod
    def compute_vibration_features(vibration_data: Dict[str, np.ndarray]) -> Dict:
        """
        Compute vibration-specific features

        Args:
            vibration_data: Dictionary of vibration axes (x, y, z)

        Returns:
            Dictionary of vibration features
        """
        features = {}

        # Total vibration magnitude
        if all(axis in vibration_data for axis in ["x", "y", "z"]):
            vib_x = vibration_data["x"]
            vib_y = vibration_data["y"]
            vib_z = vibration_data["z"]

            # Vector magnitude
            total_vibration = np.sqrt(vib_x**2 + vib_y**2 + vib_z**2)

            features["vibration_total_mean"] = float(np.mean(total_vibration))
            features["vibration_total_std"] = float(np.std(total_vibration))
            features["vibration_total_max"] = float(np.max(total_vibration))
            features["vibration_total_rms"] = float(
                np.sqrt(np.mean(total_vibration**2))
            )

            # Crest factor (peak / RMS)
            rms = features["vibration_total_rms"]
            if rms > 0:
                features["vibration_crest_factor"] = float(
                    np.max(total_vibration) / rms
                )

            # Kurtosis (measure of impulsiveness)
            features["vibration_kurtosis"] = float(
                pd.Series(total_vibration).kurtosis()
            )

            # Skewness (measure of asymmetry)
            features["vibration_skewness"] = float(pd.Series(total_vibration).skew())

        return features

    @staticmethod
    def compute_envelope_spectrum(
        signal_data: np.ndarray, sampling_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute envelope spectrum for bearing fault detection

        Args:
            signal_data: Time-domain signal
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (frequencies, envelope_spectrum)
        """
        # Hilbert transform for envelope detection
        analytic_signal = signal.hilbert(signal_data)
        envelope = np.abs(analytic_signal)

        # FFT of envelope
        envelope_fft = rfft(envelope)
        envelope_freqs = rfftfreq(len(envelope), 1 / sampling_rate)
        envelope_spectrum = np.abs(envelope_fft)

        return envelope_freqs, envelope_spectrum
