"""AudioSource for pre-analyzing audio files.

This module provides the AudioSource class which pre-analyzes an audio file
and produces AudioFrame objects synchronized to video frame timing.

CRITICAL INVARIANT: ALL AudioFrame objects are computed upfront in analyze() —
no lazy loading. This ensures deterministic performance and allows for
beat detection algorithms that need the full audio context.
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from my_amv.context import AudioFrame


@dataclass
class FrequencyBands:
    """Energy in frequency bands [0.0, 1.0].

    Attributes:
        bass: Low frequency energy (kick, bass guitar) - typically 20-250 Hz
        mid: Mid frequency energy (vocals, snare) - typically 250-4000 Hz
        treble: High frequency energy (hi-hat, cymbals) - typically 4000-20000 Hz
    """

    bass: float
    mid: float
    treble: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary format for AudioFrame."""
        return {"bass": self.bass, "mid": self.mid, "treble": self.treble}


class AudioSource:
    """Pre-analyzes audio and produces AudioFrame objects per video frame.

    The AudioSource loads the entire audio file, analyzes it using librosa,
    and produces a list of AudioFrame objects synchronized to video frame timing.

    INVARIANT: All AudioFrame objects are computed upfront in analyze() —
    no lazy loading. This ensures:
    - Deterministic performance during frame processing
    - Beat detection algorithms have full audio context
    - No file I/O during the main processing loop

    Example:
        source = AudioSource(audio_path=Path("music.mp3"), fps=30.0, frame_count=300)
        audio_frames = source.analyze()  # Compute all frames upfront
        frame_42 = source.get_frame(42)  # Fast lookup, no I/O
    """

    # Frequency band boundaries in Hz
    BASS_CUTOFF = 250.0
    MID_CUTOFF = 4000.0

    # Beat tracking parameters
    BEAT_DECAY_RATE = 0.3  # How fast beat strength decays between onsets

    def __init__(
        self,
        audio_path: Path,
        fps: float,
        frame_count: int,
        sample_rate: int = 22050,
    ) -> None:
        """Initialize the AudioSource.

        Args:
            audio_path: Path to the audio file (mp3, wav, etc.)
            fps: Video frame rate in frames per second
            frame_count: Total number of video frames
            sample_rate: Audio sample rate for analysis (default: 22050)

        Raises:
            FileNotFoundError: If audio_path doesn't exist
            ValueError: If fps <= 0 or frame_count <= 0
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        if frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {frame_count}")

        self.audio_path = audio_path
        self.fps = fps
        self.frame_count = frame_count
        self.sample_rate = sample_rate

        # Computed by analyze()
        self._frames: list[AudioFrame] | None = None
        self._duration: float | None = None

    def analyze(self) -> list[AudioFrame]:
        """Analyze the audio and return all AudioFrame objects.

        This method:
        1. Loads the audio file using librosa
        2. Computes beat tracking, onset detection, and spectral features
        3. Aligns audio features to video frame timing
        4. Returns a list of AudioFrame objects, one per video frame

        INVARIANT: All AudioFrame objects are computed upfront — no lazy loading.
        The result is cached, so calling analyze() multiple times returns the same list.

        Returns:
            List of AudioFrame objects, one per video frame (length == frame_count)

        Raises:
            librosa.LibrosaError: If audio loading fails
        """
        if self._frames is not None:
            return self._frames

        # Load audio
        y, sr = librosa.load(self.audio_path, sr=self.sample_rate)
        self._duration = librosa.get_duration(y=y, sr=sr)

        # Compute frame times
        frame_times = np.arange(self.frame_count) / self.fps

        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Onset strength is computed per frame, resample to our frame times
        onset_times = librosa.times_like(onset_env, sr=sr, hop_length=512)
        onset_strength = np.interp(frame_times, onset_times, onset_env)

        # Normalize onset strength to [0, 1]
        if onset_strength.max() > 0:
            onset_strength = onset_strength / onset_strength.max()

        # Detect beat onsets
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

        # Compute beat strength with exponential decay
        beat_strength = self._compute_beat_strength(frame_times, beat_times)

        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        rms_times = librosa.times_like(rms, sr=sr, hop_length=512)
        rms_energy = np.interp(frame_times, rms_times, rms)

        # Compute spectral centroid
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        sc_times = librosa.times_like(spec_centroid, sr=sr, hop_length=512)
        spectral_centroid = np.interp(frame_times, sc_times, spec_centroid)

        # Compute frequency band energies
        frequency_bands = self._compute_frequency_bands(y, sr, frame_times)

        # Build AudioFrame objects
        self._frames = []
        for i, t in enumerate(frame_times):
            self._frames.append(
                AudioFrame(
                    frame_index=i,
                    time_seconds=t,
                    beat_strength=float(beat_strength[i]),
                    onset_strength=float(onset_strength[i]),
                    rms_energy=float(rms_energy[i]),
                    spectral_centroid=float(spectral_centroid[i]),
                    frequency_bands=frequency_bands[i].to_dict(),
                )
            )

        return self._frames

    def _compute_beat_strength(
        self, frame_times: np.ndarray, beat_times: np.ndarray
    ) -> np.ndarray:
        """Compute beat strength with exponential decay from beat onsets.

        Beat strength peaks at 1.0 on beat onsets and decays exponentially
        between beats.

        Args:
            frame_times: Time of each video frame in seconds
            beat_times: Time of each beat onset in seconds

        Returns:
            Array of beat strength values [0, 1] for each frame
        """
        beat_strength = np.zeros_like(frame_times)

        # For each beat, add contribution to all frames
        for beat_time in beat_times:
            # Time since this beat
            delta = frame_times - beat_time
            # Only consider frames after the beat
            mask = delta >= 0
            # Exponential decay: e^(-decay_rate * t)
            beat_strength[mask] = np.maximum(
                beat_strength[mask], np.exp(-self.BEAT_DECAY_RATE * delta[mask])
            )

        return beat_strength

    def _compute_frequency_bands(
        self, y: np.ndarray, sr: int, frame_times: np.ndarray
    ) -> list[FrequencyBands]:
        """Compute frequency band energies for each frame.

        Uses STFT to decompose the audio into frequency bins, then aggregates
        into bass, mid, and treble bands.

        Args:
            y: Audio signal
            sr: Sample rate
            frame_times: Time of each video frame

        Returns:
            List of FrequencyBands, one per frame
        """
        # Compute STFT
        stft = librosa.stft(y, hop_length=512)
        magnitude = np.abs(stft)

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=magnitude.shape[0] * 2 - 1)

        # Find frequency band indices
        bass_idx = freqs <= self.BASS_CUTOFF
        mid_idx = (freqs > self.BASS_CUTOFF) & (freqs <= self.MID_CUTOFF)
        treble_idx = freqs > self.MID_CUTOFF

        # Compute band energies
        bass_energy = np.mean(magnitude[bass_idx, :], axis=0)
        mid_energy = np.mean(magnitude[mid_idx, :], axis=0)
        treble_energy = np.mean(magnitude[treble_idx, :], axis=0)

        # Time for each STFT frame
        stft_times = librosa.times_like(magnitude[0, :], sr=sr, hop_length=512)

        # Interpolate to video frame times
        bass_interp = np.interp(frame_times, stft_times, bass_energy)
        mid_interp = np.interp(frame_times, stft_times, mid_energy)
        treble_interp = np.interp(frame_times, stft_times, treble_energy)

        # Normalize each band independently to [0, 1]
        def normalize(x: np.ndarray) -> np.ndarray:
            if x.max() > 0:
                return x / x.max()
            return x

        bass_interp = normalize(bass_interp)
        mid_interp = normalize(mid_interp)
        treble_interp = normalize(treble_interp)

        # Build FrequencyBands objects
        bands_list = []
        for i in range(len(frame_times)):
            bands_list.append(
                FrequencyBands(
                    bass=float(bass_interp[i]),
                    mid=float(mid_interp[i]),
                    treble=float(treble_interp[i]),
                )
            )

        return bands_list

    def get_frame(self, index: int) -> AudioFrame:
        """Get the AudioFrame for a specific video frame.

        analyze() must be called before get_frame().

        Args:
            index: Frame index (0-based)

        Returns:
            AudioFrame for the requested frame

        Raises:
            RuntimeError: If analyze() hasn't been called
            IndexError: If index is out of range
        """
        if self._frames is None:
            raise RuntimeError("analyze() must be called before get_frame()")

        if index < 0 or index >= len(self._frames):
            raise IndexError(
                f"Frame index {index} out of range [0, {len(self._frames)})"
            )

        return self._frames[index]

    @property
    def duration(self) -> float:
        """Get audio duration in seconds.

        Returns:
            Duration in seconds

        Raises:
            RuntimeError: If analyze() hasn't been called
        """
        if self._duration is None:
            raise RuntimeError("analyze() must be called before accessing duration")
        return self._duration

    @property
    def frames(self) -> list[AudioFrame]:
        """Get all analyzed AudioFrame objects.

        Returns:
            List of all AudioFrame objects

        Raises:
            RuntimeError: If analyze() hasn't been called
        """
        if self._frames is None:
            raise RuntimeError("analyze() must be called before accessing frames")
        return self._frames
