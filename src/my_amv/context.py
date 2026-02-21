"""FrameContext and AudioFrame for per-frame processing state.

This module provides the core data structures for managing per-frame state
during video processing.
"""

from dataclasses import dataclass, field
from typing import Any

from my_amv.types import Layer, LayerKey, RGBArray


@dataclass(frozen=True)
class AudioFrame:
    """Per-frame audio analysis data.

    The AudioSource pre-analyzes the full audio track before frame processing
    begins. Each VideoFrame is paired with its corresponding AudioFrame
    based on time synchronization.

    Frozen to prevent accidental mutation during processing.
    """

    frame_index: int
    """Index of this frame in the audio analysis (0-based)."""

    time_seconds: float
    """Time position in seconds from the start of the audio."""

    beat_strength: float
    """Beat strength at this frame [0.0, 1.0].
    Peaks at beat onsets detected by librosa.
    """

    onset_strength: float
    """Onset strength [0.0, 1.0].
    Detects note/attack onsets for more rhythmic effects.
    """

    rms_energy: float
    """RMS (root mean square) energy of the audio signal [0.0, ∞).
    Higher values indicate louder audio.
    """

    spectral_centroid: float
    """Spectral centroid in Hz.
    Indicates the "brightness" of the audio - higher values mean more
    high-frequency content.
    """

    frequency_bands: dict[str, float]
    """Energy in frequency bands [0.0, 1.0] each.

    Keys:
      - "bass": Low frequency energy (kick, bass guitar)
      - "mid": Mid frequency energy (vocals, snare)
      - "treble": High frequency energy (hi-hat, cymbals)
    """


@dataclass
class FrameContext:
    """Per-frame processing context.

    Holds all state for a single video frame:
    - Frame metadata (index, fps)
    - Audio analysis data (if audio provided)
    - Layer store (named RGB arrays)
    - Rule state (per-rule inter-frame state)

    The layer store enables the compositing system where rules can read
    from and write to named layers independently.

    The rule state store enables temporal coherence - rules can carry
    state from frame to frame (e.g., running averages, seeded randomization).
    """

    frame_index: int
    """Current frame index (0-based)."""

    frame_count: int
    """Total number of frames in the video."""

    fps: float
    """Video frame rate in frames per second."""

    audio_frame: AudioFrame | None = None
    """Audio analysis data for this frame, if audio was provided."""

    _layers: dict[LayerKey, RGBArray] = field(default_factory=dict, repr=False)
    """Named layer storage.

    Rules use get_layer() and set_layer() to access this.
    See AGENTS.md for the layer system invariant.
    """

    _rule_state: dict[str, Any] = field(default_factory=dict, repr=False)
    """Per-rule inter-frame state.

    Keys are rule names (from rule.name()).
    Rules use get_rule_state() and set_rule_state() to access this.
    """

    def get_layer(self, key: LayerKey) -> RGBArray:
        """Get a layer from the layer store.

        Args:
            key: Layer name (Layer enum or string)

        Returns:
            RGBArray for the requested layer

        Raises:
            KeyError: If the layer doesn't exist
        """
        return self._layers[key]

    def set_layer(self, key: LayerKey, arr: RGBArray) -> None:
        """Store a layer in the layer store.

        Args:
            key: Layer name (Layer enum or string)
            arr: RGBArray to store
        """
        self._layers[key] = arr

    def has_layer(self, key: LayerKey) -> bool:
        """Check if a layer exists in the layer store.

        Args:
            key: Layer name (Layer enum or string)

        Returns:
            True if the layer exists, False otherwise
        """
        return key in self._layers

    def list_layers(self) -> list[LayerKey]:
        """List all layer names currently in the store.

        Returns:
            List of LayerKey names
        """
        return list(self._layers.keys())

    def get_rule_state(self, rule_name: str) -> Any | None:
        """Get inter-frame state for a specific rule.

        Args:
            rule_name: Name of the rule (from rule.name())

        Returns:
            The rule's state, or None if not set
        """
        return self._rule_state.get(rule_name)

    def set_rule_state(self, rule_name: str, state: Any) -> None:
        """Store inter-frame state for a specific rule.

        Args:
            rule_name: Name of the rule (from rule.name())
            state: Any state object the rule wants to persist
        """
        self._rule_state[rule_name] = state

    def clear_rule_state(self, rule_name: str) -> None:
        """Clear inter-frame state for a specific rule.

        Useful for resetting at scene boundaries.

        Args:
            rule_name: Name of the rule to clear
        """
        self._rule_state.pop(rule_name, None)
