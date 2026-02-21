"""Audio-reactive effect rules.

This module provides audio-driven effects that respond to beat,
onset, and spectral features from an audio track.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from my_amv.context import FrameContext
from my_amv.rule import EffectRule
from my_amv.types import Layer, RGBArray
from my_amv.rules import register_rule


@dataclass
class AudioReactiveState:
    """State for audio-reactive effects."""

    displacement: float  # Current displacement magnitude [0, 1]
    thickness_scale: float  # Current thickness scale [0.5, 2.0]


def _apply_displacement(
    frame: RGBArray,
    displacement: float,
    max_displacement: int,
) -> RGBArray:
    """Apply pixel displacement to frame.

    Args:
        frame: Input frame (H, W, 3)
        displacement: Displacement magnitude [0, 1]
        max_displacement: Maximum displacement in pixels

    Returns:
        Displaced frame
    """
    if displacement < 0.01 or max_displacement == 0:
        return frame

    height, width = frame.shape[:2]
    offset = int(displacement * max_displacement)

    if offset == 0:
        return frame

    # Simple displacement using np.roll (shifts pixels)
    # For a more sophisticated effect, we'd use cv2.remap
    result = np.roll(frame, offset, axis=1)  # Shift horizontally

    return result


def _apply_thickness(
    frame: RGBArray,
    thickness_scale: float,
) -> RGBArray:
    """Apply thickness variation to edges.

    Args:
        frame: Input edge frame (H, W, 3)
        thickness_scale: Scale factor for thickness

    Returns:
        Frame with adjusted thickness
    """
    if abs(thickness_scale - 1.0) < 0.01:
        return frame

    # For edges, thickness is simulated by dilation/erosion
    # Use morphological operations
    import cv2

    kernel_size = int(3 * thickness_scale)
    if kernel_size < 1:
        kernel_size = 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if thickness_scale > 1.0:
        # Dilate for thicker edges
        result = cv2.dilate(frame, kernel, iterations=1)
    else:
        # Erode for thinner edges
        result = cv2.erode(frame, kernel, iterations=1)

    return result


def _get_frequency_energy(
    frequency_bands: dict[str, float] | None,
    frequency_band: Literal["bass", "mid", "treble", "all"],
) -> float:
    """Get energy for a specific frequency band.

    Args:
        frequency_bands: Dict with "bass", "mid", "treble" keys
        frequency_band: Which band to select

    Returns:
        Energy value [0, 1]
    """
    if frequency_bands is None:
        return 0.0

    if frequency_band == "all":
        # Average of all bands
        return float(np.mean(list(frequency_bands.values())))

    return float(frequency_bands.get(frequency_band, 0.0))


@register_rule
class AudioReactiveEdgeRule(EffectRule[AudioReactiveState]):
    """Audio-reactive edge displacement and thickness effect.

    Responds to beat onsets and frequency content by displacing
    and modulating the thickness of edge features.

    When no audio is available (audio_frame is None), the rule
    acts as a passthrough.

    State:
        displacement: Current displacement magnitude [0, 1]
        thickness_scale: Current thickness scale [0.5, 2.0]

    Parameters:
        sensitivity: Overall reactivity scale [0, 2] (default 1.0)
        max_displacement: Maximum pixel displacement (default 8)
        decay: Exponential decay rate per frame [0, 1] (default 0.85)
            - Higher values = slower decay (more sustain)
            - Lower values = faster decay (quick return to baseline)
        frequency_band: Which frequency band to respond to
            - "bass": Low frequencies (kick, bass guitar)
            - "mid": Mid frequencies (vocals, snare)
            - "treble": High frequencies (hi-hat, cymbals)
            - "all": Average of all bands
        effect_type: Which effects to apply
            - "displace": Only displacement
            - "thickness": Only thickness modulation
            - "both": Apply both effects
    """

    def __init__(
        self,
        sensitivity: float = 1.0,
        max_displacement: int = 8,
        decay: float = 0.85,
        frequency_band: Literal["bass", "mid", "treble", "all"] = "all",
        effect_type: Literal["displace", "thickness", "both"] = "both",
    ) -> None:
        """Initialize the AudioReactiveEdgeRule.

        Args:
            sensitivity: Overall reactivity scale
            max_displacement: Maximum pixel displacement
            decay: Exponential decay rate
            frequency_band: Frequency band to respond to
            effect_type: Which effects to apply
        """
        self.sensitivity = sensitivity
        self.max_displacement = max_displacement
        self.decay = decay
        self.frequency_band = frequency_band
        self.effect_type = effect_type
        self.input_layer = Layer.EDGES

    def name(self) -> str:
        """Return the rule name."""
        return "AudioReactiveEdgeRule"

    def configure(self, params: dict) -> None:
        """Configure the rule from parameters.

        Args:
            params: Dictionary with optional keys:
                - sensitivity: float (0-2)
                - max_displacement: int (0-50)
                - decay: float (0-1)
                - frequency_band: str ("bass" | "mid" | "treble" | "all")
                - effect_type: str ("displace" | "thickness" | "both")

        Raises:
            ValueError: If any parameter is invalid
        """
        if "sensitivity" in params:
            sensitivity = float(params["sensitivity"])
            if not (0.0 <= sensitivity <= 2.0):
                raise ValueError(f"sensitivity must be in [0.0, 2.0], got {sensitivity}")
            self.sensitivity = sensitivity

        if "max_displacement" in params:
            max_disp = int(params["max_displacement"])
            if not (0 <= max_disp <= 50):
                raise ValueError(f"max_displacement must be in [0, 50], got {max_disp}")
            self.max_displacement = max_disp

        if "decay" in params:
            decay = float(params["decay"])
            if not (0.0 <= decay <= 1.0):
                raise ValueError(f"decay must be in [0.0, 1.0], got {decay}")
            self.decay = decay

        if "frequency_band" in params:
            band = params["frequency_band"]
            if band not in ("bass", "mid", "treble", "all"):
                raise ValueError(f"Invalid frequency_band: {band}")
            self.frequency_band = band

        if "effect_type" in params:
            etype = params["effect_type"]
            if etype not in ("displace", "thickness", "both"):
                raise ValueError(f"Invalid effect_type: {etype}")
            self.effect_type = etype

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        """Apply audio-reactive effects to the frame.

        Args:
            frame: Input RGB array (H, W, 3) - typically edge map
            context: Frame context with audio_frame

        Returns:
            (effected_frame, updated_context) tuple
        """
        # Check if audio is available
        audio_frame = context.audio_frame

        if audio_frame is None:
            # No audio: passthrough
            return frame.copy(), context

        # Get previous state
        state = context.get_rule_state(self.name())

        if state is None:
            state = AudioReactiveState(displacement=0.0, thickness_scale=1.0)

        # Get frequency energy
        energy = _get_frequency_energy(audio_frame.frequency_bands, self.frequency_band)

        # Calculate beat influence
        # Beat strength + frequency energy drives the effect
        beat_strength = audio_frame.beat_strength
        combined = (beat_strength + energy) / 2.0  # Average [0, 1]

        # Apply sensitivity
        target_displacement = combined * self.sensitivity

        # On strong beat, spike the displacement
        if combined > 0.5:  # Beat threshold
            state.displacement = min(state.displacement + target_displacement, 1.0)
            state.thickness_scale = min(state.thickness_scale + 0.2 * combined, 2.0)
        else:
            # Decay
            state.displacement *= self.decay
            state.thickness_scale = 1.0 + (state.thickness_scale - 1.0) * self.decay

        # Clamp values
        state.displacement = np.clip(state.displacement, 0.0, 1.0)
        state.thickness_scale = np.clip(state.thickness_scale, 0.5, 2.0)

        # Store updated state
        context.set_rule_state(self.name(), state)

        # Apply effects
        result = frame.copy()

        if self.effect_type in ("displace", "both"):
            result = _apply_displacement(result, state.displacement, self.max_displacement)

        if self.effect_type in ("thickness", "both"):
            result = _apply_thickness(result, state.thickness_scale)

        return result, context

    def reset_state(self, context: FrameContext) -> FrameContext:
        """Clear the rule state.

        Args:
            context: Frame context

        Returns:
            Updated context with state cleared
        """
        context.clear_rule_state(self.name())
        return context
