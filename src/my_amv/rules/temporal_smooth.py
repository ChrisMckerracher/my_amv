"""Temporal smoothing rules for video coherence.

This module provides exponential moving average (EMA) smoothing
for reducing flicker and creating smooth transitions between frames.
"""

import numpy as np

from my_amv.context import FrameContext
from my_amv.rule import EffectRule
from my_amv.types import RGBArray
from my_amv.rules import register_rule


@register_rule
class TemporalSmoothRule(EffectRule[RGBArray]):
    """Exponential moving average smoothing across frames.

    Reduces flicker and creates smooth temporal transitions by blending
    each frame with a running EMA of previous frames.

    Formula:
        output = blend_factor * current + (1 - blend_factor) * ema_state

    On the first frame (or after reset), ema_state = current, so the
    output is identical to input (passthrough).

    Parameters:
        blend_factor: Weight of current frame [0.0, 1.0] (default 0.7)
            - 1.0: pure passthrough (no smoothing)
            - 0.5: equal blend of current and EMA
            - 0.0: pure EMA (max smoothing, very laggy)

    State:
        ema_frame: Running EMA of frames (RGBArray)
    """

    def __init__(self, blend_factor: float = 0.7) -> None:
        """Initialize the TemporalSmoothRule.

        Args:
            blend_factor: Weight of current frame [0.0, 1.0]
        """
        self.blend_factor = blend_factor

    def name(self) -> str:
        """Return the rule name."""
        return "TemporalSmooth"

    def configure(self, params: dict) -> None:
        """Configure the rule from parameters.

        Args:
            params: Dictionary with optional key:
                - blend_factor: float (0.0-1.0)

        Raises:
            ValueError: If blend_factor is outside [0, 1]
        """
        if "blend_factor" in params:
            factor = float(params["blend_factor"])
            if not (0.0 <= factor <= 1.0):
                raise ValueError(f"blend_factor must be in [0.0, 1.0], got {factor}")
            self.blend_factor = factor

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        """Apply temporal smoothing to the frame.

        Args:
            frame: Input RGB array (H, W, 3)
            context: Frame context

        Returns:
            (smoothed_frame, updated_context) tuple
        """
        # Get previous EMA state
        ema_state = context.get_rule_state(self.name())

        if ema_state is None:
            # First frame: EMA is the current frame
            ema_frame = frame.copy()
            output = frame.copy()
        else:
            # Check shape compatibility
            if ema_state.shape != frame.shape:
                # Shape mismatch: reset EMA
                ema_frame = frame.copy()
                output = frame.copy()
            else:
                # Apply EMA blending
                # output = blend_factor * current + (1 - blend_factor) * ema_state
                output = cv2_blend(frame, ema_state, self.blend_factor)
                ema_frame = output.copy()

        # Store updated EMA state
        context.set_rule_state(self.name(), ema_frame)

        return output, context

    def reset_state(self, context: FrameContext) -> FrameContext:
        """Clear the EMA state.

        After reset, the next frame will be treated as the first frame
        (passthrough behavior).

        Args:
            context: Frame context

        Returns:
            Updated context with state cleared
        """
        context.clear_rule_state(self.name())
        return context


def cv2_blend(
    current: RGBArray, ema: RGBArray, blend_factor: float
) -> RGBArray:
    """Blend current frame with EMA using weighted average.

    Args:
        current: Current frame (H, W, 3)
        ema: Previous EMA frame (H, W, 3)
        blend_factor: Weight for current frame [0.0, 1.0]

    Returns:
        Blended frame (H, W, 3)
    """
    # Convert to float for precision
    current_f = current.astype(np.float32)
    ema_f = ema.astype(np.float32)

    # Weighted blend
    blended = blend_factor * current_f + (1.0 - blend_factor) * ema_f

    # Clip and convert back to uint8
    return np.clip(blended, 0, 255).astype(np.uint8)
