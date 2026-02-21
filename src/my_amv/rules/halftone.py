"""Halftone and stochastic effect rules.

This module provides halftone effects with temporal coherence for
smooth evolution across video frames.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from my_amv.context import FrameContext
from my_amv.rule import EffectRule
from my_amv.types import Layer, RGBArray
from my_amv.rules import register_rule


@dataclass
class HalfToneState:
    """State for halftone dot positions."""

    dots: np.ndarray  # shape (N, 3) -> [x, y, radius]


def _generate_dot_grid(
    width: int,
    height: int,
    dot_size: int,
    density: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate a grid of random dot positions.

    Args:
        width: Image width
        height: Image height
        dot_size: Size of each dot cell
        density: Dot density [0, 1]
        rng: Random state for reproducibility

    Returns:
        Array of (x, y, radius) positions
    """
    grid_h = height // dot_size
    grid_w = width // dot_size

    dots = []

    for row in range(grid_h):
        for col in range(grid_w):
            # Random chance to place a dot based on density
            if rng.rand() < density:
                # Center position in grid cell with random offset
                x = col * dot_size + dot_size // 2 + rng.randint(-dot_size // 4, dot_size // 4)
                y = row * dot_size + dot_size // 2 + rng.randint(-dot_size // 4, dot_size // 4)

                # Radius varies slightly
                radius = dot_size // 2 * (0.5 + 0.5 * rng.rand())

                dots.append([x, y, radius])

    if not dots:
        return np.zeros((0, 3), dtype=np.float32)

    return np.array(dots, dtype=np.float32)


def _perturb_dots(
    dots: np.ndarray,
    max_step: float,
    rng: np.random.RandomState,
    width: int,
    height: int,
) -> np.ndarray:
    """Random walk perturbation of dot positions.

    Args:
        dots: Current dot positions (N, 3)
        max_step: Maximum step size
        rng: Random state
        width: Image width (for bounds)
        height: Image height (for bounds)

    Returns:
        Perturbed dot positions
    """
    if len(dots) == 0:
        return dots

    # Random walk step
    steps = rng.randn(len(dots), 3) * max_step
    steps[:, 2] = 0  # Don't change radius

    new_dots = dots + steps

    # Clamp to image bounds
    new_dots[:, 0] = np.clip(new_dots[:, 0], 0, width - 1)
    new_dots[:, 1] = np.clip(new_dots[:, 1], 0, height - 1)

    return new_dots


def _render_halftone(
    frame: RGBArray,
    dots: np.ndarray,
    dot_color: tuple[int, int, int] = (0, 0, 0),
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> RGBArray:
    """Render halftone dots onto a frame.

    Args:
        frame: Input frame (for size reference)
        dots: Dot positions (N, 3) -> [x, y, radius]
        dot_color: RGB color for dots
        background_color: RGB color for background

    Returns:
        Rendered halftone image
    """
    height, width = frame.shape[:2]
    result = np.full((height, width, 3), background_color, dtype=np.uint8)

    if len(dots) == 0:
        return result

    y, x = np.ogrid[:height, :width]

    for dot_x, dot_y, radius in dots:
        dot_x = int(dot_x)
        dot_y = int(dot_y)
        r = int(radius)

        # Create circular mask
        mask = (x - dot_x) ** 2 + (y - dot_y) ** 2 <= r**2

        # Draw dot
        result[mask] = dot_color

    return result


@register_rule
class HalfToneRandomizationRule(EffectRule[HalfToneState]):
    """Halftone effect with smooth temporal evolution.

    Creates a stochastic halftone pattern that evolves smoothly between
    frames using random walk perturbation.

    State:
        dots: Array of (x, y, radius) for each dot

    Parameters:
        dot_size: Size of dot grid cells (default 4)
        dot_density: Probability of dot in each cell [0-1] (default 0.5)
        temporal_smoothing: How much dots persist [0-1] (default 0.8)
            - 1.0: dots don't move
            - 0.0: fully random each frame
        seed: Random seed for initialization (default 42)
    """

    def __init__(
        self,
        dot_size: int = 4,
        dot_density: float = 0.5,
        temporal_smoothing: float = 0.8,
        seed: int = 42,
    ) -> None:
        """Initialize the HalfToneRandomizationRule.

        Args:
            dot_size: Size of each dot cell in pixels
            dot_density: Dot density [0.0, 1.0]
            temporal_smoothing: Temporal coherence [0.0, 1.0]
            seed: Random seed for reproducibility
        """
        self.dot_size = dot_size
        self.dot_density = dot_density
        self.temporal_smoothing = temporal_smoothing
        self.seed = seed

    def name(self) -> str:
        """Return the rule name."""
        return "HalfToneRandomization"

    def configure(self, params: dict) -> None:
        """Configure the rule from parameters.

        Args:
            params: Dictionary with optional keys:
                - dot_size: int (1-20)
                - dot_density: float (0.0-1.0)
                - temporal_smoothing: float (0.0-1.0)
                - seed: int

        Raises:
            ValueError: If any parameter is invalid
        """
        if "dot_size" in params:
            dot_size = int(params["dot_size"])
            if not (1 <= dot_size <= 20):
                raise ValueError(f"dot_size must be in [1, 20], got {dot_size}")
            self.dot_size = dot_size

        if "dot_density" in params:
            density = float(params["dot_density"])
            if not (0.0 <= density <= 1.0):
                raise ValueError(f"dot_density must be in [0.0, 1.0], got {density}")
            self.dot_density = density

        if "temporal_smoothing" in params:
            smoothing = float(params["temporal_smoothing"])
            if not (0.0 <= smoothing <= 1.0):
                raise ValueError(f"temporal_smoothing must be in [0.0, 1.0], got {smoothing}")
            self.temporal_smoothing = smoothing

        if "seed" in params:
            self.seed = int(params["seed"])

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        """Apply halftone effect to the frame.

        Args:
            frame: Input RGB array (H, W, 3)
            context: Frame context

        Returns:
            (halftone_frame, updated_context) tuple
        """
        height, width = frame.shape[:2]

        # Get previous state
        state = context.get_rule_state(self.name())

        # Create RNG seeded with frame_index for deterministic evolution
        rng = np.random.RandomState(self.seed + context.frame_index)

        if state is None or not isinstance(state, HalfToneState):
            # First frame or invalid state: generate new dots
            dots = _generate_dot_grid(
                width, height, self.dot_size, self.dot_density, rng
            )
        else:
            # Evolve existing dots
            prev_dots = state.dots

            # Generate new target positions
            new_dots = _generate_dot_grid(
                width, height, self.dot_size, self.dot_density, rng
            )

            # For simplicity, if counts don't match, just use new dots
            # A more sophisticated approach would interpolate
            if len(new_dots) == len(prev_dots):
                # Blend between old and new based on temporal_smoothing
                # Higher smoothing = stay closer to previous
                blend_factor = 1.0 - self.temporal_smoothing
                dots = prev_dots + blend_factor * (new_dots - prev_dots)
            else:
                dots = new_dots

        # Store state
        new_state = HalfToneState(dots=dots)
        context.set_rule_state(self.name(), new_state)

        # Render halftone
        result = _render_halftone(frame, dots)

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
