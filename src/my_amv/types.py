"""Core type definitions for the my_amv pipeline.

This module defines the foundational types used throughout the pipeline.
Critical invariants (from AGENTS.md):
- Layer MUST be an Enum, NOT StrEnum (preserves dict[LayerKey, RGBArray] union type)
- LayerKey = Layer | str (built-in layers use enum, user layers use str)
"""

from enum import Enum
from typing import Literal, TypeVar, Union

import numpy as np
from numpy.typing import NDArray


class Layer(Enum):
    """Named layers in the compositing system.

    MUST be a regular Enum, not StrEnum. This preserves the distinction
    between Layer enum members and arbitrary string layer names in type checking.
    """

    ORIGINAL = "original"
    MASK = "mask"
    EDGES = "edges"
    DEPTH = "depth"
    SKELETON = "skeleton"
    FINAL = "final"
    MAIN = "main"


LayerKey = Union[Layer, str]
"""Union of built-in Layer enum and user-defined string layer names.

Built-in layers (Layer enum):
  - ORIGINAL: Unmodified input frame
  - MASK: Person silhouette binary mask
  - EDGES: Edge/contour map
  - DEPTH: Depth map
  - SKELETON: Pose skeleton overlay
  - FINAL: Final composited output
  - MAIN: Default working layer for linear chain behavior

User-defined layers: Any string name (e.g., "halftone", "audio_lines")
"""

BlendMode = Literal["screen", "multiply", "add", "over", "normal"]
"""Blend modes for layer compositing.

- screen: Lighten - 1-(1-a)(1-b)
- multiply: Darken - a*b/255
- add: Sum - min(a+b, 255)
- over: Alpha compositing (foreground over background)
- normal: Foreground replaces background
"""

# Array type aliases
RGBArray = NDArray[np.uint8]  # shape (H, W, 3)
"""RGB image array with uint8 values [0, 255]."""

MaskArray = NDArray[np.uint8]  # shape (H, W) values 0|255
"""Binary mask array with values 0 or 255."""

EdgeArray = NDArray[np.uint8]  # shape (H, W)
"""Edge/contour array with uint8 values."""

DepthArray = NDArray[np.float32]  # shape (H, W) 0.0-1.0
"""Depth map array with normalized float values [0.0, 1.0]."""

StateT = TypeVar("StateT")
"""Type variable for EffectRule state.
Each rule can define its own state type for temporal coherence.
"""
