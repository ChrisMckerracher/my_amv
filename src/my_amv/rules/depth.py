"""Depth estimation rules.

This module provides depth map estimation from video frames using
Depth Anything V2 (primary) or MiDaS (fallback) models.
"""

from typing import Literal

import numpy as np

from my_amv.context import FrameContext
from my_amv.rule import EffectRule
from my_amv.types import DepthArray, Layer, LayerKey, RGBArray
from my_amv.rules import register_rule


@register_rule
class DepthMappingRule(EffectRule[None]):
    """Generate depth map from video frames.

    Uses Depth Anything V2 via transformers as primary model.
    Falls back to MiDaS via torch.hub if transformers is unavailable.

    Outputs:
        - Layer.DEPTH: Float32 depth array [0.0, 1.0]
        - "depth_visual": uint8 visualization [0, 255]

    Parameters:
        model: Model to use ("depth_anything_v2" or "midas")
        size: Model size ("small", "base", "large")
        near_bright: If True, nearer pixels are brighter
    """

    def __init__(
        self,
        model: Literal["depth_anything_v2", "midas"] = "depth_anything_v2",
        size: Literal["small", "base", "large"] = "base",
        near_bright: bool = True,
    ) -> None:
        """Initialize the DepthMappingRule.

        Args:
            model: Model selection ("depth_anything_v2" or "midas")
            size: Model size variant ("small", "base", "large")
            near_bright: If True, nearer pixels have higher values
        """
        self.model = model
        self.size = size
        self.near_bright = near_bright
        self._estimator = None
        self.output_layer = Layer.DEPTH

    def name(self) -> str:
        """Return the rule name."""
        return "DepthMap"

    def configure(self, params: dict) -> None:
        """Configure the rule from parameters.

        Args:
            params: Dictionary with optional keys:
                - model: str ("depth_anything_v2" | "midas")
                - size: str ("small" | "base" | "large")
                - near_bright: bool

        Raises:
            ValueError: If any parameter is invalid
        """
        if "model" in params:
            model = params["model"]
            if model not in ("depth_anything_v2", "midas"):
                raise ValueError(f"Invalid model: {model}")
            self.model = model

        if "size" in params:
            size = params["size"]
            if size not in ("small", "base", "large"):
                raise ValueError(f"Invalid size: {size}")
            self.size = size

        if "near_bright" in params:
            self.near_bright = bool(params["near_bright"])

    def _load_estimator(self):
        """Load the depth estimation model.

        Tries Depth Anything V2 first, falls back to MiDaS.

        Returns:
            Model object with predict() method

        Raises:
            ImportError: If neither model library is available
        """
        # Try Depth Anything V2 first
        if self.model == "depth_anything_v2":
            try:
                from transformers import pipeline

                model_size = self._get_model_size_string()
                model_name = f"depth-anything/Depth-Anything-V2-{model_size}"

                estimator = pipeline(
                    task="depth-estimation",
                    model=model_name,
                )
                return estimator
            except ImportError:
                # Fall through to MiDaS
                pass

        # Fallback to MiDaS
        try:
            import torch

            midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas_model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.small_transform

            # Wrap MiDaS to match expected interface
            class MiDaSEstimator:
                def __init__(self, model, transform):
                    self.model = model
                    self.transform = transform
                    import torch
                    self.torch = torch

                def predict(self, image: RGBArray) -> DepthArray:
                    import cv2

                    # Convert RGB to expected format
                    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    input_batch = self.transform(img).unsqueeze(0)

                    with self.torch.no_grad():
                        prediction = self.model(input_batch)
                        prediction = self.torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=image.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze()

                    output = prediction.cpu().numpy()
                    # Normalize to [0, 1]
                    output = (output - output.min()) / (output.max() - output.min() + 1e-8)
                    return output.astype(np.float32)

            return MiDaSEstimator(midas_model, transform)

        except ImportError:
            raise ImportError(
                "Depth estimation requires either transformers or torch. "
                "Install with: pip install transformers torch"
            )

    def _get_model_size_string(self) -> str:
        """Get the model size string for transformers.

        Returns:
            Model size variant name
        """
        size_map = {
            "small": "Small",
            "base": "Base",
            "large": "Large",
        }
        return size_map.get(self.size, "Base")

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        """Apply depth estimation to the frame.

        Args:
            frame: Input RGB array (H, W, 3)
            context: Frame context

        Returns:
            (depth_visualization, updated_context) tuple
            - depth_visualization: uint8 RGB visualization
            - updated_context: With Layer.DEPTH and "depth_visual" set
        """
        if self._estimator is None:
            self._estimator = self._load_estimator()

        # Run depth estimation
        depth = self._estimator.predict(frame)

        # Ensure float32 and [0, 1] range
        depth = depth.astype(np.float32)
        depth = np.clip(depth, 0.0, 1.0)

        # Invert if near_bright is False
        if not self.near_bright:
            depth = 1.0 - depth

        # Store raw depth map
        context.set_layer(Layer.DEPTH, depth)

        # Create uint8 visualization (multiply by 255)
        depth_visual = (depth * 255).astype(np.uint8)

        # Convert to RGB by replicating the channel
        depth_rgb = np.stack([depth_visual] * 3, axis=-1).astype(np.uint8)
        context.set_layer("depth_visual", depth_rgb)

        # Return the visualization as the frame output
        return depth_rgb, context
