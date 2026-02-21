"""Person segmentation rules.

Implements PersonSegmentationRule for extracting person silhouettes
using MediaPipe selfie segmentation or SAM2.
"""

from typing import Literal

import cv2
import numpy as np

from my_amv.context import FrameContext
from my_amv.rule import EffectRule
from my_amv.types import Layer, MaskArray, RGBArray


ModelType = Literal["mediapipe", "sam2"]


class PersonSegmentationRule(EffectRule[None]):
    """Extract person silhouette/mask from frames.

    Supports two segmentation models:
    - "mediapipe": MediaPipe Selfie Segmentation (lighter, no prompt needed)
    - "sam2": Segment Anything Model 2 (higher quality, optional ROI hint)

    The rule can optionally apply the mask to the current frame, zeroing
    out the background pixels.

    Attributes:
        model: Segmentation model to use
        apply_mask_to_frame: If True, apply mask to current frame (zero background)
        roi: Optional region of interest hint [x1, y1, x2, y2] for SAM2
        output_layer: Layer to store mask (default: Layer.MASK)

    Example:
        # Extract mask only
        rule = PersonSegmentationRule(model="mediapipe")

        # Extract mask and apply to frame
        rule = PersonSegmentationRule(
            model="mediapipe",
            apply_mask_to_frame=True
        )
    """

    def __init__(
        self,
        model: ModelType = "mediapipe",
        apply_mask_to_frame: bool = False,
        roi: list[int] | None = None,
        output_layer: Layer = Layer.MASK,
    ) -> None:
        """Initialize PersonSegmentationRule.

        Args:
            model: Segmentation model ("mediapipe" or "sam2")
            apply_mask_to_frame: If True, zero out background on current frame
            roi: Optional bounding box hint [x1, y1, x2, y2] for SAM2
            output_layer: Layer to store mask output
        """
        self.model = model
        self.apply_mask_to_frame = apply_mask_to_frame
        self.roi = roi
        self.output_layer = output_layer
        self.input_layer = Layer.MAIN

        # Lazy-loaded models
        self._mediapipe_model: object | None = None
        self._sam2_model: object | None = None

    def name(self) -> str:
        return "SilhouetteExtract"

    def _get_mediapipe_model(self):
        """Lazy-load MediaPipe selfie segmentation model."""
        if self._mediapipe_model is not None:
            return self._mediapipe_model

        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError(
                "Install mediapipe to use model='mediapipe'. "
                "Run: pip install mediapipe"
            ) from e

        self._mediapipe_model = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 0 = general, 1 = landscape (higher accuracy)
        )
        return self._mediapipe_model

    def _get_sam2_model(self):
        """Lazy-load SAM2 model."""
        if self._sam2_model is not None:
            return self._sam2_model

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise RuntimeError(
                "Install SAM2 to use model='sam2'. "
                "See: https://github.com/facebookresearch/segment-anything-2"
            ) from e

        # Build default SAM2 model
        # Note: User may want to configure specific model/checkpoint paths
        sam2_model = build_sam2("sam2.1_hiera_large.yaml", "sam2.1_hiera_large.pt")
        self._sam2_model = SAM2ImagePredictor(sam2_model)
        return self._sam2_model

    def _apply_mediapipe(self, frame: RGBArray) -> MaskArray:
        """Apply MediaPipe selfie segmentation.

        Args:
            frame: RGB input frame (BGR for cv2)

        Returns:
            Binary mask array (0 or 255)
        """
        model = self._get_mediapipe_model()

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process
        results = model.process(rgb_frame)

        if results.segmentation_mask is None:
            # No person detected, return empty mask
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Get segmentation mask (float 0-1)
        mask_float = results.segmentation_mask

        # Threshold at 0.5 for binary mask
        mask_binary = (mask_float > 0.5).astype(np.uint8) * 255

        return mask_binary

    def _apply_sam2(self, frame: RGBArray) -> MaskArray:
        """Apply SAM2 segmentation.

        Args:
            frame: RGB input frame

        Returns:
            Binary mask array (0 or 255)
        """
        predictor = self._get_sam2_model()

        # SAM2 expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set image
        predictor.set_image(rgb_frame)

        if self.roi is not None:
            # Use ROI hint as box prompt
            # ROI format: [x1, y1, x2, y2]
            box = np.array(self.roi)
            masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        else:
            # Use center point as prompt (default behavior)
            h, w = frame.shape[:2]
            point_coords = np.array([[w // 2, h // 2]])
            point_labels = np.array([1])  # 1 = foreground point
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )

        # Get best mask
        mask_float = masks[0]  # Shape (H, W)

        # Convert to binary
        mask_binary = (mask_float > 0.5).astype(np.uint8) * 255

        return mask_binary

    def _segment(self, frame: RGBArray) -> MaskArray:
        """Run segmentation to get person mask.

        Args:
            frame: RGB input frame

        Returns:
            Binary mask (0=background, 255=person)
        """
        if self.model == "mediapipe":
            return self._apply_mediapipe(frame)
        elif self.model == "sam2":
            return self._apply_sam2(frame)
        else:
            raise ValueError(
                f"Unknown model: {self.model}. "
                f"Valid: mediapipe, sam2"
            )

    def _apply_mask_to_frame(self, frame: RGBArray, mask: MaskArray) -> RGBArray:
        """Apply mask to frame, zeroing out background pixels.

        Args:
            frame: RGB input frame
            mask: Binary mask (0 or 255)

        Returns:
            Frame with background zeroed
        """
        result = frame.copy()
        # Where mask is 0, set frame pixels to 0
        result[mask == 0] = 0
        return result

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        """Apply person segmentation to the frame.

        Args:
            frame: Input RGB frame
            context: Frame context

        Returns:
            (output_frame, context) - mask stored in context layer
            If apply_mask_to_frame=True, output_frame has background zeroed
        """
        mask = self._segment(frame)

        # Store mask in layer (as RGB for consistency)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        context.set_layer(self.output_layer, mask_rgb)

        # Apply mask to frame if requested
        if self.apply_mask_to_frame:
            output_frame = self._apply_mask_to_frame(frame, mask)
        else:
            output_frame = frame

        return output_frame, context

    def configure(self, params: dict) -> None:
        """Configure the rule with parameters.

        Args:
            params: Dictionary with keys:
                - model: str (optional)
                - apply_mask_to_frame: bool (optional)
                - roi: list[int] (optional)
        """
        if "model" in params:
            valid = {"mediapipe", "sam2"}
            if params["model"] not in valid:
                raise ValueError(f"Invalid model: {params['model']}")
            self.model = params["model"]

        if "apply_mask_to_frame" in params:
            self.apply_mask_to_frame = bool(params["apply_mask_to_frame"])

        if "roi" in params:
            roi = params["roi"]
            if roi is not None:
                if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                    raise ValueError("roi must be a list of 4 ints: [x1, y1, x2, y2]")
                self.roi = list(roi)
            else:
                self.roi = None
