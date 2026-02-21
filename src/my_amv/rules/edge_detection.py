"""Edge detection rules.

Implements EdgeDetectionRule with support for classical (Canny, DoG, LoG, Sobel)
and AI-based (Lineart, HED, PiDiNet) edge detection algorithms.
"""

from typing import Literal

import cv2
import numpy as np

from my_amv.context import FrameContext
from my_amv.rule import EffectRule
from my_amv.types import EdgeArray, Layer, RGBArray


# Valid algorithm types
AlgorithmType = Literal[
    "canny",
    "dog",
    "log",
    "sobel",
    "lineart_realistic",
    "lineart_anime",
    "hed",
    "pidinet",
]


class EdgeDetectionRule(EffectRule[None]):
    """Detect edges in frames using configurable algorithms.

    Supports classical CV algorithms (Canny, DoG, LoG, Sobel) and
    AI-based algorithms (Lineart, HED, PiDiNet) via controlnet_aux.

    The detail_level parameter (1-10) controls edge density:
    - 1-3: Silhouette only (high blur/threshold)
    - 4-6: Structural lines (moderate settings)
    - 7-8: Secondary contours (light processing)
    - 9-10: Fine detail (minimal processing)

    Attributes:
        algorithm: Edge detection algorithm to use
        detail_level: Detail density 1-10
        output_layer: Layer to store edges (default: Layer.EDGES)
        threshold1: Low threshold for Canny (overrides detail_level)
        threshold2: High threshold for Canny (overrides detail_level)
        sigma1: First Gaussian sigma for DoG/LoG
        sigma2: Second Gaussian sigma for DoG

    Example:
        rule = EdgeDetectionRule(algorithm="canny", detail_level=5)
        # Or with explicit parameters
        rule = EdgeDetectionRule(algorithm="canny", threshold1=50, threshold2=150)
    """

    def __init__(
        self,
        algorithm: AlgorithmType = "canny",
        detail_level: int = 5,
        output_layer: Layer = Layer.EDGES,
        **kwargs,
    ) -> None:
        """Initialize EdgeDetectionRule.

        Args:
            algorithm: Edge detection algorithm
            detail_level: Detail density 1-10 (used if explicit params not provided)
            output_layer: Layer to store edge output
            **kwargs: Algorithm-specific overrides (threshold1, threshold2, sigma1, sigma2)
        """
        self.algorithm = self._validate_algorithm(algorithm)
        self.detail_level = self._validate_detail_level(detail_level)
        self.output_layer = output_layer
        self.input_layer = Layer.MAIN

        # Algorithm-specific parameter overrides
        self.threshold1 = kwargs.get("threshold1")
        self.threshold2 = kwargs.get("threshold2")
        self.sigma1 = kwargs.get("sigma1")
        self.sigma2 = kwargs.get("sigma2")
        self.sigma = kwargs.get("sigma")

        # Lazy-loaded ML models
        self._ml_model: object | None = None

    def _validate_detail_level(self, level: int) -> int:
        """Validate and clamp detail level to [1, 10]."""
        if not isinstance(level, int):
            raise ValueError(f"detail_level must be an integer, got {type(level).__name__}")
        if not 1 <= level <= 10:
            raise ValueError(f"detail_level must be in [1, 10], got {level}")
        return level

    def _validate_algorithm(self, algorithm: str) -> str:
        """Validate algorithm is supported."""
        valid = {
            "canny", "dog", "log", "sobel",
            "lineart_realistic", "lineart_anime", "hed", "pidinet"
        }
        if algorithm not in valid:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Valid: canny, dog, log, sobel, lineart_realistic, lineart_anime, hed, pidinet"
            )
        return algorithm

    def name(self) -> str:
        return "EdgeDetect"

    def _get_canny_params(self) -> tuple[float, float, float]:
        """Get Canny parameters based on detail level.

        Returns:
            (blur_sigma, low_threshold, high_threshold)
        """
        if self.threshold1 is not None and self.threshold2 is not None:
            # Use explicit thresholds, no blur
            return 0.0, float(self.threshold1), float(self.threshold2)

        # Map detail level to Canny parameters
        # Higher level = lower thresholds = more edges
        configs = {
            1: (5.0, 200, 250),
            2: (4.0, 170, 220),
            3: (3.5, 140, 190),
            4: (3.0, 120, 170),
            5: (2.5, 100, 150),
            6: (2.0, 80, 130),
            7: (1.5, 60, 110),
            8: (1.0, 40, 90),
            9: (0.5, 25, 70),
            10: (0.0, 15, 50),
        }
        return configs[self.detail_level]

    def _get_dog_params(self) -> tuple[float, float]:
        """Get Difference of Gaussians parameters based on detail level.

        Returns:
            (sigma1, sigma2) where sigma1 < sigma2
        """
        if self.sigma1 is not None and self.sigma2 is not None:
            return float(self.sigma1), float(self.sigma2)

        # Map detail level to DoG sigmas
        # Higher level = smaller ratio = more detail
        configs = {
            1: (1.0, 8.0),
            2: (1.0, 6.0),
            3: (1.0, 4.5),
            4: (1.0, 3.5),
            5: (1.0, 2.5),
            6: (1.0, 2.0),
            7: (0.8, 1.6),
            8: (0.6, 1.2),
            9: (0.5, 1.0),
            10: (0.3, 0.6),
        }
        return configs[self.detail_level]

    def _get_log_params(self) -> float:
        """Get Laplacian of Gaussian sigma based on detail level."""
        if self.sigma is not None:
            return float(self.sigma)

        # Map detail level to LoG sigma
        # Higher level = smaller sigma = more detail
        return {
            1: 5.0,
            2: 4.0,
            3: 3.5,
            4: 3.0,
            5: 2.5,
            6: 2.0,
            7: 1.5,
            8: 1.0,
            9: 0.5,
            10: 0.0,
        }[self.detail_level]

    def _apply_canny(self, gray: np.ndarray) -> EdgeArray:
        """Apply Canny edge detection.

        Args:
            gray: Grayscale input image

        Returns:
            Edge array
        """
        sigma, low, high = self._get_canny_params()

        # Apply Gaussian blur if sigma > 0
        if sigma > 0:
            ksize = int(sigma * 3) | 1  # Odd kernel size
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        else:
            blurred = gray

        edges = cv2.Canny(blurred, int(low), int(high), apertureSize=3)
        return edges

    def _apply_dog(self, gray: np.ndarray) -> EdgeArray:
        """Apply Difference of Gaussians edge detection.

        Args:
            gray: Grayscale input image

        Returns:
            Edge array
        """
        sigma1, sigma2 = self._get_dog_params()

        ksize1 = int(sigma1 * 3) | 1 if sigma1 > 0 else (1, 1)
        ksize2 = int(sigma2 * 3) | 1 if sigma2 > 0 else (1, 1)

        blur1 = cv2.GaussianBlur(gray, (ksize1, ksize1) if isinstance(ksize1, int) else ksize1, sigma1)
        blur2 = cv2.GaussianBlur(gray, (ksize2, ksize2) if isinstance(ksize2, int) else ksize2, sigma2)

        dog = cv2.absdiff(blur1, blur2)
        # Normalize to 0-255
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Threshold to get binary edges
        _, edges = cv2.threshold(dog, 30, 255, cv2.THRESH_BINARY)
        return edges

    def _apply_log(self, gray: np.ndarray) -> EdgeArray:
        """Apply Laplacian of Gaussian edge detection.

        Args:
            gray: Grayscale input image

        Returns:
            Edge array
        """
        sigma = self._get_log_params()

        if sigma > 0:
            ksize = int(sigma * 3) | 1
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        else:
            blurred = gray

        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        # Convert to absolute uint8
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)

        # Threshold
        _, edges = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
        return edges

    def _apply_sobel(self, gray: np.ndarray) -> EdgeArray:
        """Apply Sobel edge detection.

        Args:
            gray: Grayscale input image

        Returns:
            Edge array (gradient magnitude)
        """
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = cv2.magnitude(sobel_x, sobel_y)
        magnitude = np.uint8(magnitude)

        # Threshold based on detail level
        threshold = {
            1: 150, 2: 120, 3: 100, 4: 80, 5: 60,
            6: 50, 7: 40, 8: 30, 9: 20, 10: 10
        }[self.detail_level]

        _, edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        return edges

    def _get_ml_model(self):
        """Lazy-load ML model for AI-based algorithms."""
        if self._ml_model is not None:
            return self._ml_model

        try:
            from controlnet_aux import (
                HEDdetector,
                LineartAnimeDetector,
                LineartDetector,
                PidiNetDetector,
            )
        except ImportError as e:
            raise RuntimeError(
                f"Install controlnet-aux to use algorithm={self.algorithm}. "
                f"Run: pip install controlnet-aux"
            ) from e

        model_map = {
            "lineart_realistic": LineartDetector,
            "lineart_anime": LineartAnimeDetector,
            "hed": HEDdetector,
            "pidinet": PidiNetDetector,
        }

        model_cls = model_map.get(self.algorithm)
        if model_cls is None:
            raise ValueError(f"Unknown ML algorithm: {self.algorithm}")

        self._ml_model = model_cls.from_pretrained("lllyasviel/Annotators")
        return self._ml_model

    def _apply_ml_algorithm(self, frame: RGBArray) -> EdgeArray:
        """Apply AI-based edge detection algorithm.

        Args:
            frame: RGB input frame

        Returns:
            Edge array
        """
        model = self._get_ml_model()

        # Convert BGR to RGB for controlnet_aux (expects PIL RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run model
        result = model(rgb)

        # Convert back to numpy if needed
        if hasattr(result, "convert"):
            # PIL Image
            result = np.array(result.convert("L"))

        # Ensure grayscale
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        # Apply detail level thresholding
        # Higher detail level = lower threshold = more edges
        threshold = int(255 * (1.0 - (self.detail_level / 10.0) * 0.8))
        _, edges = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)

        return edges

    def _detect_edges(self, frame: RGBArray) -> EdgeArray:
        """Detect edges using the configured algorithm.

        Args:
            frame: RGB input frame

        Returns:
            Edge array
        """
        # Convert to grayscale for classical algorithms
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.algorithm == "canny":
            return self._apply_canny(gray)
        elif self.algorithm == "dog":
            return self._apply_dog(gray)
        elif self.algorithm == "log":
            return self._apply_log(gray)
        elif self.algorithm == "sobel":
            return self._apply_sobel(gray)
        elif self.algorithm in ("lineart_realistic", "lineart_anime", "hed", "pidinet"):
            return self._apply_ml_algorithm(frame)
        else:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                f"Valid: canny, dog, log, sobel, lineart_realistic, lineart_anime, hed, pidinet"
            )

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        """Apply edge detection to the frame.

        Args:
            frame: Input RGB frame
            context: Frame context

        Returns:
            (input_frame, context) - edge result stored in context layer
        """
        edges = self._detect_edges(frame)

        # Convert to RGB for layer storage (edges are grayscale)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        context.set_layer(self.output_layer, edges_rgb)

        return frame, context

    def configure(self, params: dict) -> None:
        """Configure the rule with parameters.

        Args:
            params: Dictionary with keys:
                - algorithm: str (optional)
                - detail_level: int 1-10 (optional)
                - threshold1: int (optional, Canny)
                - threshold2: int (optional, Canny)
                - sigma1: float (optional, DoG)
                - sigma2: float (optional, DoG)
                - sigma: float (optional, LoG)
        """
        if "algorithm" in params:
            valid = {"canny", "dog", "log", "sobel", "lineart_realistic", "lineart_anime", "hed", "pidinet"}
            if params["algorithm"] not in valid:
                raise ValueError(f"Invalid algorithm: {params['algorithm']}")
            self.algorithm = params["algorithm"]
            # Clear cached model if algorithm changed
            self._ml_model = None

        if "detail_level" in params:
            self.detail_level = self._validate_detail_level(params["detail_level"])

        # Allow overrides for specific algorithm parameters
        for key in ("threshold1", "threshold2", "sigma1", "sigma2", "sigma"):
            if key in params:
                setattr(self, key, params[key])
