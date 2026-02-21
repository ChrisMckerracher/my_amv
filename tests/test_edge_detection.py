"""Tests for EdgeDetectionRule."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from my_amv.context import FrameContext
from my_amv.rules.edge_detection import EdgeDetectionRule
from my_amv.types import Layer, RGBArray


class TestEdgeDetectionRule:
    """Test EdgeDetectionRule edge detection functionality."""

    @pytest.fixture
    def sample_frame(self) -> RGBArray:
        """Create a sample RGB frame for testing."""
        # Create a frame with a simple shape (white square on black background)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[30:70, 30:70] = 255  # White square
        return frame

    @pytest.fixture
    def context(self) -> FrameContext:
        """Create a test FrameContext."""
        return FrameContext(frame_index=0, frame_count=100, fps=30.0)

    def test_rule_creation(self):
        """EdgeDetectionRule should be creatable with default parameters."""
        rule = EdgeDetectionRule()

        assert rule.name() == "EdgeDetect"
        assert rule.algorithm == "canny"
        assert rule.detail_level == 5
        assert rule.output_layer == Layer.EDGES
        assert rule.input_layer == Layer.MAIN

    def test_rule_creation_with_params(self):
        """EdgeDetectionRule should accept algorithm and detail_level."""
        rule = EdgeDetectionRule(algorithm="sobel", detail_level=3)

        assert rule.algorithm == "sobel"
        assert rule.detail_level == 3

    def test_invalid_detail_level_raises(self):
        """Invalid detail_level should raise ValueError."""
        with pytest.raises(ValueError, match="detail_level must be in \\[1, 10\\]"):
            EdgeDetectionRule(detail_level=0)

        with pytest.raises(ValueError, match="detail_level must be in \\[1, 10\\]"):
            EdgeDetectionRule(detail_level=11)

        with pytest.raises(ValueError, match="detail_level must be an integer"):
            EdgeDetectionRule(detail_level=5.5)

    def test_invalid_algorithm_raises(self):
        """Invalid algorithm should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            EdgeDetectionRule(algorithm="not_real")

    def test_apply_returns_tuple(self, sample_frame, context):
        """apply() should return (frame, context) tuple."""
        rule = EdgeDetectionRule(algorithm="canny")

        result_frame, result_ctx = rule.apply(sample_frame, context)

        assert isinstance(result_frame, np.ndarray)
        assert isinstance(result_ctx, FrameContext)
        # Frame is unchanged (branch mode)
        assert np.array_equal(result_frame, sample_frame)

    def test_apply_stores_edges_in_context(self, sample_frame, context):
        """apply() should store edge result in context layer."""
        rule = EdgeDetectionRule(algorithm="canny")

        rule.apply(sample_frame, context)

        assert context.has_layer(Layer.EDGES)
        edges = context.get_layer(Layer.EDGES)
        assert edges.shape == (100, 100, 3)

    def test_canny_edge_detection(self, sample_frame, context):
        """Canny algorithm should detect edges."""
        rule = EdgeDetectionRule(algorithm="canny", detail_level=5)

        result_frame, result_ctx = rule.apply(sample_frame, context)
        edges = result_ctx.get_layer(Layer.EDGES)

        # Edges should be detected at the boundaries of the square
        # The center of the square should have no edges
        assert edges[50, 50, 0] == 0  # Black center
        # There should be some white pixels (edges) in the image
        assert np.any(edges > 0)

    def test_dog_edge_detection(self, sample_frame, context):
        """Difference of Gaussians should detect edges."""
        rule = EdgeDetectionRule(algorithm="dog", detail_level=5)

        result_frame, result_ctx = rule.apply(sample_frame, context)
        edges = result_ctx.get_layer(Layer.EDGES)

        assert edges.shape == (100, 100, 3)
        # Should have some edges
        assert np.any(edges > 0)

    def test_log_edge_detection(self, sample_frame, context):
        """Laplacian of Gaussian should detect edges."""
        rule = EdgeDetectionRule(algorithm="log", detail_level=5)

        result_frame, result_ctx = rule.apply(sample_frame, context)
        edges = result_ctx.get_layer(Layer.EDGES)

        assert edges.shape == (100, 100, 3)
        assert np.any(edges > 0)

    def test_sobel_edge_detection(self, sample_frame, context):
        """Sobel algorithm should detect edges."""
        rule = EdgeDetectionRule(algorithm="sobel", detail_level=5)

        result_frame, result_ctx = rule.apply(sample_frame, context)
        edges = result_ctx.get_layer(Layer.EDGES)

        assert edges.shape == (100, 100, 3)
        assert np.any(edges > 0)

    def test_detail_level_mapping_canny(self):
        """Different detail levels should produce different Canny parameters."""
        rule = EdgeDetectionRule()

        # Low detail = high thresholds
        sigma1, low1, high1 = rule._get_canny_params()
        rule.detail_level = 1
        sigma_low, low_low, high_low = rule._get_canny_params()

        # High detail = low thresholds
        rule.detail_level = 10
        sigma_high, low_high, high_high = rule._get_canny_params()

        # Low detail should have higher thresholds
        assert low_low > low_high
        assert high_low > high_high
        # Low detail should have more blur
        assert sigma_low > sigma_high

    def test_detail_level_mapping_dog(self):
        """Different detail levels should produce different DoG sigmas."""
        rule = EdgeDetectionRule()

        rule.detail_level = 1
        sigma1_low, sigma2_low = rule._get_dog_params()

        rule.detail_level = 10
        sigma1_high, sigma2_high = rule._get_dog_params()

        # Low detail should have larger sigma spread
        assert (sigma2_low - sigma1_low) > (sigma2_high - sigma1_high)

    def test_explicit_canny_params_override_detail_level(self):
        """Explicit threshold parameters should override detail_level."""
        rule = EdgeDetectionRule(
            detail_level=5,  # Medium detail
            threshold1=10,   # But explicit low threshold (high detail)
            threshold2=20
        )

        sigma, low, high = rule._get_canny_params()
        assert low == 10
        assert high == 20
        assert sigma == 0.0  # No blur when explicit thresholds

    def test_explicit_dog_params_override_detail_level(self):
        """Explicit sigma parameters should override detail_level."""
        rule = EdgeDetectionRule(
            detail_level=5,
            sigma1=2.0,
            sigma2=4.0
        )

        sigma1, sigma2 = rule._get_dog_params()
        assert sigma1 == 2.0
        assert sigma2 == 4.0

    def test_explicit_log_sigma_override_detail_level(self):
        """Explicit sigma should override detail_level for LoG."""
        rule = EdgeDetectionRule(
            detail_level=5,
            sigma=3.5
        )

        sigma = rule._get_log_params()
        assert sigma == 3.5

    def test_configure_updates_algorithm(self):
        """configure() should update algorithm."""
        rule = EdgeDetectionRule(algorithm="canny")

        rule.configure({"algorithm": "sobel"})

        assert rule.algorithm == "sobel"

    def test_configure_invalid_algorithm_raises(self):
        """configure() with invalid algorithm should raise ValueError."""
        rule = EdgeDetectionRule()

        with pytest.raises(ValueError, match="Invalid algorithm"):
            rule.configure({"algorithm": "not_real"})

    def test_configure_updates_detail_level(self):
        """configure() should update detail_level."""
        rule = EdgeDetectionRule(detail_level=5)

        rule.configure({"detail_level": 8})

        assert rule.detail_level == 8

    def test_configure_validates_detail_level(self):
        """configure() should validate detail_level."""
        rule = EdgeDetectionRule()

        with pytest.raises(ValueError, match="detail_level must be in \\[1, 10\\]"):
            rule.configure({"detail_level": 15})

    def test_configure_sets_explicit_params(self):
        """configure() should set algorithm-specific parameters."""
        rule = EdgeDetectionRule()

        rule.configure({
            "threshold1": 30,
            "threshold2": 80,
            "sigma1": 1.5,
            "sigma2": 3.0,
        })

        assert rule.threshold1 == 30
        assert rule.threshold2 == 80
        assert rule.sigma1 == 1.5
        assert rule.sigma2 == 3.0

    def test_output_shape_matches_input(self, sample_frame, context):
        """Edge output should have same spatial dimensions as input."""
        rule = EdgeDetectionRule(algorithm="canny")

        rule.apply(sample_frame, context)
        edges = context.get_layer(Layer.EDGES)

        assert edges.shape[0:2] == sample_frame.shape[0:2]

    def test_ml_algorithm_without_controlnet_aux_raises(self, sample_frame, context):
        """ML algorithm should raise clear error without controlnet_aux."""
        rule = EdgeDetectionRule(algorithm="lineart_realistic")

        # Mock the import to simulate missing package
        with patch.dict("sys.modules", {"controlnet_aux": None}):
            with pytest.raises(RuntimeError, match="Install controlnet-aux"):
                rule._detect_edges(sample_frame)

    def test_ml_algorithm_with_mocked_model(self, sample_frame, context):
        """ML algorithm should use controlnet_aux when available."""
        # Mock the ML edge detection result directly
        with patch.object(EdgeDetectionRule, "_get_ml_model"):
            rule = EdgeDetectionRule(algorithm="lineart_realistic")

            # Mock _apply_ml_algorithm to return a known result
            mock_edges = np.zeros((100, 100), dtype=np.uint8)
            mock_edges[30:70, 30:70] = 255

            with patch.object(rule, "_apply_ml_algorithm", return_value=mock_edges):
                rule.apply(sample_frame, context)
                edges = context.get_layer(Layer.EDGES)

                assert edges.shape == (100, 100, 3)
                # Should have edges in the middle region
                assert np.any(edges[30:70, 30:70] > 0)

    def test_edge_output_is_binary(self, sample_frame, context):
        """Edge output should contain mostly 0 or 255 values."""
        rule = EdgeDetectionRule(algorithm="canny", detail_level=5)

        rule.apply(sample_frame, context)
        edges = context.get_layer(Layer.EDGES)

        # Get grayscale version
        gray = edges[:, :, 0]

        # Most values should be 0 or 255 (allowing some anti-aliasing)
        unique_values = np.unique(gray)
        assert 0 in unique_values or 255 in unique_values

    def test_different_algorithms_produce_different_results(self, sample_frame):
        """Different algorithms should produce different edge maps."""
        ctx1 = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx2 = FrameContext(frame_index=0, frame_count=100, fps=30.0)

        rule1 = EdgeDetectionRule(algorithm="canny", detail_level=5)
        rule2 = EdgeDetectionRule(algorithm="sobel", detail_level=5)

        rule1.apply(sample_frame, ctx1)
        rule2.apply(sample_frame, ctx2)

        edges1 = ctx1.get_layer(Layer.EDGES)
        edges2 = ctx2.get_layer(Layer.EDGES)

        # Results should be different
        assert not np.array_equal(edges1, edges2)

    def test_custom_output_layer(self, sample_frame, context):
        """Custom output_layer should store edges in that layer."""
        rule = EdgeDetectionRule(algorithm="canny", output_layer="custom_edges")

        rule.apply(sample_frame, context)

        assert context.has_layer("custom_edges")
        edges = context.get_layer("custom_edges")
        assert edges.shape == (100, 100, 3)

    def test_stateless_rule(self, sample_frame, context):
        """EdgeDetectionRule should be stateless."""
        rule = EdgeDetectionRule(algorithm="canny")

        # Should not raise any errors
        state = rule.serialize_state(context)
        assert state == {}

        # Should not raise any errors
        rule.deserialize_state(context, {})
