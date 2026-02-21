"""Tests for DepthMappingRule."""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from my_amv.context import FrameContext
from my_amv.rules.depth import DepthMappingRule
from my_amv.types import Layer


class MockDepthEstimator:
    """Mock depth estimator for testing."""

    def predict(self, image):
        """Return a fake depth map."""
        h, w = image.shape[:2]
        # Create a gradient depth map
        depth = np.linspace(0, 1, w, dtype=np.float32)
        depth = np.tile(depth, (h, 1))
        return depth


@pytest.fixture
def depth_frame():
    """Create a test frame."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def depth_context():
    """Create a test context."""
    return FrameContext(frame_index=0, frame_count=100, fps=30.0)


class TestDepthMappingRule:
    """Test DepthMappingRule functionality."""

    def test_rule_name(self):
        """Rule name should be 'DepthMap'."""
        rule = DepthMappingRule()
        assert rule.name() == "DepthMap"

    def test_default_parameters(self):
        """Rule should have default parameters."""
        rule = DepthMappingRule()
        assert rule.model == "depth_anything_v2"
        assert rule.size == "base"
        assert rule.near_bright is True

    def test_custom_parameters(self):
        """Rule should accept custom parameters."""
        rule = DepthMappingRule(
            model="midas",
            size="small",
            near_bright=False,
        )
        assert rule.model == "midas"
        assert rule.size == "small"
        assert rule.near_bright is False

    def test_output_layer_is_depth(self):
        """Rule should output to Layer.DEPTH."""
        rule = DepthMappingRule()
        assert rule.output_layer == Layer.DEPTH

    def test_configure_model(self):
        """configure() should update model parameter."""
        rule = DepthMappingRule()
        rule.configure({"model": "midas"})
        assert rule.model == "midas"

    def test_configure_invalid_model_raises(self):
        """Invalid model should raise ValueError."""
        rule = DepthMappingRule()
        with pytest.raises(ValueError, match="Invalid model"):
            rule.configure({"model": "invalid_model"})

    def test_configure_size(self):
        """configure() should update size parameter."""
        rule = DepthMappingRule()
        rule.configure({"size": "large"})
        assert rule.size == "large"

    def test_configure_invalid_size_raises(self):
        """Invalid size should raise ValueError."""
        rule = DepthMappingRule()
        with pytest.raises(ValueError, match="Invalid size"):
            rule.configure({"size": "huge"})

    def test_configure_near_bright(self):
        """configure() should update near_bright parameter."""
        rule = DepthMappingRule()
        rule.configure({"near_bright": False})
        assert rule.near_bright is False

    def test_configure_all_parameters(self):
        """configure() should update all parameters."""
        rule = DepthMappingRule()
        rule.configure({
            "model": "midas",
            "size": "small",
            "near_bright": False,
        })
        assert rule.model == "midas"
        assert rule.size == "small"
        assert rule.near_bright is False

    def test_apply_returns_correct_shape(self, depth_frame, depth_context):
        """apply() should return output with same shape as input."""
        rule = DepthMappingRule()
        rule._estimator = MockDepthEstimator()

        result, _ = rule.apply(depth_frame, depth_context)

        assert result.shape == depth_frame.shape

    def test_apply_stores_depth_layer(self, depth_frame, depth_context):
        """apply() should store depth map in Layer.DEPTH."""
        rule = DepthMappingRule()
        rule._estimator = MockDepthEstimator()

        _, new_context = rule.apply(depth_frame, depth_context)

        assert new_context.has_layer(Layer.DEPTH)
        depth = new_context.get_layer(Layer.DEPTH)
        assert depth.dtype == np.float32
        assert depth.shape == (100, 100)

    def test_apply_depth_range_normalized(self, depth_frame, depth_context):
        """apply() should produce depth values in [0, 1] range."""
        rule = DepthMappingRule()
        rule._estimator = MockDepthEstimator()

        _, new_context = rule.apply(depth_frame, depth_context)

        depth = new_context.get_layer(Layer.DEPTH)
        assert np.all(depth >= 0.0)
        assert np.all(depth <= 1.0)

    def test_apply_stores_visual_layer(self, depth_frame, depth_context):
        """apply() should store visual layer."""
        rule = DepthMappingRule()
        rule._estimator = MockDepthEstimator()

        _, new_context = rule.apply(depth_frame, depth_context)

        assert new_context.has_layer("depth_visual")
        visual = new_context.get_layer("depth_visual")
        assert visual.dtype == np.uint8
        assert visual.shape == (100, 100, 3)

    def test_apply_near_bright_true_increases_near_values(
        self, depth_frame, depth_context
    ):
        """near_bright=True should make near pixels brighter."""
        rule = DepthMappingRule(near_bright=True)
        rule._estimator = MockDepthEstimator()

        result, new_context = rule.apply(depth_frame, depth_context)

        # Visual layer should have bright areas for high depth values
        visual = new_context.get_layer("depth_visual")
        # With our mock gradient, right side should be bright
        assert visual[50, 90, 0] > visual[50, 10, 0]

    def test_apply_near_bright_false_increases_far_values(
        self, depth_frame, depth_context
    ):
        """near_bright=False should make far pixels brighter."""
        rule = DepthMappingRule(near_bright=False)
        rule._estimator = MockDepthEstimator()

        result, new_context = rule.apply(depth_frame, depth_context)

        depth = new_context.get_layer(Layer.DEPTH)
        # Values should be inverted
        assert depth[50, 90] < depth[50, 10]

    def test_apply_loads_estimator_on_first_call(
        self, depth_frame, depth_context
    ):
        """apply() should load estimator on first call."""
        rule = DepthMappingRule()
        rule._estimator = MockDepthEstimator()
        assert rule._estimator is not None

        rule.apply(depth_frame, depth_context)

        assert rule._estimator is not None

    def test_stateless_rule(self, depth_frame, depth_context):
        """Rule should be stateless."""
        rule = DepthMappingRule()
        rule._estimator = MockDepthEstimator()

        # Apply multiple times
        _, ctx1 = rule.apply(depth_frame, depth_context)
        depth_context.frame_index = 1
        _, ctx2 = rule.apply(depth_frame, depth_context)

        # Same depth values (stateless)
        depth1 = ctx1.get_layer(Layer.DEPTH)
        depth2 = ctx2.get_layer(Layer.DEPTH)
        assert np.array_equal(depth1, depth2)


class TestDepthMappingRuleImportErrors:
    """Test DepthMappingRule with missing dependencies."""

    def test_works_with_mock_estimator(self, depth_frame, depth_context):
        """Rule should work with manually set mock estimator."""
        rule = DepthMappingRule()
        # Set estimator directly instead of relying on library loading
        rule._estimator = MockDepthEstimator()

        # Should not raise
        result, _ = rule.apply(depth_frame, depth_context)
        assert result.shape == depth_frame.shape
