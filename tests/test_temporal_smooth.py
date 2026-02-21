"""Tests for TemporalSmoothRule."""

import numpy as np
import pytest

from my_amv.context import FrameContext
from my_amv.rules.temporal_smooth import TemporalSmoothRule, cv2_blend


@pytest.fixture
def smooth_frame1():
    """Create first test frame."""
    return np.full((50, 50, 3), 100, dtype=np.uint8)


@pytest.fixture
def smooth_frame2():
    """Create second test frame (different from first)."""
    return np.full((50, 50, 3), 200, dtype=np.uint8)


@pytest.fixture
def smooth_context():
    """Create a test context."""
    return FrameContext(frame_index=0, frame_count=100, fps=30.0)


class TestTemporalSmoothRule:
    """Test TemporalSmoothRule functionality."""

    def test_rule_name(self):
        """Rule name should be 'TemporalSmooth'."""
        rule = TemporalSmoothRule()
        assert rule.name() == "TemporalSmooth"

    def test_default_parameters(self):
        """Rule should have default blend_factor."""
        rule = TemporalSmoothRule()
        assert rule.blend_factor == 0.7

    def test_custom_parameters(self):
        """Rule should accept custom blend_factor."""
        rule = TemporalSmoothRule(blend_factor=0.5)
        assert rule.blend_factor == 0.5

    def test_configure_blend_factor(self):
        """configure() should update blend_factor."""
        rule = TemporalSmoothRule()
        rule.configure({"blend_factor": 0.9})
        assert rule.blend_factor == 0.9

    def test_configure_invalid_blend_factor_low_raises(self):
        """blend_factor < 0 should raise ValueError."""
        rule = TemporalSmoothRule()
        with pytest.raises(ValueError, match="blend_factor must be in \\[0.0, 1.0\\]"):
            rule.configure({"blend_factor": -0.1})

    def test_configure_invalid_blend_factor_high_raises(self):
        """blend_factor > 1 should raise ValueError."""
        rule = TemporalSmoothRule()
        with pytest.raises(ValueError, match="blend_factor must be in \\[0.0, 1.0\\]"):
            rule.configure({"blend_factor": 1.5})

    def test_first_frame_is_passthrough(self, smooth_frame1, smooth_context):
        """First frame should be returned unchanged (no prior state)."""
        rule = TemporalSmoothRule(blend_factor=0.7)
        result, _ = rule.apply(smooth_frame1, smooth_context)

        # First frame = passthrough
        assert np.array_equal(result, smooth_frame1)

    def test_second_frame_is_blended(self, smooth_frame1, smooth_frame2, smooth_context):
        """Second frame should be blended with first."""
        rule = TemporalSmoothRule(blend_factor=0.7)

        # First frame
        _, ctx = rule.apply(smooth_frame1, smooth_context)
        smooth_context.frame_index = 1

        # Second frame
        result, ctx = rule.apply(smooth_frame2, ctx)

        # Result should be a blend of 100 and 200
        # With blend_factor=0.7: 0.7 * 200 + 0.3 * 100 = 140 + 30 = 170
        expected = 170
        assert result[0, 0, 0] == expected

    def test_blend_factor_one_is_passthrough(self, smooth_frame1, smooth_frame2, smooth_context):
        """blend_factor=1.0 should always return current frame (no smoothing)."""
        rule = TemporalSmoothRule(blend_factor=1.0)

        # First frame
        _, ctx = rule.apply(smooth_frame1, smooth_context)
        smooth_context.frame_index = 1

        # Second frame
        result, ctx = rule.apply(smooth_frame2, ctx)

        # Should be pure frame2 (no blending)
        assert np.array_equal(result, smooth_frame2)

    def test_blend_factor_zero_is_ema_only(self, smooth_frame1, smooth_frame2, smooth_context):
        """blend_factor=0.0 should return pure EMA (max smoothing)."""
        rule = TemporalSmoothRule(blend_factor=0.0)

        # First frame
        _, ctx = rule.apply(smooth_frame1, smooth_context)
        smooth_context.frame_index = 1

        # Second frame
        result, ctx = rule.apply(smooth_frame2, ctx)

        # Should be pure frame1 (EMA doesn't update)
        assert np.array_equal(result, smooth_frame1)

    def test_blend_factor_half(self, smooth_frame1, smooth_frame2, smooth_context):
        """blend_factor=0.5 should average current and EMA."""
        rule = TemporalSmoothRule(blend_factor=0.5)

        # First frame
        _, ctx = rule.apply(smooth_frame1, smooth_context)
        smooth_context.frame_index = 1

        # Second frame
        result, ctx = rule.apply(smooth_frame2, ctx)

        # Should be average: (100 + 200) / 2 = 150
        expected = 150
        assert result[0, 0, 0] == expected

    def test_multiple_frames_progressive_blending(self, smooth_frame1, smooth_frame2, smooth_context):
        """Multiple frames should show progressive smoothing."""
        rule = TemporalSmoothRule(blend_factor=0.7)

        # Frame 1: value 100
        result1, ctx = rule.apply(smooth_frame1, smooth_context)
        smooth_context.frame_index = 1

        # Frame 2: value 200, blended with 100
        result2, ctx = rule.apply(smooth_frame2, ctx)
        smooth_context.frame_index = 2

        # Frame 3: value 200 again
        result3, ctx = rule.apply(smooth_frame2, ctx)

        # Frame 2 should be: 0.7 * 200 + 0.3 * 100 = 170
        assert result2[0, 0, 0] == 170

        # Frame 3 should be: 0.7 * 200 + 0.3 * 170 = 140 + 51 = 191
        assert result3[0, 0, 0] == 191

    def test_shape_change_resets_ema(self, smooth_frame1, smooth_context):
        """Shape change should reset EMA (treat as new first frame)."""
        rule = TemporalSmoothRule(blend_factor=0.7)

        # First frame
        small_frame = np.full((50, 50, 3), 100, dtype=np.uint8)
        result1, ctx = rule.apply(small_frame, smooth_context)
        smooth_context.frame_index = 1

        # Second frame with different shape
        large_frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        result2, ctx = rule.apply(large_frame, ctx)

        # Should be passthrough (shape mismatch resets EMA)
        assert np.array_equal(result2, large_frame)

    def test_reset_state_clears_ema(self, smooth_frame1, smooth_context):
        """reset_state() should clear EMA state."""
        rule = TemporalSmoothRule()

        # Create state
        _, ctx = rule.apply(smooth_frame1, smooth_context)
        assert ctx.get_rule_state(rule.name()) is not None

        # Reset
        rule.reset_state(ctx)
        assert ctx.get_rule_state(rule.name()) is None

    def test_reset_state_then_passthrough(self, smooth_frame1, smooth_frame2, smooth_context):
        """After reset, next frame should be passthrough."""
        rule = TemporalSmoothRule(blend_factor=0.7)

        # First two frames
        _, ctx = rule.apply(smooth_frame1, smooth_context)
        smooth_context.frame_index = 1
        _, ctx = rule.apply(smooth_frame2, ctx)
        smooth_context.frame_index = 2

        # Reset
        rule.reset_state(ctx)

        # Next frame should be passthrough
        result, ctx = rule.apply(smooth_frame2, ctx)

        assert np.array_equal(result, smooth_frame2)

    def test_state_stored_correctly(self, smooth_frame1, smooth_context):
        """State should be stored as RGBArray in context."""
        rule = TemporalSmoothRule()

        _, ctx = rule.apply(smooth_frame1, smooth_context)

        state = ctx.get_rule_state(rule.name())
        assert isinstance(state, np.ndarray)
        assert state.shape == smooth_frame1.shape
        assert state.dtype == np.uint8


class TestCv2Blend:
    """Test cv2_blend helper function."""

    def test_blend_returns_uint8(self):
        """Should return uint8 array."""
        current = np.full((10, 10, 3), 100, dtype=np.uint8)
        ema = np.full((10, 10, 3), 200, dtype=np.uint8)

        result = cv2_blend(current, ema, 0.5)
        assert result.dtype == np.uint8

    def test_blend_half(self):
        """blend_factor=0.5 should average."""
        current = np.full((10, 10, 3), 100, dtype=np.uint8)
        ema = np.full((10, 10, 3), 200, dtype=np.uint8)

        result = cv2_blend(current, ema, 0.5)
        assert result[0, 0, 0] == 150

    def test_blend_zero(self):
        """blend_factor=0.0 should return EMA."""
        current = np.full((10, 10, 3), 100, dtype=np.uint8)
        ema = np.full((10, 10, 3), 200, dtype=np.uint8)

        result = cv2_blend(current, ema, 0.0)
        assert np.array_equal(result, ema)

    def test_blend_one(self):
        """blend_factor=1.0 should return current."""
        current = np.full((10, 10, 3), 100, dtype=np.uint8)
        ema = np.full((10, 10, 3), 200, dtype=np.uint8)

        result = cv2_blend(current, ema, 1.0)
        assert np.array_equal(result, current)

    def test_blend_clips_at_255(self):
        """Blending should clip at 255."""
        current = np.full((10, 10, 3), 255, dtype=np.uint8)
        ema = np.full((10, 10, 3), 255, dtype=np.uint8)

        result = cv2_blend(current, ema, 0.5)
        assert result[0, 0, 0] == 255

    def test_blend_clips_at_zero(self):
        """Blending should clip at 0."""
        current = np.full((10, 10, 3), 0, dtype=np.uint8)
        ema = np.full((10, 10, 3), 0, dtype=np.uint8)

        result = cv2_blend(current, ema, 0.5)
        assert result[0, 0, 0] == 0

    def test_blend_preserves_shape(self):
        """Blending should preserve input shape."""
        current = np.full((50, 30, 3), 100, dtype=np.uint8)
        ema = np.full((50, 30, 3), 200, dtype=np.uint8)

        result = cv2_blend(current, ema, 0.5)
        assert result.shape == current.shape
