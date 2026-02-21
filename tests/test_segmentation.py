"""Tests for PersonSegmentationRule."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from my_amv.context import FrameContext
from my_amv.rules.segmentation import PersonSegmentationRule
from my_amv.types import Layer, MaskArray, RGBArray


class TestPersonSegmentationRule:
    """Test PersonSegmentationRule segmentation functionality."""

    @pytest.fixture
    def sample_frame(self) -> RGBArray:
        """Create a sample RGB frame for testing."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a "person" in the middle (white square)
        frame[30:70, 30:70] = 200
        return frame

    @pytest.fixture
    def context(self) -> FrameContext:
        """Create a test FrameContext."""
        return FrameContext(frame_index=0, frame_count=100, fps=30.0)

    @pytest.fixture
    def mock_mediapipe_result(self):
        """Create a mock MediaPipe segmentation result."""
        mock_result = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.float32)
        mock_mask[30:70, 30:70] = 0.8  # Person region
        mock_result.segmentation_mask = mock_mask
        return mock_result

    def test_rule_creation(self):
        """PersonSegmentationRule should be creatable with default parameters."""
        rule = PersonSegmentationRule()

        assert rule.name() == "SilhouetteExtract"
        assert rule.model == "mediapipe"
        assert rule.apply_mask_to_frame is False
        assert rule.roi is None
        assert rule.output_layer == Layer.MASK
        assert rule.input_layer == Layer.MAIN

    def test_rule_creation_with_params(self):
        """PersonSegmentationRule should accept model and apply_mask_to_frame."""
        rule = PersonSegmentationRule(
            model="sam2",
            apply_mask_to_frame=True,
            roi=[10, 20, 50, 80]
        )

        assert rule.model == "sam2"
        assert rule.apply_mask_to_frame is True
        assert rule.roi == [10, 20, 50, 80]

    def test_invalid_model_raises(self):
        """Invalid model should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            rule = PersonSegmentationRule(model="invalid")
            rule._segment(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_apply_returns_tuple(self, sample_frame, context):
        """apply() should return (frame, context) tuple."""
        with patch.object(PersonSegmentationRule, "_segment") as mock_seg:
            mock_mask = np.zeros((100, 100), dtype=np.uint8)
            mock_seg.return_value = mock_mask

            rule = PersonSegmentationRule()
            result_frame, result_ctx = rule.apply(sample_frame, context)

            assert isinstance(result_frame, np.ndarray)
            assert isinstance(result_ctx, FrameContext)

    def test_apply_stores_mask_in_context(self, sample_frame, context):
        """apply() should store mask result in context layer."""
        with patch.object(PersonSegmentationRule, "_segment") as mock_seg:
            # Return a simple mask
            mock_mask = np.zeros((100, 100), dtype=np.uint8)
            mock_mask[30:70, 30:70] = 255
            mock_seg.return_value = mock_mask

            rule = PersonSegmentationRule()
            rule.apply(sample_frame, context)

            assert context.has_layer(Layer.MASK)
            mask = context.get_layer(Layer.MASK)
            assert mask.shape == (100, 100, 3)

    def test_mask_shape_matches_input(self, sample_frame, context):
        """Mask output should have same spatial dimensions as input."""
        with patch.object(PersonSegmentationRule, "_segment") as mock_seg:
            mock_mask = np.zeros((100, 100), dtype=np.uint8)
            mock_seg.return_value = mock_mask

            rule = PersonSegmentationRule()
            rule.apply(sample_frame, context)
            mask = context.get_layer(Layer.MASK)

            assert mask.shape[0:2] == sample_frame.shape[0:2]

    def test_apply_mask_to_frame_false_leaves_frame_unchanged(self, sample_frame, context):
        """When apply_mask_to_frame=False, frame should be unchanged."""
        with patch.object(PersonSegmentationRule, "_segment") as mock_seg:
            mock_mask = np.zeros((100, 100), dtype=np.uint8)
            mock_mask[30:70, 30:70] = 255
            mock_seg.return_value = mock_mask

            rule = PersonSegmentationRule(apply_mask_to_frame=False)
            result_frame, _ = rule.apply(sample_frame, context)

            # Frame should be unchanged
            assert np.array_equal(result_frame, sample_frame)

    def test_apply_mask_to_frame_true_zeros_background(self, sample_frame, context):
        """When apply_mask_to_frame=True, background should be zeroed."""
        with patch.object(PersonSegmentationRule, "_segment") as mock_seg:
            # Mask person in center
            mock_mask = np.zeros((100, 100), dtype=np.uint8)
            mock_mask[30:70, 30:70] = 255
            mock_seg.return_value = mock_mask

            rule = PersonSegmentationRule(apply_mask_to_frame=True)
            result_frame, _ = rule.apply(sample_frame, context)

            # Background should be zeroed (outside mask region)
            assert np.all(result_frame[0:30, :] == 0)
            assert np.all(result_frame[70:, :] == 0)
            # Person region should have original values
            assert np.all(result_frame[30:70, 30:70] == 200)

    def test_mediapipe_without_package_raises(self, sample_frame):
        """MediaPipe model should raise clear error without package."""
        rule = PersonSegmentationRule(model="mediapipe")

        with patch.dict("sys.modules", {"mediapipe": None}):
            with pytest.raises(RuntimeError, match="Install mediapipe"):
                rule._get_mediapipe_model()

    def test_sam2_without_package_raises(self, sample_frame):
        """SAM2 model should raise clear error without package."""
        rule = PersonSegmentationRule(model="sam2")

        with patch.dict("sys.modules", {"sam2": None}):
            with pytest.raises(RuntimeError, match="Install SAM2"):
                rule._get_sam2_model()

    def test_mediapipe_segmentation_mock(self, sample_frame, mock_mediapipe_result):
        """MediaPipe segmentation should work with mocked model."""
        # Mock the _get_mediapipe_model to return a mock model
        mock_model = MagicMock()
        mock_model.process.return_value = mock_mediapipe_result

        rule = PersonSegmentationRule(model="mediapipe")
        with patch.object(rule, "_get_mediapipe_model", return_value=mock_model):
            mask = rule._apply_mediapipe(sample_frame)

            # Mask should be binary (0 or 255)
            assert mask.dtype == np.uint8
            unique_vals = np.unique(mask)
            assert 0 in unique_vals or 255 in unique_vals

    def test_sam2_segmentation_mock(self, sample_frame):
        """SAM2 segmentation should work with mocked model."""
        mock_mask = np.zeros((100, 100), dtype=np.float32)
        mock_mask[30:70, 30:70] = 0.8

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = (
            np.array([mock_mask]),  # masks
            np.array([0.9]),         # scores
            np.array([[]])           # logits
        )

        rule = PersonSegmentationRule(model="sam2")
        with patch.object(rule, "_get_sam2_model", return_value=mock_predictor):
            mask = rule._apply_sam2(sample_frame)

            # Mask should be binary
            assert mask.dtype == np.uint8

    def test_sam2_with_roi_hint(self, sample_frame):
        """SAM2 should use ROI hint when provided."""
        mock_mask = np.zeros((100, 100), dtype=np.float32)
        mock_mask[30:70, 30:70] = 0.8

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = (
            np.array([mock_mask]),
            np.array([0.9]),
            np.array([[]])
        )

        rule = PersonSegmentationRule(model="sam2", roi=[10, 20, 50, 80])
        with patch.object(rule, "_get_sam2_model", return_value=mock_predictor):
            rule._apply_sam2(sample_frame)

            # Should be called with box parameter
            mock_predictor.predict.assert_called_once()
            call_kwargs = mock_predictor.predict.call_args[1]
            assert "box" in call_kwargs
            np.testing.assert_array_equal(call_kwargs["box"], np.array([10, 20, 50, 80]))

    def test_mediapipe_no_person_detected_returns_empty_mask(self, sample_frame):
        """MediaPipe should return empty mask when no person detected."""
        # Mock result with no segmentation mask
        mock_result = MagicMock()
        mock_result.segmentation_mask = None

        with patch.dict("sys.modules", {"mediapipe": MagicMock()}):
            mock_mp = MagicMock()
            mock_model = MagicMock()
            mock_model.process.return_value = mock_result
            mock_mp.solutions.selfie_segmentation.SelfieSegmentation.return_value = mock_model

            with patch.dict("sys.modules", {"mediapipe": mock_mp}):
                rule = PersonSegmentationRule(model="mediapipe")
                rule._mediapipe_model = None

                mask = rule._apply_mediapipe(sample_frame)

                # Should be all zeros
                assert np.all(mask == 0)

    def test_configure_updates_model(self):
        """configure() should update model."""
        rule = PersonSegmentationRule(model="mediapipe")

        rule.configure({"model": "sam2"})

        assert rule.model == "sam2"

    def test_configure_invalid_model_raises(self):
        """configure() with invalid model should raise ValueError."""
        rule = PersonSegmentationRule()

        with pytest.raises(ValueError, match="Invalid model"):
            rule.configure({"model": "not_real"})

    def test_configure_updates_apply_mask_to_frame(self):
        """configure() should update apply_mask_to_frame."""
        rule = PersonSegmentationRule(apply_mask_to_frame=False)

        rule.configure({"apply_mask_to_frame": True})

        assert rule.apply_mask_to_frame is True

    def test_configure_updates_roi(self):
        """configure() should update roi."""
        rule = PersonSegmentationRule()

        rule.configure({"roi": [10, 20, 50, 80]})

        assert rule.roi == [10, 20, 50, 80]

    def test_configure_sets_roi_to_none(self):
        """configure() should allow setting roi to None."""
        rule = PersonSegmentationRule(roi=[10, 20, 50, 80])

        rule.configure({"roi": None})

        assert rule.roi is None

    def test_configure_invalid_roi_raises(self):
        """configure() with invalid roi should raise ValueError."""
        rule = PersonSegmentationRule()

        with pytest.raises(ValueError, match="roi must be a list of 4 ints"):
            rule.configure({"roi": [1, 2, 3]})  # Only 3 elements

    def test_custom_output_layer(self, sample_frame, context):
        """Custom output_layer should store mask in that layer."""
        with patch.object(PersonSegmentationRule, "_segment") as mock_seg:
            mock_mask = np.zeros((100, 100), dtype=np.uint8)
            mock_seg.return_value = mock_mask

            rule = PersonSegmentationRule(output_layer="custom_mask")
            rule.apply(sample_frame, context)

            assert context.has_layer("custom_mask")
            mask = context.get_layer("custom_mask")
            assert mask.shape == (100, 100, 3)

    def test_stateless_rule(self, sample_frame, context):
        """PersonSegmentationRule should be stateless."""
        with patch.object(PersonSegmentationRule, "_segment") as mock_seg:
            mock_mask = np.zeros((100, 100), dtype=np.uint8)
            mock_seg.return_value = mock_mask

            rule = PersonSegmentationRule()

            # Should not raise any errors
            state = rule.serialize_state(context)
            assert state == {}

            # Should not raise any errors
            rule.deserialize_state(context, {})

    def test_segment_binary_mask_values(self, sample_frame, mock_mediapipe_result):
        """Segmentation mask should contain only 0 or 255 values."""
        # Mock result with continuous probability values
        mock_result = MagicMock()
        mock_mask = np.random.rand(100, 100).astype(np.float32) * 0.8 + 0.1
        mock_result.segmentation_mask = mock_mask

        with patch.dict("sys.modules", {"mediapipe": MagicMock()}):
            mock_mp = MagicMock()
            mock_model = MagicMock()
            mock_model.process.return_value = mock_result
            mock_mp.solutions.selfie_segmentation.SelfieSegmentation.return_value = mock_model

            with patch.dict("sys.modules", {"mediapipe": mock_mp}):
                rule = PersonSegmentationRule(model="mediapipe")
                rule._mediapipe_model = None

                mask = rule._apply_mediapipe(sample_frame)

                # After thresholding, should only be 0 or 255
                unique_vals = np.unique(mask)
                assert len(unique_vals) <= 2
                assert all(v in (0, 255) for v in unique_vals)

    def test_apply_mask_to_frame_works_correctly(self):
        """_apply_mask_to_frame should correctly zero background."""
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255  # Person in center

        rule = PersonSegmentationRule()
        result = rule._apply_mask_to_frame(frame, mask)

        # Background should be zeroed
        assert np.all(result[0:30, :] == 0)
        assert np.all(result[70:, :] == 0)
        assert np.all(result[:, 0:30] == 0)
        assert np.all(result[:, 70:] == 0)

        # Person region should be unchanged
        assert np.all(result[30:70, 30:70] == 128)
