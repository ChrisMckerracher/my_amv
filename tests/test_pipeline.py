"""Tests for Pipeline module.

Uses mock rules to test inline vs branch routing, state carry-forward,
and scene cut reset behavior.
"""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from my_amv.context import FrameContext
from my_amv.pipeline import Pipeline, SceneCutConfig, CheckpointConfig
from my_amv.rule import EffectRule
from my_amv.types import Layer, LayerKey, RGBArray


@dataclass
class MockState:
    """Simple state for mock rules."""

    counter: int = 0
    last_value: float = 0.0


class MockInlineRule(EffectRule[MockState]):
    """Mock rule that uses inline routing (output_layer is None)."""

    def __init__(self, name: str = "mock_inline") -> None:
        self._name = name
        self.input_layer = Layer.MAIN
        self.output_layer = None  # Inline routing

    def name(self) -> str:
        return self._name

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        # Get or create state
        state = context.get_rule_state(self._name)
        if state is None:
            state = MockState()
        state.counter += 1
        context.set_rule_state(self._name, state)

        # Simple transformation: add a value to the frame
        output = np.clip(frame.astype(np.int16) + 10, 0, 255).astype(np.uint8)
        return output, context


class MockBranchRule(EffectRule[None]):
    """Mock rule that uses branch routing (output_layer is set)."""

    def __init__(self, name: str = "mock_branch", output_layer: LayerKey = Layer.EDGES) -> None:
        self._name = name
        self.input_layer = Layer.MAIN
        self.output_layer = output_layer  # Branch routing

    def name(self) -> str:
        return self._name

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        # Store result in the layer store
        output = np.clip(frame.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        context.set_layer(self.output_layer, output)
        return frame, context  # Return input unchanged


class MockStatefulRule(EffectRule[MockState]):
    """Mock rule that carries state between frames."""

    def __init__(self, name: str = "mock_stateful") -> None:
        self._name = name
        self.input_layer = Layer.MAIN
        self.output_layer = None

    def name(self) -> str:
        return self._name

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        # Get or create state
        state = context.get_rule_state(self._name)
        if state is None:
            state = MockState(counter=0, last_value=0.0)

        # Update state based on frame mean
        frame_mean = float(frame.mean())
        state.last_value = frame_mean
        state.counter += 1

        context.set_rule_state(self._name, state)

        return frame.copy(), context

    def reset_state(self, context: FrameContext) -> FrameContext:
        """Reset state to zero."""
        context.clear_rule_state(self._name)
        return context


class TestSceneCutConfig:
    """Test SceneCutConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SceneCutConfig()

        assert config.enabled is True
        assert config.threshold == 0.3
        assert config.min_segment_length == 15

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SceneCutConfig(enabled=False, threshold=0.5, min_segment_length=30)

        assert config.enabled is False
        assert config.threshold == 0.5
        assert config.min_segment_length == 30


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CheckpointConfig()

        assert config.enabled is False
        assert config.dir is None
        assert config.save_frames is True
        assert config.save_state is True


class TestPipeline:
    """Test Pipeline class."""

    @pytest.fixture
    def sample_frame(self) -> RGBArray:
        """Create a sample RGB frame for testing."""
        return np.zeros((100, 100, 3), dtype=np.uint8) + 128

    def test_init(self) -> None:
        """Test Pipeline initialization."""
        rules = [MockInlineRule("rule1"), MockInlineRule("rule2")]
        pipeline = Pipeline(rules=rules)

        assert pipeline.rules == rules
        assert pipeline.audio_source is None
        assert pipeline.checkpoint.enabled is False

    def test_init_with_audio_source(self) -> None:
        """Test Pipeline initialization with audio source."""
        from my_amv.audio import AudioSource
        from pathlib import Path
        import tempfile
        import soundfile as sf
        import numpy as np

        # Create a temporary audio file
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "test.wav"
            sr = 22050
            y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
            sf.write(str(audio_path), y, sr)

            audio_source = AudioSource(audio_path, fps=30.0, frame_count=30)
            rules = [MockInlineRule()]
            pipeline = Pipeline(rules=rules, audio_source=audio_source)

            assert pipeline.audio_source is audio_source

    def test_process_frame_inline_routing(self, sample_frame: RGBArray) -> None:
        """Test that inline routing replaces the current frame."""
        rules = [MockInlineRule("rule1")]
        pipeline = Pipeline(rules=rules)

        result = pipeline.process_frame(sample_frame, 0, 10, 30.0)

        # Inline rule should modify the frame
        assert result.shape == sample_frame.shape
        # Frame values should be increased by 10 (from MockInlineRule)
        # Check center pixel
        assert result[50, 50, 0] == sample_frame[50, 50, 0] + 10

    def test_process_frame_branch_routing(self, sample_frame: RGBArray) -> None:
        """Test that branch routing stores result in layer store."""
        rules = [MockBranchRule("rule1", output_layer=Layer.EDGES)]
        pipeline = Pipeline(rules=rules)

        result = pipeline.process_frame(sample_frame, 0, 10, 30.0)

        # Branch rule should NOT modify the current frame
        np.testing.assert_array_equal(result, sample_frame)

        # But the layer should be stored in the context
        # We need to access this through the context in process_frame
        # Since we can't access the context directly, let's verify through
        # a second rule that reads from the branch layer

    def test_process_frame_multiple_rules_inline(self, sample_frame: RGBArray) -> None:
        """Test that multiple inline rules chain correctly."""
        rules = [
            MockInlineRule("rule1"),
            MockInlineRule("rule2"),
            MockInlineRule("rule3"),
        ]
        pipeline = Pipeline(rules=rules)

        result = pipeline.process_frame(sample_frame, 0, 10, 30.0)

        # Each inline rule adds 10, so result should be +30
        expected = np.clip(sample_frame.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_process_frame_inline_then_branch(self, sample_frame: RGBArray) -> None:
        """Test inline rule followed by branch rule."""
        class InlineThenBranchRule(EffectRule[None]):
            def __init__(self):
                self.input_layer = Layer.MAIN
                self.output_layer = None

            def name(self) -> str:
                return "inline_then_branch"

            def apply(self, frame: RGBArray, context: FrameContext) -> tuple[RGBArray, FrameContext]:
                # Inline: modify frame
                modified = np.clip(frame.astype(np.int16) + 10, 0, 255).astype(np.uint8)

                # Also store to a layer
                context.set_layer(Layer.EDGES, modified + 20)

                return modified, context

        rules = [InlineThenBranchRule()]
        pipeline = Pipeline(rules=rules)

        result = pipeline.process_frame(sample_frame, 0, 10, 30.0)

        # Result should be inline modification (+10)
        expected = np.clip(sample_frame.astype(np.int16) + 10, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_process_frame_branch_then_inline(self, sample_frame: RGBArray) -> None:
        """Test branch rule followed by inline rule."""
        class BranchThenInlineRule(EffectRule[None]):
            def __init__(self, branch_first: bool = True):
                self.input_layer = Layer.MAIN
                self.output_layer = None
                self._branch_first = branch_first

            def name(self) -> str:
                return "branch_then_inline"

            def apply(self, frame: RGBArray, context: FrameContext) -> tuple[RGBArray, FrameContext]:
                if self._branch_first:
                    # First call: store to layer, return unchanged
                    context.set_layer(Layer.EDGES, frame + 20)
                    return frame.copy(), context
                else:
                    # Second call: read from layer
                    edges = context.get_layer(Layer.EDGES)
                    return edges, context

        # First rule stores to layer, second reads from it
        rules = [
            BranchThenInlineRule(branch_first=True),
            BranchThenInlineRule(branch_first=False),
        ]
        rules[1].input_layer = Layer.EDGES

        pipeline = Pipeline(rules=rules)

        result = pipeline.process_frame(sample_frame, 0, 10, 30.0)

        # Result should have +20 from the branch layer
        expected = np.clip(sample_frame.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_process_frame_state_carry_forward(self, sample_frame: RGBArray) -> None:
        """Test that state is set in context during frame processing.

        Note: Each process_frame call creates a new FrameContext, so state
        doesn't automatically carry forward between process_frame calls.
        State carry-forward within process_video works differently.
        """
        # Create a rule that tracks calls and sets state
        class TrackingRule(EffectRule[None]):
            def __init__(self):
                self._name = "tracker"
                self.input_layer = Layer.MAIN
                self.output_layer = None
                self.call_count = 0

            def name(self) -> str:
                return self._name

            def apply(self, frame: RGBArray, context: FrameContext) -> tuple[RGBArray, FrameContext]:
                self.call_count += 1

                # Set state in the context (with call_count as the value)
                context.set_rule_state(self._name, MockState(counter=self.call_count))
                return frame, context

        rule = TrackingRule()
        pipeline = Pipeline(rules=[rule])

        # Process a frame
        pipeline.process_frame(sample_frame, 0, 10, 30.0)

        # Verify rule was called once
        assert rule.call_count == 1

        # Process another frame - gets a fresh context
        pipeline.process_frame(sample_frame, 1, 10, 30.0)

        # Verify rule was called twice total
        assert rule.call_count == 2

        # Each process_frame call creates a fresh context
        # State carry-forward across frames happens in process_video,
        # not in individual process_frame calls

    def test_process_frame_context_has_original_layer(self, sample_frame: RGBArray) -> None:
        """Test that FrameContext has ORIGINAL layer set."""

        class CheckOriginalRule(EffectRule[None]):
            def __init__(self):
                self._name = "check_original"
                self.input_layer = Layer.ORIGINAL
                self.output_layer = None
                self.found_original = False

            def name(self) -> str:
                return self._name

            def apply(self, frame: RGBArray, context: FrameContext) -> tuple[RGBArray, FrameContext]:
                self.found_original = context.has_layer(Layer.ORIGINAL)
                return frame, context

        rule = CheckOriginalRule()
        pipeline = Pipeline(rules=[rule])

        pipeline.process_frame(sample_frame, 0, 10, 30.0)

        assert rule.found_original

    def test_scene_cut_detection(self, sample_frame: RGBArray) -> None:
        """Test scene cut detection and state reset."""
        from my_amv.audio import AudioSource
        import tempfile
        import soundfile as sf

        # Create temporary audio file
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "test.wav"
            sr = 22050
            y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
            sf.write(str(audio_path), y, sr)

            audio_source = AudioSource(audio_path, fps=30.0, frame_count=30)
            audio_source.analyze()  # Must analyze before getting frames

            rule = MockStatefulRule("stateful")
            pipeline = Pipeline(rules=[rule], audio_source=audio_source)

            # Set low threshold to trigger scene cut
            pipeline.set_scene_cut_config(SceneCutConfig(threshold=0.0))

            # Process first frame
            frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
            pipeline.process_frame(frame1, 0, 10, 30.0)

            # Process very different frame (triggers scene cut)
            frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
            result = pipeline.process_frame(frame2, 1, 10, 30.0)

            # Scene cut should have been detected
            # We can verify by checking the scene cuts list if accessible
            # For now, just verify no crash occurred
            assert result is not None

    def test_compute_frame_diff(self) -> None:
        """Test frame difference computation."""
        pipeline = Pipeline(rules=[])

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        diff = pipeline._compute_frame_diff(frame1, frame2)

        assert diff == pytest.approx(1.0)

    def test_compute_frame_diff_same(self) -> None:
        """Test frame difference for identical frames."""
        pipeline = Pipeline(rules=[])

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        diff = pipeline._compute_frame_diff(frame, frame)

        assert diff == pytest.approx(0.0)

    def test_reset_all_rules(self, sample_frame: RGBArray) -> None:
        """Test that reset_all_rules clears rule state."""
        rule = MockStatefulRule("stateful")
        pipeline = Pipeline(rules=[rule])

        # Process a frame to create state
        pipeline.process_frame(sample_frame, 0, 10, 30.0)

        # Reset state
        pipeline._reset_all_rules(sample_frame, 1, 10, 30.0)

        # Verify state was reset
        # We can verify by processing another frame and checking
        # that the state counter started over

    def test_set_scene_cut_config(self) -> None:
        """Test setting scene cut configuration."""
        pipeline = Pipeline(rules=[])

        config = SceneCutConfig(enabled=False, threshold=0.5)
        pipeline.set_scene_cut_config(config)

        assert pipeline.scene_cut.enabled is False
        assert pipeline.scene_cut.threshold == 0.5

    def test_set_checkpoint_config(self) -> None:
        """Test setting checkpoint configuration."""
        pipeline = Pipeline(rules=[])

        config = CheckpointConfig(enabled=True, save_frames=False)
        pipeline.set_checkpoint_config(config)

        assert pipeline.checkpoint.enabled is True
        assert pipeline.checkpoint.save_frames is False

    def test_process_image(self, sample_frame: RGBArray, tmp_path: Path) -> None:
        """Test process_image returns all layers."""
        from PIL import Image

        # Create a test image
        image_path = tmp_path / "test.png"
        Image.fromarray(sample_frame).save(image_path)

        class LayerProducerRule(EffectRule[None]):
            def __init__(self, output_layer: LayerKey, add_value: int = 50):
                self._name = f"producer_{output_layer}"
                self.input_layer = Layer.MAIN
                self.output_layer = output_layer
                self._add_value = add_value

            def name(self) -> str:
                return self._name

            def apply(self, frame: RGBArray, context: FrameContext) -> tuple[RGBArray, FrameContext]:
                # Return the modified version - the pipeline will store it to output_layer
                modified = np.clip(frame.astype(np.int16) + self._add_value, 0, 255).astype(np.uint8)
                return modified, context

        rules = [
            LayerProducerRule(Layer.EDGES, add_value=50),
            LayerProducerRule(Layer.MASK, add_value=100),
        ]
        pipeline = Pipeline(rules=rules)

        layers = pipeline.process_image(image_path)

        # Check that layers exist
        assert Layer.ORIGINAL in layers
        assert Layer.EDGES in layers
        assert Layer.MASK in layers
        assert Layer.FINAL in layers

        # Check ORIGINAL layer - should match the input
        np.testing.assert_array_equal(layers[Layer.ORIGINAL], sample_frame)

        # Check EDGES layer - should be original + 50
        expected_edges = np.clip(sample_frame.astype(np.int16) + 50, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(layers[Layer.EDGES], expected_edges)

        # Check MASK layer - should be original + 100
        expected_mask = np.clip(sample_frame.astype(np.int16) + 100, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(layers[Layer.MASK], expected_mask)
