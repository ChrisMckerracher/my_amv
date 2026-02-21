"""Tests for FrameContext and AudioFrame."""

import numpy as np
import pytest

from my_amv.context import AudioFrame, FrameContext
from my_amv.types import Layer, RGBArray


class TestAudioFrame:
    """Test AudioFrame frozen dataclass."""

    def test_audio_frame_creation(self):
        """AudioFrame should be creatable with all fields."""
        audio = AudioFrame(
            frame_index=0,
            time_seconds=0.0,
            beat_strength=0.5,
            onset_strength=0.3,
            rms_energy=0.8,
            spectral_centroid=2000.0,
            frequency_bands={"bass": 0.7, "mid": 0.5, "treble": 0.3},
        )
        assert audio.frame_index == 0
        assert audio.time_seconds == 0.0
        assert audio.beat_strength == 0.5
        assert audio.onset_strength == 0.3
        assert audio.rms_energy == 0.8
        assert audio.spectral_centroid == 2000.0
        assert audio.frequency_bands == {"bass": 0.7, "mid": 0.5, "treble": 0.3}

    def test_audio_frame_frozen(self):
        """AudioFrame should be frozen (immutable)."""
        audio = AudioFrame(
            frame_index=0,
            time_seconds=0.0,
            beat_strength=0.5,
            onset_strength=0.3,
            rms_energy=0.8,
            spectral_centroid=2000.0,
            frequency_bands={"bass": 0.7, "mid": 0.5, "treble": 0.3},
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            audio.beat_strength = 0.9

    def test_audio_frame_frequency_bands(self):
        """AudioFrame frequency_bands should have all expected bands."""
        audio = AudioFrame(
            frame_index=0,
            time_seconds=0.0,
            beat_strength=0.5,
            onset_strength=0.3,
            rms_energy=0.8,
            spectral_centroid=2000.0,
            frequency_bands={"bass": 0.7, "mid": 0.5, "treble": 0.3},
        )
        assert "bass" in audio.frequency_bands
        assert "mid" in audio.frequency_bands
        assert "treble" in audio.frequency_bands


class TestFrameContextLayerStore:
    """Test FrameContext layer storage methods."""

    def test_frame_context_creation(self):
        """FrameContext should be creatable with required fields."""
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        assert ctx.frame_index == 0
        assert ctx.frame_count == 100
        assert ctx.fps == 30.0
        assert ctx.audio_frame is None
        assert ctx.list_layers() == []

    def test_set_and_get_layer(self):
        """set_layer and get_layer should store and retrieve layers."""
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        arr: RGBArray = np.zeros((100, 100, 3), dtype=np.uint8)

        ctx.set_layer(Layer.EDGES, arr)
        retrieved = ctx.get_layer(Layer.EDGES)

        assert np.array_equal(retrieved, arr)
        assert retrieved.shape == (100, 100, 3)

    def test_set_layer_with_string_key(self):
        """set_layer should accept string keys for custom layers."""
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        arr: RGBArray = np.zeros((100, 100, 3), dtype=np.uint8)

        ctx.set_layer("halftone", arr)
        retrieved = ctx.get_layer("halftone")

        assert np.array_equal(retrieved, arr)

    def test_has_layer(self):
        """has_layer should report layer existence correctly."""
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        arr: RGBArray = np.zeros((100, 100, 3), dtype=np.uint8)

        assert not ctx.has_layer(Layer.EDGES)
        ctx.set_layer(Layer.EDGES, arr)
        assert ctx.has_layer(Layer.EDGES)

    def test_list_layers(self):
        """list_layers should return all stored layer names."""
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        arr: RGBArray = np.zeros((100, 100, 3), dtype=np.uint8)

        assert ctx.list_layers() == []

        ctx.set_layer(Layer.EDGES, arr)
        ctx.set_layer(Layer.DEPTH, arr)
        ctx.set_layer("custom", arr)

        layers = ctx.list_layers()
        assert len(layers) == 3
        assert Layer.EDGES in layers
        assert Layer.DEPTH in layers
        assert "custom" in layers

    def test_get_layer_raises_key_error(self):
        """get_layer should raise KeyError for missing layers."""
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)

        with pytest.raises(KeyError):
            ctx.get_layer(Layer.EDGES)


class TestFrameContextRuleState:
    """Test FrameContext rule state isolation."""

    def test_get_and_set_rule_state(self):
        """set_rule_state and get_rule_state should store rule state."""
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)

        assert ctx.get_rule_state("TestRule") is None

        ctx.set_rule_state("TestRule", {"count": 5})
        state = ctx.get_rule_state("TestRule")

        assert state == {"count": 5}

    def test_rule_state_isolation(self):
        """Different rules should have isolated state."""
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)

        ctx.set_rule_state("RuleA", {"value": 1})
        ctx.set_rule_state("RuleB", {"value": 2})

        assert ctx.get_rule_state("RuleA") == {"value": 1}
        assert ctx.get_rule_state("RuleB") == {"value": 2}

    def test_clear_rule_state(self):
        """clear_rule_state should remove a rule's state."""
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)

        ctx.set_rule_state("TestRule", {"count": 5})
        assert ctx.get_rule_state("TestRule") is not None

        ctx.clear_rule_state("TestRule")
        assert ctx.get_rule_state("TestRule") is None

    def test_clear_nonexistent_rule_state(self):
        """clear_rule_state should be safe for non-existent rules."""
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)

        # Should not raise
        ctx.clear_rule_state("NonExistentRule")
        assert ctx.get_rule_state("NonExistentRule") is None


class TestFrameContextWithAudio:
    """Test FrameContext with audio data."""

    def test_frame_context_with_audio(self):
        """FrameContext should accept AudioFrame."""
        audio = AudioFrame(
            frame_index=0,
            time_seconds=0.0,
            beat_strength=0.5,
            onset_strength=0.3,
            rms_energy=0.8,
            spectral_centroid=2000.0,
            frequency_bands={"bass": 0.7, "mid": 0.5, "treble": 0.3},
        )

        ctx = FrameContext(
            frame_index=0, frame_count=100, fps=30.0, audio_frame=audio
        )

        assert ctx.audio_frame is not None
        assert ctx.audio_frame.frame_index == 0
        assert ctx.audio_frame.beat_strength == 0.5
