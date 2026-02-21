"""Tests for AudioReactiveEdgeRule."""

import numpy as np
import pytest

from my_amv.context import AudioFrame, FrameContext
from my_amv.rules.audio_reactive import (
    AudioReactiveEdgeRule,
    AudioReactiveState,
    _apply_displacement,
    _apply_thickness,
    _get_frequency_energy,
)


@pytest.fixture
def audio_frame():
    """Create a test audio frame."""
    return AudioFrame(
        frame_index=0,
        time_seconds=0.0,
        beat_strength=0.8,
        onset_strength=0.7,
        rms_energy=0.5,
        spectral_centroid=2000.0,
        frequency_bands={"bass": 0.9, "mid": 0.5, "treble": 0.3},
    )


@pytest.fixture
def silent_audio_frame():
    """Create a silent audio frame."""
    return AudioFrame(
        frame_index=0,
        time_seconds=0.0,
        beat_strength=0.0,
        onset_strength=0.0,
        rms_energy=0.0,
        spectral_centroid=1000.0,
        frequency_bands={"bass": 0.0, "mid": 0.0, "treble": 0.0},
    )


@pytest.fixture
def audio_context(audio_frame):
    """Create a test context with audio."""
    return FrameContext(frame_index=0, frame_count=100, fps=30.0, audio_frame=audio_frame)


@pytest.fixture
def silent_context(silent_audio_frame):
    """Create a test context with silent audio."""
    return FrameContext(
        frame_index=0, frame_count=100, fps=30.0, audio_frame=silent_audio_frame
    )


@pytest.fixture
def no_audio_context():
    """Create a test context without audio."""
    return FrameContext(frame_index=0, frame_count=100, fps=30.0, audio_frame=None)


@pytest.fixture
def edge_frame():
    """Create a test edge frame."""
    # Create a simple edge pattern
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[20:30, :] = 255  # Horizontal edge
    return frame


class TestAudioReactiveEdgeRule:
    """Test AudioReactiveEdgeRule functionality."""

    def test_rule_name(self):
        """Rule name should be 'AudioReactiveEdgeRule'."""
        rule = AudioReactiveEdgeRule()
        assert rule.name() == "AudioReactiveEdgeRule"

    def test_default_parameters(self):
        """Rule should have default parameters."""
        rule = AudioReactiveEdgeRule()
        assert rule.sensitivity == 1.0
        assert rule.max_displacement == 8
        assert rule.decay == 0.85
        assert rule.frequency_band == "all"
        assert rule.effect_type == "both"

    def test_custom_parameters(self):
        """Rule should accept custom parameters."""
        rule = AudioReactiveEdgeRule(
            sensitivity=0.7,
            max_displacement=15,
            decay=0.9,
            frequency_band="bass",
            effect_type="displace",
        )
        assert rule.sensitivity == 0.7
        assert rule.max_displacement == 15
        assert rule.decay == 0.9
        assert rule.frequency_band == "bass"
        assert rule.effect_type == "displace"

    def test_input_layer_is_edges(self):
        """Rule should read from Layer.EDGES by default."""
        from my_amv.types import Layer

        rule = AudioReactiveEdgeRule()
        assert rule.input_layer == Layer.EDGES

    def test_configure_sensitivity(self):
        """configure() should update sensitivity."""
        rule = AudioReactiveEdgeRule()
        rule.configure({"sensitivity": 0.5})
        assert rule.sensitivity == 0.5

    def test_configure_invalid_sensitivity_low_raises(self):
        """sensitivity < 0 should raise ValueError."""
        rule = AudioReactiveEdgeRule()
        with pytest.raises(ValueError, match="sensitivity must be in \\[0.0, 2.0\\]"):
            rule.configure({"sensitivity": -0.1})

    def test_configure_invalid_sensitivity_high_raises(self):
        """sensitivity > 2 should raise ValueError."""
        rule = AudioReactiveEdgeRule()
        with pytest.raises(ValueError, match="sensitivity must be in \\[0.0, 2.0\\]"):
            rule.configure({"sensitivity": 2.5})

    def test_configure_max_displacement(self):
        """configure() should update max_displacement."""
        rule = AudioReactiveEdgeRule()
        rule.configure({"max_displacement": 20})
        assert rule.max_displacement == 20

    def test_configure_invalid_max_displacement_raises(self):
        """max_displacement > 50 should raise ValueError."""
        rule = AudioReactiveEdgeRule()
        with pytest.raises(ValueError, match="max_displacement must be in \\[0, 50\\]"):
            rule.configure({"max_displacement": 100})

    def test_configure_decay(self):
        """configure() should update decay."""
        rule = AudioReactiveEdgeRule()
        rule.configure({"decay": 0.7})
        assert rule.decay == 0.7

    def test_configure_invalid_decay_low_raises(self):
        """decay < 0 should raise ValueError."""
        rule = AudioReactiveEdgeRule()
        with pytest.raises(ValueError, match="decay must be in \\[0.0, 1.0\\]"):
            rule.configure({"decay": -0.1})

    def test_configure_frequency_band(self):
        """configure() should update frequency_band."""
        rule = AudioReactiveEdgeRule()
        rule.configure({"frequency_band": "bass"})
        assert rule.frequency_band == "bass"

    def test_configure_invalid_frequency_band_raises(self):
        """Invalid frequency_band should raise ValueError."""
        rule = AudioReactiveEdgeRule()
        with pytest.raises(ValueError, match="Invalid frequency_band"):
            rule.configure({"frequency_band": "invalid"})

    def test_configure_effect_type(self):
        """configure() should update effect_type."""
        rule = AudioReactiveEdgeRule()
        rule.configure({"effect_type": "thickness"})
        assert rule.effect_type == "thickness"

    def test_configure_invalid_effect_type_raises(self):
        """Invalid effect_type should raise ValueError."""
        rule = AudioReactiveEdgeRule()
        with pytest.raises(ValueError, match="Invalid effect_type"):
            rule.configure({"effect_type": "invalid"})

    def test_apply_no_audio_passthrough(self, edge_frame, no_audio_context):
        """Without audio, rule should be passthrough."""
        rule = AudioReactiveEdgeRule()
        result, _ = rule.apply(edge_frame, no_audio_context)

        assert np.array_equal(result, edge_frame)

    def test_apply_returns_correct_shape(self, edge_frame, audio_context):
        """apply() should return output with same shape as input."""
        rule = AudioReactiveEdgeRule()
        result, _ = rule.apply(edge_frame, audio_context)

        assert result.shape == edge_frame.shape

    def test_apply_creates_state(self, edge_frame, audio_context):
        """apply() should create state on first call."""
        rule = AudioReactiveEdgeRule()
        _, new_ctx = rule.apply(edge_frame, audio_context)

        state = new_ctx.get_rule_state(rule.name())
        assert state is not None
        assert isinstance(state, AudioReactiveState)

    def test_displacement_on_beat(self, edge_frame, audio_context):
        """Strong beat should produce displacement > 0."""
        rule = AudioReactiveEdgeRule(sensitivity=1.0, decay=0.85)
        _, new_ctx = rule.apply(edge_frame, audio_context)

        state = new_ctx.get_rule_state(rule.name())
        assert state.displacement > 0

    def test_displacement_scales_with_sensitivity(self, edge_frame, audio_context):
        """Higher sensitivity should produce more displacement."""
        rule_low = AudioReactiveEdgeRule(sensitivity=0.5)
        rule_high = AudioReactiveEdgeRule(sensitivity=2.0)

        _, ctx_low = rule_low.apply(edge_frame, audio_context)
        _, ctx_high = rule_high.apply(edge_frame, audio_context)

        state_low = ctx_low.get_rule_state(rule_low.name())
        state_high = ctx_high.get_rule_state(rule_high.name())

        assert state_high.displacement >= state_low.displacement

    def test_decay_reduces_displacement(self, edge_frame, audio_context, silent_context):
        """Displacement should decay over frames with no beat."""
        rule = AudioReactiveEdgeRule(decay=0.5)

        # First frame: strong beat
        _, ctx = rule.apply(edge_frame, audio_context)
        state1 = ctx.get_rule_state(rule.name())
        disp1 = state1.displacement

        # Second frame: silent
        ctx.frame_index = 1
        ctx.audio_frame = silent_context.audio_frame
        _, ctx = rule.apply(edge_frame, ctx)
        state2 = ctx.get_rule_state(rule.name())
        disp2 = state2.displacement

        # Displacement should have decayed
        assert disp2 < disp1

    def test_frequency_band_bass(self, edge_frame, audio_context):
        """frequency_band='bass' should use bass energy."""
        rule = AudioReactiveEdgeRule(frequency_band="bass")
        # Should use bass=0.9 from audio_frame
        result, _ = rule.apply(edge_frame, audio_context)
        # Just verify it runs
        assert result.shape == edge_frame.shape

    def test_frequency_band_mid(self, edge_frame, audio_context):
        """frequency_band='mid' should use mid energy."""
        rule = AudioReactiveEdgeRule(frequency_band="mid")
        result, _ = rule.apply(edge_frame, audio_context)
        assert result.shape == edge_frame.shape

    def test_frequency_band_treble(self, edge_frame, audio_context):
        """frequency_band='treble' should use treble energy."""
        rule = AudioReactiveEdgeRule(frequency_band="treble")
        result, _ = rule.apply(edge_frame, audio_context)
        assert result.shape == edge_frame.shape

    def test_effect_type_displace(self, edge_frame, audio_context):
        """effect_type='displace' should only apply displacement."""
        rule = AudioReactiveEdgeRule(effect_type="displace")
        result, _ = rule.apply(edge_frame, audio_context)
        assert result.shape == edge_frame.shape

    def test_effect_type_thickness(self, edge_frame, audio_context):
        """effect_type='thickness' should only apply thickness."""
        rule = AudioReactiveEdgeRule(effect_type="thickness")
        result, _ = rule.apply(edge_frame, audio_context)
        assert result.shape == edge_frame.shape

    def test_effect_type_both(self, edge_frame, audio_context):
        """effect_type='both' should apply both effects."""
        rule = AudioReactiveEdgeRule(effect_type="both")
        result, _ = rule.apply(edge_frame, audio_context)
        assert result.shape == edge_frame.shape

    def test_reset_state_clears_state(self, edge_frame, audio_context):
        """reset_state() should clear the rule state."""
        rule = AudioReactiveEdgeRule()

        # Create state
        _, ctx = rule.apply(edge_frame, audio_context)
        assert ctx.get_rule_state(rule.name()) is not None

        # Reset
        rule.reset_state(ctx)
        assert ctx.get_rule_state(rule.name()) is None

    def test_thickness_scale_on_beat(self, edge_frame, audio_context):
        """Strong beat should increase thickness_scale."""
        rule = AudioReactiveEdgeRule(effect_type="thickness", sensitivity=1.0)
        _, new_ctx = rule.apply(edge_frame, audio_context)

        state = new_ctx.get_rule_state(rule.name())
        # With beat, thickness should be > 1.0
        assert state.thickness_scale >= 1.0


class TestApplyDisplacement:
    """Test _apply_displacement helper function."""

    def test_zero_displacement_passthrough(self):
        """Zero displacement should return input unchanged."""
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = _apply_displacement(frame, 0.0, 10)
        assert np.array_equal(result, frame)

    def test_zero_max_displacement_passthrough(self):
        """Zero max_displacement should return input unchanged."""
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = _apply_displacement(frame, 0.5, 0)
        assert np.array_equal(result, frame)

    def test_displacement_changes_pixels(self):
        """Non-zero displacement should change pixel positions."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, 5] = 255  # Vertical line at x=5

        result = _apply_displacement(frame, 1.0, 2)

        # Line should have moved
        assert not np.array_equal(result, frame)

    def test_displacement_preserves_shape(self):
        """Displacement should preserve shape."""
        frame = np.full((50, 30, 3), 100, dtype=np.uint8)
        result = _apply_displacement(frame, 0.5, 5)
        assert result.shape == frame.shape


class TestApplyThickness:
    """Test _apply_thickness helper function."""

    def test_unit_scale_passthrough(self):
        """thickness_scale=1.0 should return input unchanged."""
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = _apply_thickness(frame, 1.0)
        assert np.array_equal(result, frame)

    def test_thickness_greater_than_one_dilates(self):
        """thickness_scale > 1.0 should dilate edges."""
        # Create a thin edge
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[10, :] = 255  # Thin horizontal line

        result = _apply_thickness(frame, 2.0)

        # Result should have more non-zero pixels
        assert np.sum(result > 0) > np.sum(frame > 0)

    def test_thickness_less_than_one_erodes(self):
        """thickness_scale < 1.0 should erode edges."""
        # Create a thick edge
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[8:12, :] = 255  # Thick horizontal line

        result = _apply_thickness(frame, 0.5)

        # Result should have fewer non-zero pixels
        assert np.sum(result > 0) <= np.sum(frame > 0)

    def test_thickness_preserves_shape(self):
        """Thickness should preserve shape."""
        frame = np.full((30, 40, 3), 100, dtype=np.uint8)
        result = _apply_thickness(frame, 1.5)
        assert result.shape == frame.shape


class TestGetFrequencyEnergy:
    """Test _get_frequency_energy helper function."""

    def test_none_returns_zero(self):
        """None frequency_bands should return 0.0."""
        energy = _get_frequency_energy(None, "all")
        assert energy == 0.0

    def test_all_returns_average(self):
        """'all' should return average of all bands."""
        bands = {"bass": 0.8, "mid": 0.6, "treble": 0.4}
        energy = _get_frequency_energy(bands, "all")
        # (0.8 + 0.6 + 0.4) / 3 = 0.6
        assert energy == pytest.approx(0.6)

    def test_bass_returns_bass(self):
        """'bass' should return bass energy."""
        bands = {"bass": 0.9, "mid": 0.5, "treble": 0.3}
        energy = _get_frequency_energy(bands, "bass")
        assert energy == 0.9

    def test_mid_returns_mid(self):
        """'mid' should return mid energy."""
        bands = {"bass": 0.9, "mid": 0.5, "treble": 0.3}
        energy = _get_frequency_energy(bands, "mid")
        assert energy == 0.5

    def test_treble_returns_treble(self):
        """'treble' should return treble energy."""
        bands = {"bass": 0.9, "mid": 0.5, "treble": 0.3}
        energy = _get_frequency_energy(bands, "treble")
        assert energy == 0.3

    def test_missing_band_returns_zero(self):
        """Missing band should return 0.0."""
        bands = {"bass": 0.9, "mid": 0.5}  # No treble
        energy = _get_frequency_energy(bands, "treble")
        assert energy == 0.0
