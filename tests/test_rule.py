"""Tests for EffectRule ABC and CompositeRule."""

import numpy as np
import pytest

from my_amv.context import AudioFrame, FrameContext
from my_amv.rule import CompositeRule, EffectRule, LayerBlend
from my_amv.types import Layer, RGBArray


class MockEffectRule(EffectRule[dict]):
    """Mock EffectRule for testing."""

    def __init__(self, name: str = "MockRule") -> None:
        self._name = name
        self.state_value = 0

    def name(self) -> str:
        return self._name

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        # Simple identity transformation
        return frame.copy(), context

    def configure(self, params: dict) -> None:
        if "value" in params:
            self.state_value = params["value"]

    def serialize_state(self, context: FrameContext) -> dict:
        return {"value": self.state_value}

    def deserialize_state(self, context: FrameContext, state: dict) -> FrameContext:
        self.state_value = state.get("value", 0)
        return context


class TestEffectRuleABC:
    """Test EffectRule abstract base class."""

    def test_mock_rule_implements_interface(self):
        """Mock rule should implement all required methods."""
        rule = MockEffectRule()

        assert rule.name() == "MockRule"
        assert hasattr(rule, "apply")
        assert hasattr(rule, "configure")
        assert hasattr(rule, "reset_state")
        assert hasattr(rule, "serialize_state")
        assert hasattr(rule, "deserialize_state")
        assert rule.input_layer == Layer.MAIN
        assert rule.output_layer is None

    def test_rule_apply_returns_tuple(self):
        """Rule.apply() should return (frame, context) tuple."""
        rule = MockEffectRule()
        frame: RGBArray = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)

        result_frame, result_ctx = rule.apply(frame, ctx)

        assert isinstance(result_frame, np.ndarray)
        assert isinstance(result_ctx, FrameContext)
        assert result_frame.shape == (100, 100, 3)

    def test_rule_configure(self):
        """Rule.configure() should update rule parameters."""
        rule = MockEffectRule()

        rule.configure({"value": 42})
        assert rule.state_value == 42

    def test_rule_reset_state(self):
        """Rule.reset_state() should clear rule state from context."""
        rule = MockEffectRule()
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)

        ctx.set_rule_state("MockRule", {"count": 5})
        assert ctx.get_rule_state("MockRule") is not None

        rule.reset_state(ctx)
        assert ctx.get_rule_state("MockRule") is None

    def test_rule_serialize_deserialize_state(self):
        """Rule state should be serializable and deserializable."""
        rule = MockEffectRule()
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)

        rule.configure({"value": 123})

        state = rule.serialize_state(ctx)
        assert state == {"value": 123}

        rule2 = MockEffectRule("MockRule2")
        rule2.deserialize_state(ctx, state)
        assert rule2.state_value == 123


class TestLayerBlend:
    """Test LayerBlend dataclass."""

    def test_layer_blend_creation(self):
        """LayerBlend should store its attributes."""
        blend = LayerBlend(layer=Layer.EDGES, blend_mode="screen", opacity=0.8)

        assert blend.layer == Layer.EDGES
        assert blend.blend_mode == "screen"
        assert blend.opacity == 0.8

    def test_layer_blend_with_string_layer(self):
        """LayerBlend should accept string layer names."""
        blend = LayerBlend(layer="custom", blend_mode="add", opacity=1.0)

        assert blend.layer == "custom"
        assert blend.blend_mode == "add"
        assert blend.opacity == 1.0


class TestCompositeRule:
    """Test CompositeRule blending functionality."""

    def test_composite_rule_creation(self):
        """CompositeRule should be creatable with layers and background."""
        layers = [
            LayerBlend(layer=Layer.EDGES, blend_mode="screen", opacity=0.8),
            LayerBlend(layer="depth", blend_mode="multiply", opacity=0.5),
        ]
        rule = CompositeRule(layers=layers, background=Layer.MAIN)

        assert rule.name() == "CompositeRule"
        assert len(rule.layers) == 2
        assert rule.background == Layer.MAIN

    def test_composite_rule_default_background(self):
        """CompositeRule should default to Layer.MAIN background."""
        layers = [LayerBlend(layer=Layer.EDGES, blend_mode="screen", opacity=0.8)]
        rule = CompositeRule(layers=layers)

        assert rule.background == Layer.MAIN

    def test_composite_rule_name(self):
        """CompositeRule name should be fixed."""
        rule = CompositeRule(layers=[])
        assert rule.name() == "CompositeRule"

    def test_composite_apply_with_layers(self):
        """CompositeRule.apply() should blend layers from context."""
        # Create test frames
        base_frame: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)
        edge_frame: RGBArray = np.full((10, 10, 3), 50, dtype=np.uint8)

        # Create context with layer store
        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx.set_layer(Layer.EDGES, edge_frame)

        # Create rule
        layers = [LayerBlend(layer=Layer.EDGES, blend_mode="screen", opacity=1.0)]
        rule = CompositeRule(layers=layers, background=Layer.MAIN)

        result, _ = rule.apply(base_frame, ctx)

        # Screen blend: 1 - (1-100)(1-50)/255 = 1 - (155*205)/255 ≈ 125.5
        # Due to rounding, should be close to 125
        assert result.shape == (10, 10, 3)
        # The result should be brighter than base due to screen blend
        assert result[0, 0, 0] > base_frame[0, 0, 0]

    def test_blend_mode_multiply(self):
        """Multiply blend mode should darken pixels."""
        base: RGBArray = np.full((10, 10, 3), 200, dtype=np.uint8)
        overlay: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)

        # Multiply: 200 * 100 / 255 ≈ 78
        expected = (200 * 100) // 255

        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx.set_layer("overlay", overlay)

        rule = CompositeRule(
            layers=[LayerBlend(layer="overlay", blend_mode="multiply", opacity=1.0)],
            background=Layer.MAIN,
        )

        result, _ = rule.apply(base, ctx)
        assert result[0, 0, 0] == expected

    def test_blend_mode_add(self):
        """Add blend mode should sum pixel values."""
        base: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)
        overlay: RGBArray = np.full((10, 10, 3), 80, dtype=np.uint8)

        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx.set_layer("overlay", overlay)

        rule = CompositeRule(
            layers=[LayerBlend(layer="overlay", blend_mode="add", opacity=1.0)],
            background=Layer.MAIN,
        )

        result, _ = rule.apply(base, ctx)
        assert result[0, 0, 0] == 180  # 100 + 80

    def test_blend_mode_add_clips_at_255(self):
        """Add blend mode should clip at 255."""
        base: RGBArray = np.full((10, 10, 3), 200, dtype=np.uint8)
        overlay: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)

        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx.set_layer("overlay", overlay)

        rule = CompositeRule(
            layers=[LayerBlend(layer="overlay", blend_mode="add", opacity=1.0)],
            background=Layer.MAIN,
        )

        result, _ = rule.apply(base, ctx)
        assert result[0, 0, 0] == 255  # clipped

    def test_blend_mode_normal(self):
        """Normal blend mode should replace with overlay."""
        base: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)
        overlay: RGBArray = np.full((10, 10, 3), 200, dtype=np.uint8)

        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx.set_layer("overlay", overlay)

        rule = CompositeRule(
            layers=[LayerBlend(layer="overlay", blend_mode="normal", opacity=1.0)],
            background=Layer.MAIN,
        )

        result, _ = rule.apply(base, ctx)
        assert result[0, 0, 0] == 200  # overlay replaces base

    def test_opacity_less_than_one(self):
        """Opacity should scale the overlay."""
        base: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)
        overlay: RGBArray = np.full((10, 10, 3), 255, dtype=np.uint8)

        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx.set_layer("overlay", overlay)

        rule = CompositeRule(
            layers=[LayerBlend(layer="overlay", blend_mode="normal", opacity=0.5)],
            background=Layer.MAIN,
        )

        result, _ = rule.apply(base, ctx)
        # 255 * 0.5 = 127.5, rounds to 127
        assert result[0, 0, 0] == 127

    def test_opacity_raises_when_invalid(self):
        """Opacity outside [0, 1] should raise ValueError."""
        base: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)
        overlay: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)

        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx.set_layer("overlay", overlay)

        rule = CompositeRule(
            layers=[LayerBlend(layer="overlay", blend_mode="normal", opacity=1.5)],
            background=Layer.MAIN,
        )

        with pytest.raises(ValueError, match="Opacity must be in \\[0, 1\\]"):
            rule.apply(base, ctx)

    def test_missing_layer_raises_key_error(self):
        """Missing layer in context should raise KeyError."""
        base: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)

        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        # Don't add the layer to context

        rule = CompositeRule(
            layers=[LayerBlend(layer="missing", blend_mode="normal", opacity=1.0)],
            background=Layer.MAIN,
        )

        with pytest.raises(KeyError):
            rule.apply(base, ctx)

    def test_unknown_blend_mode_raises(self):
        """Unknown blend mode should raise ValueError."""
        # This would require modifying the blend mode check
        # For now, we test the default valid modes work
        base: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)
        overlay: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)

        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx.set_layer("overlay", overlay)

        # Create rule and manually set invalid blend mode
        rule = CompositeRule(
            layers=[LayerBlend(layer="overlay", blend_mode="normal", opacity=1.0)],
            background=Layer.MAIN,
        )
        rule.layers[0].blend_mode = "invalid"  # type: ignore

        with pytest.raises(ValueError, match="Unknown blend mode"):
            rule.apply(base, ctx)

    def test_composite_with_background_from_context(self):
        """Composite should use background from context when not MAIN."""
        base: RGBArray = np.full((10, 10, 3), 50, dtype=np.uint8)
        bg: RGBArray = np.full((10, 10, 3), 150, dtype=np.uint8)
        overlay: RGBArray = np.full((10, 10, 3), 255, dtype=np.uint8)

        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx.set_layer("bg", bg)
        ctx.set_layer("overlay", overlay)

        rule = CompositeRule(
            layers=[LayerBlend(layer="overlay", blend_mode="add", opacity=1.0)],
            background="bg",
        )

        result, _ = rule.apply(base, ctx)
        # 150 + 255 = 405, clips to 255
        assert result[0, 0, 0] == 255

    def test_composite_configure(self):
        """CompositeRule.configure() should update layers."""
        rule = CompositeRule(layers=[])

        rule.configure(
            {
                "layers": [
                    {"layer": "edges", "blend_mode": "screen", "opacity": 0.7},
                    {"layer": "depth", "blend_mode": "multiply", "opacity": 0.5},
                ],
                "background": "custom_bg",
            }
        )

        assert len(rule.layers) == 2
        assert rule.layers[0].layer == "edges"
        assert rule.layers[0].blend_mode == "screen"
        assert rule.layers[0].opacity == 0.7
        assert rule.layers[1].layer == "depth"
        assert rule.background == "custom_bg"

    def test_multiple_layers_blend_in_order(self):
        """Multiple layers should blend in order (back to front)."""
        base: RGBArray = np.full((10, 10, 3), 100, dtype=np.uint8)
        layer1: RGBArray = np.full((10, 10, 3), 50, dtype=np.uint8)
        layer2: RGBArray = np.full((10, 10, 3), 30, dtype=np.uint8)

        ctx = FrameContext(frame_index=0, frame_count=100, fps=30.0)
        ctx.set_layer("layer1", layer1)
        ctx.set_layer("layer2", layer2)

        # layer2 (30) should be applied after layer1 (50)
        # Using add mode: 100 + 50 + 30 = 180
        rule = CompositeRule(
            layers=[
                LayerBlend(layer="layer1", blend_mode="add", opacity=1.0),
                LayerBlend(layer="layer2", blend_mode="add", opacity=1.0),
            ],
            background=Layer.MAIN,
        )

        result, _ = rule.apply(base, ctx)
        assert result[0, 0, 0] == 180
