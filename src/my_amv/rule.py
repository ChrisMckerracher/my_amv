"""EffectRule abstract base class and composable processing units.

This module defines the core abstraction for the composable effect pipeline.
The EffectRule ABC defines the interface that all processing rules implement.

ROUTING INVARIANT (from AGENTS.md):
- The Pipeline (not the rule) handles routing based on output_layer
- If rule.output_layer is None → inline: result replaces current frame
- If rule.output_layer is not None → branch: result goes to layer store
- CompositeRule is special — it reads multiple named layers and merges them
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic

import numpy as np

from my_amv.context import FrameContext
from my_amv.types import BlendMode, Layer, LayerKey, RGBArray, StateT


class EffectRule(ABC, Generic[StateT]):
    """Abstract base class for composable processing rules.

    Each EffectRule:
    - Reads from input_layer (default: Layer.MAIN)
    - Writes to output_layer (default: None = inline replacement)
    - Applies some transformation to the frame
    - Can carry inter-frame state for temporal coherence

    Type Parameters:
        StateT: The type of state this rule carries between frames

    The routing behavior (inline vs branch) is handled by the Pipeline,
    not by the rule itself. Rules always return (output_frame, context).
    """

    #: Which layer to read from. Default: Layer.MAIN
    input_layer: LayerKey = Layer.MAIN

    #: Which layer to write to. Default: None = inline (replace current frame)
    output_layer: LayerKey | None = None

    @abstractmethod
    def name(self) -> str:
        """Return the name of this rule.

        Used for:
        - State key in FrameContext._rule_state
        - Logging and debugging
        - Configuration file references

        Returns:
            Rule name (must be unique per rule type instance)
        """

    @abstractmethod
    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        """Apply this rule to a frame.

        Args:
            frame: Input RGB array (H, W, 3)
            context: Frame processing context

        Returns:
            (output_frame, updated_context) tuple
            - output_frame: Processed RGB array
            - updated_context: Context with any state updates
        """

    def configure(self, params: dict) -> None:
        """Configure this rule with parameters.

        Called during pipeline setup from the config file.
        Override to validate and apply rule-specific parameters.

        Default implementation does nothing (no parameters).

        Args:
            params: Dictionary of parameter names to values

        Raises:
            ValueError: If any parameter is invalid
        """
        pass

    def reset_state(self, context: FrameContext) -> FrameContext:
        """Clear this rule's inter-frame state.

        Called at scene boundaries or when requested.

        Args:
            context: Current frame context

        Returns:
            Updated context with this rule's state cleared
        """
        context.clear_rule_state(self.name())
        return context

    def serialize_state(self, context: FrameContext) -> dict:
        """Serialize this rule's state for checkpointing.

        Override to return a JSON-serializable representation of state.
        Useful for long-running video jobs that can be resumed.

        Args:
            context: Current frame context

        Returns:
            JSON-serializable dict (default: empty dict)
        """
        return {}

    def deserialize_state(
        self, context: FrameContext, state: dict
    ) -> FrameContext:
        """Deserialize and restore this rule's state.

        Override to restore state from a previously serialized checkpoint.

        Args:
            context: Current frame context
            state: Previously serialized state dict

        Returns:
            Updated context with restored state
        """
        return context


@dataclass
class LayerBlend:
    """Configuration for blending a single layer in CompositeRule.

    Attributes:
        layer: Name of the layer to blend
        blend_mode: How to blend this layer (screen, multiply, add, over, normal)
        opacity: Layer opacity [0.0, 1.0]
    """

    layer: LayerKey
    blend_mode: BlendMode
    opacity: float


class CompositeRule(EffectRule[None]):
    """Merge multiple named layers into a single output.

    CompositeRule is special — it reads multiple named layers from the
    context and blends them together using configurable blend modes.

    Unlike normal rules, CompositeRule doesn't use input_layer (it reads
    from multiple layers) and output_layer must be set (branch mode).

    Blend Modes:
        screen: 1 - (1-a)(1-b) — lightens, useful for adding light
        multiply: a*b/255 — darkens, useful for shadows
        add: min(a+b, 255) — simple addition, clips at 255
        over: Alpha compositing (not implemented, requires alpha channel)
        normal: b (foreground replaces background)

    Example:
        rule = CompositeRule(
            layers=[
                LayerBlend(layer="edges", blend_mode="screen", opacity=0.8),
                LayerBlend(layer="depth", blend_mode="multiply", opacity=0.5),
            ],
            background=Layer.MAIN,
        )
    """

    def __init__(
        self,
        layers: list[LayerBlend],
        background: LayerKey | None = None,
    ) -> None:
        """Initialize CompositeRule.

        Args:
            layers: List of layers to blend, in back-to-front order
            background: Background layer to start with (default: Layer.MAIN)
        """
        self.layers = layers
        self.background = background if background is not None else Layer.MAIN
        # CompositeRule always outputs to a layer (branch mode)
        self.output_layer = Layer.FINAL
        self.input_layer = Layer.MAIN  # Not used, but required by interface

    def name(self) -> str:
        return "CompositeRule"

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        """Composite multiple layers into a single output.

        Args:
            frame: Input frame (ignored, we read from context layers)
            context: Frame context with layer store

        Returns:
            (composited_frame, context)

        Raises:
            KeyError: If any referenced layer doesn't exist
            ValueError: If opacity is outside [0, 1]
        """
        # Start with background layer
        if self.background == Layer.MAIN:
            result = frame.copy()
        else:
            result = context.get_layer(self.background).copy()

        # Blend each layer in order
        for layer_blend in self.layers:
            if not (0.0 <= layer_blend.opacity <= 1.0):
                raise ValueError(f"Opacity must be in [0, 1], got {layer_blend.opacity}")

            layer_arr = context.get_layer(layer_blend.layer)
            result = self._blend(result, layer_arr, layer_blend)

        return result, context

    def _blend(self, base: RGBArray, overlay: RGBArray, config: LayerBlend) -> RGBArray:
        """Blend overlay onto base using the configured blend mode.

        Args:
            base: Background RGB array
            overlay: Foreground RGB array
            config: LayerBlend configuration

        Returns:
            Blended RGB array
        """
        mode = config.blend_mode
        opacity = config.opacity

        # Apply opacity to overlay
        if opacity < 1.0:
            overlay = (overlay * opacity).astype(np.uint8)

        if mode == "normal":
            # Foreground replaces background
            result = overlay
        elif mode == "screen":
            # 1 - (1-a)(1-b)
            # Convert to float for computation
            base_f = base.astype(np.float32)
            overlay_f = overlay.astype(np.float32)
            result = 255 - (255 - base_f) * (255 - overlay_f) / 255
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif mode == "multiply":
            # a * b / 255
            result = (base.astype(np.float32) * overlay.astype(np.float32) / 255).astype(
                np.uint8
            )
        elif mode == "add":
            # min(a + b, 255)
            result = np.minimum(base.astype(np.int16) + overlay.astype(np.int16), 255).astype(
                np.uint8
            )
        elif mode == "over":
            # Alpha compositing - not implemented for RGB without alpha
            # For now, just do normal blend
            # TODO: Implement proper alpha compositing if needed
            result = overlay
        else:
            raise ValueError(f"Unknown blend mode: {mode}")

        return result

    def configure(self, params: dict) -> None:
        """Configure the composite rule.

        Args:
            params: Must contain 'layers' list, optional 'background'

        Raises:
            ValueError: If configuration is invalid
        """
        if "layers" in params:
            # Rebuild layers list from config
            layers_list = []
            for layer_dict in params["layers"]:
                layers_list.append(
                    LayerBlend(
                        layer=layer_dict["layer"],
                        blend_mode=layer_dict["blend_mode"],
                        opacity=layer_dict["opacity"],
                    )
                )
            self.layers = layers_list

        if "background" in params:
            self.background = params["background"]
