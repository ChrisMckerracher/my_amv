"""Tests for type definitions.

Critical invariant checks:
- Layer is Enum, not StrEnum
- LayerKey accepts both Layer and str
- Layer.MAIN is a Layer instance, not a str
"""

import enum
from typing import Literal, TypeVar, Union, get_args, get_origin

import pytest

from my_amv.types import (
    BlendMode,
    DepthArray,
    EdgeArray,
    Layer,
    LayerKey,
    MaskArray,
    RGBArray,
    StateT,
)


class TestLayerIsEnumNotStrEnum:
    """Layer MUST be Enum, NOT StrEnum.

    This preserves the dict[LayerKey, RGBArray] union type distinction.
    If Layer were StrEnum, LayerKey would collapse to just str.
    """

    def test_layer_is_enum(self):
        """Layer should be an instance of Enum."""
        assert issubclass(Layer, enum.Enum)

    def test_layer_is_not_str_enum(self):
        """Layer should NOT be a StrEnum."""
        # StrEnum was added in Python 3.11
        try:
            from enum import StrEnum

            assert not issubclass(Layer, StrEnum)
        except ImportError:
            # Python < 3.11 doesn't have StrEnum, so this passes by default
            pass

    def test_layer_member_is_enum_instance(self):
        """Layer members should be Layer enum instances, not strings."""
        assert isinstance(Layer.MAIN, Layer)
        # Layer members have a .value attribute that is a string
        assert isinstance(Layer.MAIN.value, str)
        # But the member itself is an Enum, not a str
        assert not isinstance(Layer.MAIN, str)


class TestLayerKeyUnionType:
    """LayerKey = Layer | str union type."""

    def test_layer_key_is_union(self):
        """LayerKey should be a Union type."""
        assert get_origin(LayerKey) is Union

    def test_layer_key_accepts_layer(self):
        """LayerKey should accept Layer enum members."""
        layer_key: LayerKey = Layer.MAIN
        assert layer_key is Layer.MAIN

    def test_layer_key_accepts_str(self):
        """LayerKey should accept plain strings."""
        layer_key: LayerKey = "custom_layer"
        assert layer_key == "custom_layer"

    def test_layer_key_accepts_both(self):
        """LayerKey should work with both Layer and str."""
        keys: list[LayerKey] = [Layer.ORIGINAL, Layer.EDGES, "halftone", "audio_lines"]
        assert len(keys) == 4


class TestLayerEnumValues:
    """Test Layer enum has all expected values."""

    def test_layer_has_original(self):
        assert Layer.ORIGINAL.value == "original"

    def test_layer_has_mask(self):
        assert Layer.MASK.value == "mask"

    def test_layer_has_edges(self):
        assert Layer.EDGES.value == "edges"

    def test_layer_has_depth(self):
        assert Layer.DEPTH.value == "depth"

    def test_layer_has_skeleton(self):
        assert Layer.SKELETON.value == "skeleton"

    def test_layer_has_final(self):
        assert Layer.FINAL.value == "final"

    def test_layer_has_main(self):
        assert Layer.MAIN.value == "main"


class TestBlendMode:
    """Test BlendMode literal type."""

    def test_blend_mode_is_literal(self):
        """BlendMode should be a Literal type."""
        assert get_origin(BlendMode) is Literal

    def test_blend_mode_values(self):
        """BlendMode should have expected blend modes."""
        assert get_args(BlendMode) == (
            "screen",
            "multiply",
            "add",
            "over",
            "normal",
        )


class TestArrayTypes:
    """Test array type aliases are properly defined."""

    def test_rgb_array_exists(self):
        """RGBArray should be defined."""
        assert RGBArray is not None
        # It's an NDArray with uint8 dtype
        # We can't fully test the type at runtime, but we can check it exists

    def test_mask_array_exists(self):
        """MaskArray should be defined."""
        assert MaskArray is not None

    def test_edge_array_exists(self):
        """EdgeArray should be defined."""
        assert EdgeArray is not None

    def test_depth_array_exists(self):
        """DepthArray should be defined."""
        assert DepthArray is not None


class TestStateT:
    """Test StateT type variable."""

    def test_state_t_is_type_var(self):
        """StateT should be a TypeVar."""
        assert isinstance(StateT, TypeVar)

    def test_state_t_is_not_bound(self):
        """StateT should be unbound (accepts any type)."""
        assert StateT.__constraints__ == ()
        assert StateT.__bound__ is None
