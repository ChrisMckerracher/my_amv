"""Built-in EffectRule implementations.

This module exports all available rule classes for dynamic registration.
Import from this module to make rules available to the pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Rule registry for dynamic lookup
_RULE_REGISTRY: dict[str, type] = {}


def register_rule(cls: type) -> type:
    """Decorator to register an EffectRule class for dynamic lookup.

    Usage:
        @register_rule
        class MyRule(EffectRule):
            ...
    """
    _RULE_REGISTRY[cls.__name__] = cls
    return cls


def get_rule(name: str) -> type | None:
    """Get a registered rule class by name.

    Args:
        name: Rule class name (e.g., "DepthMappingRule")

    Returns:
        The rule class if found, None otherwise
    """
    return _RULE_REGISTRY.get(name)


def list_rules() -> list[str]:
    """List all registered rule names.

    Returns:
        List of rule class names
    """
    return list(_RULE_REGISTRY.keys())


# Lazy imports to avoid circular dependency
# The rule modules import register_rule from this module
if not TYPE_CHECKING:
    from my_amv.rules.audio_reactive import AudioReactiveEdgeRule
    from my_amv.rules.depth import DepthMappingRule
    from my_amv.rules.edge_detection import EdgeDetectionRule
    from my_amv.rules.halftone import HalfToneRandomizationRule
    from my_amv.rules.segmentation import PersonSegmentationRule
    from my_amv.rules.temporal_smooth import TemporalSmoothRule

    __all__ = [
        "AudioReactiveEdgeRule",
        "DepthMappingRule",
        "EdgeDetectionRule",
        "HalfToneRandomizationRule",
        "PersonSegmentationRule",
        "TemporalSmoothRule",
        "register_rule",
        "get_rule",
        "list_rules",
    ]
else:
    __all__ = [
        "AudioReactiveEdgeRule",
        "DepthMappingRule",
        "EdgeDetectionRule",
        "HalfToneRandomizationRule",
        "PersonSegmentationRule",
        "TemporalSmoothRule",
        "register_rule",
        "get_rule",
        "list_rules",
    ]
