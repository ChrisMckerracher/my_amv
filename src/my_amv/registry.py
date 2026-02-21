"""EffectRule registration and discovery.

This module provides a decorator-based registration system for EffectRule
subclasses. Rules are registered by name and can be instantiated dynamically
from config files.
"""

from importlib import import_module
from typing import Any

from typing_extensions import TYPE_CHECKING

from my_amv.rule import EffectRule

if TYPE_CHECKING:
    from collections.abc import Callable

# Global registry of rule classes
# Key: rule class name (e.g., "EdgeDetectionRule")
# Value: EffectRule subclass
_REGISTRY: dict[str, type[EffectRule]] = {}


def register_rule(cls: type[EffectRule]) -> type[EffectRule]:
    """Decorator to register an EffectRule subclass in the global registry.

    Usage:
        ```python
        @register_rule
        class EdgeDetectionRule(EffectRule[None]):
            ...
        ```

    The rule is registered by its class name. This name is used to reference
    the rule in config files.

    Args:
        cls: EffectRule subclass to register

    Returns:
        The same class (unchanged), for use as a decorator

    Raises:
        ValueError: If a rule with the same name is already registered
    """
    name = cls.__name__
    if name in _REGISTRY:
        raise ValueError(f"Rule '{name}' is already registered")
    _REGISTRY[name] = cls
    return cls


def get_registry() -> dict[str, type[EffectRule]]:
    """Get a copy of the global rule registry.

    Returns:
        Dictionary mapping rule names to EffectRule classes
    """
    return dict(_REGISTRY)


def get_rule(name: str, params: dict[str, Any] | None = None) -> EffectRule:
    """Instantiate a rule by name from the registry.

    Args:
        name: Rule class name (e.g., "EdgeDetectionRule")
        params: Optional parameters to pass to the rule's configure() method

    Returns:
        Instantiated EffectRule subclass

    Raises:
        KeyError: If the rule name is not in the registry
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Unknown rule '{name}'. Available rules: {available or '(none)'}"
        )

    rule_cls = _REGISTRY[name]
    rule = rule_cls()

    if params:
        rule.configure(params)

    return rule


def discover_rules() -> None:
    """Import all my_amv.rules.* submodules to trigger @register_rule decorators.

    This function should be called once at startup (e.g., when loading a config
    or starting the CLI) to ensure all rule classes are registered.

    The function imports each rules submodule, which causes the @register_rule
    decorators on each rule class to execute, populating the global registry.

    This function is idempotent - calling it multiple times will not re-register
    rules or raise errors.

    Note:
        This function uses import_module within a try/except for each submodule
        to gracefully handle optional dependencies (e.g., rules that require
        packages that may not be installed).
    """
    # Register CompositeRule (built-in, from rule.py)
    # Import here to avoid circular dependency
    from my_amv.rule import CompositeRule

    if "CompositeRule" not in _REGISTRY:
        _REGISTRY["CompositeRule"] = CompositeRule

    rule_modules = [
        "my_amv.rules.segmentation",
        "my_amv.rules.edge_detection",
        "my_amv.rules.depth",
        "my_amv.rules.halftone",
        "my_amv.rules.audio_reactive",
        "my_amv.rules.temporal_smooth",
    ]

    for module_name in rule_modules:
        try:
            import_module(module_name)
        except ImportError:
            # Skip modules that can't be imported (e.g., missing dependencies)
            pass

    # Merge rules registered via my_amv.rules.register_rule (separate registry)
    # into the main registry so both registration paths work.
    try:
        from my_amv.rules import _RULE_REGISTRY as rules_registry  # type: ignore[attr-defined]
        for name, cls in rules_registry.items():
            if name not in _REGISTRY:
                _REGISTRY[name] = cls
    except (ImportError, AttributeError):
        pass


def clear_registry() -> None:
    """Clear all registered rules.

    This is primarily useful for testing. Production code should not need to
    clear the registry.
    """
    _REGISTRY.clear()
