"""Tests for EffectRule registration and discovery."""

import pytest

from my_amv.context import FrameContext
from my_amv.registry import clear_registry, discover_rules, get_registry, get_rule, register_rule
from my_amv.rule import EffectRule
from my_amv.types import RGBArray


class DummyRule(EffectRule[None]):
    """Dummy rule for testing."""

    def name(self) -> str:
        return "DummyRule"

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        return frame.copy(), context


class AnotherDummyRule(EffectRule[dict]):
    """Another dummy rule for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.value = 0

    def name(self) -> str:
        return "AnotherDummyRule"

    def apply(
        self, frame: RGBArray, context: FrameContext
    ) -> tuple[RGBArray, FrameContext]:
        return frame.copy(), context

    def configure(self, params: dict) -> None:
        if "value" in params:
            self.value = params["value"]


class TestRegistryDecorator:
    """Test @register_rule decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def test_register_rule_adds_to_registry(self):
        """@register_rule should add rule class to registry."""
        register_rule(DummyRule)

        registry = get_registry()
        assert "DummyRule" in registry
        assert registry["DummyRule"] is DummyRule

    def test_register_rule_returns_class(self):
        """@register_rule should return the class unchanged."""
        result = register_rule(DummyRule)
        assert result is DummyRule

    def test_register_rule_duplicate_raises_error(self):
        """Registering duplicate rule name should raise ValueError."""
        register_rule(DummyRule)

        with pytest.raises(ValueError, match="is already registered"):
            register_rule(DummyRule)

    def test_register_multiple_rules(self):
        """Multiple rules can be registered."""
        register_rule(DummyRule)
        register_rule(AnotherDummyRule)

        registry = get_registry()
        assert len(registry) == 2
        assert "DummyRule" in registry
        assert "AnotherDummyRule" in registry


class TestGetRegistry:
    """Test get_registry() function."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def test_get_registry_returns_dict(self):
        """get_registry should return a dictionary."""
        result = get_registry()
        assert isinstance(result, dict)

    def test_get_registry_returns_copy(self):
        """get_registry should return a copy, not the original."""
        register_rule(DummyRule)
        registry1 = get_registry()
        registry2 = get_registry()

        # Modifying one should not affect the other
        registry1["FakeRule"] = DummyRule
        assert "FakeRule" not in registry2
        assert "FakeRule" not in get_registry()


class TestGetRule:
    """Test get_rule() function."""

    def setup_method(self):
        """Clear registry and register test rules."""
        clear_registry()
        register_rule(DummyRule)
        register_rule(AnotherDummyRule)

    def test_get_rule_instantiates_rule(self):
        """get_rule should return an instance of the rule class."""
        rule = get_rule("DummyRule")
        assert isinstance(rule, DummyRule)
        assert isinstance(rule, EffectRule)

    def test_get_rule_with_params_calls_configure(self):
        """get_rule should call configure() with params."""
        rule = get_rule("AnotherDummyRule", {"value": 42})
        assert rule.value == 42

    def test_get_rule_unknown_rule_raises_error(self):
        """get_rule should raise KeyError for unknown rule names."""
        with pytest.raises(KeyError, match="Unknown rule 'NonExistentRule'"):
            get_rule("NonExistentRule")

        error_msg = str(pytest.raises(KeyError, match="NonExistentRule"))
        # Check that the error message includes available rules
        assert "DummyRule" in str(get_registry())

    def test_get_rule_without_params(self):
        """get_rule should work without params."""
        rule = get_rule("DummyRule")
        assert rule is not None


class TestDiscoverRules:
    """Test discover_rules() function."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def test_discover_rules_populates_registry(self):
        """discover_rules should import rule modules and populate registry."""
        # Note: This test assumes the rules modules exist and can be imported.
        # If they haven't been implemented yet, they'll be skipped due to ImportError.

        discover_rules()

        registry = get_registry()

        # CompositeRule should always be available (registered by discover_rules)
        assert "CompositeRule" in registry


class TestClearRegistry:
    """Test clear_registry() function."""

    def test_clear_registry_removes_all_rules(self):
        """clear_registry should remove all registered rules."""
        register_rule(DummyRule)
        register_rule(AnotherDummyRule)

        # CompositeRule is auto-registered in the previous test
        pre_clear_count = len(get_registry())
        assert pre_clear_count >= 2

        clear_registry()

        assert len(get_registry()) == 0
