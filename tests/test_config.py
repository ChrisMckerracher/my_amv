"""Tests for configuration loading and validation."""

import json
from pathlib import Path

import pytest
import yaml

from my_amv.config import (
    PipelineConfig,
    RuleConfig,
    build_pipeline,
    load_config,
    validate_layer_references,
    validate_rule_names,
)
from my_amv.registry import clear_registry, register_rule
from my_amv.rule import CompositeRule, EffectRule
from my_amv.types import Layer


class DummyRule(EffectRule[None]):
    """Dummy rule for testing."""

    def name(self) -> str:
        return "DummyRule"

    def apply(self, frame, context):
        return frame.copy(), context


class AnotherRule(EffectRule[None]):
    """Another dummy rule for testing."""

    def name(self) -> str:
        return "AnotherRule"

    def apply(self, frame, context):
        return frame.copy(), context


class TestRuleConfig:
    """Test RuleConfig Pydantic model."""

    def test_rule_config_creation(self):
        """RuleConfig should accept valid parameters."""
        config = RuleConfig(name="DummyRule", params={"value": 42})
        assert config.name == "DummyRule"
        assert config.input_layer is None
        assert config.output_layer is None
        assert config.params == {"value": 42}

    def test_rule_config_with_layers(self):
        """RuleConfig should accept input and output layers."""
        config = RuleConfig(
            name="DummyRule",
            input_layer="edges",
            output_layer="halftone",
            params={"strength": 0.5},
        )
        assert config.input_layer == "edges"
        assert config.output_layer == "halftone"

    def test_rule_config_empty_name_raises_error(self):
        """Empty rule name should raise validation error."""
        with pytest.raises(ValueError, match="Rule name must not be empty"):
            RuleConfig(name="")

        with pytest.raises(ValueError, match="Rule name must not be empty"):
            RuleConfig(name="   ")

    def test_rule_config_default_params(self):
        """Default params should be empty dict."""
        config = RuleConfig(name="DummyRule")
        assert config.params == {}


class TestPipelineConfig:
    """Test PipelineConfig Pydantic model."""

    def test_pipeline_config_creation(self):
        """PipelineConfig should accept valid parameters."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[RuleConfig(name="DummyRule")],
        )
        assert config.input == Path("input.mp4").resolve()
        assert config.output == Path("output.mp4").resolve()
        assert config.audio is None
        assert config.frame_range is None
        assert len(config.rules) == 1

    def test_pipeline_config_with_all_fields(self):
        """PipelineConfig should accept all optional fields."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            audio=Path("music.mp3"),
            frame_range=(0, 100),
            output_frames_dir=Path("frames"),
            checkpoint_dir=Path("checkpoints"),
            rules=[
                RuleConfig(name="DummyRule"),
                RuleConfig(name="AnotherRule"),
            ],
        )
        assert config.audio == Path("music.mp3").resolve()
        assert config.frame_range == (0, 100)
        assert config.output_frames_dir == Path("frames").resolve()
        assert config.checkpoint_dir == Path("checkpoints").resolve()
        assert len(config.rules) == 2

    def test_pipeline_config_default_rules(self):
        """Default rules should be empty list."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
        )
        assert config.rules == []

    def test_frame_range_negative_start_raises_error(self):
        """Frame range with negative start should raise error."""
        with pytest.raises(ValueError, match="Frame range start must be >= 0"):
            PipelineConfig(
                input=Path("input.mp4"),
                output=Path("output.mp4"),
                frame_range=(-1, 100),
            )

    def test_frame_range_end_before_start_raises_error(self):
        """Frame range with end < start (and not -1) should raise error."""
        with pytest.raises(ValueError, match="Frame range end.*must be >= start"):
            PipelineConfig(
                input=Path("input.mp4"),
                output=Path("output.mp4"),
                frame_range=(100, 50),
            )

    def test_frame_range_end_negative_one_is_valid(self):
        """Frame range end=-1 should be valid (means 'all frames')."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            frame_range=(0, -1),
        )
        assert config.frame_range == (0, -1)

    def test_frame_range_equal_start_end_is_valid(self):
        """Frame range with start == end should be valid."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            frame_range=(50, 50),
        )
        assert config.frame_range == (50, 50)


class TestLoadConfig:
    """Test load_config() function."""

    def test_load_yaml_config(self, tmp_path):
        """Should load valid YAML config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
input: video.mp4
output: result.mp4
rules:
  - name: DummyRule
    params:
      value: 42
"""
        )

        config = load_config(config_file)
        assert isinstance(config, PipelineConfig)
        assert config.input.name == "video.mp4"
        assert config.output.name == "result.mp4"
        assert len(config.rules) == 1
        assert config.rules[0].name == "DummyRule"
        assert config.rules[0].params == {"value": 42}

    def test_load_json_config(self, tmp_path):
        """Should load valid JSON config file."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "input": "video.mp4",
                    "output": "result.mp4",
                    "rules": [{"name": "DummyRule", "params": {"value": 42}}],
                }
            )
        )

        config = load_config(config_file)
        assert isinstance(config, PipelineConfig)
        assert config.input.name == "video.mp4"
        assert config.output.name == "result.mp4"

    def test_load_config_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_config_unsupported_format(self, tmp_path):
        """Should raise ValueError for unsupported file format."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("not a valid config")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(config_file)

    def test_load_config_empty_file(self, tmp_path):
        """Should raise ValueError for empty file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="Config file is empty"):
            load_config(config_file)

    def test_load_config_invalid_yaml(self, tmp_path):
        """Should raise ValueError for malformed YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(":\n  invalid: yaml: content:")

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            load_config(config_file)

    def test_load_config_invalid_json(self, tmp_path):
        """Should raise ValueError for malformed JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{invalid json}")

        with pytest.raises(ValueError, match="Failed to parse JSON"):
            load_config(config_file)

    def test_load_config_non_dict_raises_error(self, tmp_path):
        """Should raise ValueError if config is not a dict."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2")

        with pytest.raises(ValueError, match="Config must be a dictionary"):
            load_config(config_file)

    def test_load_config_validation_error(self, tmp_path):
        """Should raise ValueError if Pydantic validation fails."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("input: video.mp4\noutput: result.mp4\nrules:\n  - name: ''")

        with pytest.raises(ValueError, match="Config validation failed"):
            load_config(config_file)


class TestValidateRuleNames:
    """Test validate_rule_names() function."""

    def setup_method(self):
        """Clear registry and register test rules."""
        clear_registry()
        register_rule(DummyRule)
        register_rule(AnotherRule)
        register_rule(CompositeRule)

    def test_validate_rule_names_all_valid(self):
        """Should pass when all rule names are registered."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[
                RuleConfig(name="DummyRule"),
                RuleConfig(name="AnotherRule"),
            ],
        )

        # Should not raise
        validate_rule_names(config)

    def test_validate_rule_names_unknown_rule(self):
        """Should raise ValueError for unknown rule name."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[
                RuleConfig(name="DummyRule"),
                RuleConfig(name="NonExistentRule"),
            ],
        )

        with pytest.raises(ValueError, match="Unknown rule 'NonExistentRule'"):
            validate_rule_names(config)

    def test_validate_rule_names_empty_registry(self):
        """Should raise error when no rules are registered."""
        # Clear the registry
        clear_registry()

        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[RuleConfig(name="AnyRule")],
        )

        with pytest.raises(ValueError, match="No rules are registered"):
            validate_rule_names(config)


class TestValidateLayerReferences:
    """Test validate_layer_references() function."""

    def test_validate_layers_valid_chain(self):
        """Should pass for valid layer chain."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[
                RuleConfig(name="DummyRule", output_layer="edges"),
                RuleConfig(name="AnotherRule", input_layer="edges"),
            ],
        )

        # Should not raise
        validate_layer_references(config)

    def test_validate_layers_builtin_layer_available(self):
        """Built-in layers should always be available."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[
                RuleConfig(name="DummyRule", input_layer="original"),
                RuleConfig(name="AnotherRule", input_layer="main"),
            ],
        )

        validate_layer_references(config)

    def test_validate_layers_forward_reference_raises_error(self):
        """Should raise error for forward layer reference."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[
                RuleConfig(
                    name="DummyRule", input_layer="edges"
                ),  # edges doesn't exist yet
            ],
        )

        with pytest.raises(ValueError, match="input_layer 'edges' does not exist"):
            validate_layer_references(config)

    def test_validate_layers_user_layer_becomes_available(self):
        """User-created layers should become available after creation."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[
                RuleConfig(name="DummyRule", output_layer="edges"),
                RuleConfig(name="AnotherRule", input_layer="edges", output_layer="halftone"),
                RuleConfig(name="DummyRule", input_layer="halftone"),
            ],
        )

        validate_layer_references(config)

    def test_validate_layers_main_layer_is_default(self):
        """Layer.MAIN should be the default input_layer."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[
                RuleConfig(name="DummyRule"),  # No input_layer specified
            ],
        )

        validate_layer_references(config)


class TestBuildPipeline:
    """Test build_pipeline() function."""

    def setup_method(self):
        """Clear registry and register test rules."""
        clear_registry()
        register_rule(DummyRule)
        register_rule(AnotherRule)

    def test_build_pipeline_returns_rules(self):
        """Should return list of EffectRule instances."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[
                RuleConfig(name="DummyRule"),
                RuleConfig(name="AnotherRule"),
            ],
        )

        rules = build_pipeline(config)
        assert len(rules) == 2
        assert isinstance(rules[0], DummyRule)
        assert isinstance(rules[1], AnotherRule)

    def test_build_pipeline_passes_params(self):
        """Should pass params to rule.configure()."""
        class ConfigurableRule(EffectRule[None]):
            def __init__(self):
                super().__init__()
                self.value = 0

            def name(self):
                return "ConfigurableRule"

            def apply(self, frame, context):
                return frame.copy(), context

            def configure(self, params):
                self.value = params.get("value", 0)

        clear_registry()
        register_rule(ConfigurableRule)

        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[RuleConfig(name="ConfigurableRule", params={"value": 42})],
        )

        rules = build_pipeline(config)
        assert rules[0].value == 42

    def test_build_pipeline_sets_input_output_layers(self):
        """Should set input_layer and output_layer on rules."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[
                RuleConfig(name="DummyRule", input_layer="original", output_layer="edges"),
            ],
        )

        rules = build_pipeline(config)
        assert rules[0].input_layer == "original"
        assert rules[0].output_layer == "edges"

    def test_build_pipeline_unknown_rule_raises_error(self):
        """Should raise ValueError for unknown rule name."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[RuleConfig(name="NonExistentRule")],
        )

        with pytest.raises(ValueError, match="Unknown rule 'NonExistentRule'"):
            build_pipeline(config)

    def test_build_pipeline_invalid_layer_reference_raises_error(self):
        """Should raise ValueError for invalid layer reference."""
        config = PipelineConfig(
            input=Path("input.mp4"),
            output=Path("output.mp4"),
            rules=[RuleConfig(name="DummyRule", input_layer="nonexistent")],
        )

        with pytest.raises(ValueError, match="input_layer 'nonexistent' does not exist"):
            build_pipeline(config)
