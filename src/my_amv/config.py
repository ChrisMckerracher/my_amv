"""Configuration loading and validation.

This module provides Pydantic models for loading and validating pipeline
configuration from YAML or JSON files.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from my_amv.registry import discover_rules, get_rule
from my_amv.rule import EffectRule


class RuleConfig(BaseModel):
    """Configuration for a single EffectRule.

    Attributes:
        name: Rule class name (must be registered in the registry)
        input_layer: Which layer to read from (None = Layer.MAIN)
        output_layer: Which layer to write to (None = inline mode)
        params: Rule-specific parameters passed to configure()
    """

    name: str
    input_layer: str | None = Field(
        default=None, description="Layer to read from (null = Layer.MAIN)"
    )
    output_layer: str | None = Field(
        default=None, description="Layer to write to (null = inline)"
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Rule parameters")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Rule name must not be empty."""
        if not v or not v.strip():
            raise ValueError("Rule name must not be empty")
        return v


class PipelineConfig(BaseModel):
    """Configuration for the entire pipeline.

    Attributes:
        input: Path to input video file
        output: Path to output video file
        audio: Optional path to separate audio file
        frame_range: Optional (start, end) frame range (end=-1 for all frames)
        output_frames_dir: Optional directory to save individual frames
        checkpoint_dir: Optional directory for checkpoint files
        rules: List of rule configurations to apply in order
    """

    input: Path = Field(description="Input video file path")
    output: Path = Field(description="Output video file path")
    audio: Path | None = Field(default=None, description="Optional audio file path")
    frame_range: tuple[int, int] | None = Field(
        default=None, description="Frame range (start, end), end=-1 for all"
    )
    output_frames_dir: Path | None = Field(
        default=None, description="Directory to save individual frames"
    )
    checkpoint_dir: Path | None = Field(
        default=None, description="Checkpoint directory for resumable processing"
    )
    rules: list[RuleConfig] = Field(
        default_factory=list, description="List of effect rules to apply"
    )

    @field_validator("input", "output", "audio", "output_frames_dir", "checkpoint_dir")
    @classmethod
    def paths_must_be_absolute_or_resolved(cls, v: Path | None) -> Path | None:
        """Resolve paths to absolute paths."""
        if v is None:
            return None
        return v.resolve()

    @field_validator("frame_range")
    @classmethod
    def frame_range_must_be_valid(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        """Frame range must have start <= end or end == -1."""
        if v is None:
            return None
        start, end = v
        if start < 0:
            raise ValueError(f"Frame range start must be >= 0, got {start}")
        if end != -1 and end < start:
            raise ValueError(
                f"Frame range end ({end}) must be >= start ({start}) or -1 for all frames"
            )
        return v


def load_config(path: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML or JSON file.

    Args:
        path: Path to config file (.yaml, .yml, or .json)

    Returns:
        Validated PipelineConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid or has unknown fields
        yaml.YAMLError: If YAML parsing fails
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    path = path.resolve()
    suffix = path.suffix.lower()

    with path.open("r") as f:
        if suffix in (".yaml", ".yml"):
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse YAML config: {e}") from e
        elif suffix == ".json":
            import json

            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON config: {e}") from e
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. "
                "Supported formats: .yaml, .yml, .json"
            )

    if data is None:
        raise ValueError("Config file is empty")

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dictionary, got {type(data).__name__}")

    # Parse and validate with Pydantic
    try:
        config = PipelineConfig(**data)
    except Exception as e:
        raise ValueError(f"Config validation failed: {e}") from e

    return config


def validate_rule_names(config: PipelineConfig) -> None:
    """Validate that all rule names in the config are registered.

    This is called before building the pipeline to provide clear error
    messages if a rule name is misspelled or not yet implemented.

    Args:
        config: Pipeline configuration to validate

    Raises:
        ValueError: If any rule name is not in the registry
    """
    from my_amv.registry import get_registry

    registry = get_registry()

    if not registry:
        raise ValueError(
            "No rules are registered. Make sure discover_rules() has been called."
        )

    available = ", ".join(sorted(registry.keys()))

    for rule_config in config.rules:
        name = rule_config.name
        if name not in registry:
            raise ValueError(
                f"Unknown rule '{name}'. Available rules: {available}"
            )


def validate_layer_references(config: PipelineConfig) -> None:
    """Validate that input_layer references exist at the point they're used.

    This performs a static analysis of the rule chain to detect forward
    references to layers that haven't been created yet.

    Args:
        config: Pipeline configuration to validate

    Raises:
        ValueError: If any input_layer references a non-existent layer
    """
    from my_amv.types import Layer

    # Only ORIGINAL and MAIN are available at the start.
    # Other built-in layers (MASK, EDGES, DEPTH, SKELETON) need to be created
    # by rules before they can be used as input_layer.
    available_layers = {
        Layer.ORIGINAL.value,  # The unmodified input frame
        Layer.MAIN.value,  # Default working layer
    }

    # Layer.FINAL is only created at the end, not available during processing
    # Layer.MASK, EDGES, DEPTH, SKELETON must be explicitly created by rules

    for i, rule_config in enumerate(config.rules):
        # Check input_layer
        input_layer = rule_config.input_layer or Layer.MAIN.value
        if input_layer not in available_layers:
            raise ValueError(
                f"Rule {i} ({rule_config.name}): "
                f"input_layer '{input_layer}' does not exist. "
                f"Available layers at this point: {sorted(available_layers)}"
            )

        # After this rule runs, output_layer becomes available
        output_layer = rule_config.output_layer
        if output_layer:
            available_layers.add(output_layer)
        else:
            # Inline mode: the output goes back to the main chain
            # The layer itself is available as Layer.MAIN
            pass


def build_pipeline(config: PipelineConfig) -> list[EffectRule]:
    """Build a list of EffectRule instances from the configuration.

    This function:
    1. Discovers all available rules
    2. Validates rule names
    3. Validates layer references
    4. Instantiates each rule with its parameters

    Args:
        config: Validated pipeline configuration

    Returns:
        List of instantiated EffectRule objects

    Raises:
        ValueError: If validation fails
    """
    # Discover all rules
    discover_rules()

    # Validate rule names exist
    validate_rule_names(config)

    # Validate layer references don't have forward references
    validate_layer_references(config)

    # Build the rule chain
    rules: list[EffectRule] = []
    for rule_config in config.rules:
        rule = get_rule(rule_config.name, rule_config.params)

        # Set input/output layers
        if rule_config.input_layer is not None:
            rule.input_layer = rule_config.input_layer
        if rule_config.output_layer is not None:
            rule.output_layer = rule_config.output_layer

        rules.append(rule)

    return rules
