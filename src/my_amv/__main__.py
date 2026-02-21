"""CLI entry point for my_amv.

This module provides the command-line interface for running the AMV pipeline.
Uses typer for argument parsing and command routing.

Usage:
    python -m my_amv run config.yaml
    python -m my_amv run --input video.mp4 --output out.mp4 --audio music.mp3
    python -m my_amv list-rules
"""

from pathlib import Path
from typing import Optional

import typer

from my_amv.audio import AudioSource
from my_amv.config import PipelineConfig, RuleConfig, build_pipeline, load_config
from my_amv.pipeline import Pipeline
from my_amv.registry import discover_rules, get_registry
from my_amv.rule import CompositeRule

# Register CompositeRule so it's always available
discover_rules()

# Create the main typer app
app = typer.Typer(
    name="my_amv",
    help="AMV (Anime Music Video) art pipeline: video + audio -> processed art video",
    add_completion=False,
)


@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to pipeline config file (YAML or JSON)",
        exists=True,
        resolve_path=True,
    ),
    input: Optional[Path] = typer.Option(
        None,
        "--input", "-i",
        help="Input video file",
        exists=True,
        resolve_path=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output video file",
        resolve_path=True,
    ),
    audio: Optional[Path] = typer.Option(
        None,
        "--audio", "-a",
        help="Audio file to mux into output",
        exists=True,
        resolve_path=True,
    ),
    frame_range: Optional[str] = typer.Option(
        None,
        "--frame-range", "-r",
        help="Frame range as 'start,end' (use -1 for end to process all frames)",
    ),
    output_frames: Optional[Path] = typer.Option(
        None,
        "--output-frames", "-f",
        help="Directory to save individual frames",
        resolve_path=True,
    ),
    checkpoint_dir: Optional[Path] = typer.Option(
        None,
        "--checkpoint-dir",
        help="Directory for checkpoint files",
        resolve_path=True,
    ),
) -> None:
    """Run the AMV pipeline on a video.

    Either --config or (--input + --output) must be provided.

    Example using config file:
        my_amv run --config pipeline.yaml

    Example using CLI arguments:
        my_amv run --input video.mp4 --output result.mp4 --audio music.mp3
    """
    try:
        # Validate arguments
        if config is None and (input is None or output is None):
            typer.echo(
                "Error: Either --config or (--input + --output) must be provided.\n"
                "Use 'my_amv run --help' for usage information.",
                err=True,
            )
            raise typer.Exit(1)

        # Load configuration
        if config is not None:
            # Load from config file
            typer.echo(f"Loading config from: {config}")
            pipeline_config = load_config(config)
        else:
            # Build config from CLI arguments
            if input is None or output is None:
                # This should never happen due to the check above, but mypy needs it
                typer.echo("Error: --input and --output are required.", err=True)
                raise typer.Exit(1)

            typer.echo("Building config from CLI arguments")

            # Parse frame range if provided
            parsed_frame_range = None
            if frame_range is not None:
                try:
                    parts = frame_range.split(",")
                    if len(parts) != 2:
                        raise ValueError
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    parsed_frame_range = (start, end)
                except ValueError:
                    typer.echo(
                        f"Error: Invalid frame range '{frame_range}'. "
                        "Use format 'start,end' (e.g., '0,100' or '0,-1').",
                        err=True,
                    )
                    raise typer.Exit(1)

            pipeline_config = PipelineConfig(
                input=input,
                output=output,
                audio=audio,
                frame_range=parsed_frame_range,
                output_frames_dir=output_frames,
                checkpoint_dir=checkpoint_dir,
                rules=[],  # Empty rule chain for CLI-only mode
            )

        # Display available rules
        registry = get_registry()

        if not registry:
            typer.echo("Warning: No rules registered. The pipeline will have no effect.", err=True)

        # Build the pipeline
        typer.echo(f"Input: {pipeline_config.input}")
        typer.echo(f"Output: {pipeline_config.output}")
        if pipeline_config.audio:
            typer.echo(f"Audio: {pipeline_config.audio}")
        if pipeline_config.frame_range:
            typer.echo(f"Frame range: {pipeline_config.frame_range[0]} to {pipeline_config.frame_range[1]}")

        # Build rule chain
        rules = build_pipeline(pipeline_config)
        typer.echo(f"Rules: {len(rules)}")

        for i, rule in enumerate(rules):
            typer.echo(f"  {i + 1}. {rule.name()}")
            if rule.input_layer != "main":
                typer.echo(f"     input_layer: {rule.input_layer}")
            if rule.output_layer is not None:
                typer.echo(f"     output_layer: {rule.output_layer}")

        # Build AudioSource if audio file is provided
        audio_source = None
        if pipeline_config.audio is not None:
            typer.echo("Analyzing audio...")
            audio_source = AudioSource(pipeline_config.audio)

        # Run the pipeline
        pipeline = Pipeline(
            rules=rules,
            audio_source=audio_source,
            checkpoint_dir=pipeline_config.checkpoint_dir,
        )

        typer.echo("\nStarting pipeline...")
        output = pipeline.process_video(
            input_path=pipeline_config.input,
            output_path=pipeline_config.output,
            audio_path=pipeline_config.audio,
            frame_range=pipeline_config.frame_range,
            output_frames_dir=pipeline_config.output_frames_dir,
        )
        typer.echo(f"\nDone! Output: {output}")

    except Exception as e:
        # Display error message
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("list-rules")
def list_rules() -> None:
    """List all available EffectRule classes.

    Shows the rule name and a brief description if available.
    """
    # Discover all rules (already done at import time, but call for completeness)
    discover_rules()

    registry = get_registry()

    if not registry:
        typer.echo("No rules registered.")
        raise typer.Exit(0)

    typer.echo(f"Available rules ({len(registry)}):")
    typer.echo("")

    # Sort rule names alphabetically
    for name in sorted(registry.keys()):
        rule_cls = registry[name]
        typer.echo(f"  {name}")

        # Try to get docstring
        if rule_cls.__doc__:
            # Get first line of docstring
            doc = rule_cls.__doc__.strip().split("\n")[0]
            typer.echo(f"    {doc}")


def main() -> None:
    """Entry point for `python -m my_amv`."""
    app()


if __name__ == "__main__":
    main()
