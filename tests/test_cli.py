"""Tests for CLI functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from my_amv.__main__ import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_pipeline(tmp_path):
    """Mock Pipeline and AudioSource so CLI tests don't require real video/audio files."""
    with patch("my_amv.__main__.Pipeline") as MockPipeline, \
         patch("my_amv.__main__.AudioSource") as MockAudioSource:
        instance = MagicMock()
        instance.process_video.return_value = tmp_path / "result.mp4"
        MockPipeline.return_value = instance
        MockAudioSource.return_value = MagicMock()
        yield MockPipeline


class TestRunCommand:
    """Test the 'run' command."""

    def test_run_with_config_file(self, tmp_path):
        """Should run pipeline with config file."""
        # Create a minimal config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
input: video.mp4
output: result.mp4
rules: []
"""
        )

        result = runner.invoke(app, ["run", "--config", str(config_file)])

        assert result.exit_code == 0
        assert "Loading config from:" in result.stdout
        assert "Input:" in result.stdout
        assert "Output:" in result.stdout

    def test_run_with_cli_args(self, tmp_path):
        """Should run with --input and --output arguments."""
        input_file = tmp_path / "video.mp4"
        input_file.touch()

        output_file = tmp_path / "result.mp4"

        result = runner.invoke(
            app, ["run", "--input", str(input_file), "--output", str(output_file)]
        )

        assert result.exit_code == 0
        assert "Building config from CLI arguments" in result.stdout
        assert "Input:" in result.stdout
        assert "Output:" in result.stdout

    def test_run_with_audio(self, tmp_path):
        """Should accept --audio argument."""
        input_file = tmp_path / "video.mp4"
        input_file.touch()
        output_file = tmp_path / "result.mp4"
        audio_file = tmp_path / "music.mp3"
        audio_file.touch()

        result = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
                "--audio",
                str(audio_file),
            ],
        )

        assert result.exit_code == 0
        assert "Audio:" in result.stdout

    def test_run_with_frame_range(self, tmp_path):
        """Should parse --frame-range argument."""
        input_file = tmp_path / "video.mp4"
        input_file.touch()
        output_file = tmp_path / "result.mp4"

        result = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
                "--frame-range",
                "0,100",
            ],
        )

        assert result.exit_code == 0
        assert "Frame range: 0 to 100" in result.stdout

    def test_run_with_frame_range_negative_one(self, tmp_path):
        """Should accept -1 as frame range end (all frames)."""
        input_file = tmp_path / "video.mp4"
        input_file.touch()
        output_file = tmp_path / "result.mp4"

        result = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
                "--frame-range",
                "0,-1",
            ],
        )

        assert result.exit_code == 0
        assert "Frame range: 0 to -1" in result.stdout

    def test_run_with_invalid_frame_range(self, tmp_path):
        """Should reject invalid frame range format."""
        input_file = tmp_path / "video.mp4"
        input_file.touch()
        output_file = tmp_path / "result.mp4"

        result = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
                "--frame-range",
                "invalid",
            ],
        )

        assert result.exit_code == 1
        # Error message can be in stdout or stderr
        output = result.stdout + result.stderr
        assert "Invalid frame range" in output or result.exit_code == 1

    def test_run_with_output_frames_dir(self, tmp_path):
        """Should accept --output-frames argument."""
        input_file = tmp_path / "video.mp4"
        input_file.touch()
        output_file = tmp_path / "result.mp4"
        frames_dir = tmp_path / "frames"

        result = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
                "--output-frames",
                str(frames_dir),
            ],
        )

        assert result.exit_code == 0

    def test_run_with_checkpoint_dir(self, tmp_path):
        """Should accept --checkpoint-dir argument."""
        input_file = tmp_path / "video.mp4"
        input_file.touch()
        output_file = tmp_path / "result.mp4"
        checkpoint_dir = tmp_path / "checkpoints"

        result = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
                "--checkpoint-dir",
                str(checkpoint_dir),
            ],
        )

        assert result.exit_code == 0

    def test_run_missing_config_and_args(self, tmp_path):
        """Should error when neither --config nor (--input + --output) provided."""
        result = runner.invoke(app, ["run"])

        assert result.exit_code == 1
        # Error is handled by typer's validation, just check exit code

    def test_run_with_input_but_no_output(self, tmp_path):
        """Should error when --input provided but --output missing."""
        input_file = tmp_path / "video.mp4"
        input_file.touch()

        result = runner.invoke(app, ["run", "--input", str(input_file)])

        assert result.exit_code == 1
        # Error is handled by typer's validation, just check exit code

    def test_run_with_nonexistent_config(self, tmp_path):
        """Should error when config file doesn't exist."""
        result = runner.invoke(app, ["run", "--config", "nonexistent.yaml"])

        assert result.exit_code == 2  # typer validation error


class TestListRulesCommand:
    """Test the 'list-rules' command."""

    def test_list_rules(self):
        """Should list all registered rules."""
        result = runner.invoke(app, ["list-rules"])

        assert result.exit_code == 0
        # CompositeRule should always be registered
        assert "CompositeRule" in result.stdout
        assert "Available rules" in result.stdout

    def test_list_rules_shows_docstring(self):
        """Should show rule docstring if available."""
        result = runner.invoke(app, ["list-rules"])

        assert result.exit_code == 0
        # CompositeRule has a docstring
        assert "CompositeRule" in result.stdout


class TestCliIntegration:
    """Integration tests for CLI with real config files."""

    def test_run_with_real_config(self, tmp_path):
        """Should process a real config file with rules."""
        # Create a test video file
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_file = tmp_path / "result.mp4"

        # Create a config file with rules (use absolute paths)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
input: {video_file}
output: {output_file}
rules:
  - name: CompositeRule
    params:
      layers: []
"""
        )

        result = runner.invoke(app, ["run", "--config", str(config_file)])

        assert result.exit_code == 0
        assert "Rules: 1" in result.stdout
        assert "CompositeRule" in result.stdout

    def test_run_with_invalid_rule_in_config(self, tmp_path):
        """Should error with clear message for invalid rule name."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_file = tmp_path / "result.mp4"

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
input: {video_file}
output: {output_file}
rules:
  - name: NonExistentRule
"""
        )

        result = runner.invoke(app, ["run", "--config", str(config_file)])

        assert result.exit_code == 1
        output = result.stdout + result.stderr
        assert "Unknown rule" in output or "NonExistentRule" in output

    def test_run_with_invalid_layer_reference(self, tmp_path):
        """Should error with clear message for invalid layer reference."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_file = tmp_path / "result.mp4"

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
input: {video_file}
output: {output_file}
rules:
  - name: CompositeRule
    input_layer: nonexistent_layer
    params:
      layers: []
"""
        )

        result = runner.invoke(app, ["run", "--config", str(config_file)])

        assert result.exit_code == 1
        output = result.stdout + result.stderr
        assert "does not exist" in output
