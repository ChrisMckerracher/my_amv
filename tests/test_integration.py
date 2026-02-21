"""End-to-end integration tests for the my_amv pipeline.

These tests exercise real rule chains with synthetic data (no heavy ML models).
They verify the full pipeline path: input -> rules -> frame processing -> output.

Marked with pytest.mark.integration and skipped if ffmpeg is unavailable.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from my_amv.pipeline import Pipeline
from my_amv.rules.edge_detection import EdgeDetectionRule
from my_amv.rules.temporal_smooth import TemporalSmoothRule
from my_amv.types import Layer

# Skip entire module if ffmpeg is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None,
        reason="ffmpeg not found on PATH",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_frame(
    width: int = 256,
    height: int = 256,
    value: int = 128,
    draw_shape: bool = True,
) -> np.ndarray:
    """Create a synthetic RGB frame with an optional centered rectangle."""
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    if draw_shape:
        y1, y2 = height // 4, 3 * height // 4
        x1, x2 = width // 4, 3 * width // 4
        frame[y1:y2, x1:x2] = 255
    return frame


def _write_synthetic_video(
    path: Path,
    num_frames: int = 30,
    width: int = 256,
    height: int = 256,
    fps: float = 30.0,
) -> Path:
    """Write a synthetic MP4 video using ffmpeg (pipe raw frames in).

    Uses num_frames that are multiples of fps to avoid duration-rounding issues.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for i in range(num_frames):
        base_val = 60 + (i * 5) % 180
        frame = _make_synthetic_frame(width, height, value=base_val, draw_shape=True)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0, f"ffmpeg failed: {proc.stderr.read().decode()}"
    assert path.exists() and path.stat().st_size > 0
    return path


def _write_synthetic_wav(path: Path, duration: float = 1.0, sr: int = 22050) -> Path:
    """Write a synthetic sine-tone WAV file."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), signal, sr)
    assert path.exists()
    return path


# ---------------------------------------------------------------------------
# Test 1 – Image pipeline
# ---------------------------------------------------------------------------

class TestImagePipeline:
    """Process a single synthetic image through a rule chain."""

    def test_image_pipeline_produces_output(self, tmp_path: Path) -> None:
        """EdgeDetection + TemporalSmooth on a single PNG -> correct shape."""
        img = _make_synthetic_frame(256, 256, value=128, draw_shape=True)
        input_path = tmp_path / "input.png"
        Image.fromarray(img).save(input_path)

        edge_rule = EdgeDetectionRule(algorithm="canny", detail_level=5)
        edge_rule.output_layer = None  # inline
        smooth_rule = TemporalSmoothRule(blend_factor=0.7)

        pipeline = Pipeline(rules=[edge_rule, smooth_rule])
        layers = pipeline.process_image(input_path)

        assert Layer.FINAL in layers
        final = layers[Layer.FINAL]
        assert final.shape == (256, 256, 3)
        assert final.dtype == np.uint8

        assert Layer.ORIGINAL in layers
        np.testing.assert_array_equal(layers[Layer.ORIGINAL], img)


# ---------------------------------------------------------------------------
# Test 2 – Short video pipeline (no audio)
# ---------------------------------------------------------------------------

class TestVideoPipelineNoAudio:
    """Process a synthetic video through EdgeDetect + TemporalSmooth."""

    def test_video_pipeline_produces_mp4(self, tmp_path: Path) -> None:
        """30-frame video -> processed frames + valid MP4 output."""
        video_path = _write_synthetic_video(
            tmp_path / "input.mp4", num_frames=30, fps=30.0,
        )
        output_path = tmp_path / "output.mp4"

        edge_rule = EdgeDetectionRule(algorithm="canny", detail_level=5)
        edge_rule.output_layer = None  # inline
        smooth_rule = TemporalSmoothRule(blend_factor=0.7)

        pipeline = Pipeline(rules=[edge_rule, smooth_rule])
        result = pipeline.process_video(
            input_path=video_path, output_path=output_path,
        )

        # Output file exists and is a valid video
        assert result.exists()
        assert result.stat().st_size > 0

        # Processed frames directory has the expected PNGs
        frames_dir = output_path.parent / "output_frames"
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        assert len(frame_files) >= 28  # allow small ffprobe rounding

        # Consecutive frames should differ (temporal smooth carries state)
        f0 = np.array(Image.open(frame_files[0]))
        f1 = np.array(Image.open(frame_files[1]))
        assert not np.array_equal(f0, f1), "Consecutive frames should differ"


# ---------------------------------------------------------------------------
# Test 3 – Video + audio muxing
# ---------------------------------------------------------------------------

class TestVideoAudioMuxing:
    """Process video and mux a separate audio track into the output."""

    def test_output_mp4_has_audio_track(self, tmp_path: Path) -> None:
        """Output MP4 should contain an audio stream after muxing."""
        video_path = _write_synthetic_video(
            tmp_path / "input.mp4", num_frames=30, fps=30.0,
        )
        audio_path = _write_synthetic_wav(tmp_path / "audio.wav", duration=1.0)
        output_path = tmp_path / "output.mp4"

        edge_rule = EdgeDetectionRule(algorithm="canny", detail_level=5)
        edge_rule.output_layer = None

        pipeline = Pipeline(rules=[edge_rule])
        result = pipeline.process_video(
            input_path=video_path,
            output_path=output_path,
            audio_path=audio_path,
        )

        # Probe output streams
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(result),
            ],
            capture_output=True, text=True,
        )
        streams = probe.stdout.strip().splitlines()
        assert "video" in streams, "Output should have a video stream"
        assert "audio" in streams, "Output should have an audio stream"


# ---------------------------------------------------------------------------
# Test 4 – Checkpointing + resume
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    """Verify that processing can be interrupted and resumed."""

    def test_checkpoint_and_resume(self, tmp_path: Path) -> None:
        """Process first half, stop, resume – all frames in final output."""
        video_path = _write_synthetic_video(
            tmp_path / "input.mp4", num_frames=30, fps=30.0,
        )
        checkpoint_dir = tmp_path / "checkpoints"
        output_path = tmp_path / "output.mp4"
        frames_dir = tmp_path / "output_frames"

        edge_rule = EdgeDetectionRule(algorithm="canny", detail_level=5)
        edge_rule.output_layer = None

        # --- Pass 1: process only frames 0-14 (first half) ---
        pipeline1 = Pipeline(rules=[edge_rule], checkpoint_dir=checkpoint_dir)
        pipeline1.process_video(
            input_path=video_path,
            output_path=output_path,
            output_frames_dir=frames_dir,
            frame_range=(0, 15),
        )

        # Verify checkpoint files exist
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))
        assert len(checkpoint_files) == 15

        # Record mtimes for first-pass frames
        first_pass_mtimes = {}
        for i in range(15):
            fp = frames_dir / f"frame_{i:06d}.png"
            assert fp.exists(), f"frame {i} should exist after first pass"
            first_pass_mtimes[i] = fp.stat().st_mtime

        # --- Pass 2: resume, process remaining frames ---
        edge_rule2 = EdgeDetectionRule(algorithm="canny", detail_level=5)
        edge_rule2.output_layer = None

        pipeline2 = Pipeline(rules=[edge_rule2], checkpoint_dir=checkpoint_dir)
        pipeline2.process_video(
            input_path=video_path,
            output_path=output_path,
            output_frames_dir=frames_dir,
        )

        # All frames present
        all_frames = sorted(frames_dir.glob("frame_*.png"))
        assert len(all_frames) >= 28  # allow small ffprobe rounding

        # Frames 0-14 should NOT have been reprocessed (mtime unchanged)
        for i in range(15):
            fp = frames_dir / f"frame_{i:06d}.png"
            assert fp.stat().st_mtime == first_pass_mtimes[i], (
                f"frame {i} was reprocessed on resume – should have been skipped"
            )

        # Final output video exists
        assert output_path.exists()
        assert output_path.stat().st_size > 0
