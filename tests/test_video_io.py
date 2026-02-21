"""Tests for VideoReader and VideoWriter.

Tests use synthetic videos generated with ffmpeg. Tests are skipped
if ffmpeg is not available.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from my_amv.video_io import VideoReader, VideoWriter, VideoMetadata

# Try to import ffmpeg - skip tests if not available
pytest.importorskip("ffmpeg", reason="ffmpeg not available")

# Also check that ffmpeg executable is available
@pytest.fixture(scope="module", autouse=True)
def check_ffmpeg() -> None:
    """Skip all tests if ffmpeg executable is not available."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg executable not found")


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a synthetic 10-frame test video.

    Uses ffmpeg to generate a simple test pattern video.

    Returns:
        Path to the synthetic video file
    """
    video_path = tmp_path / "synthetic.mp4"

    # Use ffmpeg to generate a test video
    # Test source: 10 frames of a color gradient
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", f"color=c=red:size=320x240:duration=0.33:r=30",
        "-vf", "hue=h=0",  # Keep hue constant
        "-c:v", "libx264",
        "-t", "0.33",  # ~10 frames at 30fps
        "-y",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        pytest.skip(f"Failed to create synthetic video: {result.stderr.decode()}")

    return video_path


@pytest.fixture
def synthetic_audio(tmp_path: Path) -> Path:
    """Create a synthetic audio file.

    Uses ffmpeg to generate a simple sine wave audio.

    Returns:
        Path to the synthetic audio file
    """
    audio_path = tmp_path / "synthetic.mp3"

    # Use ffmpeg to generate a sine wave
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=1",
        "-y",
        str(audio_path),
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        pytest.skip(f"Failed to create synthetic audio: {result.stderr.decode()}")

    return audio_path


class TestVideoMetadata:
    """Test VideoMetadata dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metadata = VideoMetadata(
            fps=30.0,
            frame_count=300,
            width=1920,
            height=1080,
            duration=10.0,
            codec="h264",
        )

        result = metadata.to_dict()

        assert result == {
            "fps": 30.0,
            "frame_count": 300,
            "width": 1920,
            "height": 1080,
            "duration": 10.0,
            "codec": "h264",
        }


class TestVideoReader:
    """Test VideoReader class."""

    def test_init(self) -> None:
        """Test VideoReader initialization."""
        reader = VideoReader()

        assert reader.ffmpeg_path is not None
        assert reader.ffprobe_path is not None

    def test_get_metadata(self, synthetic_video: Path) -> None:
        """Test getting video metadata."""
        reader = VideoReader()
        metadata = reader.get_metadata(synthetic_video)

        assert isinstance(metadata, VideoMetadata)
        assert metadata.fps > 0
        assert metadata.frame_count > 0
        assert metadata.width > 0
        assert metadata.height > 0
        assert metadata.duration > 0
        assert isinstance(metadata.codec, str)

    def test_get_metadata_mp4(self, synthetic_video: Path) -> None:
        """Test that MP4 format is supported."""
        reader = VideoReader()
        # Should not raise ValueError
        metadata = reader.get_metadata(synthetic_video)
        assert metadata is not None

    def test_get_metadata_file_not_found(self) -> None:
        """Test get_metadata with non-existent file."""
        reader = VideoReader()

        with pytest.raises(FileNotFoundError):
            reader.get_metadata(Path("/nonexistent/video.mp4"))

    def test_get_metadata_unsupported_format(self, tmp_path: Path) -> None:
        """Test get_metadata with unsupported format."""
        reader = VideoReader()

        # Create a .txt file (not a video)
        txt_file = tmp_path / "not_a_video.txt"
        txt_file.write_text("This is not a video")

        with pytest.raises(ValueError, match="Unsupported video format"):
            reader.get_metadata(txt_file)

    def test_extract_frames(self, synthetic_video: Path, tmp_path: Path) -> None:
        """Test extracting frames from video."""
        reader = VideoReader()
        output_dir = tmp_path / "frames"

        metadata = reader.get_metadata(synthetic_video)
        frame_files = reader.extract_frames(synthetic_video, output_dir)

        assert len(frame_files) > 0
        assert all(f.suffix == ".png" for f in frame_files)
        assert all(f.exists() for f in frame_files)

        # Frame count should be approximately correct (may vary slightly)
        assert abs(len(frame_files) - metadata.frame_count) <= 1

    def test_extract_frames_with_range(self, synthetic_video: Path, tmp_path: Path) -> None:
        """Test extracting a range of frames."""
        reader = VideoReader()
        output_dir = tmp_path / "frames_range"

        # Extract first 5 frames
        frame_files = reader.extract_frames(
            synthetic_video,
            output_dir,
            frame_range=(0, 5),
        )

        assert len(frame_files) <= 5
        assert all(f.exists() for f in frame_files)

    def test_extract_frames_file_not_found(self, tmp_path: Path) -> None:
        """Test extract_frames with non-existent file."""
        reader = VideoReader()
        output_dir = tmp_path / "frames"

        with pytest.raises(FileNotFoundError):
            reader.extract_frames(Path("/nonexistent/video.mp4"), output_dir)

    def test_extract_frames_invalid_range(self, synthetic_video: Path, tmp_path: Path) -> None:
        """Test extract_frames with invalid frame range."""
        reader = VideoReader()
        output_dir = tmp_path / "frames"

        with pytest.raises(ValueError):
            reader.extract_frames(synthetic_video, output_dir, frame_range=(5, 2))

    def test_extract_frames_with_step(self, synthetic_video: Path, tmp_path: Path) -> None:
        """Test extracting every Nth frame."""
        reader = VideoReader()
        output_dir = tmp_path / "frames_step"

        # Extract every 2nd frame
        frame_files = reader.extract_frames(
            synthetic_video,
            output_dir,
            step=2,
        )

        # Should have about half the frames
        metadata = reader.get_metadata(synthetic_video)
        assert len(frame_files) <= (metadata.frame_count + 1) // 2


class TestVideoWriter:
    """Test VideoWriter class."""

    def test_init(self) -> None:
        """Test VideoWriter initialization."""
        writer = VideoWriter()

        assert writer.ffmpeg_path is not None

    def test_assemble_video(self, synthetic_video: Path, tmp_path: Path) -> None:
        """Test assembling frames into a video."""
        # First extract frames
        reader = VideoReader()
        frames_dir = tmp_path / "source_frames"
        reader.extract_frames(synthetic_video, frames_dir)

        # Then reassemble
        writer = VideoWriter()
        output_path = tmp_path / "reassembled.mp4"

        # Get original metadata for fps
        metadata = reader.get_metadata(synthetic_video)

        result = writer.assemble_video(
            frames_dir,
            output_path,
            fps=metadata.fps,
        )

        assert result == output_path
        assert output_path.exists()

        # Verify the output video is valid
        output_metadata = reader.get_metadata(output_path)
        assert output_metadata.fps == metadata.fps

    def test_assemble_video_empty_dir(self, tmp_path: Path) -> None:
        """Test assembling video from empty directory."""
        writer = VideoWriter()
        empty_dir = tmp_path / "empty_frames"
        empty_dir.mkdir()

        output_path = tmp_path / "output.mp4"

        with pytest.raises(FileNotFoundError, match="No frame files found"):
            writer.assemble_video(empty_dir, output_path, fps=30.0)

    def test_assemble_video_dir_not_found(self, tmp_path: Path) -> None:
        """Test assembling video from non-existent directory."""
        writer = VideoWriter()
        nonexistent_dir = tmp_path / "nonexistent_frames"
        output_path = tmp_path / "output.mp4"

        with pytest.raises(FileNotFoundError):
            writer.assemble_video(nonexistent_dir, output_path, fps=30.0)

    def test_mux_audio(
        self,
        synthetic_video: Path,
        synthetic_audio: Path,
        tmp_path: Path,
    ) -> None:
        """Test multiplexing audio into video."""
        writer = VideoWriter()
        output_path = tmp_path / "with_audio.mp4"

        result = writer.mux_audio(synthetic_video, synthetic_audio, output_path)

        assert result == output_path
        assert output_path.exists()

        # Verify the output is valid
        reader = VideoReader()
        metadata = reader.get_metadata(output_path)
        assert metadata.fps > 0

    def test_mux_audio_video_not_found(self, synthetic_audio: Path, tmp_path: Path) -> None:
        """Test mux_audio with non-existent video."""
        writer = VideoWriter()
        output_path = tmp_path / "output.mp4"

        with pytest.raises(FileNotFoundError):
            writer.mux_audio(Path("/nonexistent/video.mp4"), synthetic_audio, output_path)

    def test_mux_audio_audio_not_found(self, synthetic_video: Path, tmp_path: Path) -> None:
        """Test mux_audio with non-existent audio."""
        writer = VideoWriter()
        output_path = tmp_path / "output.mp4"

        with pytest.raises(FileNotFoundError):
            writer.mux_audio(synthetic_video, Path("/nonexistent/audio.mp3"), output_path)
