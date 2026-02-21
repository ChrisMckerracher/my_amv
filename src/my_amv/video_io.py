"""Video frame extraction and reassembly.

This module provides VideoReader and VideoWriter classes for:
- Extracting frames from video files
- Assembling frames back into video files
- Multiplexing audio into video files

Uses FFmpeg for all video operations, ensuring consistent behavior
across different video formats and codecs.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from my_amv.types import RGBArray


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file.

    Attributes:
        fps: Frame rate in frames per second
        frame_count: Total number of frames
        width: Video width in pixels
        height: Video height in pixels
        duration: Video duration in seconds
        codec: Video codec name
    """

    fps: float
    frame_count: int
    width: int
    height: int
    duration: float
    codec: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "codec": self.codec,
        }


class VideoReader:
    """Read video files and extract frames.

    Uses FFmpeg to:
    - Get video metadata
    - Extract frames as PNG images
    - Handle various video formats (MP4, MOV, etc.)

    Example:
        reader = VideoReader()
        metadata = reader.get_metadata(Path("video.mp4"))
        frames = reader.extract_frames(
            Path("video.mp4"),
            Path("frames/"),
            frame_range=(0, 100),
        )
    """

    # Supported video formats
    SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    # FFmpeg probe command timeout in seconds
    PROBE_TIMEOUT = 30

    # Frame extraction timeout per frame
    EXTRACT_TIMEOUT_PER_FRAME = 1

    def __init__(self, ffmpeg_path: str | None = None, ffprobe_path: str | None = None) -> None:
        """Initialize VideoReader.

        Args:
            ffmpeg_path: Optional path to ffmpeg executable
            ffprobe_path: Optional path to ffprobe executable

        Raises:
            RuntimeError: If ffmpeg or ffprobe is not available
        """
        self.ffmpeg_path = ffmpeg_path or self._find_executable("ffmpeg")
        self.ffprobe_path = ffprobe_path or self._find_executable("ffprobe")

        if self.ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found - please install ffmpeg")

        if self.ffprobe_path is None:
            raise RuntimeError("ffprobe not found - please install ffmpeg")

    def _find_executable(self, name: str) -> str | None:
        """Find an executable in PATH.

        Args:
            name: Name of the executable

        Returns:
            Path to the executable, or None if not found
        """
        return shutil.which(name)

    def get_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            VideoMetadata object with video information

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video format is not supported
            RuntimeError: If ffprobe fails to parse the video
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        suffix = video_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Run ffprobe to get metadata
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.PROBE_TIMEOUT,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed: {e.stderr}") from e
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffprobe timeout after {self.PROBE_TIMEOUT}s") from None

        # Parse JSON output
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}") from e

        # Extract stream data
        if "streams" not in data or not data["streams"]:
            raise RuntimeError("No video stream found in file")

        stream = data["streams"][0]

        # Parse frame rate (may be "30000/1001" format)
        r_frame_rate = stream.get("r_frame_rate", "30/1")
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(r_frame_rate)

        # Get other metadata
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        codec = stream.get("codec_name", "unknown")

        if width == 0 or height == 0:
            raise RuntimeError("Invalid video dimensions")

        # Get duration
        if "format" not in data or "duration" not in data["format"]:
            raise RuntimeError("No duration information found")

        duration = float(data["format"]["duration"])

        # Calculate frame count
        frame_count = int(duration * fps)

        return VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration=duration,
            codec=codec,
        )

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        frame_range: tuple[int, int] | None = None,
        step: int = 1,
    ) -> list[Path]:
        """Extract frames from a video file as PNG images.

        Args:
            video_path: Path to the input video file
            output_dir: Directory to save extracted frames
            frame_range: Optional (start, end) frame range (inclusive start, exclusive end)
            step: Extract every Nth frame (default: 1)

        Returns:
            List of paths to extracted frame images

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video format is not supported or parameters are invalid
            RuntimeError: If frame extraction fails
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        suffix = video_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")

        # Get metadata to check video validity
        metadata = self.get_metadata(video_path)

        if metadata.frame_count == 0:
            raise ValueError("Video file contains no frames")

        # Determine frame range
        if frame_range is None:
            start_frame, end_frame = 0, metadata.frame_count
        else:
            start_frame, end_frame = frame_range

        if start_frame < 0 or end_frame > metadata.frame_count:
            raise ValueError(
                f"Frame range {frame_range} out of bounds [0, {metadata.frame_count}]"
            )

        if start_frame >= end_frame:
            raise ValueError(
                f"Invalid frame range: start ({start_frame}) >= end ({end_frame})"
            )

        # Check disk space
        self._check_disk_space(output_dir, metadata, start_frame, end_frame, step)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        # Use -vf select and -vsync 0 for precise frame extraction
        output_pattern = str(output_dir / "frame_%06d.png")

        cmd = [
            self.ffmpeg_path,
            "-v", "error",
            "-i", str(video_path),
        ]

        # Add frame range and step filters
        if start_frame > 0 or step > 1:
            # select filter: 'not(mod(n,step))' selects every step-th frame
            # setpts filter adjusts timestamps
            select_expr = f"gte(n,{start_frame})"
            if step > 1:
                select_expr = f"({select_expr})*not(mod(n-{start_frame},{step}))"
            cmd.extend(["-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB"])

        # Output settings
        cmd.extend([
            "-vsync", "0",
            "-start_number", str(start_frame),
            "-frames:v", str((end_frame - start_frame + step - 1) // step),
            output_pattern,
        ])

        timeout = (end_frame - start_frame) * self.EXTRACT_TIMEOUT_PER_FRAME + 30

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr}") from e
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg timeout after {timeout}s") from None

        # List extracted frames
        frame_files = sorted(output_dir.glob("frame_*.png"))

        if not frame_files:
            raise RuntimeError("No frames were extracted")

        return frame_files

    def _check_disk_space(
        self,
        output_dir: Path,
        metadata: VideoMetadata,
        start_frame: int,
        end_frame: int,
        step: int,
    ) -> None:
        """Check if there's enough disk space for extracted frames.

        Args:
            output_dir: Directory where frames will be saved
            metadata: Video metadata
            start_frame: Start frame index
            end_frame: End frame index
            step: Frame step

        Raises:
            RuntimeError: If disk space is insufficient
        """
        import psutil

        try:
            disk_usage = psutil.disk_usage(output_dir.anchor)
            available_bytes = disk_usage.free
        except (OSError, AttributeError):
            # Can't check disk space - proceed anyway
            return

        # Estimate frame size (uncompressed PNG)
        # Rough estimate: width * height * 3 bytes * 1.5 for PNG overhead
        estimated_frame_size = metadata.width * metadata.height * 3 * 1.5

        frame_count = (end_frame - start_frame + step - 1) // step
        estimated_total = estimated_frame_size * frame_count

        # Warn if using more than 90% of available space
        if estimated_total > available_bytes * 0.9:
            free_mb = available_bytes / (1024 * 1024)
            needed_mb = estimated_total / (1024 * 1024)
            raise RuntimeError(
                f"Insufficient disk space: {free_mb:.0f}MB free, "
                f"estimated {needed_mb:.0f}MB needed for {frame_count} frames"
            )


class VideoWriter:
    """Assemble frames into video files and mux audio.

    Uses FFmpeg to:
    - Assemble numbered PNG images into a video
    - Multiplex audio into a video file
    - Handle various output formats and codecs

    Example:
        writer = VideoWriter()
        writer.assemble_video(
            frames_dir=Path("frames/"),
            output_path=Path("output.mp4"),
            fps=30.0,
        )
        writer.mux_audio(
            video_path=Path("output.mp4"),
            audio_path=Path("audio.mp3"),
            output_path=Path("final.mp4"),
        )
    """

    # Default codec for MP4 output
    DEFAULT_CODEC = "libx264"

    # FFmpeg command timeout factor (seconds per frame)
    TIMEOUT_PER_FRAME = 0.5

    def __init__(self, ffmpeg_path: str | None = None) -> None:
        """Initialize VideoWriter.

        Args:
            ffmpeg_path: Optional path to ffmpeg executable

        Raises:
            RuntimeError: If ffmpeg is not available
        """
        self.ffmpeg_path = ffmpeg_path or self._find_executable("ffmpeg")

        if self.ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found - please install ffmpeg")

    def _find_executable(self, name: str) -> str | None:
        """Find an executable in PATH."""
        return shutil.which(name)

    def assemble_video(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float,
        codec: str = "libx264",
        crf: int = 23,
    ) -> Path:
        """Assemble numbered PNG frames into a video file.

        Args:
            frames_dir: Directory containing numbered frame_XXXXXX.png files
            output_path: Path for the output video file
            fps: Frame rate in frames per second
            codec: Video codec (default: libx264)
            crf: Constant Rate Factor for quality (default: 23, lower = better)

        Returns:
            Path to the output video file

        Raises:
            FileNotFoundError: If frames directory doesn't exist or is empty
            ValueError: If parameters are invalid
            RuntimeError: If video assembly fails
        """
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        # Find frame files
        frame_files = sorted(frames_dir.glob("frame_*.png"))

        if not frame_files:
            raise FileNotFoundError(f"No frame files found in {frames_dir}")

        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        # Create output parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        input_pattern = str(frames_dir / "frame_%06d.png")

        cmd = [
            self.ffmpeg_path,
            "-v", "error",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", codec,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-y",  # Overwrite output file
            str(output_path),
        ]

        timeout = len(frame_files) * self.TIMEOUT_PER_FRAME + 60

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr}") from e
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg timeout after {timeout}s") from None

        return output_path

    def mux_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> Path:
        """Multiplex audio into a video file.

        The audio is copied without re-encoding (-c:a copy) and truncated
        to match the video duration.

        Args:
            video_path: Path to the input video file
            audio_path: Path to the audio file
            output_path: Path for the output video file

        Returns:
            Path to the output video file

        Raises:
            FileNotFoundError: If input files don't exist
            RuntimeError: If muxing fails
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create output parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        # -shortest truncates the longer stream to match the shorter
        cmd = [
            self.ffmpeg_path,
            "-v", "error",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-c:a", "aac",  # Encode audio to AAC (widely compatible)
            "-shortest",  # Truncate to the shorter stream
            "-y",  # Overwrite output file
            str(output_path),
        ]

        timeout = 300  # 5 minutes default

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr}") from e
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg timeout after {timeout}s") from None

        return output_path
