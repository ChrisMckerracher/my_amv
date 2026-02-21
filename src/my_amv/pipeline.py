"""Pipeline orchestrates rules and video I/O.

This module provides the Pipeline class which coordinates the application
of EffectRule objects to video frames, handling layer routing, scene cut
detection, checkpointing, and resume functionality.

ROUTING INVARIANT (from AGENTS.md):
- If rule.output_layer is None → inline: result replaces current frame
- If rule.output_layer is not None → branch: result goes to layer store
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from my_amv.audio import AudioSource
from my_amv.context import AudioFrame, FrameContext
from my_amv.rule import EffectRule
from my_amv.types import Layer, LayerKey, RGBArray
from my_amv.video_io import VideoReader, VideoWriter


@dataclass
class SceneCutConfig:
    """Configuration for scene cut detection.

    Attributes:
        enabled: Whether scene cut detection is enabled
        threshold: Frame difference threshold [0, 1] - higher = less sensitive
        min_segment_length: Minimum frames between scene cuts
    """

    enabled: bool = True
    threshold: float = 0.3
    min_segment_length: int = 15


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing.

    Attributes:
        enabled: Whether checkpointing is enabled
        dir: Directory to save checkpoints
        save_frames: Whether to save frame PNGs
        save_state: Whether to save rule state JSON
    """

    enabled: bool = False
    dir: Path | None = None
    save_frames: bool = True
    save_state: bool = True


@dataclass
class ProcessResult:
    """Result of video processing.

    Attributes:
        output_path: Path to the output video
        frames_processed: Number of frames processed
        scene_cuts: Frame indices where scene cuts were detected
        checkpoints_created: Number of checkpoint files created
        resumed_from: Checkpoint index resumed from (0 if no resume)
    """

    output_path: Path
    frames_processed: int
    scene_cuts: list[int] = field(default_factory=list)
    checkpoints_created: int = 0
    resumed_from: int = 0


class Pipeline:
    """Orchestrates the application of EffectRule objects to video frames.

    The Pipeline manages:
    - Rule chaining with inline vs branch layer routing
    - Frame context construction with audio synchronization
    - Scene cut detection and state reset
    - Checkpointing and resume for long-running jobs

    Example:
        pipeline = Pipeline(
            rules=[rule1, rule2, rule3],
            audio_source=audio_source,
        )
        pipeline.process_video(
            input_path=Path("input.mp4"),
            output_path=Path("output.mp4"),
        )
    """

    def __init__(
        self,
        rules: list[EffectRule],
        audio_source: AudioSource | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        """Initialize the Pipeline.

        Args:
            rules: List of EffectRule objects to apply in order
            audio_source: Optional AudioSource for audio-reactive effects
            checkpoint_dir: Optional directory for checkpoint/resume support
        """
        self.rules = rules
        self.audio_source = audio_source

        # Configure checkpointing
        self.checkpoint = CheckpointConfig(
            enabled=checkpoint_dir is not None,
            dir=checkpoint_dir,
        )

        # Configure scene cut detection
        self.scene_cut = SceneCutConfig()

    def process_frame(
        self,
        frame: RGBArray,
        frame_index: int,
        frame_count: int,
        fps: float,
    ) -> RGBArray:
        """Process a single frame through all rules.

        This method:
        1. Builds a FrameContext with Layer.ORIGINAL set
        2. Applies each rule in order
        3. Handles inline vs branch routing based on output_layer
        4. Returns the final frame

        Args:
            frame: Input RGB array (H, W, 3)
            frame_index: Current frame index (0-based)
            frame_count: Total number of frames
            fps: Video frame rate

        Returns:
            Processed RGB array
        """
        # Get audio frame if available
        audio_frame: AudioFrame | None = None
        if self.audio_source is not None:
            audio_frame = self.audio_source.get_frame(frame_index)

        # Build context with ORIGINAL layer set
        context = FrameContext(
            frame_index=frame_index,
            frame_count=frame_count,
            fps=fps,
            audio_frame=audio_frame,
        )
        context.set_layer(Layer.ORIGINAL, frame.copy())

        # Start with the original frame as current
        current_frame = frame.copy()

        # Apply each rule in order
        for rule in self.rules:
            # Get input frame for this rule
            if rule.input_layer == Layer.MAIN:
                input_frame = current_frame
            else:
                input_frame = context.get_layer(rule.input_layer)

            # Apply the rule
            output_frame, context = rule.apply(input_frame, context)

            # Handle routing based on output_layer
            if rule.output_layer is None:
                # Inline: result replaces current frame
                current_frame = output_frame
            else:
                # Branch: result goes to layer store, current frame unchanged
                context.set_layer(rule.output_layer, output_frame)

        return current_frame

    def process_image(self, input_path: Path) -> dict[LayerKey, RGBArray]:
        """Process a single image and return all layers.

        This is useful for:
        - Previewing the effect on a still image
        - Debugging layer composition
        - Generating thumbnails

        Args:
            input_path: Path to input image

        Returns:
            Dictionary of all layer names to RGBArrays
        """
        from PIL import Image

        # Load image
        img = Image.open(input_path).convert("RGB")
        frame = np.array(img).astype(np.uint8)

        # Create a minimal context for single image processing
        context = FrameContext(
            frame_index=0,
            frame_count=1,
            fps=30.0,
        )
        context.set_layer(Layer.ORIGINAL, frame)

        current_frame = frame.copy()

        # Apply each rule
        for rule in self.rules:
            if rule.input_layer == Layer.MAIN:
                input_frame = current_frame
            else:
                input_frame = context.get_layer(rule.input_layer)

            output_frame, context = rule.apply(input_frame, context)

            if rule.output_layer is None:
                current_frame = output_frame
            else:
                context.set_layer(rule.output_layer, output_frame)

        # Add final frame to layers
        context.set_layer(Layer.FINAL, current_frame)

        # Return all layers
        return context._layers

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        audio_path: Path | None = None,
        frame_range: tuple[int, int] | None = None,
        output_frames_dir: Path | None = None,
    ) -> Path:
        """Process a video file through all rules.

        This method:
        1. Extracts frames from the input video
        2. Processes each frame through the rule chain
        3. Detects scene cuts and resets rule state
        4. Saves checkpoints if enabled
        5. Reassembles the output video
        6. Muxes audio if provided

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            audio_path: Optional path to separate audio file for muxing
            frame_range: Optional (start, end) frame range to process
            output_frames_dir: Optional directory to save individual frames

        Returns:
            Path to the output video file

        Raises:
            FileNotFoundError: If input video doesn't exist
            ValueError: If video format is unsupported
        """
        from PIL import Image

        # Ensure output parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get video metadata
        reader = VideoReader()
        metadata = reader.get_metadata(input_path)
        fps = metadata["fps"]
        frame_count = metadata["frame_count"]

        # Determine frame range
        if frame_range is None:
            start_frame, end_frame = 0, frame_count
        else:
            start_frame, end_frame = frame_range

        # Setup checkpointing
        start_from_frame = 0
        if self.checkpoint.enabled and self.checkpoint.dir:
            start_from_frame = self._find_resume_checkpoint(
                self.checkpoint.dir, start_frame, end_frame
            )

        # Setup output frames directory
        if output_frames_dir is None:
            frames_dir = output_path.parent / f"{output_path.stem}_frames"
        else:
            frames_dir = output_frames_dir

        frames_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames if not resuming
        extracted_frames_dir = frames_dir / "extracted"
        if start_from_frame == start_frame:
            reader.extract_frames(
                input_path,
                extracted_frames_dir,
                frame_range=(start_frame, end_frame),
            )

        # Process each frame
        scene_cuts: list[int] = []
        previous_frame: RGBArray | None = None

        for frame_index in range(start_from_frame, end_frame):
            # Load frame
            frame_path = extracted_frames_dir / f"frame_{frame_index:06d}.png"
            frame = np.array(Image.open(frame_path))

            # Detect scene cut
            if self.scene_cut.enabled and previous_frame is not None:
                diff = self._compute_frame_diff(previous_frame, frame)
                if diff > self.scene_cut.threshold:
                    scene_cuts.append(frame_index)
                    self._reset_all_rules(frame, frame_index, frame_count, fps)

            # Process frame
            processed = self.process_frame(frame, frame_index, frame_count, fps)

            # Save processed frame
            output_frame_path = frames_dir / f"frame_{frame_index:06d}.png"
            Image.fromarray(processed).save(output_frame_path)

            # Save checkpoint
            if self.checkpoint.enabled and self.checkpoint.dir:
                self._save_checkpoint(
                    frame_index,
                    output_frame_path,
                    scene_cuts,
                )

            previous_frame = processed

        # Assemble output video
        writer = VideoWriter()
        writer.assemble_video(frames_dir, output_path, fps)

        # Mux audio if provided
        if audio_path is not None:
            temp_output = output_path.parent / f"{output_path.stem}_no_audio.mp4"
            output_path.rename(temp_output)
            writer.mux_audio(temp_output, audio_path, output_path)
            temp_output.unlink()

        return output_path

    def _compute_frame_diff(self, frame1: RGBArray, frame2: RGBArray) -> float:
        """Compute normalized difference between two frames.

        Uses mean absolute difference normalized to [0, 1].

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Difference value in [0, 1], where 1 = completely different
        """
        diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
        return float(diff.mean() / 255.0)

    def _reset_all_rules(
        self,
        frame: RGBArray,
        frame_index: int,
        frame_count: int,
        fps: float,
    ) -> None:
        """Reset all rules' inter-frame state.

        Called at scene boundaries to clear temporal state.

        Args:
            frame: Current frame
            frame_index: Current frame index
            frame_count: Total frame count
            fps: Frame rate
        """
        for rule in self.rules:
            context = FrameContext(
                frame_index=frame_index,
                frame_count=frame_count,
                fps=fps,
            )
            rule.reset_state(context)

    def _find_resume_checkpoint(
        self,
        checkpoint_dir: Path,
        start_frame: int,
        end_frame: int,
    ) -> int:
        """Find the latest checkpoint to resume from.

        Searches for the highest-numbered checkpoint file within the frame range.

        Args:
            checkpoint_dir: Directory containing checkpoints
            start_frame: Start of frame range
            end_frame: End of frame range

        Returns:
            Frame index to resume from (or start_frame if no checkpoint found)
        """
        if not checkpoint_dir.exists():
            return start_frame

        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoint_files:
            return start_frame

        # Find highest frame index with a checkpoint
        max_frame = start_frame
        for cf in checkpoint_files:
            try:
                frame_num = int(cf.stem.split("_")[1])
                if start_frame <= frame_num < end_frame and frame_num > max_frame:
                    max_frame = frame_num
            except (ValueError, IndexError):
                continue

        if max_frame > start_frame:
            # Restore state from checkpoint
            self._restore_checkpoint(max_frame)
            return max_frame + 1

        return start_frame

    def _save_checkpoint(
        self,
        frame_index: int,
        frame_path: Path,
        scene_cuts: list[int],
    ) -> None:
        """Save a checkpoint for the current frame.

        Args:
            frame_index: Current frame index
            frame_path: Path to the processed frame
            scene_cuts: List of scene cut indices detected so far
        """
        if not self.checkpoint.dir:
            return

        self.checkpoint.dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self.checkpoint.dir / f"checkpoint_{frame_index:06d}.json"

        # Collect rule states
        rule_states: dict[str, dict] = {}
        if self.checkpoint.save_state:
            for rule in self.rules:
                context = FrameContext(
                    frame_index=frame_index,
                    frame_count=1,
                    fps=30.0,
                )
                rule_states[rule.name()] = rule.serialize_state(context)

        checkpoint_data = {
            "frame_index": frame_index,
            "frame_path": str(frame_path),
            "scene_cuts": scene_cuts,
            "rule_states": rule_states,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def _restore_checkpoint(self, frame_index: int) -> None:
        """Restore pipeline state from a checkpoint.

        Args:
            frame_index: Frame index of the checkpoint to restore
        """
        if not self.checkpoint.dir:
            return

        checkpoint_path = self.checkpoint.dir / f"checkpoint_{frame_index:06d}.json"

        if not checkpoint_path.exists():
            return

        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        # Restore rule states
        if "rule_states" in checkpoint_data and self.checkpoint.save_state:
            for rule in self.rules:
                rule_name = rule.name()
                if rule_name in checkpoint_data["rule_states"]:
                    context = FrameContext(
                        frame_index=frame_index,
                        frame_count=1,
                        fps=30.0,
                    )
                    state = checkpoint_data["rule_states"][rule_name]
                    rule.deserialize_state(context, state)

    def set_scene_cut_config(self, config: SceneCutConfig) -> None:
        """Configure scene cut detection.

        Args:
            config: SceneCutConfig with detection parameters
        """
        self.scene_cut = config

    def set_checkpoint_config(self, config: CheckpointConfig) -> None:
        """Configure checkpointing.

        Args:
            config: CheckpointConfig with checkpoint parameters
        """
        self.checkpoint = config
