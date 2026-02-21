Feature: Video Pipeline with Temporal Coherence
  As an artist creating AMV-style music video art,
  I want to process a video file paired with a separate audio file through the pipeline,
  So that I can produce a stylized art video with music, using temporally coherent effects.

  Background:
    Given the video pipeline is installed and configured
    And FFmpeg is available on the system
    And a separate audio file may optionally be provided alongside the video

  # --- Video Input and Frame Extraction ---

  Scenario: Extract frames from an MP4 video file
    Given a video file "dance.mp4" at 30fps with 300 frames
    When I run frame extraction with default settings
    Then 300 individual frames are extracted as PNG files
    And each frame is named with a zero-padded sequential number
    And each frame matches the video's native resolution

  Scenario: Extract frames from a MOV video file
    Given a video file "performance.mov" at 24fps with 240 frames
    When I run frame extraction with default settings
    Then 240 individual frames are extracted as PNG files
    And each frame preserves the original color space

  Scenario: Extract frames at a reduced rate
    Given a video file "dance.mp4" at 30fps with 300 frames
    When I run frame extraction at every 3rd frame
    Then 100 frames are extracted
    And the extracted frames are evenly spaced across the video duration

  Scenario: Extract a specific frame range
    Given a video file "dance.mp4" at 30fps with 300 frames
    When I run frame extraction for frames 60 through 120
    Then 61 frames are extracted
    And the first extracted frame corresponds to frame 60 of the source video
    And the last extracted frame corresponds to frame 120 of the source video

  # --- Frame-by-Frame Processing ---

  Scenario: Process all frames through an EffectRule chain
    Given a video file "dance.mp4" with 300 frames
    And an EffectRule chain configured as [EdgeDetect, DepthComposite]
    When I run the video pipeline
    Then all 300 frames are processed through the rule chain
    And the output directory contains 300 processed frame PNGs

  Scenario: Process video and output as video file
    Given a video file "dance.mp4" at 30fps
    And an EffectRule chain configured as [EdgeDetect]
    When I run the video pipeline with video output enabled
    Then the output is an MP4 file at 30fps
    And the output video has the same duration as the input
    And the output video contains processed frames in sequence

  Scenario: Process video and output both video and individual frames
    Given a video file "dance.mp4" at 30fps
    And an EffectRule chain configured as [SilhouetteExtract]
    When I run the video pipeline with video and frame output enabled
    Then the output includes an MP4 video file
    And the output includes a directory of individual PNG frames
    And the video and frame outputs are identical frame-for-frame

  # --- Temporal Coherence ---

  Scenario: Stateless effects produce consistent output across frames
    Given a video file of a person standing still for 30 frames
    And an EffectRule chain configured as [EdgeDetect] with detail level 5
    When I run the video pipeline
    Then the edge map output is nearly identical across all 30 frames
    And there is no visible flickering when the output is played back

  Scenario: Stateful effects evolve smoothly across frames
    Given a video file "dance.mp4" with 300 frames
    And an EffectRule chain configured as [HalfToneRandomization] with temporal_smoothing 0.8
    When I run the video pipeline
    Then the halftone pattern changes gradually between consecutive frames
    And no consecutive frame pair shows an abrupt pattern discontinuity
    And the output video appears smoothly animated when played at normal speed

  Scenario: Inter-frame state carries forward correctly
    Given a video file with 10 frames
    And a stateful EffectRule that increments a counter per frame
    When I run the video pipeline
    Then the rule's state counter is 1 on frame 1
    And the rule's state counter is 10 on frame 10
    And state was not reset between frames

  Scenario: Scene cut resets inter-frame state
    Given a video file with a scene cut at frame 50
    And a stateful EffectRule with temporal smoothing
    When I run the video pipeline
    Then the rule's inter-frame state is carried forward through frames 1-49
    And the rule's inter-frame state is reset at frame 50
    And no blending artifacts appear across the scene cut boundary

  Scenario: Temporal smoothing blends consecutive frames
    Given a video file of a person with subtle movement over 30 frames
    And an EffectRule chain configured as [TemporalSmooth] with blend_factor 0.7
    When I run the video pipeline
    Then each output frame is a blend of the current and previous processed frames
    And the output is smoother than processing each frame independently
    And the output retains the overall motion of the subject

  # --- Checkpointing and Resume ---

  Scenario: Interrupted processing can be resumed
    Given a video file with 300 frames
    And the pipeline was interrupted after processing 150 frames
    When I resume the video pipeline
    Then processing continues from frame 151
    And frames 1-150 are not reprocessed
    And the final output includes all 300 processed frames

  Scenario: Checkpoint preserves inter-frame state
    Given a video file with 300 frames
    And a stateful EffectRule chain
    And the pipeline was interrupted after processing 150 frames
    When I resume the video pipeline
    Then the EffectRule state is restored from the checkpoint at frame 150
    And frame 151 output is identical to what it would have been without interruption

  # --- Audio Muxing ---

  Scenario: Separate audio file is muxed into output video
    Given a video file "dance.mp4" at 30fps
    And a separate audio file "soundtrack.mp3"
    And an EffectRule chain configured as [EdgeDetect]
    When I run the video pipeline with video output enabled
    Then the output MP4 contains the processed video frames
    And the output MP4 contains the audio from "soundtrack.mp3"
    And the audio is synchronized with the video starting at frame 0

  Scenario: Output video without audio file has no audio track
    Given a video file "dance.mp4" at 30fps
    And no separate audio file is provided
    When I run the video pipeline with video output enabled
    Then the output MP4 contains only the processed video frames
    And the output MP4 has no audio track

  Scenario: Audio longer than video is truncated in output
    Given a video file with 300 frames at 30fps (10 seconds)
    And a separate audio file that is 30 seconds long
    When I run the video pipeline with video output enabled
    Then the output video is 10 seconds long
    And the audio track in the output is truncated to 10 seconds

  Scenario: Audio shorter than video leaves remaining video silent
    Given a video file with 300 frames at 30fps (10 seconds)
    And a separate audio file that is 5 seconds long
    When I run the video pipeline with video output enabled
    Then the output video is 10 seconds long
    And the first 5 seconds have audio
    And the remaining 5 seconds are silent

  Scenario: Audio is passed through without re-encoding
    Given a video file "dance.mp4"
    And a separate audio file "soundtrack.mp3"
    When I run the video pipeline with video output enabled
    Then the audio stream in the output is copied from the source without re-encoding
    And the audio quality is identical to the input audio file

  # --- Output Configuration ---

  Scenario: Output video codec is configurable
    Given a video file "dance.mp4"
    And an EffectRule chain configured as [EdgeDetect]
    When I run the video pipeline with output codec "h265"
    Then the output video is encoded with H.265
    And the output file is a valid MP4

  Scenario: Output frame rate matches input by default
    Given a video file at 24fps
    When I run the video pipeline
    Then the output video is 24fps

  Scenario: Output frame rate can be overridden
    Given a video file at 30fps
    When I run the video pipeline with output frame rate 15
    Then the output video is 15fps
    And only every 2nd processed frame is included in the output video

  # --- Error Handling ---

  Scenario: Unsupported video format produces clear error
    Given a video file "clip.avi" in an unsupported format
    When I attempt to run the video pipeline
    Then the pipeline returns an error listing supported formats as MP4 and MOV
    And no output files are created

  Scenario: Corrupted video file produces clear error
    Given a corrupted video file "broken.mp4" that cannot be decoded
    When I attempt to run the video pipeline
    Then the pipeline returns an error indicating the video could not be decoded
    And the error includes the frame number where decoding failed if applicable

  Scenario: Empty video file produces clear error
    Given a video file with 0 frames
    When I attempt to run the video pipeline
    Then the pipeline returns an error indicating the video contains no frames
    And no output files are created

  Scenario: Insufficient disk space warning
    Given a video file with 10000 frames at 4K resolution
    When I run the video pipeline with frame output enabled
    And available disk space is less than the estimated output size
    Then the pipeline warns about potentially insufficient disk space before processing begins
    And the user can choose to proceed or cancel
