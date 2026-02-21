# Product Brief: Edge Detection & Topology Pipeline for Art

## Problem Statement

An artist working with photographs and video of people wants to extract clean structural representations -- edges, silhouettes, contours, and depth -- to use as foundations for drawing and painting. Today, manually tracing references is slow and imprecise, while automated tools produce noisy, cluttered output that requires significant cleanup before it is usable as an art base layer.

When working with video, the problem compounds: frame-by-frame processing produces flickering, temporally incoherent output where effects jump discontinuously between frames. The artist needs effects that evolve smoothly across time, not ones applied independently to each frame.

The artist needs an AMV (Anime Music Video) art pipeline that takes a video of a person plus a separate audio/music file and produces minimal, noise-free structural outputs -- with temporal coherence and audio-reactive effects -- that can be directly used as under-layers for artistic work. The pipeline must support composable, chainable processing rules ("EffectRules") so the artist can mix and match effects (edge detection, halftone, depth compositing, audio-reactive displacement, etc.) and have them carry state smoothly across video frames, synchronized with the music.

## User Persona

**Name:** Visual artist / illustrator
**Background:** Creates drawings and paintings that reference real people. Works digitally (tablet/desktop) and sometimes prints reference layers for traditional media.
**Pain points:**
- Manual tracing of photo references is tedious and time-consuming
- Existing edge detection tools (e.g., Photoshop "Find Edges") produce too much noise -- texture, background clutter, hair detail -- making the output unusable without heavy cleanup
- Needs the *structure* of a person (pose, silhouette, major contours) without distracting surface detail
- Wants to experiment with depth as an additional artistic dimension (foreground/background separation, layered compositions)
- When processing video, frame-by-frame effects flicker and jump -- there is no temporal coherence, making the output unusable for animation or video art
- Wants to combine multiple effects (edge detect + halftone + depth composite) but current tools require manual scripting with no composability

**Goal:** Produce clean, minimal structural representations of people from photos and video that serve as a canvas to draw or paint over, with smooth temporal continuity for video and a composable effect pipeline.

## Core Outcomes

Success means the artist can:

1. **Clean silhouette extraction** -- A binary or near-binary outline of a person's body, isolated from background, with smooth contours and no interior noise.
2. **Contour/edge map** -- Major structural edges (limbs, torso, head, clothing folds) without texture noise (skin pores, fabric weave, background detail).
3. **Depth map** -- A grayscale or colored depth representation showing relative distance of body parts from camera, usable as a shading guide or compositional layer.
4. **Topology mesh (stretch goal)** -- A simplified wireframe or polygon mesh derived from edges, representing the 3D surface topology of the person's form.
5. **Video pipeline with temporal coherence** -- Process video files frame-by-frame and produce output video where effects evolve smoothly across frames with no flickering or discontinuous jumps.
6. **Composable EffectRule chains** -- Define ordered sequences of processing rules (e.g., EdgeDetect -> HalfToneRandomization -> DepthComposite) that can be swapped, reordered, and individually configured, with each rule able to carry inter-frame state for temporal continuity.

Each output should be clean enough to place directly beneath a digital drawing layer with no preprocessing. For video outputs, temporal coherence must be visually smooth when played back at normal speed.

## Input / Output Specification

### Inputs

| Input | Format | Notes |
|-------|--------|-------|
| Single photograph | JPEG, PNG, TIFF | Primary use case. Resolution up to 4K. |
| Video file | MP4, MOV | Full video processing with frame extraction. Process all frames or a specified range. Video's own audio track is ignored. |
| Audio file (optional) | MP3, WAV, FLAC, AAC, OGG | Separate music/audio file. Analyzed for beat detection, spectral features, and onset data used by audio-reactive rules. Duration should match or exceed the video. |
| Region of interest (optional) | Bounding box or mask | Artist may want to focus on a specific person in a multi-person scene. |
| EffectRule chain config | YAML or JSON | Ordered list of rules with per-rule parameters. |

### Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Silhouette mask | PNG (binary alpha) | Clean person outline, transparent background. |
| Edge map | PNG (black lines on white/transparent) | Structural contour lines at configurable detail levels. |
| Depth map | PNG (grayscale 16-bit) or EXR | Per-pixel relative depth. Near = bright, far = dark (or configurable). |
| Topology mesh | SVG or OBJ | Wireframe or simplified polygon representation derived from edges. |
| Layered composite | PSD or multi-layer PNG | All outputs stacked as layers for import into drawing software. |
| Processed video | MP4 (H.264/H.265) | Full video with EffectRule chain applied, temporal coherence preserved. If an audio file was provided, it is muxed into the output video. |
| Frame sequence | Numbered PNGs in directory | Individual processed frames for fine-grained artistic control. |

### Detail Level Control

The artist needs control over how much detail appears in edge and contour outputs. A "detail level" parameter (e.g., 1-10 scale) should allow:
- **Level 1-3:** Silhouette only (outer boundary)
- **Level 4-6:** Major structural lines (limbs, head, torso divisions)
- **Level 7-8:** Secondary contours (clothing folds, facial features)
- **Level 9-10:** Fine detail (fingers, fabric texture, hair strands)

## Algorithms to Investigate

### Classical Edge Detection
- **Canny edge detector** -- tunable thresholds, well-understood
- **Sobel / Scharr operators** -- gradient-based, fast
- **Laplacian of Gaussian (LoG)** -- blob/edge detection
- **Structured Edge Detection (Dollar & Zitnick)** -- learned edge detection with classical approach

### AI-Based Edge Detection
- **HED (Holistically-Nested Edge Detection)** -- deep learning multi-scale edges
- **RCF (Richer Convolutional Features)** -- improved HED
- **BDCN (Bi-Directional Cascade Network)** -- state-of-art learned edges
- **DexiNed** -- edge detection without per-dataset training
- **PiDiNet** -- lightweight pixel-difference networks

### Person Segmentation (for silhouette isolation)
- **MediaPipe Pose / Selfie Segmentation** -- real-time, lightweight
- **SAM (Segment Anything Model)** -- zero-shot segmentation
- **Mask R-CNN / PointRend** -- instance segmentation

### Depth Estimation
- **MiDaS** -- monocular depth estimation, robust across domains
- **Depth Anything** -- recent foundation model for depth
- **ZoeDepth** -- metric depth estimation
- **Marigold** -- diffusion-based depth estimation

### Topology / Mesh
- **Contour tracing + simplification** (Douglas-Peucker)
- **Delaunay triangulation** from edge points
- **Marching squares** for 2D topology from depth maps
- **PIFuHD** -- pixel-aligned implicit function for 3D human reconstruction

## Video Pipeline

### Frame Extraction and Processing
- Accept MP4 and MOV video files as input
- Extract frames at the video's native frame rate, or at a user-specified rate (e.g., every Nth frame)
- Process frames sequentially through the configured EffectRule chain
- Reassemble processed frames into an output video file (MP4, H.264 or H.265)
- Optionally output individual frames as numbered PNGs alongside the video

### Temporal Coherence
- Effects must be "frame-aware" -- they should not be applied independently to each frame
- Each EffectRule can carry state from frame N to frame N+1 (inter-frame state)
- Randomized or stochastic effects (e.g., halftone randomization) must evolve smoothly across frames, not jump discontinuously
- Approaches to temporal coherence include:
  - **Temporal smoothing:** blend current frame output with previous frame(s)
  - **Optical flow registration:** align effect parameters to motion between frames
  - **Explicit state carry-forward:** each rule maintains internal state that evolves incrementally per frame
  - **Seeded randomization with interpolation:** random seeds interpolate between keyframes rather than resetting per frame
- The pipeline should detect scene cuts and reset inter-frame state at boundaries

## EffectRule System

### Core Abstraction
An **EffectRule** is a composable, chainable unit of processing. Each rule:
- Reads from a named input layer and writes to a named output layer (defaulting to `main` for linear chain behavior)
- Accepts a frame (image) plus optional metadata (depth map, edge map, frame index, previous state)
- Produces a processed frame plus updated state
- Exposes rule-specific configuration parameters
- Can optionally carry inter-frame state for temporal continuity

### Chain Composition
- Rules are composed into an ordered chain: `[Rule_A, Rule_B, Rule_C]`
- The output of Rule_A feeds into Rule_B, which feeds into Rule_C
- The chain is defined in a configuration file (YAML or JSON)
- Rules can be added, removed, or reordered without code changes

### Rule Interface
Each EffectRule implements:
- `process(frame, context) -> (processed_frame, updated_state)` -- the core processing method
- `configure(params)` -- set rule-specific parameters
- `reset_state()` -- clear inter-frame state (e.g., at scene boundaries)
- `get_state() / set_state(state)` -- serialize/deserialize state for checkpointing

### Example Rules
| Rule Name | Description | Stateful? |
|-----------|-------------|-----------|
| EdgeDetect | Detect edges at configurable detail level | No |
| SilhouetteExtract | Extract person silhouette mask | No |
| DepthMap | Generate monocular depth map | No |
| HalfToneRandomization | Apply halftone dot pattern with randomized placement | Yes -- dot pattern evolves smoothly across frames |
| TemporalSmooth | Blend current frame with exponential moving average of previous frames | Yes -- maintains running average |
| DepthComposite | Composite depth map as a visual layer over the frame | No |
| ContourOverlay | Overlay simplified contour lines onto the frame | No |
| ThresholdBinarize | Binarize frame to black/white at configurable threshold | No |
| AudioReactiveEdgeRule | Displace, pulse, or warp person edges in sync with a separate audio file | Yes -- carries pre-computed audio analysis data and decay state across frames |
| CompositeRule | Merge multiple named layers into a final output using configurable blend modes and opacity | No |

### Layer System

Rules operate on **named layers** rather than a single linear frame pipeline. This enables independent parallel effects on the same source data.

#### Named Layers
The pipeline maintains a set of named image layers that rules can read from and write to:

| Layer Name | Description | Created By |
|------------|-------------|------------|
| `original` | The unmodified input frame | Pipeline (automatic) |
| `mask` | Person silhouette binary mask | SilhouetteExtract |
| `edges` | Edge/contour map | EdgeDetect |
| `depth` | Depth map | DepthMap |
| `main` | Default working layer (linear chain behavior) | Pipeline (automatic) |
| *(user-defined)* | Custom layers (e.g., `halftone`, `audio_lines`) | Any rule via `output_layer` |

#### input_layer / output_layer

Each rule declares:
- **`input_layer`** -- which named layer it reads from (default: `main`)
- **`output_layer`** -- which named layer it writes to (default: `main`)

This means:
- Two rules can both read from the same layer (e.g., both read `edges`) and produce independent results on separate output layers, with no cross-contamination
- A rule that writes to `halftone` does not affect a rule that writes to `audio_lines`, even if both read from `edges`
- Rules without explicit layer configuration default to `input_layer: main` and `output_layer: main`, preserving standard linear chain behavior

#### CompositeRule

A special **CompositeRule** merges multiple named layers into a final output. It supports:
- **Blend modes:** `screen`, `multiply`, `add`, `over` (alpha compositing)
- **Per-layer opacity**
- **Layer ordering** (back-to-front compositing order)

Example: halftone dots on one layer + audio-reactive lines on another, composited together with screen blend mode onto the depth map background.

#### Layer Configuration Example
```yaml
rules:
  - name: EdgeDetect
    output_layer: "edges"
    params:
      algorithm: "hed"
      detail_level: 5
  - name: HalfToneRandomization
    input_layer: "edges"
    output_layer: "halftone"
    params:
      dot_size: 4
  - name: AudioReactiveEdgeRule
    input_layer: "edges"
    output_layer: "audio_lines"
    params:
      sensitivity: 0.7
  - name: CompositeRule
    params:
      layers:
        - layer: "halftone"
          blend_mode: "screen"
          opacity: 0.8
        - layer: "audio_lines"
          blend_mode: "add"
          opacity: 1.0
      background: "depth"
```

### Configuration Format
```yaml
pipeline:
  input: "video.mp4"
  audio: "soundtrack.mp3"  # separate audio file, muxed into output
  output: "processed.mp4"
  frame_range: [0, -1]  # all frames
  rules:
    - name: EdgeDetect
      params:
        algorithm: "hed"
        detail_level: 5
    - name: HalfToneRandomization
      params:
        dot_size: 4
        randomization_strength: 0.3
        temporal_smoothing: 0.8
    - name: DepthComposite
      params:
        algorithm: "depth_anything"
        blend_mode: "multiply"
        opacity: 0.5
```

### Modularity Requirements
- New rules can be added by implementing the EffectRule interface -- no changes to the pipeline core
- Rules are discovered at runtime (plugin-style registration)
- Rules can be developed and tested independently on single images before being used in video chains
- Invalid rule configurations produce clear errors before processing begins

## Non-Functional Requirements

| Requirement | Specification |
|-------------|---------------|
| Processing mode | **Offline / batch.** Real-time is not required. |
| Latency target (image) | < 30 seconds per image on consumer GPU (RTX 3060 or equivalent). CPU fallback acceptable at slower speeds. |
| Latency target (video) | Proportional to frame count. A 10-second 30fps video (300 frames) should complete in under 2 hours on consumer GPU. |
| Resolution | Must support input up to 4096x4096 (image) or 4K video. Output at same resolution as input. |
| Output format (image) | Lossless PNG for raster outputs. SVG for vector/topology. |
| Output format (video) | MP4 (H.264 or H.265) for video. Optional frame-by-frame PNG export. |
| Platform | macOS (Apple Silicon) primary. Linux secondary. |
| Dependencies | Python ecosystem. Prefer PyTorch for ML models. FFmpeg for video I/O. |
| Reproducibility | Same input + same parameters + same rule chain = same output (deterministic where possible). |
| Configurability | All algorithm and rule parameters exposed via YAML/JSON config file or CLI flags. |
| No cloud dependency | All processing runs locally. No API calls to external services. |
| Temporal coherence | Video output must be visually smooth at playback speed. No flickering or per-frame discontinuities. |
| Checkpointing | Long video jobs can be interrupted and resumed without reprocessing completed frames. |

## User Stories

### US-1: Silhouette Extraction
```
As a visual artist preparing reference layers,
I want to extract a clean silhouette of a person from a photograph,
So that I can use it as a base outline for drawing without manually tracing.
```

### US-2: Adjustable Edge Detail
```
As an artist who works at varying levels of abstraction,
I want to control how much edge detail appears in the output (from simple silhouette to fine contours),
So that I can choose the right level of structural guidance for each piece.
```

### US-3: Depth Map Generation
```
As an artist exploring dimensional composition,
I want to generate a depth map from a single photograph of a person,
So that I can use depth as a shading guide or to create layered compositions.
```

### US-4: Topology Wireframe
```
As an artist studying human form and structure,
I want to generate a simplified wireframe mesh from detected edges,
So that I can understand and reference the 3D topology of a pose.
```

### US-5: Noise-Free Output
```
As an artist who needs clean reference layers,
I want edge detection output that excludes background clutter, skin texture, and other visual noise,
So that I only see the structural lines I need without spending time cleaning up artifacts.
```

### US-6: Batch Processing
```
As an artist working with video reference or photo series,
I want to process multiple frames or images in a batch,
So that I can generate consistent reference layers across a sequence.
```

### US-7: Video Processing with Temporal Coherence
```
As an artist creating video-based art or animation references,
I want to process a video file through the pipeline and get a processed video back,
So that I can use temporally coherent edge/depth/effect output as a foundation for video art.
```

### US-8: Composable Effect Chains
```
As an artist who experiments with layered visual effects,
I want to define an ordered chain of processing rules (e.g., edge detect then halftone then depth composite),
So that I can mix and match effects without writing code and iterate on the combination quickly.
```

### US-9: Smooth Stochastic Effects Across Frames
```
As an artist applying randomized effects (like halftone or stippling) to video,
I want the randomized pattern to evolve smoothly across frames rather than jumping discontinuously,
So that the output video looks intentional and animated rather than flickering.
```

### US-10: Swappable Rules
```
As an artist comparing different approaches,
I want to swap one rule in my chain for another (e.g., replace Canny with HED) without reconfiguring the rest,
So that I can A/B test different algorithms within the same pipeline.
```

### US-11: Audio-Reactive Edge Effects
```
As an artist creating AMV-style music video art,
I want to pair a video of a person with a separate music file and have detected edges pulse, warp, and displace in sync with the music,
So that my visual edge layers are synchronized with the soundtrack for dynamic, beat-driven compositions.
```

## Success Criteria

1. **Silhouette accuracy:** Person silhouette cleanly separated from background in 90%+ of test images without manual correction.
2. **Edge cleanliness:** At detail level 4-6, output contains only major structural contours -- no texture noise, no background edges.
3. **Depth plausibility:** Depth map correctly orders body parts by relative distance (e.g., extended hand closer than torso) in 85%+ of test images.
4. **Artist usability:** Output can be loaded directly into Procreate, Photoshop, or Krita as a reference layer with no preprocessing.
5. **Algorithm comparison:** At least 3 classical and 3 AI-based edge detection approaches benchmarked side-by-side on the same test set.
6. **Video temporal coherence:** Processed video plays back smoothly with no visible flickering or frame-to-frame discontinuities when viewed at normal speed.
7. **Effect chain composability:** At least 3 different EffectRule chains can be defined, executed, and produce distinct visual results from the same input.
8. **Stateful rule continuity:** A stateful rule (e.g., HalfToneRandomization) produces output that evolves smoothly across a 30-frame test clip, with no abrupt pattern changes between consecutive frames.
9. **Rule swappability:** Replacing a single rule in a chain (e.g., Canny -> HED) produces a valid different output without errors or reconfiguration of other rules in the chain.
10. **Audio-reactive edges:** When AudioReactiveEdgeRule is applied to a video paired with a separate audio file, edge displacement and thickness visibly react to beat onsets and frequency band energy, with smooth decay between beats.
11. **Audio muxing:** Output video has the provided audio file muxed in, synchronized with the processed visual frames.

## Out of Scope

- **Real-time / live video processing** -- This is an offline pipeline. Video is processed in batch, not streamed live.
- **Multi-person scene decomposition** -- Focus is on single-person extraction. Multi-person is a future extension.
- **Full 3D model reconstruction** -- Topology mesh is a 2.5D simplification, not a production-quality 3D scan.
- **Style transfer or artistic rendering** -- The pipeline produces structural data and applies composable effects, but does not perform AI style transfer (e.g., neural style transfer, diffusion-based stylization).
- **Training custom models** -- Investigation uses pre-trained models. Fine-tuning is a separate effort.
- **Web UI or mobile app** -- CLI and scriptable Python API are sufficient for this investigation phase.
- **Color or texture extraction** -- Focus is on structure (edges, depth, topology), not appearance.
- **Audio synthesis or modification** -- The pipeline reads and analyzes a separate audio file for reactive effects, and muxes it into the output video, but does not generate, edit, or re-encode audio content. Audio is read-only and passed through unchanged.
- **GPU-based video encoding** -- Output video encoding uses FFmpeg with CPU codecs. Hardware-accelerated encoding is a future optimization.

## Open Questions

1. Should the pipeline support transparent/alpha-channel input (e.g., pre-cut subjects)?
2. Is there a preferred line weight or stroke style for edge output (constant width vs. pressure-varied)?
3. Should depth maps be absolute (metric) or relative? Relative is simpler but metric enables more precise 3D work.
4. What test dataset will be used? (Studio portraits, candid photos, varied lighting, varied poses?)
5. Should the topology mesh preserve anatomical landmarks (joints, spine) or just follow contour geometry?
6. What is the maximum video duration the pipeline should handle? (Minutes? Hours?)
7. Should the EffectRule chain config support conditional branching (if/else), or is strict linear chaining sufficient for now?
8. How should scene cuts be detected -- threshold on frame difference, or explicit user annotation?
9. Should inter-frame state be serializable to disk for long jobs, or is in-memory state sufficient?
10. Should the pipeline support variable frame rate (VFR) input video, or only constant frame rate (CFR)?

## Dependencies

- Python 3.10+
- PyTorch 2.x with MPS (Apple Silicon) or CUDA support
- OpenCV for classical algorithms
- FFmpeg for video frame extraction and reassembly
- librosa for audio analysis (beat detection, spectral features, onset detection)
- Pre-trained model weights (downloaded on first run)
