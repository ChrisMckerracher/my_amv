# Technical Design: Edge Detection + Topology + Depth Mapping Pipeline

**Status**: Draft (Rev 8 -- layer-based compositing verified, final layer extraction in video loop)
**Author**: Architect Agent
**Date**: 2026-02-20
**Updated**: 2026-02-20
**Purpose**: AMV-style art pipeline: video of a person + separate music track -> processed art video with audio-reactive effects
**Product Brief**: `/Users/chrismck/Code/my_amv/docs/plans/product/edge-topology-brief.md`
**Feature Files**: `/Users/chrismck/Code/my_amv/docs/features/`

---

## Product Requirements Summary

The following constraints from the product brief drive architectural decisions:

| Requirement | Specification |
|-------------|---------------|
| Input model | Video file + separate audio file (AMV-style, two independent inputs) |
| Processing mode | Offline/batch (no real-time requirement) |
| Platform | macOS Apple Silicon (MPS) primary, Linux (CUDA) secondary |
| Resolution | Up to 4096x4096, output matches input |
| Latency | < 30 seconds per image on consumer GPU |
| Determinism | Same input + params = same output |
| Local only | No cloud APIs |
| Framework | Python + PyTorch |

**Required outputs per image**:
1. Silhouette mask -- binary alpha PNG
2. Edge/contour map -- black lines on transparent PNG, configurable detail level (1-10)
3. Depth map -- 16-bit grayscale PNG or EXR
4. Topology mesh -- SVG (2D wireframe) or OBJ (depth-enhanced 2.5D)
5. Layered composite -- PSD-compatible file stacking all outputs

---

## Table of Contents

**Part A: Image Pipeline (Algorithms & Per-Frame Processing)**
1. [Classical Edge Detection Algorithms](#1-classical-edge-detection-algorithms)
2. [AI-Based Edge/Silhouette Detection](#2-ai-based-edgesilhouette-detection)
3. [Person-Specific Detection](#3-person-specific-detection)
4. [Depth Mapping](#4-depth-mapping)
5. [Topology Generation](#5-topology-generation)
6. [Detail Level Parameter](#6-detail-level-parameter)
7. [Recommended Image Pipeline](#7-recommended-pipeline)
8. [Output Format Specifications](#8-output-format-specifications)
9. [Determinism and Reproducibility](#9-determinism-and-reproducibility)

**Part B: Video Pipeline, EffectRule System & Audio**
12. [EffectRule System Architecture](#12-effectrule-system-architecture)
13. [Video Pipeline Architecture](#13-video-pipeline-architecture)
14. [Temporal Coherence Strategies](#14-temporal-coherence-strategies)
15. [Audio Analysis and Audio-Reactive Rules](#15-audio-analysis-and-audio-reactive-rules)

**Part C: Infrastructure**
10. [Tech Stack](#10-tech-stack)
11. [ADR Log](#11-adr-log)

---

## 1. Classical Edge Detection Algorithms

Classical methods operate on pixel intensity gradients. They are fast, deterministic, and require no GPU, but produce raw edges without semantic understanding (they detect ALL edges, not just person contours).

### 1.1 Canny Edge Detector

**How it works**: Multi-stage -- Gaussian blur -> gradient computation (Sobel) -> non-maximum suppression -> hysteresis thresholding (dual threshold).

**Key parameters**:
- `threshold1` (low): Edges below this are discarded. Typical: 50-100.
- `threshold2` (high): Edges above this are kept. Typical: 150-200.
- `apertureSize`: Sobel kernel size (3, 5, or 7). Larger = smoother gradients.
- Gaussian `sigma` (applied before Canny): Controls blur. Higher sigma = fewer fine details.

**Artistic trade-offs**:
- Low thresholds: More detail, more noise. Creates textured/busy aesthetic.
- High thresholds: Clean lines but may lose subtle contours.
- Best for: Clean line art when combined with pre-segmented input (person mask applied first).

```python
import cv2
edges = cv2.Canny(gray_image, threshold1=80, threshold2=160, apertureSize=3)
```

### 1.2 Sobel / Scharr Operators

**How they work**: First-order derivative filters detecting horizontal and vertical intensity gradients separately.

- **Sobel**: 3x3 kernel (or larger). Good general-purpose gradient detection.
- **Scharr**: Modified 3x3 kernel with better rotational symmetry. More accurate for fine gradients.

**Artistic trade-offs**:
- Produce gradient magnitude images, not binary edges. Good for creating "glow" or "energy field" effects.
- Can combine X and Y gradients at different weights for directional emphasis.
- Best for: Gradient-based artistic effects, not clean line extraction.

```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = cv2.magnitude(sobel_x, sobel_y)
```

### 1.3 Laplacian of Gaussian (LoG)

**How it works**: Second-order derivative. Applies Gaussian blur then Laplacian operator. Detects edges as zero-crossings.

**Artistic trade-offs**:
- Detects edges at all orientations simultaneously.
- Very sensitive to noise -- requires careful sigma tuning.
- Creates a "wireframe" effect with double edges at strong boundaries.
- Best for: Ethereal/ghost-like edge effects.

### 1.4 Difference of Gaussians (DoG)

**How it works**: Subtracts two Gaussian-blurred versions of the image (different sigmas). Approximates LoG but is computationally cheaper.

**Artistic trade-offs**:
- The sigma ratio controls which scale of features survive.
- Small ratio (e.g., 1.0 vs 1.6): Fine detail.
- Large ratio (e.g., 1.0 vs 4.0): Only major contours.
- Best for: Band-pass filtered artistic effects, selective detail levels.

```python
blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
dog = blur1 - blur2
```

### 1.5 Structured Edge Detection (Structured Forests)

**How it works**: Uses random decision forests trained on edge/non-edge patches. Semi-classical (uses ML but not deep learning).

**Artistic trade-offs**:
- More semantically aware than pure gradient methods.
- Produces softer, more natural-looking edges.
- Available in OpenCV's `ximgproc` module.
- Best for: Natural-looking edges that bridge the gap between classical and AI methods.

```python
edge_detector = cv2.ximgproc.createStructuredEdgeDetection("model.yml.gz")
edges = edge_detector.detectEdges(np.float32(image) / 255.0)
```

### 1.6 When to Use Each (Summary)

| Method | Speed | Clean Lines | Noise Sensitivity | Semantic Awareness | Art Use Case |
|--------|-------|-------------|-------------------|-------------------|--------------|
| Canny | Fast | High | Medium | None | Clean line art (on pre-masked input) |
| Sobel/Scharr | Fast | Low (gradient) | Medium | None | Gradient glow effects |
| LoG | Fast | Medium | High | None | Ethereal wireframe |
| DoG | Fast | Medium | Low | None | Scale-selective contours |
| Structured Forests | Medium | High | Low | Low | Natural sketch look |

**Key insight for art use**: Classical methods work best AFTER person segmentation. Apply a person mask first, then run edge detection on the masked region. This eliminates background noise entirely.

---

## 2. AI-Based Edge / Silhouette Detection

AI methods produce semantically meaningful edges -- they understand object boundaries rather than just pixel gradients. This is critical for isolating person contours cleanly.

### 2.1 HED (Holistically-Nested Edge Detection)

**Architecture**: VGG16 backbone with side outputs from multiple layers, fused for multi-scale edge prediction.

**Strengths**:
- Produces soft, natural-looking edges similar to hand-drawn outlines.
- Multi-scale: captures both fine details and overall contours.
- Well-supported in ControlNet ecosystem.

**Limitations**:
- Detects ALL edges (not person-specific). Still needs masking for person-only output.
- Older architecture (2015). Outperformed by newer methods in accuracy.

**Speed**: ~50ms on GPU, ~500ms on CPU.

**Usage**:
```python
# Via ControlNet preprocessor (controlnet_aux)
from controlnet_aux import HEDdetector
hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
result = hed(image)
```

### 2.2 ControlNet Preprocessors (Recommended for Art)

The ControlNet ecosystem provides several edge/line preprocessors optimized for different artistic styles:

| Preprocessor | Style | Best For |
|--------------|-------|----------|
| `softedge_hed` | Soft, painterly edges | Recoloring, restyling |
| `softedge_pidinet` | Clean soft edges | General edge-guided generation |
| `lineart_standard` | Standard line art | Technical illustration |
| `lineart_realistic` | Detailed line art | Realistic illustrations |
| `lineart_anime` | Anime-style lines | Anime/manga aesthetic |
| `scribble_hed` | Thick, pronounced edges | Bold structural outlines |

**Recommendation**: `lineart_realistic` or `lineart_anime` produce the cleanest single-line contours for art canvas use. `softedge_hed` best preserves natural edge quality.

**Installation**:
```python
pip install controlnet_aux
from controlnet_aux import LineartDetector, LineartAnimeDetector
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
result = lineart(image)
```

### 2.3 RCF (Richer Convolutional Features)

**Architecture**: Based on VGG16 like HED, but uses features from ALL convolutional layers (not just side outputs), providing richer multi-scale edge information.

**Strengths**:
- Richer feature representation than HED.
- Better at capturing fine-grained edges.

**Limitations**:
- Can produce noisy edges -- sensitive to textures and surface details.
- Older architecture (2017). Outperformed by newer methods.
- Not ideal for our pipeline due to noise sensitivity.

### 2.4 BDCN (Bi-Directional Cascade Network)

**Architecture**: Bi-directional cascade structure with scale-specific supervision at different network layers.

**Strengths**:
- Better scale-specific edge detection than HED/RCF.
- Achieves strong benchmark scores (.890/.899/.934 on building edge benchmarks).
- More discriminative feature learning per scale.

**Limitations**:
- Higher computational cost than PiDiNet.
- Still detects all edges (not person-specific).

### 2.5 DexiNed (Dense Extreme Inception Network)

**Architecture**: Dense inception-based edge detection network. Designed for zero-shot transfer (no per-dataset fine-tuning needed).

**Strengths**:
- Excellent zero-shot performance -- works well on unseen domains.
- Strong boundary accuracy (.893/.897/.940 on benchmarks).
- No dataset-specific training required.

**Limitations**:
- Slower than PiDiNet.
- Larger model than lightweight alternatives.

### 2.6 PiDiNet (Pixel Difference Network)

**Architecture**: Lightweight network using pixel difference convolution (PDC), mimicking traditional edge operators with learned parameters.

**Strengths**:
- Very lightweight and fast -- near real-time on CPU.
- Accuracy close to heavier models (HED, RCF, BDCN).
- Available as ControlNet preprocessor (`softedge_pidinet`).
- Good balance of quality and speed for our pipeline.

**Limitations**:
- Slightly less accurate than BDCN/DexiNed on benchmarks.
- Still detects all edges (not person-specific).

```python
# Via ControlNet preprocessor
from controlnet_aux import PidiNetDetector
pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
result = pidinet(image)
```

### 2.7 AI Edge Model Comparison

| Model | Year | Architecture | Speed | Accuracy | Zero-Shot | Art Suitability |
|-------|------|-------------|-------|----------|-----------|-----------------|
| HED | 2015 | VGG16 + side outputs | Fast | Good | Fair | Good (soft edges) |
| RCF | 2017 | VGG16 + all layers | Medium | Good+ | Fair | Fair (noisy) |
| BDCN | 2019 | Bi-directional cascade | Medium | Very Good | Good | Good |
| DexiNed | 2020 | Dense inception | Medium | Very Good | Excellent | Good |
| PiDiNet | 2021 | Pixel difference conv | Very Fast | Good | Good | Good (lightweight) |
| ControlNet Lineart | 2023 | Various | Fast | Excellent | Excellent | Excellent (art-tuned) |

**Recommendation for art pipeline**: ControlNet lineart preprocessors remain the best choice because they are specifically tuned for artistic line output. For benchmarking/comparison purposes, include PiDiNet (fast) and DexiNed (accurate) alongside.

### 2.8 SAM 2.1 (Segment Anything Model 2) -- For Person Segmentation to Edges

**Architecture**: Transformer-based promptable segmentation model. Works on images and video.

**Strengths**:
- State-of-the-art segmentation quality.
- Zero-shot: works on any object class without fine-tuning.
- SAM 2.1 improved handling of occlusions and similar objects.
- Video support with temporal consistency (streaming memory architecture).
- Promptable: point, box, or mask prompts.

**For our pipeline**:
- Use SAM2 to generate a pixel-perfect person mask.
- Extract contours from the mask boundary using OpenCV.
- This gives clean silhouette edges with zero background noise.

**Limitations**:
- Requires a prompt (point/box click on the person). Can be automated with a person detector.
- Heavier than dedicated segmentation models (~300ms per frame on GPU).

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor(build_sam2("sam2.1_hiera_large.yaml", "sam2.1_hiera_large.pt"))
predictor.set_image(image)
masks, scores, _ = predictor.predict(point_coords=[[x, y]], point_labels=[1])
```

### 2.9 RMBG 2.0 (Background Removal)

**Architecture**: Built on BiRefNet (Bilateral Reference Network). Trained on 15,000+ high-quality labeled images.

**Strengths**:
- Outputs 8-bit alpha matte (not binary mask) -- smooth, anti-aliased edges.
- Excellent for person silhouettes. Handles hair, transparent fabrics.
- Fast inference. Lightweight compared to SAM.

**For our pipeline**:
- Primary recommendation for person silhouette extraction.
- The alpha matte can be thresholded for binary mask or used as-is for soft edges.
- Contours from the matte give very clean person outlines.

**Limitations**:
- Source-available for non-commercial use. Commercial license required for production.
- Trained primarily on stock photos -- may struggle with unusual poses or heavy occlusion.

```python
from transformers import pipeline
pipe = pipeline("image-segmentation", model="briaai/RMBG-2.0", trust_remote_code=True)
result = pipe(image)  # Returns PIL Image with alpha channel
```

### 2.10 U2-Net (Salient Object Detection)

**Architecture**: Nested U-structure with Residual U-blocks (RSU). 44MB model (lightweight).

**Strengths**:
- Very small model -- runs well on CPU.
- Good at detecting the "main subject" in an image.
- Has been used for portrait generation and art transfer applications.
- Multiple variants: U2-Net (full, 176MB), U2-Net-P (portable, 4.7MB).

**For our pipeline**:
- Good lightweight alternative to RMBG for salient person detection.
- The portable version enables real-time processing.
- Outputs a saliency map that can be thresholded for a mask.

**Limitations**:
- Not person-specific -- detects the most "salient" object.
- Edge quality lower than RMBG 2.0 or SAM 2.
- No alpha matte -- binary or soft saliency map.

### 2.11 Comparison Matrix

| Model | Edge Quality | Person-Specific | Speed (GPU) | Speed (CPU) | Model Size | Noise Level |
|-------|-------------|----------------|-------------|-------------|------------|-------------|
| HED | Good (soft) | No | 50ms | 500ms | ~56MB | Medium |
| ControlNet Lineart | Excellent | No | 80ms | 1s | ~200MB | Low |
| SAM 2.1 | Excellent (via mask) | Promptable | 300ms | Slow | ~2.4GB | Very Low |
| RMBG 2.0 | Excellent (alpha) | Yes (foreground) | 100ms | 800ms | ~176MB | Very Low |
| U2-Net | Good | No (salient) | 60ms | 300ms | ~176MB | Low |
| U2-Net-P | Fair | No (salient) | 30ms | 150ms | ~4.7MB | Medium |

---

## 3. Person-Specific Detection

These models provide structural understanding of the human body -- skeleton keypoints, body part segmentation, and pose topology.

### 3.1 MediaPipe Pose

**What it provides**: 33 body landmarks (keypoints) in real-time.

**Strengths**:
- Extremely fast: designed for mobile/web. Runs on CPU at 30+ FPS.
- No GPU required.
- Cross-platform (Python, JS, Android, iOS).
- Includes face mesh (468 landmarks) and hand tracking (21 per hand).

**For our pipeline**:
- Provides skeleton topology (connect-the-dots body wireframe).
- Lightweight enough for real-time video processing.
- Skeleton can be rendered as artistic wireframe overlay.

**Limitations**:
- Single-person only (in default mode).
- Keypoints only -- no pixel-level segmentation.
- Accuracy drops with unusual poses, heavy occlusion, or non-standard camera angles.

```python
import mediapipe as mp
mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = results.pose_landmarks  # 33 keypoints
```

### 3.2 OpenPose

**What it provides**: Multi-person body (25 keypoints), hand (21 per hand), face (70 points), foot (6 points) detection.

**Strengths**:
- Multi-person detection (bottom-up approach).
- Part Affinity Fields for robust limb association.
- Industry standard for motion capture reference in film/games.

**Limitations**:
- Slower than MediaPipe (requires GPU for real-time).
- Complex installation (C++ with Python bindings).
- Development has slowed -- DWPose is the modern successor.

**For our pipeline**:
- Use if multi-person skeleton topology is needed.
- Good for dense artistic wireframes with hand/face detail.

### 3.3 DWPose (Recommended)

**What it provides**: Whole-body pose estimation -- 133 keypoints (body + hands + face).

**Architecture**: Two-stage distillation based on RTMPose. Trained on UBody dataset.

**Strengths**:
- State-of-the-art accuracy on COCO-WholeBody benchmark.
- Faster and more accurate than OpenPose.
- Better whole-body coverage (133 keypoints vs OpenPose's ~130 with separate models).
- Single unified model vs OpenPose's separate body/hand/face models.
- Well-integrated with ControlNet ecosystem.

**Limitations**:
- Requires GPU for best performance.
- Top-down approach: needs person detector first (typically uses YOLO or RTMDet).

```python
from controlnet_aux import DWposeDetector
dwpose = DWposeDetector()
result = dwpose(image)  # Returns pose visualization
```

### 3.4 Mask R-CNN / PointRend (Instance Segmentation)

**What it provides**: Per-instance segmentation masks with bounding boxes and class labels.

**Mask R-CNN**:
- Two-stage detector: Region proposal (RPN) followed by mask prediction.
- Instance-aware: separate mask per person in multi-person scenes.
- Available in torchvision and Detectron2.

**PointRend**:
- Extension of Mask R-CNN that refines mask boundaries using point-based rendering.
- Produces significantly sharper mask edges than standard Mask R-CNN.
- Iteratively refines uncertain boundary pixels.

**For our pipeline**:
- PointRend is a strong option for silhouette extraction when edge sharpness is critical.
- Better boundary quality than YOLO-seg, though slower.
- Instance segmentation allows per-person processing in multi-person scenes.

**Limitations**:
- Slower than YOLO-seg (two-stage architecture).
- Detectron2 has complex installation (not a simple pip install).
- RMBG 2.0 still produces cleaner alpha mattes for single-person use.

```python
# Detectron2 PointRend
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import detectron2.projects.point_rend as point_rend

cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file("configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(image)
# Filter for person class (class 0 in COCO)
```

### 3.5 DeepLabV3+ (Semantic Segmentation)

**What it provides**: Pixel-level semantic segmentation with "person" as one of 21 PASCAL VOC classes.

**Strengths**:
- Pixel-perfect person segmentation (not just keypoints).
- Pre-trained on COCO/PASCAL VOC -- person class well-represented.
- Available in torchvision with pre-trained weights.
- Can distinguish person from background at pixel level.

**For our pipeline**:
- Alternative to RMBG/SAM for person mask generation.
- Advantage: explicitly trained on "person" class.
- Mask edges can be extracted as person contours.

**Limitations**:
- Not as sharp at boundaries as RMBG 2.0 or SAM 2.
- Segments ALL people in scene (no instance segmentation in base model).

```python
import torchvision
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()
# Class index 15 = person in PASCAL VOC
```

### 3.6 YOLO Segmentation (YOLOv8/v11-seg)

**What it provides**: Instance segmentation -- per-person masks with bounding boxes.

**Strengths**:
- Extremely fast (real-time on consumer GPU).
- Instance-aware: separate mask per person.
- Single model for detection + segmentation.
- Latest versions (YOLOv11) have excellent accuracy.

**For our pipeline**:
- Best option for multi-person scenarios where each person needs separate processing.
- Fast enough for video (60+ FPS on GPU).
- Provides bounding boxes that can seed SAM2 prompts.

**Limitations**:
- Mask resolution lower than SAM2 or DeepLabV3+.
- Edge quality not as refined for art purposes.

```python
from ultralytics import YOLO
model = YOLO("yolo11x-seg.pt")
results = model(image)
for result in results:
    masks = result.masks  # Per-instance segmentation masks
```

### 3.7 Comparison for Person Detection

| Model | Output Type | Multi-Person | Speed | Edge Quality | Topology |
|-------|------------|-------------|-------|-------------|----------|
| MediaPipe | 33 keypoints | No | Very Fast (CPU) | N/A | Skeleton |
| OpenPose | 25-130 keypoints | Yes | Medium (GPU) | N/A | Skeleton |
| DWPose | 133 keypoints | Yes | Fast (GPU) | N/A | Full skeleton |
| Mask R-CNN | Instance masks | Yes (instance) | Medium | Good | None |
| PointRend | Instance masks | Yes (instance) | Medium | Very Good | None |
| DeepLabV3+ | Pixel mask | Yes (semantic) | Medium | Good | None |
| YOLO-seg | Instance masks | Yes (instance) | Very Fast | Fair | None |

---

## 4. Depth Mapping

Monocular depth estimation generates a depth map from a single 2D image. For art, the depth map serves as a "canvas layer" showing spatial relationships.

### 4.1 MiDaS v3.1

**Architecture**: Multiple encoder options (BEiT, Swin, SwinV2, Next-ViT, LeViT). Trained on 12 datasets.

**Output**: Relative depth (no metric scale). Values indicate relative distance, not absolute meters.

**Strengths**:
- Mature, well-tested. Good cross-domain generalization.
- Multiple model sizes for speed/accuracy trade-off.
- BEiT-L-512 variant: best accuracy in the MiDaS family.

**Limitations**:
- Relative depth only -- no absolute measurements.
- Superseded by Depth Anything V2 in most benchmarks.
- Edge quality in depth maps is moderate.

**For our pipeline**:
- Legacy option. Use Depth Anything V2 instead unless compatibility requires MiDaS.

```python
import torch
model = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512")
```

### 4.2 ZoeDepth

**Architecture**: Builds on MiDaS with adaptive depth binning. Domain-specific heads for indoor/outdoor.

**Output**: Metric depth (absolute scale in meters).

**Strengths**:
- Fastest inference (~0.17s per image).
- Metric depth output (actual distances).
- Good cross-domain transfer via domain routing.

**Limitations**:
- Least accurate of modern methods (MAE: 3.087m in outdoor natural scenes).
- Significantly worse than Depth Anything V2 in benchmarks.
- Edge artifacts in depth maps.

**For our pipeline**:
- Use only if absolute metric depth is required AND speed is critical.
- Not recommended for art use due to lower edge quality.

### 4.3 Depth Anything V2 (Recommended)

**Architecture**: DINOv2 encoder with DPT decoder. Student-teacher training with synthetic data.

**Output**: Relative depth (high quality). Metric fine-tuned variants available.

**Strengths**:
- State-of-the-art accuracy. Best overall performance across benchmarks.
- Excellent edge quality in depth maps -- sharp object boundaries.
- Multiple scales: Small (25M), Base (98M), Large (335M), Giant (1.3B params).
- 10x faster than Stable Diffusion-based depth methods.
- Video variant (Video Depth Anything) for temporally consistent depth.

**Limitations**:
- Relative depth by default (fine-tuned metric variants exist but less accurate).
- Large model (Giant) requires significant VRAM.

**For our pipeline**:
- Primary recommendation for depth maps.
- Use `Depth-Anything-V2-Large` for best quality/speed balance.
- Use `Depth-Anything-V2-Small` for real-time video.
- Video Depth Anything for temporally stable video depth.

```python
from transformers import pipeline
depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
result = depth_pipe(image)
depth_map = result["depth"]  # PIL Image
```

### 4.4 Marigold (Diffusion-Based Depth)

**Architecture**: Fine-tuned Stable Diffusion model repurposed for monocular depth estimation. Uses the diffusion denoising process to generate depth maps.

**Output**: Relative depth. High visual quality with sharp boundaries.

**Strengths**:
- Leverages rich visual knowledge from Stable Diffusion's pre-training.
- Excellent zero-shot performance on unseen domains.
- Very sharp object boundaries in depth maps -- aligns closely with human perception.
- Best quality at 768x768 resolution (Stable Diffusion's native resolution).
- CVPR 2024 Oral presentation, Best Paper Award Candidate.

**Limitations**:
- Significantly slower than feed-forward models: requires multiple diffusion steps per image.
- At native 768x768 resolution, upscaling needed for 4096x4096 output.
- Non-deterministic by default (diffusion sampling has randomness) -- requires fixed seed for reproducibility.
- Higher VRAM usage than Depth Anything V2 due to diffusion backbone.

**For our pipeline**:
- Include as optional high-quality depth method.
- Best for cases where depth boundary sharpness is the top priority and speed is not.
- Determinism achievable via fixed random seed + single denoising step variant.

```python
from diffusers import MarigoldDepthPipeline
import torch

pipe = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-0", torch_dtype=torch.float16)
pipe = pipe.to("mps")  # or "cuda"
depth = pipe(image, num_inference_steps=10, generator=torch.Generator().manual_seed(42))
depth_map = depth.prediction[0]  # numpy array
```

### 4.5 Apple Depth Pro

**Architecture**: Multi-scale Vision Transformer. Trained on real + synthetic data.

**Output**: Metric depth (absolute scale) + focal length estimation.

**Strengths**:
- Sharp boundary detail -- best-in-class edge quality in depth maps.
- Captures fine details (hair, thin objects, mesh/cage structures).
- Metric output without requiring camera intrinsics.
- Fast: 2.25 megapixel depth map in 0.3 seconds.
- Includes focal length estimation.

**Limitations**:
- Larger model footprint.
- Apple license (open-source but check terms for commercial use).
- Newer model -- less community tooling and integration.

**For our pipeline**:
- Best option when depth map edge sharpness is the priority.
- Excellent for art use: the sharp boundaries translate to clean depth-based contours.
- Use alongside or instead of Depth Anything V2 when metric depth or edge sharpness matters most.

```python
import depth_pro
model, transform = depth_pro.create_model_and_transforms()
model.eval()
prediction = model.infer(transform(image))
depth = prediction["depth"]  # Metric depth in meters
```

### 4.6 Depth Model Comparison

| Model | Depth Type | Edge Quality | Speed (GPU) | Accuracy | Model Size | Art Suitability | Deterministic |
|-------|-----------|-------------|-------------|----------|------------|-----------------|---------------|
| MiDaS v3.1 | Relative | Good | 200ms | Good | ~400MB | Medium | Yes |
| ZoeDepth | Metric | Fair | 170ms | Fair | ~350MB | Low | Yes |
| Depth Anything V2-L | Relative | Excellent | 220ms | Excellent | ~1.3GB | High | Yes |
| Depth Anything V2-S | Relative | Good | 50ms | Good | ~100MB | Medium | Yes |
| Marigold | Relative | Excellent+ | 3-10s | Excellent | ~2GB | Very High | With fixed seed |
| Apple Depth Pro | Metric | Excellent+ | 300ms | Excellent | ~1.8GB | Highest | Yes |

**Recommendation**: Depth Anything V2-Large as the default (best accuracy/speed/determinism balance). Apple Depth Pro when edge sharpness or metric depth is paramount. Marigold as optional premium quality mode when speed is not a concern.

**30-second latency target**: All feed-forward models (MiDaS, ZoeDepth, Depth Anything, Depth Pro) complete well within 30 seconds even at 4096x4096. Marigold may require reduced inference steps or tiled processing at high resolution to meet the target.

---

## 5. Topology Generation

Topology transforms edge maps and depth data into structured geometric representations.

### 5.1 Contour Finding from Edge Maps (OpenCV)

**The core approach**: Convert edge detection output to vector contours.

```python
import cv2
import numpy as np

# From binary edge image
contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Simplify contours (reduce point count while preserving shape)
epsilon = 0.001 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

# Filter by area to remove noise
significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
```

**Parameters for art**:
- `RETR_EXTERNAL`: Only outer contours (silhouette only).
- `RETR_TREE`: Full hierarchy (inner details like facial features).
- `CHAIN_APPROX_SIMPLE`: Compresses segments to endpoints.
- `epsilon` in `approxPolyDP`: Controls simplification. Higher = fewer points, more abstract.

### 5.2 Mesh Generation from Depth + Edges

**Approach**: Convert depth map pixels to 3D point cloud, then create a mesh surface.

```python
import numpy as np
import open3d as o3d

# Depth map to point cloud
h, w = depth_map.shape
fx, fy = focal_length, focal_length  # From camera or estimated
cx, cy = w / 2, h / 2

# Create organized point cloud
points = []
for v in range(h):
    for u in range(w):
        z = depth_map[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points.append([x, y, z])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))

# Mesh reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
```

### 5.3 Delaunay Triangulation for Topology

**Use case**: Creates a triangulated mesh from edge points, producing a geometric/low-poly artistic effect.

```python
from scipy.spatial import Delaunay
import numpy as np

# Extract edge points from contour or edge map
edge_points = np.argwhere(edge_image > 0)  # (y, x) coordinates
# Subsample for performance
indices = np.random.choice(len(edge_points), size=min(5000, len(edge_points)), replace=False)
sampled_points = edge_points[indices]

# Triangulate
tri = Delaunay(sampled_points)

# Visualize
import matplotlib.pyplot as plt
plt.triplot(sampled_points[:, 1], sampled_points[:, 0], tri.simplices, linewidth=0.5)
```

**Art applications**:
- Low-poly portrait effect: triangulate face/body contour points, fill triangles with average color from original image.
- Wireframe art: render only the triangle edges.
- Density variation: more triangles in detailed areas (face), fewer in smooth areas (torso).

### 5.4 Douglas-Peucker Contour Simplification

**Use case**: Reduce contour point count while preserving shape. Critical for producing clean SVG output and controlling topology density.

**How it works**: Iteratively removes points that deviate less than `epsilon` from the line between their neighbors. The `epsilon` parameter directly controls the fidelity/simplicity trade-off.

```python
import cv2

# Extract contours from edge map
contours, _ = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Simplify each contour with Douglas-Peucker
simplified = []
for contour in contours:
    arc_len = cv2.arcLength(contour, closed=True)
    # epsilon as fraction of perimeter -- ties to detail level
    epsilon = 0.002 * arc_len  # Low epsilon = high fidelity
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    simplified.append(approx)
```

**Mapping to detail level**:
- Detail 1-3 (silhouette): `epsilon = 0.01 * arc_length` (heavy simplification)
- Detail 4-6 (structure): `epsilon = 0.003 * arc_length`
- Detail 7-8 (secondary): `epsilon = 0.001 * arc_length`
- Detail 9-10 (fine): `epsilon = 0.0003 * arc_length` (minimal simplification)

### 5.5 Marching Squares (Iso-contours from Depth Maps)

**Use case**: Extract iso-depth contour lines from depth maps, producing topographic-style contour visualization.

**How it works**: Marching squares processes a 2D scalar field (depth map) and extracts contour lines at specified threshold values. Similar to contour lines on a topographic map.

```python
import numpy as np
from skimage import measure

# Extract iso-contours from depth map at multiple levels
depth_normalized = depth_map / depth_map.max()
levels = np.linspace(0.1, 0.9, num=8)  # 8 contour levels

all_contours = []
for level in levels:
    contours = measure.find_contours(depth_normalized, level)
    all_contours.extend(contours)

# Render as SVG or overlay
import svgwrite
dwg = svgwrite.Drawing("topo_contours.svg", size=(width, height))
for contour in all_contours:
    points = [(float(x), float(y)) for y, x in contour]
    dwg.add(dwg.polyline(points, stroke="black", fill="none", stroke_width=0.5))
dwg.save()
```

**Art applications**:
- Topographic body maps showing depth as contour lines.
- Layered depth visualization (each contour = a depth slice).
- Combined with edge map: structural edges + depth contours = rich topology.

### 5.6 Edge-Based Graph Generation

**Use case**: Represent the person's structure as a mathematical graph for algorithmic art.

```python
import networkx as nx

# From contour points, build a graph
G = nx.Graph()
for contour in contours:
    points = contour.reshape(-1, 2)
    for i in range(len(points) - 1):
        G.add_edge(tuple(points[i]), tuple(points[i + 1]))
    G.add_edge(tuple(points[-1]), tuple(points[0]))  # Close contour

# From skeleton keypoints (e.g., DWPose)
skeleton_edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # Right arm
    (0, 5), (5, 6), (6, 7), (7, 8),   # Left arm
    # ... etc
]
for kp1, kp2 in skeleton_edges:
    G.add_edge(keypoints[kp1], keypoints[kp2])
```

---

## 6. Detail Level Parameter

The product requires a single integer parameter (1-10) that controls edge output density. This parameter must map to concrete algorithm settings.

### 6.1 Detail Level Definition

| Level | Label | Visible Content | Excluded Content |
|-------|-------|----------------|------------------|
| 1 | Outer silhouette only | Body outline | Everything interior |
| 2-3 | Silhouette + limb separation | Head, arms, legs as distinct shapes | Clothing, facial features |
| 4-5 | Major structural lines | Limb contours, torso divisions | Skin texture, fabric weave |
| 6 | Structural + basic clothing | Major clothing boundaries | Fine wrinkles, hair strands |
| 7-8 | Secondary contours | Clothing folds, facial feature outlines | Skin pores, fine hair |
| 9-10 | Fine detail | Fingers, hair strands, fabric texture | Background noise (always excluded) |

### 6.2 Implementation: Detail Level to Algorithm Parameters

The detail level maps to different strategies depending on the edge detection algorithm:

**Strategy A: Canny threshold mapping (classical)**
```python
def canny_params_for_detail_level(level: int) -> dict:
    """Map detail level 1-10 to Canny parameters."""
    # Higher level = lower thresholds = more edges detected
    configs = {
        1:  {"blur_sigma": 5.0, "low": 200, "high": 250},
        2:  {"blur_sigma": 4.0, "low": 170, "high": 220},
        3:  {"blur_sigma": 3.5, "low": 140, "high": 190},
        4:  {"blur_sigma": 3.0, "low": 120, "high": 170},
        5:  {"blur_sigma": 2.5, "low": 100, "high": 150},
        6:  {"blur_sigma": 2.0, "low": 80,  "high": 130},
        7:  {"blur_sigma": 1.5, "low": 60,  "high": 110},
        8:  {"blur_sigma": 1.0, "low": 40,  "high": 90},
        9:  {"blur_sigma": 0.5, "low": 25,  "high": 70},
        10: {"blur_sigma": 0.0, "low": 15,  "high": 50},
    }
    return configs[level]
```

**Strategy B: AI edge + contour area filtering**

For AI-based edge detectors (HED, Lineart, PiDiNet), which produce soft probability maps:

```python
def filter_edges_by_detail_level(edge_probs: np.ndarray, level: int, mask: np.ndarray) -> np.ndarray:
    """Filter AI edge detector output by detail level.

    Args:
        edge_probs: Soft edge probability map (0.0-1.0), from AI edge detector.
        level: Detail level 1-10.
        mask: Binary person mask.

    Returns:
        Filtered binary edge map.
    """
    # Higher level = lower threshold = more edges pass through
    threshold = 1.0 - (level / 10.0) * 0.8  # Maps 1->0.92, 10->0.20

    # Binarize
    edges = (edge_probs > threshold).astype(np.uint8) * 255

    # Apply person mask
    edges = edges & mask

    # Remove small contours (noise) -- stricter at low detail levels
    min_area = max(10, 500 - (level * 50))  # Maps 1->450px, 10->10px
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(edges)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean, [c], -1, 255, thickness=1)

    return clean
```

**Strategy C: Multi-method blending by level**

| Level Range | Primary Method | Fallback |
|-------------|---------------|----------|
| 1-3 | Silhouette contour from mask | Canny (high threshold) on mask |
| 4-6 | AI Lineart + area filter | Canny (medium threshold) + area filter |
| 7-8 | AI Lineart + low filter | Canny (low threshold) + low filter |
| 9-10 | AI Lineart (minimal filter) | Canny (very low threshold) |

At levels 1-3, the edge map IS the silhouette -- we just use the mask contour with Douglas-Peucker simplification. Interior edges only appear at level 4+.

### 6.3 Determinism of Detail Level

The detail level parameter must be deterministic: same image + same level = same output. This is satisfied because:
- All threshold values are deterministic functions of the level integer.
- Contour area filtering is deterministic.
- No randomness is involved in the edge detection or filtering.

---

## 7. Recommended Pipeline

### 7.1 Architecture Overview

```
Input Image/Frame + Config (detail_level, seed, algorithms)
       |
       v
+--------------------+
| Init:              |  set_deterministic(seed)
| Set deterministic  |  device = select_device()
+--------------------+
       |
       v
+--------------------+
| Stage 1: Person    |  RMBG 2.0 (single person)
| Segmentation       |  OR YOLO-seg -> RMBG 2.0 (multi-person)
+--------------------+
       |
       v  (person mask + masked image)
       |
  +----+--------+--------+
  |              |        |
  v              v        v
+-----------+ +--------+ +----------+
| Stage 2   | |Stage 3 | |Stage 3b  |
| Edge Det  | |Depth   | |Pose/     |
| +detail   | |Map     | |Skeleton  |
| level     | |        | |(optional)|
+-----------+ +--------+ +----------+
  |              |        |
  v              v        v
+--------------------+
| Stage 4:           |
| Topology Gen       |
| (optional)         |
+--------------------+
       |
       v
+--------------------+
| Stage 5: Output    |  Resolution matching + format conversion
| Assembly           |  PSD composite, 16-bit depth, SVG, OBJ
+--------------------+
       |
       v
+-----------------------------+
| Outputs:                    |
| - silhouette.png (alpha)    |
| - edges.png (transparent)   |
| - depth.png (16-bit)        |
| - depth.exr (float, opt)    |
| - topology.svg / .obj       |
| - composite.psd             |
+-----------------------------+
```

### 7.2 Stage 1: Person Segmentation / Masking

**Primary approach (single person)**:
```python
from transformers import pipeline as hf_pipeline

# RMBG 2.0 for clean foreground extraction
segmenter = hf_pipeline("image-segmentation", model="briaai/RMBG-2.0", trust_remote_code=True)
result = segmenter(input_image)
alpha_mask = result  # PIL Image with alpha channel
```

**Multi-person approach**:
```python
from ultralytics import YOLO

# Step 1: Detect and segment each person
yolo = YOLO("yolo11x-seg.pt")
results = yolo(input_image, classes=[0])  # class 0 = person

# Step 2: For each detected person, refine with RMBG or SAM2
for detection in results[0]:
    bbox = detection.boxes.xyxy[0]
    coarse_mask = detection.masks.data[0]
    # Crop and refine with RMBG 2.0 for cleaner edges
```

**Output**: Binary or alpha mask of person(s). Masked image (person on transparent/black background).

### 7.3 Stage 2: Edge Detection on Masked Person

**Primary approach (clean art lines)**:
```python
from controlnet_aux import LineartDetector
import cv2
import numpy as np

# Apply mask to isolate person
masked_image = apply_mask(input_image, person_mask)

# Option A: AI-based lineart (best for art)
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
edges_ai = lineart(masked_image)

# Option B: Classical Canny on masked input (fastest)
gray = cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2GRAY)
edges_canny = cv2.Canny(gray, 80, 160)

# Option C: Contour from mask boundary (cleanest silhouette)
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
silhouette = np.zeros_like(binary_mask)
cv2.drawContours(silhouette, contours, -1, 255, thickness=2)
```

**Noise reduction strategy**:
1. Apply person mask BEFORE edge detection (eliminates all background edges).
2. Use Gaussian blur on masked image before Canny (sigma=1.0-2.0).
3. Post-process: remove small contours by area threshold.
4. Optional: morphological operations (dilate/erode) to clean edges.

```python
# Post-processing for minimal noise
def clean_edges(edge_image, min_contour_area=100):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(edge_image)
    significant = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    cv2.drawContours(clean, significant, -1, 255, thickness=1)
    return clean
```

### 7.4 Stage 3: Depth Map Generation

```python
from transformers import pipeline as hf_pipeline
import numpy as np

# Depth Anything V2 (recommended)
depth_estimator = hf_pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
result = depth_estimator(input_image)
depth_map = np.array(result["depth"])

# Apply person mask to depth map
person_depth = depth_map * (person_mask / 255.0)

# Normalize for visualization
depth_normalized = cv2.normalize(person_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Optional: Apply colormap for artistic depth visualization
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
```

**For sharper depth edges (art priority)**:
```python
# Apple Depth Pro (when edge sharpness matters most)
import depth_pro
model, transform = depth_pro.create_model_and_transforms()
model.eval()
prediction = model.infer(transform(input_image))
depth_metric = prediction["depth"]  # Metric depth in meters
```

### 7.5 Stage 3b: Pose / Skeleton (Optional)

```python
from controlnet_aux import DWposeDetector

# DWPose for full body skeleton
dwpose = DWposeDetector()
pose_image = dwpose(input_image)  # Visualization with keypoints + connections

# For raw keypoints (for topology generation)
# Use MMPose/RTMPose directly for programmatic access
from mmpose.apis import MMPoseInferencer
inferencer = MMPoseInferencer("rtmpose-l")
result = next(inferencer(input_image))
keypoints = result["predictions"][0]["keypoints"]  # List of (x, y) coordinates
```

### 7.6 Stage 4: Topology Generation (Optional)

```python
from scipy.spatial import Delaunay
import numpy as np

# Option A: Delaunay triangulation of edge points
edge_points = np.argwhere(clean_edge_image > 0)
# Subsample for manageable mesh density
step = max(1, len(edge_points) // 3000)
sampled = edge_points[::step]
triangulation = Delaunay(sampled)

# Option B: Contour-based vector topology
contours, hierarchy = cv2.findContours(clean_edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Simplify for clean topology
simplified = [cv2.approxPolyDP(c, epsilon=2.0, closed=True) for c in contours]

# Option C: Depth-aware 3D mesh
# Combine edge points with depth values for 2.5D mesh
edge_3d_points = []
for y, x in sampled:
    z = depth_map[y, x]
    edge_3d_points.append([x, y, z])
tri_3d = Delaunay(np.array(edge_3d_points)[:, :2])  # Triangulate in 2D, use Z for height
```

### 7.7 Stage 5: Output Assembly

Combines all stage outputs into final deliverables.

```python
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer

def assemble_composite(silhouette, edges, depth, topology_svg, output_path):
    """Assemble all outputs into a PSD-compatible layered file."""
    h, w = silhouette.shape[:2]

    # Create PSD with layers
    # Using psd-tools or pytoshop for PSD generation
    # Alternative: TIFF with layers via tifffile, or ORA (OpenRaster)
    pass
```

**Output format details are in Section 8.**

---

## 8. Output Format Specifications

All outputs must match input resolution and be stackable as layers.

### 8.1 Silhouette Mask

| Property | Value |
|----------|-------|
| Format | PNG, 8-bit RGBA |
| Channels | Alpha channel only (RGB = black) |
| Person region | Alpha = 255 (opaque) |
| Background | Alpha = 0 (transparent) |
| Edge treatment | Anti-aliased (smooth contour boundary) |
| Resolution | Matches input |

### 8.2 Edge / Contour Map

| Property | Value |
|----------|-------|
| Format | PNG, 8-bit RGBA |
| Line pixels | Black (0, 0, 0, 255) |
| Background | Transparent (0, 0, 0, 0) |
| Line weight | Anti-aliased, consistent width |
| Detail level | Controlled by `detail_level` param (1-10) |
| Resolution | Matches input |

### 8.3 Depth Map

| Property | Value |
|----------|-------|
| Primary format | 16-bit grayscale PNG (0-65535 range) |
| Alternative format | 32-bit float EXR (linear float values) |
| Visualization format | 8-bit color PNG (perceptually uniform colormap) |
| Convention | Near = bright (high values), Far = dark (low values) |
| Person region | Full depth range |
| Background | Zero (when person isolation enabled) |
| Resolution | Matches input |

```python
import numpy as np
from PIL import Image
import cv2

def save_depth_16bit(depth_float: np.ndarray, path: str):
    """Save depth map as 16-bit PNG."""
    # Normalize to 0-65535 range
    d_min, d_max = depth_float[depth_float > 0].min(), depth_float.max()
    depth_norm = (depth_float - d_min) / (d_max - d_min)
    depth_16 = (depth_norm * 65535).astype(np.uint16)
    cv2.imwrite(path, depth_16)

def save_depth_exr(depth_float: np.ndarray, path: str):
    """Save depth map as 32-bit float EXR."""
    import OpenEXR, Imath
    h, w = depth_float.shape
    header = OpenEXR.Header(w, h)
    header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    exr = OpenEXR.OutputFile(path, header)
    exr.writePixels({"Z": depth_float.astype(np.float32).tobytes()})
    exr.close()

def save_depth_color(depth_float: np.ndarray, path: str):
    """Save depth visualization with perceptually uniform colormap."""
    depth_8 = cv2.normalize(depth_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(depth_8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(path, colored)
```

### 8.4 Topology Mesh

| Property | Value |
|----------|-------|
| 2D format | SVG (polyline paths, triangle wireframe) |
| 2.5D format | OBJ (vertices with Z from depth, faces from triangulation) |
| SVG properties | Scalable, opens in Illustrator/Inkscape |
| OBJ properties | Opens in Blender/MeshLab |

### 8.5 Layered Composite (PSD-Compatible)

| Property | Value |
|----------|-------|
| Format | PSD or OpenRaster (ORA) |
| Layer 1 | Silhouette mask |
| Layer 2 | Edge map (at requested detail level) |
| Layer 3 | Depth map (colorized visualization) |
| Layer 4 | Topology wireframe (rasterized from SVG) |
| Layer order | Topology on top, silhouette on bottom |
| Compatibility | Photoshop, Krita, GIMP (via ORA) |

```python
# PSD generation using pytoshop (lightweight PSD writer)
import pytoshop
from pytoshop.enums import ColorMode
import numpy as np

def create_psd_composite(layers_dict: dict, output_path: str, width: int, height: int):
    """Create PSD with multiple layers.

    Args:
        layers_dict: {"layer_name": numpy_rgba_array, ...}
        output_path: Path to save PSD file
        width, height: Image dimensions
    """
    psd = pytoshop.PsdFile(num_channels=4, height=height, width=width,
                           color_mode=ColorMode.rgb)
    for name, rgba_array in layers_dict.items():
        layer = pytoshop.layers.ChannelImageData.from_image(rgba_array)
        psd_layer = pytoshop.layers.LayerRecord(
            name=name, top=0, left=0, bottom=height, right=width,
            channels={-1: layer[3], 0: layer[0], 1: layer[1], 2: layer[2]}
        )
        psd.layer_and_mask_info.layer_info.layer_records.append(psd_layer)
    with open(output_path, "wb") as f:
        psd.write(f)
```

---

## 9. Determinism and Reproducibility

The product requires that identical inputs with identical parameters produce identical outputs.

### 9.1 Sources of Non-Determinism

| Component | Default Deterministic? | How to Fix |
|-----------|----------------------|------------|
| OpenCV classical (Canny, Sobel) | Yes | N/A |
| PyTorch feed-forward models | Mostly | Set `torch.use_deterministic_algorithms(True)` |
| PyTorch on MPS (Apple Silicon) | Partially | Some ops non-deterministic on MPS; use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` |
| PyTorch on CUDA | Partially | Set `torch.backends.cudnn.deterministic = True` |
| Marigold (diffusion) | No | Fix seed: `torch.Generator().manual_seed(seed)` |
| Delaunay triangulation (SciPy) | Yes | N/A |
| Random subsampling | No | Fix seed: `np.random.seed(seed)` |

### 9.2 Determinism Configuration

```python
import torch
import numpy as np
import random

def set_deterministic(seed: int = 42):
    """Configure all sources for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
```

### 9.3 Resolution Handling for 4096x4096

Most models have native processing resolutions below 4096x4096. Strategy:

| Model | Native Resolution | 4K Strategy |
|-------|------------------|-------------|
| RMBG 2.0 | 1024x1024 | Process at native, upscale mask with bilinear |
| Depth Anything V2 | 518x518 (encoder) | Process full image (internally handles tiling) |
| Apple Depth Pro | 1536x1536 | Process at native, upscale with bicubic |
| Marigold | 768x768 | Tiled processing or process at native + upscale |
| ControlNet Lineart | 512x512 | Process at native, upscale with nearest-neighbor |
| Canny/Sobel | Any | Process at full resolution natively |

**Upscaling masks and edges**: Use nearest-neighbor for binary masks (no interpolation artifacts). Use bilinear for soft masks and depth maps.

**Output guarantee**: All outputs are resized/upscaled to match input resolution before saving.

---

## 10. Tech Stack

### 10.1 Python Libraries

**Core**:
| Library | Purpose | Version |
|---------|---------|---------|
| `opencv-python` | Image processing, contours, classical edge detection | >= 4.8 |
| `numpy` | Array operations | >= 1.24 |
| `Pillow` | Image I/O and manipulation | >= 10.0 |
| `torch` | Deep learning inference | >= 2.1 |
| `torchvision` | Pre-trained models (DeepLabV3+) | >= 0.16 |

**AI Models**:
| Library | Purpose | Version |
|---------|---------|---------|
| `transformers` | Hugging Face model hub (Depth Anything, RMBG) | >= 4.36 |
| `controlnet_aux` | ControlNet preprocessors (HED, Lineart, DWPose) | >= 0.0.7 |
| `ultralytics` | YOLO detection + segmentation | >= 8.1 |
| `segment-anything-2` | SAM 2 segmentation | latest |
| `mediapipe` | Lightweight pose estimation | >= 0.10 |

**Topology / Visualization**:
| Library | Purpose | Version |
|---------|---------|---------|
| `scipy` | Delaunay triangulation | >= 1.11 |
| `scikit-image` | Marching squares, contour finding | >= 0.21 |
| `open3d` | 3D point cloud and mesh | >= 0.17 |
| `networkx` | Graph-based topology | >= 3.1 |
| `matplotlib` | Visualization and export | >= 3.8 |
| `svgwrite` | SVG vector output | >= 1.4 |

**Video / Audio**:
| Library | Purpose | Version |
|---------|---------|---------|
| FFmpeg (system) | Video frame extraction, reassembly, audio muxing (primary) | >= 5.0 |
| `ffmpeg-python` | **Optional**: Pythonic FFmpeg bindings for advanced codec control | >= 0.2.0 |
| `av` (PyAV) | **Optional**: FFmpeg Python bindings for in-process video I/O | >= 12.0 |
| `librosa` | Audio analysis (spectral features, beat detection, onset, RMS) | >= 0.10 |
| `soundfile` | Audio file I/O (backend for librosa) | >= 0.12 |
| `madmom` | **Optional**: RNN-based beat/onset detection (higher accuracy) | >= 0.17 |
| `PyYAML` | Pipeline config parsing | >= 6.0 |

**Output Formats**:
| Library | Purpose | Version |
|---------|---------|---------|
| `pytoshop` | PSD file generation (layered composites) | >= 1.2 |
| `OpenEXR` | EXR depth map export (32-bit float) | >= 3.2 |
| `tifffile` | Alternative layered TIFF output | >= 2023.4 |

### 10.2 Models: Local vs API

| Model | Local | API Available | Recommended |
|-------|-------|---------------|-------------|
| RMBG 2.0 | Yes (HF) | Replicate, fal.ai | Local |
| Depth Anything V2 | Yes (HF) | Replicate | Local |
| Apple Depth Pro | Yes (GitHub) | No | Local |
| SAM 2.1 | Yes (GitHub) | Replicate | Local |
| YOLO-seg | Yes (pip) | Roboflow | Local |
| DWPose | Yes (controlnet_aux) | No | Local |
| ControlNet preprocessors | Yes (controlnet_aux) | Replicate | Local |

**Recommendation**: Run all models locally. The inference is fast enough and avoids API costs/latency. Use APIs only for prototyping or if GPU is unavailable.

### 10.3 Hardware Requirements

**Minimum (CPU-only, limited)**:
- 16GB RAM
- Any modern CPU (Apple Silicon M1+ recommended for CPU inference)
- Can run: MediaPipe, OpenCV classical, U2-Net-P, Canny
- Cannot practically run: SAM 2, Depth Anything V2-Large, DWPose

**Recommended (GPU)**:
- 8GB+ VRAM GPU (RTX 3060 / 3070 or equivalent)
- 32GB system RAM
- Can run: All models at medium scale
- Depth Anything V2-Base, RMBG 2.0, YOLO-seg, DWPose

**Optimal (Large models)**:
- 16GB+ VRAM GPU (RTX 4080, 4090, A100)
- 64GB system RAM
- Can run: All models at full scale
- Depth Anything V2-Large/Giant, Apple Depth Pro, SAM 2.1 Hiera-Large

**Apple Silicon (M1 Pro/Max/Ultra, M2, M3, M4)**:
- Unified memory architecture -- share RAM between CPU and GPU.
- MPS (Metal Performance Shaders) backend in PyTorch.
- 16GB unified memory: equivalent to ~8GB VRAM scenario.
- 32GB+ unified memory: can run most large models.
- Good performance for inference but slower than dedicated NVIDIA GPUs.

### 10.4 Installation

```bash
# Core
pip install opencv-python numpy Pillow torch torchvision

# AI Models
pip install transformers controlnet_aux ultralytics mediapipe diffusers

# Depth Pro (Apple)
pip install git+https://github.com/apple/ml-depth-pro.git

# SAM 2
pip install git+https://github.com/facebookresearch/sam2.git

# Topology
pip install scipy scikit-image open3d networkx matplotlib svgwrite

# Video / Audio
# FFmpeg must be installed separately: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)
pip install librosa soundfile pyyaml

# Output formats
pip install pytoshop OpenEXR tifffile
```

### 10.5 Platform-Specific Notes

**macOS Apple Silicon (MPS) -- Primary Platform**:
```python
# Device selection
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

- PyTorch MPS support is mature as of PyTorch 2.1+.
- Some operations fall back to CPU silently on MPS -- monitor with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`.
- Unified memory means no explicit GPU memory limit, but large models can cause memory pressure.
- `torch.use_deterministic_algorithms(True)` may not be fully supported on MPS -- use `warn_only=True`.

**Linux CUDA -- Secondary Platform**:
- Full determinism supported with `cudnn.deterministic = True`.
- CUDA 11.8+ recommended for PyTorch 2.x.
- Monitor VRAM with `torch.cuda.memory_summary()`.

---

## 11. ADR Log

### ADR-001: Primary Person Segmentation Model

**Status**: Proposed

**Context**: Need to extract person from background as the first pipeline stage. Multiple options exist (RMBG 2.0, SAM 2.1, DeepLabV3+, YOLO-seg).

**Decision**: Use RMBG 2.0 as primary for single-person images. Use YOLO-seg for detection + RMBG 2.0 for refinement in multi-person scenarios. SAM 2.1 as fallback for difficult cases.

**Consequences**:
- Clean alpha mattes with smooth edges suitable for art.
- Non-commercial license for RMBG 2.0 (acceptable for personal art projects).
- YOLO provides fast person detection to feed into RMBG.

**Alternatives Considered**:
- SAM 2.1 alone: Higher quality but requires manual prompting or additional detector. Heavier.
- DeepLabV3+: Good but coarser edges than RMBG 2.0.
- YOLO-seg alone: Fast but lower mask quality at edges.

### ADR-002: Primary Depth Estimation Model

**Status**: Proposed

**Context**: Need depth maps with clean edges for art use. Options: MiDaS, ZoeDepth, Depth Anything V2, Apple Depth Pro.

**Decision**: Use Depth Anything V2-Large as the default. Offer Apple Depth Pro as optional upgrade for maximum edge sharpness.

**Consequences**:
- Best accuracy/speed balance with Depth Anything V2.
- Excellent edge quality for art applications.
- Relative depth is sufficient for art (absolute measurements not needed).
- Apple Depth Pro adds metric depth capability when needed.

**Alternatives Considered**:
- MiDaS v3.1: Superseded by Depth Anything V2 in accuracy.
- ZoeDepth: Fastest but lowest accuracy and poorest edge quality.
- Apple Depth Pro only: Highest edge quality but larger model and less community tooling.

### ADR-003: Edge Detection Strategy

**Status**: Proposed

**Context**: Need clean edges with minimal noise for art canvas. Must detect only person contours.

**Decision**: Two-pass approach: (1) Segment person first (Stage 1), (2) Apply edge detection only to segmented person. Primary edge method: ControlNet lineart preprocessor. Fallback: Canny on masked input.

**Consequences**:
- Mask-first approach eliminates all background noise by design.
- AI lineart produces art-quality lines.
- Canny fallback is CPU-friendly for real-time.
- Two-pass is slower than single-pass but dramatically cleaner.

**Alternatives Considered**:
- Single-pass HED/lineart on full image: Detects all edges including background. Requires post-hoc cleanup.
- Classical only: Fast but produces noisy, non-semantic edges.

### ADR-004: Detail Level Implementation

**Status**: Proposed

**Context**: Product requires a single integer parameter (1-10) controlling edge output density, from silhouette-only (1) to fine detail (10).

**Decision**: Multi-strategy approach:
- Levels 1-3: Use mask contour with Douglas-Peucker simplification (no interior edges).
- Levels 4-10: Use AI lineart on masked person, with threshold/area filtering scaled by level.
- All levels: Apply person mask first (background noise eliminated by construction).

**Consequences**:
- Clean separation between silhouette-only and structural edge modes.
- Deterministic: level maps to fixed thresholds and filter parameters.
- Smooth progression from simple to detailed.

**Alternatives Considered**:
- Single algorithm with only threshold adjustment: Hard to prevent texture noise at high detail without explicit filtering.
- Separate algorithm per level range: More complex to maintain but allows best algorithm per range.

### ADR-005: Output Format Strategy

**Status**: Proposed

**Context**: Product requires PSD-compatible layered composite, 16-bit depth PNG, EXR, and SVG topology.

**Decision**: Use `pytoshop` for PSD generation (lightweight, no Photoshop dependency). Use `OpenEXR` for EXR depth. Use `svgwrite` for SVG topology. All outputs at input resolution.

**Consequences**:
- Pure Python PSD generation without heavy dependencies.
- EXR requires OpenEXR C library (available via pip).
- All outputs are resolution-matched for direct layer stacking.

**Alternatives Considered**:
- GIMP Script-Fu for PSD: Requires GIMP installation.
- OpenRaster (ORA) instead of PSD: Less compatible with Photoshop but simpler format.
- TIFF with layers: Supported by some editors but less universal than PSD.

### ADR-006: Determinism Strategy

**Status**: Proposed

**Context**: Product requires same input + same parameters = same output.

**Decision**: Set global deterministic mode at pipeline startup. Fix all random seeds. Use `warn_only=True` for MPS where full determinism is not guaranteed. Document known non-deterministic edge cases.

**Consequences**:
- Deterministic on CUDA with `cudnn.deterministic = True`.
- Mostly deterministic on MPS (some floating-point ops may vary by ~1 bit).
- Marigold requires fixed seed for reproducibility.
- Performance impact: ~5-10% slower with deterministic mode enabled on CUDA.

### ADR-007: EffectRule as Core Abstraction

**Status**: Proposed

**Context**: The pipeline needs composable, chainable processing units that can carry state across video frames, be configured via YAML, and be extended without modifying core code.

**Decision**: Define an `EffectRule` abstract base class with `apply(frame, context) -> (frame, context)`. The return of both frame AND context allows rules to publish computed outputs (masks, depth maps) for downstream consumption. Rules are composed into a `Pipeline` that executes sequentially, threading both frame and context through. Inter-frame state is namespaced per rule inside `FrameContext.rule_state`. Rules are discovered via a plugin registry.

**Key design choices**:
- `apply()` returns `(frame, context)` -- not just `frame` -- so rules can publish outputs onto the shared context bus.
- Context carries `original_image` so downstream rules can always access the unmodified input.
- `context.previous_frames` (dict keyed by rule name) gives rules access to any other rule's output from the previous video frame, enabling cross-rule temporal blending.
- Rule state is namespaced by rule name (`context.get_rule_state(self.name())`), preventing rules from clobbering each other.
- Pipeline owns video I/O: `Pipeline(rules=[...], audio_path="music.mp3")` with `process_video(video_path, output_path)` for end-to-end processing.
- Linear chain only (no DAG/fan-out). A `FanOutRule` could be added later if needed.

**Consequences**:
- All processing (edge detection, depth, halftone, audio-reactive) uses the same interface.
- New rules can be added as Python modules without touching pipeline core.
- Chain configuration is purely declarative (YAML/JSON).
- State management is per-rule, namespaced inside FrameContext.
- Rule validation happens at config load time, before any processing.
- Context persists across frames, automatically carrying inter-frame state and previous_frames.

**Alternatives Considered**:
- `EffectRule` as `Protocol` instead of `ABC`: Protocol enables structural subtyping (duck typing) and avoids inheritance. However, ABC provides a clearer contract via `@abstractmethod`, default method implementations (`configure`, `reset_state`, `param_schema`), and better IDE support. ABC chosen because the default implementations provide significant value.
- `process(frame, context) -> frame` (context not returned): Simpler but prevents rules from publishing outputs for downstream consumption. Would require out-of-band communication.
- Functional pipeline (functions instead of classes): Simpler but no state management for stateful effects.
- Node graph (DAG instead of chain): More flexible but much more complex. Linear chaining is sufficient for now (product confirms no conditional branching needed).
- Middleware pattern: Similar to chain but with wrap-around semantics. Unnecessary complexity.

### ADR-008: Video I/O via FFmpeg Subprocess

**Status**: Proposed

**Context**: Need to extract frames from video, reassemble processed frames into output video, and mux a separate audio track into the output (AMV-style: video and audio are independent inputs).

**Decision**: Use FFmpeg via subprocess calls for all video I/O. No Python-native video encoding library. FFmpeg handles frame extraction, reassembly, and muxing the separate audio track into the output.

**Consequences**:
- FFmpeg is the industry standard for video processing. Handles all codecs, containers, and edge cases.
- FFmpeg must be installed separately (not pip-installable). Document in setup instructions.
- Subprocess calls are well-tested and avoid complex Python bindings.
- Frame exchange via PNG files on disk is simple and debuggable (can inspect any frame).
- Disk I/O overhead is acceptable for offline batch processing.
- Audio muxing uses FFmpeg's `-i audio.mp3 -shortest` to sync separate audio to rendered frames.

**Alternatives Considered**:
- OpenCV VideoCapture/VideoWriter: Simpler API but limited codec support and poor error handling.
- PyAV (FFmpeg Python bindings): Avoids subprocess overhead but adds complex dependency.
- moviepy: High-level but heavy dependency chain and limited codec control.

### ADR-009: Audio Analysis with librosa

**Status**: Proposed

**Context**: Audio-reactive rules need per-frame beat detection, spectral features, and onset strength aligned to video frames.

**Decision**: Use librosa as the primary audio analysis library for all spectral features, band energy, RMS, onset detection, and default beat tracking. Pre-compute all features before frame processing via `AudioSource`. Align features to video frames using `hop_length = sr / fps`. Optionally use madmom for beat/onset detection when higher accuracy is needed (see ADR-012).

**Consequences**:
- librosa is the standard Python audio analysis library. Well-documented, actively maintained.
- Pre-computation means audio analysis runs once, not per-frame. Fast and deterministic.
- hop_length alignment ensures features are time-synced to video frames.
- librosa works with MP3, WAV, FLAC, OGG (via soundfile/audioread backends).
- madmom available as optional enhancement for beat tracking (RNN + DBN, MIREX-winning accuracy).

**Alternatives Considered**:
- madmom only: Superior beat detection but limited spectral analysis. Not suitable as sole library.
- aubio: Fast C core but limited format support and less customizable. Speed advantage irrelevant for offline batch processing.
- essentia: More comprehensive but heavier dependency and less Pythonic API.
- torchaudio: PyTorch-native but less feature-rich for music analysis.
- Manual FFT: Too low-level. Would replicate what librosa already does.

### ADR-010: Temporal Coherence Default Strategy

**Status**: Proposed

**Context**: Processed video must not flicker. Multiple strategies exist (EMA, optical flow, state carry-forward).

**Decision**: Default strategy is per-rule state carry-forward (Strategy 3). Stateful rules maintain their own temporal state. TemporalSmoothRule (EMA) is available as an optional chain rule. Optical flow is available but not default due to cost.

**Consequences**:
- Each stateful rule is responsible for its own temporal coherence.
- Users can add TemporalSmoothRule to any chain for additional smoothing.
- No global temporal smoothing imposed on all rules (stateless rules like EdgeDetect are already deterministic and should not flicker on static input).
- Optical flow available as a specialized rule for motion-heavy scenes.

### ADR-011: AudioFrame as Typed Dataclass on FrameContext

**Status**: Proposed

**Context**: Audio-reactive rules need per-frame audio data. The original design used `audio_features: Optional[dict]` on FrameContext -- an untyped dict that rules had to blindly access by string keys. This is error-prone and undiscoverable.

**Decision**: Introduce `AudioFrame` as a typed dataclass (defined in Section 12.1). FrameContext carries `audio: Optional[AudioFrame]` instead of `audio_features: Optional[dict]`. AudioSource pre-computes `AudioFrame` objects and Pipeline injects them per frame.

**Consequences**:
- Type safety: rules access `context.audio.bands["bass"]` with IDE autocompletion and static analysis.
- AudioFrame fields are explicit: `rms`, `beat`, `beat_strength`, `spectrum`, `bands`, `onset_strength`, `spectral_centroid`, `tempo_bpm`.
- AudioSource encapsulates all librosa/madmom calls. Rules never import audio libraries directly.
- Pipeline owns audio injection (`process_frame` sets `context.audio`). Rules are decoupled from data loading.
- If no audio is available, `context.audio` is `None` and rules passthrough.

**Alternatives Considered**:
- Untyped dict: Flexible but error-prone. No IDE support, no validation.
- Audio features as separate FrameContext fields (e.g., `context.rms`, `context.beat`): Too many fields, clutters the context with audio-specific data that most rules don't need.
- Rules call AudioSource directly: Breaks the "rules are pure functions of (frame, context)" principle.

### ADR-012: madmom as Optional Beat Detection Enhancement

**Status**: Proposed

**Context**: librosa's beat tracker uses dynamic programming on onset strength. madmom uses RNN + DBN (recurrent neural network + dynamic Bayesian network), which won MIREX beat tracking evaluations and handles complex rhythms (syncopation, tempo changes) significantly better.

**Decision**: madmom is an optional dependency. `AudioSource(use_madmom_beats=True)` activates it. If madmom is not installed, AudioSource falls back to librosa's beat tracker with a warning.

**Consequences**:
- Default installation does not require madmom (simpler setup).
- Users processing music with complex rhythms can opt into madmom for better beat accuracy.
- The fallback is seamless -- same AudioFrame output structure regardless of which beat tracker was used.
- madmom requires Cython build, which can be tricky on some platforms. Making it optional avoids installation friction.

### ADR-013: Layer-Based Compositing Model

**Status**: Proposed

**Context**: A pure linear chain (each rule receives the previous rule's output) forces rules to process in strict sequence. When two rules both need the same clean source (e.g. both `HalfToneRandomizationRule` and `AudioReactiveEdgeRule` want clean edges), the second rule must either recompute the source or rely on `context.edge_map`. Reordering rules is fragile and makes intent unclear.

**Decision**: Add `context.layers: dict[str, np.ndarray]` to `FrameContext`, and `input_layer`/`output_layer` optional attributes to `EffectRule`. Rules that set `output_layer` are "branch rules" — they write to a named layer buffer and the flowing frame continues unchanged. `CompositeRule` merges named layers with configurable blend modes.

**Consequences**:
- Two rules can independently read from the same named layer (e.g. `"edges"`) without interfering.
- Compositing intent is explicit in the YAML config rather than implied by rule order.
- Backward compatible: rules without `input_layer`/`output_layer` behave identically to before (inline, transform the flowing frame).
- The `Pipeline.process_frame` loop gains a small amount of routing logic but remains O(N) in the number of rules.

**Alternatives considered**:
- Rely on `context.edge_map` + rule ordering: works but fragile and implicit.
- Full DAG execution model: more powerful but significantly more complex; deferred to a future version if needed.

---

## Appendix: Quick-Start Code

Minimal working pipeline combining the recommended components with product requirements (detail level, determinism, output formats).

```python
"""
Edge + Depth Pipeline for Art (Quick-Start)
Requires: pip install opencv-python numpy Pillow torch transformers controlnet_aux
Usage: python pipeline.py <image_path> [detail_level] [seed]
"""
import cv2
import numpy as np
import random
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline
from controlnet_aux import LineartDetector


def set_deterministic(seed: int = 42):
    """Configure all sources for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def filter_edges_by_detail(edge_img: np.ndarray, mask: np.ndarray, level: int) -> np.ndarray:
    """Filter edges by detail level. Levels 1-3 use silhouette only."""
    if level <= 3:
        # Silhouette only: contour from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(mask)
        # Douglas-Peucker simplification -- heavier at lower levels
        epsilon_frac = 0.01 - (level - 1) * 0.003  # 0.01, 0.007, 0.004
        for c in contours:
            eps = epsilon_frac * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            cv2.drawContours(result, [approx], -1, 255, thickness=2)
        return result
    else:
        # AI edges filtered by level
        threshold = int(255 * (1.0 - (level / 10.0) * 0.8))
        edges = ((edge_img > threshold) & (mask > 0)).astype(np.uint8) * 255
        # Remove small contours
        min_area = max(10, 500 - (level * 50))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(edges)
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(clean, [c], -1, 255, thickness=1)
        return clean


def process_image(image_path: str, detail_level: int = 5, seed: int = 42,
                  output_prefix: str = "output"):
    """Process a single image through the full pipeline."""
    set_deterministic(seed)
    input_size = None  # Will store original resolution

    # Load image
    image = Image.open(image_path).convert("RGB")
    input_size = image.size  # (width, height)

    # --- Stage 1: Person segmentation (RMBG 2.0) ---
    segmenter = hf_pipeline("image-segmentation", model="briaai/RMBG-2.0",
                            trust_remote_code=True)
    mask_image = segmenter(image)
    mask_array = np.array(mask_image)
    if mask_array.ndim == 3 and mask_array.shape[2] == 4:
        person_mask = mask_array[:, :, 3]
    else:
        person_mask = mask_array
    # Ensure mask matches input resolution
    if (person_mask.shape[1], person_mask.shape[0]) != input_size:
        person_mask = cv2.resize(person_mask, input_size, interpolation=cv2.INTER_NEAREST)
    binary_mask = (person_mask > 128).astype(np.uint8) * 255

    # Save silhouette (RGBA with alpha mask)
    silhouette_rgba = np.zeros((*binary_mask.shape, 4), dtype=np.uint8)
    silhouette_rgba[:, :, 3] = binary_mask
    Image.fromarray(silhouette_rgba).save(f"{output_prefix}_silhouette.png")

    # --- Stage 2: Edge detection with detail level ---
    image_np = np.array(image)
    masked = image_np.copy()
    masked[binary_mask == 0] = 0
    masked_pil = Image.fromarray(masked)

    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
    edges_raw = np.array(lineart(masked_pil).convert("L"))
    # Resize if needed
    if (edges_raw.shape[1], edges_raw.shape[0]) != input_size:
        edges_raw = cv2.resize(edges_raw, input_size, interpolation=cv2.INTER_NEAREST)

    edges_filtered = filter_edges_by_detail(edges_raw, binary_mask, detail_level)

    # Save as black lines on transparent (RGBA)
    edges_rgba = np.zeros((*edges_filtered.shape, 4), dtype=np.uint8)
    edges_rgba[edges_filtered > 0] = [0, 0, 0, 255]  # Black, opaque
    Image.fromarray(edges_rgba).save(f"{output_prefix}_edges.png")

    # --- Stage 3: Depth map ---
    depth_estimator = hf_pipeline("depth-estimation",
                                   model="depth-anything/Depth-Anything-V2-Large-hf")
    depth_result = depth_estimator(image)
    depth_map = np.array(depth_result["depth"], dtype=np.float32)
    # Resize to input resolution
    if (depth_map.shape[1], depth_map.shape[0]) != input_size:
        depth_map = cv2.resize(depth_map, input_size, interpolation=cv2.INTER_LINEAR)
    # Mask to person
    depth_map[binary_mask == 0] = 0
    # Save 16-bit PNG
    d_min = depth_map[depth_map > 0].min() if (depth_map > 0).any() else 0
    d_max = depth_map.max()
    depth_norm = np.zeros_like(depth_map)
    if d_max > d_min:
        depth_norm[depth_map > 0] = (depth_map[depth_map > 0] - d_min) / (d_max - d_min)
    depth_16 = (depth_norm * 65535).astype(np.uint16)
    cv2.imwrite(f"{output_prefix}_depth.png", depth_16)
    # Save color visualization
    depth_8 = (depth_norm * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(f"{output_prefix}_depth_color.png", depth_colored)

    print(f"Done. Outputs: {output_prefix}_silhouette.png, "
          f"{output_prefix}_edges.png (detail={detail_level}), "
          f"{output_prefix}_depth.png (16-bit), "
          f"{output_prefix}_depth_color.png")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    if path:
        process_image(path, detail_level=level, seed=seed)
    else:
        print("Usage: python pipeline.py <image_path> [detail_level=5] [seed=42]")
```

---

---

## 12. EffectRule System Architecture

The EffectRule system is the core abstraction that makes the pipeline composable and extensible. Every processing operation -- edge detection, depth mapping, halftone, audio-reactive displacement -- is implemented as an EffectRule.

### 12.1 EffectRule Interface

The central design decision is that `apply()` returns **both** the processed frame and an updated context. This allows rules to publish computed outputs (masks, depth maps, edge maps) into the context for downstream rules to consume.

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Literal, TypeAlias, TypeVar
import numpy as np
from numpy.typing import NDArray

# ─── Array type aliases ───────────────────────────────────────────────────────

RGBArray:   TypeAlias = NDArray[np.uint8]    # (H, W, 3)
MaskArray:  TypeAlias = NDArray[np.uint8]    # (H, W)    values 0 | 255
EdgeArray:  TypeAlias = NDArray[np.uint8]    # (H, W)
DepthArray: TypeAlias = NDArray[np.float32]  # (H, W)    0.0–1.0

# ─── Layer ────────────────────────────────────────────────────────────────────

class Layer(Enum):
    """Built-in layer keys. Distinct from str — cannot be confused with user keys."""
    ORIGINAL = "original"   # unmodified input frame (set by Pipeline)
    MASK     = "mask"       # person segmentation mask (PersonSegmentationRule)
    EDGES    = "edges"      # edge / contour map (EdgeDetectionRule)
    DEPTH    = "depth"      # depth map (DepthMapRule)
    SKELETON = "skeleton"   # pose keypoint visualisation (SkeletonOverlayRule)
    FINAL    = "final"      # final composite output (CompositeRule default)

# User-defined layers ("halftone", "audio_lines", …) are plain str.
# Built-in layers are Layer members. The union is the full key space.
LayerKey: TypeAlias = Layer | str

# ─── BlendMode ────────────────────────────────────────────────────────────────

BlendMode: TypeAlias = Literal["screen", "multiply", "add", "over", "normal"]

# ─── Rule state generic ───────────────────────────────────────────────────────

StateT = TypeVar("StateT")

# ─── AudioFrame ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AudioFrame:
    """Per-frame audio analysis aligned to a single video frame.

    Pre-computed by AudioSource before frame processing begins.
    Available to any rule via context.audio.
    """
    rms: float                      # Overall RMS loudness (0-1 normalized)
    beat: bool                      # True if this frame aligns with a detected beat onset
    beat_strength: float            # Beat confidence / onset envelope value (0-1)
    spectrum: np.ndarray            # Full FFT magnitude spectrum for this frame's window
    bands: dict[str, float]         # Frequency band energies: {"bass": 0-1, "mid": 0-1, "treble": 0-1}
    onset_strength: float           # General onset detection envelope value (0-1)
    spectral_centroid: float        # Spectral centroid (brightness, 0-1 normalized)
    tempo_bpm: float                # Estimated global tempo in BPM (same for all frames)


@dataclass
class FrameContext:
    """Shared state flowing through the rule chain.

    All shared data lives in typed dicts keyed by LayerKey (Layer | str).
    Built-in layers use Layer enum members; user-defined layers use plain str.
    No named fields for specific outputs — fully extensible without dataclass changes.
    """
    # --- Per-frame metadata ---
    frame_index: int
    total_frames: int           # 0 for single-image mode
    fps: float                  # 0 for single-image mode
    timestamp_sec: float

    # --- Audio (injected per frame by Pipeline, None if no audio source) ---
    audio: AudioFrame | None = None

    # --- Layer buffers: all shared rule outputs and compositing targets ---
    # Layer members for built-ins, str for user-defined ("halftone", "audio_lines", …)
    _layers:          dict[LayerKey, RGBArray] = field(default_factory=dict)
    _previous_frames: dict[LayerKey, RGBArray] = field(default_factory=dict)

    # --- Per-rule inter-frame state, keyed by rule class name ---
    _rule_state: dict[str, object] = field(default_factory=dict)

    # --- Arbitrary pass-through metadata ---
    metadata: dict[str, object] = field(default_factory=dict)

    # layer accessors
    def get_layer(self, key: LayerKey) -> RGBArray | None:
        return self._layers.get(key)

    def set_layer(self, key: LayerKey, value: RGBArray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Layer value must be ndarray, got {type(value)}")
        self._layers[key] = value

    def get_previous(self, key: LayerKey) -> RGBArray | None:
        return self._previous_frames.get(key)

    # rule state accessors — StateT inferred from the rule's Generic parameter
    def get_rule_state(self, rule: EffectRule[StateT]) -> StateT | None:
        return self._rule_state.get(rule.name())  # type: ignore[return-value]

    def set_rule_state(self, rule: EffectRule[StateT], state: StateT) -> None:
        self._rule_state[rule.name()] = state  # type: ignore[assignment]


class EffectRule(ABC, Generic[StateT]):
    """Base class for all composable processing rules.

    Generic over StateT — the type of this rule's inter-frame state.
    Stateless rules: EffectRule[None].

    Layer routing (optional):
    - input_layer: reads from context.get_layer(input_layer) instead of flowing frame.
    - output_layer: writes to context.set_layer(output_layer, …) — branch rule,
      flowing frame continues unchanged.
    - Neither set: inline rule, transforms the flowing frame directly.
    """

    input_layer:  LayerKey | None = None
    output_layer: LayerKey | None = None

    @abstractmethod
    def apply(self, frame: RGBArray, context: FrameContext) -> tuple[RGBArray, FrameContext]:
        """Process a single frame. Returns (processed_frame, updated_context)."""
        ...

    def initial_state(self) -> StateT | None:
        """Override to return the initial inter-frame state for this rule."""
        return None

    def configure(self, params: dict[str, object]) -> None:
        schema = self.param_schema()
        for key, value in params.items():
            if key in schema or hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Unknown parameter '{key}' for {self.__class__.__name__}. "
                    f"Valid: {list(schema.keys())}"
                )

    def reset_state(self, context: FrameContext) -> FrameContext:
        """Reset inter-frame state to initial. Called on scene cuts."""
        context.set_rule_state(self, self.initial_state())
        return context

    @property
    def is_stateful(self) -> bool:
        """Whether this rule carries state across frames."""
        return False

    @classmethod
    def param_schema(cls) -> dict:
        """Return parameter names, types, defaults, and descriptions."""
        return {}

    @classmethod
    def name(cls) -> str:
        """Rule name used for config files and state namespacing."""
        return cls.__name__
```

### 12.1.1 How FrameContext Flows Through the Chain

```
Frame_0 (original image)
   |
   +-- context = FrameContext(frame_index=0, original_image=frame, ...)
   |
   v
[PersonSegmentationRule.apply(frame, context)]
   |  -> sets context.layers[Layer.MASK] = <mask>
   |  -> returns (frame, context)     # frame unchanged (passthrough)
   v
[EdgeDetectionRule.apply(frame, context)]
   |  -> reads context.layers[Layer.MASK] to restrict edges to person
   |  -> sets context.layers[Layer.EDGES] = <edges>
   |  -> returns (edge_image, context) # frame IS the edge image now
   v
[HalfToneRandomizationRule.apply(frame, context)]
   |  -> reads context.get_rule_state("HalfToneRandomizationRule") for prev offsets
   |  -> writes updated offsets back to rule_state
   |  -> returns (halftoned_frame, context)
   v
[DepthMapRule.apply(frame, context)]
   |  -> reads context.layers[Layer.ORIGINAL] (NOT the halftoned frame) for depth
   |  -> sets context.layers[Layer.DEPTH] = <depth>
   |  -> returns (depth_composited_frame, context)
   v
Output frame
```

**Key design points**:

1. **Frame flows linearly**: Each rule receives the previous rule's output frame and can transform it. The frame is the "main data stream" through the chain.

2. **Context is a shared bus**: Rules publish intermediate outputs (masks, depth maps, edges) onto the context. Downstream rules can read these. This avoids rules needing to recompute upstream outputs.

3. **Original image always available**: `context.original_image` is always the unmodified input. Rules like DepthMapRule should compute depth from the original, not from a halftoned/edge-detected frame.

4. **State is namespaced per rule**: Each rule reads/writes its own slice of `context.rule_state` using `context.get_rule_state(self.name())`. This prevents rules from clobbering each other's state.

5. **Context persists across frames**: The pipeline reuses the same FrameContext object across frames (updating `frame_index`, `timestamp_sec`, etc.), which means `rule_state` and `previous_frames` carry forward automatically. At scene cuts, `reset_state()` is called on each rule.

6. **Previous-frame outputs accessible**: `context.previous_frames` is a dict keyed by rule name, holding each rule's output frame from the previous video frame. Pipeline updates this automatically. Rules can use this for temporal blending (e.g., comparing current edges to previous edges for motion-aware smoothing).

### 12.1.2 Architecture Questions Answered

**Q: Should rules be able to branch (fan-out/fan-in) or only linear chains?**
A: **Linear chains only** for now. This is a deliberate simplification. The product brief confirms linear chaining is sufficient (Open Question #7). Fan-out would require a DAG execution engine with dependency resolution, which adds significant complexity for minimal near-term benefit. If needed later, a `FanOutRule` could be implemented as a single rule that internally spawns parallel sub-chains and merges results.

**Q: How do rules access previous-frame outputs from other rules?**
A: Two mechanisms:
1. **`context.previous_frames`**: Dict keyed by rule name, holding each rule's output frame from the previous video frame. Updated automatically by Pipeline at the end of each frame. Example: `prev_edges = context.previous_frames.get("EdgeDetectionRule")` to access the previous frame's edge output for temporal blending or motion estimation.
2. **`context.rule_state`**: For a rule's own custom inter-frame state (not just the output frame). Example: `context.get_rule_state(self.name())["prev_offsets"]` for halftone offsets.

For current-frame outputs from upstream rules, read context attributes directly: `context.person_mask`, `context.depth_map`, `context.edge_map`, etc.

**Q: How are rules configured?**
A: Three mechanisms, in priority order:
1. **YAML/JSON config** (Section 12.5): Declarative, primary method.
2. **Constructor params**: `EdgeDetectionRule(algorithm="hed", detail_level=7)` for programmatic use.
3. **CLI overrides**: `--rule.EdgeDetect.detail_level=8` overrides config file values.

### 12.2 Chain Execution Engine

```python
from typing import List
import yaml
import importlib
import pkgutil


class Pipeline:
    """Ordered chain of EffectRules. The core execution engine.

    AMV-style: takes a list of rules and an optional audio_path.
    Audio is a separate file (MP3, WAV, FLAC) -- not from the video.
    """

    def __init__(self, rules: List[EffectRule],
                 audio_path: str | None = None):
        if not rules:
            raise ValueError("Pipeline requires at least one rule")
        self.rules = rules
        self.audio_path = audio_path      # Path to separate audio file (for lazy init)
        self.audio_source: AudioSource | None = None  # Initialized in process_video()

    def process_frame(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, FrameContext]:
        """Run frame through all rules in order, threading context through.

        Automatically injects audio data into context.audio if an AudioSource
        is attached and the current frame index has audio data.

        Args:
            frame: Input image for this frame.
            context: FrameContext with metadata and inter-frame state.

        Returns:
            (processed_frame, updated_context)
        """
        # Inject audio data for this frame (if available)
        if self.audio_source is not None:
            context.audio = self.audio_source.get_frame(context.frame_index)
        else:
            context.audio = None

        # Seed layers with "original" so rules can always access the unmodified frame
        context.layers["original"] = frame

        result = frame
        current_outputs = {}  # Collect each rule's output for previous_frames

        for rule in self.rules:
            # Layer routing: determine which frame this rule receives
            if rule.input_layer is not None:
                input_frame = context.layers.get(rule.input_layer, result)
            else:
                input_frame = result

            output_frame, context = rule.apply(input_frame, context)

            # Record this rule's output for next frame's previous_frames
            current_outputs[rule.name()] = output_frame

            if rule.output_layer is not None:
                # Branch rule: store output to named layer, flowing frame unchanged
                context.layers[rule.output_layer] = output_frame
            else:
                # Inline rule: advance the flowing frame
                result = output_frame

        # Update previous_frames for next frame (rules can read these for temporal blending)
        context.previous_frames = current_outputs

        return result, context

    def reset_all_state(self, context: FrameContext) -> FrameContext:
        """Reset inter-frame state on all stateful rules (scene cut)."""
        for rule in self.rules:
            if rule.is_stateful:
                context = rule.reset_state(context)
        return context

    def get_checkpoint(self, context: FrameContext) -> dict:
        """Serialize all rule states from context for checkpointing."""
        return {
            rule.name(): context.get_rule_state(rule.name())
            for rule in self.rules
            if rule.is_stateful
        }

    def restore_checkpoint(self, context: FrameContext, checkpoint: dict) -> FrameContext:
        """Restore all rule states into context from checkpoint."""
        for rule_name, state in checkpoint.items():
            context.set_rule_state(rule_name, state)
        return context

    def process_video(self, video_path: str, output_path: str,
                      seed: int = 42, codec: str = "h264") -> None:
        """End-to-end video processing: extract frames, apply rules, reassemble.

        This is the main entry point for AMV-style processing:
        video_path is the input video, audio_path was set at construction.
        The output video has the separate audio track muxed in.

        Args:
            video_path: Path to input video (MP4, MOV, etc.)
            output_path: Path for output video (MP4)
            seed: Random seed for determinism
            codec: Output codec ("h264" or "h265")
        """
        set_deterministic(seed)

        fps = get_video_fps(video_path)
        temp_dir = "tmp_frames"
        output_frame_dir = "tmp_processed"
        total_frames = extract_frames(video_path, temp_dir)

        # Pre-analyze separate audio file (if provided)
        if self.audio_path:
            self.audio_source = AudioSource(self.audio_path, total_frames, fps)

        # Scene cut detection
        scene_cuts = detect_scene_cuts(temp_dir, total_frames)

        # Initialize persistent FrameContext
        context = FrameContext(
            frame_index=0, total_frames=total_frames, fps=fps,
            timestamp_sec=0.0, original_image=None,
        )

        os.makedirs(output_frame_dir, exist_ok=True)

        for i in range(total_frames):
            frame = cv2.imread(os.path.join(temp_dir, f"frame_{i+1:06d}.png"))

            if i in scene_cuts:
                context = self.reset_all_state(context)
                context.previous_frames = {}  # Clear cross-rule temporal data

            context.frame_index = i
            context.timestamp_sec = i / fps
            context.original_image = frame
            # Clear per-frame shared outputs
            context.person_mask = None
            context.depth_map = None
            context.edge_map = None
            context.skeleton = None
            context.contours = None

            processed, context = self.process_frame(frame, context)
            # Use "final" layer if it exists (layer-based compositing), else flowing frame
            output = context.layers.get("final", processed)
            cv2.imwrite(os.path.join(output_frame_dir, f"frame_{i+1:06d}.png"), output)

            # Periodic checkpoint
            if (i + 1) % 100 == 0:
                cp = {"next_frame": i + 1, "rule_states": self.get_checkpoint(context)}
                with open(output_path + ".checkpoint", "wb") as f:
                    pickle.dump(cp, f)

        # Reassemble video and mux separate audio track
        reassemble_video(output_frame_dir, output_path, fps, codec, self.audio_path)

    @classmethod
    def from_config(cls, config: dict, registry: "RuleRegistry") -> "Pipeline":
        """Build pipeline from parsed config dict (including audio_path)."""
        rules = []
        for rule_spec in config.get("rules", []):
            rule_name = rule_spec["name"]
            params = rule_spec.get("params", {})
            rule_cls = registry.get(rule_name)
            rule = rule_cls()
            rule.configure(params)
            rules.append(rule)
        audio_path = config.get("pipeline", {}).get("audio")
        return cls(rules, audio_path=audio_path)
```

**Usage pattern (AMV-style: video + separate audio file)**:
```python
# Simplest: construct and run end-to-end
pipeline = Pipeline(
    rules=[
        PersonSegmentationRule(model="rmbg2"),
        EdgeDetectionRule(algorithm="lineart_realistic", detail_level=6),
        AudioReactiveEdgeRule(sensitivity=0.8, decay_rate=0.85),
        DepthMapRule(model="depth_anything_v2_large"),
    ],
    audio_path="music.mp3",  # Separate audio file, NOT from video
)

# Single call: extracts frames, processes, reassembles with audio muxed in
pipeline.process_video("dance.mp4", "output.mp4")
```

**Low-level usage (manual frame loop)**:
```python
pipeline = Pipeline(rules=[...], audio_path="music.mp3")
# Manually init AudioSource for custom frame range
pipeline.audio_source = AudioSource("music.mp3", total_frames=total, fps=fps)

context = FrameContext(frame_index=0, total_frames=total, fps=fps,
                       timestamp_sec=0.0, original_image=None)

for i, frame in enumerate(video_reader):
    context.frame_index = i
    context.timestamp_sec = i / fps
    context.original_image = frame
    output_frame, context = pipeline.process_frame(frame, context)
    video_writer.write(output_frame)
```

### 12.3 Rule Registry and Plugin Discovery

```python
class RuleRegistry:
    """Discovers and registers EffectRule implementations."""

    def __init__(self):
        self._rules: dict[str, type[EffectRule]] = {}

    def register(self, rule_cls: type[EffectRule]) -> None:
        self._rules[rule_cls.name()] = rule_cls

    def get(self, name: str) -> type[EffectRule]:
        if name not in self._rules:
            available = ", ".join(sorted(self._rules.keys()))
            raise ValueError(f"Unknown rule '{name}'. Available: {available}")
        return self._rules[name]

    def list_rules(self) -> list[dict]:
        """Return list of registered rules with metadata."""
        return [
            {"name": cls.name(), "params": cls.param_schema(), "stateful": cls().is_stateful}
            for cls in self._rules.values()
        ]

    def discover_plugins(self, package_name: str = "amv_rules") -> None:
        """Auto-discover EffectRule subclasses in a package."""
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package_name}.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and issubclass(attr, EffectRule)
                        and attr is not EffectRule):
                    self.register(attr)
```

### 12.4 Example Rule Implementations

The following rules map to the pipeline stages defined in Section 7, now expressed as EffectRules.

**PersonSegmentationRule (Stage 1)**
```python
class PersonSegmentationRule(EffectRule):
    """Extract person mask and publish to context. Frame is passed through unchanged."""

    def __init__(self):
        self.model = "rmbg2"  # "rmbg2", "sam2", "yolo_seg", "deeplabv3"
        self._segmenter = None

    def apply(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, FrameContext]:
        if self._segmenter is None:
            self._segmenter = self._load_model()

        mask = self._segmenter.predict(frame)
        context.person_mask = mask  # Publish for downstream rules
        return frame, context  # Frame passes through unmodified

    def _load_model(self):
        # Lazy-load model based on self.model config
        ...

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "model": {"type": "str", "default": "rmbg2",
                      "choices": ["rmbg2", "sam2", "yolo_seg", "deeplabv3", "pointrend"]},
        }
```

**EdgeDetectionRule (Stage 2)**
```python
class EdgeDetectionRule(EffectRule):
    """Detect edges with configurable algorithm and detail level.
    Publishes edge_map to context. Returns edge image as frame."""

    def __init__(self):
        self.algorithm = "lineart"
        self.detail_level = 5

    def apply(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, FrameContext]:
        # Use original image for edge detection (not a previously processed frame)
        source = context.original_image
        mask = context.person_mask

        edges = self._detect_edges(source, mask)
        context.edge_map = edges  # Publish for downstream rules
        return edges, context     # Frame becomes the edge image

    def _detect_edges(self, frame, mask):
        # Delegates to Section 6 detail level logic
        ...

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "algorithm": {"type": "str", "default": "lineart",
                          "choices": ["canny", "hed", "lineart", "lineart_anime",
                                      "pidinet", "dexined"]},
            "detail_level": {"type": "int", "default": 5, "min": 1, "max": 10},
        }
```

**DepthMapRule (Stage 3)**
```python
class DepthMapRule(EffectRule):
    """Generate depth map and publish to context.
    Returns depth visualization as frame."""

    def __init__(self):
        self.model = "depth_anything_v2_large"
        self.output_mode = "colormap"  # "colormap", "grayscale", "composite"
        self.composite_blend = 0.5
        self._estimator = None

    def apply(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, FrameContext]:
        if self._estimator is None:
            self._estimator = self._load_model()

        # Always compute depth from original image
        depth = self._estimator.predict(context.original_image)

        # Mask to person if available
        if context.person_mask is not None:
            depth[context.person_mask == 0] = 0

        context.depth_map = depth  # Publish raw depth for downstream rules

        # Return visualization based on output_mode
        if self.output_mode == "composite":
            vis = self._composite(frame, depth)
        elif self.output_mode == "colormap":
            vis = self._colormap(depth)
        else:
            vis = self._grayscale(depth)
        return vis, context

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "model": {"type": "str", "default": "depth_anything_v2_large",
                      "choices": ["midas", "zoedepth", "depth_anything_v2_small",
                                  "depth_anything_v2_large", "depth_pro", "marigold"]},
            "output_mode": {"type": "str", "default": "colormap",
                            "choices": ["colormap", "grayscale", "composite"]},
            "composite_blend": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
        }
```

**HalfToneRandomizationRule (Stateful)**
```python
class HalfToneRandomizationRule(EffectRule):
    """Halftone effect with smooth temporal evolution across frames."""

    def __init__(self):
        self.density = 0.5
        self.dot_size = 4
        self.evolution_rate = 0.1  # How fast dots evolve between frames

    def apply(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, FrameContext]:
        state = context.get_rule_state(self.name())
        prev_offsets = state.get("prev_offsets")

        h, w = frame.shape[:2]
        grid_h, grid_w = h // self.dot_size, w // self.dot_size

        # Generate new target offsets
        rng = np.random.RandomState(42 + context.frame_index)
        new_offsets = rng.randn(grid_h, grid_w) * self.density

        # Smooth evolution: blend with previous offsets
        if prev_offsets is not None and prev_offsets.shape == new_offsets.shape:
            offsets = prev_offsets + self.evolution_rate * (new_offsets - prev_offsets)
        else:
            offsets = new_offsets

        # Store updated offsets in context state
        context.set_rule_state(self.name(), {"prev_offsets": offsets})

        result = self._apply_halftone(frame, offsets)
        return result, context

    def reset_state(self, context: FrameContext) -> FrameContext:
        context.set_rule_state(self.name(), {})
        return context

    @property
    def is_stateful(self) -> bool:
        return True

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "density": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "dot_size": {"type": "int", "default": 4, "min": 1, "max": 20},
            "evolution_rate": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
        }
```

**SkeletonOverlayRule (Stage 3b)**
```python
class SkeletonOverlayRule(EffectRule):
    """Render DWPose skeleton overlay. Publishes keypoints to context."""

    def __init__(self):
        self.model = "dwpose"  # "dwpose", "mediapipe", "openpose"
        self.line_thickness = 2
        self.overlay_mode = "overlay"  # "overlay" (on top of frame) or "standalone"

    def apply(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, FrameContext]:
        keypoints = self._detect_pose(context.original_image)
        context.skeleton = keypoints  # Publish for downstream rules

        if self.overlay_mode == "standalone":
            canvas = np.zeros_like(frame)
        else:
            canvas = frame.copy()

        result = self._draw_skeleton(canvas, keypoints)
        return result, context

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "model": {"type": "str", "default": "dwpose",
                      "choices": ["dwpose", "mediapipe", "openpose"]},
            "line_thickness": {"type": "int", "default": 2, "min": 1, "max": 10},
            "overlay_mode": {"type": "str", "default": "overlay",
                             "choices": ["overlay", "standalone"]},
        }
```

**TopologyRule (Stage 4)**
```python
class TopologyRule(EffectRule):
    """Generate topology mesh from edges and/or depth. Publishes contours to context."""

    def __init__(self):
        self.method = "delaunay"  # "delaunay", "contour", "marching_squares"
        self.target_triangles = 500
        self.use_depth = False  # Enable 2.5D with depth values

    def apply(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, FrameContext]:
        edges = context.edge_map
        depth = context.depth_map if self.use_depth else None

        if edges is None:
            return frame, context  # No edges available, passthrough

        contours, mesh = self._generate_topology(edges, depth)
        context.contours = contours  # Publish for downstream rules

        result = self._render_topology(frame, mesh)
        return result, context

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "method": {"type": "str", "default": "delaunay",
                       "choices": ["delaunay", "contour", "marching_squares"]},
            "target_triangles": {"type": "int", "default": 500, "min": 10, "max": 10000},
            "use_depth": {"type": "bool", "default": False},
        }
```

**TemporalSmoothRule (Stateful, general-purpose)**
```python
class TemporalSmoothRule(EffectRule):
    """Exponential moving average smoothing across frames.
    Can be inserted anywhere in the chain to smooth any rule's output."""

    def __init__(self):
        self.blend_factor = 0.7  # Weight of current frame

    def apply(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, FrameContext]:
        state = context.get_rule_state(self.name())
        prev_frame = state.get("prev_frame")

        if prev_frame is None or prev_frame.shape != frame.shape:
            context.set_rule_state(self.name(), {"prev_frame": frame.copy()})
            return frame, context

        blended = cv2.addWeighted(
            frame, self.blend_factor,
            prev_frame, 1.0 - self.blend_factor, 0
        )
        context.set_rule_state(self.name(), {"prev_frame": blended.copy()})
        return blended, context

    def reset_state(self, context: FrameContext) -> FrameContext:
        context.set_rule_state(self.name(), {})
        return context

    @property
    def is_stateful(self) -> bool:
        return True

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "blend_factor": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0},
        }
```

### 12.4.1 Summary: All Built-In Rules

| Rule Name | Pipeline Stage | Stateful | Publishes to Context | Reads from Context |
|-----------|---------------|----------|---------------------|--------------------|
| PersonSegmentationRule | 1 (mask) | No | `person_mask` | -- |
| EdgeDetectionRule | 2 (edges) | No | `edge_map` | `person_mask` |
| DepthMapRule | 3 (depth) | No | `depth_map` | `person_mask`, `original_image` |
| SkeletonOverlayRule | 3b (pose) | No | `skeleton` | `original_image` |
| TopologyRule | 4 (topology) | No | `contours` | `edge_map`, `depth_map` |
| HalfToneRandomizationRule | artistic | Yes | -- | -- |
| TemporalSmoothRule | utility | Yes | -- | -- |
| AudioReactiveEdgeRule | audio | Yes | -- | `person_mask`, `audio` |
| ThresholdBinarizeRule | utility | No | -- | -- |
| DepthCompositeRule | utility | No | -- | `depth_map` |
| ContourOverlayRule | utility | No | -- | `contours` |

### 12.5 CompositeRule

`CompositeRule` is a built-in branch rule that merges multiple named layers into one using a configurable blend mode. It always reads from `context.layers` and writes to `context.layers` — the flowing frame is never modified.

```python
class CompositeRule(EffectRule[None]):
    """Merge multiple named layers with blend modes.

    Always a branch rule: reads via context.get_layer(), writes via context.set_layer().
    The flowing frame passes through unchanged.
    """

    def __init__(
        self,
        source_layers: list[LayerKey],
        blend_mode: BlendMode = "screen",
        output_layer: LayerKey = Layer.FINAL,
        opacities: list[float] | None = None,
    ) -> None:
        self.source_layers = source_layers
        self.blend_mode = blend_mode
        self.output_layer = output_layer
        self.opacities = opacities

    def apply(self, frame: RGBArray, context: FrameContext) -> tuple[RGBArray, FrameContext]:
        ops = self.opacities or [1.0] * len(self.source_layers)
        result: NDArray[np.float32] | None = None
        for key, opacity in zip(self.source_layers, ops):
            layer = context.get_layer(key)
            if layer is None:
                continue
            layer_f = (layer.astype(np.float32) / 255.0) * opacity
            if result is None:
                result = layer_f
            elif self.blend_mode == "screen":
                result = 1.0 - (1.0 - result) * (1.0 - layer_f)
            elif self.blend_mode == "multiply":
                result = result * layer_f
            elif self.blend_mode == "add":
                result = np.clip(result + layer_f, 0.0, 1.0)
            elif self.blend_mode == "over":
                result = np.maximum(result, layer_f)
            else:  # "normal"
                result = layer_f

        if result is not None:
            context.set_layer(self.output_layer, (np.clip(result, 0, 1) * 255).astype(np.uint8))
        return frame, context  # flowing frame unchanged

    @classmethod
    def name(cls) -> str:
        return "CompositeRule"
```

**Blend modes**:
| Mode | Formula | Art Use |
|------|---------|---------|
| `screen` | `1-(1-a)(1-b)` | Merge light elements on dark bg (lines, glows) |
| `multiply` | `a*b` | Darken, shadows, depth |
| `add` | `clip(a+b)` | Additive glow, energy effects |
| `over` | `max(a,b)` | Simple overlay, no interaction |
| `normal` | `b` | Replace with top layer |

**Inline vs branch rule summary**:
- **Inline rule** (no `output_layer`): receives and transforms the flowing frame. Next rule receives this rule's output.
- **Branch rule** (has `output_layer`): writes to `context.layers[output_layer]`; flowing frame continues unchanged. Enables two rules to read the same source independently.
- **CompositeRule**: always a branch rule — only touches `context.layers`.

### 12.6 Configuration Format (YAML)

**Linear chain (simple, no layer routing)**:
```yaml
pipeline:
  input: "dance.mp4"
  audio: "music.mp3"
  output: "processed.mp4"
  seed: 42
  output_codec: "h264"

rules:
  - name: PersonSegmentationRule
    params: {}
  - name: EdgeDetectionRule
    params:
      algorithm: "lineart_realistic"
      detail_level: 6
  - name: HalfToneRandomizationRule
    params:
      dot_size: 4
      evolution_rate: 0.1
  - name: AudioReactiveEdgeRule
    params:
      sensitivity: 0.7
      decay_rate: 0.85
      displacement_scale: 15
```

**Layer-based compositing (halftone + audio lines on separate layers)**:
```yaml
rules:
  - name: PersonSegmentationRule
    output_layer: mask            # branch: writes mask, frame unchanged

  - name: EdgeDetectionRule
    output_layer: edges           # branch: writes edges, frame unchanged

  - name: HalfToneRandomizationRule
    input_layer: edges            # reads clean edges (not flowing frame)
    output_layer: halftone        # branch: writes to halftone layer

  - name: AudioReactiveEdgeRule
    input_layer: edges            # ALSO reads clean edges independently
    output_layer: audio_lines     # branch: writes to audio_lines layer

  - name: CompositeRule
    params:
      source_layers: [halftone, audio_lines]
      blend_mode: screen
      output_layer: final         # result in context.layers["final"]
```

---

## 13. Video Pipeline Architecture

### 13.1 Pipeline Overview

```
Input Video (MP4/MOV) + Separate Audio File (MP3/WAV/FLAC) + Config (YAML)
       |                        |
       v                        v
+----------------------------+  +----------------------------+
| Video Pre-Processing       |  | Audio Pre-Analysis         |
| - Frame extraction (FFmpeg)|  | - AudioSource(audio_path)  |
| - Scene cut detection      |  | - librosa spectral/beat    |
+----------------------------+  | - Per-frame AudioFrame[]   |
       |                        +----------------------------+
       |                                |
       v                                v
+----------------------------------------------------+
| Per-Frame Processing Loop                          |
| For frame_i in range:                              |
|   1. Load frame from disk                          |
|   2. Update FrameContext (frame_index, timestamp)   |
|   3. Pipeline injects context.audio from AudioSource|
|   4. Scene cut? Reset state                        |
|   5. Pipeline.process_frame (all rules in sequence)|
|   6. Save processed frame                          |
|   7. Checkpoint (periodic)                         |
+----------------------------------------------------+
       |
       v
+----------------------------+
| Post-Processing            |
| - Reassemble video (FFmpeg)|
| - Mux separate audio track |  <-- Audio file muxed into output
| - Cleanup temp frames      |
+----------------------------+
       |
       v
Output: processed.mp4 (with audio) + frames/ (optional)
```

**AMV-style workflow**: The video and audio are independent inputs. The video contains footage of a person (e.g., dance footage, anime clips). The audio is a separate music track (e.g., MP3). They are combined during post-processing when FFmpeg muxes the audio track into the rendered output video. Audio analysis drives audio-reactive effects (e.g., `AudioReactiveEdgeRule`) but does not modify the audio itself.

### 13.2 Frame Extraction (FFmpeg)

```python
import subprocess
import os


def extract_frames(video_path: str, output_dir: str, frame_range: tuple = None,
                   every_nth: int = 1) -> int:
    """Extract frames from video using FFmpeg.

    Args:
        video_path: Path to input video.
        output_dir: Directory for extracted PNG frames.
        frame_range: Optional (start, end) frame indices.
        every_nth: Extract every Nth frame (1 = all frames).

    Returns:
        Number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = ["ffmpeg", "-i", video_path]

    if frame_range:
        start, end = frame_range
        # Get FPS to compute timestamps
        fps = get_video_fps(video_path)
        cmd += ["-ss", str(start / fps)]
        if end >= 0:
            cmd += ["-to", str((end + 1) / fps)]

    if every_nth > 1:
        cmd += ["-vf", f"select='not(mod(n\\,{every_nth}))'", "-vsync", "vfr"]

    cmd += [os.path.join(output_dir, "frame_%06d.png")]

    subprocess.run(cmd, check=True, capture_output=True)
    return len([f for f in os.listdir(output_dir) if f.endswith(".png")])


def get_video_fps(video_path: str) -> float:
    """Get video frame rate using FFprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", video_path],
        capture_output=True, text=True
    )
    import json
    data = json.loads(result.stdout)
    for stream in data["streams"]:
        if stream["codec_type"] == "video":
            r = stream["r_frame_rate"].split("/")
            return float(r[0]) / float(r[1])
    raise ValueError("No video stream found")


def reassemble_video(frame_dir: str, output_path: str, fps: float,
                     codec: str = "h264", audio_path: str = None):
    """Reassemble frames into video, muxing the separate audio track.

    The audio_path is the user's standalone music file (MP3/WAV/FLAC),
    NOT extracted from the input video. FFmpeg muxes it into the output
    so the final rendered video plays with the music.
    """
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%06d.png"),
    ]

    if audio_path:
        cmd += ["-i", audio_path, "-shortest"]

    codec_flag = "libx264" if codec == "h264" else "libx265"
    cmd += [
        "-c:v", codec_flag, "-pix_fmt", "yuv420p",
        "-crf", "18",  # High quality
    ]

    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", "192k"]

    cmd += [output_path]
    subprocess.run(cmd, check=True, capture_output=True)
```

### 13.3 Main Processing Loop

The primary entry point is `Pipeline.process_video()` (defined in Section 12.2). It handles frame extraction, audio analysis, the per-frame processing loop, checkpointing, and reassembly with audio muxing.

**Config-driven usage**:
```python
def main(config_path: str):
    """Entry point: load config and run pipeline."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    registry = RuleRegistry()
    registry.discover_plugins()
    pipeline = Pipeline.from_config(config, registry)

    video_path = config["pipeline"]["input"]
    output_path = config["pipeline"]["output"]
    seed = config["pipeline"].get("seed", 42)
    codec = config["pipeline"].get("output_codec", "h264")

    pipeline.process_video(video_path, output_path, seed=seed, codec=codec)
```

**Programmatic usage**:
```python
pipeline = Pipeline(
    rules=[
        PersonSegmentationRule(model="rmbg2"),
        EdgeDetectionRule(algorithm="lineart_realistic", detail_level=6),
        AudioReactiveEdgeRule(sensitivity=0.8, decay_rate=0.85),
    ],
    audio_path="music.mp3",  # Separate audio file
)
pipeline.process_video("dance.mp4", "output.mp4")
```

The `process_video()` method internally:
1. Extracts frames from video via FFmpeg (Section 13.2)
2. Creates `AudioSource` from `self.audio_path` (if provided)
3. Detects scene cuts (Section 13.4)
4. Initializes a persistent `FrameContext` with `previous_frames`, `rule_state`
5. Loops over frames: injects audio, resets on scene cuts, runs all rules, records outputs to `context.previous_frames`, checkpoints every 100 frames
6. Reassembles output video via FFmpeg, muxing the separate audio track

### 13.4 Scene Cut Detection

```python
def detect_scene_cuts(frame_dir: str, total_frames: int,
                      threshold: float = 30.0) -> set[int]:
    """Detect scene cuts by measuring inter-frame difference.

    Uses histogram difference between consecutive frames. A large
    difference indicates a scene cut.

    Args:
        frame_dir: Directory containing extracted frames.
        total_frames: Total number of frames.
        threshold: Histogram difference threshold for scene cut.

    Returns:
        Set of frame indices where scene cuts occur.
    """
    scene_cuts = set()
    prev_hist = None

    for i in range(total_frames):
        frame_path = os.path.join(frame_dir, f"frame_{i+1:06d}.png")
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold / 100.0:  # Normalize threshold
                scene_cuts.add(i)

        prev_hist = hist

    return scene_cuts
```

### 13.5 Checkpointing and Resume

**Strategy**: Every N frames (default: 100), serialize:
- The index of the next frame to process
- All stateful rule states (via `get_state()`)
- Saved as a pickle file alongside the output

**On resume**:
- Load checkpoint
- Restore rule states via `set_state()`
- Continue from the stored frame index
- Already-processed frames in `output_frame_dir` are not overwritten

**Limitation**: If the config or input changes between runs, the checkpoint is invalid. A hash of the config should be included in the checkpoint for validation.

### 13.6 Performance Budget (Video)

Product requirement: 300 frames (10s @ 30fps) < 2 hours on consumer GPU.

| Stage | Per-Frame (GPU) | 300 Frames | Notes |
|-------|----------------|------------|-------|
| Frame extraction | ~2ms | ~0.6s | FFmpeg, fast |
| Person mask (RMBG 2.0) | ~100ms | ~30s | |
| Depth map (DA V2-L) | ~220ms | ~66s | |
| Edge detection (Lineart) | ~80ms | ~24s | |
| Rule chain (typical 3 rules) | ~50ms | ~15s | Depends on rules |
| Frame save (PNG) | ~20ms | ~6s | |
| **Total per frame** | **~470ms** | **~141s** | ~2.4 minutes |
| Video reassembly | -- | ~30s | FFmpeg |
| Audio analysis | -- | ~5s | One-time |
| **Grand total** | | **~3 minutes** | Well within 2 hours |

At 4K resolution, multiply by ~2-3x due to model upscaling overhead: ~6-9 minutes for 300 frames. Still well within budget.

---

## 14. Temporal Coherence Strategies

Temporal coherence ensures processed video looks smooth when played back. Without it, per-frame processing produces flickering.

### 14.1 Strategy 1: Temporal Smoothing (EMA)

**How it works**: Exponential moving average of processed frames. Each output frame blends with the previous output.

```
output[t] = alpha * processed[t] + (1 - alpha) * output[t-1]
```

**Parameters**: `alpha` (blend_factor): 0.0-1.0. Lower = smoother but more ghosting.

**Best for**: Reducing flicker in edge maps, depth maps, any per-pixel output.

**Limitations**: Introduces ghosting/trailing on fast motion. Not suitable for rapid scene changes.

**Implementation**: `TemporalSmoothRule` (Section 12.4).

### 14.2 Strategy 2: Optical Flow Registration

**How it works**: Compute optical flow between frame N and frame N+1. Warp the previous output according to the flow, then blend with the current output.

```python
import cv2

def flow_warp_blend(prev_output, curr_output, prev_frame, curr_frame, alpha=0.5):
    """Warp previous output using optical flow, then blend."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                         None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Warp previous output to align with current frame
    h, w = prev_output.shape[:2]
    flow_map = np.column_stack([
        (np.arange(w)[np.newaxis, :] + flow[..., 0]).flatten(),
        (np.arange(h)[:, np.newaxis] + flow[..., 1]).flatten()
    ]).reshape(h, w, 2).astype(np.float32)
    warped_prev = cv2.remap(prev_output, flow_map[..., 0], flow_map[..., 1],
                             cv2.INTER_LINEAR)

    # Blend warped previous with current
    return cv2.addWeighted(curr_output, alpha, warped_prev, 1 - alpha, 0)
```

**Best for**: Motion-heavy scenes where simple EMA causes ghosting.

**Limitations**: Optical flow computation is expensive (~50-100ms per frame). Flow errors at occlusion boundaries cause artifacts.

### 14.3 Strategy 3: Explicit State Carry-Forward

**How it works**: Each EffectRule maintains internal state that evolves incrementally. For example, a halftone rule maintains its dot grid positions and smoothly interpolates them between frames.

**Best for**: Stochastic/randomized effects (halftone, stippling, noise patterns). The state ensures patterns evolve smoothly rather than regenerating randomly per frame.

**Implementation**: Each stateful rule stores its previous output or parameters and interpolates toward the new target.

```python
class HalfToneRandomizationRule(EffectRule):
    """Halftone with smooth temporal evolution."""

    def __init__(self):
        self.dot_size = 4
        self.randomization_strength = 0.3
        self.temporal_smoothing = 0.8

    def apply(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, FrameContext]:
        state = context.get_rule_state(self.name())
        prev_offsets = state.get("prev_offsets")

        # Generate new random offsets for this frame
        h, w = frame.shape[:2]
        new_offsets = np.random.randn(h // self.dot_size, w // self.dot_size) * self.randomization_strength

        # Smooth with previous offsets
        if prev_offsets is not None and prev_offsets.shape == new_offsets.shape:
            smoothed = (self.temporal_smoothing * prev_offsets +
                       (1 - self.temporal_smoothing) * new_offsets)
        else:
            smoothed = new_offsets

        context.set_rule_state(self.name(), {"prev_offsets": smoothed})
        return self._apply_halftone(frame, smoothed), context

    @property
    def is_stateful(self) -> bool:
        return True
```

### 14.4 Strategy 4: Seeded Randomization with Interpolation

**How it works**: Instead of random seeds per frame, define keyframe seeds and interpolate between them. This ensures smooth transitions in stochastic effects.

```python
def interpolated_seed(frame_index: int, keyframe_interval: int = 30, base_seed: int = 42):
    """Generate smoothly interpolated random state between keyframes."""
    keyframe_a = (frame_index // keyframe_interval) * keyframe_interval
    keyframe_b = keyframe_a + keyframe_interval
    t = (frame_index - keyframe_a) / keyframe_interval  # 0.0 to 1.0

    rng_a = np.random.RandomState(base_seed + keyframe_a)
    rng_b = np.random.RandomState(base_seed + keyframe_b)

    noise_a = rng_a.randn(100)  # Example noise vector
    noise_b = rng_b.randn(100)

    # Smooth interpolation (cosine for smooth transitions)
    t_smooth = (1 - np.cos(t * np.pi)) / 2
    return noise_a * (1 - t_smooth) + noise_b * t_smooth
```

**Best for**: Effects that need controlled randomness (stippling, grain, dither patterns).

### 14.5 Strategy 5: Model-Native Temporal Consistency

Some AI models have built-in temporal consistency mechanisms. Using these avoids the need for post-hoc smoothing.

**Video Depth Anything**

Released January 2025. Extends Depth Anything V2 with temporal consistency for video.
- Generates consistent depth maps for long videos (5+ minutes).
- Uses a recurrent architecture that maintains temporal state between frames.
- Drop-in replacement for Depth Anything V2 when processing video.
- No need for additional temporal smoothing on depth output.

```python
# Video Depth Anything produces temporally consistent depth across frames
# The model internally maintains a state buffer across frames
from transformers import pipeline as hf_pipeline
depth_pipe = hf_pipeline("depth-estimation", model="depth-anything/Video-Depth-Anything")
# Process frames sequentially -- the model handles temporal consistency
for frame in video_frames:
    depth = depth_pipe(frame)  # Internally consistent with previous frames
```

**Recommendation for DepthMapRule**: When processing video, prefer Video Depth Anything over standard Depth Anything V2. The model handles temporal consistency natively, eliminating the need for a TemporalSmoothRule after depth estimation.

**SAM 2 Video Mode (Streaming Memory)**

SAM 2 has a dedicated video segmentation mode with streaming memory architecture:
- Maintains a per-session memory module that captures information about the target object.
- Tracks the selected person throughout all video frames, even through temporary occlusions.
- Produces temporally consistent person masks without frame-to-frame flickering.
- Requires a single prompt (point/box click) on the first frame; tracks automatically thereafter.

```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor("sam2.1_hiera_large.yaml", "sam2.1_hiera_large.pt")
state = predictor.init_state(video_path="dance.mp4")

# Prompt on first frame
predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=[[x, y]], labels=[1])

# Propagate through all frames -- produces temporally consistent masks
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    person_mask = masks[0]  # Temporally consistent mask
```

**Recommendation for PersonSegmentationRule**: When processing video, SAM 2 video mode is the most temporally consistent option for person masking. However, it requires an initial prompt. For fully automatic operation, use RMBG 2.0 per-frame with TemporalSmoothRule applied to the mask output.

**RAFT Optical Flow (Alternative to Farneback)**

RAFT (Recurrent All-Pairs Field Transforms) is a deep learning optical flow method that significantly outperforms classical Farneback:
- Higher accuracy, especially at motion boundaries.
- Better handling of large displacements.
- Available in torchvision: `torchvision.models.optical_flow.raft_large()`.
- GPU-accelerated (~100ms per frame pair on consumer GPU).

```python
import torchvision
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(device)

# Compute flow between consecutive frames
prev_tensor = torch.from_numpy(prev_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
curr_tensor = torch.from_numpy(curr_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
flow = model(prev_tensor.to(device), curr_tensor.to(device))[-1]  # Last iteration = best
```

**Comparison: Farneback vs RAFT**

| Property | Farneback (OpenCV) | RAFT (torchvision) |
|----------|-------------------|-------------------|
| Speed | ~30ms (CPU) | ~100ms (GPU) |
| Accuracy | Good (classical) | Excellent (SOTA) |
| Motion boundaries | Fair | Excellent |
| Large displacements | Poor | Good |
| GPU required | No | Yes |
| Dependencies | OpenCV only | torch + torchvision |

**Recommendation**: Use Farneback for CPU-only scenarios. Use RAFT when GPU is available and motion accuracy matters (fast-moving subjects, complex poses).

### 14.6 Recommended Combination

| Effect Type | Recommended Strategy | Notes |
|------------|---------------------|-------|
| Person mask (video) | Strategy 5: SAM 2 video mode | Native temporal tracking |
| Person mask (auto, no prompt) | Strategy 1 (EMA) on RMBG 2.0 output | |
| Depth map (video) | Strategy 5: Video Depth Anything | Native temporal consistency |
| Depth map (single image) | No smoothing needed | Deterministic per-frame |
| Edge detection | Strategy 1 (EMA) with high alpha (0.8-0.9) | |
| Halftone / stippling | Strategy 3 (state carry-forward) | Evolution rate controls smoothness |
| Audio-reactive displacement | Strategy 3 (decay state) | Instant rise, smooth decay |
| Fast motion scenes | Strategy 2 (RAFT optical flow) + Strategy 1 | |
| Random noise patterns | Strategy 4 (seeded interpolation) | |

---

## 15. Audio Analysis and Audio-Reactive Rules

### 15.1 Audio Analysis Library Comparison

Three Python libraries were evaluated for audio analysis:

| Property | librosa | madmom | aubio |
|----------|---------|--------|-------|
| **Focus** | General audio/MIR analysis | Beat/onset/tempo (state-of-art neural models) | Real-time onset/beat/pitch detection |
| **Core language** | Python + NumPy | Python + NumPy (neural nets via C) | C (Python bindings) |
| **Beat detection** | Dynamic programming beat tracker | RNN + DBN (superior accuracy) | Tempo-based heuristic |
| **Onset detection** | Spectral flux, multi-method | RNN-based (MIREX-winning) | Spectral methods (fast) |
| **Spectral analysis** | Full STFT, mel, chroma, MFCC | Limited (focused on rhythm) | Basic FFT, mel |
| **Format support** | MP3, WAV, FLAC, OGG, AAC (via soundfile/audioread) | WAV, FLAC (via ffmpeg optional) | WAV primarily, MP3 limited |
| **Installation** | `pip install librosa` (pure Python) | `pip install madmom` (requires cython build) | `pip install aubio` (C extension) |
| **Maintenance** | Actively maintained, well-documented | Less active (academic origin) | Stable but slow updates |
| **Speed** | Moderate (batch-oriented) | Slower (neural network inference) | Fast (C core, real-time capable) |
| **Customizability** | High (many parameters per feature) | Low (opinionated models) | Moderate |

**Decision**: Use **librosa as primary** for all spectral analysis, band energy, RMS, and spectral centroid. Optionally use **madmom for beat/onset detection only** when higher accuracy is needed (madmom's RNN-based beat tracker outperforms librosa's dynamic programming tracker on complex rhythms). aubio is not recommended -- its speed advantage is irrelevant for offline/batch processing, and its limited format support is a drawback.

See ADR-009 for the full decision rationale and ADR-011 for the AudioFrame data model.

### 15.2 AudioFrame and AudioSource

The `AudioFrame` dataclass (defined in Section 12.1 alongside `FrameContext`) carries per-frame audio analysis. The `AudioSource` class reads a **standalone audio file** (not extracted from the video) and produces frame-aligned `AudioFrame` objects. This is an AMV-style workflow: the video and audio are independent inputs.

```python
from collections import deque

import librosa
import numpy as np


class AudioSource:
    """Pre-analyzes a standalone audio file and provides per-frame AudioFrame lookup.

    AMV-style: audio is a separate file (MP3, WAV, FLAC, etc.), not
    extracted from the video. The audio file is loaded once, analyzed,
    and aligned to video frames by timestamp (frame_index / fps).

    The Pipeline injects context.audio = audio_source.get_frame(i) each frame.
    The same audio file is muxed into the output video during post-processing.
    """

    def __init__(self, audio_path: str, total_frames: int, fps: float,
                 use_madmom_beats: bool = False):
        """Analyze standalone audio file and build frame-aligned feature array.

        Args:
            audio_path: Path to standalone audio file (MP3, WAV, FLAC, OGG, AAC).
                        This is NOT extracted from the video -- it is a separate
                        music track provided by the user.
            total_frames: Number of video frames to align with.
            fps: Video frame rate for time alignment. Each AudioFrame corresponds
                 to a video frame: frame i covers audio from i/fps to (i+1)/fps seconds.
            use_madmom_beats: If True, use madmom RNN beat tracker instead
                              of librosa (better for complex rhythms).
        """
        self._audio_path = audio_path
        self.total_frames = total_frames
        self.fps = fps
        self._frames: list[AudioFrame] = []

        # Load standalone audio file
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        # hop_length aligns STFT frames to video frames
        hop_length = int(sr / fps)

        # --- Spectral analysis (librosa) ---
        stft = np.abs(librosa.stft(y, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr)

        # Frequency band masks
        bass_mask = (freqs >= 20) & (freqs <= 250)
        mid_mask = (freqs > 250) & (freqs <= 4000)
        treble_mask = (freqs > 4000) & (freqs <= 20000)

        # Per-frame band energies (normalized 0-1)
        bass = self._normalize(np.sum(stft[bass_mask, :] ** 2, axis=0))
        mid = self._normalize(np.sum(stft[mid_mask, :] ** 2, axis=0))
        treble = self._normalize(np.sum(stft[treble_mask, :] ** 2, axis=0))

        # Onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_env = self._normalize(onset_env)

        # RMS energy
        rms = self._normalize(librosa.feature.rms(y=y, hop_length=hop_length)[0])

        # Spectral centroid
        centroid = self._normalize(
            librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        )

        # --- Beat detection ---
        if use_madmom_beats:
            beat_set, tempo = self._madmom_beats(audio_path, hop_length, sr)
        else:
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=hop_length
            )
            beat_set = set(beat_frames.tolist())
            tempo = float(tempo)

        # --- Build per-frame AudioFrame objects ---
        for i in range(total_frames):
            if i < len(bass):
                self._frames.append(AudioFrame(
                    rms=float(rms[i]) if i < len(rms) else 0.0,
                    beat=(i in beat_set),
                    beat_strength=float(onset_env[i]) if i < len(onset_env) else 0.0,
                    spectrum=stft[:, i] if i < stft.shape[1] else np.zeros(stft.shape[0]),
                    bands={"bass": float(bass[i]), "mid": float(mid[i]),
                           "treble": float(treble[i])},
                    onset_strength=float(onset_env[i]) if i < len(onset_env) else 0.0,
                    spectral_centroid=float(centroid[i]) if i < len(centroid) else 0.0,
                    tempo_bpm=tempo,
                ))
            else:
                # Beyond audio duration: silent frame
                self._frames.append(AudioFrame(
                    rms=0.0, beat=False, beat_strength=0.0,
                    spectrum=np.zeros(stft.shape[0]),
                    bands={"bass": 0.0, "mid": 0.0, "treble": 0.0},
                    onset_strength=0.0, spectral_centroid=0.0, tempo_bpm=tempo,
                ))

    def get_frame(self, frame_index: int) -> AudioFrame | None:
        """Get AudioFrame for a video frame index. Returns None if out of range."""
        if 0 <= frame_index < len(self._frames):
            return self._frames[frame_index]
        return None

    @property
    def audio_path(self) -> str:
        """Path to the original audio file (for muxing into output video)."""
        return self._audio_path

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        mx = arr.max()
        return arr / mx if mx > 0 else arr

    @staticmethod
    def _madmom_beats(audio_path: str, hop_length: int,
                      sr: int) -> tuple[set[int], float]:
        """Use madmom RNN beat tracker for higher accuracy.

        Falls back to librosa if madmom is not installed.
        """
        try:
            import madmom
            proc = madmom.features.beats.RNNBeatProcessor()(audio_path)
            beats_sec = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)(proc)
            # Convert beat times (seconds) to video frame indices
            beat_set = {int(t * sr / hop_length) for t in beats_sec}
            # Estimate tempo from beat intervals
            if len(beats_sec) > 1:
                intervals = np.diff(beats_sec)
                tempo = 60.0 / np.median(intervals)
            else:
                tempo = 0.0
            return beat_set, float(tempo)
        except ImportError:
            import warnings
            warnings.warn("madmom not installed, falling back to librosa beat tracker")
            y, _ = librosa.load(audio_path, sr=sr, mono=True)
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=hop_length
            )
            return set(beat_frames.tolist()), float(tempo)
```

### 15.3 AudioReactiveEdgeRule Implementation

This rule reads `context.audio` (an `AudioFrame`) and uses it to modulate detected edges. It carries inter-frame state for smooth beat decay via `context.rule_state`.

**State schema** (stored in `context.rule_state["AudioReactiveEdgeRule"]`):
```
{
    "beat_decay": float,          # Current decay factor from last beat (0-1, decays each frame)
    "displacement_field": ndarray, # Spatial displacement field (H x W float32), decays each frame
    "band_history": deque,         # Last N frames of band energy for smoothing
}
```

```python
class AudioReactiveEdgeRule(EffectRule):
    """Displace person edges in sync with audio.

    Reads: context.audio (AudioFrame), context.person_mask, context.edge_map
    Publishes: nothing (modifies frame only)
    State: beat_decay, displacement_field, band_history

    Behavior:
    - On beat onset: spike edge displacement/thickness (instant rise)
    - Between beats: smooth exponential decay
    - Bass energy -> large structural edge displacement (silhouette, limbs)
    - Treble energy -> fine detail edge displacement (hair, fabric texture)
    - Mid energy -> medium features (clothing folds, facial contours)
    - RMS amplitude -> continuous breathing/undulation of all edges
    """

    def __init__(self):
        self.sensitivity = 0.7          # Overall reactivity (0-2)
        self.decay_rate = 0.85          # Beat decay per frame (0 = instant, 1 = never)
        self.displacement_scale = 15    # Max pixels of edge displacement
        self.breathing_scale = 3        # Max pixels of RMS-driven breathing
        self.band_weights = {"bass": 1.0, "mid": 0.4, "treble": 0.8}
        self.history_length = 8         # Frames of band history for smoothing

    def apply(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, FrameContext]:
        audio = context.audio
        if audio is None:
            return frame, context  # No audio: passthrough

        state = context.get_rule_state(self.name())

        # Initialize state on first frame
        if "beat_decay" not in state:
            state["beat_decay"] = 0.0
            state["displacement_field"] = np.zeros(frame.shape[:2], dtype=np.float32)
            state["band_history"] = deque(maxlen=self.history_length)

        beat_decay = state["beat_decay"]
        disp_field = state["displacement_field"]
        band_history = state["band_history"]

        # --- Beat response: instant rise, smooth decay ---
        if audio.beat:
            beat_decay = min(1.0, audio.beat_strength * self.sensitivity)
        else:
            beat_decay *= self.decay_rate

        # --- Frequency-spatial mapping ---
        # Compute weighted energy from bands
        weighted_energy = sum(
            audio.bands.get(band, 0.0) * self.band_weights.get(band, 0.0)
            for band in ("bass", "mid", "treble")
        )

        # Smooth band energy using history
        band_history.append(weighted_energy)
        smoothed_energy = sum(band_history) / len(band_history)

        # Target displacement = beat component + continuous energy component
        beat_displacement = beat_decay * self.displacement_scale
        energy_displacement = smoothed_energy * self.sensitivity * self.displacement_scale * 0.5
        breathing = audio.rms * self.breathing_scale

        total_displacement = beat_displacement + energy_displacement + breathing

        # --- Build spatial displacement field ---
        # Bass -> large radius kernel (affects silhouette edges)
        # Treble -> small radius kernel (affects fine detail edges)
        bass_radius = max(1, int(audio.bands.get("bass", 0) * self.displacement_scale))
        treble_radius = max(1, int(audio.bands.get("treble", 0) * self.displacement_scale * 0.3))

        # Decay existing displacement field toward new target
        target_field = np.full(frame.shape[:2], total_displacement, dtype=np.float32)
        disp_field = self.decay_rate * disp_field + (1 - self.decay_rate) * target_field

        # --- Apply displacement to edges ---
        result = frame
        if total_displacement > 0.5:
            result = self._apply_edge_displacement(
                frame, context.person_mask, context.edge_map,
                disp_field, bass_radius, treble_radius,
            )

        # Save state
        state["beat_decay"] = beat_decay
        state["displacement_field"] = disp_field
        state["band_history"] = band_history
        context.set_rule_state(self.name(), state)

        return result, context

    def _apply_edge_displacement(
        self, frame: np.ndarray, mask: Optional[np.ndarray],
        edge_map: Optional[np.ndarray], disp_field: np.ndarray,
        bass_radius: int, treble_radius: int,
    ) -> np.ndarray:
        """Dilate/displace edges based on spatial displacement field.

        Uses edge_map from context if available (from EdgeDetectionRule),
        otherwise detects edges locally with Canny.
        """
        # Use pre-computed edges if available, else detect locally
        if edge_map is not None:
            edges = edge_map.copy()
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

        # Mask to person region if available
        if mask is not None:
            edges = edges & mask

        # Multi-scale displacement: large kernel for bass, small for treble
        if bass_radius > 1:
            kernel_bass = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (bass_radius * 2 + 1, bass_radius * 2 + 1)
            )
            edges_bass = cv2.dilate(edges, kernel_bass, iterations=1)
        else:
            edges_bass = edges

        if treble_radius > 1 and treble_radius != bass_radius:
            kernel_treble = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (treble_radius * 2 + 1, treble_radius * 2 + 1)
            )
            edges_treble = cv2.dilate(edges, kernel_treble, iterations=1)
        else:
            edges_treble = edges

        # Combine: bass displacement + treble detail
        displaced = np.maximum(edges_bass, edges_treble)

        # Composite displaced edges onto frame
        frame_out = frame.copy()
        frame_out[displaced > 0] = [255, 255, 255]
        return frame_out

    def reset_state(self, context: FrameContext) -> FrameContext:
        context.set_rule_state(self.name(), {})
        return context

    @property
    def is_stateful(self) -> bool:
        return True

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "sensitivity": {"type": "float", "default": 0.7, "min": 0.0, "max": 2.0,
                            "description": "Overall audio reactivity"},
            "decay_rate": {"type": "float", "default": 0.85, "min": 0.0, "max": 0.99,
                           "description": "Beat decay per frame (higher = slower fade)"},
            "displacement_scale": {"type": "int", "default": 15, "min": 1, "max": 100,
                                   "description": "Max pixels of beat-driven displacement"},
            "breathing_scale": {"type": "int", "default": 3, "min": 0, "max": 20,
                                "description": "Max pixels of RMS-driven breathing"},
            "band_weights": {"type": "dict",
                             "default": {"bass": 1.0, "mid": 0.4, "treble": 0.8},
                             "description": "Weight per frequency band"},
            "history_length": {"type": "int", "default": 8, "min": 1, "max": 30,
                               "description": "Frames of band energy smoothing history"},
        }
```

### 15.4 Audio-Reactive Design Principles

1. **Pre-compute all audio analysis before frame processing**. `AudioSource` runs all analysis at construction time. Audio features are static -- they depend only on the audio file, not on video frames. This avoids redundant work and enables deterministic behavior.

2. **Pipeline owns audio injection**. `Pipeline.process_frame()` sets `context.audio` from `AudioSource` each frame. Rules never call `AudioSource` directly. This keeps rules decoupled from data loading.

3. **Instant rise, smooth decay**. Beat onsets cause immediate displacement peaks. Between beats, displacement decays exponentially (`decay_rate`). This matches how humans perceive musical energy -- sharp attacks, gradual release.

4. **Frequency-to-spatial mapping**. Bass energy drives large structural edge displacement (silhouette, limbs) using large morphological kernels. Treble energy drives fine detail edge displacement (hair, fabric) using small kernels. Mid energy contributes to medium features. This creates a natural-feeling audio-visual correspondence where low frequencies feel "heavy" and high frequencies feel "sparkly".

5. **Continuous breathing via RMS**. Beyond beat-driven spikes, the overall loudness (RMS) drives a continuous undulation of edges. This makes the visual feel alive even between beats, similar to how real audio visualizers pulse constantly.

6. **Band history smoothing**. A sliding window (`band_history`) smooths frequency band energy across frames to prevent jitter. Without this, frame-to-frame FFT variations would cause edges to twitch erratically.

7. **Reads upstream context**. AudioReactiveEdgeRule reads `context.edge_map` (from EdgeDetectionRule) and `context.person_mask` (from PersonSegmentationRule) when available. This avoids redundant edge detection and ensures displacement is applied to the same edges the user sees.

8. **Graceful degradation**. No audio file = `context.audio` is `None` = passthrough. Silent audio = zero displacement. Missing `person_mask` = edges detected on full frame. Missing `edge_map` = local Canny detection. Corrupted audio = `AudioSource.__init__` raises `librosa.util.exceptions.ParameterError` before processing starts.

---

## Technical Design Checklist

- [x] Component boundaries are clearly defined (image: 5 stages, video: 3 phases, EffectRule interface)
- [x] Data flow is documented (image pipeline: Section 7.1, video pipeline: Section 13.1)
- [x] API contracts specify request/response schemas (EffectRule interface: Section 12.1, FrameContext)
- [x] Error handling strategy is defined (fallback models, feature file scenarios, rule validation)
- [x] State management approach documented (Section 12: EffectRule state, checkpointing, scene cuts)
- [x] Dependencies (external services, libraries) are listed (Section 10)
- [x] Performance considerations are addressed (image: 30s target, video: 2hr/300frames budget in 13.6)
- [x] Task decomposition recommendations are included (image stages, video phases, rule modularity)
- [x] Output format specifications defined (Section 8: PNG, 16-bit PNG, EXR, SVG, OBJ, PSD, MP4)
- [x] Determinism strategy documented (Section 9: seeds, deterministic algorithms)
- [x] Detail level parameter specified (Section 6: mapping 1-10 to algorithm params)
- [x] Resolution handling for 4096x4096 documented (Section 9.3)
- [x] Platform support defined (Section 10.5: MPS primary, CUDA secondary)
- [x] Temporal coherence strategies documented (Section 14: EMA, optical flow, state carry-forward)
- [x] Audio analysis pipeline documented (Section 15: librosa/madmom comparison, AudioSource, AudioFrame, per-frame features)
- [x] Plugin/extensibility architecture documented (Section 12.3: RuleRegistry, auto-discovery)
- [x] Layer-based compositing model documented (Section 12.5: CompositeRule, input_layer/output_layer, blend modes, ADR-013)
- [x] Checkpointing and resume documented (Section 13.5)
- [x] Scene cut detection documented (Section 13.4)
- [ ] Security implications -- N/A (local processing, no network calls required)
- [ ] Migration strategy -- N/A (greenfield project)

## Product Requirements Traceability

| Product Requirement | Design Section | Status |
|--------------------|---------------|--------|
| Silhouette mask (binary alpha PNG) | 7.2, 8.1 | Covered |
| Edge map (configurable detail 1-10) | 6, 7.3 | Covered |
| Depth map (16-bit PNG / EXR) | 7.4, 8.3 | Covered |
| Topology mesh (SVG / OBJ) | 7.6, 8.4 | Covered |
| Layered composite (PSD) | 7.7, 8.5 | Covered |
| Processed video (MP4 H.264/H.265) | 13.2 | Covered |
| Frame sequence output (numbered PNGs) | 13.2, 13.3 | Covered |
| Audio muxing into output video | 13.2 | Covered |
| Offline/batch processing | 7.1, 13.1 | Covered |
| Local only (no cloud APIs) | 10.2 | Covered |
| macOS MPS primary | 10.5 | Covered |
| Resolution up to 4096x4096 / 4K video | 9.3, 13.6 | Covered |
| Latency < 30s per image | 4.6 | Covered |
| Video latency < 2hr / 300 frames | 13.6 | Covered |
| Deterministic output | 9 | Covered |
| Temporal coherence (no flickering) | 14 | Covered |
| EffectRule composable chains | 12 | Covered |
| Rule chain YAML/JSON config | 12.6 | Covered |
| Layer-based compositing (named layers, CompositeRule, blend modes) | 12.5, 12.6, ADR-013 | Covered |
| Stateful rules (inter-frame state) | 12.1, 12.4, 14.3 | Covered |
| State serialization / checkpointing | 12.2, 13.5 | Covered |
| Scene cut detection + state reset | 13.4 | Covered |
| Rule plugin discovery | 12.3 | Covered |
| Audio analysis (beat, spectral, onset) | 15.1, 15.2 | Covered |
| AudioFrame typed data model | 12.1, 15.2, ADR-011 | Covered |
| AudioSource pre-analysis + Pipeline injection | 12.2, 15.2 | Covered |
| Audio-reactive edge displacement | 15.3 | Covered |
| Frequency-spatial band mapping (bass->structural, treble->fine) | 15.3, 15.4 | Covered |
| Beat decay + RMS breathing | 15.3, 15.4 | Covered |
| Classical edge algorithms evaluated | 1 | Covered |
| AI edge algorithms evaluated | 2 | Covered |
| Depth algorithms evaluated | 4 | Covered |
| Topology algorithms evaluated | 5 | Covered |
