# AGENTS.md — AI Agent Instructions for `my_amv`

This file contains instructions for AI agents (Claude Code, Cursor, Copilot, etc.) working on this codebase.

---

## Architecture Diagram

**`docs/architecture.html` must be kept up to date at all times.**

This self-contained HTML file is the canonical visual reference for the codebase architecture. Any agent that modifies the architecture must update the diagram in the same commit/PR.

### When to update `docs/architecture.html`

Update the diagram whenever you change:

| Change | Diagram section(s) to update |
|--------|-------------------------------|
| Add/remove/rename a type, field, or method | **Type System** tab |
| Change `Layer` enum values | **Type System** tab + **Layer System** tab |
| Change `FrameContext` fields or accessors | **Type System** tab + **Data Flow** tab |
| Change `EffectRule` interface or `apply()` signature | **Type System** tab + **Call Stack** tab |
| Change `Pipeline.process_video` / `process_frame` logic | **Data Flow** tab + **Call Stack** tab |
| Change inline vs branch rule routing | **Layer System** tab |
| Add/remove a major processing stage | **Data Flow** tab + **Call Stack** tab |
| Change `AudioSource` or `AudioFrame` | **Type System** tab + **Data Flow** tab |
| Add/remove a layer in the compositing model | **Layer System** tab |

### How to update the diagram

The diagram is a single self-contained HTML file — all JS and CSS are inline.

1. Open `docs/architecture.html` and locate the relevant `<script>` section or data object.
2. The data driving each tab is defined near the top of each tab's JS block as a plain JS object/array — look for comments like `/* TYPE_SYSTEM_DATA */`, `/* CALL_STACK_DATA */`, etc.
3. Update the data to reflect your changes.
4. Verify the page renders correctly by opening it in a browser.
5. Update the `lastUpdated` variable at the top of the `<script>` block to today's date (ISO format: `YYYY-MM-DD`).

---

## Project Overview

`my_amv` is an AMV (Anime Music Video) art pipeline. It takes:
- `video.mp4` — source video
- `music.mp3` — separate audio track

And produces `output.mp4` with the audio muxed in, after applying a chain of `EffectRule`s frame-by-frame.

See `docs/plans/architect/edge-topology-tech-design.md` for the full technical design.
See `docs/plans/product/edge-topology-brief.md` for the product brief.
See `docs/features/` for Gherkin feature files (acceptance criteria).

---

## Key Architectural Invariants

These must not be violated without updating the architecture docs:

1. **`Layer` is an `Enum`, not `StrEnum`.**
   `Layer.MASK` is a `Layer` instance, not a `str`. This preserves the `dict[LayerKey, RGBArray]` union type distinction in mypy. Do not change this to `StrEnum`.

2. **`LayerKey = Layer | str`.**
   Built-in layers use `Layer` enum members. User-defined / rule-specific layers use plain `str`. Do not collapse this to just `str`.

3. **`FrameContext` has no named output fields.**
   All layer data lives in `_layers: dict[LayerKey, RGBArray]`. Access via `get_layer()` / `set_layer()`. Do not add named fields like `person_mask` or `depth_map`.

4. **`EffectRule.apply()` signature is fixed:**
   ```python
   def apply(self, frame: RGBArray, context: FrameContext) -> tuple[RGBArray, FrameContext]
   ```
   The return is always `(output_frame, context)`. Do not change this signature.

5. **Inline vs Branch routing:**
   - If `rule.output_layer is None` → **inline**: result replaces the current frame.
   - If `rule.output_layer is not None` → **branch**: result goes to `context.set_layer(rule.output_layer, ...)`, current frame is unchanged.

6. **`AudioSource` pre-analyzes the full audio before frame processing begins.**
   Do not stream/lazy-load audio frames — all `AudioFrame` objects are computed upfront.

7. **`FrameContext._rule_state` is keyed by rule name (`rule.name()`).**
   Rules must use a unique, stable `name()` to avoid state key collisions.

---

## Codebase Structure (planned)

```
my_amv/
├── AGENTS.md                        ← this file
├── docs/
│   ├── architecture.html            ← interactive architecture diagram (keep up to date!)
│   ├── plans/
│   │   ├── product/edge-topology-brief.md
│   │   └── architect/edge-topology-tech-design.md
│   └── features/
│       ├── edge_detection.feature
│       ├── depth_mapping.feature
│       ├── topology_generation.feature
│       ├── video_pipeline.feature
│       ├── effect_rules.feature
│       └── audio_reactive.feature
├── src/
│   └── my_amv/
│       ├── types.py                 ← Layer, LayerKey, BlendMode, RGBArray, etc.
│       ├── context.py               ← FrameContext, AudioFrame
│       ├── rule.py                  ← EffectRule[StateT] ABC, CompositeRule
│       ├── audio.py                 ← AudioSource
│       ├── pipeline.py              ← Pipeline
│       └── rules/
│           ├── segmentation.py      ← PersonSegmentationRule
│           ├── edge_detection.py    ← EdgeDetectionRule
│           ├── depth.py             ← DepthMappingRule
│           ├── halftone.py          ← HalfToneRandomizationRule
│           ├── audio_reactive.py    ← AudioReactiveEdgeRule
│           └── ...
└── tests/
```

---

## Type Reference (quick cheatsheet)

```python
Layer(Enum)        # ORIGINAL, MASK, EDGES, DEPTH, SKELETON, FINAL
LayerKey           # = Layer | str
BlendMode          # = Literal["screen","multiply","add","over","normal"]
RGBArray           # NDArray[np.uint8]  shape (H, W, 3)
MaskArray          # NDArray[np.uint8]  shape (H, W)  values 0|255
EdgeArray          # NDArray[np.uint8]  shape (H, W)
DepthArray         # NDArray[np.float32] shape (H, W)  0.0–1.0
StateT             # TypeVar for EffectRule state
AudioFrame         # frozen dataclass — per-frame audio features
FrameContext       # mutable dataclass — all frame data + layer store
EffectRule[StateT] # ABC — composable processing unit
CompositeRule      # EffectRule[None] — multi-layer blend
AudioSource        # pre-analyzes audio → list[AudioFrame]
Pipeline           # orchestrates rules + video I/O
```
