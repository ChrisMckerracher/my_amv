# Dependency Audit
*Audited 2026-02-21 by auditor@epic-my-amv*

## Summary

7 deps can be removed, 2 are missing, 1 is redundant.

---

## Remove (unused / bloat)

| Package | Reason |
|---------|--------|
| `diffusers` | Not imported anywhere. Pulled in speculatively. ~1GB. |
| `accelerate` | Only needed by diffusers. Remove with it. |
| `torchvision` | Not imported anywhere in src/ or tests/. |
| `ffmpeg-python` | Not imported — FFmpeg is called via subprocess directly. |
| `scikit-image` | Not imported anywhere. |
| `scipy` | Not imported anywhere. |
| `opencv-contrib-python` | Redundant with `opencv-python` — both map to `cv2`. Only one needed. |
| `open3d` (optional) | Not imported. Remove from `[3d]` extras or keep as future placeholder. |

---

## Add (used but undeclared)

| Package | Used in | Notes |
|---------|---------|-------|
| `mediapipe` | `rules/segmentation.py` | Conditional import, but should be declared |
| `psutil` | `video_io.py` | Used for disk space checking |

> `sam2` is also imported conditionally in segmentation.py but is not a standard pip package — needs investigation.

---

## Keep (actively used)

| Package | Import | Used in |
|---------|--------|---------|
| `torch` | `torch` | Multiple rules — core ML runtime |
| `opencv-python` | `cv2` | Core image processing throughout |
| `numpy` | `numpy` | Everywhere |
| `Pillow` | `PIL` | output.py and tests |
| `librosa` | `librosa` | audio.py |
| `soundfile` | `soundfile` | Tests |
| `controlnet-aux` | `controlnet_aux` | edge_detection.py (lazy-loaded) |
| `transformers` | pipeline | depth.py (try/except) |
| `pydantic` | `pydantic` | config.py |
| `pyyaml` | `yaml` | config.py + tests |
| `typer` | `typer` | `__main__.py` CLI |
| `svgwrite` | `svgwrite` | output.py (try/except) |

---

## Action items

- [ ] Remove 7 unused deps from `pyproject.toml`
- [ ] Add `mediapipe` and `psutil` to `pyproject.toml`
- [ ] Investigate `sam2` — find correct package name or make it a proper optional dep
- [ ] Verify `opencv-contrib-python` can be dropped (check if any contrib module like `xphoto`, `ximgproc` etc. is used)
- [ ] Run `uv sync` after cleanup and re-run `uv run pytest` to confirm nothing breaks

---

## M1 MacBook impact

After cleanup, the install drops `diffusers` + `accelerate` + `torchvision` + `scipy` + `scikit-image` — saving ~1.5-2GB.
Core heavy deps that remain: `torch` (CPU/MPS), `controlnet-aux`, `transformers`. These are genuinely needed.
