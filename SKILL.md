# SKILL: Interactive Architecture Diagram (HTML)

A guide for generating self-contained, readable, interactive architecture diagrams as a single HTML file. Distilled from building `docs/architecture.html`.

---

## Output Requirements

- **Single HTML file** — no external CDN dependencies, all JS/CSS inline
- **Self-contained** — opens directly in any browser, no server needed
- **Dark theme** — VSCode/Linear aesthetic: near-black background, colored syntax highlights
- **Four tabs** minimum: Type System, Data Flow, Call Stack, Layer System (adapt to project)
- **Last-updated timestamp** and a "kept up to date" note referencing AGENTS.md
- **Monospace font** throughout: `'Cascadia Code','Fira Code','JetBrains Mono','SF Mono',Consolas,monospace`

---

## Readability Rules (hard-won)

### Font sizes — never go below these minimums
| Context | Min size |
|---------|----------|
| Body / code lines | 13px |
| Node names, labels | 14px |
| Badges, section headers | 10px |
| Smallest annotation | 11px |

### Color contrast
```css
--text:   #e8ecf8;   /* primary text — bright */
--muted:  #8892aa;   /* secondary — must be visible, NOT #4a5070 */
--border: #252540;
--bg:     #0a0a14;
--card:   #141428;
```

Comments and annotations must be `#6a7a9a` minimum — `#353550` is effectively invisible on dark backgrounds.

### SVG vs Canvas for diagrams
**Always use SVG, never Canvas for diagrams with text.**
- Canvas text is blurry at small sizes and does not scale
- SVG text is always crisp, scales perfectly, and supports `<title>` tooltips
- Reserve canvas only for pixel-level effects (particles, generative art)

---

## Type System Tab — Column Layout to Minimize Tangles

The single most important rule: **arrange nodes in dependency order, left to right**. Every arrow should flow forward (left→right). Backwards arrows create spaghetti.

### Algorithm

1. Topologically sort your types by dependency
2. Assign each type to the leftmost column where all its dependencies are already placed
3. Within a column, stack nodes vertically with enough gap (120–140px between node tops)

### Derive columns from the dependency graph — don't assume fixed labels

Column labels and counts must come from the actual type graph, not a fixed template. The algorithm:

```
1. Build a directed graph: edge A → B means "A is used by B" (B depends on A)
2. Compute in-degree for every node (how many things it depends on)
3. Assign column = longest path from any root (Kahn's algorithm / BFS from roots):
     col[n] = max(col[dep] + 1 for dep in dependencies[n])
4. Within each column, sort nodes vertically to minimize crossing:
     - Nodes that connect to the same target should be close together
     - Nodes connected to the bottom of the previous column go near the bottom
5. Label columns from the actual node kinds present (inspect what's in each column)
```

Example result for the AMV project:
```
Col 0  x=24    BlendMode, NDArrays, StateT, Layer      ← no dependencies
Col 1  x=310   LayerKey, AudioFrame                    ← depend only on col 0
Col 2  x=610   EffectRule, FrameContext                ← depend on col 0+1
Col 3  x=930   CompositeRule, AudioSource              ← depend on col 2
Col 4  x=1250  Pipeline                                ← depends on col 2+3
```

The column spacing (260–320px between columns) should scale with the widest node in each column, not be fixed.

### Arrow routing
- **Standard (L→R):** exit right edge of source, enter left edge of target
  ```js
  const gap = Math.max(tx - fx, 30);
  const ctrl = gap * 0.5;
  d = `M${fx},${fy} C${fx+ctrl},${fy} ${tx-ctrl},${ty} ${tx},${ty}`;
  ```
- **Backwards (same column or R→L):** left-side loop — exit left edge, arc 50–60px left, enter left edge of target
  ```js
  const lx = Math.min(f.x, t.x) - 55;
  d = `M${f.x},${fy} C${lx},${fy} ${lx},${ty} ${t.x},${ty}`;
  ```
- **Multiple arrows into the same node:** use `ty_off` (±8–12px) to spread entry points so arrowheads don't overlap

### Arrow visibility
```
stroke-width: 2
opacity: 0.7         (hover → 1.0 + stroke-width 3)
arrowhead: 10×8px markerWidth/Height
```

### Correct coordinate system for arrows
Use `node.offsetLeft / node.offsetTop` — NOT `getBoundingClientRect()`.
`offsetLeft/Top` returns coordinates in the inner div's untransformed space, which matches the SVG coordinate system directly. `getBoundingClientRect()` returns screen-space coordinates that are corrupted by the CSS pan/zoom transform.

---

## Pan, Zoom, and Node Drag

This is non-negotiable for any diagram with more than ~6 nodes. Users must be able to navigate.

### HTML structure
```html
<div id="view-types" style="overflow:hidden; cursor:grab">
  <div id="type-inner" style="position:absolute; top:0; left:0; transform-origin:0 0">
    <svg id="type-svg" style="position:absolute; top:0; left:0; pointer-events:none"></svg>
    <!-- nodes: position:absolute inside type-inner -->
  </div>
  <div id="hud" style="position:absolute; bottom:14px; right:14px">
    <!-- legend, hints — NOT inside type-inner so it doesn't zoom -->
  </div>
</div>
```

### Zoom toward cursor (scroll)
```js
let scale = 0.78, panX = 20, panY = 20;

view.addEventListener('wheel', e => {
  e.preventDefault();
  const r = view.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const factor = e.deltaY < 0 ? 1.12 : 1/1.12;
  const ns = Math.max(0.2, Math.min(3.5, scale * factor));
  // keep the point under the cursor fixed:
  panX = mx - (ns/scale) * (mx - panX);
  panY = my - (ns/scale) * (my - panY);
  scale = ns;
  inner.style.transform = `translate(${panX}px,${panY}px) scale(${scale})`;
}, { passive: false });
```

### Pan (drag background)
```js
let panning = false, panStartX = 0, panStartY = 0;

view.addEventListener('mousedown', e => {
  if (e.target.closest('.type-node') || e.target.closest('#hud')) return;
  panning = true;
  panStartX = e.clientX - panX;
  panStartY = e.clientY - panY;
  view.classList.add('panning'); // cursor:grabbing
});
document.addEventListener('mousemove', e => {
  if (!panning) return;
  panX = e.clientX - panStartX;
  panY = e.clientY - panStartY;
  inner.style.transform = `translate(${panX}px,${panY}px) scale(${scale})`;
});
document.addEventListener('mouseup', () => { panning = false; view.classList.remove('panning'); });
```

### Node drag (+ click-to-expand disambiguation)
```js
let dragNode = null, dragOffX = 0, dragOffY = 0, dragMoved = false, dragStartCX = 0, dragStartCY = 0;

view.addEventListener('mousedown', e => {
  const node = e.target.closest('.type-node');
  if (!node) return; // handled by pan above
  dragNode = node;
  dragStartCX = e.clientX; dragStartCY = e.clientY;
  dragMoved = false;
  const ir = inner.getBoundingClientRect();
  dragOffX = (e.clientX - ir.left) / scale - node.offsetLeft;
  dragOffY = (e.clientY - ir.top)  / scale - node.offsetTop;
  e.preventDefault(); // prevent text selection
});

document.addEventListener('mousemove', e => {
  if (!dragNode) return;
  if (!dragMoved && (Math.abs(e.clientX-dragStartCX)>4 || Math.abs(e.clientY-dragStartCY)>4)) {
    dragMoved = true;
    dragNode.style.transition = 'none';
  }
  if (dragMoved) {
    const ir = inner.getBoundingClientRect();
    dragNode.style.left = Math.max(0, (e.clientX - ir.left)/scale - dragOffX) + 'px';
    dragNode.style.top  = Math.max(0, (e.clientY - ir.top) /scale - dragOffY) + 'px';
    drawArrows(); // redraw live
  }
});

document.addEventListener('mouseup', () => {
  if (dragNode) {
    dragNode.style.transition = '';
    if (!dragMoved) {
      // short tap = click → toggle expand
      dragNode.classList.toggle('expanded');
      setTimeout(drawArrows, 50);
    }
    dragNode = null;
  }
});
```

**Key insight:** the 4px move threshold disambiguates drag from click. Do NOT attach a separate `click` listener to nodes — handle expand/collapse in `mouseup` when `!dragMoved`.

---

## Type Node Color Coding

```
Enum            --enum:    #c084fc  (purple)
Frozen dataclass--frozen:  #34d5c0  (teal)
Mutable dataclass--mutable:#60a5fa  (blue)
Abstract class  --abc:     #fb923c  (orange)
Concrete class  --concrete:#4ade80  (green)
TypeAlias       --alias:   #94a3b8  (gray)
TypeVar         --tvar:    #fbbf24  (yellow)
```

Border: `rgba(color, 0.35)` collapsed → full color expanded/hover.
Badge bg: `rgba(color, 0.12)`.

---

## Data Flow Tab

Use a static `<svg viewBox="0 0 W H">` for the data flow diagram. Key principles:
- Top-to-bottom or left-to-right flow
- Large central "pipeline" box with internal steps listed
- Input boxes on left, output on right
- Each step inside the pipeline gets its own color-coded `<rect>` background
- Arrow markers defined in `<defs>`, one per color used
- Label the arrows near their midpoints with small `<rect>` + `<text>` combos for readability

---

## Call Stack Tab

Expandable tree built from a JS data structure. Key rules:
- Distinguish `loop` nodes (dashed border box) from regular call nodes
- Use `+` / `−` toggle indicators (16px square, turns accent color when open)
- Code colors: function names `#79b8ff`, params `#e9c46a`, types `#c084fc`, comments `#6a7a9a`
- Comments **must** be at least `#6a7a9a` — never `#353550` (invisible)
- Indent children with `padding-left: 24px; border-left: 1px solid var(--border)`

---

## Layer System Tab

Step-through showing state accumulation. Key rules:
- Left column: pipeline steps (click to activate, or Prev/Next buttons)
- Right column: live FrameContext state panel updating as you step
- New entries animate in with `translateX(-6px) → 0` + flash highlight
- `opacity: 0 → 1` on each layer entry, staggered with `setTimeout(i * 60ms)`
- Badge colors: BRANCH (yellow), INLINE (blue), COMPOSITE (green), INIT (gray)

---

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Column labels float above empty space | **Don't use column text labels.** Nodes start at different y positions across columns, so a label at `top: 4px` will hover above a gap where its column has no nodes. The node badges (ABC, TypeAlias, Enum, etc.) already describe each node's kind — let the left→right arrow flow communicate the column grouping instead. |
| Arrows all go right→left or cross freely | Use column layout sorted by dependency order |
| Text blurry at small scale | Use SVG not canvas; don't go below 11px |
| "Muted" text invisible | `--muted` must be at least `#6a7a9a`, never `#353550` or `#4a5070` |
| Zoom/pan broken when nodes are dragged | Use `offsetLeft/offsetTop` not `getBoundingClientRect` for arrow coordinates |
| Click and drag conflict on nodes | Use 4px move threshold; handle expand in `mouseup` with `!dragMoved` |
| Multiple arrows pile up at one node | Use `ty_off` (±8–12px) to spread entry points |
| HUD (legend/hint) zooms with diagram | Put HUD outside `#type-inner`, as a sibling — it won't receive the transform |
| Canvas animation makes text unreadable | Replace with static SVG; animation isn't worth the text quality loss |
