# Zoom / Pan / Display Pipeline — Complete History & Architecture Memo

**Purpose:** This document chronicles every design decision, bug fix, and edge case for the zoom/pan/resolution/resize/display system. Use it as a regression checklist after any rewrite of the display pipeline.

---

## Part 1: Chronological History

### v3 (Pre-February 2026) — First Zoom Controls
- Basic zoom controls added (details sparse in changelog)
- No pan system yet

### v7 (Pre-February 2026) — Marker Positioning Fix
- **Bug:** Picker balloons/markers used raw canvas pixel coordinates instead of CSS display coordinates, so they appeared in wrong positions at any non-1:1 display size
- **Fix:** Convert canvas coordinates accounting for zoom level before placing markers

### v19 (February 4, 2026) — Resizable Preview Panel
- **New:** Drag edges/corner to resize the image preview area
- **New:** Responsive layout — vertical stacking on narrow viewports, horizontal scroll for many targets

### v20 (February 4, 2026) — Resize Bug Fixes
- **Bug fix:** Picker balloons follow image on resize (they weren't updating positions)
- **Bug fix:** Resize handles accessible during loading (z-index was blocking them)

### v24 (February 8-9, 2026) — THE BIG REWRITE
This was the major restructuring session. Almost everything about how the image displays was rebuilt.

#### Panzoom Library Integration
- Added `@panzoom/panzoom@4.6.1` via CDN
- **Critical design choice:** Panzoom handles ONLY panning (translate). Zoom is NOT done via Panzoom's scale — instead, zoom is done by re-rendering the canvas at the zoomed pixel size. This avoids the CSS `transform: scale()` blur problem.
- Panzoom configured with `maxScale: 1, minScale: 1, disableZoom: true`
- Pan is disabled by default, enabled only during Alt+drag sessions
- Pan is allowed at zoom > 1x OR when vertically cropped

#### Pixel-Perfect Zoom Rendering
- **Problem solved:** CSS `transform: scale()` stretches a rasterized bitmap, producing blur
- **Solution:** Two-canvas system:
  - Hidden `canvas` (id="imageCanvas") holds the original-resolution pixel data
  - Visible `displayCanvas` (created in JS) is rendered at actual zoomed pixel size
  - `renderAtCurrentZoom()` draws the hidden canvas onto displayCanvas at the target resolution
  - Result: crisp pixels at any zoom level, up to 2x retina density

#### The CSS Scale Trick (Instant Feedback)
- When the user scrolls to zoom, waiting 150ms for a pixel-perfect re-render would feel laggy
- **Solution:** Immediately apply `displayCanvas.style.transform = scale(zoomLevel / lastRenderedZoom)` for instant visual feedback (may be slightly blurry), then debounce the high-quality `renderAtCurrentZoom()` after 150ms
- Once the hi-res render completes, the CSS scale is reset to `scale(1)` — the canvas is now pixel-perfect

#### Zoom-to-Cursor Math
- When scrolling to zoom, the point under the mouse cursor stays fixed on screen
- Formula: `newPan = mousePos - (mousePos - oldPan) * (newZoom / oldZoom)`
- Same math used for both scroll wheel and zoom slider (slider zooms to viewport center instead of cursor)

#### Scroll Sensitivity & Snap-to-1x
- Reduced scroll zoom sensitivity from +/-15% to +/-7% per tick
- **Snap-to-1x threshold:** If `newZoom < 1.02`, snap it to exactly `1.0` — prevents "partial-zoom limbo" where you're at 101% and things feel slightly off
- Zoom range clamped: min=1.0, max=4.0

#### Deferred Layout Teardown at 1x
- When zooming back to 1x, the wrapper height needs to be unlocked and classes need to be removed
- **Problem:** If you do this immediately during continuous scroll, layout thrashes
- **Solution:** Defer the 1x cleanup by 200ms. If user scrolls back up within 200ms, cancel the cleanup. Only tear down if zoom stays at 1x.
- **Guard:** Teardown only happens if `!isVerticallyCropped` — if the user manually shrunk the wrapper, keep it shrunk

#### Sticky Overlays
- `updateStickyOverlays()` positions the zoom controls and picker overlay at the vertical center of the visible portion of the canvas wrapper
- Accounts for partial scroll: if the wrapper is half-scrolled off the top of the viewport, the overlay centers in the still-visible half
- **Edge clamping:** Overlays can't go past wrapper edges — clamped to `halfHeight` from each edge
- Triggers: scroll, resize, layout changes, zoom changes

#### Other v24 Changes
- Button overflow fix (`flex-wrap: wrap` to button groups)
- Ctrl key toggle for overlay peek (hide overlays temporarily)
- Upload zone centering (flex instead of margin-top)

### v25 (February 9, 2026) — Theme & Canvas Border
- Canvas wrapper border changed to `2px solid rgba(120, 150, 175, 0.35)` for better boundary visibility
- This is cosmetic but important for the user to see where the canvas ends

### v26 (February 9, 2026) — Vertical Cropping & Pan Boundaries

#### Vertical Image Cropping
- **New feature:** Dragging the bottom resize handle UP now clips/hides overflow instead of rescaling the image
- User can then pan to see hidden parts (pan enabled even at zoom=1 when cropped)
- Added `isVerticallyCropped` state and `checkVerticallyCropped()` function
- **Detection:** Compares wrapper height to natural image display height at current width. If wrapper is shorter by more than 2px, it's cropped.
- **2px tolerance** prevents flicker at the exact boundary

#### Pan Boundary Constraints
- **Bug fixed:** Before this, image edges could be dragged past the viewport edges, showing empty space
- `constrainPan()` function added:
  - If image is larger than wrapper: can pan from `0` to `-(canvasSize - wrapperSize)`
  - If image is smaller than wrapper: centers it, no panning allowed in that axis
  - **Recursive guard:** `_constrainingPan` flag prevents infinite loop from panzoomchange event
  - **CSS scale awareness:** Accounts for temporary CSS scale when hi-res render hasn't finished yet

#### Reset View Enhancement
- Reset Zoom button now also resets wrapper height to natural image aspect ratio
- Added "Reset View" text label
- **Multiple fixes** to properly restore wrapper height (was fragile)
- Resets: zoom to 1, pan to (0,0), isVerticallyCropped to false, clears manual width/height/flex overrides, restores wrapper aspect-ratio CSS, removes 'zoomed'/'can-pan' classes

#### Other v26 Changes
- Text selection prevention during drag operations (`user-select: none`)
- Responsive layout collapse fix (`flex-wrap: wrap` to `nowrap` to prevent hover causing single-column collapse)

### v29 (February 13, 2026) — Opacity Fast Path & Overlay Scroll Clamping
- **Overlay scroll clamping refined:** `updateStickyOverlays()` clamps overlay `top` so overlays stop at wrapper edges instead of getting cut off
- The cached fast recolor path (`_opacityCache`, `recolorImageOpacityFast()`) was added here — it goes through the full display pipeline (`updateDisplayCanvas` → `renderAtCurrentZoom`) which is part of what makes it slow

### v30 (February 14, 2026) — WebGL Direct Display (Option A)
This was a major overhaul of the display pipeline. Recoloring now runs as WebGL fragment shaders, rendering directly to a visible canvas at display resolution.

#### Three-Canvas Architecture
- **Hidden `canvas` (#imageCanvas):** 2D context, holds original-resolution pixel data. Still used for coordinate mapping, export, and as the source texture for WebGL.
- **`displayCanvas` (#displayCanvas):** 2D context, CPU fallback. Used when WebGL is unavailable or for initial image display before recoloring.
- **`webglCanvas` (#webglDisplayCanvas):** WebGL context, primary GPU-rendered display. Created in HTML, styled with `image-rendering: high-quality`.

#### No readPixels in Hot Path
- WebGL shaders render directly to the visible `webglCanvas` at display resolution — the user sees the GPU output immediately.
- CPU pixel data (needed for export, strip rebuild, luminosity post-processing) obtained lazily via `syncWebGLToCPU()`.

#### `renderWebGLToDisplay(type)`
- Core renderer. Sets `webglCanvas` dimensions to zoom-appropriate retina resolution using `devicePixelRatio` (not hardcoded `2`).
- Renders using cached shader uniforms (`_lastSimpleUniforms` or `_lastRBFUniforms`).
- Manages CSS sizing to match display dimensions.
- Called by `doRecolorSimpleWebGL()` and `doRecolorRBFWebGL()`.

#### `syncWebGLToCPU()`
- Lazy readback: re-renders at full image resolution (not display resolution), reads pixels via `readPixels`, flips Y rows (readPixels returns bottom-to-top), writes to hidden `canvas`.
- Re-renders display afterward to restore display-resolution output.

#### Cached WebGL State
- Uniform locations: `_simpleUniformLocs`, `_rbfUniformLocs` — avoids `gl.getUniformLocation()` every render.
- Buffers: `_webglPositionBuffer`, `_webglTexCoordBuffer` — fullscreen quad created once.
- Image texture: `_webglImageTexture` — uploaded once per image load.

#### Y-Flip
- Vertex shader flips texture Y: `1.0 - a_texCoord.y` (WebGL texture origin is bottom-left, image data is top-left).
- `syncWebGLToCPU` does row-swap after `readPixels` to match 2D canvas top-to-bottom convention.

#### Canvas Visibility Rules
- WebGL active (`_lastWebGLRenderType` = 'simple'/'rbf') → show `webglCanvas`, hide `displayCanvas` + `canvas`.
- CPU active (`_lastWebGLRenderType` = null) → show `displayCanvas`, hide `webglCanvas` + `canvas`.
- Initial/reset → show `displayCanvas` (original image via `drawImage()`).

#### `getVisibleCanvas()` Helper
- Returns whichever canvas is currently visible (webgl > display > canvas).
- Used for coordinate mapping (`getCanvasCoords`), bounding rect calculations, etc.

#### Opacity Fast Path (Updated)
- `recolorImageOpacityFast()` recalculates `diffLab` → `doRecolorSimpleWebGL()` → renders directly to `webglCanvas`. No readPixels.
- Strip rebuild debounced 200ms (requires CPU sync).

#### Texture Filtering
- Image textures use LINEAR filtering (smooth interpolation).
- LUT textures (RBF 3D grid) use NEAREST filtering (exact lookup).

#### CPU Fallback
- When WebGL context creation fails, all rendering falls back to `displayCanvas` with `drawImage()`.
- `_lastWebGLRenderType = null` signals CPU mode.

### v31 (February 15, 2026) — Logo Swap & Scroll Reset
- **Scroll to top on reload:** `window.scrollTo(0, 0)` in DOMContentLoaded.
- **Dual logo swap:** Tall logo shown initially, short logo replaces it at `image-loaded` stage. Reset restores tall logo. (Header-only change, no display pipeline impact.)

---

## Part 2: Current Architecture (As of v31)

### DOM Structure
```
canvasWrapper (overflow: hidden, position: relative)
  canvasInner (Panzoom-transformed via CSS translate)
    canvas#imageCanvas (HIDDEN — holds original-res pixel data, 2D context)
    canvas#displayCanvas (CPU FALLBACK — rendered at zoomed pixel size, 2D context)
    canvas#webglDisplayCanvas (PRIMARY — GPU-rendered recolor output, WebGL context)
    div.picker-marker (children positioned in canvasInner's local coordinate space)
```

### Canvas Roles
| Canvas | Purpose | Visible? | Size |
|--------|---------|----------|------|
| `canvas` (#imageCanvas) | Working pixel buffer — original pixel data, source for WebGL texture | No (display:none) | Original image dimensions (e.g. 2000x1500) |
| `displayCanvas` (#displayCanvas) | CPU fallback — used when WebGL unavailable or before first recolor | When WebGL inactive | `displayWidth * retina` pixel buffer, `displayWidth` CSS size |
| `webglCanvas` (#webglDisplayCanvas) | Primary display — GPU-rendered recolor shader output | When WebGL active | `displayWidth * devicePixelRatio` pixel buffer, `displayWidth` CSS size |

### The Full Render Pipeline (per recolor or zoom change)

#### WebGL Path (primary, v30+):
1. Recolor algorithm runs as WebGL fragment shader via `doRecolorSimpleWebGL()` or `doRecolorRBFWebGL()`
2. `renderWebGLToDisplay(type)` called:
   - Calculates display size from wrapper width + image aspect ratio + zoom
   - Sets `webglCanvas` dimensions to `displaySize * devicePixelRatio`
   - Binds cached shader program, sets cached uniform values
   - Draws fullscreen quad — shader output goes directly to screen
   - Sets CSS `width/height` to `displaySize`
   - Shows `webglCanvas`, hides `displayCanvas`
3. CPU sync (only when needed for export/strips/luminosity):
   - `syncWebGLToCPU()` re-renders at full image resolution
   - `readPixels()` → Y-flip → write to hidden `canvas`
   - Re-renders display at display resolution

#### CPU Fallback Path (when WebGL unavailable):
1. Recolor algorithm writes pixels to hidden `canvas` via `ctx.putImageData()`
2. `updateDisplayCanvas()` called:
   - Calculates base display size from wrapper width + image aspect ratio
   - Applies CSS `transform: scale()` for instant visual feedback
   - Debounces `renderAtCurrentZoom()` at 150ms
3. `renderAtCurrentZoom()`:
   - Calculates display size: `baseWidth * zoomLevel`
   - Calculates retina scale: `min(devicePixelRatio, imageWidth/displayWidth)`
   - Sets `displayCanvas.width/height` to `displaySize * retinaScale`
   - `displayCtx.drawImage(canvas, 0, 0, renderWidth, renderHeight)`
   - Sets CSS `width/height` to `displaySize`
   - Clears CSS scale transform
   - Updates `lastRenderedZoom`

### Zoom Flow
```
User scrolls → calculate new zoomLevel (±7%, snap to 1x if < 1.02)
  → zoom-to-cursor pan math
  → lock wrapper height, add 'zoomed' class
  → panzoomInstance.pan(newPan)
  → updateDisplayCanvas() → CSS scale instant + debounced hi-res render
  → constrainPan() → clamp edges
  → updateMarkers(), updateZoomDisplay(), updateStickyOverlays()
```

### Pan Flow
```
User Alt+drags → enable Panzoom pan
  → Panzoom handles mousemove → CSS translate on canvasInner
  → panzoomchange event fires
  → syncFromPanzoom() → read panX/panY
  → constrainPan() → clamp to edges
  → updateMarkers()
User mouseup → disable Panzoom pan
```

### Resize Flow
```
User drags resize handle → track delta
  → right: set canvasArea width, clear flex
  → bottom: set wrapper height, clear aspect-ratio (enables cropping)
  → corner: both
User releases → checkVerticallyCropped()
  → updateDisplayCanvas() → re-render at new size
  → updateMarkers(), updateStickyOverlays()
```

### Coordinate Mapping (screen click → image pixel)
```
getCanvasCoords(event):
  visibleCanvas = getVisibleCanvas()  // webglCanvas > displayCanvas > canvas
  rect = visibleCanvas.getBoundingClientRect()  // includes pan transform
  scaleX = canvas.width / rect.width            // image pixels per display pixel
  scaleY = canvas.height / rect.height
  imageX = floor((clientX - rect.left) * scaleX)
  imageY = floor((clientY - rect.top) * scaleY)
```
- `getVisibleCanvas()` returns whichever canvas is currently displayed (webgl > display > canvas)
- This works because `getBoundingClientRect()` accounts for Panzoom's CSS translate
- Markers placed in canvasInner's local space use `style.width` instead of `getBoundingClientRect()`

---

## Part 3: Edge Cases & Guards

### 1. Snap-to-1x Threshold
- **What:** If zoom calculation produces < 1.02, snap to exactly 1.0
- **Why:** Prevents lingering at 101% where layout can't decide if it's "zoomed" or not
- **Where:** Scroll handler (line ~903) and zoom slider (line ~4111)

### 2. Deferred 1x Layout Teardown
- **What:** When zoom returns to 1x, wait 200ms before removing 'zoomed' class and unlocking wrapper height
- **Why:** Continuous scroll through 1x would cause layout thrashing (lock/unlock/lock/unlock)
- **Guard:** Only tears down if `!isVerticallyCropped`
- **Where:** End of scroll zoom handler and zoom slider handler

### 3. Recursive constrainPan Prevention
- **What:** `_constrainingPan` flag prevents `constrainPan()` from calling itself
- **Why:** `panzoomInstance.pan()` fires a `panzoomchange` event, which calls `constrainPan()` again
- **Where:** constrainPan function (line ~4179)

### 4. CSS Scale vs Pixel-Perfect Mismatch
- **What:** Between instant CSS scale and debounced hi-res render, `constrainPan()` must handle both states
- **How:** `canvasW = Math.max(cW, wW * zoomLevel / lastRenderedZoom)` — uses the larger of actual canvas size or estimated zoomed size
- **Where:** constrainPan (lines ~4195-4196)

### 5. Mid-Layout-Transition Guard
- **What:** If wrapper has zero width (e.g., during flex transition), skip rendering
- **Guard:** `if (baseWidth < 1 || baseHeight < 1) return;`
- **Where:** renderAtCurrentZoom (line ~4330)

### 6. Vertical Cropping Detection Tolerance
- **What:** 2px tolerance when comparing wrapper height to natural height
- **Why:** Prevents flicker at exact boundary where floating-point rounding could toggle cropping on/off
- **Where:** checkVerticallyCropped (line ~4486)

### 7. Pan Allowed Conditions
- **What:** Pan only enabled when `zoomLevel > 1 || isVerticallyCropped`
- **Where:** Mousedown handler (line ~834), constrainPan (line ~4180)

### 8. Overlay Edge Clamping
- **What:** Overlays (zoom controls, picker panel) can't extend past wrapper top/bottom edges
- **How:** Clamp center position to `[halfHeight, wrapperHeight - halfHeight]`
- **Why:** Without this, overlays get cut off by `overflow: hidden` on wrapper
- **Where:** updateStickyOverlays clampCenter helper (line ~606)

### 9. Overlay Off-Screen Skip
- **What:** If wrapper is completely scrolled off viewport, skip overlay positioning
- **Guard:** `if (visibleHeight <= 0) return;`
- **Where:** updateStickyOverlays (line ~596)

### 10. Marker Drag Bounds Clamping
- **What:** When dragging a picker marker, clamp coords to `[0, width-1]` and `[0, height-1]`
- **Why:** Prevents out-of-bounds pixel reads from imageData
- **Where:** Marker drag handler (lines ~856-857)

### 11. Resize Minimum Dimensions
- **What:** Minimum width 300px, minimum height 150px during resize handle drag
- **Why:** Prevents collapsing the canvas area to unusable size
- **Where:** Resize mousemove handler (lines ~4414, ~4421)

### 12. Pan Centering at Small Zoom
- **What:** If zoomed image is smaller than wrapper in one axis, center it instead of allowing pan
- **How:** `minX = maxX = (wrapperWidth - canvasWidth) / 2`
- **Where:** constrainPan boundary math (lines ~4213-4225)

### 13. WebGL Canvas Visibility Sync (v30)
- **What:** Only one canvas visible at a time — `webglCanvas` when WebGL active, `displayCanvas` when CPU fallback
- **Why:** Two visible canvases would z-fight or double-render
- **Guard:** `renderWebGLToDisplay()` shows webglCanvas and hides displayCanvas; CPU path does the reverse
- **Where:** `renderWebGLToDisplay()`, `updateDisplayCanvas()`, `renderAtCurrentZoom()`

### 14. WebGL Y-Flip (v30)
- **What:** WebGL texture coordinates are bottom-to-top; image data is top-to-bottom
- **How:** Vertex shader flips Y: `1.0 - a_texCoord.y`. `syncWebGLToCPU()` does row-swap after `readPixels`.
- **Why:** Without both flips, image appears upside down on screen or in CPU readback

### 15. WebGL Lazy CPU Sync (v30)
- **What:** `syncWebGLToCPU()` only called when CPU pixel data is actually needed
- **Why:** `readPixels` is expensive and would negate the GPU performance benefit
- **When triggered:** Export, strip rebuild (debounced 200ms), luminosity post-processing
- **Guard:** Re-renders display at display resolution after sync (sync renders at image resolution)

### 16. WebGL Context Loss Fallback (v30)
- **What:** If WebGL context creation fails, entire pipeline falls back to CPU 2D canvas rendering
- **Guard:** `_lastWebGLRenderType = null` signals CPU mode; all render paths check this

---

## Part 4: Regression Test Checklist

Use this list to manually verify correct behavior after any display pipeline changes.

### Zoom Tests
- [ ] Scroll wheel zooms in/out smoothly (7% per tick)
- [ ] Zoom centers on cursor position (point under mouse stays fixed)
- [ ] Zoom slider works, centers on viewport center
- [ ] Zoom snaps to exactly 100% when close (scroll back from 107% should hit 100%, not 99.something%)
- [ ] Maximum zoom is 400%
- [ ] Cannot zoom below 100%
- [ ] Display is crisp/pixel-perfect at all zoom levels (no CSS blur)
- [ ] Zoom feels responsive — instant visual feedback before hi-res render

### Pan Tests
- [ ] Alt+drag pans the image when zoomed > 1x
- [ ] Cannot pan when at 1x zoom (unless vertically cropped)
- [ ] Image edges cannot be dragged past wrapper edges (no empty space visible)
- [ ] Pan works correctly after zoom-to-cursor (no jump)
- [ ] Releasing Alt+drag stops panning immediately
- [ ] Pan in picker mode requires Alt key

### Vertical Cropping Tests
- [ ] Drag bottom resize handle UP — image clips, doesn't rescale
- [ ] Can pan vertically to see clipped content (even at 1x zoom)
- [ ] Pan boundaries work correctly while cropped
- [ ] Reset View restores full image (un-crops)

### Resize Tests
- [ ] Right edge resize changes preview width
- [ ] Bottom edge resize changes preview height
- [ ] Corner resize changes both
- [ ] Minimum size enforced (300px wide, 150px tall)
- [ ] Image re-renders correctly after resize
- [ ] Markers update positions after resize
- [ ] No text selection during resize drag

### Display Quality Tests
- [ ] At 1x zoom, image displays at natural size with retina sharpness
- [ ] At 2x zoom, image still crisp (no interpolation blur)
- [ ] Recolored image displays correctly (not original)
- [ ] After recolor, zoom still works correctly
- [ ] After zoom, recolor still works correctly

### Reset View Tests
- [ ] Resets zoom to 100%
- [ ] Resets pan to (0, 0)
- [ ] Un-crops vertical cropping
- [ ] Restores wrapper to natural aspect ratio
- [ ] Clears manual resize overrides
- [ ] Zoom slider shows 100%

### Marker/Picker Tests
- [ ] Click on canvas in picker mode places marker at correct pixel
- [ ] Markers follow zoom (scale and reposition correctly)
- [ ] Markers follow pan (move with image)
- [ ] Dragging marker updates color at correct position
- [ ] Marker clicks work at zoomed+panned state
- [ ] Color picked is correct regardless of zoom level

### Overlay Tests
- [ ] Zoom controls overlay stays vertically centered in visible canvas area
- [ ] Picker overlay stays vertically centered in visible canvas area
- [ ] Overlays don't overflow past wrapper top/bottom edges
- [ ] Scrolling page repositions overlays smoothly
- [ ] Overlays disappear when canvas fully scrolled off screen

### Window Resize Tests
- [ ] Shrinking window re-renders canvas at new size
- [ ] Growing window re-renders canvas at new size
- [ ] Markers reposition correctly on window resize
- [ ] Overlays reposition correctly on window resize
- [ ] Zoom level preserved across window resize

### WebGL Display Tests (v30)
- [ ] Recolor renders via WebGL (webglCanvas visible, displayCanvas hidden)
- [ ] Image appears correct orientation (not flipped/mirrored)
- [ ] Display is crisp at retina resolution (uses devicePixelRatio)
- [ ] Opacity slider updates display without readPixels lag
- [ ] Export produces correct image (syncWebGLToCPU readback works)
- [ ] Color strips rebuild correctly after debounced CPU sync
- [ ] Luminosity post-processing works (triggers CPU sync)
- [ ] Switching between Simple and RBF algorithms — both render via WebGL
- [ ] Loading a new image — webglCanvas resets, texture re-uploaded
- [ ] Reset/remove image — falls back to displayCanvas for initial state
- [ ] CPU fallback works if WebGL context unavailable (displayCanvas takes over)
- [ ] Coordinate mapping (`getCanvasCoords`) works against webglCanvas
- [ ] Picker markers position correctly over webglCanvas

### Edge Case Combinations
- [ ] Zoom to 3x → resize wrapper smaller → pan → reset view (everything clean?)
- [ ] Vertically crop → zoom in → pan → zoom out to 1x (pan still works because cropped?)
- [ ] Rapid scroll zoom in and out (no layout thrashing, no stuck state)
- [ ] Zoom to 2x → recolor image → zoom still correct?
- [ ] Opacity slider while zoomed — display updates correctly?
- [ ] Load new image while zoomed — resets to 1x with correct sizing?
- [ ] Export image while zoomed — exports at original resolution, not display resolution?
- [ ] WebGL recolor → zoom → pan → export (full pipeline end-to-end)
- [ ] Rapid opacity slider drag while zoomed — no flicker or canvas visibility glitch

---

*Created: February 14, 2026*
*Updated: February 15, 2026*
*For use as regression reference during display pipeline refactoring*
