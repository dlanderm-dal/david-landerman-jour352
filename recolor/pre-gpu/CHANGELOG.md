# Recolor Preview Tool - Changelog

This document compiles all feature requests, bug fixes, and changes from past conversations.

---

## Version History Summary

| Version | Date | Key Changes |
|---------|------|-------------|
| v2 | Pre-Feb | Base tool with k-means extraction, RBF recoloring |
| v3 | Pre-Feb | Added picker mode, zoom controls |
| v6 | Pre-Feb | Color harmony wheel added |
| v7 | Pre-Feb | Fixed color wheel (was black), fixed balloon positioning |
| v11 | Pre-Feb | Column-based mapping introduced |
| v12 | Pre-Feb | Origin bank for overflow colors, persistent numbered markers |
| v13 | Pre-Feb | Drag-and-drop origin swatches, responsive column layout |
| v14 | Pre-Feb | Algorithm toggle (Simple vs RBF), interactive harmony wheel |
| v18 | Pre-Feb | Theme presets, Adobe Color import |
| v19 | Pre-Feb | Origin X buttons, add swatch +, per-swatch pickers, config save/load, responsive layout |
| v20 | Pre-Feb | Bug fixes for v19, color tolerance slider, target category labels, category-based grouping |
| v21 | Feb 4 | Per-column bypass toggle, live preview toggle, persistent picker selections |
| v22 | Feb 4 | Recolored distribution strip, Re-extract fix, Adobe instructions, title update |
| v23 | Feb 4 | Fixed bidirectional color tolerance re-extract |
| v24 | Feb 8-9 | Progressive disclosure UI, Panzoom integration, major UI restructuring |
| v25 | Feb 9 | Template 7 theme applied, Lock a Color/Harmony toggle, collapsible origin, 14 changes |
| v26 | Feb 9 | Theme presets auto-apply, recolor history, step circles, vertical crop, 14 changes |
| v27 | Feb 10 | Tutorial overlay system, panel renames, button renames, slider fix |
| v28 | Feb 13 | Per-origin/target opacity, harmony controls rewrite, luminosity reset, revert removal |
| v29 | Feb 13 | Synthetic background opacity layering, vertical opacity slider, cached fast recolor |

---

## v29 Changes (February 13, 2026) - Session 2

### Opacity Behavior Overhaul
1. **Synthetic background layering for opacity** - Opacity no longer blends between origin and target colors. Instead, it now simulates the target color layered over the active background at the given opacity. The active background is determined by column 0 (CATEGORY_BACKGROUND): if the background column is bypassed/locked, uses average of original background origin colors; otherwise uses the target color for the background column. Added `getActiveBackgroundLab()` function.

2. **Vertical opacity slider UI** - Replaced the sun emoji button + horizontal popup slider with a compact vertical slider to the left of each origin swatch. 44px tall, rotated 270deg, starts translucent (35% opacity), visible on hover or when custom value set. Small percentage label appears at bottom when < 100%. Double-click resets to 100%.

3. **Cached fast recolor path** - Added `_opacityCache` and `recolorImageOpacityFast()` to avoid re-running the full recolor pipeline when only opacity changes. The cache stores full-opacity target LAB values and recomputes only the opacity blend. Combined with `autoRecolorOpacityOnly()` using `requestAnimationFrame` throttling for smooth slider interaction during live preview. Cache invalidated on palette/mapping changes, preserved on opacity-only changes.

---

## v28 Changes (February 13, 2026) - Session 1

### New Features
1. **Per-target opacity slider** - Each target swatch's color picker now includes an opacity slider (0-100%). Value saved when Apply is clicked. Effective opacity = origin opacity * target opacity.

2. **Per-origin opacity tool** - Sun icon button on each origin swatch opens a popup with slider, numeric input, and reset button for controlling opacity per-origin.

3. **Harmony controls rewrite** - Renamed Lock On Harmony buttons to "Generate Palette", "Randomize Everything", "Adjust Existing Colors to Harmony". Added base hue slider with rainbow gradient, saturation min/max, and lightness min/max controls. `harmonizePalette()` now respects `columnBypass`.

4. **Luminosity slider reset button** - Added reset button to restore luminosity to 0.

5. **Revert button removed** - Removed revert (↩) button from target swatches. Lock button now shows bypass origins preview with individual origin colors when hovered.

6. **Color Analysis panel split** - Split Color Analysis into a separate optional panel (`colorAnalysisOptionalPanel`) that can be collapsed independently.

### Bug Fixes
1. **Tutorial overlay category sync** - Fixed tutorial dropdown to sync with active category, added `categoryToTutorialStep()` helper.
2. **Reset button behavior** - Reset now clears all target swatches to null, resets opacities and luminosity, unchecks live preview.

---

## v27 Changes (February 10, 2026)

### Major New Feature: Tutorial Overlay System
- Semi-transparent dark overlay covering top ~20% of image viewer with step-by-step guided dialogues for color picking
- Step 1: "Select Background" (cyan highlight)
- Step 2: "Select Locked Colors" (orange highlight) with note about background separation
- Step 3+: "Select Accent Color N" (green highlight) with consolidation note
- "Next Color" button advances through categories
- Red "That's all my colors, let's move on!" button (same as Apply As Origin) from Accent 1 onward
- Right side shows Picking For selector, selected colors list, and Clear Selections
- "Hide Tutorial" collapses to tab; tab can re-expand
- Ctrl key temporarily hides ALL overlays (distinct from collapse)
- Tutorial only shows uncollapsed on FIRST picker activation; subsequent activations show collapsed tab
- Includes zoom/option-drag hint and Ctrl hide hint

### Panel and Button Renames
- "Color Analysis" → "Color Analysis (reference)"
- "Palette Mapping" → "Review Color Bucketing"
- Target Selector button → "Look's Good!"
- "Target Choice" → "Pick your new colors"
- "Apply Recolor" → "Apply this recolor"
- "Live Preview" → "Recolor Live"
- "Save Config" → "Save recolor configuration"
- "Export Saved" → "Export" (with active state highlighting when configs saved)

### UI Reorganization
- Moved Pick Colors button to top of Color Analysis box
- Moved "Apply this recolor" button between Theme Presets and Color Adjust divider
- Color consolidation slider track fix (was invisible - changed `var(--bg)` to `var(--border)`)

---

## v26 Changes (February 9, 2026) - Session 3

### New Features
1. **Theme presets auto-apply** - Clicking a theme applies immediately, removed separate "Apply Theme" button
2. **Recolor history system** - Replaced saved configurations list with automatic history. Every previewed recolor is logged. Each entry has individual save button with green checkmark "Saved!" text. Only saved configs are exported. Large blue "Save recolor configuration" button.
3. **Adobe Color link duplication** - Duplicate "Open Adobe Color" link above the collapsible details section
4. **Color Adjust box** - Wrapped algorithm toggle and Luminosity section inside gray box labeled "Color Adjust (optional)" with "Recolor algorithm" sublabel
5. **Step number circles** - Added numbered step circles (1-5) to Upload, Pick Colors, Target Selector, Apply Recolor, and Save Config buttons
6. **Upload icon redesign** - Replaced camera emoji with inline SVG (mountain landscape + sun in rectangle, Lucide/Feather style)
7. **Vertical image cropping** - Dragging bottom handle UP now clips/hides overflow instead of rescaling; user can pan to see hidden parts. Added `isVerticallyCropped` state and `checkVerticallyCropped()` function.
8. **Reset View enhancement** - Reset Zoom button now also resets wrapper height to natural image aspect ratio. Added "Reset View" text label.
9. **Origin drag hint** - "Drag to reorganize (optional)" text at top of Origin collapsible
10. **Live toggle relocation** - Moved Live toggle from Palette Mapping to below Apply Recolor button. When Live checked, Apply button greys out with "(Live is Active)" text.

### Bug Fixes
1. **Adobe Color link margin** - Reduced awkward top padding
2. **Origin collapsible re-opening** - Fixed un-collapsing when hitting Apply Recolor
3. **Auto-record live recolors** - Added debounced history entry when Live toggle auto-recolors
4. **Responsive layout collapse** - Fixed hover on left side causing single-column collapse (changed `flex-wrap: wrap` to `nowrap`)
5. **Save/History box** - Wrapped config buttons and history in labeled "Save / History" box
6. **Pick Colors button with step circle** - Fixed step circle disappearing when tool activated
7. **Lock toggle history** - Lock on/off now creates history entries
8. **Config save button visibility** - Fixed invisible white-on-white save button
9. **Pan boundary constraints** - Image edges can no longer be dragged past viewport edges
10. **Reset View button** - Multiple fixes to properly restore wrapper height
11. **Title change** - Changed to "Recolor Preview Tool" (removed version number)
12. **Text selection during resize** - Added `user-select: none` during drag operations

---

## v25 Changes (February 9, 2026) - Session 2

### Theme Application
1. **Template 7 theme applied** - Transformed entire app from dark theme (Space Grotesk, orange `#f97316`, dark `#0a0a0b`) to teal/coastal light theme (Plus Jakarta Sans, teal `#2b9eb3`, light `#f4f7f9`). Updated all CSS custom properties, borders, shadows, and border radii.

### New Features
2. **Image border visibility** - Canvas wrapper border changed to `2px solid rgba(120, 150, 175, 0.35)` for better boundary visibility
3. **Quick Harmony Bar** - Duplicated harmony controls below image preview, next to reset button, with sync between sidebar and quick bar
4. **Lock a Harmony / Lock a Color toggle** - Two-panel toggle switch in both sidebar and quick bar. "Lock a Harmony" works like existing behavior. "Lock a Color" lets you lock one color's hue and generate compatible harmonies. Custom swatch picker dropdowns matching Add Color bank visual pattern.
5. **Locked origin colors in Lock a Color picker** - Origin colors from locked/bank show as compact chips with short notation (B, L, A1, A2)
6. **Harmony algorithm rewrite** - Full rewrite of `randomizeLockColor()` with proper offset distribution, bank origin support, and +/-5 degree hue jitter
7. **Collapsible Origin section** - Origin swatches wrapped in `<details>` collapsible, collapses on target selection, stays open on config import

### Bug Fixes
8. **Shuffle updates live preview** - Shuffle now calls `autoRecolorImage()` and respects locked targets
9. **Recolored strip with locked targets** - Built `effectivePalette` substituting original colors for bypassed columns
10. **Config preserves locks** - `columnBypass` now saved/loaded in configs
11. **Harmony result label layout** - Fixed layout shift from different harmony name lengths
12. **Config backfills picker data** - Added `pickedColors`, `pickedPositions`, `pickedCategories` to config save/load
13. **Picker overlay button text** - "(add or subtract)" on new line, white text when active on teal background
14. **Hide picker sub-elements when inactive** - Hides swatch list, Apply, Clear, and balloons when picker deactivated; restores on re-engagement
15. **Harmony wheel for bank origins** - Added reference marker for bank colors on harmony wheel
16. **Tool legend updates** - Added shuffle and reset icons to reference section

---

## v24 Changes (February 8-9, 2026) - Session 1

### Major UI Restructuring: Progressive Disclosure System
Added a `uiStage` state machine tracking: `initial` → `image-loaded` → `colors-picked` → `target-selection` → `complete`. Panels progressively revealed as user advances. One-way disclosure - once revealed, panels never re-hide.

#### Stage 1 (Initial / Page Load)
- Image Preview centered at 50% width, sidebar hidden
- Algorithm toggle, color consolidation, and color strips moved to sidebar panels
- Zoom and pan hints displayed

#### Stage 2 (Image Uploaded)
- Image Preview moves left (60% width with draggable resize)
- Color Analysis panel appears with distribution strip, consolidation tool, Pick Colors button

#### Stage 3 (Colors Picked)
- Picker auto-disengages on Apply As Origin
- Picker instructions collapse

#### Stage 4 (Apply as Origin)
- Palette Mapping panel appears with origins loaded
- Targets start as null (grey X marks) - extensive null guards added across ~15 functions
- Target Selector button appears

#### Stage 5 (Target Selector Clicked)
- Mini palette buttons appear (paint/lock/revert)
- Add Color swatches show all extracted colors
- Target Choice panel with Import Theme, Color Harmony, Theme Presets, Luminosity
- Tool legend panel with emoji reference

#### Stage 6 (Complete)
- Image recolors, all tools remain visible permanently

### Panzoom Library Integration
- Added `@panzoom/panzoom@4.6.1` via CDN
- Panzoom handles ONLY panning; zoom via direct canvas re-rendering at zoomed pixel size (pixel-perfect, no CSS blur)
- Zoom-to-cursor math: `newPan = mousePos - (mousePos - oldPan) * (newZoom / oldZoom)`
- Reduced scroll sensitivity from +/-15% to +/-7%, snap-to-1x threshold, deferred layout teardown

### Other Changes
1. **Button overflow fix** - Added `flex-wrap: wrap` to button groups
2. **Sticky Pick Colors and Zoom overlays** - `updateStickyOverlays()` with scroll/resize listeners
3. **Renamed "Color Tolerance (Advanced)"** to "Color Consolidation (Increase this to combine similar colors)"
4. **Upload zone centering** - Replaced `margin-top: 100px` with flex centering
5. **Color Consolidation controls reorganized** - Default/Apply buttons below slider, Apply turns orange when dirty
6. **Color consolidation backend** - Improved `mergeColorsWithTolerance()` to use weighted LAB averaging
7. **Luminosity rework** - Removed from shader pipeline, created `applyLuminosityPostProcess()` as post-processing
8. **Picker category bug fix** - `Math.max(targetCount, 5)` ensures accent options always available
9. **Ctrl toggle for overlay peek** - Press Ctrl to hide picker overlay and zoom controls
10. **Randomize Harmony button** - Random base color with selected harmony model
11. **UI theme templates** - Created template.html + 9 themed variations for visual experimentation

---

## v23 Changes (February 4, 2026)

### Bug Fixes
1. **Color tolerance re-extract now works bidirectionally** - Fixed issue where increasing tolerance and re-extracting would merge colors, but lowering tolerance couldn't restore the original colors. Now stores raw k-means extraction separately (`rawColorDistribution`) and applies tolerance merging on top of it during re-extract.

---

## v22 Changes (February 4, 2026)

### New Features
1. **Recolored Distribution Strip** - Second color strip below the original showing recolored image distribution
2. **Adobe Color Instructions** - Collapsible instructions panel with direct link and step-by-step guide
3. **Updated Title** - "Graphic Design: Palette Recolor Preview Tool (v22)" with subtitle

### Bug Fixes
1. **Re-extract no longer overwrites picked colors** - Only recalculates percentages for existing origins

---

## v21 Changes (February 4, 2026)

### New Features
1. **Target category labels** - "Background", "Accent 1", "Accent 2", etc. below each target
2. **Color picker category selector** - Dropdown for Background, Locked, Accent categories with short labels
3. **Category-based origin grouping** - Colors grouped by category on Apply As Origin
4. **Persistent picker selections** - Preserved when closing/reopening picker
5. **Clickable category labels** - Click B/L/A1/A2 to cycle categories
6. **Per-column bypass/lock toggle** - Lock button next to each target's picker
7. **Live Preview toggle** - Auto-recolor on changes when enabled
8. **Remove Image button** - Clears everything for new image
9. **Theme/Adobe import triggers recolor** - Properly triggers recolor when Live enabled
10. **Instructions/Credits section** - Collapsible usage instructions and credits

---

## v20 Changes (February 4, 2026)

### Bug Fixes
1. Revert button now updates preview
2. Resize handles accessible during loading (z-index fix)
3. Picker balloons follow image on resize
4. Add swatch + appears in first empty column
5. Extra Colors bank always visible once image loaded
6. Shared color picker respects Set button (no auto-apply on drag)
7. Fixed duplicate markers after Apply As Origin

### New Features
1. **Color Tolerance Slider** - 0-50 slider controlling color merge aggressiveness
2. **Target swatch X buttons** - Same hover X delete style as origin swatches

---

## v19 Changes (February 4, 2026)

### New Features
1. **Origin swatch X buttons** - Remove individual origins from mapping
2. **Add swatch + button** - Dropdown of available colors with hex and percentages
3. **Per-swatch color pickers** - Full gradient picker with hue slider, hex input, preview
4. **Configuration save/load** - Save, load, export/import JSON configs
5. **Resizable preview panel** - Drag edges/corner to resize
6. **Responsive layout** - Vertical stacking on narrow viewports, horizontal scroll for many targets

---

## Bug Fixes Log

### Color Wheel Was Black (Fixed in v7)
- **Cause:** `hslToRgb()` expected saturation/lightness as 0-100, code passed decimals
- **Fix:** `hslToRgb(angle, 0.8, 0.5)` → `hslToRgb(angle, 80, 50)`

### Balloon/Marker Positioning Broken (Fixed in v7)
- **Cause:** Markers used raw canvas coordinates instead of CSS display coordinates
- **Fix:** Convert canvas coordinates accounting for zoom level

### Recolor Only Applied to First Origin in Column (Fixed v14)
- **Cause:** RBF included "bank" origins with zero diffLab, diluting transformation
- **Fix:** Added Simple algorithm that handles many-to-one mapping correctly

### Many-to-One Color Mapping Crash (Fixed v3)
- **Cause:** Tried to access `newLab[i]` for nonexistent indices
- **Fix:** `origin[i]` maps to `target[i % targetCount]`

### Zoomed Image Blurry (Fixed v24)
- **Cause:** Panzoom CSS `transform: scale()` stretches rasterized bitmap
- **Fix:** Pan-only Panzoom; zoom via direct canvas rendering at pixel resolution

### Accent Categories Missing in Picker (Fixed v24)
- **Cause:** `for (i=1; i<targetCount)` with targetCount=1 produced zero options
- **Fix:** `Math.max(targetCount, 5)` ensures at least 4 accent options

### Color Merging Not Perceptually Accurate (Fixed v24)
- **Cause:** Averaging done in RGB space
- **Fix:** Changed to LAB space averaging in `mergeColorsWithTolerance()`

---

*Last updated: February 13, 2026*
*Compiled from Claude Code conversation history*
