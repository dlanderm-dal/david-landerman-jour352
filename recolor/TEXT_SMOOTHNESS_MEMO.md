# Text Smoothness in Palette Recoloring — Research Memo

**Problem:** Antialiased text in images looks bad after recoloring. The smooth gradient pixels ("rim") between letterforms and background get mapped to wrong colors, creating visible halos or graininess.

**Root cause:** Per-pixel independent color remapping breaks the spatial relationship between neighboring pixels. An antialiased rim pixel that is a 50/50 blend of black text and white background should become a 50/50 blend of the new text color and new background color — but it may instead get pulled toward an unrelated palette entry.

---

## Findings from Linked GitHub Repos

None of the six linked repos address this problem. They all do per-pixel independent recoloring:

- **ziap/repalette** — Nearest-neighbor + optional error-diffusion dithering (Floyd-Steinberg, etc.)
- **ashish0kumar/tint** — Shepard's method (inverse-distance-weighted), exposes `power` param (default 2.5)
- **rupareddy5/Palette-based-photo-recoloring** & **pipi3838/DIP_Final** — Chang 2015 SIGGRAPH paper implementation
- **danielgafni/repalette** — Deep neural networks/GANs (too heavy for browser)
- **b-z/photo_recoloring** — Another Chang 2015 implementation with L/ab separation

---

## Practical Solutions (Ranked by Impact & Feasibility)

### 1. Luminance-Chrominance Separation (Highest Impact, Moderate Shader Change)

Split recoloring into separate L and a,b transforms in LAB space:
- Map the pixel's L value through linear interpolation between old/new palette luminances
- Shift only the a,b channels using the weighted blend

Since text edges are primarily luminance variations, preserving L-channel gradients directly preserves antialiasing. This is what the Chang 2015 SIGGRAPH paper does.

**Implementation:** Modify the shader to apply `dLab` only to a,b channels, compute L shift separately as a linear ramp between the two nearest palette colors' luminance values.

### 2. Two-Nearest-Color Clamping on Edges (Single-Pass Shader)

Add Sobel edge detection (4 neighbor texture samples). On edge pixels, clamp palette consideration to only the 2 closest colors. Prevents contamination from unrelated palette entries.

```glsl
// Edge detection in shader
vec3 left  = rgb2lab(texture2D(u_image, v_texCoord + vec2(-1.0/u_width, 0.0)).rgb);
vec3 right = rgb2lab(texture2D(u_image, v_texCoord + vec2( 1.0/u_width, 0.0)).rgb);
vec3 up    = rgb2lab(texture2D(u_image, v_texCoord + vec2(0.0, -1.0/u_height)).rgb);
vec3 down  = rgb2lab(texture2D(u_image, v_texCoord + vec2(0.0,  1.0/u_height)).rgb);
float gradient = length(right - left) + length(down - up);
float edgeFactor = smoothstep(5.0, 30.0, gradient);
float sharpness = mix(u_blendSharpness, 0.5, edgeFactor); // softer on edges
```

Cost: 4 additional texture lookups + 4 rgb2lab conversions per pixel. Requires `u_width`/`u_height` uniforms.

### 3. Expose Blend Sharpness as User Control (Easiest)

Current `u_blendSharpness = 2.0` may be too aggressive for edge pixels. Let users dial it down when they see halos. Similar to `ashish0kumar/tint`'s `power` parameter.

### 4. Bilateral Filter Post-Pass (Most General, More Complex)

Two-pass rendering: recolor first, then bilateral filter that smooths halos while preserving real edges. Requires framebuffer object setup.

References:
- [Shadertoy bilateral filter](https://www.shadertoy.com/view/4dfGDH)
- [GLSL bilateral filter](https://github.com/tranvansang/bilateral-filter)
- [Bilateral Grid (Chen 2007 SIGGRAPH)](https://groups.csail.mit.edu/graphics/bilagrid/)

### 5. Improved Weight Function (Two-Nearest Barycentric)

For each pixel, find the two nearest palette colors, compute blend weight as relative position along the line segment between them:
- `t = dist_to_nearest1 / (dist_to_nearest1 + dist_to_nearest2)`
- `dLab = (1-t) * dLab[nearest1] + t * dLab[nearest2]`

For pixels far from the line between the two colors, fall back to full weighted blend.

### 6. Gradient-Domain / Poisson Reconstruction (Academic)

Compute recolored image, then solve Poisson equation to find image whose gradients match the original but has new colors. Fully preserves antialiasing but requires global linear solve — not suitable for single-pass fragment shader.

Reference: [Gradient-Preserving Color Transfer (Xiao 2009)](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2009.01566.x)

---

## Notes on Current RBF Shader

The RBF shader already uses manual trilinear interpolation on the 3D LUT (8-corner weighted blend), so `NEAREST` texture filtering is correct. If RBF still shows halos, the issue is in the LUT values themselves diverging too sharply at adjacent grid points, not in the interpolation.

---

## Recommended Implementation Order

1. **Luminance-chrominance separation** (Approach 1) — highest impact for text specifically
2. **Two-nearest-color clamping on edges** (Approach 2) — prevents contamination
3. **Expose blend sharpness slider** (Approach 3) — cheap UX escape hatch

All three can be implemented within the existing single-pass fragment shader without framebuffer objects or multi-pass rendering.

---

*Created: February 15, 2026*
*Updated: February 15, 2026*

## Implementation Status (v32)

### Completed
- **Full-resolution rendering** — Root cause of most text bleeding was zoom-dependent downsampling. Shader now always renders at `imgWidth × imgHeight`. This was the highest-impact fix.
- **Distance-based attenuation** — Gaussian attenuation `exp(-minDist² / 1800)` (sigma=30 ΔE) in Simple mode. Pixels far from all palette colors receive weaker shifts. Implemented in both GLSL shader and CPU fallback.

### Tried & Rolled Back
- **blendSharpness 2→8** — Too aggressive. Recolored subtle gray table lines that should have been left alone.
- **RBF ngrid 16→32** — No improvement on bleeding; rolled back.

### Not Yet Tried
- Luminance-chrominance separation (Approach 1)
- Two-nearest-color clamping on edges (Approach 2)
- Expose blend sharpness as user slider (Approach 3)
- Bilateral filter post-pass (Approach 4)
