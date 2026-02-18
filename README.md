# 🌹 Blooming Rose Field &nbsp;<a href="https://uk.mathworks.com/matlabcentral/fileexchange/183283-blooming-rose-field"><img src="https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg" height="30"></a>&nbsp;<a href="https://matlab.mathworks.com/open/github/v1?repo=Vasileios-Bellos/BloomingRoseField"><img src="https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg" height="30"></a>&nbsp;<a href="https://vasileios-bellos.github.io/BloomingRoseField/"><img src="https://img.shields.io/badge/Live_Demo-Interactive_3D-e6454d?style=flat" height="30"></a>

An interactive 3D flower field rendered entirely on MATLAB using parametric surfaces. A warm point light hovers above the field - roses within its influence radius bloom open; those outside gradually close back into buds. Guide the light with your mouse, arrow keys, or let it sweep the field on its own. Each rose sits on a botanically-inspired stem with a curved Bézier spine, five cupping sepals, and a handful of thorns - all built from first principles with no external meshes or textures. The rose head geometry is adapted from [Eric Ludlam's original work](https://github.com/Vasileios-Bellos/BloomingRoseField?tab=readme-ov-file#acknowledgements).

This project extends [Blooming Rose](https://github.com/Vasileios-Bellos/BloomingRose), which animates a single rose through a scripted bloom sequence, into a real-time interactive scene with multiple roses responding dynamically to user input.

<p align="center">
  <img src="BloomingRoseField.gif" alt="Blooming Rose Field" width="70%">
</p>

## Quick Start

```matlab
BloomingRoseField
```

The scene opens on fullscreen. Move your mouse to guide the light across the field and watch nearby roses bloom.

## Control Modes

Three modes let you guide the light source. Press **1**, **2**, or **3** to switch at any time - transitions are seamless, with the new mode picking up from the light's current position.

| Mode | Input | Behavior |
|------|-------|----------|
| **1 - Mouse / Touch** | Cursor movement | Light follows the mouse with smooth interpolation (default) |
| **2 - Arrow Keys** | Hold arrow keys | Light moves continuously, aligned to screen directions |
| **3 - Auto (Lissajous)** | Hands-free | Light traces a Lissajous path sweeping the full field |

The Lissajous path in mode 3 follows the trapezoidal field shape, with Y driven by `cos(t × 0.7)` and X by `sin(t × 1.0)`, producing an asymmetric figure-eight that covers the entire planting area. A gentle Z bobble (`sin(t × 0.4)`) adds vertical motion.

## Controls

| Key | Action |
|-----|--------|
| `1` | Mouse / Touch mode (default) |
| `2` | Arrow Keys mode |
| `3` | Auto (Lissajous) mode |
| `Space` | Pause / Unpause |
| `R` | Start / Stop recording |
| `Q` / `X` / `Esc` | Quit |
| Arrow keys | Move light (mode 2) |
| Mouse move | Light follows cursor (mode 1) |

## Colormap Library

Each rose is randomly assigned one of 19 colormaps at startup, all modeled after real cultivar colors. The palette is a subset of the 32 colormaps available in [Blooming Rose](https://github.com/Vasileios-Bellos/BloomingRose), covering the real-variety family:

*Classic Red*, *Juliet* (David Austin), *Amnesia*, *Quicksand*, *Sahara*, *Coral Reef*, *Hot Pink*, *Blush*, *Ocean Song*, *Golden Mustard*, *Ivory*, *Free Spirit*, *Burgundy*, *Rose Gold*, *White Mondial*, *Mint Green*, *Black Baccara*, *Café Latte*, and *Aobara* (Suntory Applause).

Vertex coloring uses `C = hypot(hypot(X, Y), Z × 0.9)` mapped through each rose's 256-entry colormap, giving smooth petal-depth shading under Gouraud lighting.

## Parameters

All parameters are defined at the top of the script.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `referenceFPS` | `30` | Target frame rate (`numeric` = real-time, `'fixed'` = constant timestep) |
| `nRoses` | `15` | Number of roses in the field |
| `n` | `100` | Mesh resolution (n × n grid per rose) |
| `A_rose` | `1.995653` | Petal height coefficient |
| `B_rose` | `1.27689` | Petal curl coefficient |
| `petalNum` | `3.6` | Petals per revolution |
| `nBloomLevels` | `60` | Discrete bloom states to precompute |
| `roseScale` | `0.5` | Base rose size (multiplied by per-rose random variation) |
| `bloomRate` | `3.0` | Bloom speed when inside influence radius |
| `unbloomRate` | `1.5` | Unbloom speed when outside influence radius |
| `stemLength` | `2.8` | Total stem length |
| `stemRadiusTop` | `0.032` | Radius near calyx |
| `stemRadiusBot` | `0.026` | Radius at base |
| `stemLeanMax` | `0.12` | Max lateral lean of top (fraction of height) |
| `stemBowMax` | `0.18` | Max mid-stem bow outward (fraction of height) |
| `nSepals` | `5` | Sepals per rose |
| `sepalLength` | `0.26` | Tip-to-base length |
| `sepalWidth` | `0.08` | Max width at midpoint |
| `sepalDroop` | `0.08` | Outward droop |
| `thornsPerStem` | `4` | Base thorn count (actual varies ±1 per stem) |
| `thornHeight` | `0.06` | Cone height |
| `thornRadius` | `0.015` | Cone base radius |
| `lightHeight` | `2.5` | Hover height above ground |
| `influenceRadius` | `2.0` | Bloom trigger radius |
| `fieldFrontY` | `-4` | Field front edge (near camera) |
| `fieldBackY` | `4` | Field back edge (far from camera) |
| `fieldFrontW` | `5.5` | Full width at front edge |
| `fieldBackW` | `11` | Full width at back edge |

## Recording and Exporting

Press **R** during any control mode to start recording. The HUD hides and the title bar shows **● REC**. Press **R** again to stop - the figure closes and an export dialog appears with three format options:

| Format | Notes |
|--------|-------|
| **MP4 Video** | Configurable FPS (default 60), Quality 95 |
| **Animated GIF** | Configurable FPS, optional dithering, global 256-color palette |
| **PNG Sequence** | Numbered frames, zero-padded filenames |

A **frame skip** control lets you keep every Nth frame for temporal downsampling (useful for smaller GIFs). The dialog loops after each export, so you can save the same recording in multiple formats before closing.

During recording, the frame timestep is fixed (independent of machine speed) to ensure even frame spacing in the output.

## Web Demo

A browser-based port is [available as a live demo](https://vasileios-bellos.github.io/BloomingRoseField/). Built with Three.js, it reproduces the animation with real-time 3D rendering and touch support, preserving the exact parametric equations, Bézier stem curves, Frenet-frame tube meshing, Rodrigues rotation and all 19 colormaps from the original MATLAB script. It renders 700 roses at 150×150 mesh resolution blooming over 120 frames, using a shared index buffer and frame-rate-independent delta-time scaling normalized to 30 fps.

## Standalone App (Windows)

`BloomingRoseField_installer.exe` installs a compiled standalone version that runs without the need for a local MATLAB installation. The installer will automatically download and install the MATLAB Runtime (R2025b) if it is not already present on the system.

## Technical Details

### Rose Head

A 100×100 parametric surface mesh defined by three constants from Eric Ludlam's original concept (`A = 1.995653`, `B = 1.27689`, `petalNum = 3.6`). The petal envelope equation is:

```
x = 1 − ½ · ((5/4) · (1 − mod(petalNum·θ, 2π)/π)² − ¼)²
```

60 bloom levels are precomputed at startup using `cospi`-based easing curves mapped from Eric Ludlam's original 48-level scheme. Each level stores a full n×n mesh of XYZ coordinates, indexed at runtime by the quantized bloom state.

### Stems

Each stem is a cubic Bézier curve from the ground plane to the rose head, with four control points: a vertical rise from the base (`P0 → P1`), a lateral curve through the mid-section (`P1 → P2`), and a smooth ease into the rose head (`P2 → P3`). Random lean gives each stem top a lateral offset from directly above its base; random mid-stem bow (roughly perpendicular to the lean) creates gentle S-curves and outward arcs. The Bézier spine is meshed into a tube by sweeping a circle along Frenet frames (tangent, normal, binormal from the curve derivative). The radius tapers from 0.032 (calyx) to 0.026 (base) with a Gaussian bulge at the calyx junction. The stem-top tangent determines the rose head orientation.

### Sepals

Five pointed leaf surfaces per rose with a `sin(πu)^0.6` width profile tapered by `(1 − u³)`. Placed at equal angular intervals around the stem top with inward cupping via `(1 − v²)` displacement. Orientation is set by a combined Rodrigues + Z rotation matrix (`buildHeadRotation`) derived from the stem-top tangent.

### Thorns

3–5 cones per stem (base count `thornsPerStem ± 1`) with `(1 − u)^1.5` radius falloff, placed along 12%–85% of the stem spine. Each thorn tilts 30° toward the stem tangent using a local frame built from the spine's Frenet vectors, with random size variation.

### Field Layout

Roses are placed within a trapezoidal region - narrow at the camera (5.5 units wide) and wide at the back (11 units wide) - matching the perspective frustum so the field fills the screen evenly. Placement uses a 3-zone strategy with equal rose count per zone:

- **Zone 1** (front 25%) - nearest to camera, narrowest strip
- **Zone 2** (middle 40%) - mid-field
- **Zone 3** (back 35%) - farthest from camera, widest strip

Equal count in narrower front zones produces higher apparent density near the camera, where detail matters most. Within each zone, Poisson-disk rejection sampling enforces a minimum separation scaled to the field area. Mouse and arrow-key movement is clamped to the trapezoid boundary plus a small margin.

### Bloom Dynamics

Each frame, the Euclidean distance from every rose head to the light source is computed in a single vectorized operation. Roses within the influence radius bloom at a rate proportional to proximity, with a quadratic falloff (`0.25 + 0.75 · proximity²`) that provides a floor to prevent choppy transitions at the edge. Roses outside the radius gradually close at a fixed `unbloomRate`. The continuous bloom state is quantized to an integer level, and only roses whose quantized level actually changed since the last frame get their surface geometry updated - the rest are skipped entirely.

### Performance

All 60 bloom-level meshes are precomputed and stored as cell arrays at startup. Per-rose transforms (scaled rotation matrix and world offset) are cached once and reused every frame. The skip-based update strategy means that on a typical frame, only a handful of the `nRoses` surfaces need a `set(hRoses(ri), ...)` call. Frame timing uses delta-time scaling for frame-rate independence, with optional fixed-timestep mode for recording.

## Requirements

MATLAB R2020a or later (uses `vecnorm`, `ndgrid`, `cospi`). No toolboxes required for the animation itself. GIF exporting uses `rgb2ind` from the Image Processing Toolbox.

## File Structure

```
BloomingRoseField.m              - MATLAB script: interactive field with 3 control modes, recording, exporting
BloomingRoseField_installer.exe  - Standalone app installer (Windows, requires MATLAB Runtime R2025b)
index.html                       - Web demo: Three.js port with 700 roses and touch support
BloomingRoseField.gif            - Animated GIF preview (low resolution)
BloomingRoseField.mp4            - Video recording (high resolution)
```

## Acknowledgements

Rose head parametric equations by **[Eric Ludlam](https://www.mathworks.com/matlabcentral/profile/authors/869244)**, from "[Blooming Rose](https://uk.mathworks.com/matlabcentral/communitycontests/contests/6/entries/13857)" - [MATLAB Flipbook Mini Hack](https://uk.mathworks.com/matlabcentral/communitycontests/contests/6/entries) contest (2023). [Source code on GitHub](https://github.com/zappo2/digital-art-with-matlab/tree/master/flowers).

## Author

**[Vasilis Bellos](https://www.mathworks.com/matlabcentral/profile/authors/13754969)** - rose field concept, colormap library, quantized bloom dynamics, interactive GUI, recording/exporting pipeline and Three.js web port.

## License

[MIT](LICENSE)
