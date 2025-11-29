# Cinematic Navigation – Assignment 4

End-to-end pipeline for cinematic navigation in Gaussian Splatting scenes. The CLI uses Hydra configs to plan safe paths, render smooth coverage videos (including 360° orbits), and optionally run object detection on rendered frames.

## Quickstart
1. Install deps (consider a virtualenv):  
   ```bash
   pip install -r requirements.txt
   ```
2. Run the default tour (taichi renderer):  
   ```bash
   python -m src.main
   ```
3. Switch renderers or tune parameters via Hydra overrides, e.g.:  
   ```bash
   python -m src.main render.renderer=gsplat render.gsplat_device=cuda render.preview=true
   python -m src.main scene_path=/path/to/scene.ply exploration.num_waypoints=16 render.enable_360=false
   ```

Outputs land in `outputs/` (and Hydra run logs in `outputs/runs/...`).

## What’s inside
- **Path planning & exploration** (`src/explorer.py`, `src/path_planner.py`): builds an occupancy grid from the PLY, keeps the agent inside the main room, plans A* paths with collision checks, then smooths them with Catmull–Rom splines for cinematic motion. Clearance checks use a KD-tree built from splat positions.
- **Rendering engines** (`src/renderer.py`): taichi painter’s algorithm splatter (default) plus a gsplat path (uses GPU if installed). Both render per-pose RGB frames and can preview live via OpenCV.
- **360° coverage**: optional orbit appended after the coverage path; radius and frame count are configurable.
- **Object detection (optional)** (`src/detector.py`): lightweight HOG-based pass on rendered frames; saves JSON if enabled.
- **Analysis helper** (`src/analyze.py`): prints bounding boxes and stats for a scene.

## Configs
`configs/config.yaml` holds defaults. Key knobs:
- `scene_path`: PLY input.
- `exploration.*`: grid resolution, clearance, number of waypoints, camera height above the estimated floor, smoothing samples, FPS.
- `render.*`: renderer choice (`taichi`|`gsplat`), output size, FOV, 360° orbit toggle, preview stride, output paths.
- `detection.*`: enable and JSON output path.

Hydra lets you save runs to custom folders or compose multiple overrides; see `hydra.run.dir` in the config.

## Notes on renderers
- **Taichi**: CPU-backed kernel by default; adjust `ti.init` in `src/renderer.py` if you want Metal/CUDA.
- **gsplat**: Install the package + a matching PyTorch build and GPU drivers; set `render.renderer=gsplat` and `render.gsplat_device=cuda`. The code falls back to taichi if gsplat is missing or errors out.

## Real-time preview
Set `render.preview=true` to stream frames with OpenCV during rendering (disabled if OpenCV is not installed).

## 360-degree video
Enabled by default via `render.enable_360`; adjust `render.orbit_radius` and `render.orbit_frames` to control coverage.

## Limitations
- Large scenes may take time to render with the CPU Taichi path; gsplat is recommended when available.
- The default object detection is basic (HOG people detector). Integrate your own detector in `src/detector.py` if needed.
