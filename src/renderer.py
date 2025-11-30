from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R

from path_planner import CameraPose, Trajectory


def save_trajectory(trajectory: Trajectory, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    poses_out = [
        {
            "position": pose.position.tolist(),
            "quaternion_xyzw": pose.rotation.tolist(),
        }
        for pose in trajectory.poses
    ]
    data = {"mode": trajectory.mode, "poses": poses_out}
    out_path = output_dir / "camera_path.json"
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)

    np.savez_compressed(
        output_dir / "camera_path.npz",
        positions=np.stack([p.position for p in trajectory.poses], axis=0),
        quaternions=np.stack([p.rotation for p in trajectory.poses], axis=0),
        mode=trajectory.mode,
    )
    return out_path


def assemble_video(frames_dir: Path, output_path: Path, fps: int = 30) -> None:
    import imageio.v2 as imageio

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frame_files:
            writer.append_data(imageio.imread(frame))


class Renderer:
    """
    Thin wrapper around gsplat/diff-gaussian-rasterization. Defaults to a dry-run
    that only stores the camera path so development on CPU-only machines remains possible.
    """

    def __init__(
        self,
        ply_path: str,
        output_dir: Path,
        width: int = 800,
        height: int = 800,
        backend: str = "none",
        device: str = "cuda",
        rng_seed: int = 0,
    ) -> None:
        self.ply_path = ply_path
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.backend = backend
        self.device = device
        self.rng_seed = rng_seed
        self.backend_available = False

        self.torch = None
        self.gsplat = None
        self._render_fn = None

        if backend == "gsplat":
            try:
                import torch  # type: ignore
                import gsplat  # type: ignore

                self.torch = torch
                self.gsplat = gsplat
                self._render_fn = self._resolve_render_fn(gsplat)
                if self._render_fn is not None:
                    self.backend_available = True
                else:
                    print(
                        "gsplat found but no known render function; rendering disabled.\n"
                        "Expected entrypoints include: gsplat.rendering.rasterization (v1.5+), "
                        "gsplat.render, gsplat.rendering.render/forward/render_cuda."
                    )
            except ImportError:
                print("gsplat not available; falling back to dry-run.")

    # ------------------------------------------------------------------ #
    def render_trajectory(
        self,
        trajectory: Trajectory,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        background: Optional[np.ndarray] = None,
        max_points: Optional[int] = None,
        tile_size: int = 16,
        render_mode: str = "RGB",
        packed: bool = True,
    ) -> Optional[Path]:
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        if self.backend != "gsplat" or not self.backend_available:
            print("Rendering disabled; wrote camera_path files instead.")
            return None

        torch = self.torch  # type: ignore
        assert torch is not None and self._render_fn is not None
        gaussians = self._load_gaussians(max_points=max_points)
        fx = fx or float(self.width / 2)
        fy = fy or float(self.height / 2)
        cx = cx or float(self.width / 2)
        cy = cy or float(self.height / 2)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=self.device, dtype=torch.float32)

        bg = torch.zeros(3, device=self.device, dtype=torch.float32)
        if background is not None:
            bg = torch.tensor(background, device=self.device, dtype=torch.float32)
        bg = bg.to(self.device).float()

        # backgrounds shape must match image_dims + (channels,)
        backgrounds = bg if packed else bg.view(1, 1, -1)

        colors, sh_degree = self._prepare_colors(gaussians)
        
        print(f"[Renderer] Starting render of {len(trajectory.poses)} frames...")
        
        for idx, pose in enumerate(trajectory.poses):
            view = self._pose_to_w2c(pose)
            view_t = torch.tensor(view, device=self.device, dtype=torch.float32)
            try:
                image, _, _ = self._render_fn(
                    means=gaussians["means"],
                    quats=gaussians["rots"],
                    scales=gaussians["scales"],
                    opacities=gaussians["opacity"],
                    colors=colors,
                    viewmats=view_t[None, ...],
                    Ks=K[None, ...],
                    width=self.width,
                    height=self.height,
                    sh_degree=sh_degree,
                    packed=packed,
                    backgrounds=backgrounds,
                    tile_size=tile_size,
                    render_mode=render_mode,
                )
            except Exception as e:
                raise RuntimeError(f"gsplat render failed at frame {idx}: {e}") from e

            # rasterization returns (..., C, H, W, 3); drop batch/camera dims.
            if image.ndim == 5:
                img_np = image[0, 0].clamp(0, 1).detach().cpu().numpy()
            elif image.ndim == 4:
                img_np = image[0].clamp(0, 1).detach().cpu().numpy()
            else:
                img_np = image.clamp(0, 1).detach().cpu().numpy()
            self._write_frame(frames_dir, idx, img_np)
        return frames_dir

    # ------------------------------------------------------------------ #
    def _load_gaussians(self, max_points: Optional[int] = None) -> Dict[str, Any]:
        ply = PlyData.read(self.ply_path)
        v = ply["vertex"].data
        torch = self.torch  # type: ignore
        assert torch is not None
        names = set(v.dtype.names or [])

        def need(field: str) -> np.ndarray:
            if field not in names:
                raise KeyError(f"PLY missing required field '{field}' for rendering")
            return v[field]

        coords = np.stack([need("x"), need("y"), need("z")], axis=-1)
        
        # --- ACTIVATION FIX: Opacity (Sigmoid) ---
        if "opacity" in names:
            op_raw = v["opacity"]
            # Heuristic: standard 3DGS stores logits (negative/positive floats). 
            # If values are outside [0,1], we assume logits.
            if op_raw.min() < 0 or op_raw.max() > 1:
                opacity_np = 1 / (1 + np.exp(-op_raw))
            else:
                opacity_np = np.clip(op_raw, 0.0, 1.0)
        else:
            opacity_np = np.ones(len(v), dtype=np.float32)

        # --- ACTIVATION FIX: Scale (Exp) ---
        if {"scale_0", "scale_1", "scale_2"} <= names:
            scales_raw = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)
            # Standard 3DGS stores log-scales. We must apply exp() to get physical size.
            scales_np = np.exp(scales_raw)
        else:
            scales_np = np.ones((len(v), 3), dtype=np.float32) * 0.01

        rots_np = (
            np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1)
            if {"rot_0", "rot_1", "rot_2", "rot_3"} <= names
            else np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(v), 1))
        )
        
        # Normalize quaternions (crucial for valid rendering)
        rots_norm = np.linalg.norm(rots_np, axis=-1, keepdims=True)
        rots_np = rots_np / (rots_norm + 1e-9)

        # SH coefficients (f_dc_*, f_rest_*)
        sh_fields = [name for name in names if name.startswith("f_dc") or name.startswith("f_rest")]
        shs = None
        if sh_fields:
            shs = np.stack([v[field] for field in sh_fields], axis=-1)

        coords, opacity_np, scales_np, rots_np, shs = self._maybe_subsample(
            coords, opacity_np, scales_np, rots_np, shs, max_points
        )

        means = torch.tensor(coords, device=self.device, dtype=torch.float32)
        opacity = torch.tensor(opacity_np, device=self.device, dtype=torch.float32)
        scales = torch.tensor(scales_np, device=self.device, dtype=torch.float32)
        rots = torch.tensor(rots_np, device=self.device, dtype=torch.float32)
        shs_t = torch.tensor(shs, device=self.device, dtype=torch.float32) if shs is not None else None

        return {"means": means, "opacity": opacity, "scales": scales, "rots": rots, "shs": shs_t}

    def _maybe_subsample(
        self,
        coords: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        rots: np.ndarray,
        shs: Optional[np.ndarray],
        max_points: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if max_points is None or coords.shape[0] <= max_points:
            return coords, opacities, scales, rots, shs
        rng = np.random.default_rng(self.rng_seed)
        idx = rng.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[idx]
        opacities = opacities[idx]
        scales = scales[idx]
        rots = rots[idx]
        shs = shs[idx] if shs is not None else None
        return coords, opacities, scales, rots, shs

    def _prepare_colors(self, gaussians: Dict[str, Any]):
        torch = self.torch  # type: ignore
        assert torch is not None
        shs = gaussians.get("shs")
        if shs is None:
            # fallback: use opaque white
            colors = torch.ones((gaussians["means"].shape[0], 3), device=self.device, dtype=torch.float32)
            return colors, None

        # shs shape: (N, F). Convert to (N, K, 3)
        N, F = shs.shape
        if F % 3 != 0:
            raise ValueError(f"Unexpected SH layout: {F} not divisible by 3")
        K = F // 3
        sh_degree = int(round(math.sqrt(K) - 1))
        if (sh_degree + 1) ** 2 != K:
            # Fallback for weird SH counts, treat as Degree 0 (Diffuse)
            print(f"[Warning] SH count {F} doesn't map to a standard degree. Truncating to Degree 0.")
            colors = shs[:, :3].view(N, 1, 3)
            return colors, 0
            
        colors = shs.view(N, K, 3)
        return colors, sh_degree

    def _resolve_render_fn(self, gsplat_mod: Any):
        """
        Handle gsplat API differences across releases.
        Tries a handful of known entrypoints and returns the first callable found.
        """
        # Prefer documented rasterization entrypoint (gsplat>=1.5)
        if hasattr(gsplat_mod, "rendering"):
            rendering = gsplat_mod.rendering
            for name in ("rasterization", "render", "forward", "render_cuda"):
                fn = getattr(rendering, name, None)
                if callable(fn):
                    return fn
        # Older/alternate layout
        for name in ("render",):
            fn = getattr(gsplat_mod, name, None)
            if callable(fn):
                return fn
        return None

    def _pose_to_w2c(self, pose: CameraPose) -> np.ndarray:
        rot = R.from_quat(pose.rotation).as_matrix()
        trans = pose.position.reshape(3, 1)
        w2c = np.eye(4)
        w2c[:3, :3] = rot.T
        w2c[:3, 3] = (-rot.T @ trans).reshape(3)
        return w2c

    def _write_frame(self, frames_dir: Path, idx: int, image: np.ndarray) -> None:
        import imageio.v2 as imageio

        frame_path = frames_dir / f"frame_{idx:04d}.png"
        imageio.imwrite(frame_path, (image * 255).astype(np.uint8))