from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

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
    ) -> None:
        self.ply_path = ply_path
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.backend = backend
        self.device = device
        self.backend_available = False

        self.torch = None
        self.gsplat = None

        if backend == "gsplat":
            try:
                import torch  # type: ignore
                import gsplat  # type: ignore

                self.torch = torch
                self.gsplat = gsplat
                self.backend_available = True
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
    ) -> Optional[Path]:
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        if self.backend != "gsplat" or not self.backend_available:
            print("Rendering disabled; wrote camera_path files instead.")
            return None

        torch = self.torch  # type: ignore
        assert torch is not None and self.gsplat is not None
        gaussians = self._load_gaussians()
        fx = fx or float(self.width / 2)
        fy = fy or float(self.height / 2)
        cx = cx or float(self.width / 2)
        cy = cy or float(self.height / 2)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=self.device, dtype=torch.float32)

        bg = torch.zeros(3, device=self.device, dtype=torch.float32)
        if background is not None:
            bg = torch.tensor(background, device=self.device, dtype=torch.float32)

        for idx, pose in enumerate(trajectory.poses):
            view = self._pose_to_w2c(pose)
            view_t = torch.tensor(view, device=self.device, dtype=torch.float32)
            try:
                image = self.gsplat.render(
                    means3d=gaussians["means"],
                    scales=gaussians["scales"],
                    rotations=gaussians["rots"],
                    opacities=gaussians["opacity"],
                    shs=gaussians.get("shs"),
                    colors_precomp=None,
                    cov3d_precomp=None,
                    viewmats=view_t[None, ...],
                    Ks=K[None, ...],
                    width=self.width,
                    height=self.height,
                    packed=False,
                    background=bg,
                )
            except Exception as e:
                raise RuntimeError(f"gsplat render failed at frame {idx}: {e}") from e

            # gsplat may return either tensor or tuple; handle both.
            if isinstance(image, (list, tuple)):
                image = image[0]
            img_np = (
                image.clamp(0, 1)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            self._write_frame(frames_dir, idx, img_np)
        return frames_dir

    # ------------------------------------------------------------------ #
    def _load_gaussians(self) -> Dict[str, Any]:
        ply = PlyData.read(self.ply_path)
        v = ply["vertex"].data
        torch = self.torch  # type: ignore
        assert torch is not None
        names = set(v.dtype.names or [])

        def need(field: str) -> np.ndarray:
            if field not in names:
                raise KeyError(f"PLY missing required field '{field}' for rendering")
            return v[field]

        means = torch.tensor(
            np.stack([need("x"), need("y"), need("z")], axis=-1),
            device=self.device,
            dtype=torch.float32,
        )
        if "opacity" in names:
            opacity_np = np.clip(v["opacity"], 0.0, 1.0)
        else:
            opacity_np = np.ones(len(v), dtype=np.float32)
        opacity = torch.tensor(opacity_np, device=self.device, dtype=torch.float32)

        if {"scale_0", "scale_1", "scale_2"} <= names:
            scales_np = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)
        else:
            scales_np = np.ones((len(v), 3), dtype=np.float32)
        scales = torch.tensor(scales_np, device=self.device, dtype=torch.float32)

        if {"rot_0", "rot_1", "rot_2", "rot_3"} <= names:
            rots_np = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1)
        else:
            rots_np = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(v), 1))
        rots = torch.tensor(rots_np, device=self.device, dtype=torch.float32)

        # SH coefficients (f_dc_*, f_rest_*)
        sh_fields = [name for name in names if name.startswith("f_dc") or name.startswith("f_rest")]
        shs = None
        if sh_fields:
            shs = torch.tensor(
                np.stack([v[field] for field in sh_fields], axis=-1),
                device=self.device,
                dtype=torch.float32,
            )

        return {"means": means, "opacity": opacity, "scales": scales, "rots": rots, "shs": shs}

    def _pose_to_w2c(self, pose: CameraPose) -> np.ndarray:
        rot = R.from_quat(pose.rotation).as_matrix()
        trans = pose.position.reshape(3, 1)
        w2c = np.eye(4)
        w2c[:3, :3] = rot.T
        w2c[:3, 3] = -rot.T @ trans
        return w2c

    def _write_frame(self, frames_dir: Path, idx: int, image: np.ndarray) -> None:
        import imageio.v2 as imageio

        frame_path = frames_dir / f"frame_{idx:04d}.png"
        imageio.imwrite(frame_path, (image * 255).astype(np.uint8))
