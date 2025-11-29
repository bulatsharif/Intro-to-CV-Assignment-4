import logging
import math
import os
from dataclasses import dataclass
from typing import List, Sequence

import imageio.v2 as imageio
import numpy as np

from .path_planner import CameraPose


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = eye - target
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = eye
    return np.linalg.inv(mat).astype(np.float32)


@dataclass
class RenderConfig:
    width: int
    height: int
    fov_y: float
    renderer: str
    gsplat_device: str = "cpu"
    preview: bool = False
    preview_stride: int = 3


class GaussianRenderer:
    def __init__(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        scales: np.ndarray,
        cfg: RenderConfig,
    ):
        self.positions = positions
        self.colors = colors
        self.scales = scales.mean(axis=1)  # approx size per splat
        self.cfg = cfg
        self._taichi = None
        self._render_kernel = None

    def _ensure_taichi(self):
        if self._taichi is not None:
            return
        import taichi as ti

        ti.init(arch=ti.cpu)
        self._taichi = ti

        @ti.kernel
        def render_splat_kernel(
            points: ti.types.ndarray(),
            colors: ti.types.ndarray(),
            scales: ti.types.ndarray(),
            indices: ti.types.ndarray(),
            image: ti.types.ndarray(),
            w: ti.i32,
            h: ti.i32,
            view_matrix: ti.types.ndarray(),
            fx: ti.f32,
            fy: ti.f32,
            cx: ti.f32,
            cy: ti.f32,
        ):
            for i in range(indices.shape[0]):
                idx = indices[i]
                p_x = points[idx, 0]
                p_y = points[idx, 1]
                p_z = points[idx, 2]
                scale_world = scales[idx]

                cam_x = (
                    view_matrix[0, 0] * p_x
                    + view_matrix[0, 1] * p_y
                    + view_matrix[0, 2] * p_z
                    + view_matrix[0, 3]
                )
                cam_y = (
                    view_matrix[1, 0] * p_x
                    + view_matrix[1, 1] * p_y
                    + view_matrix[1, 2] * p_z
                    + view_matrix[1, 3]
                )
                cam_z = (
                    view_matrix[2, 0] * p_x
                    + view_matrix[2, 1] * p_y
                    + view_matrix[2, 2] * p_z
                    + view_matrix[2, 3]
                )

                if cam_z > 0.1:
                    u_center = (cam_x / cam_z) * fx + cx
                    v_center = (cam_y / cam_z) * fy + cy
                    radius_px = (scale_world * fx / cam_z) * 1.5
                    if radius_px < 1.0:
                        radius_px = 1.0
                    if radius_px > 18.0:
                        radius_px = 18.0
                    r_int = int(radius_px)
                    u_min = int(u_center - r_int)
                    u_max = int(u_center + r_int + 1)
                    v_min = int(v_center - r_int)
                    v_max = int(v_center + r_int + 1)
                    if u_min < 0:
                        u_min = 0
                    if u_max > w:
                        u_max = w
                    if v_min < 0:
                        v_min = 0
                    if v_max > h:
                        v_max = h
                    for v in range(v_min, v_max):
                        for u in range(u_min, u_max):
                            dx = u - u_center
                            dy = v - v_center
                            dist_sq = dx * dx + dy * dy
                            if dist_sq <= radius_px * radius_px:
                                image[v, u, 0] = colors[idx, 0]
                                image[v, u, 1] = colors[idx, 1]
                                image[v, u, 2] = colors[idx, 2]

        self._render_kernel = render_splat_kernel

    def _render_taichi(self, pose: CameraPose) -> np.ndarray:
        self._ensure_taichi()
        ti = self._taichi

        view_mat = look_at(pose.position, pose.target, pose.up)
        row3 = view_mat[2, :3]
        t_z = view_mat[2, 3]
        depths = self.positions @ row3 + t_z
        indices = np.argsort(depths)[::-1].astype(np.int32)

        image = np.zeros(
            (self.cfg.height, self.cfg.width, 3), dtype=np.float32
        )
        fx = (self.cfg.width / 2) / math.tan(self.cfg.fov_y / 2)
        fy = (self.cfg.height / 2) / math.tan(self.cfg.fov_y / 2)
        cx, cy = self.cfg.width / 2, self.cfg.height / 2

        self._render_kernel(
            self.positions,
            self.colors,
            self.scales,
            indices,
            image,
            self.cfg.width,
            self.cfg.height,
            view_mat,
            fx,
            fy,
            cx,
            cy,
        )
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    def _render_gsplat(self, pose: CameraPose) -> np.ndarray:
        try:
            import torch
            import gsplat
        except ImportError:
            logging.warning(
                "gsplat not available, falling back to taichi renderer"
            )
            return self._render_taichi(pose)

        try:
            device = torch.device(self.cfg.gsplat_device)
            means = torch.from_numpy(self.positions).to(device)
            colors = torch.from_numpy(self.colors).to(device)
            scales = torch.from_numpy(self.scales[:, None].repeat(3, axis=1)).to(device)
            opacities = torch.ones((self.positions.shape[0], 1), device=device) * 0.9
            quats = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], device=device
            ).repeat(self.positions.shape[0], 1)

            cam = torch.tensor(look_at(pose.position, pose.target, pose.up), device=device)
            tan_fovx = math.tan(self.cfg.fov_y / 2) * (self.cfg.width / self.cfg.height)
            tan_fovy = math.tan(self.cfg.fov_y / 2)
            rendered = gsplat.render(
                means3D=means,
                colors=colors,
                opacities=opacities,
                scales=scales,
                rotations=quats,
                width=self.cfg.width,
                height=self.cfg.height,
                tan_fovx=tan_fovx,
                tan_fovy=tan_fovy,
                bg_color=torch.zeros(3, device=device),
                viewmatrix=cam[:3, :],
            )
            image = rendered["render"] if isinstance(rendered, dict) else rendered
            image = torch.clamp(image, 0.0, 1.0)
            return (image.cpu().numpy() * 255.0).astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            logging.warning("gsplat render failed (%s), using taichi", exc)
            return self._render_taichi(pose)

    def render_sequence(
        self, poses: Sequence[CameraPose], preview: bool = False
    ) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        for idx, pose in enumerate(poses):
            method = self.cfg.renderer
            if method == "gsplat":
                frame = self._render_gsplat(pose)
            else:
                frame = self._render_taichi(pose)
            frames.append(frame)
            if preview and idx % max(1, self.cfg.preview_stride) == 0:
                self._preview_frame(frame, idx)
        if preview:
            self._close_preview()
        return frames

    def _preview_frame(self, frame: np.ndarray, idx: int) -> None:
        try:
            import cv2
        except ImportError:
            logging.info("Preview frame %d (cv2 not installed)", idx)
            return
        cv2.imshow("preview", frame[:, :, ::-1])
        cv2.waitKey(1)

    def _close_preview(self) -> None:
        try:
            import cv2
        except ImportError:
            return
        cv2.destroyAllWindows()


def write_video(frames: Sequence[np.ndarray], out_path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with imageio.get_writer(out_path, mode="I", fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    logging.info("Saved video to %s", out_path)
