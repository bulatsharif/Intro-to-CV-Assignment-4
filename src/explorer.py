import logging
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


@dataclass
class SceneData:
    points: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    center: np.ndarray
    floor_height: float
    kd_tree: cKDTree


@dataclass
class GridMap:
    resolution: float
    origin: np.ndarray  # (x_min, z_min)
    occupancy: np.ndarray  # True if inflated obstacle
    density: np.ndarray
    room_mask: np.ndarray  # main connected free space


def load_scene(path: str) -> SceneData:
    logging.info("Loading scene %s", path)
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    pos = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=-1).astype(
        np.float32
    )

    dc = np.stack(
        (vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]), axis=-1
    ).astype(np.float32)
    colors = np.clip(0.5 + dc * 0.28209, 0.0, 1.0)

    scales = np.exp(
        np.stack((vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]), axis=-1)
    ).astype(np.float32)
    bounds_min = pos.min(axis=0)
    bounds_max = pos.max(axis=0)
    floor_height = float(np.percentile(pos[:, 1], 2.0))
    kd_tree = cKDTree(pos)
    return SceneData(
        points=pos,
        colors=colors,
        scales=scales,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        center=(bounds_min + bounds_max) / 2.0,
        floor_height=floor_height,
        kd_tree=kd_tree,
    )


class SceneExplorer:
    def __init__(
        self,
        scene: SceneData,
        grid_resolution: float,
        clearance_m: float,
        room_height: float = 2.4,
        seed: int = 17,
    ):
        self.scene = scene
        self.grid_resolution = grid_resolution
        self.clearance_m = clearance_m
        self.room_height = room_height
        self.random = random.Random(seed)
        self.grid_map = self._build_grid()

    def _build_grid(self) -> GridMap:
        height_mask = (self.scene.points[:, 1] - self.scene.floor_height) <= self.room_height
        x = self.scene.points[height_mask, 0]
        z = self.scene.points[height_mask, 2]
        x_min, x_max = x.min(), x.max()
        z_min, z_max = z.min(), z.max()
        origin = np.array([x_min, z_min], dtype=np.float32)
        gx = int(math.ceil((x_max - x_min) / self.grid_resolution)) + 1
        gz = int(math.ceil((z_max - z_min) / self.grid_resolution)) + 1
        density = np.zeros((gz, gx), dtype=np.int32)

        ix = ((x - x_min) / self.grid_resolution).astype(np.int32)
        iz = ((z - z_min) / self.grid_resolution).astype(np.int32)
        np.add.at(density, (iz, ix), 1)

        non_zero = density[density > 0]
        threshold = np.percentile(non_zero, 70) if non_zero.size > 0 else 1
        obstacle_mask = density > threshold
        inflated = self._inflate(obstacle_mask, self.clearance_m)

        free_mask = ~inflated
        room_mask = self._largest_component(free_mask)
        logging.info(
            "Grid built with shape %s, free ratio %.2f",
            density.shape,
            room_mask.mean(),
        )
        return GridMap(
            resolution=self.grid_resolution,
            origin=origin,
            occupancy=inflated,
            density=density,
            room_mask=room_mask,
        )

    def _inflate(self, mask: np.ndarray, clearance: float) -> np.ndarray:
        radius = max(1, int(math.ceil(clearance / self.grid_resolution)))
        padded = np.pad(mask, radius, mode="edge")
        inflated = mask.copy()
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dz * dz <= radius * radius:
                    inflated |= padded[
                        radius + dz : radius + dz + mask.shape[0],
                        radius + dx : radius + dx + mask.shape[1],
                    ]
        return inflated

    def _largest_component(self, free_mask: np.ndarray) -> np.ndarray:
        visited = -np.ones_like(free_mask, dtype=np.int32)
        h, w = free_mask.shape
        comp_id = 0
        best_id = -1
        best_size = 0
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(h):
            for c in range(w):
                if not free_mask[r, c] or visited[r, c] != -1:
                    continue
                stack = [(r, c)]
                visited[r, c] = comp_id
                size = 0
                while stack:
                    cr, cc = stack.pop()
                    size += 1
                    for dr, dc in offsets:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and free_mask[nr, nc]
                            and visited[nr, nc] == -1
                        ):
                            visited[nr, nc] = comp_id
                            stack.append((nr, nc))
                if size > best_size:
                    best_size = size
                    best_id = comp_id
                comp_id += 1
        if best_id == -1:
            return free_mask
        return visited == best_id

    def grid_to_world(self, cell: Tuple[int, int], height: float) -> np.ndarray:
        col, row = cell
        x = self.grid_map.origin[0] + (col + 0.5) * self.grid_map.resolution
        z = self.grid_map.origin[1] + (row + 0.5) * self.grid_map.resolution
        return np.array([x, height, z], dtype=np.float32)

    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        col = int((pos[0] - self.grid_map.origin[0]) / self.grid_map.resolution)
        row = int((pos[2] - self.grid_map.origin[1]) / self.grid_map.resolution)
        return col, row

    def _farthest_point_sampling(
        self, candidates: np.ndarray, k: int, start_idx: int
    ) -> List[int]:
        selected = [start_idx]
        dists = np.full(len(candidates), np.inf, dtype=np.float32)
        for _ in range(1, k):
            ref = candidates[selected[-1]]
            diff = candidates - ref
            dist_sq = np.sum(diff * diff, axis=1)
            dists = np.minimum(dists, dist_sq)
            idx = int(np.argmax(dists))
            selected.append(idx)
        return selected

    def choose_waypoints(self, num: int) -> List[Tuple[int, int]]:
        free_coords = np.argwhere(self.grid_map.room_mask)
        if free_coords.size == 0:
            raise RuntimeError("No free space found in scene grid")
        center = np.array(
            self.world_to_grid(self.scene.center), dtype=np.int32
        )[::-1]  # row, col
        start_idx = int(np.argmin(np.sum((free_coords - center) ** 2, axis=1)))
        sampled_ids = self._farthest_point_sampling(free_coords, num, start_idx)
        waypoints = []
        for idx in sampled_ids:
            r, c = free_coords[idx]
            waypoints.append((c, r))
        return waypoints

    def snap_to_free(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        c, r = cell
        if (
            0 <= r < self.grid_map.room_mask.shape[0]
            and 0 <= c < self.grid_map.room_mask.shape[1]
            and self.grid_map.room_mask[r, c]
        ):
            return cell
        h, w = self.grid_map.room_mask.shape
        best = None
        best_d = 1e9
        for rr in range(h):
            for cc in range(w):
                if not self.grid_map.room_mask[rr, cc]:
                    continue
                d = (rr - r) ** 2 + (cc - c) ** 2
                if d < best_d:
                    best_d = d
                    best = (cc, rr)
        if best is None:
            return cell
        return best

    def collision_free(self, pos: np.ndarray) -> bool:
        dist, _ = self.scene.kd_tree.query(pos)
        return dist >= self.clearance_m
