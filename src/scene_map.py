from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np
from plyfile import PlyData


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


@dataclass
class SceneBounds:
    min_bound: np.ndarray
    max_bound: np.ndarray

    @property
    def size(self) -> np.ndarray:
        return self.max_bound - self.min_bound

    @property
    def max_dimension(self) -> float:
        return float(np.max(self.size))

    @property
    def center(self) -> np.ndarray:
        return (self.min_bound + self.max_bound) * 0.5


class SceneMap:
    """
    Handles PLY ingestion, robust bounding box computation, and voxel occupancy grid.
    """

    def __init__(
        self,
        ply_path: str,
        voxel_size: float = 0.5,
        opacity_threshold: float = 0.5,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
    ) -> None:
        self.ply_path = ply_path
        self.voxel_size = voxel_size
        self.opacity_threshold = opacity_threshold
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

        self.points: Optional[np.ndarray] = None
        self.opacities: Optional[np.ndarray] = None

        self.filtered_points: Optional[np.ndarray] = None
        self.bounds: Optional[SceneBounds] = None
        self.centroid: Optional[np.ndarray] = None
        self.floor_height: Optional[float] = None

        self.occupied_voxels: Set[Tuple[int, int, int]] = set()
        self.grid_shape: Optional[Tuple[int, int, int]] = None

        self._load_and_process()

    # Public helpers -----------------------------------------------------
    def is_free(self, idx: Tuple[int, int, int]) -> bool:
        if self.grid_shape is None:
            return False
        x, y, z = idx
        gx, gy, gz = self.grid_shape
        if not (0 <= x < gx and 0 <= y < gy and 0 <= z < gz):
            return False
        return idx not in self.occupied_voxels

    def world_to_grid(self, point: np.ndarray) -> Tuple[int, int, int]:
        assert self.bounds is not None
        relative = (point - self.bounds.min_bound) / self.voxel_size
        return tuple(np.floor(relative).astype(int))  # type: ignore[return-value]

    def grid_to_world(self, idx: Tuple[int, int, int]) -> np.ndarray:
        assert self.bounds is not None
        base = np.array(idx, dtype=float) * self.voxel_size
        # Shift to voxel center for smoother paths.
        return self.bounds.min_bound + base + self.voxel_size * 0.5

    def get_center_grid(self) -> Tuple[int, int, int]:
        assert self.bounds is not None
        center = self.bounds.center
        return self.world_to_grid(center)

    def find_nearest_free(
        self, start_idx: Tuple[int, int, int], max_radius: int = 5
    ) -> Tuple[int, int, int]:
        """
        If the chosen start voxel is occupied, search outward in a cube until a free voxel is found.
        """
        if self.is_free(start_idx):
            return start_idx
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        candidate = (start_idx[0] + dx, start_idx[1] + dy, start_idx[2] + dz)
                        if self.is_free(candidate):
                            return candidate
        return start_idx

    # Internal pipeline --------------------------------------------------
    def _load_and_process(self) -> None:
        ply = PlyData.read(self.ply_path)
        vertex = ply["vertex"]

        coords = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
        self.points = coords

        if "opacity" in vertex:
            opacities = np.asarray(vertex["opacity"])
            if opacities.min() < 0 or opacities.max() > 1:
                opacities = sigmoid(opacities)
        else:
            opacities = np.ones(coords.shape[0], dtype=np.float32)

        self.opacities = opacities
        mask = opacities >= self.opacity_threshold
        filtered = coords[mask]
        if filtered.shape[0] == 0:
            raise ValueError("No points remain after opacity filtering; adjust threshold.")

        self.filtered_points = filtered

        min_bound = np.percentile(filtered, self.percentile_low, axis=0)
        max_bound = np.percentile(filtered, self.percentile_high, axis=0)
        self.bounds = SceneBounds(min_bound=min_bound, max_bound=max_bound)

        # Use mean for orbit look-at; center uses robust bounds.
        self.centroid = np.mean(filtered, axis=0)
        self.floor_height = float(min_bound[1])

        self._voxelize(filtered)

    def _voxelize(self, points: np.ndarray) -> None:
        assert self.bounds is not None
        shifted = points - self.bounds.min_bound
        grid_indices = np.floor(shifted / self.voxel_size).astype(int)
        self.occupied_voxels = set(map(tuple, grid_indices))

        grid_extents = np.ceil((self.bounds.max_bound - self.bounds.min_bound) / self.voxel_size).astype(int) + 1
        self.grid_shape = (int(grid_extents[0]), int(grid_extents[1]), int(grid_extents[2]))

    # Reporting ----------------------------------------------------------
    def describe(self) -> str:
        assert self.bounds is not None
        occupied = len(self.occupied_voxels)
        grid_vol = math.prod(self.grid_shape) if self.grid_shape else 0
        sparsity = occupied / grid_vol if grid_vol else 0.0
        lines = [
            f"PLY: {self.ply_path}",
            f"Voxel size: {self.voxel_size} m",
            f"Bounds (1-{100 - self.percentile_high}%): min={self.bounds.min_bound}, max={self.bounds.max_bound}",
            f"Max dimension: {self.bounds.max_dimension:.2f} m",
            f"Occupied voxels: {occupied} / {grid_vol} ({sparsity:.2%})",
        ]
        return "\n".join(lines)
