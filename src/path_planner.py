from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from scene_map import SceneMap


VOXEL_MOVES = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == dy == dz == 0)
]


@dataclass
class CameraPose:
    position: np.ndarray  # shape (3,)
    rotation: np.ndarray  # quaternion (x, y, z, w)


@dataclass
class Trajectory:
    poses: List[CameraPose]
    mode: str  # "orbit" or "explorer"


class PathPlanner:
    def __init__(self, scene: SceneMap) -> None:
        self.scene = scene

    # ------------------------------------------------------------------ #
    def plan(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.bounds is not None
        if self.scene.bounds.max_dimension < 10.0:
            return self.plan_orbit(num_frames=num_frames)
        return self.plan_explorer(num_frames=num_frames)

    # ------------------------------------------------------------------ #
    def plan_orbit(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.bounds is not None
        assert self.scene.centroid is not None

        radius = max(self.scene.bounds.max_dimension * 0.8, self.scene.voxel_size * 4)
        base_height = self.scene.centroid[1] + self.scene.bounds.max_dimension * 0.15
        height_variation = self.scene.bounds.max_dimension * 0.05

        angles = np.linspace(0.0, 2 * math.pi, num_frames, endpoint=False)
        poses: List[CameraPose] = []
        for i, theta in enumerate(angles):
            spiral_scale = 0.9 + 0.1 * math.sin(0.5 * theta)
            r = radius * spiral_scale
            pos = np.array(
                [
                    self.scene.centroid[0] + r * math.cos(theta),
                    base_height + height_variation * math.sin(1.5 * theta),
                    self.scene.centroid[2] + r * math.sin(theta),
                ]
            )
            quat = self._look_at(pos, self.scene.centroid)
            poses.append(CameraPose(position=pos, rotation=quat))

        return Trajectory(poses=poses, mode="orbit")

    # ------------------------------------------------------------------ #
    def plan_explorer(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.bounds is not None

        center_world = self.scene.bounds.center
        start_world = center_world.copy()
        start_world[1] = self.scene.floor_height + 1.5 if self.scene.floor_height is not None else center_world[1]
        start_idx = self.scene.world_to_grid(start_world)
        start_idx = self.scene.find_nearest_free(start_idx, max_radius=5)

        a_idx, _ = self._bfs_farthest(start_idx)
        b_idx, _ = self._bfs_farthest(a_idx)

        path_grid = self._astar(a_idx, b_idx)
        if len(path_grid) < 2:
            # Degenerate case; stay put.
            pos = self.scene.grid_to_world(start_idx)
            quat = self._look_forward(np.array([0, 0, 1], dtype=float))
            return Trajectory(poses=[CameraPose(position=pos, rotation=quat)], mode="explorer")

        path_world = np.stack([self.scene.grid_to_world(p) for p in path_grid], axis=0)

        smooth_pos, tangents = self._smooth_path(path_world, num_frames=num_frames)
        quats = self._orientations_from_tangents(smooth_pos, tangents)

        poses = [
            CameraPose(position=pos, rotation=quat)
            for pos, quat in zip(smooth_pos, quats)
        ]
        return Trajectory(poses=poses, mode="explorer")

    # ------------------------------------------------------------------ #
    def _bfs_farthest(self, start: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], float]:
        visited = {start}
        q = deque([(start, 0)])
        farthest = start
        far_dist = 0.0

        while q:
            node, dist = q.popleft()
            if dist > far_dist:
                far_dist = dist
                farthest = node
            for nbr in self._neighbors(node):
                if nbr in visited:
                    continue
                if not self.scene.is_free(nbr):
                    continue
                visited.add(nbr)
                q.append((nbr, dist + 1))
        return farthest, far_dist

    def _astar(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        open_heap: List[Tuple[float, float, Tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start))

        came_from: dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        g_score = {start: 0.0}

        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for nbr in self._neighbors(current):
                if not self.scene.is_free(nbr):
                    continue
                tentative_g = g_curr + self._euclidean(current, nbr)
                if tentative_g < g_score.get(nbr, float("inf")):
                    came_from[nbr] = current
                    g_score[nbr] = tentative_g
                    f_score = tentative_g + self._euclidean(nbr, goal)
                    heapq.heappush(open_heap, (f_score, tentative_g, nbr))

        # If no path found, return the start.
        return [start]

    def _neighbors(self, node: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        x, y, z = node
        neighbors: List[Tuple[int, int, int]] = []
        for dx, dy, dz in VOXEL_MOVES:
            nbr = (x + dx, y + dy, z + dz)
            neighbors.append(nbr)
        return neighbors

    def _reconstruct_path(
        self, came_from: dict[Tuple[int, int, int], Tuple[int, int, int]], current: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------ #
    def _smooth_path(
        self, points: np.ndarray, num_frames: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(points) < 4:
            # Linear interpolation fallback.
            t = np.linspace(0, 1, len(points))
            ts = np.linspace(0, 1, num_frames)
            interp = np.stack([np.interp(ts, t, points[:, dim]) for dim in range(3)], axis=-1)
            tangents = np.gradient(interp, axis=0)
            return interp, tangents

        distances = np.zeros(len(points))
        distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        total = distances[-1]
        # Ensure strictly increasing to satisfy CubicSpline
        for i in range(1, len(distances)):
            if distances[i] <= distances[i - 1]:
                distances[i] = distances[i - 1] + 1e-4
        total = distances[-1]
        if total <= 1e-6:
            interp = np.repeat(points[:1], num_frames, axis=0)
            tangents = np.zeros_like(interp)
            return interp, tangents

        splines = [CubicSpline(distances, points[:, dim], bc_type="natural") for dim in range(3)]
        ts = np.linspace(0, total, num_frames)
        interp = np.stack([s(ts) for s in splines], axis=-1)
        tangents = np.stack([s(ts, 1) for s in splines], axis=-1)
        return interp, tangents

    def _orientations_from_tangents(
        self, positions: np.ndarray, tangents: np.ndarray
    ) -> np.ndarray:
        quats = []
        for pos, tan in zip(positions, tangents):
            if np.linalg.norm(tan) < 1e-6:
                tan = np.array([0.0, 0.0, 1.0])
            forward = tan / (np.linalg.norm(tan) + 1e-9)
            up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(forward, up)) > 0.95:
                up = np.array([1.0, 0.0, 0.0])
            right = np.cross(up, forward)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                right = np.array([1.0, 0.0, 0.0])
                right_norm = 1.0
            right /= right_norm
            up_corrected = np.cross(forward, right)
            rot = np.stack([right, up_corrected, -forward], axis=1)

            # Enforce right-handed, non-degenerate rotation.
            det = np.linalg.det(rot)
            if det < 0:
                rot[:, 0] *= -1  # flip right vector
            if not np.isfinite(rot).all() or abs(np.linalg.det(rot)) < 1e-6:
                rot = np.eye(3)

            quat = R.from_matrix(rot).as_quat()
            quats.append(quat)
        return np.asarray(quats)

    def _look_at(self, position: np.ndarray, target: np.ndarray) -> np.ndarray:
        forward = target - position
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, 1.0])
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, up)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right) + 1e-9
        up_corrected = np.cross(forward, right)
        rot = np.stack([right, up_corrected, -forward], axis=1)
        if np.linalg.det(rot) < 0:
            rot[:, 0] *= -1
        return R.from_matrix(rot).as_quat()

    def _look_forward(self, forward: np.ndarray) -> np.ndarray:
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, up)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right) + 1e-9
        up_corrected = np.cross(forward, right)
        rot = np.stack([right, up_corrected, -forward], axis=1)
        if np.linalg.det(rot) < 0:
            rot[:, 0] *= -1
        return R.from_matrix(rot).as_quat()

    @staticmethod
    def _euclidean(a: Sequence[int], b: Sequence[int]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
