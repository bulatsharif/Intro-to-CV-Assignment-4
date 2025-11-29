import logging
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .explorer import SceneData, SceneExplorer


@dataclass
class CameraPose:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray


class GridPathPlanner:
    def __init__(
        self,
        explorer: SceneExplorer,
        camera_height: float,
        max_step_m: float,
        look_ahead: int,
        smoothing_samples: int,
    ):
        self.explorer = explorer
        self.camera_height = camera_height
        self.max_step_cells = max(1, int(max_step_m / explorer.grid_resolution))
        self.look_ahead = look_ahead
        self.smoothing_samples = smoothing_samples

    def _neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        c, r = cell
        res = []
        max_delta = self.max_step_cells
        for dc in range(-max_delta, max_delta + 1):
            for dr in range(-max_delta, max_delta + 1):
                if dc == 0 and dr == 0:
                    continue
                nc, nr = c + dc, r + dr
                if (
                    0 <= nr < self.explorer.grid_map.room_mask.shape[0]
                    and 0 <= nc < self.explorer.grid_map.room_mask.shape[1]
                    and self.explorer.grid_map.room_mask[nr, nc]
                    and self._line_is_free(cell, (nc, nr))
                ):
                    res.append((nc, nr))
        return res

    def _line_is_free(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        ca, ra = a
        cb, rb = b
        steps = int(max(abs(ca - cb), abs(ra - rb)))
        if steps <= 1:
            return True
        for i in range(1, steps):
            t = i / steps
            c = int(round(ca + (cb - ca) * t))
            r = int(round(ra + (rb - ra) * t))
            if not self.explorer.grid_map.room_mask[r, c]:
                return False
        return True

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        import heapq

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self._neighbors(current):
                tentative = g_score[current] + self._step_cost(current, neighbor)
                if tentative < g_score.get(neighbor, 1e9):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f = tentative + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
        logging.warning("A* failed between %s and %s", start, goal)
        return [start, goal]

    def _step_cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        density = self.explorer.grid_map.density
        free_mask = self.explorer.grid_map.room_mask
        ca, ra = a
        cb, rb = b
        base = math.hypot(ca - cb, ra - rb)
        proximity_penalty = 1.0 + 0.1 * min(
            density[ra, ca] if free_mask[ra, ca] else 10,
            density[rb, cb] if free_mask[rb, cb] else 10,
        )
        return base * proximity_penalty

    def _catmull_rom(self, pts: Sequence[np.ndarray], samples: int) -> List[np.ndarray]:
        if len(pts) < 2:
            return list(pts)
        padded = [pts[0]] + list(pts) + [pts[-1]]
        smoothed = []
        for i in range(1, len(padded) - 2):
            p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
            for t in np.linspace(0, 1, samples, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                point = (
                    0.5
                    * (
                        (2 * p1)
                        + (-p0 + p2) * t
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                    )
                )
                smoothed.append(point)
        smoothed.append(pts[-1])
        return smoothed

    def _world_positions(self, cells: List[Tuple[int, int]]) -> List[np.ndarray]:
        positions = []
        for c, r in cells:
            positions.append(self.explorer.grid_to_world((c, r), self.camera_height))
        return positions

    def plan_through_waypoints(self, waypoints: List[Tuple[int, int]]) -> List[np.ndarray]:
        if not waypoints:
            raise ValueError("No waypoints provided")
        snapped = [self.explorer.snap_to_free(w) for w in waypoints]
        full_path: List[Tuple[int, int]] = [snapped[0]]
        for nxt in snapped[1:]:
            segment = self._astar(full_path[-1], nxt)
            if len(segment) > 1:
                full_path.extend(segment[1:])
        positions = self._world_positions(full_path)
        return self._enforce_clearance(positions)

    def _enforce_clearance(self, positions: List[np.ndarray]) -> List[np.ndarray]:
        safe_positions: List[np.ndarray] = []
        for pos in positions:
            if self.explorer.collision_free(pos):
                safe_positions.append(pos)
                continue
            cell = self.explorer.world_to_grid(pos)
            snapped = self.explorer.snap_to_free(cell)
            safe_positions.append(self.explorer.grid_to_world(snapped, pos[1]))
        return safe_positions

    def smooth_path(self, positions: List[np.ndarray]) -> List[np.ndarray]:
        if len(positions) < 3:
            return positions
        smoothed = self._catmull_rom(positions, self.smoothing_samples)
        return self._enforce_clearance(smoothed)

    def build_camera_poses(self, positions: List[np.ndarray], look_target: np.ndarray) -> List[CameraPose]:
        poses: List[CameraPose] = []
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for i, pos in enumerate(positions):
            j = min(len(positions) - 1, i + self.look_ahead)
            forward = positions[j] - pos
            if np.linalg.norm(forward) < 1e-3:
                forward = look_target - pos
            target = pos + forward
            poses.append(CameraPose(position=pos, target=target, up=up))
        return poses

    def build_orbit(self, center: np.ndarray, radius: float, frames: int) -> List[CameraPose]:
        poses: List[CameraPose] = []
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for t in np.linspace(0, 2 * math.pi, frames, endpoint=False):
            x = center[0] + radius * math.cos(t)
            z = center[2] + radius * math.sin(t)
            pos = np.array([x, center[1], z], dtype=np.float32)
            if not self.explorer.collision_free(pos):
                cell = self.explorer.world_to_grid(pos)
                snapped = self.explorer.snap_to_free(cell)
                pos = self.explorer.grid_to_world(snapped, center[1])
            poses.append(CameraPose(position=pos, target=center, up=up))
        return poses
