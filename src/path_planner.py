from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

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

    def plan(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.bounds is not None
        # Heuristic: If scene is small (<10m), orbit. If large, explore.
        print(f"[Planner] Scene Dimension: {self.scene.bounds.max_dimension:.2f}m")
        if self.scene.bounds.max_dimension < 10.0:
            print("[Planner] Mode: ORBIT (Small Object)")
            return self.plan_orbit(num_frames=num_frames)
        print("[Planner] Mode: EXPLORER (Large Environment)")
        return self.plan_explorer(num_frames=num_frames)

    # ------------------------------------------------------------------ #
    def plan_orbit(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.bounds is not None
        assert self.scene.centroid is not None

        radius = max(self.scene.bounds.max_dimension * 0.9, self.scene.voxel_size * 5)
        base_height = self.scene.centroid[1] + self.scene.bounds.max_dimension * 0.1
        height_variation = self.scene.bounds.max_dimension * 0.05

        angles = np.linspace(0.0, 2 * math.pi, num_frames, endpoint=False)
        poses: List[CameraPose] = []
        for i, theta in enumerate(angles):
            spiral_scale = 1.0 + 0.05 * math.sin(4 * theta)
            r = radius * spiral_scale
            
            x = self.scene.centroid[0] + r * math.cos(theta)
            z = self.scene.centroid[2] + r * math.sin(theta)
            y = base_height + height_variation * math.sin(2 * theta)
            
            pos = np.array([x, y, z])
            quat = self._look_at(pos, self.scene.centroid)
            poses.append(CameraPose(position=pos, rotation=quat))

        return Trajectory(poses=poses, mode="orbit")

    # ------------------------------------------------------------------ #
    def plan_explorer(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.bounds is not None
        
        # --- FIX 1: Robust Start Position Finding ---
        # Instead of bounding box center (which might be outside), use the Data Centroid
        # (The average position of all points is usually inside the room)
        start_world = self.scene.centroid.copy()
        
        # Adjust Height: Find local floor
        # We look at points in a cylinder around the centroid to find the floor Y
        if self.scene.filtered_points is not None:
             # Take points within 5m of center
             dists = np.linalg.norm(self.scene.filtered_points[:, [0, 2]] - start_world[[0, 2]], axis=1)
             nearby_points = self.scene.filtered_points[dists < 5.0]
             if len(nearby_points) > 0:
                 # Floor is the 5th percentile of Y in this area
                 floor_y = np.percentile(nearby_points[:, 1], 5)
                 print(f"[Planner] Detected Floor Y: {floor_y:.2f}")
                 start_world[1] = floor_y + 1.8  # Eye level
             else:
                 print("[Planner] Warning: No points near center, using global centroid.")

        start_idx = self.scene.world_to_grid(start_world)
        
        # Ensure we start in free space (Search Spiral)
        print(f"[Planner] Searching for free space near {start_idx}...")
        start_idx = self.scene.find_nearest_free(start_idx, max_radius=20)
        
        # --- FIX 2: Enclosure Check (Am I outside?) ---
        # If we are "outside", rays will escape to infinity. We want to be "inside".
        # If outside, move towards the center of the bounding box until enclosed.
        if not self._is_enclosed(start_idx):
            print("[Planner] Start point seems outside. Moving towards center...")
            center_idx = self.scene.get_center_grid()
            # March from current -> center until we hit a wall, then step through it
            start_idx = self._march_to_enclosure(start_idx, center_idx)

        print(f"[Planner] Final Start Voxel: {start_idx}")

        # 2. Strategy: Find the "Diameter"
        print("[Planner] Flood Filling to find Point A...")
        a_idx, _ = self._bfs_farthest(start_idx)
        print(f"[Planner] Point A found at {a_idx}. Flood Filling for Point B...")
        b_idx, _ = self._bfs_farthest(a_idx)
        print(f"[Planner] Point B found at {b_idx}.")

        # 3. Pathfinding
        print(f"[Planner] Running A* from {a_idx} to {b_idx}...")
        path_grid = self._astar(a_idx, b_idx)
        
        print(f"[Planner] A* Path Length: {len(path_grid)} voxels")

        if len(path_grid) < 5:
            print("[Planner] CRITICAL: Path too short. Trying simple A -> Start -> B path.")
            # Fallback: Just go from Start to A
            path_grid = self._astar(start_idx, a_idx)
            if len(path_grid) < 5:
                 print("[Planner] ERROR: Agent is trapped. Hovering.")
                 pos = self.scene.grid_to_world(start_idx)
                 quat = self._look_forward(np.array([0, 0, 1], dtype=float))
                 return Trajectory(poses=[CameraPose(position=pos, rotation=quat)]*num_frames, mode="explorer")

        # 4. Convert to World Coordinates
        path_world = np.stack([self.scene.grid_to_world(p) for p in path_grid], axis=0)

        # 5. Subsampling (Fix Jerkiness)
        subsample_step = max(1, len(path_world) // 20) 
        subsample_step = min(subsample_step, 5)
        
        control_points = path_world[::subsample_step]
        if not np.array_equal(control_points[-1], path_world[-1]):
            control_points = np.vstack([control_points, path_world[-1]])

        # 6. Generate Smooth Spline
        smooth_pos, tangents = self._smooth_path(control_points, num_frames=num_frames)
        
        # 7. Generate Rotations
        quats = self._orientations_from_tangents(smooth_pos, tangents)

        poses = [
            CameraPose(position=pos, rotation=quat)
            for pos, quat in zip(smooth_pos, quats)
        ]
        return Trajectory(poses=poses, mode="explorer")

    # ------------------------------------------------------------------ #
    def _is_enclosed(self, idx: Tuple[int, int, int]) -> bool:
        # Cast rays in +X, -X, +Z, -Z. If at least 3 hit walls, we are "inside".
        # If rays hit grid bounds, we are "outside".
        hits = 0
        directions = [(1,0,0), (-1,0,0), (0,0,1), (0,0,-1)]
        gx, gy, gz = self.scene.grid_shape
        
        for dx, dy, dz in directions:
            cx, cy, cz = idx
            steps = 0
            while 0 <= cx < gx and 0 <= cy < gy and 0 <= cz < gz:
                if not self.scene.is_free((cx, cy, cz)):
                    hits += 1
                    break
                cx += dx
                cy += dy
                cz += dz
                steps += 1
                if steps > 50: # Limit ray length
                    break
        return hits >= 3

    def _march_to_enclosure(self, start: Tuple[int, int, int], target: Tuple[int, int, int]) -> Tuple[int, int, int]:
        # Simple line drawing algo to walk from start -> target
        # Stop when we find a free voxel that satisfies _is_enclosed
        curr = np.array(start)
        goal = np.array(target)
        vec = goal - curr
        dist = np.linalg.norm(vec)
        if dist < 1: return start
        
        step_vec = vec / dist
        
        for i in range(int(dist)):
            candidate = tuple(np.round(curr + step_vec * i).astype(int))
            if self.scene.is_free(candidate):
                if self._is_enclosed(candidate):
                    return candidate
        
        return start # Failed to find better, stick to original

    def _bfs_farthest(self, start: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], float]:
        visited = {start}
        q = deque([(start, 0)])
        farthest = start
        far_dist = 0.0
        
        grid_shape = self.scene.grid_shape
        
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
        
        gx, gy, gz = self.scene.grid_shape
        margin = 2 

        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for nbr in self._neighbors(current):
                nx, ny, nz = nbr
                
                if not self.scene.is_free(nbr):
                    continue
                
                # Check Boundary Margin
                if (nx < margin or nx >= gx - margin or
                    ny < margin or ny >= gy - margin or
                    nz < margin or nz >= gz - margin):
                    if nbr != goal: 
                        continue

                tentative_g = g_curr + self._euclidean(current, nbr)
                if tentative_g < g_score.get(nbr, float("inf")):
                    came_from[nbr] = current
                    g_score[nbr] = tentative_g
                    f_score = tentative_g + self._euclidean(nbr, goal)
                    heapq.heappush(open_heap, (f_score, tentative_g, nbr))

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

    def _smooth_path(
        self, points: np.ndarray, num_frames: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(points) < 4:
            t = np.linspace(0, 1, len(points))
            ts = np.linspace(0, 1, num_frames)
            interp = np.stack([np.interp(ts, t, points[:, dim]) for dim in range(3)], axis=-1)
            tangents = np.gradient(interp, axis=0)
            return interp, tangents

        distances = np.zeros(len(points))
        distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        total_dist = distances[-1]
        
        for i in range(1, len(distances)):
            if distances[i] <= distances[i - 1]:
                distances[i] = distances[i - 1] + 1e-4
                
        splines = [CubicSpline(distances, points[:, dim], bc_type="natural") for dim in range(3)]
        
        lin_t = np.linspace(0, 1, num_frames)
        smooth_t = 6*(lin_t**5) - 15*(lin_t**4) + 10*(lin_t**3)
        sample_dists = smooth_t * total_dist
        
        interp = np.stack([s(sample_dists) for s in splines], axis=-1)
        tangents = np.stack([s(sample_dists, 1) for s in splines], axis=-1)
        return interp, tangents

    def _orientations_from_tangents(
        self, positions: np.ndarray, tangents: np.ndarray
    ) -> np.ndarray:
        quats = []
        for i in range(len(positions)):
            tan = tangents[i]
            if np.linalg.norm(tan) < 1e-6:
                tan = np.array([0.0, 0.0, 1.0])
            
            forward = tan / (np.linalg.norm(tan) + 1e-9)
            up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(forward, up)) > 0.95:
                up = np.array([1.0, 0.0, 0.0])
            right = np.cross(up, forward)
            right /= (np.linalg.norm(right) + 1e-9)
            up_corrected = np.cross(forward, right)
            
            rot_mat = np.stack([right, up_corrected, -forward], axis=1)
            if np.linalg.det(rot_mat) < 0:
                rot_mat[:, 0] *= -1
            q = R.from_matrix(rot_mat).as_quat()
            quats.append(q)

        quats = np.array(quats)
        smoothed_quats = [quats[0]]
        alpha = 0.05 
        curr_rot = R.from_quat(quats[0])
        for i in range(1, len(quats)):
            target_rot = R.from_quat(quats[i])
            key_rots = R.from_quat(np.stack([curr_rot.as_quat(), target_rot.as_quat()]))
            slerp = Slerp([0, 1], key_rots)
            curr_rot = slerp([alpha])[0]
            smoothed_quats.append(curr_rot.as_quat())
            
        return np.array(smoothed_quats)

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