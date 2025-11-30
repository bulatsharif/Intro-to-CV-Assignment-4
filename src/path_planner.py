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

    # ------------------------------------------------------------------ #
    def plan(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.bounds is not None
        # Heuristic: If scene is small, orbit. If large, explore.
        if self.scene.bounds.max_dimension < 10.0:
            return self.plan_orbit(num_frames=num_frames)
        return self.plan_explorer(num_frames=num_frames)

    # ------------------------------------------------------------------ #
    def plan_orbit(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.bounds is not None
        assert self.scene.centroid is not None

        # Orbit slightly further out to ensure we see the object
        radius = max(self.scene.bounds.max_dimension * 0.9, self.scene.voxel_size * 5)
        base_height = self.scene.centroid[1] + self.scene.bounds.max_dimension * 0.1
        height_variation = self.scene.bounds.max_dimension * 0.05

        angles = np.linspace(0.0, 2 * math.pi, num_frames, endpoint=False)
        poses: List[CameraPose] = []
        for i, theta in enumerate(angles):
            # Add subtle bobbing for "drone-like" feel
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

        # 1. Start Position: Center of the map, but elevated
        center_world = self.scene.bounds.center
        start_world = center_world.copy()
        
        # Robust floor detection: Use 5th percentile of Y to find "true floor" ignoring noise
        if self.scene.filtered_points is not None:
             floor_y = np.percentile(self.scene.filtered_points[:, 1], 5)
             start_world[1] = floor_y + 1.8  # Eye level / Drone height
        
        start_idx = self.scene.world_to_grid(start_world)
        
        # Ensure we start in free space
        start_idx = self.scene.find_nearest_free(start_idx, max_radius=10)
        
        # 2. Strategy: Find the "Diameter" of the livable space
        # Find Point A (Deepest point from start)
        a_idx, _ = self._bfs_farthest(start_idx)
        # Find Point B (Opposite end from A)
        b_idx, _ = self._bfs_farthest(a_idx)

        # 3. Pathfinding
        path_grid = self._astar(a_idx, b_idx)
        
        if len(path_grid) < 2:
            print("[PathPlanner] Warning: Path too short. Hovering in place.")
            pos = self.scene.grid_to_world(start_idx)
            quat = self._look_forward(np.array([0, 0, 1], dtype=float))
            return Trajectory(poses=[CameraPose(position=pos, rotation=quat)]*num_frames, mode="explorer")

        # 4. Convert to World Coordinates
        path_world = np.stack([self.scene.grid_to_world(p) for p in path_grid], axis=0)

        # 5. FIX JERKINESS: Subsample the path
        # A* on a grid is zig-zaggy. We pick every Nth point to force the spline to smooth it out.
        # For a 0.5m grid, picking every 4th point means control points are ~2m apart.
        subsample_step = max(1, len(path_world) // 15) # Ensure we have at least 15 control points if possible
        subsample_step = min(subsample_step, 5) # But don't skip too much
        
        control_points = path_world[::subsample_step]
        # Always include the exact end point
        if not np.array_equal(control_points[-1], path_world[-1]):
            control_points = np.vstack([control_points, path_world[-1]])

        # 6. Generate Smooth Spline with Ease-In/Ease-Out
        smooth_pos, tangents = self._smooth_path(control_points, num_frames=num_frames)
        
        # 7. Generate Rotations (Smoothed)
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
        
        # Margin check for "Going Outside"
        # We prefer points that are NOT on the absolute edge of the grid
        grid_shape = self.scene.grid_shape
        assert grid_shape is not None
        margin = 2 

        while q:
            node, dist = q.popleft()
            if dist > far_dist:
                far_dist = dist
                farthest = node
            
            for nbr in self._neighbors(node):
                if nbr in visited:
                    continue
                # Strict bound check inside BFS to keep logic consistent
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
        
        # SAFETY MARGIN: Don't let A* go to the absolute edge of the voxel grid
        # This prevents clipping through outer walls into the void
        assert self.scene.grid_shape is not None
        gx, gy, gz = self.scene.grid_shape
        margin = 2 # 2 voxels = 1 meter buffer from the bounding box edge

        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for nbr in self._neighbors(current):
                nx, ny, nz = nbr
                
                # 1. Check if occupied
                if not self.scene.is_free(nbr):
                    continue
                
                # 2. Check Boundary Margin (Fix for "Going Outside")
                # If we are too close to the grid edge, treat as obstacle
                if (nx < margin or nx >= gx - margin or
                    ny < margin or ny >= gy - margin or
                    nz < margin or nz >= gz - margin):
                    # Exception: If the goal itself is near the edge, allow it
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

    # ------------------------------------------------------------------ #
    def _smooth_path(
        self, points: np.ndarray, num_frames: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        if len(points) < 4:
            # Fallback for tiny paths
            t = np.linspace(0, 1, len(points))
            ts = np.linspace(0, 1, num_frames)
            interp = np.stack([np.interp(ts, t, points[:, dim]) for dim in range(3)], axis=-1)
            tangents = np.gradient(interp, axis=0)
            return interp, tangents

        # Calculate arc-length parameterization
        distances = np.zeros(len(points))
        distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        total_dist = distances[-1]
        
        # Fix duplicates
        for i in range(1, len(distances)):
            if distances[i] <= distances[i - 1]:
                distances[i] = distances[i - 1] + 1e-4
                
        # Create Spline
        splines = [CubicSpline(distances, points[:, dim], bc_type="natural") for dim in range(3)]
        
        # FIX SPEED: Use Ease-In/Ease-Out time mapping
        # Instead of linear time (constant speed), we map [0..1] -> [0..TotalDist] non-linearly
        lin_t = np.linspace(0, 1, num_frames)
        
        # Smoothstep function: 3x^2 - 2x^3 (Classic S-curve)
        # Or even smoother: 6x^5 - 15x^4 + 10x^3
        smooth_t = 6*(lin_t**5) - 15*(lin_t**4) + 10*(lin_t**3)
        
        # Map back to distance
        sample_dists = smooth_t * total_dist
        
        interp = np.stack([s(sample_dists) for s in splines], axis=-1)
        tangents = np.stack([s(sample_dists, 1) for s in splines], axis=-1)
        
        return interp, tangents

    def _orientations_from_tangents(
        self, positions: np.ndarray, tangents: np.ndarray
    ) -> np.ndarray:
        quats = []
        
        # 1. Generate Raw Target Orientations
        for i in range(len(positions)):
            tan = tangents[i]
            if np.linalg.norm(tan) < 1e-6:
                tan = np.array([0.0, 0.0, 1.0])
            
            # Look forward along tangent
            forward = tan / (np.linalg.norm(tan) + 1e-9)
            up = np.array([0.0, 1.0, 0.0])
            
            # Handle gimbal lock (looking straight up/down)
            if abs(np.dot(forward, up)) > 0.95:
                up = np.array([1.0, 0.0, 0.0])
                
            right = np.cross(up, forward)
            right /= (np.linalg.norm(right) + 1e-9)
            
            up_corrected = np.cross(forward, right)
            
            rot_mat = np.stack([right, up_corrected, -forward], axis=1)
            
            # Fix flipping
            if np.linalg.det(rot_mat) < 0:
                rot_mat[:, 0] *= -1
                
            q = R.from_matrix(rot_mat).as_quat()
            quats.append(q)

        # 2. FIX JITTER: Smooth Rotations (SLERP)
        # We apply a rolling average window to the quaternions
        quats = np.array(quats)
        smoothed_quats = [quats[0]]
        
        # Simple exponential smoothing for rotations
        # q_new = Slerp(q_old, q_target, alpha)
        alpha = 0.1 # Very smooth, slow response to turns
        
        curr_rot = R.from_quat(quats[0])
        
        for i in range(1, len(quats)):
            target_rot = R.from_quat(quats[i])
            
            # Slerp between current smoothed state and the noisy target
            # Note: Scipy Slerp expects times, so we just interpolate 0..1
            key_rots = R.from_quat(np.stack([curr_rot.as_quat(), target_rot.as_quat()]))
            slerp = Slerp([0, 1], key_rots)
            
            curr_rot = slerp([alpha])[0] # Take a small step towards target
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