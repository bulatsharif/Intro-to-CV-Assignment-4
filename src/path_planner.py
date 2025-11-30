from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import distance_transform_edt  # <--- NEW: For Distance Field

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
        self.cost_map: Optional[np.ndarray] = None

    def plan(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.bounds is not None
        
        # 0. Build the Navigation Cost Map (The "Virtual Cage")
        self._build_cost_map()
        
        print(f"[Planner] Scene Dimension: {self.scene.bounds.max_dimension:.2f}m")
        if self.scene.bounds.max_dimension < 10.0:
            print("[Planner] Mode: ORBIT (Small Object)")
            return self.plan_orbit(num_frames=num_frames)
        print("[Planner] Mode: EXPLORER (Large Environment)")
        return self.plan_explorer(num_frames=num_frames)

    def _build_cost_map(self):
        """
        Creates a 3D grid where:
        - Value 0 = Obstacle
        - Value High = Near Obstacle or Near Boundary (High Cost)
        - Value Low = Safe central space (Low Cost)
        """
        assert self.scene.grid_shape is not None
        shape = self.scene.grid_shape
        
        # 1. Create Binary Grid (0=Free, 1=Occupied)
        # We invert logic for distance transform: 1=Feature (Obstacle), 0=Background
        grid = np.zeros(shape, dtype=bool)
        for (x, y, z) in self.scene.occupied_voxels:
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                grid[x, y, z] = 1
                
        # 2. Mark Boundaries as Obstacles (The "Cage")
        # This prevents the path from hugging the edge of the map
        pad = 2
        grid[0:pad, :, :] = 1; grid[-pad:, :, :] = 1
        grid[:, 0:pad, :] = 1; grid[:, -pad:, :] = 1
        grid[:, :, 0:pad] = 1; grid[:, :, -pad:] = 1
        
        # 3. Compute Euclidean Distance Transform
        # dist[x,y,z] = distance to nearest obstacle
        # We compute distance on the INVERTED grid (distance to solid)
        dist_field = distance_transform_edt(1 - grid.astype(int))
        
        # 4. Create Cost Map
        # Base cost = 1.0
        # If dist < 2.0 (1 meter), add penalty (don't hug walls)
        # If dist is HUGE (very open space), it's fine.
        # We actually want to maximize distance from walls, so we invert distance to get cost.
        
        # Sigmoid-like cost: High near walls, Low in center
        # Max dist in a room might be ~10-20 voxels.
        # Cost = 1 + (MaxDist / (Dist + epsilon))
        self.cost_map = 1.0 + (5.0 / (dist_field + 0.5))
        
        # Mark obstacles as infinity
        self.cost_map[grid == 1] = float('inf')
        
        print("[Planner] Cost Map Built. Virtual Cage Active.")

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
        
        # 1. Robust Start: Centroid + Floor check
        start_world = self.scene.centroid.copy()
        if self.scene.filtered_points is not None:
             dists = np.linalg.norm(self.scene.filtered_points[:, [0, 2]] - start_world[[0, 2]], axis=1)
             nearby_points = self.scene.filtered_points[dists < 5.0]
             if len(nearby_points) > 0:
                 floor_y = np.percentile(nearby_points[:, 1], 5)
                 start_world[1] = floor_y + 1.8 

        start_idx = self.scene.world_to_grid(start_world)
        print(f"[Planner] Searching for best start near {start_idx}...")
        start_idx = self._find_safest_start(start_idx, max_radius=20)
        
        print(f"[Planner] Final Start Voxel: {start_idx}")

        # 2. Find Key Points (A and B) using Cost-Aware Search
        print("[Planner] Finding Point A...")
        a_idx, _ = self._bfs_cost_aware(start_idx)
        print(f"[Planner] Point A: {a_idx}. Finding Point B...")
        b_idx, _ = self._bfs_cost_aware(a_idx)
        print(f"[Planner] Point B: {b_idx}.")

        # 3. Pathfinding (Cost Aware A*)
        print(f"[Planner] Running Safety-A* from {a_idx} to {b_idx}...")
        path_grid = self._astar_cost_aware(a_idx, b_idx)
        print(f"[Planner] Path Length: {len(path_grid)}")

        if len(path_grid) < 5:
            print("[Planner] Path too short. Fallback to Start->A.")
            path_grid = self._astar_cost_aware(start_idx, a_idx)
            if len(path_grid) < 5:
                 pos = self.scene.grid_to_world(start_idx)
                 quat = self._look_forward(np.array([0, 0, 1], dtype=float))
                 return Trajectory(poses=[CameraPose(position=pos, rotation=quat)]*num_frames, mode="explorer")

        # 4. World Conversion
        path_world = np.stack([self.scene.grid_to_world(p) for p in path_grid], axis=0)

        # 5. Subsampling
        subsample_step = max(1, len(path_world) // 20) 
        subsample_step = min(subsample_step, 5)
        control_points = path_world[::subsample_step]
        if not np.array_equal(control_points[-1], path_world[-1]):
            control_points = np.vstack([control_points, path_world[-1]])

        # 6. Smoothing
        smooth_pos, tangents = self._smooth_path(control_points, num_frames=num_frames)
        quats = self._orientations_from_tangents(smooth_pos, tangents)

        poses = [
            CameraPose(position=pos, rotation=quat)
            for pos, quat in zip(smooth_pos, quats)
        ]
        return Trajectory(poses=poses, mode="explorer")

    # ------------------------------------------------------------------ #
    def _find_safest_start(self, center_idx: Tuple[int, int, int], max_radius: int) -> Tuple[int, int, int]:
        """Finds a free voxel with the lowest cost (safest) in the vicinity."""
        best_node = center_idx
        min_cost = float('inf')
        
        # Spiral out
        for r in range(max_radius):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    for dz in range(-r, r+1):
                        nx, ny, nz = center_idx[0]+dx, center_idx[1]+dy, center_idx[2]+dz
                        if self._is_valid(nx, ny, nz):
                            c = self.cost_map[nx, ny, nz] # type: ignore
                            if c < min_cost:
                                min_cost = c
                                best_node = (nx, ny, nz)
            if min_cost < 10.0: # Found a reasonably safe spot
                return best_node
        return best_node

    def _is_valid(self, x, y, z):
        s = self.scene.grid_shape
        return 0 <= x < s[0] and 0 <= y < s[1] and 0 <= z < s[2]

    def _bfs_cost_aware(self, start: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], float]:
        """BFS that avoids high-cost areas to find 'functional' distance."""
        visited = {start}
        q = deque([(start, 0)])
        farthest = start
        max_dist = 0
        
        while q:
            node, dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest = node
            
            for nbr in self._neighbors(node):
                if nbr in visited: continue
                
                # Safety Check: Don't traverse into 'infinity' cost (walls)
                nx, ny, nz = nbr
                if self.cost_map[nx, ny, nz] > 1000: # type: ignore
                    continue
                    
                visited.add(nbr)
                q.append((nbr, dist + 1))
        return farthest, max_dist

    def _astar_cost_aware(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        open_heap: List[Tuple[float, float, Tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start))
        came_from: dict = {}
        g_score = {start: 0.0}

        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for nbr in self._neighbors(current):
                nx, ny, nz = nbr
                
                # Retrieve Safety Cost
                if not self._is_valid(nx, ny, nz): continue
                step_cost = self.cost_map[nx, ny, nz] # type: ignore
                
                if step_cost > 1000: # Virtual Wall / Real Wall
                    continue

                tentative_g = g_curr + step_cost # Weighted A*
                
                if tentative_g < g_score.get(nbr, float("inf")):
                    came_from[nbr] = current
                    g_score[nbr] = tentative_g
                    f = tentative_g + self._euclidean(nbr, goal)
                    heapq.heappush(open_heap, (f, tentative_g, nbr))

        return [start]

    def _neighbors(self, node: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        x, y, z = node
        neighbors: List[Tuple[int, int, int]] = []
        for dx, dy, dz in VOXEL_MOVES:
            nbr = (x + dx, y + dy, z + dz)
            neighbors.append(nbr)
        return neighbors

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _smooth_path(self, points: np.ndarray, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
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

    def _orientations_from_tangents(self, positions: np.ndarray, tangents: np.ndarray) -> np.ndarray:
        quats = []
        for i in range(len(positions)):
            tan = tangents[i]
            if np.linalg.norm(tan) < 1e-6: tan = np.array([0.0, 0.0, 1.0])
            
            forward = tan / (np.linalg.norm(tan) + 1e-9)
            up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(forward, up)) > 0.95: up = np.array([1.0, 0.0, 0.0])
            
            right = np.cross(up, forward)
            right /= (np.linalg.norm(right) + 1e-9)
            up_corrected = np.cross(forward, right)
            
            rot_mat = np.stack([right, up_corrected, -forward], axis=1)
            if np.linalg.det(rot_mat) < 0: rot_mat[:, 0] *= -1
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

    def _look_at(self, position, target):
        forward = target - position
        if np.linalg.norm(forward) < 1e-6: forward = np.array([0.0, 0.0, 1.0])
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, up)) > 0.95: up = np.array([1.0, 0.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right) + 1e-9
        up_corrected = np.cross(forward, right)
        rot = np.stack([right, up_corrected, -forward], axis=1)
        if np.linalg.det(rot) < 0: rot[:, 0] *= -1
        return R.from_matrix(rot).as_quat()

    def _look_forward(self, forward):
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, up)) > 0.95: up = np.array([1.0, 0.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right) + 1e-9
        up_corrected = np.cross(forward, right)
        rot = np.stack([right, up_corrected, -forward], axis=1)
        if np.linalg.det(rot) < 0: rot[:, 0] *= -1
        return R.from_matrix(rot).as_quat()

    @staticmethod
    def _euclidean(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)