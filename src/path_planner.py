from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure, label

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
    position: np.ndarray
    rotation: np.ndarray


@dataclass
class Trajectory:
    poses: List[CameraPose]
    mode: str


class PathPlanner:
    def __init__(self, scene: SceneMap) -> None:
        self.scene = scene
        self.cost_map: Optional[np.ndarray] = None
        self.interior_mask: Optional[np.ndarray] = None # <--- NEW: strict bounds

    def plan(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.bounds is not None
        
        # Heuristic check
        if self.scene.bounds.max_dimension < 10.0:
            print("[Planner] Small scene detected. Using Orbit.")
            return self.plan_orbit(num_frames=num_frames)

        print("[Planner] Large scene detected. Using Sealed Explorer.")
        return self.plan_sealed_explorer(num_frames=num_frames)

    def plan_sealed_explorer(self, num_frames: int = 600) -> Trajectory:
        assert self.scene.grid_shape is not None
        
        # 1. Find Safe Start (Sandwich Method)
        center_x, center_z = self.scene.get_center_grid()[0], self.scene.get_center_grid()[2]
        start_idx = self._find_vertical_sandwich_start(center_x, center_z)
        print(f"[Planner] Sandwich Start: {start_idx}")

        # 2. Build The "Sealed" Map (Caulk the windows)
        print("[Planner] Sealing building leaks (Morphological Dilation)...")
        self._build_topology_mask(seed_point=start_idx)
        
        # 3. Find Key Points (Strictly Inside)
        print("[Planner] Finding Point A (Interior)...")
        a_idx, _ = self._bfs_interior(start_idx)
        print(f"[Planner] Point A: {a_idx}. Finding Point B...")
        b_idx, _ = self._bfs_interior(a_idx)
        print(f"[Planner] Point B: {b_idx}.")

        # 4. Pathfinding
        print(f"[Planner] Running Sealed-A*...")
        path_grid = self._astar_sealed(a_idx, b_idx)
        print(f"[Planner] Path Length: {len(path_grid)}")

        # Fallback if A* fails
        if len(path_grid) < 5:
            path_grid = self._astar_sealed(start_idx, a_idx)
            if len(path_grid) < 5:
                 # Last resort: hover
                 pos = self.scene.grid_to_world(start_idx)
                 quat = self._look_forward(np.array([0, 0, 1], dtype=float))
                 return Trajectory(poses=[CameraPose(position=pos, rotation=quat)]*num_frames, mode="explorer")

        # 5. Smooth & Convert
        path_world = np.stack([self.scene.grid_to_world(p) for p in path_grid], axis=0)
        
        subsample_step = max(1, len(path_world) // 15)
        control_points = path_world[::subsample_step]
        if not np.array_equal(control_points[-1], path_world[-1]):
            control_points = np.vstack([control_points, path_world[-1]])

        smooth_pos, tangents = self._smooth_path(control_points, num_frames=num_frames)
        quats = self._orientations_from_tangents(smooth_pos, tangents)
        
        poses = [CameraPose(position=pos, rotation=quat) for pos, quat in zip(smooth_pos, quats)]
        return Trajectory(poses=poses, mode="explorer")

    def _build_topology_mask(self, seed_point: Tuple[int, int, int]):
        """
        Creates a 'Strict Interior' mask by thickening walls and finding the 
        connected component belonging to the seed point.
        """
        shape = self.scene.grid_shape
        # 1. Base Obstacles
        obstacles = np.zeros(shape, dtype=bool)
        for (x, y, z) in self.scene.occupied_voxels:
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                obstacles[x, y, z] = 1
        
        # 2. Thicken Walls (Seal gaps < 1.5m)
        # 3 iterations of dilation = 1.5 meters of thickness added
        struct = generate_binary_structure(3, 1) # 6-connectivity
        thick_walls = binary_dilation(obstacles, structure=struct, iterations=3)
        
        # 3. Define Free Space (Inverse of thick walls)
        free_space = ~thick_walls
        
        # Ensure start point isn't accidentally walled off by the thickening
        # If it is, clear a small bubble around it
        sx, sy, sz = seed_point
        if not free_space[sx, sy, sz]:
            print("[Planner] Warning: Start point covered by thick walls. Carving bubble.")
            r = 2
            x_min, x_max = max(0, sx-r), min(shape[0], sx+r+1)
            y_min, y_max = max(0, sy-r), min(shape[1], sy+r+1)
            z_min, z_max = max(0, sz-r), min(shape[2], sz+r+1)
            free_space[x_min:x_max, y_min:y_max, z_min:z_max] = 1
            obstacles[x_min:x_max, y_min:y_max, z_min:z_max] = 0 # Also clear from cost map base
            
        # 4. Connected Components (Labeling)
        labeled_array, num_features = label(free_space, structure=struct)
        
        # 5. Select the component containing the seed
        seed_label = labeled_array[sx, sy, sz]
        if seed_label == 0:
            raise RuntimeError("Start point is in void after clearing? Should not happen.")
            
        self.interior_mask = (labeled_array == seed_label)
        vol_voxels = np.sum(self.interior_mask)
        print(f"[Planner] Interior Volume Defined: {vol_voxels} voxels.")

        # 6. Build Cost Map (Restricted to this mask)
        # Use original obstacles for distance field (precision), but cap with mask
        dist_field = distance_transform_edt(1 - obstacles.astype(int))
        self.cost_map = 1.0 + (8.0 / (dist_field + 0.5))
        
        # Apply the Mask: Infinite cost everywhere outside the bubble
        self.cost_map[~self.interior_mask] = float('inf')

    def _astar_sealed(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
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
                if not self._is_valid(nx, ny, nz): continue
                
                # Retrieve Cost (will be inf if outside mask)
                step_cost = self.cost_map[nx, ny, nz] # type: ignore
                if step_cost == float('inf'): continue

                tentative_g = g_curr + step_cost
                if tentative_g < g_score.get(nbr, float("inf")):
                    came_from[nbr] = current
                    g_score[nbr] = tentative_g
                    f = tentative_g + self._euclidean(nbr, goal)
                    heapq.heappush(open_heap, (f, tentative_g, nbr))
        return [start]

    def _bfs_interior(self, start: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], float]:
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
                nx, ny, nz = nbr
                if not self._is_valid(nx, ny, nz): continue
                
                # Strict Check: Must be in the Interior Mask
                if not self.interior_mask[nx, ny, nz]: continue # type: ignore
                
                visited.add(nbr)
                q.append((nbr, dist + 1))
        return farthest, max_dist

    # --- Boilerplate Helpers (Same as before) ---
    def plan_orbit(self, num_frames=600) -> Trajectory:
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

    def _find_vertical_sandwich_start(self, cx: int, cz: int) -> Tuple[int, int, int]:
        gy = self.scene.grid_shape[1]
        current_span_start = -1
        spans = [] 
        for y in range(gy):
            is_solid = (cx, y, cz) in self.scene.occupied_voxels
            if not is_solid:
                if current_span_start == -1: current_span_start = y
            else:
                if current_span_start != -1:
                    length = y - current_span_start
                    if length > 2 and length < (gy * 0.8): 
                        spans.append((current_span_start, length))
                    current_span_start = -1
        if not spans: return (cx, gy//2, cz)
        spans.sort(key=lambda x: x[1], reverse=True)
        best_span = spans[0]
        start_y = best_span[0] + best_span[1] // 2
        return (cx, start_y, cz)

    def _is_valid(self, x, y, z):
        s = self.scene.grid_shape
        return 0 <= x < s[0] and 0 <= y < s[1] and 0 <= z < s[2]

    def _neighbors(self, node):
        x, y, z = node
        neighbors = []
        for dx, dy, dz in VOXEL_MOVES:
            neighbors.append((x+dx, y+dy, z+dz))
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
            if distances[i] <= distances[i - 1]: distances[i] = distances[i - 1] + 1e-4
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