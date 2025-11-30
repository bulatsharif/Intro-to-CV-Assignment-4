from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import distance_transform_edt, binary_closing, label

from scene_map import SceneMap


# 26-connectivity for smoother pathfinding
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
        self.grid_shape = self.scene.grid_shape
        self.interior_mask: Optional[np.ndarray] = None
        self.exterior_mask: Optional[np.ndarray] = None
        self.obstacle_grid: Optional[np.ndarray] = None
        self.dist_field: Optional[np.ndarray] = None

    def plan(self, num_frames: int = 600) -> Trajectory:
        if self.scene.grid_shape is None:
            print("[Planner] Scene not voxelized. Using fallback Orbit.")
            return self.plan_orbit(num_frames)

        print("[Planner] analyzing scene geometry...")
        self._analyze_structure()
        
        # Heuristic: Do we have a substantial interior?
        indoor_volume = np.sum(self.interior_mask)
        print(f"[Planner] Detected {indoor_volume} indoor voxels.")

        if indoor_volume > 200: 
            print("[Planner] Enclosed building detected. Mode: INDOOR EXPLORER")
            try:
                return self.plan_indoor_tour(num_frames)
            except Exception as e:
                print(f"[Planner] Indoor planning failed ({e}). Falling back to Orbit.")
                return self.plan_orbit(num_frames)
        else:
            print("[Planner] Open/Outdoor scene detected. Mode: ORBIT")
            return self.plan_orbit(num_frames)

    def _analyze_structure(self):
        shape = self.scene.grid_shape
        
        # 1. Base Grid (1=Solid, 0=Air)
        grid = np.zeros(shape, dtype=bool)
        for (x, y, z) in self.scene.occupied_voxels:
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                grid[x, y, z] = 1
        
        self.obstacle_grid = grid

        # 2. Close Holes
        print("[Planner] sealing mesh gaps...")
        structure = np.ones((3, 3, 3)) 
        closed_grid = binary_closing(grid, structure=structure, iterations=1)

        # 3. Identify "Outside" via Flood Fill
        empty_space = ~closed_grid
        labeled, n_components = label(empty_space)
        boundary_labels = set()
        
        for dim in range(3):
            boundary_labels.update(np.unique(np.take(labeled, 0, axis=dim)))
            boundary_labels.update(np.unique(np.take(labeled, -1, axis=dim)))
            
        self.exterior_mask = np.zeros(shape, dtype=bool)
        for l in boundary_labels:
            if l == 0: continue 
            self.exterior_mask |= (labeled == l)
            
        # 4. Identify "Inside"
        self.interior_mask = (~grid) & (~self.exterior_mask)

        # 5. Distance Field
        self.dist_field = distance_transform_edt(~grid)

    def plan_indoor_tour(self, num_frames: int) -> Trajectory:
        masked_dist = self.dist_field.copy()
        masked_dist[~self.interior_mask] = 0
        
        start_idx = np.unravel_index(np.argmax(masked_dist), self.grid_shape) # type: ignore
        
        if masked_dist[start_idx] == 0:
            raise RuntimeError("No valid interior point found.")
            
        print(f"[Planner] Start Voxel: {start_idx} (Clearance: {masked_dist[start_idx]:.2f})")

        # Keypoints
        target_b, _ = self._bfs_bounded(start_idx)
        target_c, _ = self._bfs_bounded(target_b)
        
        print(f"[Planner] Tour: Start->{target_b}->{target_c}->Start")
        
        # Paths
        path1 = self._astar_bounded(start_idx, target_b)
        path2 = self._astar_bounded(target_b, target_c)
        path3 = self._astar_bounded(target_c, start_idx)
        
        full_path_grid = path1 + path2[1:] + path3[1:]
        
        # Smooth
        path_world = np.stack([self.scene.grid_to_world(p) for p in full_path_grid], axis=0)
        subsample = max(1, len(path_world) // 15)
        control_points = path_world[::subsample]
        if not np.allclose(control_points[0], control_points[-1]):
             control_points = np.vstack([control_points, control_points[0]])

        smooth_pos, tangents = self._smooth_path(control_points, num_frames=num_frames)
        quats = self._orientations_from_tangents(smooth_pos, tangents)
        
        poses = [CameraPose(position=pos, rotation=quat) for pos, quat in zip(smooth_pos, quats)]
        return Trajectory(poses=poses, mode="indoor")

    def _bfs_bounded(self, start: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], float]:
        q = deque([(start, 0)])
        visited = {start}
        farthest = start
        max_dist = 0
        
        while q:
            curr, dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest = curr
            
            for dx, dy, dz in VOXEL_MOVES:
                nbr = (curr[0]+dx, curr[1]+dy, curr[2]+dz)
                if nbr in visited: continue
                if not (0 <= nbr[0] < self.grid_shape[0] and 
                        0 <= nbr[1] < self.grid_shape[1] and 
                        0 <= nbr[2] < self.grid_shape[2]):
                    continue
                if self.exterior_mask[nbr]: continue
                if self.obstacle_grid[nbr]: continue
                
                visited.add(nbr)
                q.append((nbr, dist + 1))
        return farthest, max_dist

    def _astar_bounded(self, start, goal) -> List[Tuple[int, int, int]]:
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            for dx, dy, dz in VOXEL_MOVES:
                nbr = (current[0]+dx, current[1]+dy, current[2]+dz)
                if not (0 <= nbr[0] < self.grid_shape[0] and 
                        0 <= nbr[1] < self.grid_shape[1] and 
                        0 <= nbr[2] < self.grid_shape[2]):
                    continue
                
                cost = 1.0
                dist = self.dist_field[nbr]
                if dist < 1.0: cost += 1000 
                elif dist < 2.0: cost += 10 
                if self.exterior_mask[nbr]: cost += 500.0 
                if self.obstacle_grid[nbr]: cost += 9999.0
                
                new_g = g_score[current] + cost
                if new_g < g_score.get(nbr, float('inf')):
                    came_from[nbr] = current
                    g_score[nbr] = new_g
                    priority = new_g + self._heuristic(nbr, goal)
                    heapq.heappush(open_set, (priority, nbr))
        return [start] 

    def _heuristic(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def plan_orbit(self, num_frames=600) -> Trajectory:
        print("[Planner] Generating safe orbit trajectory...")
        assert self.scene.bounds is not None
        assert self.scene.centroid is not None
        
        radius = self.scene.bounds.max_dimension * 0.8
        y_level = self.scene.centroid[1] + (self.scene.bounds.max_dimension * 0.2)
        center = self.scene.centroid
        
        poses = []
        for i in range(num_frames):
            theta = 2 * math.pi * (i / num_frames)
            x = center[0] + radius * math.cos(theta)
            z = center[2] + radius * math.sin(theta)
            pos = np.array([x, y_level, z])
            poses.append(CameraPose(position=pos, rotation=self._look_at(pos, center)))
            
        return Trajectory(poses=poses, mode="orbit")

    def _smooth_path(self, points: np.ndarray, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        points = points[np.insert(np.diff(points, axis=0).astype(bool).any(1), 0, True)]
        if len(points) < 4:
            t_curr = np.linspace(0, 1, len(points))
            t_new = np.linspace(0, 1, num_frames)
            res = np.zeros((num_frames, 3))
            tan = np.zeros((num_frames, 3))
            for i in range(3):
                res[:, i] = np.interp(t_new, t_curr, points[:, i])
            tan[1:] = res[1:] - res[:-1]
            tan[0] = tan[1]
            return res, tan

        dists = np.zeros(len(points))
        dists[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        dists /= dists[-1]
        
        cs = CubicSpline(dists, points, bc_type='periodic')
        ts = np.linspace(0, 1, num_frames)
        return cs(ts), cs(ts, 1)

    def _orientations_from_tangents(self, positions: np.ndarray, tangents: np.ndarray) -> np.ndarray:
        quats = []
        for i in range(len(positions)):
            t = tangents[i]
            # Normalize tangent (Forward direction)
            if np.linalg.norm(t) < 1e-6:
                forward = np.array([0.0, 0.0, 1.0])
            else:
                forward = t / np.linalg.norm(t)
            
            # --- FIX: Construct Safe Rotation Matrix (Det=1) ---
            # Camera looks down -Z. So Camera Z-axis = -Forward
            z_axis = -forward
            
            # World Up (Approximate)
            world_up = np.array([0.0, 1.0, 0.0])
            
            # Camera X (Right) = cross(Z_axis, WorldUp)
            # (If Z is Back, and Up is Up, then Right is Z x Up?)
            # Let's test: Z=(0,0,-1) [Back], Up=(0,1,0). 
            # (-1)k x j = -(-i) = i = (1,0,0). Correct.
            x_axis = np.cross(z_axis, world_up)
            
            if np.linalg.norm(x_axis) < 1e-3:
                # Singularity (Looking straight up/down)
                x_axis = np.array([1.0, 0.0, 0.0])
            
            x_axis /= np.linalg.norm(x_axis)
            
            # Camera Y (Up) = cross(Z, X)
            # k x i = j. Correct.
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            
            # Matrix columns: [Right, Up, Back]
            rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
            
            r = R.from_matrix(rot_mat)
            quats.append(r.as_quat())
            
        # Smooth rotations
        slerp_quats = [quats[0]]
        for i in range(1, len(quats)):
            prev = R.from_quat(slerp_quats[-1])
            curr = R.from_quat(quats[i])
            key_times = [0, 1]
            key_rots = R.from_quat([prev.as_quat(), curr.as_quat()])
            interpolator = Slerp(key_times, key_rots)
            slerp_quats.append(interpolator([0.1])[0].as_quat())
            
        return np.array(slerp_quats)

    def _look_at(self, position, target):
        forward = target - position
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, 1.0])
        else:
            forward /= np.linalg.norm(forward)
            
        # Use same robust logic as above
        z_axis = -forward
        world_up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, world_up)
        if np.linalg.norm(x_axis) < 1e-3: x_axis = np.array([1.0, 0.0, 0.0])
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        
        rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
        return R.from_matrix(rot_mat).as_quat()