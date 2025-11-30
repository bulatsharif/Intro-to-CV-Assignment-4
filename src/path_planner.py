from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Set

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
        # If "Indoor" voxels account for a visible amount of space, go Indoor.
        indoor_volume = np.sum(self.interior_mask)
        print(f"[Planner] Detected {indoor_volume} indoor voxels.")

        # Threshold: ~500 voxels (approx 2.5m x 2.5m x 2.5m room at 0.5m resolution)
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
        """
        Distinguishes between 'Inside' (Rooms) and 'Outside' (Sky/Void).
        """
        shape = self.scene.grid_shape
        
        # 1. Base Grid (1=Solid, 0=Air)
        # We assume SceneMap provides occupied voxels.
        grid = np.zeros(shape, dtype=bool)
        for (x, y, z) in self.scene.occupied_voxels:
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                grid[x, y, z] = 1
        
        self.obstacle_grid = grid

        # 2. Close Holes (Virtual Seal)
        # Closes gaps of ~1-2 voxels (windows/doors) to define "Inside".
        # This prevents the "Outside" flood fill from leaking in.
        print("[Planner] sealing mesh gaps...")
        structure = np.ones((3, 3, 3)) # 3x3x3 connectivity
        closed_grid = binary_closing(grid, structure=structure, iterations=1)

        # 3. Identify "Outside" via Flood Fill from corners
        # Invert: 1=Empty, 0=Solid
        empty_space = ~closed_grid
        
        # Label connected components of empty space
        labeled, n_components = label(empty_space)
        
        # Find which components touch the boundary of the bounding box
        boundary_labels = set()
        
        # Check all 6 faces
        for dim in range(3):
            boundary_labels.update(np.unique(np.take(labeled, 0, axis=dim)))
            boundary_labels.update(np.unique(np.take(labeled, -1, axis=dim)))
            
        # Create Exterior Mask (0 is usually background/solid in label output, check docs)
        # label 0 is the 'False' region (Solid), so we skip it.
        self.exterior_mask = np.zeros(shape, dtype=bool)
        for l in boundary_labels:
            if l == 0: continue 
            self.exterior_mask |= (labeled == l)
            
        # 4. Identify "Inside"
        # Inside = Empty Space AND NOT Exterior
        # We use the original grid for "Empty Space" to allow navigation near detailed walls,
        # but we use the closed-form calculation to define the region.
        # Ideally: Inside is any empty voxel that isn't connected to the sky.
        self.interior_mask = (~grid) & (~self.exterior_mask)

        # 5. Distance Field (Distance to nearest SOLID obstacle)
        # Used for centering the camera
        self.dist_field = distance_transform_edt(~grid)

    def plan_indoor_tour(self, num_frames: int) -> Trajectory:
        # 1. Find Start Point (Deepest point inside)
        # We mask the distance field to only consider Interior voxels
        masked_dist = self.dist_field.copy()
        masked_dist[~self.interior_mask] = 0
        
        start_idx = np.unravel_index(np.argmax(masked_dist), self.grid_shape) # type: ignore
        
        if masked_dist[start_idx] == 0:
            raise RuntimeError("No valid interior point found.")
            
        print(f"[Planner] Start Voxel: {start_idx} (Clearance: {masked_dist[start_idx]:.2f})")

        # 2. Find Keypoints (Tour)
        # Point A: Start
        # Point B: Furthest reachable point from A (inside)
        target_b, _ = self._bfs_bounded(start_idx)
        
        # Point C: Furthest reachable point from B
        target_c, _ = self._bfs_bounded(target_b)
        
        print(f"[Planner] Tour: Start->{target_b}->{target_c}->Start")
        
        # 3. Generate Path Segments
        # Start -> B
        path1 = self._astar_bounded(start_idx, target_b)
        # B -> C
        path2 = self._astar_bounded(target_b, target_c)
        # C -> Start
        path3 = self._astar_bounded(target_c, start_idx)
        
        # Combine
        full_path_grid = path1 + path2[1:] + path3[1:]
        
        # 4. Smooth & Convert
        path_world = np.stack([self.scene.grid_to_world(p) for p in full_path_grid], axis=0)
        
        # Subsample for spline
        subsample = max(1, len(path_world) // 15)
        control_points = path_world[::subsample]
        # Ensure loop closure
        if not np.allclose(control_points[0], control_points[-1]):
             control_points = np.vstack([control_points, control_points[0]])

        smooth_pos, tangents = self._smooth_path(control_points, num_frames=num_frames)
        quats = self._orientations_from_tangents(smooth_pos, tangents)
        
        poses = [CameraPose(position=pos, rotation=quat) for pos, quat in zip(smooth_pos, quats)]
        return Trajectory(poses=poses, mode="indoor")

    def _bfs_bounded(self, start: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], float]:
        """BFS that strictly stays within the Interior/Boundaries."""
        q = deque([(start, 0)])
        visited = {start}
        farthest = start
        max_dist = 0
        
        while q:
            curr, dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest = curr
            
            # Optimization: Don't expand too much if we just want a distant point
            # if dist > 200: continue 

            for dx, dy, dz in VOXEL_MOVES:
                nbr = (curr[0]+dx, curr[1]+dy, curr[2]+dz)
                if nbr in visited: continue
                
                # Bounds check
                if not (0 <= nbr[0] < self.grid_shape[0] and 
                        0 <= nbr[1] < self.grid_shape[1] and 
                        0 <= nbr[2] < self.grid_shape[2]):
                    continue
                
                # STRICT CONSTRAINT: Must be interior or close to it
                # We allow moving in 'interior_mask' 
                # We disallow moving in 'exterior_mask'
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
                
                # Cost Calculation
                # Base cost = 1
                cost = 1.0
                
                # Penalty for being close to walls (keep away from walls)
                dist = self.dist_field[nbr]
                if dist < 1.0: cost += 1000 # Virtually solid
                elif dist < 2.0: cost += 10 # Uncomfortable
                
                # Penalty for Exterior (Soft Barrier)
                # We allow it with high cost if it's the only way to connect rooms
                if self.exterior_mask[nbr]: 
                    cost += 500.0 
                
                # Penalty for Obstacles
                if self.obstacle_grid[nbr]:
                    cost += 9999.0
                
                new_g = g_score[current] + cost
                if new_g < g_score.get(nbr, float('inf')):
                    came_from[nbr] = current
                    g_score[nbr] = new_g
                    priority = new_g + self._heuristic(nbr, goal)
                    heapq.heappush(open_set, (priority, nbr))
                    
        return [start] # Failed

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
        
        # Calculate safe radius based on bounds
        radius = self.scene.bounds.max_dimension * 0.8
        
        # Height: Look from slightly above
        y_level = self.scene.centroid[1] + (self.scene.bounds.max_dimension * 0.2)
        center = self.scene.centroid
        
        poses = []
        for i in range(num_frames):
            theta = 2 * math.pi * (i / num_frames)
            x = center[0] + radius * math.cos(theta)
            z = center[2] + radius * math.sin(theta)
            pos = np.array([x, y_level, z])
            
            # Simple "Look At Center"
            poses.append(CameraPose(position=pos, rotation=self._look_at(pos, center)))
            
        return Trajectory(poses=poses, mode="orbit")

    # --- Geometry Helpers ---
    def _smooth_path(self, points: np.ndarray, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        # remove duplicates
        points = points[np.insert(np.diff(points, axis=0).astype(bool).any(1), 0, True)]
        
        if len(points) < 4:
            # Linear fallback
            t_curr = np.linspace(0, 1, len(points))
            t_new = np.linspace(0, 1, num_frames)
            res = np.zeros((num_frames, 3))
            tan = np.zeros((num_frames, 3))
            for i in range(3):
                res[:, i] = np.interp(t_new, t_curr, points[:, i])
            # Finite diff tangents
            tan[1:] = res[1:] - res[:-1]
            tan[0] = tan[1]
            return res, tan

        # Cubic Spline
        # Parameterize by distance
        dists = np.zeros(len(points))
        dists[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        dists /= dists[-1] # Normalize 0..1
        
        cs = CubicSpline(dists, points, bc_type='periodic')
        
        ts = np.linspace(0, 1, num_frames)
        smooth_pos = cs(ts)
        smooth_tan = cs(ts, 1) # First derivative
        
        return smooth_pos, smooth_tan

    def _orientations_from_tangents(self, positions: np.ndarray, tangents: np.ndarray) -> np.ndarray:
        quats = []
        for i in range(len(positions)):
            t = tangents[i]
            norm = np.linalg.norm(t)
            if norm < 1e-6: t = np.array([0, 0, 1])
            else: t = t / norm
            
            # Look forward
            # Up vector assumption: Y is up
            right = np.cross(np.array([0, 1, 0]), t)
            if np.linalg.norm(right) < 0.01: # looking straight up/down
                right = np.array([1, 0, 0])
            right /= np.linalg.norm(right)
            
            up = np.cross(t, right)
            
            # Construct rotation matrix [right, up, -forward] (OpenGL style view)
            # But here we need Camera-to-World. 
            # In computer graphics, camera looks down -Z. 
            # So Forward = -Z_cam => Z_cam = -Forward
            rot_mat = np.stack([right, up, -t], axis=1)
            
            r = R.from_matrix(rot_mat)
            quats.append(r.as_quat())
            
        # Smooth rotations
        slerp_quats = [quats[0]]
        for i in range(1, len(quats)):
            prev = R.from_quat(slerp_quats[-1])
            curr = R.from_quat(quats[i])
            
            # Simple exp smoothing or SLERP
            key_times = [0, 1]
            key_rots = R.from_quat([prev.as_quat(), curr.as_quat()])
            interpolator = Slerp(key_times, key_rots)
            slerp_quats.append(interpolator([0.1])[0].as_quat())
            
        return np.array(slerp_quats)

    def _look_at(self, position, target):
        forward = target - position
        return R.from_matrix(self._build_lookat_matrix(forward)).as_quat()

    def _build_lookat_matrix(self, forward):
        forward = forward / (np.linalg.norm(forward) + 1e-6)
        right = np.cross(np.array([0, 1, 0]), forward)
        if np.linalg.norm(right) < 0.01: right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        return np.stack([right, up, -forward], axis=1)