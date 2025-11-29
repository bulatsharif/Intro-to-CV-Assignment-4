import numpy as np
from plyfile import PlyData
import sys
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def analyze_scene(ply_path, voxel_size=0.5, opacity_threshold=0.5):
    print(f"Loading {ply_path}...")
    
    if not os.path.exists(ply_path):
        print(f"Error: File not found at {ply_path}")
        return

    try:
        plydata = PlyData.read(ply_path)
    except Exception as e:
        print(f"Error reading PLY: {e}")
        return

    # Extract coordinates
    vertex_data = plydata['vertex']
    x = np.array(vertex_data['x'])
    y = np.array(vertex_data['y'])
    z = np.array(vertex_data['z'])
    
    points = np.stack((x, y, z), axis=-1)
    total_points = points.shape[0]
    
    # Extract and normalize Opacity
    # Gaussian Splatting PLY files usually store 'opacity'
    if 'opacity' in vertex_data:
        opacities = np.array(vertex_data['opacity'])
        
        # Heuristic: Check if opacities are logits (raw values) or probabilities [0,1]
        # Logits usually range from -inf to +inf. Probabilities are strictly 0-1.
        if opacities.min() < 0 or opacities.max() > 1:
            print("Note: Detected raw opacity logits. Applying sigmoid...")
            opacities = sigmoid(opacities)
    else:
        print("Warning: No 'opacity' field found. Assuming all points are visible.")
        opacities = np.ones(total_points)

    # Filter 'Solid' Points (The structure of the building)
    mask = opacities > opacity_threshold
    solid_points = points[mask]
    solid_count = solid_points.shape[0]
    
    print(f"Data Loaded. Filtering noise...")

    # Robust Bounding Box (Using Percentiles to ignore floaters)
    # We use 1% and 99% to cut off the outliers
    min_bound = np.percentile(solid_points, 1, axis=0)
    max_bound = np.percentile(solid_points, 99, axis=0)
    
    scene_size = max_bound - min_bound
    scene_center = (min_bound + max_bound) / 2

    # --- VOXELIZATION SIMULATION ---
    # We shift points so min_bound is at (0,0,0) for grid calculation
    shifted_points = solid_points - min_bound
    
    # Calculate grid indices
    grid_indices = (shifted_points / voxel_size).astype(int)
    
    # Identify unique occupied voxels
    # We use a set of tuples for O(1) lookups
    occupied_voxels = set(map(tuple, grid_indices))
    
    # Calculate Scene Center in Voxel Coordinates
    center_voxel = ((scene_center - min_bound) / voxel_size).astype(int)
    is_center_occupied = tuple(center_voxel) in occupied_voxels

    # --- REPORT ---
    print("\n" + "="*40)
    print("       SCENE ANALYSIS REPORT       ")
    print("="*40)
    print(f"File: {os.path.basename(ply_path)}")
    print(f"Total Gaussian Points: {total_points:,}")
    print(f"Solid Points (opacity > {opacity_threshold}): {solid_count:,} ({solid_count/total_points:.1%})")
    print("-" * 40)
    print(f"Robust Bounds (1%-99%):")
    print(f"  Min: [{min_bound[0]:.2f}, {min_bound[1]:.2f}, {min_bound[2]:.2f}]")
    print(f"  Max: [{max_bound[0]:.2f}, {max_bound[1]:.2f}, {max_bound[2]:.2f}]")
    print(f"  Dimensions: {scene_size[0]:.2f}m x {scene_size[1]:.2f}m x {scene_size[2]:.2f}m")
    print("-" * 40)
    print(f"Voxel Grid Analysis (Size = {voxel_size}m):")
    print(f"  Unique Occupied Voxels: {len(occupied_voxels):,}")
    
    # Estimated Volume
    grid_dim = ((max_bound - min_bound) / voxel_size).astype(int) + 1
    total_grid_volume = grid_dim[0] * grid_dim[1] * grid_dim[2]
    sparsity = len(occupied_voxels) / total_grid_volume
    
    print(f"  Grid Dimensions: {grid_dim[0]} x {grid_dim[1]} x {grid_dim[2]}")
    print(f"  Sparsity (Occupied/Total): {sparsity:.2%} (Lower is emptier)")
    print("-" * 40)
    print(f"HEURISTIC CHECK:")
    print(f"  Center Voxel: {tuple(center_voxel)}")
    if is_center_occupied:
        print("  Status: OCCUPIED -> Likely an OBJECT/OUTDOOR scene (Orbit Strategy)")
    else:
        print("  Status: EMPTY    -> Likely an INDOOR/ROOM scene (Flood Fill Strategy)")
    print("="*40 + "\n")

if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR PLY FILE
    # You can also pass the file as an argument: python analyze_scene.py path/to/file.ply
    
    ply_file_path = "input-data/ConferenceHall_uncompressed.ply" 
    
    if len(sys.argv) > 1:
        ply_file_path = sys.argv[1]
        
    analyze_scene(ply_file_path)