import torch
import numpy as np
from plyfile import PlyData
import taichi as ti
import imageio
import math

# Initialize Taichi (Metal backend for Mac)
ti.init(arch=ti.metal)

@ti.kernel
def render_splat_kernel(
    points: ti.types.ndarray(),      # (N, 3) XYZ
    colors: ti.types.ndarray(),      # (N, 3) RGB
    scales: ti.types.ndarray(),      # (N)   Scalar Scale
    indices: ti.types.ndarray(),     # (N)   Sorted indices
    image: ti.types.ndarray(),       # (H, W, 3) Output
    w: ti.i32,
    h: ti.i32,
    view_matrix: ti.types.ndarray(), # (4, 4)
    fx: ti.f32,
    fy: ti.f32,
    cx: ti.f32,
    cy: ti.f32
):
    # Render Back-to-Front (Painter's Algorithm)
    for i in range(indices.shape[0]):
        idx = indices[i]
        
        # 1. Load Data
        p_x = points[idx, 0]
        p_y = points[idx, 1]
        p_z = points[idx, 2]
        scale_world = scales[idx]

        # 2. Transform to Camera Space
        cam_x = view_matrix[0, 0] * p_x + view_matrix[0, 1] * p_y + view_matrix[0, 2] * p_z + view_matrix[0, 3]
        cam_y = view_matrix[1, 0] * p_x + view_matrix[1, 1] * p_y + view_matrix[1, 2] * p_z + view_matrix[1, 3]
        cam_z = view_matrix[2, 0] * p_x + view_matrix[2, 1] * p_y + view_matrix[2, 2] * p_z + view_matrix[2, 3]

        # 3. Project to Screen
        if cam_z > 0.2: # Near clipping
            # Perspective projection
            u_center = (cam_x / cam_z) * fx + cx
            v_center = (cam_y / cam_z) * fy + cy
            
            # 4. Calculate Splat Radius
            # Radius shrinks as Z increases (perspective).
            # Multiplier 1.5 ensures slight overlap to fill gaps (Solidify).
            radius_px = (scale_world * fx / cam_z) * 1.5
            
            # Clamp radius to keep performance high (don't draw massive blobs)
            if radius_px < 1.0: radius_px = 1.0
            if radius_px > 20.0: radius_px = 20.0
            
            r_int = int(radius_px)
            
            # 5. Draw Circle (Splat)
            # Loop over a bounding box around the center
            u_min = int(u_center - r_int)
            u_max = int(u_center + r_int + 1)
            v_min = int(v_center - r_int)
            v_max = int(v_center + r_int + 1)

            # Clip to screen bounds
            if u_min < 0: u_min = 0
            if u_max > w: u_max = w
            if v_min < 0: v_min = 0
            if v_max > h: v_max = h

            # Pixel Loop
            for v in range(v_min, v_max):
                for u in range(u_min, u_max):
                    dx = u - u_center
                    dy = v - v_center
                    dist_sq = dx*dx + dy*dy
                    
                    # If inside circle, draw
                    if dist_sq <= radius_px * radius_px:
                        image[v, u, 0] = colors[idx, 0]
                        image[v, u, 1] = colors[idx, 1]
                        image[v, u, 2] = colors[idx, 2]

def load_ply_with_scales(path):
    print(f"Loading {path}...")
    plydata = PlyData.read(path)
    v = plydata['vertex']
    
    # Positions
    pos = np.stack((v['x'], v['y'], v['z']), axis=-1)
    
    # Colors (SH DC)
    dc_colors = np.stack((v['f_dc_0'], v['f_dc_1'], v['f_dc_2']), axis=-1)
    rgb = 0.5 + (dc_colors * 0.28209)
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # Scales (New!)
    # 3DGS stores log(scale). We need exp(scale).
    # We average the 3 scales (x,y,z) to get one "size" for our circle.
    s0 = np.exp(v['scale_0'])
    s1 = np.exp(v['scale_1'])
    s2 = np.exp(v['scale_2'])
    # Geometric mean or arithmetic mean is fine for approximation
    avg_scale = (s0 + s1 + s2) / 3.0
    
    return pos.astype(np.float32), rgb.astype(np.float32), avg_scale.astype(np.float32)

def get_view_matrix(eye, center, up):
    z = eye - center
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.eye(4)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = eye
    return np.linalg.inv(mat).astype(np.float32)

def main():
    # CONFIG
    ply_path = "input-data/Theater_uncompressed.ply" 
    width, height = 960, 540
    
    # Camera Setup (Try to frame the room)
    cam_pos = np.array([-1.6, -0.5, -6.0], dtype=np.float32) # Moved closer
    cam_target = np.array([0.0, 0.0, 10.0], dtype=np.float32) # Look deep into room
    cam_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    fov_y = math.radians(70)

    # LOAD
    xyz, rgb, scales = load_ply_with_scales(ply_path)
    num_points = xyz.shape[0]
    print(f"Loaded {num_points} splats.")

    # SORT
    print("Sorting...")
    view_mat_np = get_view_matrix(cam_pos, cam_target, cam_up)
    R_row3 = view_mat_np[2, :3]
    t_z = view_mat_np[2, 3]
    depths = xyz @ R_row3 + t_z
    indices = np.argsort(depths)[::-1].astype(np.int32) # Far -> Near

    # RENDER
    print("Rendering Splats...")
    image_buffer = np.zeros((height, width, 3), dtype=np.float32)
    fx = (width / 2) / math.tan(fov_y / 2)
    fy = (height / 2) / math.tan(fov_y / 2)
    cx, cy = width / 2, height / 2
    
    render_splat_kernel(
        xyz, rgb, scales, indices, image_buffer, 
        width, height, view_mat_np, fx, fy, cx, cy
    )

    # SAVE
    imageio.imwrite("output_splat_render.png", (image_buffer * 255).astype(np.uint8))
    print("Saved output_splat_render.png")

if __name__ == "__main__":
    main()