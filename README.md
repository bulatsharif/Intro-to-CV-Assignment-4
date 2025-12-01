# **Cinematic Navigation Agent for 3D Gaussian Splatting**

## **🎥 Overview**

This project implements an intelligent autonomous agent capable of exploring 3D Gaussian Splatting scenes. It ingests raw PLY point clouds, analyzes the geometry via voxelization, plans collision-free cinematic paths using a cost-aware A* algorithm, and generates rendered videos with semantic object detection.

**Key Features:**

* **Hybrid Navigation Modes:** Automatically detects scene scale to switch between **Orbit Mode** (for object-centric scenes) and **Explorer Mode** (for indoor environments).
* **Voxel-Based Safety:** Discretizes the scene into a grid to prevent wall clipping.
* **Cinematic Smoothing:** Uses Cubic Splines and SLERP (Spherical Linear Interpolation) for drone-like camera motion.
* **Vertical Containment:** Features "Vertical Clamping" and "Sandwich Start" algorithms to keep the agent inside buildings and prevent it from flying out of the roof.
* **Integrated Rendering:** A custom wrapper around gsplat for high-performance CUDA rendering.
* **Visual Analytics:** Integrated YOLOv8 for 2D semantic object labeling on the final video.

## **🛠️ Installation**

### **Prerequisites**

* Python 3.8+
* **CUDA-enabled GPU** (Required for the gsplat renderer)

### **Setup**

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Note: gsplat requires a working CUDA toolkit environment.*

## **🚀 Usage Guide**

### **1. Basic Path Generation (CPU Only)**

If you only want to calculate the camera trajectory without rendering pixels (useful for debugging the path):

```bash
python src/main.py --ply input-data/scene.ply --frames 600
```

*Output:* Creates camera_path.json and .npz files in outputs/.

### **2. Full Pipeline (Render + Video)**

To generate the path, render the frames using CUDA, and compile an MP4:

```bash
python src/main.py \
  --ply input-data/ConferenceHall.ply \
  --render --render_backend gsplat \
  --width 800 --height 800 \
  --video
```

### **3. Object Detection (Bonus)**

Apply YOLOv8 detection overlays to the rendered video:

```bash
python src/main.py \
  --ply input-data/ConferenceHall.ply \
  --render --render_backend gsplat \
  --detect --yolo_model yolov8n.pt \
  --video
```

### **Key Arguments**

| Flag | Description | Default |
| :---- | :---- | :---- |
| --ply | Path to the .ply file. | required |
| --voxel_size | Grid resolution in meters. | 0.5 |
| --frames | Total number of frames in the video. | 600 |
| --render | Enable rendering (requires GPU). | False |
| --detect | Run YOLO object detection on frames. | False |
| --packed | Use packed memory layout for faster rendering. | True |

## **🧠 Algorithm Descriptions**

### **Scene Analysis (src/scene_map.py)**

* **Ingestion:** Reads PLY files, handling both probability (0-1) and logit (raw) opacity scales via heuristic detection.
* **Voxelization:** Points are filtered by opacity (threshold 0.2) and mapped to a 3D binary grid.
* **Robust Bounds:** Uses 1st and 99th percentiles to determine scene boundaries, ignoring floating noise artifacts.

### **Path Planning (src/path_planner.py)**

* **Cost Map:** Computes a **Euclidean Distance Transform (EDT)**. Voxels near walls have high costs; open spaces have low costs. This creates a "gravity field" pulling the camera to the center of rooms.
* **Vertical Sandwich:** To find a safe start point, the planner scans the center Y-column of the grid to find the largest vertical gap between "floor" and "ceiling" voxels.
* **Smoothing:** Raw A* grid paths are smoothed using scipy.interpolate.CubicSpline for positions and spherical interpolation for rotations.

### **Rendering (src/renderer.py)**

* **Data Normalization:** Automatically detects if scale/opacity parameters need Exp/Sigmoid activations.
* **SH Sorting:** Ensures Spherical Harmonics coefficients are loaded in the correct order to prevent color artifacts.

## **⚠️ Known Limitations**

1. **2D-Only Detection:** The current object detector provides 2D bounding boxes on the video. It does not yet map these detections back to 3D world coordinates.
2. **Memory Usage:** Large scenes (>2M points) may require significant VRAM. Use --max_points to subsample if you run out of memory.
3. **Glass Surfaces:** Highly transparent surfaces (opacity < threshold) may be treated as empty space, potentially causing collision issues in glass-heavy scenes.