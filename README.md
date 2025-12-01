# **Cinematic Navigation Agent for 3D Gaussian Splatting**

## **🎥 Overview**

This project implements an intelligent autonomous agent capable of exploring 3D Gaussian Splatting scenes. It analyzes the geometry of a point cloud, plans professional cinematic paths, avoids obstacles, and generates high-quality rendered videos with optional object detection.

**Key Features:**

* **Robust Navigation:** Uses Voxel-based A\* with Distance Fields to keep the camera inside buildings.  
* **Vertical Clamping:** Automatically detects floor and ceiling heights to prevent the agent from escaping into the sky/void.  
* **Cinematic Smoothing:** Cubic Spline interpolation with smoothstep (ease-in/ease-out) acceleration.  
* **Smart Heuristics:** Automatically switches between **Orbit Mode** (for small objects) and **Explorer Mode** (for large environments).  
* **Hybrid Computer Vision:** Combines 2D YOLOv8 detection with 3D Ray-Casting against the voxel map for 3D localization.

## **🛠️ Installation**

### **Prerequisites**

* Python 3.8+  
* CUDA-enabled GPU (Required for rendering)

### **Setup**

1. Clone the repository:  
   git clone \<repository\_url\>  
   cd \<repository\_name\>

2. Install dependencies:  
   pip install \-r requirements.txt

   *Note: For the renderer, gsplat requires a CUDA-compatible environment.*  
3. Install Object Detection weights (Automatic on first run):  
   pip install ultralytics

## **🚀 Usage Guide**

### **Basic Command (Generate Path Only)**

Generate a camera path without rendering (fast, runs on CPU):

python src/main.py \--ply input-data/scene.ply \--frames 1800

### **Full Render Pipeline**

Generate the path, render frames using CUDA, and assemble a video:

python src/main.py \\  
  \--ply input-data/ConferenceHall\_uncompressed.ply \\  
  \--render \--render\_backend gsplat \--device cuda \\  
  \--width 800 \--height 800 \\  
  \--frames 1800 \\  
  \--video \\  
  \--output\_dir outputs/demo\_run

### **With Object Detection (Bonus)**

Enable YOLO detection and 3D localization overlays:

python src/main.py \\  
  \--ply input-data/scene.ply \\  
  \--render \--render\_backend gsplat \\  
  \--detect \--video \\  
  \--frames 1800

### **Key Flags**

| Flag | Description | Default |
| :---- | :---- | :---- |
| \--ply | Path to the .ply Gaussian Splat file. | Required |
| \--voxel\_size | Resolution of the navigation grid (meters). | 0.5 |
| \--opacity\_threshold | Minimum opacity to consider a point "solid". | 0.2 |
| \--detect | Enable YOLO object detection \+ 3D localization. | False |
| \--video | Assemble rendered frames into MP4. | False |

## **🧠 Algorithm Descriptions**

### **1\. Scene Analysis (src/scene\_map.py)**

The system ingests the PLY file and converts the sparse cloud into a **Binary Voxel Grid**.

* **Noise Filtering:** Discards points with opacity \< 0.2.  
* **Voxelization:** Discretizes the world into 0.5m cubes.  
* **Robust Bounds:** Uses 1st-99th percentile to ignore "floater" artifacts common in scanned data.

### **2\. The "Vertical Clamp" Planner (src/path\_planner.py)**

Standard A\* algorithms often "leak" out of buildings through windows or open roofs. Our planner solves this with three distinct layers of logic:

* **Vertical Sandwich Start:** To ensure the agent starts inside the building, we scan the vertical column at the center of the map. We identify the largest "gap" between solid voxels (representing the space between floor and ceiling) and place the start point there.  
* **Vertical Clamping:** We calculate the 5th and 90th percentiles of the scene's height. Any voxel below the floor or above the ceiling is assigned infinite cost, forming a "lid" on the environment.  
* **Cost Map & Ceiling Check:**  
  * We compute a Euclidean Distance Transform (EDT) to create a "gravity" field that pulls the camera toward the center of rooms ($Cost \= 1 \+ \\frac{10}{Dist}$).  
  * During A\* traversal, we perform a **Ceiling Check**: If a node has no solid voxels above it (meaning it's outdoors/under the sky), we add a massive penalty, forcing the agent to stay under the roof.

### **3\. Rendering Engine (src/renderer.py)**

A custom wrapper around gsplat.

* **Fixes:** Correctly interprets log-scale sizes (exp) and logit opacities (sigmoid) found in standard PLY files.  
* **Sorting:** Properly sorts Spherical Harmonics coefficients to prevent color noise artifacts.

### **4\. 3D Object Detection (src/detector.py)**

* **2D Step:** YOLOv8 detects bounding boxes in the rendered frame.  
* **3D Step (Ray Casting):** A ray is cast from the camera center through the pixel coordinates. The ray marches through the **Voxel Grid** until it hits an Occupied voxel. This returns the precise 3D world coordinate of the object.

## **⚠️ Known Limitations**

1. **Glass Surfaces:** Transparent glass has low opacity and may be ignored by the voxelizer, potentially causing collisions if the navigation grid doesn't register it as a wall.  
2. **Disconnected Floors:** If a multi-story building connects floors via a staircase narrower than the voxel\_size (0.5m), the planner might view the floors as disconnected islands.  
3. **Open-Air Scenes:** The "Ceiling Check" logic assumes an indoor environment. For strictly outdoor scenes (like a park), the "Explorer" mode might struggle to find a "roof," though the cost map will still guide it through open spaces.