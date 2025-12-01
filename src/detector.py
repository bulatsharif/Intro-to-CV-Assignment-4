from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cuda"):
        """
        Initializes the YOLO detector.
        args:
            model_name: 'yolov8n.pt' (nano) is fastest, 'yolov8x.pt' is most accurate.
            device: 'cuda' or 'cpu'.
        """
        print(f"[Detector] Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        
        # Force move to device if available
        if device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
            print("[Detector] Model loaded on CUDA GPU.")
        else:
            print("[Detector] Warning: Running on CPU.")

    def process_frames(self, input_dir: Path, output_dir: Path, conf: float = 0.25):
        """
        Reads frames from input_dir, detects objects, draws boxes, and saves to output_dir.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get sorted list of frames
        frame_files = sorted(list(input_dir.glob("frame_*.png")))
        if not frame_files:
            print(f"[Detector] No frames found in {input_dir}")
            return

        print(f"[Detector] Processing {len(frame_files)} frames...")
        
        # Process loop
        # We use a loop instead of batch inference to ensure filenames match exactly 1:1
        for frame_path in tqdm(frame_files, desc="Detecting Objects"):
            # Ultralytics handles loading automatically
            # stream=True prevents accumulating gradients/memory
            results = self.model(frame_path, conf=conf, verbose=False, stream=True)
            
            for result in results:
                # Plot returns a BGR numpy array (ready for OpenCV)
                annotated_frame = result.plot()
                
                # Save to output directory with same filename
                save_path = output_dir / frame_path.name
                cv2.imwrite(str(save_path), annotated_frame)

        print(f"[Detector] Detection complete. Frames saved to {output_dir}")