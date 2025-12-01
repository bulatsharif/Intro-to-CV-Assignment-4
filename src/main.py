from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from path_planner import PathPlanner
from renderer import Renderer, assemble_video, save_trajectory
from scene_map import SceneMap
from detector import ObjectDetector  # <--- NEW IMPORT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cinematic Navigation Agent for Gaussian Splatting scenes")
    # --- Input / Scene ---
    parser.add_argument("--ply", type=str, default="input-data/ConferenceHall_uncompressed.ply", help="Path to .ply file")
    parser.add_argument("--voxel_size", type=float, default=0.5, help="Voxel size in meters")
    parser.add_argument("--opacity_threshold", type=float, default=0.2, help="Opacity threshold for filtering floaters")
    
    # --- Planner ---
    parser.add_argument("--frames", type=int, default=600, help="Number of frames to sample")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory (defaults to outputs/run-<timestamp>)")
    
    # --- Renderer ---
    parser.add_argument("--render", action="store_true", help="Render frames with gsplat (requires GPU)")
    parser.add_argument("--render_backend", choices=["none", "gsplat"], default="none", help="Rendering backend")
    parser.add_argument("--width", type=int, default=800, help="Render width")
    parser.add_argument("--height", type=int, default=800, help="Render height")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for rendering")
    parser.add_argument("--max_points", type=int, default=None, help="Subsample gaussians to save GPU memory")
    parser.add_argument("--tile_size", type=int, default=16, help="gsplat tile_size")
    parser.add_argument("--render_mode", type=str, default="RGB", help="gsplat render_mode")
    parser.add_argument("--packed", dest="packed", action="store_true", help="Use packed rendering (default)")
    parser.set_defaults(packed=True)
    
    # --- Object Detection (NEW) ---
    parser.add_argument("--detect", action="store_true", help="Run YOLO object detection on rendered frames")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLO model version (n, s, m, l, x)")

    # --- Video ---
    parser.add_argument("--video", action="store_true", help="Assemble rendered frames into an mp4")
    parser.add_argument("--fps", type=int, default=30, help="FPS for video assembly")
    
    return parser.parse_args()


def ensure_output_dir(path_arg: str) -> Path:
    if path_arg:
        out = Path(path_arg)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path("outputs") / f"run-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    # 1. Analyze Scene
    scene = SceneMap(
        args.ply,
        voxel_size=args.voxel_size,
        opacity_threshold=args.opacity_threshold,
    )
    print(scene.describe())

    # 2. Plan Path
    planner = PathPlanner(scene)
    trajectory = planner.plan(num_frames=args.frames)
    save_trajectory(trajectory, output_dir)

    # 3. Render Frames
    frames_dir: Optional[Path] = None
    if args.render:
        renderer = Renderer(
            ply_path=args.ply,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            backend=args.render_backend,
            device=args.device,
        )
        # Note: This saves frames to output_dir/frames/
        frames_dir = renderer.render_trajectory(
            trajectory,
            max_points=args.max_points,
            tile_size=args.tile_size,
            render_mode=args.render_mode,
            packed=args.packed,
        )

    # 4. Object Detection (Integrated Step)
    detected_frames_dir: Optional[Path] = None
    frames_base = frames_dir if frames_dir is not None else output_dir / "frames"

    if args.detect:
        if frames_base.exists() and any(frames_base.glob("frame_*.png")):
            detector = ObjectDetector(model_name=args.yolo_model, device=args.device)
            detected_frames_dir = output_dir / "frames_detected"
            detector.process_frames(frames_base, detected_frames_dir)
        else:
            print("[Main] Warning: Cannot run detection. No frames found.")

    # 5. Assemble Videos
    if args.video:
        # Video 1: Clean Render
        if frames_base.exists():
            clean_video_path = output_dir / "trajectory_clean.mp4"
            print(f"[Main] Assembling clean video: {clean_video_path}")
            assemble_video(frames_base, clean_video_path, fps=args.fps)
        
        # Video 2: Detected Render
        if detected_frames_dir and detected_frames_dir.exists():
            detected_video_path = output_dir / "trajectory_detected.mp4"
            print(f"[Main] Assembling detected video: {detected_video_path}")
            assemble_video(detected_frames_dir, detected_video_path, fps=args.fps)

    print(f"Done. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()