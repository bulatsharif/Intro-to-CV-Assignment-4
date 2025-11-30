from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from path_planner import PathPlanner
from renderer import Renderer, assemble_video, save_trajectory
from scene_map import SceneMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cinematic Navigation Agent for Gaussian Splatting scenes")
    parser.add_argument("--ply", type=str, default="input-data/ConferenceHall_uncompressed.ply", help="Path to .ply file")
    parser.add_argument("--voxel_size", type=float, default=0.5, help="Voxel size in meters")
    parser.add_argument("--opacity_threshold", type=float, default=0.5, help="Opacity threshold for filtering floaters")
    parser.add_argument("--frames", type=int, default=600, help="Number of frames to sample")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory (defaults to outputs/run-<timestamp>)")
    parser.add_argument("--render", action="store_true", help="Render frames with gsplat (requires GPU)")
    parser.add_argument("--render_backend", choices=["none", "gsplat"], default="none", help="Rendering backend")
    parser.add_argument("--width", type=int, default=800, help="Render width")
    parser.add_argument("--height", type=int, default=800, help="Render height")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for rendering")
    parser.add_argument("--max_points", type=int, default=None, help="Subsample gaussians to this many points to save GPU memory")
    parser.add_argument("--tile_size", type=int, default=16, help="gsplat tile_size (higher = faster, more memory)")
    parser.add_argument("--render_mode", type=str, default="RGB", help="gsplat render_mode (RGB, D, ED, RGB+D, RGB+ED)")
    parser.add_argument("--packed", dest="packed", action="store_true", help="Use packed rendering (lower memory, default)")
    parser.add_argument("--unpacked", dest="packed", action="store_false", help="Use unpacked rendering (faster, more memory)")
    parser.set_defaults(packed=True)
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

    scene = SceneMap(
        args.ply,
        voxel_size=args.voxel_size,
        opacity_threshold=args.opacity_threshold,
    )
    print(scene.describe())

    planner = PathPlanner(scene)
    trajectory = planner.plan(num_frames=args.frames)
    save_trajectory(trajectory, output_dir)

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
        frames_dir = renderer.render_trajectory(
            trajectory,
            max_points=args.max_points,
            tile_size=args.tile_size,
            render_mode=args.render_mode,
            packed=args.packed,
        )

    if args.video:
        frames_base = frames_dir if frames_dir is not None else output_dir / "frames"
        if frames_base.exists() and any(frames_base.glob("frame_*.png")):
            assemble_video(frames_base, output_dir / "trajectory.mp4", fps=args.fps)
        else:
            print("Skipping video assembly: no frames found. Enable --render or ensure frames exist.")

    print(f"Done. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
