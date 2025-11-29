import logging
import math
import os
from dataclasses import dataclass
from typing import List

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from .detector import detect_objects, save_detections
from .explorer import SceneExplorer, load_scene
from .path_planner import CameraPose, GridPathPlanner
from .renderer import GaussianRenderer, RenderConfig, write_video


@dataclass
class PipelineArtifacts:
    poses: List[CameraPose]
    frames: List


def build_pipeline(cfg: DictConfig) -> PipelineArtifacts:
    logging.info("Using configuration:\\n%s", OmegaConf.to_yaml(cfg))
    scene_path = to_absolute_path(cfg.scene_path)
    scene = load_scene(scene_path)

    cam_height = scene.floor_height + cfg.exploration.camera_height
    explorer = SceneExplorer(
        scene=scene,
        grid_resolution=cfg.exploration.grid_resolution,
        clearance_m=cfg.exploration.clearance_m,
        room_height=cfg.exploration.room_height,
        seed=cfg.exploration.seed,
    )
    waypoints = explorer.choose_waypoints(cfg.exploration.num_waypoints)
    planner = GridPathPlanner(
        explorer=explorer,
        camera_height=cam_height,
        max_step_m=cfg.exploration.max_step_m,
        look_ahead=cfg.exploration.look_ahead,
        smoothing_samples=cfg.exploration.smoothing_samples,
    )
    coarse_positions = planner.plan_through_waypoints(waypoints)
    smooth_positions = planner.smooth_path(coarse_positions)
    poses = planner.build_camera_poses(smooth_positions, scene.center)

    if cfg.render.enable_360:
        orbit_center = scene.center.copy()
        orbit_center[1] = cam_height
        orbit = planner.build_orbit(
            orbit_center, radius=cfg.render.orbit_radius, frames=cfg.render.orbit_frames
        )
        poses.extend(orbit)

    render_cfg = RenderConfig(
        width=cfg.render.width,
        height=cfg.render.height,
        fov_y=math.radians(cfg.render.fov_y_deg),
        renderer=cfg.render.renderer,
        gsplat_device=cfg.render.gsplat_device,
        preview=cfg.render.preview,
        preview_stride=cfg.render.preview_stride,
    )
    renderer = GaussianRenderer(
        positions=scene.points,
        colors=scene.colors,
        scales=scene.scales,
        cfg=render_cfg,
    )
    frames = renderer.render_sequence(poses, preview=cfg.render.preview)
    return PipelineArtifacts(poses=poses, frames=frames)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    artifacts = build_pipeline(cfg)
    out_dir = to_absolute_path(cfg.render.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, cfg.render.output_name)
    write_video(artifacts.frames, out_path=out_path, fps=cfg.exploration.fps)

    if cfg.detection.enable:
        detections = detect_objects(artifacts.frames)
        if cfg.detection.save_json:
            save_detections(detections, to_absolute_path(cfg.detection.output_json))


if __name__ == "__main__":
    run()
