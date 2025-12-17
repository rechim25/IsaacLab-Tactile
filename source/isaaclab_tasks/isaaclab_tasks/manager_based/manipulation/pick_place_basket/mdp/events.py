# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for the pick and place basket task."""

from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the default pose for robots in all envs."""
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Add Gaussian noise to joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def _sample_pose(pose_range: dict[str, tuple[float, float]]) -> list[float]:
    """Sample a single pose from the given ranges."""
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    return [random.uniform(r[0], r[1]) for r in range_list]


def randomize_cube_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    scale_range: tuple[float, float] = (0.8, 1.2),
):
    """
    Randomize cube scale uniformly within scale_range.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        cube_cfg: Scene entity config for the cube.
        scale_range: (min_scale, max_scale) - uniform scale multiplier range.
                     Default (0.8, 1.2) gives ±20% size variation.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    import isaacsim.core.utils.prims as prim_utils
    
    cube = env.scene[cube_cfg.name]
    
    for cur_env in env_ids.tolist():
        # Sample random uniform scale
        scale = random.uniform(scale_range[0], scale_range[1])
        
        # Get the prim path for this environment's cube
        prim_path = cube.cfg.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{cur_env}")
        
        # Apply scale to the prim
        prim = prim_utils.get_prim_at_path(prim_path)
        if prim.IsValid():
            from pxr import UsdGeom
            xformable = UsdGeom.Xformable(prim)
            # Clear existing scale ops and set new uniform scale
            scale_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                    break
            if scale_op is None:
                scale_op = xformable.AddScaleOp()
            scale_op.Set((scale, scale, scale))


def randomize_cube_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
):
    """
    Randomize cube color to a vibrant color (avoiding white, grey, black).
    
    Picks from saturated colors with good visibility for manipulation tasks.
    Works with procedural CuboidCfg that uses PreviewSurfaceCfg material.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    from pxr import Gf
    import isaacsim.core.utils.prims as prim_utils
    
    # Vibrant colors (R, G, B) - avoiding white, grey, black
    VIBRANT_COLORS = [
        # Primary colors
        (0.95, 0.15, 0.15),  # Bright red
        (0.15, 0.75, 0.15),  # Bright green
        (0.15, 0.35, 0.95),  # Bright blue
        # Secondary colors
        (0.95, 0.85, 0.1),   # Yellow
        (0.95, 0.5, 0.1),    # Orange
        (0.75, 0.15, 0.85),  # Purple
        (0.1, 0.85, 0.85),   # Cyan
        (0.95, 0.4, 0.7),    # Pink
        (0.5, 0.95, 0.1),    # Lime
        # Warm tones
        (0.85, 0.25, 0.1),   # Vermilion
        (0.95, 0.35, 0.2),   # Coral
        (0.9, 0.6, 0.2),     # Amber
        (0.85, 0.45, 0.55),  # Rose
        (0.7, 0.2, 0.3),     # Crimson
        (0.95, 0.7, 0.4),    # Peach
        (0.8, 0.3, 0.1),     # Rust
        # Cool tones
        (0.2, 0.5, 0.85),    # Sky blue
        (0.1, 0.6, 0.7),     # Teal
        (0.3, 0.7, 0.6),     # Seafoam
        (0.4, 0.2, 0.7),     # Indigo
        (0.55, 0.35, 0.85),  # Violet
        (0.2, 0.8, 0.6),     # Mint
        (0.1, 0.4, 0.6),     # Ocean
        # Earth tones (saturated)
        (0.6, 0.4, 0.2),     # Bronze
        (0.7, 0.5, 0.3),     # Tan
        (0.5, 0.35, 0.2),    # Brown
        # Misc vibrant
        (0.85, 0.1, 0.55),   # Magenta
        (0.95, 0.2, 0.8),    # Hot pink
        (0.6, 0.9, 0.3),     # Chartreuse
        (0.3, 0.85, 0.4),    # Spring green
        (0.9, 0.9, 0.2),     # Lemon
        (0.7, 0.6, 0.9),     # Lavender
        (0.4, 0.7, 0.9),     # Light blue
        (0.9, 0.6, 0.7),     # Salmon
    ]
    
    for cur_env in env_ids.tolist():
        # Pick a random color
        color = random.choice(VIBRANT_COLORS)
        
        # Get the shader prim path for procedural CuboidCfg
        # CuboidCfg creates: /Cube/geometry/mesh and material at /Cube/geometry/material/Shader
        shader_path = f"/World/envs/env_{cur_env}/Cube/geometry/material/Shader"
        
        shader_prim = prim_utils.get_prim_at_path(shader_path)
        if shader_prim and shader_prim.IsValid():
            # PreviewSurfaceCfg uses inputs:diffuseColor
            attr = shader_prim.GetAttribute("inputs:diffuseColor")
            if attr and attr.IsValid():
                attr.Set(Gf.Vec3f(*color))


def randomize_cube_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
    cube_pose_range: dict[str, tuple[float, float]] = {},
    basket_pose_range: dict[str, tuple[float, float]] = {},
    min_separation: float = 0.15,
    max_sample_tries: int = 100,
):
    """
    Randomize cube position ensuring it doesn't overlap with basket.
    
    The cube is spawned in cube_pose_range, and we ensure it's at least
    min_separation away from the basket position.
    """
    if env_ids is None:
        return

    cube: RigidObject = env.scene[cube_cfg.name]
    basket: RigidObject = env.scene[basket_cfg.name]

    for cur_env in env_ids.tolist():
        # Get basket position for this environment
        basket_pos = basket.data.root_pos_w[cur_env, :3] - env.scene.env_origins[cur_env, :3]
        
        # Sample cube pose that is far enough from basket
        for _ in range(max_sample_tries):
            cube_pose = _sample_pose(cube_pose_range)
            cube_xy = torch.tensor(cube_pose[:2], device=env.device)
            basket_xy = basket_pos[:2]
            
            if torch.linalg.vector_norm(cube_xy - basket_xy) > min_separation:
                break
        
        # Write cube pose to simulation
        pose_tensor = torch.tensor([cube_pose], device=env.device)
        positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
        orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
        
        cube.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device)
        )
        cube.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device)
        )


def randomize_basket_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
    pose_range: dict[str, tuple[float, float]] = {},
):
    """Randomize basket position within the given pose range."""
    if env_ids is None:
        return

    basket: RigidObject = env.scene[basket_cfg.name]

    for cur_env in env_ids.tolist():
        basket_pose = _sample_pose(pose_range)
        
        # Write basket pose to simulation
        pose_tensor = torch.tensor([basket_pose], device=env.device)
        positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
        orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
        
        basket.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device)
        )
        basket.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device)
        )

