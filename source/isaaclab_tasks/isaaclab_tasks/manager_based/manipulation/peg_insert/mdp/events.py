# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set robot to a default joint pose.
    
    Args:
        env: The environment.
        env_ids: The environment IDs to reset.
        default_pose: The default joint positions (including gripper).
        asset_cfg: The robot asset configuration.
    """
    # Extract robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Set joint positions
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, :] = torch.tensor(default_pose, device=env.device)
    
    # Set joint velocities to zero
    joint_vel = torch.zeros_like(joint_pos)
    
    # Write to simulation
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_object_to_default_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_position: list[float],
    default_orientation: list[float],
    asset_cfg: SceneEntityCfg,
):
    """Reset an object to a default pose.
    
    Args:
        env: The environment.
        env_ids: The environment IDs to reset.
        default_position: Default [x, y, z] position.
        default_orientation: Default [w, x, y, z] quaternion orientation.
        asset_cfg: The object asset configuration.
    """
    # Extract object asset (could be Articulation or RigidObject)
    obj: Articulation = env.scene[asset_cfg.name]
    
    # Set pose
    positions = torch.tensor(default_position, device=env.device).unsqueeze(0).repeat(len(env_ids), 1)
    positions += env.scene.env_origins[env_ids, 0:3]
    orientations = torch.tensor(default_orientation, device=env.device).unsqueeze(0).repeat(len(env_ids), 1)
    
    # Write to simulation
    obj.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


# -----------------------------------------------------------------------------
# Randomization utilities for reset-time diversity
# -----------------------------------------------------------------------------

def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfgs: list[SceneEntityCfg],
):
    """Randomize the root pose of one or more objects uniformly within ranges.

    Args:
        env: Environment.
        env_ids: Environments to reset.
        pose_range: Dict with keys among ["x","y","z","roll","pitch","yaw"]. Values are (min,max).
        asset_cfgs: List of assets to randomize.
    """
    for asset_cfg in asset_cfgs:
        asset: Articulation = env.scene[asset_cfg.name]
        # Sample pose deltas
        def rng(key: str) -> tuple[float, float]:
            return pose_range.get(key, (0.0, 0.0))

        ranges = torch.tensor(
            [rng("x"), rng("y"), rng("z"), rng("roll"), rng("pitch"), rng("yaw")],
            device=env.device,
            dtype=torch.float32,
        )
        deltas = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

        # Base on default root state
        default_root = asset.data.default_root_state[env_ids].clone()
        pos = default_root[:, 0:3] + env.scene.env_origins[env_ids] + deltas[:, 0:3]
        orn_delta = math_utils.quat_from_euler_xyz(deltas[:, 3], deltas[:, 4], deltas[:, 5])
        orn = math_utils.quat_mul(default_root[:, 3:7], orn_delta)

        asset.write_root_pose_to_sim(torch.cat([pos, orn], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Add Gaussian noise to the robot joint positions at reset (keeps gripper).

    The noisy joint positions are clamped to the soft limits before being written to sim.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    noise = math_utils.sample_gaussian(mean, std, joint_pos.shape, device=env.device)
    joint_pos += noise
    # keep last two (gripper) exactly at default if present
    if joint_pos.shape[1] >= 2:
        joint_pos[:, -2:] = robot.data.default_joint_pos[env_ids, -2:]

    limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def set_pose_from_discrete_set(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    poses_xyz_yaw: list[tuple[float, float, float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Pick one pose from a discrete set for each env and apply to asset.

    Each pose is (x, y, z, yaw). Orientation is yaw-only around Z.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_options = len(poses_xyz_yaw)
    choices = torch.randint(low=0, high=num_options, size=(len(env_ids),), device=env.device)
    poses = torch.tensor(poses_xyz_yaw, device=env.device)[choices]

    pos = poses[:, 0:3] + env.scene.env_origins[env_ids]
    orn = math_utils.quat_from_euler_xyz(torch.zeros_like(poses[:, 3]), torch.zeros_like(poses[:, 3]), poses[:, 3])

    asset.write_root_pose_to_sim(torch.cat([pos, orn], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
