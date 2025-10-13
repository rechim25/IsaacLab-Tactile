# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

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

