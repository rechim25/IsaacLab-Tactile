# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the pick and place basket task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_in_basket(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
    xy_threshold: float = 0.08,
    height_threshold: float = 0.05,
    height_offset: float = 0.02,
) -> torch.Tensor:
    """Check if the cube is placed inside the basket.
    
    Args:
        env: The environment.
        robot_cfg: Configuration for the robot.
        cube_cfg: Configuration for the cube.
        basket_cfg: Configuration for the basket.
        xy_threshold: Maximum XY distance between cube and basket center.
        height_threshold: Maximum height difference from expected height.
        height_offset: Expected height of cube above basket base.
    
    Returns:
        Boolean tensor indicating success for each environment.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    basket: RigidObject = env.scene[basket_cfg.name]

    # Compute position difference
    pos_diff = cube.data.root_pos_w - basket.data.root_pos_w
    
    # Check XY alignment (cube should be centered in basket)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    
    # Check height (cube should be slightly above basket base)
    height_dist = pos_diff[:, 2]  # Positive if cube is above basket origin
    height_ok = torch.logical_and(
        height_dist > 0,  # Cube should be above basket base
        height_dist < (height_offset + height_threshold)
    )
    
    # Combined success: cube is in XY range and at correct height
    success = torch.logical_and(xy_dist < xy_threshold, height_ok)
    
    # Check gripper is open (cube is released)
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"
        
        gripper_open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
        gripper_threshold = env.cfg.gripper_threshold
        
        gripper_open_1 = torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]] - gripper_open_val) < gripper_threshold
        gripper_open_2 = torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]] - gripper_open_val) < gripper_threshold
        gripper_open = torch.logical_and(gripper_open_1, gripper_open_2)
        
        success = torch.logical_and(success, gripper_open)
    else:
        raise ValueError("No gripper_joint_names found in environment config")
    
    return success

