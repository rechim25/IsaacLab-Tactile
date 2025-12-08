# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector position in the environment frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector orientation in the world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]
    return ee_frame_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gripper joint positions."""
    robot: Articulation = env.scene[robot_cfg.name]
    # Get gripper joint indices (last 2 joints for Franka)
    gripper_joint_indices = [-2, -1]
    return robot.data.joint_pos[:, gripper_joint_indices]


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Object position in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: Articulation = env.scene[object_cfg.name]
    
    # Get object position in world frame (relative to env origin)
    obj_pos_w = obj.data.root_pos_w - env.scene.env_origins
    
    # Get robot position in world frame (relative to env origin)
    robot_pos_w = robot.data.root_pos_w - env.scene.env_origins
    
    # Transform object position to robot's root frame
    robot_quat_w = robot.data.root_quat_w
    obj_pos_rel = obj_pos_w - robot_pos_w
    obj_pos_b = math_utils.quat_apply_inverse(robot_quat_w, obj_pos_rel)
    
    return obj_pos_b


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Object orientation in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: Articulation = env.scene[object_cfg.name]
    
    # Get orientations in world frame
    obj_quat_w = obj.data.root_quat_w
    robot_quat_w = robot.data.root_quat_w
    
    # Transform object orientation to robot's root frame
    obj_quat_b = math_utils.quat_mul(math_utils.quat_conjugate(robot_quat_w), obj_quat_w)
    
    return obj_quat_b


def peg_to_hole_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the peg relative to the hole in the robot's root frame.
    
    This provides the relative position vector from the peg to the hole, which is
    useful for the policy to understand the insertion direction and distance.
    """
    peg: Articulation = env.scene["peg"]
    hole: Articulation = env.scene["hole"]
    
    # Get positions in world frame
    peg_pos_w = peg.data.root_pos_w - env.scene.env_origins
    hole_pos_w = hole.data.root_pos_w - env.scene.env_origins
    
    # Compute relative position
    peg_to_hole = hole_pos_w - peg_pos_w
    
    return peg_to_hole


def peg_to_hole_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The Euclidean distance from the peg to the hole."""
    peg_to_hole = peg_to_hole_position(env)
    return torch.norm(peg_to_hole, dim=-1, keepdim=True)


def peg_alignment_to_hole(env: ManagerBasedRLEnv) -> torch.Tensor:
    """How well the peg is aligned with the hole orientation.
    
    Returns the dot product between the peg's up vector and the hole's up vector.
    A value close to 1.0 means good alignment.
    """
    peg: Articulation = env.scene["peg"]
    hole: Articulation = env.scene["hole"]
    
    # Get orientations
    peg_quat = peg.data.root_quat_w
    hole_quat = hole.data.root_quat_w
    
    # Get up vectors (z-axis) in world frame
    peg_up = math_utils.quat_rotate(peg_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device))
    hole_up = math_utils.quat_rotate(hole_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device))
    
    # Compute alignment (dot product)
    alignment = torch.sum(peg_up * hole_up, dim=-1, keepdim=True)
    
    return alignment

