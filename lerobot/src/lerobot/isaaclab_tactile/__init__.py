# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
IsaacLab Tactile integration module for LeRobot.

This module provides adapters and utilities for training SmolVLA policies on
IsaacLab tactile data and evaluating them in IsaacLab environments.

Key conventions:
- State (policy input): 11D = [eef_pos_b(3), eef_rot6d_b(6), gripper_qpos(2)]
- Action (policy output): 7D = [Δpos_b(3), Δaxis_angle_b(3), gripper(1)]
- All pose data is expressed in robot base/root-link frame for frame-invariance.
"""

from .policy_io import (
    axis_angle_to_quat,
    decode_action_policy_to_isaaclab,
    encode_action_isaaclab_to_policy,
    encode_state_isaaclab_to_policy,
    quat_conjugate,
    quat_multiply,
    quat_to_axis_angle,
    quat_to_rot6d,
    quat_to_rotmat,
    rot6d_to_rotmat,
    world_to_base_pos,
    world_to_base_quat,
)

__all__ = [
    # Frame transforms
    "world_to_base_pos",
    "world_to_base_quat",
    # Quaternion utilities
    "quat_conjugate",
    "quat_multiply",
    "quat_to_rotmat",
    "quat_to_rot6d",
    "quat_to_axis_angle",
    "axis_angle_to_quat",
    # Rot6D utilities
    "rot6d_to_rotmat",
    # High-level encode/decode
    "encode_state_isaaclab_to_policy",
    "encode_action_isaaclab_to_policy",
    "decode_action_policy_to_isaaclab",
]
