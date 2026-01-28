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
IsaacLab Tactile Policy I/O Adapter.

This module provides pure functions for converting between IsaacLab's native
observation/action formats and the canonical policy format used by SmolVLA.

Conventions:
-----------
State (policy input): 11D vector
    [eef_pos_b(3), eef_rot6d_b(6), gripper_qpos(2)]
    - eef_pos_b: EE position in robot base/root-link frame
    - eef_rot6d_b: EE orientation in base frame as Rot6D (first 2 cols of rotation matrix)
    - gripper_qpos: 2D gripper joint positions

Action (policy output): 7D vector
    [Δpos_b(3), Δaxis_angle_b(3), gripper(1)]
    - Δpos_b: EE translation delta in robot base frame
    - Δaxis_angle_b: EE rotation delta in robot base frame (axis-angle = axis * angle)
    - gripper: scalar gripper command (-1 to 1)

Quaternion convention: (x, y, z, w) following IsaacLab/robosuite convention.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Quaternion Utilities
# =============================================================================


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute the conjugate (inverse for unit quaternions) of quaternion(s).

    Args:
        q: Quaternion(s) in (x, y, z, w) format, shape (..., 4)

    Returns:
        Conjugate quaternion(s), shape (..., 4)
    """
    q = np.asarray(q)
    result = q.copy()
    result[..., :3] = -result[..., :3]
    return result


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions: q_result = q1 * q2 (Hamilton product).

    Args:
        q1: First quaternion(s) in (x, y, z, w) format, shape (..., 4)
        q2: Second quaternion(s) in (x, y, z, w) format, shape (..., 4)

    Returns:
        Product quaternion(s), shape (..., 4)
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([x, y, z, w], axis=-1)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion(s) to 3x3 rotation matrix/matrices.

    Args:
        q: Quaternion(s) in (x, y, z, w) format, shape (..., 4)

    Returns:
        Rotation matrix/matrices, shape (..., 3, 3)
    """
    q = np.asarray(q)
    # Normalize
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-10)

    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Pre-compute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Build rotation matrix
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)

    # Stack into matrix
    row0 = np.stack([r00, r01, r02], axis=-1)
    row1 = np.stack([r10, r11, r12], axis=-1)
    row2 = np.stack([r20, r21, r22], axis=-1)

    return np.stack([row0, row1, row2], axis=-2)


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion(s) to 6D rotation representation.

    The 6D representation consists of the first two columns of the rotation matrix,
    flattened: [col0(3), col1(3)] = 6D.

    Args:
        q: Quaternion(s) in (x, y, z, w) format, shape (..., 4)

    Returns:
        6D rotation representation, shape (..., 6)
    """
    rotmat = quat_to_rotmat(q)  # (..., 3, 3)
    # Extract first two columns and flatten
    col0 = rotmat[..., :, 0]  # (..., 3)
    col1 = rotmat[..., :, 1]  # (..., 3)
    return np.concatenate([col0, col1], axis=-1)  # (..., 6)


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion(s) to axis-angle representation.

    Args:
        q: Quaternion(s) in (x, y, z, w) format, shape (..., 4)

    Returns:
        Axis-angle vector(s) = axis * angle, shape (..., 3)
    """
    q = np.asarray(q)
    original_shape = q.shape[:-1]

    # Flatten for processing
    q_flat = q.reshape(-1, 4)
    n = q_flat.shape[0]

    result = np.zeros((n, 3), dtype=q.dtype)

    for i in range(n):
        qi = q_flat[i]
        # Ensure w is in valid range
        w = np.clip(qi[3], -1.0, 1.0)
        sin_half = np.sqrt(1.0 - w * w)

        if sin_half > 1e-10:
            angle = 2.0 * np.arccos(w)
            axis = qi[:3] / sin_half
            result[i] = axis * angle

    return result.reshape((*original_shape, 3))


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation to quaternion(s).

    Args:
        aa: Axis-angle vector(s) = axis * angle, shape (..., 3)

    Returns:
        Quaternion(s) in (x, y, z, w) format, shape (..., 4)
    """
    aa = np.asarray(aa)
    original_shape = aa.shape[:-1]

    # Flatten for processing
    aa_flat = aa.reshape(-1, 3)
    n = aa_flat.shape[0]

    result = np.zeros((n, 4), dtype=aa.dtype)

    for i in range(n):
        v = aa_flat[i]
        angle = np.linalg.norm(v)

        if angle < 1e-10:
            # Zero rotation -> identity quaternion
            result[i] = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            axis = v / angle
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)
            result[i, :3] = axis * sin_half
            result[i, 3] = cos_half

    return result.reshape((*original_shape, 4))


# =============================================================================
# Rot6D Utilities
# =============================================================================


def rot6d_to_rotmat(r6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.

    Uses Gram-Schmidt orthonormalization to ensure valid rotation matrix.

    Args:
        r6d: 6D rotation representation, shape (..., 6)

    Returns:
        Rotation matrix/matrices, shape (..., 3, 3)
    """
    r6d = np.asarray(r6d)
    original_shape = r6d.shape[:-1]

    # Flatten for processing
    r6d_flat = r6d.reshape(-1, 6)
    n = r6d_flat.shape[0]

    result = np.zeros((n, 3, 3), dtype=r6d.dtype)

    for i in range(n):
        a1 = r6d_flat[i, :3]
        a2 = r6d_flat[i, 3:6]

        # Gram-Schmidt orthonormalization
        b1 = a1 / (np.linalg.norm(a1) + 1e-10)
        b2_orth = a2 - np.dot(b1, a2) * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth) + 1e-10)
        b3 = np.cross(b1, b2)

        result[i, :, 0] = b1
        result[i, :, 1] = b2
        result[i, :, 2] = b3

    return result.reshape((*original_shape, 3, 3))


# =============================================================================
# Frame Transform Utilities
# =============================================================================


def world_to_base_pos(
    p_w: np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
) -> np.ndarray:
    """
    Transform position(s) from world frame to robot base frame.

    p_b = R(base_quat_w)^T * (p_w - base_pos_w)

    Args:
        p_w: Position(s) in world frame, shape (..., 3)
        base_pos_w: Base position in world frame, shape (..., 3) or (3,)
        base_quat_w: Base orientation in world frame (x,y,z,w), shape (..., 4) or (4,)

    Returns:
        Position(s) in base frame, shape (..., 3)
    """
    p_w = np.asarray(p_w)
    base_pos_w = np.asarray(base_pos_w)
    base_quat_w = np.asarray(base_quat_w)

    # Translate to base origin
    p_translated = p_w - base_pos_w

    # Rotate by inverse of base orientation
    # R^T * v = R_inv * v, and R_inv corresponds to quat_conjugate
    base_quat_inv = quat_conjugate(base_quat_w)
    R_inv = quat_to_rotmat(base_quat_inv)

    # Apply rotation: p_b = R_inv @ p_translated
    # Handle batched case
    if p_translated.ndim == 1:
        p_b = R_inv @ p_translated
    else:
        # (..., 3, 3) @ (..., 3, 1) -> (..., 3, 1) -> (..., 3)
        p_b = np.einsum("...ij,...j->...i", R_inv, p_translated)

    return p_b


def world_to_base_quat(
    q_w: np.ndarray,
    base_quat_w: np.ndarray,
) -> np.ndarray:
    """
    Transform orientation from world frame to robot base frame.

    q_b = base_quat_w^{-1} * q_w

    Args:
        q_w: Orientation in world frame (x,y,z,w), shape (..., 4)
        base_quat_w: Base orientation in world frame (x,y,z,w), shape (..., 4) or (4,)

    Returns:
        Orientation in base frame (x,y,z,w), shape (..., 4)
    """
    base_quat_inv = quat_conjugate(base_quat_w)
    return quat_multiply(base_quat_inv, q_w)


def base_to_world_pos(
    p_b: np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
) -> np.ndarray:
    """
    Transform position(s) from robot base frame to world frame.

    p_w = R(base_quat_w) * p_b + base_pos_w

    Args:
        p_b: Position(s) in base frame, shape (..., 3)
        base_pos_w: Base position in world frame, shape (..., 3) or (3,)
        base_quat_w: Base orientation in world frame (x,y,z,w), shape (..., 4) or (4,)

    Returns:
        Position(s) in world frame, shape (..., 3)
    """
    p_b = np.asarray(p_b)
    base_pos_w = np.asarray(base_pos_w)
    base_quat_w = np.asarray(base_quat_w)

    R = quat_to_rotmat(base_quat_w)

    # Apply rotation: p_rotated = R @ p_b
    if p_b.ndim == 1:
        p_rotated = R @ p_b
    else:
        p_rotated = np.einsum("...ij,...j->...i", R, p_b)

    return p_rotated + base_pos_w


def base_to_world_quat(
    q_b: np.ndarray,
    base_quat_w: np.ndarray,
) -> np.ndarray:
    """
    Transform orientation from robot base frame to world frame.

    q_w = base_quat_w * q_b

    Args:
        q_b: Orientation in base frame (x,y,z,w), shape (..., 4)
        base_quat_w: Base orientation in world frame (x,y,z,w), shape (..., 4) or (4,)

    Returns:
        Orientation in world frame (x,y,z,w), shape (..., 4)
    """
    return quat_multiply(base_quat_w, q_b)


# =============================================================================
# High-Level Encode/Decode Functions
# =============================================================================


def encode_state_isaaclab_to_policy(
    eef_pos_w: np.ndarray,
    eef_quat_w: np.ndarray,
    gripper_qpos: np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
) -> np.ndarray:
    """
    Encode IsaacLab raw observations into the canonical 11D policy state.

    State = [eef_pos_b(3), eef_rot6d_b(6), gripper_qpos(2)]

    Args:
        eef_pos_w: EE position in world frame, shape (3,) or (B, 3)
        eef_quat_w: EE orientation in world frame (x,y,z,w), shape (4,) or (B, 4)
        gripper_qpos: Gripper joint positions, shape (2,) or (B, 2)
        base_pos_w: Base position in world frame, shape (3,) or (B, 3)
        base_quat_w: Base orientation in world frame (x,y,z,w), shape (4,) or (B, 4)

    Returns:
        11D policy state vector, shape (11,) or (B, 11)
    """
    eef_pos_w = np.asarray(eef_pos_w)
    eef_quat_w = np.asarray(eef_quat_w)
    gripper_qpos = np.asarray(gripper_qpos)
    base_pos_w = np.asarray(base_pos_w)
    base_quat_w = np.asarray(base_quat_w)

    # Transform EE pose to base frame
    eef_pos_b = world_to_base_pos(eef_pos_w, base_pos_w, base_quat_w)
    eef_quat_b = world_to_base_quat(eef_quat_w, base_quat_w)

    # Convert orientation to Rot6D
    eef_rot6d_b = quat_to_rot6d(eef_quat_b)

    # Concatenate into 11D state
    if eef_pos_b.ndim == 1:
        state = np.concatenate([eef_pos_b, eef_rot6d_b, gripper_qpos])
    else:
        state = np.concatenate([eef_pos_b, eef_rot6d_b, gripper_qpos], axis=-1)

    return state


def encode_action_isaaclab_to_policy(
    delta_pos_w: np.ndarray,
    delta_rot_w: np.ndarray,
    gripper_cmd: np.ndarray,
    base_quat_w: np.ndarray,
) -> np.ndarray:
    """
    Encode IsaacLab action (world-frame deltas) into the canonical 7D policy action.

    Action = [Δpos_b(3), Δaxis_angle_b(3), gripper(1)]

    Note: Rotation deltas are transformed from world frame to base frame by
    rotating the axis-angle vector.

    Args:
        delta_pos_w: Position delta in world frame, shape (3,) or (B, 3)
        delta_rot_w: Rotation delta in world frame (axis-angle), shape (3,) or (B, 3)
        gripper_cmd: Gripper command, shape (1,) or scalar or (B, 1) or (B,)
        base_quat_w: Base orientation in world frame (x,y,z,w), shape (4,) or (B, 4)

    Returns:
        7D policy action vector, shape (7,) or (B, 7)
    """
    delta_pos_w = np.asarray(delta_pos_w)
    delta_rot_w = np.asarray(delta_rot_w)
    gripper_cmd = np.atleast_1d(np.asarray(gripper_cmd))
    base_quat_w = np.asarray(base_quat_w)

    # Transform position delta to base frame (only rotation, no translation)
    # Δp_b = R(base_quat_w)^T * Δp_w
    base_quat_inv = quat_conjugate(base_quat_w)
    R_inv = quat_to_rotmat(base_quat_inv)

    if delta_pos_w.ndim == 1:
        delta_pos_b = R_inv @ delta_pos_w
        delta_rot_b = R_inv @ delta_rot_w
    else:
        delta_pos_b = np.einsum("...ij,...j->...i", R_inv, delta_pos_w)
        delta_rot_b = np.einsum("...ij,...j->...i", R_inv, delta_rot_w)

    # Ensure gripper_cmd has correct shape
    if gripper_cmd.ndim == 0:
        gripper_cmd = gripper_cmd.reshape(1)
    elif gripper_cmd.ndim == 1 and delta_pos_b.ndim > 1:
        gripper_cmd = gripper_cmd.reshape(-1, 1)

    # Concatenate into 7D action
    if delta_pos_b.ndim == 1:
        action = np.concatenate([delta_pos_b, delta_rot_b, gripper_cmd])
    else:
        if gripper_cmd.ndim == 1:
            gripper_cmd = gripper_cmd[:, None]
        action = np.concatenate([delta_pos_b, delta_rot_b, gripper_cmd], axis=-1)

    return action


def decode_action_policy_to_isaaclab(
    action: np.ndarray,
    base_quat_w: np.ndarray,
) -> dict:
    """
    Decode the canonical 7D policy action into IsaacLab-compatible format.

    Takes action in base frame and converts to world frame for IsaacLab controller.

    Args:
        action: 7D policy action [Δpos_b(3), Δaxis_angle_b(3), gripper(1)],
                shape (7,) or (B, 7)
        base_quat_w: Base orientation in world frame (x,y,z,w), shape (4,) or (B, 4)

    Returns:
        Dictionary with:
            - delta_pos_w: Position delta in world frame, shape (3,) or (B, 3)
            - delta_rot_w: Rotation delta in world frame (axis-angle), shape (3,) or (B, 3)
            - gripper_cmd: Gripper command, shape (1,) or (B, 1)
    """
    action = np.asarray(action)
    base_quat_w = np.asarray(base_quat_w)

    # Extract components
    if action.ndim == 1:
        delta_pos_b = action[:3]
        delta_rot_b = action[3:6]
        gripper_cmd = action[6:7]
    else:
        delta_pos_b = action[..., :3]
        delta_rot_b = action[..., 3:6]
        gripper_cmd = action[..., 6:7]

    # Transform from base frame to world frame
    R = quat_to_rotmat(base_quat_w)

    if delta_pos_b.ndim == 1:
        delta_pos_w = R @ delta_pos_b
        delta_rot_w = R @ delta_rot_b
    else:
        delta_pos_w = np.einsum("...ij,...j->...i", R, delta_pos_b)
        delta_rot_w = np.einsum("...ij,...j->...i", R, delta_rot_b)

    return {
        "delta_pos_w": delta_pos_w,
        "delta_rot_w": delta_rot_w,
        "gripper_cmd": gripper_cmd,
    }


def decode_action_policy_to_isaaclab_7d(
    action: np.ndarray,
    base_quat_w: np.ndarray,
) -> np.ndarray:
    """
    Decode the canonical 7D policy action into IsaacLab 7D action format.

    Convenience function that returns a flat 7D array suitable for IsaacLab step().

    Args:
        action: 7D policy action [Δpos_b(3), Δaxis_angle_b(3), gripper(1)],
                shape (7,) or (B, 7)
        base_quat_w: Base orientation in world frame (x,y,z,w), shape (4,) or (B, 4)

    Returns:
        7D action for IsaacLab: [Δpos_w(3), Δrot_w(3), gripper(1)],
        shape (7,) or (B, 7)
    """
    result = decode_action_policy_to_isaaclab(action, base_quat_w)

    if action.ndim == 1:
        return np.concatenate([
            result["delta_pos_w"],
            result["delta_rot_w"],
            result["gripper_cmd"],
        ])
    else:
        return np.concatenate([
            result["delta_pos_w"],
            result["delta_rot_w"],
            result["gripper_cmd"],
        ], axis=-1)
