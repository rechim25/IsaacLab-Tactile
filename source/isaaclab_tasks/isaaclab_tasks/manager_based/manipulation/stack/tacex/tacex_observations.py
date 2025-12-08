# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacEx observation functions for Isaac Lab environments.

These functions provide tactile observations from TacEx sensors
for use in the Isaac Lab observation manager.

The observations include:
- Tactile RGB images (from Taxim optical simulation)
- Marker flow fields (from FOTS simulation)
- Height maps (gelpad deformation)
- Contact forces (estimated from deformation)

References:
- TacEx Paper: https://arxiv.org/pdf/2411.04776
- TacEx Site: https://sites.google.com/view/tacex
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tacex_tactile_rgb(
    env: ManagerBasedRLEnv,
    sensor_name: str = "left_finger_tactile",
) -> torch.Tensor:
    """Get tactile RGB image from TacEx simulation.

    The RGB image is generated using TacEx's Taxim simulation,
    which maps gelpad surface normals to light intensities using
    a polynomial lookup table.

    Args:
        env: The Isaac Lab environment with TacEx manager attached.
        sensor_name: Name of the TacEx sensor to query.

    Returns:
        Tactile RGB image of shape (num_envs, H, W, 3).
        Returns zeros if TacEx is not initialized.
    """
    if not hasattr(env, "tacex") or env.tacex is None:
        # Return zeros if TacEx not configured
        return torch.zeros(
            (env.num_envs, 64, 64, 3),
            dtype=torch.float32,
            device=env.device,
        )

    return env.tacex.get_tactile_rgb(sensor_name)


def tacex_tactile_rgb_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get tactile RGB from left finger sensor."""
    return tacex_tactile_rgb(env, "left_finger_tactile")


def tacex_tactile_rgb_right(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get tactile RGB from right finger sensor."""
    return tacex_tactile_rgb(env, "right_finger_tactile")


def tacex_marker_flow(
    env: ManagerBasedRLEnv,
    sensor_name: str = "left_finger_tactile",
) -> torch.Tensor:
    """Get marker motion flow from TacEx FOTS simulation.

    The marker flow represents displacement of gel surface markers
    due to contact, useful for slip detection and force estimation.

    Args:
        env: The Isaac Lab environment with TacEx manager attached.
        sensor_name: Name of the TacEx sensor to query.

    Returns:
        Marker flow of shape (num_envs, H, W, 2) with x,y displacements.
        Returns zeros if TacEx is not initialized.
    """
    if not hasattr(env, "tacex") or env.tacex is None:
        return torch.zeros(
            (env.num_envs, 64, 64, 2),
            dtype=torch.float32,
            device=env.device,
        )

    return env.tacex.get_marker_flow(sensor_name)


def tacex_marker_flow_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get marker flow from left finger sensor."""
    return tacex_marker_flow(env, "left_finger_tactile")


def tacex_marker_flow_right(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get marker flow from right finger sensor."""
    return tacex_marker_flow(env, "right_finger_tactile")


def tacex_height_map(
    env: ManagerBasedRLEnv,
    sensor_name: str = "left_finger_tactile",
) -> torch.Tensor:
    """Get gelpad height/deformation map from TacEx.

    The height map represents the indentation depth at each point
    on the gelpad surface, computed from GIPC soft body deformation.

    Args:
        env: The Isaac Lab environment with TacEx manager attached.
        sensor_name: Name of the TacEx sensor to query.

    Returns:
        Height map of shape (num_envs, H, W).
        Returns zeros if TacEx is not initialized.
    """
    if not hasattr(env, "tacex") or env.tacex is None:
        return torch.zeros(
            (env.num_envs, 64, 64),
            dtype=torch.float32,
            device=env.device,
        )

    return env.tacex.get_height_map(sensor_name)


def tacex_height_map_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get height map from left finger sensor."""
    return tacex_height_map(env, "left_finger_tactile")


def tacex_height_map_right(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get height map from right finger sensor."""
    return tacex_height_map(env, "right_finger_tactile")


def tacex_contact_force(
    env: ManagerBasedRLEnv,
    sensor_name: str = "left_finger_tactile",
) -> torch.Tensor:
    """Get estimated contact force from TacEx tactile data.

    The contact force is estimated from the gelpad deformation.

    Args:
        env: The Isaac Lab environment with TacEx manager attached.
        sensor_name: Name of the TacEx sensor to query.

    Returns:
        Contact force vector of shape (num_envs, 3).
        Returns zeros if TacEx is not initialized.
    """
    if not hasattr(env, "tacex") or env.tacex is None:
        return torch.zeros(
            (env.num_envs, 3),
            dtype=torch.float32,
            device=env.device,
        )

    return env.tacex.get_contact_force(sensor_name)


def tacex_contact_force_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get contact force from left finger sensor."""
    return tacex_contact_force(env, "left_finger_tactile")


def tacex_contact_force_right(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get contact force from right finger sensor."""
    return tacex_contact_force(env, "right_finger_tactile")


def tacex_combined_tactile_rgb(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get combined tactile RGB from both finger sensors.

    Concatenates the left and right finger tactile images along
    a new dimension for processing both together.

    Args:
        env: The Isaac Lab environment with TacEx manager attached.

    Returns:
        Combined tactile RGB of shape (num_envs, 2, H, W, 3).
        Returns zeros if TacEx is not initialized.
    """
    left = tacex_tactile_rgb_left(env)
    right = tacex_tactile_rgb_right(env)
    return torch.stack([left, right], dim=1)


def tacex_combined_marker_flow(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get combined marker flow from both finger sensors.

    Concatenates the left and right finger marker flows along
    a new dimension for processing both together.

    Args:
        env: The Isaac Lab environment with TacEx manager attached.

    Returns:
        Combined marker flow of shape (num_envs, 2, H, W, 2).
        Returns zeros if TacEx is not initialized.
    """
    left = tacex_marker_flow_left(env)
    right = tacex_marker_flow_right(env)
    return torch.stack([left, right], dim=1)


def tacex_is_contact(
    env: ManagerBasedRLEnv,
    sensor_name: str = "left_finger_tactile",
    threshold: float = 0.001,
) -> torch.Tensor:
    """Check if there is contact detected by the tactile sensor.

    Contact is determined by checking if the maximum height map value
    exceeds a threshold (indicating deformation from contact).

    Args:
        env: The Isaac Lab environment with TacEx manager attached.
        sensor_name: Name of the TacEx sensor to query.
        threshold: Minimum height map value to consider as contact.

    Returns:
        Float tensor of shape (num_envs, 1) indicating contact (1.0 or 0.0).
    """
    height_map = tacex_height_map(env, sensor_name)
    # Check if max deformation exceeds threshold
    max_deformation = height_map.view(env.num_envs, -1).max(dim=1).values
    is_contact = (max_deformation > threshold).float().unsqueeze(-1)
    return is_contact


def tacex_is_contact_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check contact on left finger."""
    return tacex_is_contact(env, "left_finger_tactile")


def tacex_is_contact_right(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check contact on right finger."""
    return tacex_is_contact(env, "right_finger_tactile")


# Export all observation functions
__all__ = [
    "tacex_tactile_rgb",
    "tacex_tactile_rgb_left",
    "tacex_tactile_rgb_right",
    "tacex_marker_flow",
    "tacex_marker_flow_left",
    "tacex_marker_flow_right",
    "tacex_height_map",
    "tacex_height_map_left",
    "tacex_height_map_right",
    "tacex_contact_force",
    "tacex_contact_force_left",
    "tacex_contact_force_right",
    "tacex_combined_tactile_rgb",
    "tacex_combined_marker_flow",
    "tacex_is_contact",
    "tacex_is_contact_left",
    "tacex_is_contact_right",
]



