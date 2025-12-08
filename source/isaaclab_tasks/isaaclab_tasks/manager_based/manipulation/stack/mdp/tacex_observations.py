# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacEx GelSight sensor observation functions for Isaac Lab environments.

These functions read tactile data from TacEx GelSight sensors configured
in the scene. The sensors must be added to the scene config using
GelSightMiniCfg or GelSightSensorCfg.

References:
- TacEx Paper: https://arxiv.org/pdf/2411.04776
- TacEx Site: https://sites.google.com/view/tacex
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gelsight_tactile_rgb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gsmini_left"),
) -> torch.Tensor:
    """Get tactile RGB image from a TacEx GelSight sensor.

    The RGB image is generated using TacEx's Taxim simulation,
    which maps gelpad surface normals to light intensities.

    Args:
        env: The Isaac Lab environment.
        sensor_cfg: Configuration specifying which sensor to read.

    Returns:
        Tactile RGB image of shape (num_envs, H, W, 3).
    """
    sensor = env.scene.sensors.get(sensor_cfg.name)
    if sensor is None:
        # Return zeros if sensor not configured
        return torch.zeros(
            (env.num_envs, 64, 64, 3),
            dtype=torch.float32,
            device=env.device,
        )

    return sensor.data.output.get("tactile_rgb", torch.zeros(
        (env.num_envs, 64, 64, 3),
        dtype=torch.float32,
        device=env.device,
    ))


def gelsight_height_map(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gsmini_left"),
) -> torch.Tensor:
    """Get height map (gelpad deformation) from a TacEx GelSight sensor.

    The height map represents the indentation depth at each point
    on the gelpad surface.

    Args:
        env: The Isaac Lab environment.
        sensor_cfg: Configuration specifying which sensor to read.

    Returns:
        Height map of shape (num_envs, H, W).
    """
    sensor = env.scene.sensors.get(sensor_cfg.name)
    if sensor is None:
        return torch.zeros(
            (env.num_envs, 64, 64),
            dtype=torch.float32,
            device=env.device,
        )

    return sensor.data.output.get("height_map", torch.zeros(
        (env.num_envs, 64, 64),
        dtype=torch.float32,
        device=env.device,
    ))


def gelsight_marker_motion(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gsmini_left"),
) -> torch.Tensor:
    """Get marker motion from a TacEx GelSight sensor (FOTS simulation).

    Marker motion represents the displacement of gel surface markers
    due to contact, useful for slip detection.

    Args:
        env: The Isaac Lab environment.
        sensor_cfg: Configuration specifying which sensor to read.

    Returns:
        Marker motion of shape (num_envs, 2, num_markers, 2).
        Dim 1: [initial_pos, current_pos]
        Dim 3: [x, y] position in image coordinates
    """
    sensor = env.scene.sensors.get(sensor_cfg.name)
    if sensor is None:
        return torch.zeros(
            (env.num_envs, 2, 99, 2),  # 11*9 = 99 markers default
            dtype=torch.float32,
            device=env.device,
        )

    return sensor.data.output.get("marker_motion", torch.zeros(
        (env.num_envs, 2, 99, 2),
        dtype=torch.float32,
        device=env.device,
    ))


def gelsight_indentation_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gsmini_left"),
) -> torch.Tensor:
    """Get indentation depth (scalar) from a TacEx GelSight sensor.

    Indentation depth indicates how deep objects are pressing into
    the gelpad surface.

    Args:
        env: The Isaac Lab environment.
        sensor_cfg: Configuration specifying which sensor to read.

    Returns:
        Indentation depth of shape (num_envs, 1).
    """
    sensor = env.scene.sensors.get(sensor_cfg.name)
    if sensor is None:
        return torch.zeros(
            (env.num_envs, 1),
            dtype=torch.float32,
            device=env.device,
        )

    depth = sensor.indentation_depth
    if depth is None:
        return torch.zeros(
            (env.num_envs, 1),
            dtype=torch.float32,
            device=env.device,
        )

    return depth.unsqueeze(-1)


def gelsight_is_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gsmini_left"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """Check if there is contact on a TacEx GelSight sensor.

    Contact is determined by checking if indentation depth exceeds
    a threshold.

    Args:
        env: The Isaac Lab environment.
        sensor_cfg: Configuration specifying which sensor to read.
        threshold: Minimum indentation depth (mm) to consider as contact.

    Returns:
        Float tensor of shape (num_envs, 1) indicating contact (1.0 or 0.0).
    """
    sensor = env.scene.sensors.get(sensor_cfg.name)
    if sensor is None:
        return torch.zeros(
            (env.num_envs, 1),
            dtype=torch.float32,
            device=env.device,
        )

    depth = sensor.indentation_depth
    if depth is None:
        return torch.zeros(
            (env.num_envs, 1),
            dtype=torch.float32,
            device=env.device,
        )

    is_contact = (depth > threshold).float().unsqueeze(-1)
    return is_contact


# Export all observation functions
__all__ = [
    "gelsight_tactile_rgb",
    "gelsight_height_map",
    "gelsight_marker_motion",
    "gelsight_indentation_depth",
    "gelsight_is_contact",
]

