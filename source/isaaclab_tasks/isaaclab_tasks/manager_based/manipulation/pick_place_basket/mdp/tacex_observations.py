# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""TacEx GelSight sensor observation functions for pick and place basket environment.

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


def gelsight_pseudo_force_geometric(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gsmini_left"),
) -> torch.Tensor:
    """Compute normalized 3D pseudo-force from height map (geometric method).

    Derives a force-like vector from gelpad deformation geometry:
    - Fz (normal): Total deformation depth, normalized to [0, 1]
    - Fx, Fy (shear): Center of pressure offset, in [-1, 1]

    This uses pure geometry without optical simulation artifacts.

    Args:
        env: The Isaac Lab environment.
        sensor_cfg: Configuration specifying which sensor to read.

    Returns:
        Normalized pseudo-force vector of shape (num_envs, 3) as [Fx, Fy, Fz].
        Fx, Fy in [-1, 1], Fz in [0, 1].
    """
    sensor = env.scene.sensors.get(sensor_cfg.name)
    if sensor is None:
        return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)

    height_map = sensor.data.output.get("height_map")
    if height_map is None:
        return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)

    # Height map baseline (no contact) - typically ~29mm for GelSight Mini
    baseline = height_map.amax(dim=(-2, -1), keepdim=True)

    # Deformation: positive = pressed in
    deformation = (baseline - height_map).clamp(min=0)  # (num_envs, H, W)

    # Normal force (Fz): total deformation
    Fz_raw = deformation.sum(dim=(-2, -1))  # (num_envs,)

    # Shear forces (Fx, Fy): center of pressure offset from center
    H, W = height_map.shape[-2:]

    # Coordinate grids normalized to [-1, 1]
    y_coords = torch.linspace(-1, 1, H, device=env.device).view(1, H, 1)
    x_coords = torch.linspace(-1, 1, W, device=env.device).view(1, 1, W)

    # Weighted center of pressure (already in [-1, 1] from normalized coords)
    total_def = Fz_raw.unsqueeze(-1).unsqueeze(-1) + 1e-6
    Fx_raw = (deformation * x_coords).sum(dim=(-2, -1)) / total_def.squeeze()
    Fy_raw = (deformation * y_coords).sum(dim=(-2, -1)) / total_def.squeeze()

    # Normalize each component
    Fx = Fx_raw.clamp(-1, 1)
    Fy = Fy_raw.clamp(-1, 1)

    # Fz: normalize by estimated max deformation
    max_deformation = H * W * 0.5  # ~0.5mm max depth per pixel
    Fz = (Fz_raw / max_deformation).clamp(0, 1)

    return torch.stack([Fx, Fy, Fz], dim=-1)


def gelsight_pseudo_force_photometric(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gsmini_left"),
) -> torch.Tensor:
    """Compute 3D pseudo-force from tactile RGB (photometric method).

    Derives a force-like vector from the tactile RGB image colors:
    - R channel encodes surface gradient in X direction
    - G channel encodes surface gradient in Y direction
    - Intensity changes indicate contact pressure

    This uses the optical simulation output which may better capture
    fine surface details and lighting-based gradient information.

    Args:
        env: The Isaac Lab environment.
        sensor_cfg: Configuration specifying which sensor to read.

    Returns:
        Pseudo-force vector of shape (num_envs, 3) as [Fx, Fy, Fz].
    """
    sensor = env.scene.sensors.get(sensor_cfg.name)
    if sensor is None:
        return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)

    tactile_rgb = sensor.data.output.get("tactile_rgb")
    if tactile_rgb is None:
        return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)

    # Tactile RGB is in [0, 1] range, neutral (no contact) ~0.5 per channel
    # Center around zero for gradient computation
    rgb_centered = tactile_rgb - 0.5  # (num_envs, H, W, 3)

    # R channel → X gradient (shear in X)
    # G channel → Y gradient (shear in Y)
    grad_x = rgb_centered[..., 0].mean(dim=(-2, -1))  # (num_envs,)
    grad_y = rgb_centered[..., 1].mean(dim=(-2, -1))  # (num_envs,)

    # Contact intensity from deviation of all channels from baseline
    # More deviation = stronger contact
    rgb_deviation = (rgb_centered.abs()).mean(dim=(-2, -1, -3))  # (num_envs,)

    # Scale factors (empirically tuned)
    shear_scale = 10.0
    normal_scale = 100.0

    Fx = grad_x * rgb_deviation * shear_scale
    Fy = grad_y * rgb_deviation * shear_scale
    Fz = rgb_deviation * normal_scale

    return torch.stack([Fx, Fy, Fz], dim=-1)


# Export all observation functions
__all__ = [
    "gelsight_tactile_rgb",
    "gelsight_height_map",
    "gelsight_marker_motion",
    "gelsight_indentation_depth",
    "gelsight_is_contact",
    "gelsight_pseudo_force_geometric",
    "gelsight_pseudo_force_photometric",
]

