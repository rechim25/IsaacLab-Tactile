# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_inserted(env: ManagerBasedRLEnv, threshold: float = 0.04) -> torch.Tensor:
    """Terminate when the peg is successfully inserted into the hole.
    
    Success criteria:
    1. Peg is centered over the hole (XY distance < 2.5mm)
    2. Peg has descended to the target depth (Z position relative to hole)
    
    Args:
        env: The environment instance.
        threshold: The height threshold as a fraction of hole height. Default is 0.04 (4%).
    
    Returns:
        Boolean tensor indicating which environments have successfully inserted the peg.
    """
    peg: Articulation = env.scene["peg"]
    hole: Articulation = env.scene["hole"]
    
    # Get positions in world frame
    peg_pos_w = peg.data.root_pos_w - env.scene.env_origins
    hole_pos_w = hole.data.root_pos_w - env.scene.env_origins
    
    # Compute XY distance (centering criterion)
    xy_dist = torch.norm(peg_pos_w[:, :2] - hole_pos_w[:, :2], dim=-1)
    is_centered = xy_dist < 0.0025  # 2.5mm threshold
    
    # Compute Z displacement (insertion depth criterion)
    # Peg base should be at or below hole top surface
    hole_height = 0.025  # from factory_hole_8mm config
    z_disp = peg_pos_w[:, 2] - hole_pos_w[:, 2]
    height_threshold = hole_height * threshold
    is_inserted = z_disp < height_threshold
    
    # Success requires both centering and insertion
    success = torch.logical_and(is_centered, is_inserted)
    
    return success

