# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacEx tactile simulation integration for Isaac Lab stack environments.

This module provides integration with TacEx (GelSight Tactile Simulation in Isaac Sim)
for tactile sensing in manipulation tasks.

References:
- TacEx Paper: https://arxiv.org/pdf/2411.04776
- TacEx Site: https://sites.google.com/view/tacex
"""

from .tacex_manager import TacExManager, TacExSensorConfig
from .tacex_env import TacExManagerBasedRLEnv
from .tacex_observations import *  # noqa: F401, F403

__all__ = ["TacExManager", "TacExSensorConfig", "TacExManagerBasedRLEnv"]

