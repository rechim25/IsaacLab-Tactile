# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Peg insertion manipulation environment.
"""

import gymnasium as gym

from . import agents
from .peg_insert_env_cfg import PegInsertEnvCfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Peg-Insert-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka.peg_insert_joint_pos_env_cfg:FrankaPegInsertEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Peg-Insert-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka.peg_insert_joint_pos_env_cfg:FrankaPegInsertEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Peg-Insert-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka.peg_insert_ik_rel_env_cfg:FrankaPegInsertEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Peg-Insert-Franka-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka.peg_insert_ik_rel_env_cfg:FrankaPegInsertEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

