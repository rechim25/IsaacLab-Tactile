# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    pick_place_basket_joint_pos_env_cfg,
    pick_place_basket_ik_rel_env_cfg,
    pick_place_basket_ik_rel_tacex_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Pick-Place-Basket-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pick_place_basket_joint_pos_env_cfg.FrankaPickPlaceBasketEnvCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Pick-Place-Basket-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pick_place_basket_ik_rel_env_cfg.FrankaPickPlaceBasketEnvCfg,
    },
    disable_env_checker=True,
)

##
# TacEx Tactile Sensing
##

gym.register(
    id="Isaac-Pick-Place-Basket-Franka-IK-Rel-TacEx-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pick_place_basket_ik_rel_tacex_env_cfg.FrankaPickPlaceBasketTacExEnvCfg,
    },
    disable_env_checker=True,
)

