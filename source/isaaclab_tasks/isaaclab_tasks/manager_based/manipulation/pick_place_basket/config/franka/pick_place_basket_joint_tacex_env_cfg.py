# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""TacEx-enabled Franka Pick and Place Basket with joint-position control."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place_basket import mdp

from .pick_place_basket_ik_rel_tacex_env_cfg import FrankaPickPlaceBasketTacExEnvCfg


@configclass
class FrankaPickPlaceBasketJointTacExEnvCfg(FrankaPickPlaceBasketTacExEnvCfg):
    """TacEx environment with absolute 7-DoF joint targets + binary gripper."""

    def __post_init__(self):
        super().__post_init__()

        # Absolute joint-space control to match LeRobot joint action convention:
        # action = [arm_joint_pos_target(7), gripper(1)].
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=False,
        )
