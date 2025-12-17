# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""TacEx-enabled Franka Pick and Place Basket environment configuration.

This module provides a manager-based Pick and Place Basket environment with TacEx
GelSight tactile sensing integration.

TacEx (https://sites.google.com/view/tacex) provides:
- GPU-accelerated tactile RGB simulation via Taxim
- Marker motion simulation via FOTS
- GIPC soft body simulation for gelpad deformation

References:
- TacEx Paper: https://arxiv.org/pdf/2411.04776
- TacEx Site: https://sites.google.com/view/tacex
"""

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.pick_place_basket import mdp

from . import pick_place_basket_joint_pos_env_cfg

# TacEx imports
try:
    from tacex.simulation_approaches.gpu_taxim import TaximSimulatorCfg
    from tacex_assets import TACEX_ASSETS_DATA_DIR
    from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
        FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
    )
    from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

    TACEX_AVAILABLE = True
except ImportError:
    TACEX_AVAILABLE = False
    print("[TacEx] TacEx not installed. Using stub mode.")

# Fallback to standard Franka if TacEx not available
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class TacExObservationsCfg:
    """Observation specifications including TacEx tactile observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive and state observations for policy."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        cube_pos = ObsTerm(func=mdp.cube_position_in_world_frame)
        cube_quat = ObsTerm(func=mdp.cube_orientation_in_world_frame)
        basket_pos = ObsTerm(func=mdp.basket_position_in_world_frame)
        basket_quat = ObsTerm(func=mdp.basket_orientation_in_world_frame)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class VisionCfg(ObsGroup):
        """Visual camera observations."""

        table_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class TactileCfg(ObsGroup):
        """TacEx GelSight tactile sensor observations.

        Provides tactile RGB images from GelSight sensors mounted on
        the Franka gripper fingers.
        """

        # Left finger tactile RGB (Taxim simulation)
        tactile_rgb_left = ObsTerm(
            func=mdp.gelsight_tactile_rgb,
            params={"sensor_cfg": SceneEntityCfg("gsmini_left")},
        )

        # Right finger tactile RGB (Taxim simulation)
        tactile_rgb_right = ObsTerm(
            func=mdp.gelsight_tactile_rgb,
            params={"sensor_cfg": SceneEntityCfg("gsmini_right")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class TactileStateCfg(ObsGroup):
        """TacEx low-dimensional tactile state observations.

        Provides indentation depth and contact detection from tactile data.
        """

        # Indentation depth (how deep objects press into gelpad)
        indentation_left = ObsTerm(
            func=mdp.gelsight_indentation_depth,
            params={"sensor_cfg": SceneEntityCfg("gsmini_left")},
        )
        indentation_right = ObsTerm(
            func=mdp.gelsight_indentation_depth,
            params={"sensor_cfg": SceneEntityCfg("gsmini_right")},
        )

        # Binary contact detection
        is_contact_left = ObsTerm(
            func=mdp.gelsight_is_contact,
            params={"sensor_cfg": SceneEntityCfg("gsmini_left"), "threshold": 0.1},
        )
        is_contact_right = ObsTerm(
            func=mdp.gelsight_is_contact,
            params={"sensor_cfg": SceneEntityCfg("gsmini_right"), "threshold": 0.1},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TactileForceCfg(ObsGroup):
        """TacEx pseudo-force observations derived from tactile data.

        Provides 3D pseudo-force vectors computed from gelpad deformation.
        Two methods available:
        - Geometric: from height_map (pure geometry)
        - Photometric: from tactile_rgb (optical gradients)
        """

        # Geometric pseudo-force (from height_map deformation)
        force_geometric_left = ObsTerm(
            func=mdp.gelsight_pseudo_force_geometric,
            params={"sensor_cfg": SceneEntityCfg("gsmini_left")},
        )
        force_geometric_right = ObsTerm(
            func=mdp.gelsight_pseudo_force_geometric,
            params={"sensor_cfg": SceneEntityCfg("gsmini_right")},
        )

        # Photometric pseudo-force (from tactile_rgb gradients)
        force_photometric_left = ObsTerm(
            func=mdp.gelsight_pseudo_force_photometric,
            params={"sensor_cfg": SceneEntityCfg("gsmini_left")},
        )
        force_photometric_right = ObsTerm(
            func=mdp.gelsight_pseudo_force_photometric,
            params={"sensor_cfg": SceneEntityCfg("gsmini_right")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask completion observations (for hierarchical/skill learning)."""

        cube_grasped = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )
        cube_in_basket = ObsTerm(
            func=mdp.cube_in_basket,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "cube_cfg": SceneEntityCfg("cube"),
                "basket_cfg": SceneEntityCfg("basket"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    vision: VisionCfg = VisionCfg()
    tactile: TactileCfg = TactileCfg()
    tactile_state: TactileStateCfg = TactileStateCfg()
    tactile_force: TactileForceCfg = TactileForceCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaPickPlaceBasketTacExEnvCfg(pick_place_basket_joint_pos_env_cfg.FrankaPickPlaceBasketEnvCfg):
    """TacEx-enabled Franka Pick and Place Basket environment configuration.

    This environment adds GelSight tactile sensors to the Franka fingertips
    using the TacEx simulation framework.

    Features:
    - Tactile RGB images from both fingers (Taxim simulation)
    - Marker motion flow for slip detection (FOTS simulation)
    - Contact force estimation from gelpad deformation
    - Visual cameras (table + wrist) for visuomotor learning
    """

    observations: TacExObservationsCfg = TacExObservationsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Use TacEx Franka with GelSight sensors if available
        if TACEX_AVAILABLE:
            self.scene.robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot"
            )

            # Configure GelSight sensors on left and right fingers
            # Left finger sensor
            self.scene.gsmini_left = GelSightMiniCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case_left",
                sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
                    prim_path_appendix="/Camera",
                    update_period=0,
                    resolution=(256, 256),
                    data_types=["depth"],
                    clipping_range=(0.015, 0.029),
                ),
                device="cuda",
                debug_vis=True,
                optical_sim_cfg=TaximSimulatorCfg(
                    calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/GelSight_Mini/calibs/640x480",
                    gelpad_height=GelSightMiniCfg().gelpad_dimensions.height,
                    gelpad_to_camera_min_distance=0.024,
                    with_shadow=False,
                    tactile_img_res=(256, 256),
                    device="cuda",
                ),
                marker_motion_sim_cfg=None,  # Disable FOTS for speed
                data_types=["tactile_rgb", "height_map"],
            )

            # Right finger sensor
            self.scene.gsmini_right = GelSightMiniCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case_right",
                sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
                    prim_path_appendix="/Camera",
                    update_period=0,
                    resolution=(256, 256),
                    data_types=["depth"],
                    clipping_range=(0.015, 0.029),
                ),
                device="cuda",
                debug_vis=True,
                optical_sim_cfg=TaximSimulatorCfg(
                    calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/GelSight_Mini/calibs/640x480",
                    gelpad_height=GelSightMiniCfg().gelpad_dimensions.height,
                    gelpad_to_camera_min_distance=0.024,
                    with_shadow=False,
                    tactile_img_res=(256, 256),
                    device="cuda",
                ),
                marker_motion_sim_cfg=None,  # Disable FOTS for speed
                data_types=["tactile_rgb", "height_map"],
            )
        else:
            # Fallback to standard Franka
            self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set IK-relative actions for teleoperation compatibility
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Set wrist camera (mounted on panda_hand)
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.11, 0.0, -0.12),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.6),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )

        # Camera rendering settings
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"

        # List of image observations
        self.image_obs_list = ["table_cam", "wrist_cam"]

