# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka robot configuration for pick and place basket with joint position control."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.pick_place_basket import mdp
from isaaclab_tasks.manager_based.manipulation.pick_place_basket.pick_place_basket_env_cfg import (
    PickPlaceBasketEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Randomize cube size (±15% variation by default)
    randomize_cube_scale = EventTerm(
        func=mdp.randomize_cube_scale,
        mode="reset",
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "scale_range": (0.85, 1.15),  # Single parameter: (min_scale, max_scale)
        },
    )

    # Randomize basket position first (in a reachable region)
    # Keep Y range closer to center to avoid IK issues at workspace edges
    randomize_basket_position = EventTerm(
        func=mdp.randomize_basket_pose,
        mode="reset",
        params={
            "basket_cfg": SceneEntityCfg("basket"),
            "pose_range": {
                "x": (0.50, 0.60),   # Front-right of table
                "y": (0.12, 0.20),   # Right side (clear gap from cube region)
                "z": (0.0203, 0.0203),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    # Randomize cube position (in a region well-separated from basket)
    # Y ranges don't overlap: cube Y <= -0.05, basket Y >= 0.12
    randomize_cube_position = EventTerm(
        func=mdp.randomize_cube_pose,
        mode="reset",
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "basket_cfg": SceneEntityCfg("basket"),
            "cube_pose_range": {
                "x": (0.35, 0.50),   # Left-center of table
                "y": (-0.18, -0.05), # Left side (clear gap from basket)
                "z": (0.0203, 0.0203),
                "yaw": (-0.5, 0.5),
            },
            "basket_pose_range": {
                "x": (0.50, 0.60),
                "y": (0.12, 0.20),
                "z": (0.0203, 0.0203),
                "yaw": (-0.2, 0.2),
            },
            "min_separation": 0.25,  # Large safety margin for reliable grasping
        },
    )


@configclass
class FrankaPickPlaceBasketEnvCfg(PickPlaceBasketEnvCfg):
    """Franka pick and place basket environment with joint position control."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        
        # Utilities for gripper status check
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # Rigid body properties for cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Cube - using a procedurally generated cuboid for size randomization flexibility
        # Using blue block as default, can be changed for color randomization
        self.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.42, -0.12, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube")],
            ),
        )

        # Basket properties
        basket_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Basket - using a sorting bin scaled to appropriate size
        self.scene.basket = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Basket",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.15, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
                scale=(1.0, 1.0, 2.0),  # Scaled to be a reasonable basket size
                rigid_props=basket_properties,
                semantic_tags=[("class", "basket")],
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

