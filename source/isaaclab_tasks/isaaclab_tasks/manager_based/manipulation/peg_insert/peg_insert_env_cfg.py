# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class PegInsertSceneCfg(InteractiveSceneCfg):
    """Configuration for the peg insert scene with a robot, peg, and hole."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Peg (to be grasped by gripper) - using ArticulationCfg like factory direct task
    peg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_peg_8mm.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, -0.1, 0.0203),  # On table surface, close to hole
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    # Hole (target) - sits on table surface
    hole = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Hole",
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_hole_8mm.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=10.0,  # High damping to keep it stable
                angular_damping=10.0,  # High damping to prevent rotation
                max_linear_velocity=0.0,  # Prevent movement
                max_angular_velocity=0.0,  # Prevent rotation
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  # Heavy to stay in place
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0),  # On table surface, same as stack cubes
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # robot state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        
        # end-effector state
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        
        # peg state
        peg_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("peg")})
        peg_quat = ObsTerm(func=mdp.object_orientation_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("peg")})
        
        # hole state (target)
        hole_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("hole")})
        hole_quat = ObsTerm(func=mdp.object_orientation_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("hole")})
        
        # relative state
        peg_to_hole_pos = ObsTerm(func=mdp.peg_to_hole_position)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Image observations group (not concatenated)."""

        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Terminate if peg drops below table
    peg_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("peg")},
    )
    
    # Success termination when peg is inserted
    success = DoneTerm(func=mdp.peg_inserted)


@configclass
class EventCfg:
    """Configuration for events."""

    # Robot initialization
    init_robot_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            # Good starting pose for Franka (similar to factory task)
            "default_pose": [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0, 0.04, 0.04],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Reset hole pose from a discrete preset set (for diverse demos)
    reset_hole_pose = EventTerm(
        func=mdp.set_pose_from_discrete_set,
        mode="reset",
        params={
            # Keep hole near center/positive-Y so it's not overlapping with peg presets
            "poses_xyz_yaw": [
                (0.55,  0.00, 0.0, 0.0),
                (0.58,  0.06, 0.0, 0.2),
                (0.52,  0.04, 0.0, -0.2),
            ],
            "asset_cfg": SceneEntityCfg("hole"),
        },
    )

    # Randomize peg position uniformly within ranges each reset
    reset_peg_pose = EventTerm(
        func=mdp.set_pose_from_discrete_set,
        mode="reset",
        params={
            # Peg presets on negative-Y side; avoids immediate success/stack collisions
            "poses_xyz_yaw": [
                (0.55, -0.10, 0.0203, 0.0),
                (0.50, -0.08, 0.0203, 0.2),
                (0.60, -0.12, 0.0203, -0.2),
            ],
            "asset_cfg": SceneEntityCfg("peg"),
        },
    )

    # Note: We intentionally do not randomize robot joints at reset to keep
    # a consistent end-effector orientation and gripper open state.


@configclass
class PegInsertEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the peg insertion environment."""

    # Scene settings
    scene: PegInsertSceneCfg = PegInsertSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers for this task
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = self.decimation

        # PhysX settings similar to Factory task for precise contact simulation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1

