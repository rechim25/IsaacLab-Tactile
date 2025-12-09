# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a cube stacking state machine.

The state machine generates demonstrations automatically without human teleoperation.
It uses the `warp` library to run the state machine in parallel on the GPU.

Usage:
    # Run with visualization
    ./isaaclab.sh -p scripts/environments/state_machine/stack_cube_sm.py --num_envs 16

    # Run headless to collect data faster
    ./isaaclab.sh -p scripts/environments/state_machine/stack_cube_sm.py --num_envs 64 --headless

    # Save demonstrations to HDF5
    ./isaaclab.sh -p scripts/environments/state_machine/stack_cube_sm.py --num_envs 16 --save_demos --output_file ./datasets/stack_demos.hdf5
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Stack cube state machine for automatic demo generation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--save_demos", action="store_true", default=False, help="Save demonstrations to HDF5 file.")
parser.add_argument("--output_file", type=str, default="./datasets/stack_cube_demos.hdf5", help="Output HDF5 file path.")
parser.add_argument("--num_demos", type=int, default=100, help="Number of demonstrations to collect.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
import os
from collections.abc import Sequence

import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class StackSmState:
    """States for the stack state machine."""
    REST = wp.constant(0)
    APPROACH_ABOVE_CUBE2 = wp.constant(1)
    APPROACH_CUBE2 = wp.constant(2)
    GRASP_CUBE2 = wp.constant(3)
    LIFT_CUBE2 = wp.constant(4)
    APPROACH_ABOVE_CUBE1 = wp.constant(5)
    PLACE_CUBE2_ON_CUBE1 = wp.constant(6)
    RELEASE_CUBE2 = wp.constant(7)
    RETREAT_FROM_STACK = wp.constant(8)
    APPROACH_ABOVE_CUBE3 = wp.constant(9)
    APPROACH_CUBE3 = wp.constant(10)
    GRASP_CUBE3 = wp.constant(11)
    LIFT_CUBE3 = wp.constant(12)
    APPROACH_ABOVE_STACK = wp.constant(13)
    PLACE_CUBE3_ON_STACK = wp.constant(14)
    RELEASE_CUBE3 = wp.constant(15)
    DONE = wp.constant(16)


class StackSmWaitTime:
    """Additional wait times (in s) for states before switching."""
    REST = wp.constant(0.1)
    APPROACH_ABOVE = wp.constant(0.3)
    APPROACH = wp.constant(0.4)
    GRASP = wp.constant(0.3)
    LIFT = wp.constant(0.3)
    PLACE = wp.constant(0.3)
    RELEASE = wp.constant(0.2)
    RETREAT = wp.constant(0.2)
    DONE = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    cube1_pose: wp.array(dtype=wp.transform),
    cube2_pose: wp.array(dtype=wp.transform),
    cube3_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    """State machine for stacking cubes."""
    tid = wp.tid()
    
    # Get current state
    state = cycled_state[tid]
    wait_time = sm_wait_time[tid]
    
    # Get current positions
    ee_pos = wp.transform_get_translation(ee_pose[tid])
    cube1_pos = wp.transform_get_translation(cube1_pose[tid])
    cube2_pos = wp.transform_get_translation(cube2_pose[tid])
    cube3_pos = wp.transform_get_translation(cube3_pose[tid])
    des_ee_pos = wp.transform_get_translation(des_ee_pose[tid])
    
    # Height offsets
    approach_height = 0.15
    grasp_height = 0.02
    stack_height_1 = 0.05  # Height of first cube
    stack_height_2 = 0.10  # Height of two stacked cubes
    
    # Decrement wait time
    wait_time = wait_time - dt[tid]
    
    # State transitions
    if state == StackSmState.REST:
        # Wait at rest
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if wait_time <= 0.0:
            sm_state[tid] = StackSmState.APPROACH_ABOVE_CUBE2
            sm_wait_time[tid] = StackSmWaitTime.APPROACH_ABOVE
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.APPROACH_ABOVE_CUBE2:
        # Move above cube 2 (red)
        des_pos = wp.vec3(cube2_pos[0], cube2_pos[1], cube2_pos[2] + approach_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.APPROACH_CUBE2
            sm_wait_time[tid] = StackSmWaitTime.APPROACH
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.APPROACH_ABOVE
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.APPROACH_CUBE2:
        # Move down to cube 2
        des_pos = wp.vec3(cube2_pos[0], cube2_pos[1], cube2_pos[2] + grasp_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.GRASP_CUBE2
            sm_wait_time[tid] = StackSmWaitTime.GRASP
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.APPROACH
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.GRASP_CUBE2:
        # Close gripper
        des_pos = wp.vec3(cube2_pos[0], cube2_pos[1], cube2_pos[2] + grasp_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.CLOSE
        if wait_time <= 0.0:
            sm_state[tid] = StackSmState.LIFT_CUBE2
            sm_wait_time[tid] = StackSmWaitTime.LIFT
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.LIFT_CUBE2:
        # Lift cube 2
        des_pos = wp.vec3(cube2_pos[0], cube2_pos[1], cube2_pos[2] + approach_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.APPROACH_ABOVE_CUBE1
            sm_wait_time[tid] = StackSmWaitTime.APPROACH_ABOVE
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.LIFT
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.APPROACH_ABOVE_CUBE1:
        # Move above cube 1 (blue)
        des_pos = wp.vec3(cube1_pos[0], cube1_pos[1], cube1_pos[2] + approach_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.PLACE_CUBE2_ON_CUBE1
            sm_wait_time[tid] = StackSmWaitTime.PLACE
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.APPROACH_ABOVE
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.PLACE_CUBE2_ON_CUBE1:
        # Place cube 2 on cube 1
        des_pos = wp.vec3(cube1_pos[0], cube1_pos[1], cube1_pos[2] + stack_height_1 + grasp_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.RELEASE_CUBE2
            sm_wait_time[tid] = StackSmWaitTime.RELEASE
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.PLACE
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.RELEASE_CUBE2:
        # Release cube 2
        des_ee_pose[tid] = des_ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if wait_time <= 0.0:
            sm_state[tid] = StackSmState.RETREAT_FROM_STACK
            sm_wait_time[tid] = StackSmWaitTime.RETREAT
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.RETREAT_FROM_STACK:
        # Move up from stack
        des_pos = wp.vec3(cube1_pos[0], cube1_pos[1], cube1_pos[2] + approach_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.APPROACH_ABOVE_CUBE3
            sm_wait_time[tid] = StackSmWaitTime.APPROACH_ABOVE
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.RETREAT
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.APPROACH_ABOVE_CUBE3:
        # Move above cube 3 (green)
        des_pos = wp.vec3(cube3_pos[0], cube3_pos[1], cube3_pos[2] + approach_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.APPROACH_CUBE3
            sm_wait_time[tid] = StackSmWaitTime.APPROACH
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.APPROACH_ABOVE
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.APPROACH_CUBE3:
        # Move down to cube 3
        des_pos = wp.vec3(cube3_pos[0], cube3_pos[1], cube3_pos[2] + grasp_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.GRASP_CUBE3
            sm_wait_time[tid] = StackSmWaitTime.GRASP
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.APPROACH
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.GRASP_CUBE3:
        # Close gripper
        des_pos = wp.vec3(cube3_pos[0], cube3_pos[1], cube3_pos[2] + grasp_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.CLOSE
        if wait_time <= 0.0:
            sm_state[tid] = StackSmState.LIFT_CUBE3
            sm_wait_time[tid] = StackSmWaitTime.LIFT
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.LIFT_CUBE3:
        # Lift cube 3
        des_pos = wp.vec3(cube3_pos[0], cube3_pos[1], cube3_pos[2] + approach_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.APPROACH_ABOVE_STACK
            sm_wait_time[tid] = StackSmWaitTime.APPROACH_ABOVE
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.LIFT
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.APPROACH_ABOVE_STACK:
        # Move above the stack (cube1 + cube2)
        des_pos = wp.vec3(cube1_pos[0], cube1_pos[1], cube1_pos[2] + approach_height + stack_height_1)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.PLACE_CUBE3_ON_STACK
            sm_wait_time[tid] = StackSmWaitTime.PLACE
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.APPROACH_ABOVE
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.PLACE_CUBE3_ON_STACK:
        # Place cube 3 on stack
        des_pos = wp.vec3(cube1_pos[0], cube1_pos[1], cube1_pos[2] + stack_height_2 + grasp_height)
        des_ee_pose[tid] = wp.transform(des_pos, wp.transform_get_rotation(offset[tid]))
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(ee_pos, des_pos, position_threshold) and wait_time <= 0.0:
            sm_state[tid] = StackSmState.RELEASE_CUBE3
            sm_wait_time[tid] = StackSmWaitTime.RELEASE
        elif wait_time <= 0.0:
            sm_wait_time[tid] = StackSmWaitTime.PLACE
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.RELEASE_CUBE3:
        # Release cube 3
        des_ee_pose[tid] = des_ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if wait_time <= 0.0:
            sm_state[tid] = StackSmState.DONE
            sm_wait_time[tid] = StackSmWaitTime.DONE
        else:
            sm_wait_time[tid] = wait_time
            
    elif state == StackSmState.DONE:
        # Task complete - stay in place
        des_ee_pose[tid] = des_ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        sm_wait_time[tid] = wait_time


class StackCubeStateMachine:
    """A simple state machine for stacking three cubes in a sequential manner using PyTorch."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments.
            device: The device to run the state machine on.
        """
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        
        # State machine state
        self.sm_state = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros(num_envs, device=device)
        
        # Desired poses and gripper states
        self.des_ee_pose = torch.zeros(num_envs, 7, device=device)  # pos (3) + quat (4)
        self.des_gripper_state = torch.ones(num_envs, device=device)  # 1 = open, -1 = close
        
        # Constants
        self.approach_height = 0.12
        self.grasp_height = 0.0  # Go to cube center
        self.stack_height_1 = 0.05
        self.stack_height_2 = 0.10
        self.position_threshold = 0.03
        self.speed_scale = 3.0  # Speed multiplier

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = range(self.num_envs)
        
        self.sm_state[env_ids] = 0  # REST
        self.sm_wait_time[env_ids] = 0.1

    def compute(
        self,
        ee_pose: torch.Tensor,
        cube1_pose: torch.Tensor,
        cube2_pose: torch.Tensor,
        cube3_pose: torch.Tensor,
        default_ee_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the desired end-effector pose and gripper state.

        Args:
            ee_pose: Current end-effector pose (num_envs, 7).
            cube1_pose: Pose of cube 1 / blue (num_envs, 7).
            cube2_pose: Pose of cube 2 / red (num_envs, 7).
            cube3_pose: Pose of cube 3 / green (num_envs, 7).
            default_ee_quat: Default end-effector orientation (num_envs, 4).

        Returns:
            Desired end-effector pose and gripper state.
        """
        # Decrement wait time
        self.sm_wait_time -= self.dt
        
        # Get positions
        ee_pos = ee_pose[:, :3]
        cube1_pos = cube1_pose[:, :3]
        cube2_pos = cube2_pose[:, :3]
        cube3_pos = cube3_pose[:, :3]
        
        # Process each state
        for state_val in range(17):
            mask = self.sm_state == state_val
            if not mask.any():
                continue
                
            if state_val == 0:  # REST
                self.des_ee_pose[mask, :3] = ee_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_ee_quat[mask]
                self.des_gripper_state[mask] = 1.0
                transition = mask & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 1
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 1:  # APPROACH_ABOVE_CUBE2
                des_pos = cube2_pos.clone()
                des_pos[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_ee_quat[mask]
                self.des_gripper_state[mask] = 1.0
                dist = torch.norm(ee_pos - des_pos, dim=1)
                transition = mask & (dist < self.position_threshold) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 2
                self.sm_wait_time[transition] = 0.15
                
            elif state_val == 2:  # APPROACH_CUBE2
                des_pos = cube2_pos.clone()
                des_pos[:, 2] += self.grasp_height
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_ee_quat[mask]
                self.des_gripper_state[mask] = 1.0
                dist = torch.norm(ee_pos - des_pos, dim=1)
                transition = mask & (dist < self.position_threshold) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 3
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 3:  # GRASP_CUBE2
                self.des_gripper_state[mask] = -1.0
                transition = mask & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 4
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 4:  # LIFT_CUBE2
                des_pos = ee_pos.clone()
                des_pos[:, 2] = self.approach_height  # Fixed lift height
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_gripper_state[mask] = -1.0
                transition = mask & (ee_pos[:, 2] > self.approach_height - 0.02) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 5
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 5:  # APPROACH_ABOVE_CUBE1
                des_pos = cube1_pos.clone()
                des_pos[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_gripper_state[mask] = -1.0
                dist = torch.norm(ee_pos - des_pos, dim=1)
                transition = mask & (dist < self.position_threshold) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 6
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 6:  # PLACE_CUBE2_ON_CUBE1
                des_pos = cube1_pos.clone()
                des_pos[:, 2] += self.stack_height_1 + self.grasp_height
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_gripper_state[mask] = -1.0
                dist = torch.norm(ee_pos - des_pos, dim=1)
                transition = mask & (dist < self.position_threshold) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 7
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 7:  # RELEASE_CUBE2
                self.des_gripper_state[mask] = 1.0
                transition = mask & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 8
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 8:  # RETREAT_FROM_STACK
                des_pos = cube1_pos.clone()
                des_pos[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_gripper_state[mask] = 1.0
                dist = torch.norm(ee_pos - des_pos, dim=1)
                transition = mask & (dist < self.position_threshold) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 9
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 9:  # APPROACH_ABOVE_CUBE3
                des_pos = cube3_pos.clone()
                des_pos[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_gripper_state[mask] = 1.0
                dist = torch.norm(ee_pos - des_pos, dim=1)
                transition = mask & (dist < self.position_threshold) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 10
                self.sm_wait_time[transition] = 0.15
                
            elif state_val == 10:  # APPROACH_CUBE3
                des_pos = cube3_pos.clone()
                des_pos[:, 2] += self.grasp_height
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_gripper_state[mask] = 1.0
                dist = torch.norm(ee_pos - des_pos, dim=1)
                transition = mask & (dist < self.position_threshold) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 11
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 11:  # GRASP_CUBE3
                self.des_gripper_state[mask] = -1.0
                transition = mask & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 12
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 12:  # LIFT_CUBE3
                des_pos = ee_pos.clone()
                des_pos[:, 2] = self.approach_height + self.stack_height_1  # Lift higher for stacking
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_gripper_state[mask] = -1.0
                transition = mask & (ee_pos[:, 2] > self.approach_height + self.stack_height_1 - 0.02) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 13
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 13:  # APPROACH_ABOVE_STACK
                des_pos = cube1_pos.clone()
                des_pos[:, 2] += self.approach_height + self.stack_height_1
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_gripper_state[mask] = -1.0
                dist = torch.norm(ee_pos - des_pos, dim=1)
                transition = mask & (dist < self.position_threshold) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 14
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 14:  # PLACE_CUBE3_ON_STACK
                des_pos = cube1_pos.clone()
                des_pos[:, 2] += self.stack_height_2 + self.grasp_height
                self.des_ee_pose[mask, :3] = des_pos[mask]
                self.des_gripper_state[mask] = -1.0
                dist = torch.norm(ee_pos - des_pos, dim=1)
                transition = mask & (dist < self.position_threshold) & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 15
                self.sm_wait_time[transition] = 0.1
                
            elif state_val == 15:  # RELEASE_CUBE3
                self.des_gripper_state[mask] = 1.0
                transition = mask & (self.sm_wait_time <= 0)
                self.sm_state[transition] = 16
                self.sm_wait_time[transition] = 1.0
                
            elif state_val == 16:  # DONE
                self.des_gripper_state[mask] = 1.0
        
        return self.des_ee_pose.clone(), self.des_gripper_state.clone()


def main():
    """Main function."""
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        "Isaac-Stack-Cube-Franka-IK-Rel-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Disable terminations for continuous operation
    env_cfg.terminations.time_out = None
    
    # Create environment
    env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-v0", cfg=env_cfg).unwrapped
    
    # Create state machine
    sm = StackCubeStateMachine(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.num_envs,
        device=env.device,
    )
    
    # Default end-effector orientation (pointing down)
    default_ee_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    # Reset environment
    obs, _ = env.reset()
    sm.reset()
    
    # Demo collection
    demo_count = 0
    episode_data = []
    
    print(f"[INFO] Starting automatic demo collection...")
    print(f"[INFO] Target: {args_cli.num_demos} demos")
    
    # Main loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get cube poses from environment
            cube1 = env.scene["cube_1"]
            cube2 = env.scene["cube_2"]
            cube3 = env.scene["cube_3"]
            ee_frame = env.scene["ee_frame"]
            
            cube1_pose = torch.cat([
                cube1.data.root_pos_w - env.scene.env_origins,
                cube1.data.root_quat_w
            ], dim=-1)
            cube2_pose = torch.cat([
                cube2.data.root_pos_w - env.scene.env_origins,
                cube2.data.root_quat_w
            ], dim=-1)
            cube3_pose = torch.cat([
                cube3.data.root_pos_w - env.scene.env_origins,
                cube3.data.root_quat_w
            ], dim=-1)
            ee_pose = torch.cat([
                ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins,
                ee_frame.data.target_quat_w[:, 0, :]
            ], dim=-1)
            
            # Compute state machine action
            des_ee_pose, gripper_state = sm.compute(
                ee_pose, cube1_pose, cube2_pose, cube3_pose, default_ee_quat
            )
            
            # Convert to action format [dx, dy, dz, droll, dpitch, dyaw, gripper]
            delta_pos = (des_ee_pose[:, :3] - ee_pose[:, :3]) * sm.speed_scale
            actions = torch.cat([
                delta_pos,
                torch.zeros(env.num_envs, 3, device=env.device),
                gripper_state.unsqueeze(-1)
            ], dim=-1)
            
            # Apply actions
            obs, _, terminated, truncated, info = env.step(actions)
            
            # Check for completed demos (state == DONE)
            done_mask = sm.sm_state == 16
            done_envs = done_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if len(done_envs) > 0:
                demo_count += len(done_envs)
                print(f"[INFO] Completed {demo_count} / {args_cli.num_demos} demos")
                
                # Reset completed environments
                sm.reset(done_envs.tolist())
                
                if demo_count >= args_cli.num_demos:
                    print(f"[INFO] Target reached! Collected {demo_count} demos.")
                    break
    
    # Close environment
    env.close()
    print(f"[INFO] Demo collection complete. Total demos: {demo_count}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

