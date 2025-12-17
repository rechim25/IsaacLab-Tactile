# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Pick and Place Basket State Machine

A simple state machine that picks up a cube from a table and places it into a basket.

States:
    0: INIT            - Wait for simulation to settle
    1: APPROACH_CUBE   - Move above the cube
    2: DESCEND_CUBE    - Lower to grasp height
    3: GRASP           - Close gripper to grasp cube
    4: LIFT            - Lift cube straight up
    5: MOVE_HIGH       - Move to high clearance above basket (avoid IK issues)
    6: DESCEND_ABOVE   - Descend to approach height above basket
    7: DESCEND_BASKET  - Lower cube into basket
    8: RELEASE         - Open gripper to release cube
    9: RETREAT         - Lift gripper back up
   10: DONE            - Task complete

Usage:
    ./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_sm.py \
        --num_envs 4 --num_demos 10
"""

import argparse
from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Pick and Place Basket State Machine")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demos to run")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401 - registers gym environments
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class PickPlaceBasketStateMachine:
    """
    A state machine for the pick and place basket task.
    
    The robot picks up a cube from the table and places it into a basket.
    Each environment runs independently with its own state.
    """
    
    def __init__(self, dt: float, num_envs: int, device: torch.device):
        """
        Initialize the state machine.
        
        Args:
            dt: Simulation timestep (seconds)
            num_envs: Number of parallel environments
            device: Torch device (cuda/cpu)
        """
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        
        # State tracking for each environment
        self.sm_state = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros(num_envs, device=device)
        
        # Desired end-effector pose [x, y, z, qw, qx, qy, qz]
        self.des_ee_pose = torch.zeros(num_envs, 7, device=device)
        
        # Gripper command: 1.0 = open, -1.0 = close
        self.des_gripper_state = torch.ones(num_envs, device=device)
        
        # Height parameters - using higher values for more IK flexibility
        self.approach_height = 0.12  # Height above object for approach
        self.grasp_height = 0.0      # Grasp at cube center (0 offset)
        self.high_clearance = 0.20   # High clearance for safe transit (increased)
        self.basket_drop_height = 0.08  # Height above basket to drop cube
        
        # Motion parameters
        self.threshold = 0.03   # Position threshold (matching stack_cube)
        self.speed_scale = 3.0  # Movement speed multiplier
        self.slow_speed = 2.0   # Slower speed for precision moves near basket
        
    def reset(self, env_ids=None, ee_pos=None):
        """Reset the state machine for specified environments.
        
        Args:
            env_ids: List of environment indices to reset. None means all.
            ee_pos: Current end-effector positions [num_envs, 3]. If provided,
                   resets des_ee_pose to current position to prevent jumping.
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        
        # Convert to tensor for indexing if it's a list
        if isinstance(env_ids, list):
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids
            
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.1
        self.des_gripper_state[env_ids] = 1.0  # Start with gripper open
        
        # Reset des_ee_pose to current position to prevent jumping to old targets
        if ee_pos is not None and len(env_ids_tensor) > 0:
            self.des_ee_pose[env_ids_tensor, :3] = ee_pos[env_ids_tensor]
            # Reset orientation to default (pointing down)
            self.des_ee_pose[env_ids_tensor, 3] = 1.0  # qw
            self.des_ee_pose[env_ids_tensor, 4:7] = 0.0  # qx, qy, qz
            
    def compute(
        self,
        ee_pose: torch.Tensor,
        cube_pose: torch.Tensor, 
        basket_pose: torch.Tensor,
        default_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the next desired end-effector pose, gripper command, and speed.
        
        Following the same pattern as stack_cube_tacex_sm.py for reliability.
        
        Returns:
            des_ee_pose: Target EE pose [num_envs, 7]
            des_gripper_state: Gripper command [num_envs]
            des_speed: Speed multiplier [num_envs]
        """
        # Decrease wait timers
        self.sm_wait_time -= self.dt
        
        # Extract positions
        ee_pos = ee_pose[:, :3]
        cube_pos = cube_pose[:, :3]
        basket_pos = basket_pose[:, :3]
        
        # Default speed for all envs
        des_speed = torch.ones(self.num_envs, device=self.device) * self.speed_scale
        
        # Process each state (0-10)
        for s in range(11):
            mask = self.sm_state == s
            if not mask.any():
                continue
            
            if s == 0:
                # STATE 0: INIT - Wait for simulation to settle
                self.des_ee_pose[mask, :3] = ee_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                
                # Transition when wait expires
                t = mask & (self.sm_wait_time <= 0)
                self.sm_state[t] = 1
                self.sm_wait_time[t] = 0.1
                
            elif s == 1:
                # STATE 1: APPROACH_CUBE - Move above the cube
                target = cube_pos.clone()
                target[:, 2] += self.approach_height
                
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                
                # Transition when position reached
                t = mask & (torch.norm(ee_pos - target, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t] = 2
                self.sm_wait_time[t] = 0.15
                
            elif s == 2:
                # STATE 2: DESCEND_CUBE - Lower to grasp height
                target = cube_pos.clone()
                target[:, 2] += self.grasp_height
                
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                
                # Transition when position reached
                t = mask & (torch.norm(ee_pos - target, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t] = 3
                self.sm_wait_time[t] = 0.1
                
            elif s == 3:
                # STATE 3: GRASP - Close gripper (maintain position at cube)
                target = cube_pos.clone()
                target[:, 2] += self.grasp_height
                
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0  # Close gripper
                
                # Transition after wait
                t = mask & (self.sm_wait_time <= 0)
                self.sm_state[t] = 4
                self.sm_wait_time[t] = 0.1
                
            elif s == 4:
                # STATE 4: LIFT - Lift cube straight up to high clearance
                target = ee_pos.clone()
                target[:, 2] = self.high_clearance
                
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0  # Keep gripper closed
                
                # Transition when high enough
                t = mask & (ee_pos[:, 2] > self.high_clearance - 0.02) & (self.sm_wait_time <= 0)
                self.sm_state[t] = 5
                self.sm_wait_time[t] = 0.1
                
            elif s == 5:
                # STATE 5: MOVE_HIGH - Move horizontally to above basket at high clearance
                # This prevents IK issues when moving between positions
                # Use slower speed for smoother IK transitions
                des_speed[mask] = self.slow_speed
                
                target = basket_pos.clone()
                target[:, 2] = self.high_clearance
                
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0  # Keep gripper closed
                
                # Transition when position reached
                t = mask & (torch.norm(ee_pos - target, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t] = 6
                self.sm_wait_time[t] = 0.1
                
            elif s == 6:
                # STATE 6: DESCEND_ABOVE - Descend to approach height above basket
                # Use slower speed for precision
                des_speed[mask] = self.slow_speed
                
                target = basket_pos.clone()
                target[:, 2] += self.approach_height
                
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0  # Keep gripper closed
                
                # Transition when position reached
                t = mask & (torch.norm(ee_pos - target, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t] = 7
                self.sm_wait_time[t] = 0.15
                
            elif s == 7:
                # STATE 7: DESCEND_BASKET - Lower into basket
                # Use slower speed for final descent
                des_speed[mask] = self.slow_speed
                
                target = basket_pos.clone()
                target[:, 2] += self.basket_drop_height
                
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0  # Keep gripper closed
                
                # Transition when position reached
                t = mask & (torch.norm(ee_pos - target, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t] = 8
                self.sm_wait_time[t] = 0.1
                
            elif s == 8:
                # STATE 8: RELEASE - Open gripper (maintain position)
                target = basket_pos.clone()
                target[:, 2] += self.basket_drop_height
                
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0  # Open gripper
                
                # Transition after wait
                t = mask & (self.sm_wait_time <= 0)
                self.sm_state[t] = 9
                self.sm_wait_time[t] = 0.3
                
            elif s == 9:
                # STATE 9: RETREAT - Lift back up
                target = basket_pos.clone()
                target[:, 2] += self.approach_height
                
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0  # Keep gripper open
                
                # Transition when high enough
                t = mask & (ee_pos[:, 2] > self.approach_height - 0.02) & (self.sm_wait_time <= 0)
                self.sm_state[t] = 10
                self.sm_wait_time[t] = 0.2
                
            elif s == 10:
                # STATE 10: DONE - Task complete, stay in place
                self.des_ee_pose[mask, :3] = ee_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
        
        return self.des_ee_pose.clone(), self.des_gripper_state.clone(), des_speed


def main():
    """Main function to run the pick and place basket demo."""
    
    # Setup environment
    env_name = "Isaac-Pick-Place-Basket-Franka-IK-Rel-v0"
    print(f"[INFO] Loading environment: {env_name}")
    
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
    
    # Disable timeout so we control when episodes end
    env_cfg.terminations.time_out = None
    
    # Create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    
    # Create state machine
    dt = env_cfg.sim.dt * env_cfg.decimation
    sm = PickPlaceBasketStateMachine(dt, env.num_envs, env.device)
    
    # Default gripper orientation (pointing down)
    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    # Reset environment and state machine
    obs, _ = env.reset()
    
    # Get initial EE position to initialize state machine properly
    ee_frame = env.scene["ee_frame"]
    initial_ee_pos = ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins
    sm.reset(ee_pos=initial_ee_pos)
    
    demo_count = 0
    print(f"[INFO] Running {args_cli.num_demos} demos with {args_cli.num_envs} parallel environments")
    
    # Main simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get current poses from the scene
            cube = env.scene["cube"]
            basket = env.scene["basket"]
            ee_frame = env.scene["ee_frame"]
            
            # Build pose tensors (position + quaternion)
            # Subtract env_origins to get positions relative to each environment
            cube_pose = torch.cat([
                cube.data.root_pos_w - env.scene.env_origins,
                cube.data.root_quat_w
            ], dim=-1)
            
            basket_pose = torch.cat([
                basket.data.root_pos_w - env.scene.env_origins,
                basket.data.root_quat_w
            ], dim=-1)
            
            ee_pose = torch.cat([
                ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins,
                ee_frame.data.target_quat_w[:, 0]
            ], dim=-1)
            
            # Compute desired pose, gripper command, and speed from state machine
            des_pose, des_gripper, des_speed = sm.compute(ee_pose, cube_pose, basket_pose, default_quat)
            
            # Convert to action (delta position + delta orientation + gripper)
            # The IK controller expects relative movements
            # Use per-environment speed (slower near basket for smoother IK)
            delta_pos = (des_pose[:, :3] - ee_pose[:, :3]) * des_speed.unsqueeze(-1)
            delta_rot = torch.zeros(env.num_envs, 3, device=env.device)  # No rotation change
            
            actions = torch.cat([delta_pos, delta_rot, des_gripper.unsqueeze(-1)], dim=-1)
            
            # Step the environment
            obs, _, terminated, truncated, _ = env.step(actions)
            
            # Detect environments that were reset by the environment (success/failure termination)
            # These need their state machine reset to sync with the new object positions
            env_reset_mask = terminated | truncated
            env_reset_ids = env_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if len(env_reset_ids) > 0:
                # Get fresh EE position after environment reset
                ee_pos_fresh = ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins
                
                for env_id in env_reset_ids.tolist():
                    # Only count as demo if state machine was close to completion (state >= 8)
                    if sm.sm_state[env_id] >= 8:
                        demo_count += 1
                        print(f"[INFO] Demo {demo_count}/{args_cli.num_demos} completed (env {env_id})")
                
                # Reset state machine for environments that were reset, passing current EE position
                sm.reset(env_reset_ids.tolist(), ee_pos_fresh)
            
            # Also check for state machine DONE state (backup in case termination doesn't fire)
            done_envs = (sm.sm_state == 10).nonzero(as_tuple=False).squeeze(-1)
            
            if len(done_envs) > 0:
                ee_pos_current = ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins
                
                for env_id in done_envs.tolist():
                    # Only count if not already counted via env reset
                    if not env_reset_mask[env_id]:
                        demo_count += 1
                        print(f"[INFO] Demo {demo_count}/{args_cli.num_demos} completed (env {env_id})")
                
                # Reset completed environments
                sm.reset(done_envs.tolist(), ee_pos_current)
            
            # Check if we've completed enough demos
            if demo_count >= args_cli.num_demos:
                print(f"[INFO] Completed {demo_count} demos!")
                break
    
    # Cleanup
    env.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
    simulation_app.close()
