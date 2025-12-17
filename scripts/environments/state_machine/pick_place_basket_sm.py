# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Pick and Place Basket State Machine with Smooth Natural Motion

A state machine that picks up a cube and places it in a basket using smooth,
natural trajectories suitable for imitation learning.

Key features for ML-friendly demonstrations:
- Smooth interpolation between waypoints (no sudden position jumps)
- Arc-based transport trajectory (natural human-like motion)
- Velocity ramping (slow start/end, faster in middle)
- Time-based progression for consistent demonstrations

States:
    0: INIT            - Wait for simulation to settle
    1: APPROACH_CUBE   - Smoothly move above the cube
    2: DESCEND_CUBE    - Lower to grasp height
    3: GRASP           - Close gripper to grasp cube
    4: LIFT            - Lift cube up
    5: TRANSPORT       - Arc motion from cube to basket (smooth parabolic path)
    6: DESCEND_BASKET  - Lower cube into basket
    7: RELEASE         - Open gripper to release cube
    8: RETREAT         - Lift gripper back up
    9: DONE            - Task complete

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
    A state machine for the pick and place basket task with smooth natural motion.
    
    The robot picks up a cube from the table and places it into a basket using
    smooth interpolated trajectories suitable for imitation learning.
    
    Each environment runs independently with its own state and interpolation progress.
    """
    
    # State constants
    STATE_INIT = 0
    STATE_APPROACH_CUBE = 1
    STATE_DESCEND_CUBE = 2
    STATE_GRASP = 3
    STATE_LIFT = 4
    STATE_TRANSPORT = 5  # Arc motion from cube to basket
    STATE_DESCEND_BASKET = 6
    STATE_RELEASE = 7
    STATE_RETREAT = 8
    STATE_DONE = 9
    
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
        
        # Interpolation progress [0, 1] for smooth motion within each state
        self.interp_progress = torch.zeros(num_envs, device=device)
        
        # Start and target positions for interpolation
        self.interp_start = torch.zeros(num_envs, 3, device=device)
        self.interp_target = torch.zeros(num_envs, 3, device=device)
        
        # Desired end-effector pose [x, y, z, qw, qx, qy, qz]
        self.des_ee_pose = torch.zeros(num_envs, 7, device=device)
        
        # Gripper command: 1.0 = open, -1.0 = close
        self.des_gripper_state = torch.ones(num_envs, device=device)
        
        # Height parameters
        self.approach_height = 0.12   # Height above object for approach
        self.grasp_height = -0.015    # Grasp slightly below cube center for secure grip
        self.lift_height = 0.15       # Height after lifting cube
        self.arc_height = 0.22        # Peak height during transport arc
        self.basket_drop_height = 0.08  # Height above basket to drop
        
        # Timing parameters (duration in seconds for each motion phase)
        # Slightly longer durations for smoother motion
        self.approach_duration = 0.7   # Time to approach cube
        self.descend_duration = 0.4    # Time to descend to cube
        self.grasp_duration = 0.2      # Time to close gripper
        self.lift_duration = 0.35      # Time to lift cube
        self.transport_duration = 0.9  # Time for arc transport
        self.basket_descend_duration = 0.4  # Time to descend into basket
        self.release_duration = 0.2    # Time to open gripper
        self.retreat_duration = 0.35   # Time to retreat
        
        # Blend factor: start next motion slightly before current finishes
        # This creates overlapping motion for smoother transitions
        self.blend_threshold = 0.92  # Start transition at 92% completion
        
        # Speed parameters for IK control
        self.base_speed = 2.5  # Base speed multiplier
        
        # Position threshold for backup transition
        self.threshold = 0.02
        
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
        self.sm_wait_time[env_ids] = 0.05  # Brief settle time
        self.interp_progress[env_ids] = 0.0
        self.des_gripper_state[env_ids] = 1.0  # Start with gripper open
        
        # Reset des_ee_pose and interpolation to current position
        if ee_pos is not None and len(env_ids_tensor) > 0:
            self.des_ee_pose[env_ids_tensor, :3] = ee_pos[env_ids_tensor]
            self.des_ee_pose[env_ids_tensor, 3] = 1.0  # qw
            self.des_ee_pose[env_ids_tensor, 4:7] = 0.0  # qx, qy, qz
            self.interp_start[env_ids_tensor] = ee_pos[env_ids_tensor]
            self.interp_target[env_ids_tensor] = ee_pos[env_ids_tensor]
    
    def _smooth_step(self, t: torch.Tensor) -> torch.Tensor:
        """Compute quintic smooth step function for very smooth motion.
        
        Uses smootherstep: 6t⁵ - 15t⁴ + 10t³
        This has zero velocity AND zero acceleration at both endpoints,
        creating much smoother transitions than cubic smoothstep.
        
        Args:
            t: Progress value(s) in [0, 1]
            
        Returns:
            Smoothed progress value(s) in [0, 1]
        """
        t = torch.clamp(t, 0.0, 1.0)
        # Quintic smootherstep: 6t⁵ - 15t⁴ + 10t³
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    
    def _arc_interpolate(
        self, 
        start: torch.Tensor, 
        end: torch.Tensor, 
        t: torch.Tensor, 
        arc_height: float
    ) -> torch.Tensor:
        """Interpolate along a parabolic arc for natural transport motion.
        
        Creates a smooth arc that peaks at the midpoint, like how humans
        naturally move objects through space.
        
        Args:
            start: Start position [N, 3]
            end: End position [N, 3]
            t: Progress [N] in [0, 1]
            arc_height: Peak height of the arc
            
        Returns:
            Interpolated position [N, 3]
        """
        # Smooth the progress for natural acceleration
        t_smooth = self._smooth_step(t)
        
        # Linear interpolation for x, y
        pos = start + (end - start) * t_smooth.unsqueeze(-1)
        
        # Parabolic arc for z: peaks at t=0.5
        # arc_offset = 4 * h * t * (1 - t) where h is additional height
        base_z = start[:, 2] + (end[:, 2] - start[:, 2]) * t_smooth
        arc_offset = 4.0 * arc_height * t_smooth * (1.0 - t_smooth)
        pos[:, 2] = base_z + arc_offset
        
        return pos
    
    def _linear_interpolate(
        self, 
        start: torch.Tensor, 
        end: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Smooth linear interpolation with ease in/out.
        
        Args:
            start: Start position [N, 3]
            end: End position [N, 3]
            t: Progress [N] in [0, 1]
            
        Returns:
            Interpolated position [N, 3]
        """
        t_smooth = self._smooth_step(t)
        return start + (end - start) * t_smooth.unsqueeze(-1)
            
    def compute(
        self,
        ee_pose: torch.Tensor,
        cube_pose: torch.Tensor, 
        basket_pose: torch.Tensor,
        default_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the next desired end-effector pose, gripper command, and speed.
        
        Uses smooth interpolation for natural, ML-friendly trajectories.
        
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
        des_speed = torch.ones(self.num_envs, device=self.device) * self.base_speed
        
        # Process each state
        for s in range(10):
            mask = self.sm_state == s
            if not mask.any():
                continue
            
            if s == self.STATE_INIT:
                # INIT - Wait for simulation to settle, stay in place
                self.des_ee_pose[mask, :3] = ee_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                
                # Transition when wait expires
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_APPROACH_CUBE
                    self.interp_progress[trans] = 0.0
                    self.interp_start[trans] = ee_pos[trans]
                    target = cube_pos.clone()
                    target[:, 2] += self.approach_height
                    self.interp_target[trans] = target[trans]
                
            elif s == self.STATE_APPROACH_CUBE:
                # Smoothly move above the cube
                self.interp_progress[mask] += self.dt / self.approach_duration
                
                # Compute interpolated position
                interp_pos = self._linear_interpolate(
                    self.interp_start, self.interp_target, self.interp_progress
                )
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                
                # Transition early for smooth blending into next motion
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    self.sm_state[trans] = self.STATE_DESCEND_CUBE
                    # Don't reset progress to 0 - use remaining progress for smooth blend
                    self.interp_progress[trans] = 0.0
                    # Use current interpolated position as new start for seamless handoff
                    self.interp_start[trans] = interp_pos[trans]
                    target = cube_pos.clone()
                    target[:, 2] += self.grasp_height
                    self.interp_target[trans] = target[trans]
                
            elif s == self.STATE_DESCEND_CUBE:
                # Lower to grasp height with smooth motion
                self.interp_progress[mask] += self.dt / self.descend_duration
                
                interp_pos = self._linear_interpolate(
                    self.interp_start, self.interp_target, self.interp_progress
                )
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                
                # Need to fully complete descent before grasping
                trans = mask & (self.interp_progress >= 1.0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_GRASP
                    self.sm_wait_time[trans] = self.grasp_duration
                
            elif s == self.STATE_GRASP:
                # Close gripper while holding position at cube
                target = cube_pos.clone()
                target[:, 2] += self.grasp_height
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0  # Close gripper
                
                # Transition after wait - seamless start to lift
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_LIFT
                    self.interp_progress[trans] = 0.0
                    # Start from current target (at cube) for seamless motion
                    self.interp_start[trans] = target[trans]
                    lift_target = cube_pos.clone()
                    lift_target[:, 2] = self.lift_height
                    self.interp_target[trans] = lift_target[trans]
                
            elif s == self.STATE_LIFT:
                # Lift cube up smoothly
                self.interp_progress[mask] += self.dt / self.lift_duration
                
                interp_pos = self._linear_interpolate(
                    self.interp_start, self.interp_target, self.interp_progress
                )
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0
                
                # Transition early for smooth blend into arc transport
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    self.sm_state[trans] = self.STATE_TRANSPORT
                    self.interp_progress[trans] = 0.0
                    # Use current interpolated position for seamless handoff
                    self.interp_start[trans] = interp_pos[trans]
                    # Target is above the basket at approach height (will descend after)
                    target = basket_pos.clone()
                    target[:, 2] += self.approach_height
                    self.interp_target[trans] = target[trans]
                
            elif s == self.STATE_TRANSPORT:
                # Arc motion from cube to basket - the natural carrying trajectory
                self.interp_progress[mask] += self.dt / self.transport_duration
                
                # Compute additional arc height for smooth parabolic motion
                arc_extra = self.arc_height - self.lift_height
                
                interp_pos = self._arc_interpolate(
                    self.interp_start, self.interp_target, 
                    self.interp_progress, arc_extra
                )
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0
                
                # Blend early into descent for continuous motion
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    self.sm_state[trans] = self.STATE_DESCEND_BASKET
                    self.interp_progress[trans] = 0.0
                    # Use current interpolated position for seamless handoff
                    self.interp_start[trans] = interp_pos[trans]
                    target = basket_pos.clone()
                    target[:, 2] += self.basket_drop_height
                    self.interp_target[trans] = target[trans]
                
            elif s == self.STATE_DESCEND_BASKET:
                # Final descent into basket
                self.interp_progress[mask] += self.dt / self.basket_descend_duration
                
                interp_pos = self._linear_interpolate(
                    self.interp_start, self.interp_target, self.interp_progress
                )
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0
                
                # Transition when complete
                trans = mask & (self.interp_progress >= 1.0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_RELEASE
                    self.sm_wait_time[trans] = self.release_duration
                
            elif s == self.STATE_RELEASE:
                # Open gripper while holding position
                target = basket_pos.clone()
                target[:, 2] += self.basket_drop_height
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0  # Open gripper
                
                # Transition after wait - seamless start to retreat
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_RETREAT
                    self.interp_progress[trans] = 0.0
                    # Start from drop position for seamless motion
                    self.interp_start[trans] = target[trans]
                    retreat_target = basket_pos.clone()
                    retreat_target[:, 2] += self.approach_height
                    self.interp_target[trans] = retreat_target[trans]
                
            elif s == self.STATE_RETREAT:
                # Smoothly retreat upward
                self.interp_progress[mask] += self.dt / self.retreat_duration
                
                interp_pos = self._linear_interpolate(
                    self.interp_start, self.interp_target, self.interp_progress
                )
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                
                # Transition early to avoid abrupt stop at end
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    self.sm_state[trans] = self.STATE_DONE
                
            elif s == self.STATE_DONE:
                # Task complete - stay in place
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
                    # Only count as demo if state machine was close to completion (state >= RELEASE)
                    if sm.sm_state[env_id] >= sm.STATE_RELEASE:
                        demo_count += 1
                        print(f"[INFO] Demo {demo_count}/{args_cli.num_demos} completed (env {env_id})")
                
                # Reset state machine for environments that were reset, passing current EE position
                sm.reset(env_reset_ids.tolist(), ee_pos_fresh)
            
            # Also check for state machine DONE state (backup in case termination doesn't fire)
            done_envs = (sm.sm_state == sm.STATE_DONE).nonzero(as_tuple=False).squeeze(-1)
            
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
