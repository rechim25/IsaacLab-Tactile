# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
TacEx Pick and Place Basket State Machine with Full Data Recording

A state machine that picks up a cube and places it in a basket using smooth,
natural trajectories with TacEx tactile sensing and full data recording
for imitation learning.

Records: video, joints, tactile RGB, tactile forces, actions.

Usage:
    ./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_tacex_sm.py \
        --num_envs 4 --num_demos 100 --enable_cameras \
        --save_demos --output_file ./datasets/pick_place_basket_tacex.hdf5
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TacEx Pick and Place Basket with full recording.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_demos", type=int, default=100)
parser.add_argument("--save_demos", action="store_true")
parser.add_argument("--output_file", type=str, default="./datasets/pick_place_basket_tacex.hdf5")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import h5py
import os
import numpy as np
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _compute_pseudo_force_geometric(height_map: torch.Tensor) -> torch.Tensor:
    """Compute normalized 3D pseudo-force from height map (geometric method).
    
    Args:
        height_map: Height map tensor of shape (H, W) in mm.
        
    Returns:
        Normalized pseudo-force vector of shape (3,) as [Fx, Fy, Fz].
        Fx, Fy in [-1, 1] (shear direction), Fz in [0, 1] (normal force).
    """
    device = height_map.device
    baseline = height_map.max()
    deformation = (baseline - height_map).clamp(min=0)
    
    # Normal force (Fz) - total deformation
    Fz_raw = deformation.sum()
    
    # Shear forces (Fx, Fy) from center of pressure offset
    H, W = height_map.shape
    y_coords = torch.linspace(-1, 1, H, device=device).view(H, 1)
    x_coords = torch.linspace(-1, 1, W, device=device).view(1, W)
    
    total_def = Fz_raw + 1e-6
    Fx_raw = (deformation * x_coords).sum() / total_def
    Fy_raw = (deformation * y_coords).sum() / total_def
    
    # Normalize each component
    Fx = Fx_raw.clamp(-1, 1)
    Fy = Fy_raw.clamp(-1, 1)
    
    # Fz: normalize by estimated max deformation
    max_deformation = H * W * 0.5
    Fz = (Fz_raw / max_deformation).clamp(0, 1)
    
    return torch.stack([Fx, Fy, Fz])


def _compute_pseudo_force_photometric(tactile_rgb: torch.Tensor) -> torch.Tensor:
    """Compute 3D pseudo-force from tactile RGB (photometric method).
    
    Args:
        tactile_rgb: Tactile RGB tensor of shape (H, W, 3) in [0, 1].
        
    Returns:
        Pseudo-force vector of shape (3,) as [Fx, Fy, Fz].
    """
    rgb_centered = tactile_rgb - 0.5
    
    grad_x = rgb_centered[..., 0].mean()
    grad_y = rgb_centered[..., 1].mean()
    rgb_deviation = rgb_centered.abs().mean()
    
    shear_scale = 10.0
    normal_scale = 100.0
    
    Fx = grad_x * rgb_deviation * shear_scale
    Fy = grad_y * rgb_deviation * normal_scale
    Fz = rgb_deviation * normal_scale
    
    return torch.stack([Fx, Fy, Fz])


class PickPlaceBasketStateMachine:
    """
    A state machine for the pick and place basket task with smooth natural motion.
    """
    
    STATE_INIT = 0
    STATE_APPROACH_CUBE = 1
    STATE_DESCEND_CUBE = 2
    STATE_GRASP = 3
    STATE_LIFT = 4
    STATE_TRANSPORT = 5
    STATE_DESCEND_BASKET = 6
    STATE_RELEASE = 7
    STATE_RETREAT = 8
    STATE_DONE = 9
    
    def __init__(self, dt: float, num_envs: int, device: torch.device):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        
        self.sm_state = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros(num_envs, device=device)
        self.interp_progress = torch.zeros(num_envs, device=device)
        self.interp_start = torch.zeros(num_envs, 3, device=device)
        self.interp_target = torch.zeros(num_envs, 3, device=device)
        self.des_ee_pose = torch.zeros(num_envs, 7, device=device)
        self.des_gripper_state = torch.ones(num_envs, device=device)
        
        # Height parameters
        self.approach_height = 0.12
        self.grasp_height = -0.015
        self.lift_height = 0.15
        self.arc_height = 0.22
        self.basket_drop_height = 0.08
        
        # Timing parameters
        self.approach_duration = 0.7
        self.descend_duration = 0.4
        self.grasp_duration = 0.2
        self.lift_duration = 0.35
        self.transport_duration = 0.9
        self.basket_descend_duration = 0.4
        self.release_duration = 0.2
        self.retreat_duration = 0.35
        
        self.blend_threshold = 0.92
        self.base_speed = 2.5
        self.threshold = 0.02
        
    def reset(self, env_ids=None, ee_pos=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        
        if isinstance(env_ids, list):
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids
            
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.05
        self.interp_progress[env_ids] = 0.0
        self.des_gripper_state[env_ids] = 1.0
        
        if ee_pos is not None and len(env_ids_tensor) > 0:
            self.des_ee_pose[env_ids_tensor, :3] = ee_pos[env_ids_tensor]
            self.des_ee_pose[env_ids_tensor, 3] = 1.0
            self.des_ee_pose[env_ids_tensor, 4:7] = 0.0
            self.interp_start[env_ids_tensor] = ee_pos[env_ids_tensor]
            self.interp_target[env_ids_tensor] = ee_pos[env_ids_tensor]
    
    def _smooth_step(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    
    def _arc_interpolate(self, start, end, t, arc_height):
        t_smooth = self._smooth_step(t)
        pos = start + (end - start) * t_smooth.unsqueeze(-1)
        base_z = start[:, 2] + (end[:, 2] - start[:, 2]) * t_smooth
        arc_offset = 4.0 * arc_height * t_smooth * (1.0 - t_smooth)
        pos[:, 2] = base_z + arc_offset
        return pos
    
    def _linear_interpolate(self, start, end, t):
        t_smooth = self._smooth_step(t)
        return start + (end - start) * t_smooth.unsqueeze(-1)
            
    def compute(self, ee_pose, cube_pose, basket_pose, default_quat):
        self.sm_wait_time -= self.dt
        ee_pos = ee_pose[:, :3]
        cube_pos = cube_pose[:, :3]
        basket_pos = basket_pose[:, :3]
        des_speed = torch.ones(self.num_envs, device=self.device) * self.base_speed
        
        for s in range(10):
            mask = self.sm_state == s
            if not mask.any():
                continue
            
            if s == self.STATE_INIT:
                self.des_ee_pose[mask, :3] = ee_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_APPROACH_CUBE
                    self.interp_progress[trans] = 0.0
                    self.interp_start[trans] = ee_pos[trans]
                    target = cube_pos.clone()
                    target[:, 2] += self.approach_height
                    self.interp_target[trans] = target[trans]
                
            elif s == self.STATE_APPROACH_CUBE:
                self.interp_progress[mask] += self.dt / self.approach_duration
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, self.interp_progress)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    self.sm_state[trans] = self.STATE_DESCEND_CUBE
                    self.interp_progress[trans] = 0.0
                    self.interp_start[trans] = interp_pos[trans]
                    target = cube_pos.clone()
                    target[:, 2] += self.grasp_height
                    self.interp_target[trans] = target[trans]
                
            elif s == self.STATE_DESCEND_CUBE:
                self.interp_progress[mask] += self.dt / self.descend_duration
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, self.interp_progress)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                trans = mask & (self.interp_progress >= 1.0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_GRASP
                    self.sm_wait_time[trans] = self.grasp_duration
                
            elif s == self.STATE_GRASP:
                target = cube_pos.clone()
                target[:, 2] += self.grasp_height
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_LIFT
                    self.interp_progress[trans] = 0.0
                    self.interp_start[trans] = target[trans]
                    lift_target = cube_pos.clone()
                    lift_target[:, 2] = self.lift_height
                    self.interp_target[trans] = lift_target[trans]
                
            elif s == self.STATE_LIFT:
                self.interp_progress[mask] += self.dt / self.lift_duration
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, self.interp_progress)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    self.sm_state[trans] = self.STATE_TRANSPORT
                    self.interp_progress[trans] = 0.0
                    self.interp_start[trans] = interp_pos[trans]
                    target = basket_pos.clone()
                    target[:, 2] += self.approach_height
                    self.interp_target[trans] = target[trans]
                
            elif s == self.STATE_TRANSPORT:
                self.interp_progress[mask] += self.dt / self.transport_duration
                arc_extra = self.arc_height - self.lift_height
                interp_pos = self._arc_interpolate(self.interp_start, self.interp_target, self.interp_progress, arc_extra)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    self.sm_state[trans] = self.STATE_DESCEND_BASKET
                    self.interp_progress[trans] = 0.0
                    self.interp_start[trans] = interp_pos[trans]
                    target = basket_pos.clone()
                    target[:, 2] += self.basket_drop_height
                    self.interp_target[trans] = target[trans]
                
            elif s == self.STATE_DESCEND_BASKET:
                self.interp_progress[mask] += self.dt / self.basket_descend_duration
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, self.interp_progress)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = -1.0
                trans = mask & (self.interp_progress >= 1.0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_RELEASE
                    self.sm_wait_time[trans] = self.release_duration
                
            elif s == self.STATE_RELEASE:
                target = basket_pos.clone()
                target[:, 2] += self.basket_drop_height
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_RETREAT
                    self.interp_progress[trans] = 0.0
                    self.interp_start[trans] = target[trans]
                    retreat_target = basket_pos.clone()
                    retreat_target[:, 2] += self.approach_height
                    self.interp_target[trans] = retreat_target[trans]
                
            elif s == self.STATE_RETREAT:
                self.interp_progress[mask] += self.dt / self.retreat_duration
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, self.interp_progress)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    self.sm_state[trans] = self.STATE_DONE
                
            elif s == self.STATE_DONE:
                self.des_ee_pose[mask, :3] = ee_pos[mask]
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                self.des_gripper_state[mask] = 1.0
        
        return self.des_ee_pose.clone(), self.des_gripper_state.clone(), des_speed


class DataRecorder:
    """Records video, joints, tactile, and actions for imitation learning."""
    
    def __init__(self, output_file: str, num_envs: int):
        self.output_file = output_file
        self.num_envs = num_envs
        self.buffers = {i: self._empty_buffer() for i in range(num_envs)}
        self.demo_count = 0
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        
        with h5py.File(output_file, "w") as f:
            f.create_group("data")
            f.attrs["format"] = "pick_place_basket_tacex"
    
    def _empty_buffer(self):
        return {
            "actions": [],
            "joint_pos": [],
            "joint_vel": [],
            "ee_pos": [],
            "ee_quat": [],
            "gripper_pos": [],
            "cube_pos": [],
            "cube_quat": [],
            "basket_pos": [],
            "basket_quat": [],
            "rgb_wrist": [],
            "rgb_table": [],
            "tactile_left": [],
            "tactile_right": [],
            "force_geometric_left": [],
            "force_geometric_right": [],
            "force_photometric_left": [],
            "force_photometric_right": [],
        }
    
    def add_step(self, env_id: int, data: dict):
        for key, val in data.items():
            if val is not None and key in self.buffers[env_id]:
                self.buffers[env_id][key].append(val)
    
    def save_episode(self, env_id: int) -> bool:
        buf = self.buffers[env_id]
        if len(buf["actions"]) < 10:
            self.buffers[env_id] = self._empty_buffer()
            return False
        
        with h5py.File(self.output_file, "a") as f:
            g = f["data"].create_group(f"demo_{self.demo_count}")
            
            for key, data in buf.items():
                if len(data) > 0:
                    arr = np.stack(data)
                    if "rgb" in key or "tactile" in key:
                        g.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
                    else:
                        g.create_dataset(key, data=arr)
            
            g.attrs["num_steps"] = len(buf["actions"])
            g.attrs["success"] = True
        
        self.demo_count += 1
        self.buffers[env_id] = self._empty_buffer()
        return True
    
    def finalize(self):
        with h5py.File(self.output_file, "a") as f:
            f.attrs["total_demos"] = self.demo_count


def main():
    # Try TacEx environment first
    try:
        env_name = "Isaac-Pick-Place-Basket-Franka-IK-Rel-TacEx-v0"
        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
        has_tacex = True
    except Exception:
        env_name = "Isaac-Pick-Place-Basket-Franka-IK-Rel-v0"
        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
        has_tacex = False
        print("[WARN] TacEx env not found, using standard env (no tactile)")
    
    env_cfg.terminations.time_out = None
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    
    sm = PickPlaceBasketStateMachine(env_cfg.sim.dt * env_cfg.decimation, env.num_envs, env.device)
    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    # Setup recorder
    recorder = DataRecorder(args_cli.output_file, env.num_envs) if args_cli.save_demos else None
    
    # Check available sensors
    has_wrist_cam = "wrist_cam" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    has_table_cam = "table_cam" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    has_gsmini_left = "gsmini_left" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    has_gsmini_right = "gsmini_right" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    
    print(f"[INFO] Environment: {env_name}")
    print(f"[INFO] Sensors: wrist_cam={has_wrist_cam}, table_cam={has_table_cam}, "
          f"gsmini_left={has_gsmini_left}, gsmini_right={has_gsmini_right}")
    
    obs, _ = env.reset()
    ee_frame = env.scene["ee_frame"]
    initial_ee_pos = ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins
    sm.reset(ee_pos=initial_ee_pos)
    
    demo_count = 0
    
    print(f"[INFO] Running {args_cli.num_demos} demos with {args_cli.num_envs} parallel environments")
    print(f"[INFO] Saving to: {args_cli.output_file}" if args_cli.save_demos else "[INFO] Not saving")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get scene data
            robot = env.scene["robot"]
            cube = env.scene["cube"]
            basket = env.scene["basket"]
            ee_frame = env.scene["ee_frame"]
            
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
            
            # Get robot base/root pose (for base-relative transforms)
            base_pos = robot.data.root_pos_w - env.scene.env_origins  # (num_envs, 3)
            base_quat = robot.data.root_quat_w  # (num_envs, 4)
            
            # Compute action
            des_pose, grip, speed = sm.compute(ee_pose, cube_pose, basket_pose, default_quat)
            delta = (des_pose[:, :3] - ee_pose[:, :3]) * speed.unsqueeze(-1)
            actions = torch.cat([delta, torch.zeros(env.num_envs, 3, device=env.device), grip.unsqueeze(-1)], -1)
            
            # Record data
            if recorder:
                for i in range(env.num_envs):
                    if sm.sm_state[i] < sm.STATE_DONE:
                        step_data = {
                            "actions": actions[i].cpu().numpy(),
                            "joint_pos": robot.data.joint_pos[i].cpu().numpy(),
                            "joint_vel": robot.data.joint_vel[i].cpu().numpy(),
                            "ee_pos": ee_pose[i, :3].cpu().numpy(),
                            "ee_quat": ee_pose[i, 3:7].cpu().numpy(),
                            "base_pos": base_pos[i].cpu().numpy(),  # Robot base position
                            "base_quat": base_quat[i].cpu().numpy(),  # Robot base quaternion (x,y,z,w)
                            "gripper_pos": robot.data.joint_pos[i, -2:].cpu().numpy(),
                            "cube_pos": cube_pose[i, :3].cpu().numpy(),
                            "cube_quat": cube_pose[i, 3:7].cpu().numpy(),
                            "basket_pos": basket_pose[i, :3].cpu().numpy(),
                            "basket_quat": basket_pose[i, 3:7].cpu().numpy(),
                        }
                        
                        # Camera data
                        if has_wrist_cam:
                            rgb = env.scene.sensors["wrist_cam"].data.output.get("rgb")
                            if rgb is not None and rgb.numel() > 0:
                                rgb_np = rgb[i].cpu().numpy()
                                if rgb_np.dtype in [np.float32, np.float64]:
                                    rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
                                step_data["rgb_wrist"] = rgb_np
                        
                        if has_table_cam:
                            rgb = env.scene.sensors["table_cam"].data.output.get("rgb")
                            if rgb is not None and rgb.numel() > 0:
                                rgb_np = rgb[i].cpu().numpy()
                                if rgb_np.dtype in [np.float32, np.float64]:
                                    rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
                                step_data["rgb_table"] = rgb_np
                        
                        # Tactile data (RGB images)
                        if has_gsmini_left:
                            tac = env.scene.sensors["gsmini_left"].data.output.get("tactile_rgb")
                            if tac is not None and tac.numel() > 0:
                                tac_np = tac[i].cpu().numpy()
                                if tac_np.dtype in [np.float32, np.float64]:
                                    if tac_np.max() <= 1.0:
                                        tac_np = (tac_np * 255).astype(np.uint8)
                                    else:
                                        tac_np = tac_np.astype(np.uint8)
                                step_data["tactile_left"] = tac_np
                        
                        if has_gsmini_right:
                            tac = env.scene.sensors["gsmini_right"].data.output.get("tactile_rgb")
                            if tac is not None and tac.numel() > 0:
                                tac_np = tac[i].cpu().numpy()
                                if tac_np.dtype in [np.float32, np.float64]:
                                    if tac_np.max() <= 1.0:
                                        tac_np = (tac_np * 255).astype(np.uint8)
                                    else:
                                        tac_np = tac_np.astype(np.uint8)
                                step_data["tactile_right"] = tac_np
                        
                        # Tactile pseudo-force - GEOMETRIC (from height_map)
                        if has_gsmini_left:
                            hmap = env.scene.sensors["gsmini_left"].data.output.get("height_map")
                            if hmap is not None and hmap.numel() > 0:
                                step_data["force_geometric_left"] = _compute_pseudo_force_geometric(hmap[i]).cpu().numpy()
                        
                        if has_gsmini_right:
                            hmap = env.scene.sensors["gsmini_right"].data.output.get("height_map")
                            if hmap is not None and hmap.numel() > 0:
                                step_data["force_geometric_right"] = _compute_pseudo_force_geometric(hmap[i]).cpu().numpy()
                        
                        # Tactile pseudo-force - PHOTOMETRIC (from tactile_rgb)
                        if has_gsmini_left:
                            tac_rgb = env.scene.sensors["gsmini_left"].data.output.get("tactile_rgb")
                            if tac_rgb is not None and tac_rgb.numel() > 0:
                                step_data["force_photometric_left"] = _compute_pseudo_force_photometric(tac_rgb[i]).cpu().numpy()
                        
                        if has_gsmini_right:
                            tac_rgb = env.scene.sensors["gsmini_right"].data.output.get("tactile_rgb")
                            if tac_rgb is not None and tac_rgb.numel() > 0:
                                step_data["force_photometric_right"] = _compute_pseudo_force_photometric(tac_rgb[i]).cpu().numpy()
                        
                        recorder.add_step(i, step_data)
            
            # Step environment
            obs, _, terminated, truncated, _ = env.step(actions)
            
            # Check for environment resets
            env_reset_mask = terminated | truncated
            env_reset_ids = env_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if len(env_reset_ids) > 0:
                ee_pos_fresh = ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins
                
                for env_id in env_reset_ids.tolist():
                    if sm.sm_state[env_id] >= sm.STATE_RELEASE:
                        if recorder and recorder.save_episode(env_id):
                            demo_count += 1
                            print(f"[INFO] Saved demo {demo_count}/{args_cli.num_demos}")
                        elif not recorder:
                            demo_count += 1
                            print(f"[INFO] Demo {demo_count}/{args_cli.num_demos} completed")
                
                sm.reset(env_reset_ids.tolist(), ee_pos_fresh)
            
            # Check for state machine DONE state
            done_envs = (sm.sm_state == sm.STATE_DONE).nonzero(as_tuple=False).squeeze(-1)
            
            if len(done_envs) > 0:
                ee_pos_current = ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins
                
                for env_id in done_envs.tolist():
                    if not env_reset_mask[env_id]:
                        if recorder and recorder.save_episode(env_id):
                            demo_count += 1
                            print(f"[INFO] Saved demo {demo_count}/{args_cli.num_demos}")
                        elif not recorder:
                            demo_count += 1
                            print(f"[INFO] Demo {demo_count}/{args_cli.num_demos} completed")
                
                sm.reset(done_envs.tolist(), ee_pos_current)
            
            if demo_count >= args_cli.num_demos:
                break
    
    if recorder:
        recorder.finalize()
        print(f"[INFO] Done! Saved {demo_count} demos to {args_cli.output_file}")
    else:
        print(f"[INFO] Done! {demo_count} demos completed")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

