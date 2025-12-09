# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
TacEx StackCube state machine with full data recording for Pi0 training.
Records: video, joints, tactile, actions.

Usage:
    ./isaaclab.sh -p scripts/environments/state_machine/stack_cube_tacex_sm.py \
        --num_envs 8 --num_demos 100 --enable_cameras \
        --save_demos --output_file ./datasets/pi0_tacex.hdf5
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TacEx StackCube with full recording.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_demos", type=int, default=100)
parser.add_argument("--save_demos", action="store_true")
parser.add_argument("--output_file", type=str, default="./datasets/pi0_tacex.hdf5")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import h5py
import os
import numpy as np
from collections.abc import Sequence
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class StackCubeStateMachine:
    def __init__(self, dt: float, num_envs: int, device):
        self.dt, self.num_envs, self.device = float(dt), num_envs, device
        self.sm_state = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros(num_envs, device=device)
        self.des_ee_pose = torch.zeros(num_envs, 7, device=device)
        self.des_gripper_state = torch.ones(num_envs, device=device)
        self.approach_height, self.grasp_height = 0.12, 0.0
        self.stack_height_1, self.stack_height_2 = 0.05, 0.10
        self.threshold, self.speed_scale = 0.03, 3.0

    def reset(self, env_ids=None):
        if env_ids is None: env_ids = range(self.num_envs)
        self.sm_state[env_ids], self.sm_wait_time[env_ids] = 0, 0.1

    def compute(self, ee_pose, c1_pose, c2_pose, c3_pose, default_quat):
        self.sm_wait_time -= self.dt
        ee_pos, c1, c2, c3 = ee_pose[:, :3], c1_pose[:, :3], c2_pose[:, :3], c3_pose[:, :3]
        
        for s in range(17):
            mask = self.sm_state == s
            if not mask.any(): continue
            
            if s == 0:
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = ee_pos[mask], 1.0
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                t = mask & (self.sm_wait_time <= 0); self.sm_state[t], self.sm_wait_time[t] = 1, 0.1
            elif s == 1:
                dp = c2.clone(); dp[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3], self.des_ee_pose[mask, 3:7] = dp[mask], default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 2, 0.15
            elif s == 2:
                dp = c2.clone(); dp[:, 2] += self.grasp_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], 1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 3, 0.1
            elif s == 3:
                self.des_gripper_state[mask] = -1.0
                t = mask & (self.sm_wait_time <= 0); self.sm_state[t], self.sm_wait_time[t] = 4, 0.1
            elif s == 4:
                dp = ee_pos.clone(); dp[:, 2] = self.approach_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (ee_pos[:, 2] > self.approach_height - 0.02) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 5, 0.1
            elif s == 5:
                dp = c1.clone(); dp[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 6, 0.1
            elif s == 6:
                dp = c1.clone(); dp[:, 2] += self.stack_height_1 + self.grasp_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 7, 0.1
            elif s == 7:
                self.des_gripper_state[mask] = 1.0
                t = mask & (self.sm_wait_time <= 0); self.sm_state[t], self.sm_wait_time[t] = 8, 0.1
            elif s == 8:
                dp = c1.clone(); dp[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], 1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 9, 0.1
            elif s == 9:
                dp = c3.clone(); dp[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], 1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 10, 0.15
            elif s == 10:
                dp = c3.clone(); dp[:, 2] += self.grasp_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], 1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 11, 0.1
            elif s == 11:
                self.des_gripper_state[mask] = -1.0
                t = mask & (self.sm_wait_time <= 0); self.sm_state[t], self.sm_wait_time[t] = 12, 0.1
            elif s == 12:
                dp = ee_pos.clone(); dp[:, 2] = self.approach_height + self.stack_height_1
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (ee_pos[:, 2] > self.approach_height + self.stack_height_1 - 0.02) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 13, 0.1
            elif s == 13:
                dp = c1.clone(); dp[:, 2] += self.approach_height + self.stack_height_1
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 14, 0.1
            elif s == 14:
                dp = c1.clone(); dp[:, 2] += self.stack_height_2 + self.grasp_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 15, 0.1
            elif s == 15:
                self.des_gripper_state[mask] = 1.0
                t = mask & (self.sm_wait_time <= 0); self.sm_state[t], self.sm_wait_time[t] = 16, 0.5
            elif s == 16:
                self.des_gripper_state[mask] = 1.0
        
        return self.des_ee_pose.clone(), self.des_gripper_state.clone()


class DataRecorder:
    """Records video, joints, tactile, and actions for Pi0 training."""
    
    def __init__(self, output_file: str, num_envs: int):
        self.output_file = output_file
        self.num_envs = num_envs
        self.buffers = {i: self._empty_buffer() for i in range(num_envs)}
        self.demo_count = 0
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        
        # Initialize HDF5
        with h5py.File(output_file, "w") as f:
            f.create_group("data")
            f.attrs["format"] = "pi0_tacex"
    
    def _empty_buffer(self):
        return {
            "actions": [],
            "joint_pos": [],
            "joint_vel": [],
            "ee_pos": [],
            "ee_quat": [],
            "gripper_pos": [],
            "rgb_wrist": [],
            "rgb_table": [],
            "tactile_left": [],
            "tactile_right": [],
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
            
            # Save all data
            for key, data in buf.items():
                if len(data) > 0:
                    arr = np.stack(data)
                    # Compress images
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
    # Try TacEx environment
    try:
        env_name = "Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0"
        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
        has_tacex = True
    except:
        env_name = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
        has_tacex = False
        print("[WARN] TacEx env not found, using standard env (no tactile)")
    
    env_cfg.terminations.time_out = None
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    
    sm = StackCubeStateMachine(env_cfg.sim.dt * env_cfg.decimation, env.num_envs, env.device)
    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    # Setup recorder
    recorder = DataRecorder(args_cli.output_file, env.num_envs) if args_cli.save_demos else None
    
    # Check available sensors
    has_wrist_cam = "wrist_cam" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    has_table_cam = "table_cam" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    has_gsmini_left = "gsmini_left" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    has_gsmini_right = "gsmini_right" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    
    print(f"[INFO] Sensors: wrist_cam={has_wrist_cam}, table_cam={has_table_cam}, "
          f"gsmini_left={has_gsmini_left}, gsmini_right={has_gsmini_right}")
    
    obs, _ = env.reset()
    sm.reset()
    demo_count = 0
    
    print(f"[INFO] Running {env_name}, target: {args_cli.num_demos} demos")
    print(f"[INFO] Saving to: {args_cli.output_file}" if args_cli.save_demos else "[INFO] Not saving")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get scene data
            robot = env.scene["robot"]
            c1, c2, c3 = env.scene["cube_1"], env.scene["cube_2"], env.scene["cube_3"]
            ee = env.scene["ee_frame"]
            
            c1p = torch.cat([c1.data.root_pos_w - env.scene.env_origins, c1.data.root_quat_w], -1)
            c2p = torch.cat([c2.data.root_pos_w - env.scene.env_origins, c2.data.root_quat_w], -1)
            c3p = torch.cat([c3.data.root_pos_w - env.scene.env_origins, c3.data.root_quat_w], -1)
            ee_pose = torch.cat([ee.data.target_pos_w[:, 0] - env.scene.env_origins, ee.data.target_quat_w[:, 0]], -1)
            
            # Compute action
            des_pose, grip = sm.compute(ee_pose, c1p, c2p, c3p, default_quat)
            delta = (des_pose[:, :3] - ee_pose[:, :3]) * sm.speed_scale
            actions = torch.cat([delta, torch.zeros(env.num_envs, 3, device=env.device), grip.unsqueeze(-1)], -1)
            
            # Record data
            if recorder:
                for i in range(env.num_envs):
                    if sm.sm_state[i] < 16:  # Not done
                        step_data = {
                            "actions": actions[i].cpu().numpy(),
                            "joint_pos": robot.data.joint_pos[i].cpu().numpy(),
                            "joint_vel": robot.data.joint_vel[i].cpu().numpy(),
                            "ee_pos": ee_pose[i, :3].cpu().numpy(),
                            "ee_quat": ee_pose[i, 3:7].cpu().numpy(),
                            "gripper_pos": robot.data.joint_pos[i, -2:].cpu().numpy(),
                        }
                        
                        # Camera data
                        if has_wrist_cam:
                            rgb = env.scene.sensors["wrist_cam"].data.output.get("rgb")
                            if rgb is not None:
                                step_data["rgb_wrist"] = rgb[i].cpu().numpy().astype(np.uint8)
                        if has_table_cam:
                            rgb = env.scene.sensors["table_cam"].data.output.get("rgb")
                            if rgb is not None:
                                step_data["rgb_table"] = rgb[i].cpu().numpy().astype(np.uint8)
                        
                        # Tactile data
                        if has_gsmini_left:
                            tac = env.scene.sensors["gsmini_left"].data.output.get("tactile_rgb")
                            if tac is not None:
                                step_data["tactile_left"] = tac[i].cpu().numpy().astype(np.uint8)
                        if has_gsmini_right:
                            tac = env.scene.sensors["gsmini_right"].data.output.get("tactile_rgb")
                            if tac is not None:
                                step_data["tactile_right"] = tac[i].cpu().numpy().astype(np.uint8)
                        
                        recorder.add_step(i, step_data)
            
            # Step environment
            obs, _, _, _, _ = env.step(actions)
            
            # Check done
            done = (sm.sm_state == 16).nonzero(as_tuple=False).squeeze(-1)
            if len(done) > 0:
                for i in done.tolist():
                    if recorder and recorder.save_episode(i):
                        demo_count += 1
                        print(f"[INFO] Saved demo {demo_count}/{args_cli.num_demos}")
                    elif not recorder:
                        demo_count += 1
                        print(f"[INFO] {demo_count}/{args_cli.num_demos}")
                
                sm.reset(done.tolist())
                if demo_count >= args_cli.num_demos:
                    break
    
    if recorder:
        recorder.finalize()
        print(f"[INFO] Done! Saved {demo_count} demos to {args_cli.output_file}")
    else:
        print(f"[INFO] Done! {demo_count} demos")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
