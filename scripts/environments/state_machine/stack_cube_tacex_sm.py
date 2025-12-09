# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""TacEx StackCube state machine with HDF5 saving."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TacEx StackCube state machine.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_demos", type=int, default=100)
parser.add_argument("--save_demos", action="store_true")
parser.add_argument("--output_file", type=str, default="./datasets/tacex_demos.hdf5")
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
            
            if s == 0:  # REST
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = ee_pos[mask], 1.0
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
                t = mask & (self.sm_wait_time <= 0); self.sm_state[t], self.sm_wait_time[t] = 1, 0.1
            elif s == 1:  # ABOVE CUBE2
                dp = c2.clone(); dp[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3], self.des_ee_pose[mask, 3:7] = dp[mask], default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 2, 0.15
            elif s == 2:  # APPROACH CUBE2
                dp = c2.clone(); dp[:, 2] += self.grasp_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], 1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 3, 0.1
            elif s == 3:  # GRASP
                self.des_gripper_state[mask] = -1.0
                t = mask & (self.sm_wait_time <= 0); self.sm_state[t], self.sm_wait_time[t] = 4, 0.1
            elif s == 4:  # LIFT
                dp = ee_pos.clone(); dp[:, 2] = self.approach_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (ee_pos[:, 2] > self.approach_height - 0.02) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 5, 0.1
            elif s == 5:  # ABOVE CUBE1
                dp = c1.clone(); dp[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 6, 0.1
            elif s == 6:  # PLACE ON CUBE1
                dp = c1.clone(); dp[:, 2] += self.stack_height_1 + self.grasp_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 7, 0.1
            elif s == 7:  # RELEASE
                self.des_gripper_state[mask] = 1.0
                t = mask & (self.sm_wait_time <= 0); self.sm_state[t], self.sm_wait_time[t] = 8, 0.1
            elif s == 8:  # RETREAT
                dp = c1.clone(); dp[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], 1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 9, 0.1
            elif s == 9:  # ABOVE CUBE3
                dp = c3.clone(); dp[:, 2] += self.approach_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], 1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 10, 0.15
            elif s == 10:  # APPROACH CUBE3
                dp = c3.clone(); dp[:, 2] += self.grasp_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], 1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 11, 0.1
            elif s == 11:  # GRASP
                self.des_gripper_state[mask] = -1.0
                t = mask & (self.sm_wait_time <= 0); self.sm_state[t], self.sm_wait_time[t] = 12, 0.1
            elif s == 12:  # LIFT
                dp = ee_pos.clone(); dp[:, 2] = self.approach_height + self.stack_height_1
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (ee_pos[:, 2] > self.approach_height + self.stack_height_1 - 0.02) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 13, 0.1
            elif s == 13:  # ABOVE STACK
                dp = c1.clone(); dp[:, 2] += self.approach_height + self.stack_height_1
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 14, 0.1
            elif s == 14:  # PLACE ON STACK
                dp = c1.clone(); dp[:, 2] += self.stack_height_2 + self.grasp_height
                self.des_ee_pose[mask, :3], self.des_gripper_state[mask] = dp[mask], -1.0
                t = mask & (torch.norm(ee_pos - dp, dim=1) < self.threshold) & (self.sm_wait_time <= 0)
                self.sm_state[t], self.sm_wait_time[t] = 15, 0.1
            elif s == 15:  # RELEASE
                self.des_gripper_state[mask] = 1.0
                t = mask & (self.sm_wait_time <= 0); self.sm_state[t], self.sm_wait_time[t] = 16, 0.5
            elif s == 16:  # DONE
                self.des_gripper_state[mask] = 1.0
        
        return self.des_ee_pose.clone(), self.des_gripper_state.clone()


def main():
    try:
        env_name = "Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0"
        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
    except:
        env_name = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
    
    env_cfg.terminations.time_out = None
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    sm = StackCubeStateMachine(env_cfg.sim.dt * env_cfg.decimation, env.num_envs, env.device)
    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    # HDF5 setup
    if args_cli.save_demos:
        os.makedirs(os.path.dirname(args_cli.output_file) or ".", exist_ok=True)
        buffers = {i: {"actions": [], "ee_pos": []} for i in range(env.num_envs)}
        demo_idx = 0
    
    obs, _ = env.reset()
    sm.reset()
    demo_count = 0
    print(f"[INFO] Running {env_name}, target: {args_cli.num_demos} demos")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            c1, c2, c3 = env.scene["cube_1"], env.scene["cube_2"], env.scene["cube_3"]
            ee = env.scene["ee_frame"]
            c1p = torch.cat([c1.data.root_pos_w - env.scene.env_origins, c1.data.root_quat_w], -1)
            c2p = torch.cat([c2.data.root_pos_w - env.scene.env_origins, c2.data.root_quat_w], -1)
            c3p = torch.cat([c3.data.root_pos_w - env.scene.env_origins, c3.data.root_quat_w], -1)
            ee_pose = torch.cat([ee.data.target_pos_w[:, 0] - env.scene.env_origins, ee.data.target_quat_w[:, 0]], -1)
            
            des_pose, grip = sm.compute(ee_pose, c1p, c2p, c3p, default_quat)
            delta = (des_pose[:, :3] - ee_pose[:, :3]) * sm.speed_scale
            actions = torch.cat([delta, torch.zeros(env.num_envs, 3, device=env.device), grip.unsqueeze(-1)], -1)
            
            # Record
            if args_cli.save_demos:
                for i in range(env.num_envs):
                    if sm.sm_state[i] < 16:
                        buffers[i]["actions"].append(actions[i].cpu().numpy())
                        buffers[i]["ee_pos"].append(ee_pose[i, :3].cpu().numpy())
            
            obs, _, _, _, _ = env.step(actions)
            
            done = (sm.sm_state == 16).nonzero(as_tuple=False).squeeze(-1)
            if len(done) > 0:
                for i in done.tolist():
                    if args_cli.save_demos and len(buffers[i]["actions"]) > 0:
                        with h5py.File(args_cli.output_file, "a") as f:
                            if "data" not in f: f.create_group("data")
                            g = f["data"].create_group(f"demo_{demo_idx}")
                            g.create_dataset("actions", data=np.stack(buffers[i]["actions"]))
                            g.create_dataset("ee_pos", data=np.stack(buffers[i]["ee_pos"]))
                            g.attrs["success"] = True
                        demo_idx += 1
                        buffers[i] = {"actions": [], "ee_pos": []}
                    demo_count += 1
                print(f"[INFO] {demo_count}/{args_cli.num_demos} demos")
                sm.reset(done.tolist())
                if demo_count >= args_cli.num_demos: break
    
    env.close()
    print(f"[INFO] Done! {demo_count} demos" + (f" saved to {args_cli.output_file}" if args_cli.save_demos else ""))

if __name__ == "__main__":
    main()
    simulation_app.close()
