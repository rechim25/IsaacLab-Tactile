# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
TacEx-enabled StackCube state machine for automatic demo generation.

Usage:
    ./isaaclab.sh -p scripts/environments/state_machine/stack_cube_tacex_sm.py --num_envs 8 --enable_cameras
    ./isaaclab.sh -p scripts/environments/state_machine/stack_cube_tacex_sm.py --num_envs 16 --headless --num_demos 100
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TacEx StackCube state machine.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments.")
parser.add_argument("--num_demos", type=int, default=100, help="Number of demos to collect.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from collections.abc import Sequence
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class StackCubeStateMachine:
    """State machine for stacking three cubes."""

    def __init__(self, dt: float, num_envs: int, device):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.sm_state = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros(num_envs, device=device)
        self.des_ee_pose = torch.zeros(num_envs, 7, device=device)
        self.des_gripper_state = torch.ones(num_envs, device=device)
        self.approach_height = 0.15
        self.grasp_height = 0.103  # EE height above cube center (accounts for gripper offset)
        self.stack_height_1 = 0.047
        self.stack_height_2 = 0.094
        self.threshold = 0.025

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = range(self.num_envs)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.1

    def compute(self, ee_pose, cube1_pose, cube2_pose, cube3_pose, default_quat):
        self.sm_wait_time -= self.dt
        ee_pos = ee_pose[:, :3]
        c1, c2, c3 = cube1_pose[:, :3], cube2_pose[:, :3], cube3_pose[:, :3]
        
        states = [
            (0, ee_pos, 1.0, 1, 0.3),  # REST
            (1, c2.clone().add_(torch.tensor([0, 0, self.approach_height], device=self.device)), 1.0, 2, 0.4),
            (2, c2.clone().add_(torch.tensor([0, 0, self.grasp_height], device=self.device)), 1.0, 3, 0.3),
            (3, None, -1.0, 4, 0.3),  # GRASP
            (4, c2.clone().add_(torch.tensor([0, 0, self.approach_height], device=self.device)), -1.0, 5, 0.3),
            (5, c1.clone().add_(torch.tensor([0, 0, self.approach_height], device=self.device)), -1.0, 6, 0.3),
            (6, c1.clone().add_(torch.tensor([0, 0, self.stack_height_1 + self.grasp_height], device=self.device)), -1.0, 7, 0.2),
            (7, None, 1.0, 8, 0.2),  # RELEASE
            (8, c1.clone().add_(torch.tensor([0, 0, self.approach_height], device=self.device)), 1.0, 9, 0.3),
            (9, c3.clone().add_(torch.tensor([0, 0, self.approach_height], device=self.device)), 1.0, 10, 0.4),
            (10, c3.clone().add_(torch.tensor([0, 0, self.grasp_height], device=self.device)), 1.0, 11, 0.3),
            (11, None, -1.0, 12, 0.3),  # GRASP
            (12, c3.clone().add_(torch.tensor([0, 0, self.approach_height], device=self.device)), -1.0, 13, 0.3),
            (13, c1.clone().add_(torch.tensor([0, 0, self.approach_height + self.stack_height_1], device=self.device)), -1.0, 14, 0.3),
            (14, c1.clone().add_(torch.tensor([0, 0, self.stack_height_2 + self.grasp_height], device=self.device)), -1.0, 15, 0.2),
            (15, None, 1.0, 16, 1.0),  # RELEASE
            (16, None, 1.0, 16, 1.0),  # DONE
        ]
        
        for s, des_pos, grip, next_s, wait in states:
            mask = self.sm_state == s
            if not mask.any():
                continue
            self.des_gripper_state[mask] = grip
            if des_pos is not None:
                self.des_ee_pose[mask, :3] = des_pos[mask] if des_pos.dim() == 2 else des_pos
                self.des_ee_pose[mask, 3:7] = default_quat[mask]
            if s in [3, 7, 11, 15, 16]:  # Wait-only states
                trans = mask & (self.sm_wait_time <= 0)
            else:
                dist = torch.norm(ee_pos - des_pos, dim=1) if des_pos is not None and des_pos.dim() == 2 else torch.zeros(self.num_envs, device=self.device)
                trans = mask & (dist < self.threshold) & (self.sm_wait_time <= 0)
            self.sm_state[trans] = next_s
            self.sm_wait_time[trans] = wait
        
        return self.des_ee_pose.clone(), self.des_gripper_state.clone()


def main():
    try:
        env_name = "Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0"
        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
    except Exception:
        env_name = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
    
    env_cfg.terminations.time_out = None
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    
    sm = StackCubeStateMachine(env_cfg.sim.dt * env_cfg.decimation, env.num_envs, env.device)
    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    obs, _ = env.reset()
    sm.reset()
    demo_count = 0
    
    print(f"[INFO] Running {env_name}, target: {args_cli.num_demos} demos")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            c1, c2, c3 = env.scene["cube_1"], env.scene["cube_2"], env.scene["cube_3"]
            ee = env.scene["ee_frame"]
            
            cube1_pose = torch.cat([c1.data.root_pos_w - env.scene.env_origins, c1.data.root_quat_w], -1)
            cube2_pose = torch.cat([c2.data.root_pos_w - env.scene.env_origins, c2.data.root_quat_w], -1)
            cube3_pose = torch.cat([c3.data.root_pos_w - env.scene.env_origins, c3.data.root_quat_w], -1)
            ee_pose = torch.cat([ee.data.target_pos_w[:, 0] - env.scene.env_origins, ee.data.target_quat_w[:, 0]], -1)
            
            des_pose, grip = sm.compute(ee_pose, cube1_pose, cube2_pose, cube3_pose, default_quat)
            delta = des_pose[:, :3] - ee_pose[:, :3]
            actions = torch.cat([delta, torch.zeros(env.num_envs, 3, device=env.device), grip.unsqueeze(-1)], -1)
            
            obs, _, _, _, _ = env.step(actions)
            
            done = (sm.sm_state == 16).nonzero(as_tuple=False).squeeze(-1)
            if len(done) > 0:
                demo_count += len(done)
                print(f"[INFO] {demo_count}/{args_cli.num_demos} demos")
                sm.reset(done.tolist())
                if demo_count >= args_cli.num_demos:
                    break
    
    env.close()
    print(f"[INFO] Done! {demo_count} demos")

if __name__ == "__main__":
    main()
    simulation_app.close()

