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
        --save_demos --output_dir ./datasets/pick_place_basket_tacex
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TacEx Pick and Place Basket with full recording.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_demos", type=int, default=100)
parser.add_argument("--save_demos", action="store_true")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./datasets/pick_place_basket_tacex",
    help="Output directory; HDF5 is saved as <output_dir>/data.hdf5.",
)
parser.add_argument(
    "--save_failed_videos",
    action="store_true",
    help="Save videos for failed episodes to dataset_name/unsuccessful_videos (default: disabled).",
)
parser.add_argument(
    "--background_mode",
    type=str,
    default="fixed",
    choices=["fixed", "random"],
    help="Background dome texture mode.",
)
parser.add_argument(
    "--background_texture",
    type=str,
    default="small_empty_house_4k.hdr",
    help="HDR texture path used when background_mode=fixed.",
)
AppLauncher.add_app_launcher_args(parser)
# Favor stable temporal quality by default for data collection videos.
parser.set_defaults(rendering_mode="balanced")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import h5py
import os
import numpy as np
import json
import cv2
from typing import Optional, Tuple
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR


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


INDOOR_HDR_TEXTURES = [
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
]


def _configure_background_texture(env, mode: str, fixed_texture: str) -> str:
    """Set a non-white dome texture background and return the chosen texture path."""
    def _resolve_texture_path(texture: str) -> str:
        # Allow bare HDR names and auto-resolve to standard indoor HDR location.
        if texture.endswith(".hdr") and "/" not in texture:
            return f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/{texture}"
        return texture

    if mode == "random":
        idx = torch.randint(low=0, high=len(INDOOR_HDR_TEXTURES), size=(1,)).item()
        selected = INDOOR_HDR_TEXTURES[idx]
    else:
        selected = _resolve_texture_path(fixed_texture)

    try:
        light = env.scene["light"]
        light_prim = light.prims[0]
        texture_file_attr = light_prim.GetAttribute("inputs:texture:file")
        intensity_attr = light_prim.GetAttribute("inputs:intensity")
        color_attr = light_prim.GetAttribute("inputs:color")
        texture_file_attr.Set(selected)
        # Slightly brighter with neutral tint for better robot/background contrast.
        intensity_attr.Set(3800.0)
        color_attr.Set((0.78, 0.78, 0.78))
    except Exception as exc:
        print(f"[WARN] Failed to set background texture '{selected}': {exc}")
    return selected


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for quaternions in wxyz format."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_norm_wxyz(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)


def _yaw_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)


def _euler_xyz_to_quat_wxyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dim=-1,
    )


def _quat_slerp_wxyz(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Batch slerp in wxyz format with t shape (B,)."""
    q0 = _quat_norm_wxyz(q0)
    q1 = _quat_norm_wxyz(q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # Ensure shortest path.
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(-1.0, 1.0)

    t = t.unsqueeze(-1).clamp(0.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    use_lerp = sin_omega.abs() < 1e-5

    s0 = torch.sin((1.0 - t) * omega) / (sin_omega + 1e-8)
    s1 = torch.sin(t * omega) / (sin_omega + 1e-8)
    out_slerp = s0 * q0 + s1 * q1
    out_lerp = (1.0 - t) * q0 + t * q1
    out = torch.where(use_lerp, out_lerp, out_slerp)
    return _quat_norm_wxyz(out)


def _quat_to_rotvec_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (wxyz) to axis-angle rotation vector."""
    q = _quat_norm_wxyz(q)
    w = q[..., 0].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt((1.0 - w * w).clamp(min=0.0))
    axis = xyz / (sin_half.unsqueeze(-1) + 1e-8)
    rotvec = axis * angle.unsqueeze(-1)
    # Small-angle fallback.
    small = sin_half < 1e-5
    rotvec[small] = 2.0 * xyz[small]
    return rotvec


class PickPlaceBasketStateMachine:
    """
    A state machine for the pick and place basket task with smooth natural motion.
    """
    
    STATE_INIT = 0
    STATE_PRE_APPROACH_CUBE = 1
    STATE_ALIGN_ABOVE_CUBE = 2
    STATE_DESCEND_CUBE = 3
    STATE_PRE_GRASP_PAUSE = 4
    STATE_GRASP = 5
    STATE_GRASP_HOLD = 6
    STATE_LIFT_CLEAR = 7
    STATE_LIFT_ASCEND = 8
    STATE_CARRY_MID_1 = 9
    STATE_CARRY_MID_2 = 10
    STATE_PRE_PLACE_HOVER = 11
    STATE_DESCEND_BASKET = 12
    STATE_RELEASE = 13
    STATE_POST_RELEASE_PAUSE = 14
    STATE_RETREAT = 15
    STATE_DONE = 16
    NUM_STATES = 17
    PHASE_NAMES = {
        STATE_INIT: "init",
        STATE_PRE_APPROACH_CUBE: "pre_approach_cube",
        STATE_ALIGN_ABOVE_CUBE: "align_above_cube",
        STATE_DESCEND_CUBE: "descend_cube",
        STATE_PRE_GRASP_PAUSE: "pre_grasp_pause",
        STATE_GRASP: "grasp",
        STATE_GRASP_HOLD: "grasp_hold",
        STATE_LIFT_CLEAR: "lift_clear",
        STATE_LIFT_ASCEND: "lift_ascend",
        STATE_CARRY_MID_1: "carry_mid_1",
        STATE_CARRY_MID_2: "carry_mid_2",
        STATE_PRE_PLACE_HOVER: "pre_place_hover",
        STATE_DESCEND_BASKET: "descend_basket",
        STATE_RELEASE: "release",
        STATE_POST_RELEASE_PAUSE: "post_release_pause",
        STATE_RETREAT: "retreat",
        STATE_DONE: "done",
    }
    
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
        self.default_quat = torch.zeros(num_envs, 4, device=device)
        self.target_grasp_quat = torch.zeros(num_envs, 4, device=device)
        self.carry_mid_1 = torch.zeros(num_envs, 3, device=device)
        self.carry_mid_2 = torch.zeros(num_envs, 3, device=device)
        self.pre_place_hover = torch.zeros(num_envs, 3, device=device)
        self.approach_lateral = torch.zeros(num_envs, device=device)
        self.carry_lateral_1 = torch.zeros(num_envs, device=device)
        self.carry_lateral_2 = torch.zeros(num_envs, device=device)
        self.grasp_entry_offset = torch.zeros(num_envs, 2, device=device)
        self.progress_gain = torch.ones(num_envs, device=device)
        self.progress_shape = torch.ones(num_envs, device=device)
        self.duration_scale = torch.ones(num_envs, device=device)
        self.prev_ee_pos = torch.zeros(num_envs, 3, device=device)
        self.prev_cube_pos = torch.zeros(num_envs, 3, device=device)
        self.lift_ref_cube_z = torch.zeros(num_envs, device=device)
        self.filtered_pos = torch.zeros(num_envs, 3, device=device)
        self.filtered_quat = torch.zeros(num_envs, 4, device=device)
        self.filtered_speed = torch.ones(num_envs, device=device)
        self.effective_max_pos_step = torch.full((num_envs,), 0.012, device=device)
        self.filter_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.has_prev = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.grasp_stable_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.hover_align_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        # Height parameters
        self.pre_approach_height = 0.19
        self.align_height = 0.13
        self.grasp_height = -0.004
        self.min_grasp_target_z = 0.02
        self.lift_clear_height = 0.15
        self.lift_height = 0.205
        self.carry_height = 0.24
        self.pre_place_height = 0.15
        self.basket_drop_height = 0.075
        self.retreat_height = 0.18
        
        # Timing parameters
        self.pre_approach_duration = 1.5
        self.align_duration = 1.0
        self.descend_duration = 1.05
        self.pre_grasp_pause = 0.1
        self.grasp_duration = 0.75
        self.grasp_hold_min = 0.45
        self.lift_clear_duration = 0.8
        self.lift_duration = 1.0
        self.carry_mid_1_duration = 0.9
        self.carry_mid_2_duration = 0.9
        self.pre_place_hover_duration = 0.85
        self.basket_descend_duration = 0.95
        self.release_duration = 0.35
        self.post_release_pause = 0.2
        self.retreat_duration = 0.85
        
        self.blend_threshold = 0.98
        self.base_speed = 1.45
        self.carry_speed = 1.25
        self.near_contact_speed = 0.8
        self.threshold = 0.02
        self.orient_blend_start = 0.72
        self.orient_blend_end = 0.95
        # Motion profile randomization kept narrow for natural but non-snappy trajectories.
        # Approach/descend jitter tightened to eliminate left-right wobble during cube
        # approach while keeping transport-phase diversity.
        self.timing_jitter = 0.05
        self.profile_gain_min = 0.97
        self.profile_gain_max = 1.03
        self.profile_shape_min = 0.94
        self.profile_shape_max = 1.06
        self.midpoint_xy_jitter = 0.028
        self.midpoint_z_jitter = 0.012
        self.pre_place_xy_jitter = 0.014
        self.approach_lateral_jitter = 0.004
        self.grasp_entry_xy_offset = 0.003

        # Output trajectory smoothing (reduces snapping at phase boundaries)
        self.max_pos_step = 0.012
        self.max_pos_step_near = 0.005
        self.pos_filter_alpha = 0.7
        self.speed_filter_alpha = 0.22
        self.max_speed_step = 0.08
        self.quat_filter_alpha = 0.35

        # Distance-aware fine-motion control near cube.
        # Below near_cube_dist the step cap and speed are linearly reduced toward
        # the *_near values; above far_cube_dist normal parameters are used.
        self.near_cube_dist = 0.04
        self.far_cube_dist = 0.12
        self.near_cube_speed = 0.45

        # Stability gates
        self.grasp_dist_threshold = 0.055
        self.grasp_motion_coupling_threshold = 0.012
        self.grasp_stable_required = 8
        self.clearance_margin = 0.06
        self.min_lift_delta_z = 0.03
        self.hover_xy_threshold = 0.022
        self.hover_align_required = 6

        # Orientation slack randomization (radians).
        # Tightened to reduce wrist wobble near cube, keeping just enough
        # variation for dataset diversity.
        self.yaw_slack = np.deg2rad(5.0)
        self.roll_slack = np.deg2rad(2.0)
        self.pitch_slack = np.deg2rad(2.0)
        
    def reset(self, env_ids=None, ee_pos=None, ee_quat=None):
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
        self.progress_gain[env_ids] = 1.0
        self.progress_shape[env_ids] = 1.0
        self.duration_scale[env_ids] = 1.0
        self.grasp_stable_count[env_ids] = 0
        self.hover_align_count[env_ids] = 0
        self.approach_lateral[env_ids] = 0.0
        self.carry_lateral_1[env_ids] = 0.0
        self.carry_lateral_2[env_ids] = 0.0
        self.grasp_entry_offset[env_ids] = 0.0
        self.lift_ref_cube_z[env_ids] = 0.0
        self.filtered_speed[env_ids] = self.base_speed
        self.filter_initialized[env_ids] = False
        self.has_prev[env_ids] = False
        
        if ee_pos is not None and len(env_ids_tensor) > 0:
            self.des_ee_pose[env_ids_tensor, :3] = ee_pos[env_ids_tensor]
            self.des_ee_pose[env_ids_tensor, 3] = 1.0
            self.des_ee_pose[env_ids_tensor, 4:7] = 0.0
            self.interp_start[env_ids_tensor] = ee_pos[env_ids_tensor]
            self.interp_target[env_ids_tensor] = ee_pos[env_ids_tensor]
            self.prev_ee_pos[env_ids_tensor] = ee_pos[env_ids_tensor]
            self.filtered_pos[env_ids_tensor] = ee_pos[env_ids_tensor]
        if ee_quat is not None and len(env_ids_tensor) > 0:
            self.default_quat[env_ids_tensor] = _quat_norm_wxyz(ee_quat[env_ids_tensor])
            self.target_grasp_quat[env_ids_tensor] = self.default_quat[env_ids_tensor]
            self.filtered_quat[env_ids_tensor] = self.default_quat[env_ids_tensor]
    
    def _smooth_step(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    
    def _arc_interpolate(self, start, end, t, arc_height, lateral_scale: Optional[torch.Tensor] = None):
        t_smooth = self._smooth_step(t)
        pos = start + (end - start) * t_smooth.unsqueeze(-1)
        base_z = start[:, 2] + (end[:, 2] - start[:, 2]) * t_smooth
        arc_offset = 4.0 * arc_height * t_smooth * (1.0 - t_smooth)
        pos[:, 2] = base_z + arc_offset
        if lateral_scale is not None:
            dir_xy = end[:, :2] - start[:, :2]
            dir_norm = torch.linalg.norm(dir_xy, dim=-1, keepdim=True).clamp(min=1e-6)
            unit_xy = dir_xy / dir_norm
            perp_xy = torch.stack([-unit_xy[:, 1], unit_xy[:, 0]], dim=-1)
            lateral_profile = 4.0 * t_smooth * (1.0 - t_smooth)
            pos[:, :2] = pos[:, :2] + perp_xy * (lateral_scale.unsqueeze(-1) * lateral_profile.unsqueeze(-1))
        return pos
    
    def _linear_interpolate(self, start, end, t):
        t_smooth = self._smooth_step(t)
        return start + (end - start) * t_smooth.unsqueeze(-1)

    def _sample_motion_profile(self, mask: torch.Tensor):
        if not mask.any():
            return
        n = int(mask.sum().item())
        self.progress_gain[mask] = self.profile_gain_min + (
            self.profile_gain_max - self.profile_gain_min
        ) * torch.rand(n, device=self.device)
        self.progress_shape[mask] = self.profile_shape_min + (
            self.profile_shape_max - self.profile_shape_min
        ) * torch.rand(n, device=self.device)
        self.duration_scale[mask] = (
            1.0 + (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.timing_jitter
        ).clamp(0.82, 1.22)

    def _profiled_progress(self) -> torch.Tensor:
        t = self.interp_progress.clamp(0.0, 1.0)
        p = self.progress_shape.clamp(min=0.2)
        num = torch.pow(t, p)
        den = num + torch.pow(1.0 - t, p) + 1e-8
        return (num / den).clamp(0.0, 1.0)

    def _start_motion_phase(
        self,
        trans: torch.Tensor,
        next_state: int,
        start: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        self.sm_state[trans] = next_state
        self.interp_progress[trans] = 0.0
        self.interp_start[trans] = start[trans]
        self.interp_target[trans] = target[trans]
        self._sample_motion_profile(trans)

    def _grasp_target_from_cube(self, cube_pos: torch.Tensor) -> torch.Tensor:
        target = cube_pos.clone()
        z = target[:, 2] + self.grasp_height
        z = torch.maximum(z, torch.full_like(z, self.min_grasp_target_z))
        target[:, 2] = z
        return target
            
    def _compute_target_grasp_quat(self, cube_quat: torch.Tensor, default_quat: torch.Tensor) -> torch.Tensor:
        # Align wrist yaw approximately with cube yaw and inject small random slack.
        # Assumes quaternions are in wxyz.
        w, x, y, z = cube_quat.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_noise = (torch.rand_like(yaw) * 2.0 - 1.0) * self.yaw_slack
        roll_noise = (torch.rand_like(yaw) * 2.0 - 1.0) * self.roll_slack
        pitch_noise = (torch.rand_like(yaw) * 2.0 - 1.0) * self.pitch_slack

        q_yaw = _yaw_quat_wxyz(yaw + yaw_noise)
        q_noise = _euler_xyz_to_quat_wxyz(roll_noise, pitch_noise, torch.zeros_like(yaw))
        return _quat_norm_wxyz(_quat_mul_wxyz(_quat_mul_wxyz(q_noise, q_yaw), default_quat))

    def _blended_quat(self, mask: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        # Blend from default to grasp orientation as we approach contact.
        p = (progress - self.orient_blend_start) / max(self.orient_blend_end - self.orient_blend_start, 1e-6)
        p = p.clamp(0.0, 1.0)
        return _quat_slerp_wxyz(self.default_quat[mask], self.target_grasp_quat[mask], p[mask])

    def compute(self, ee_pose, cube_pose, basket_pose, gripper_qpos: Optional[torch.Tensor] = None):
        self.sm_wait_time -= self.dt
        ee_pos = ee_pose[:, :3]
        cube_pos = cube_pose[:, :3]
        basket_pos = basket_pose[:, :3]
        des_speed = torch.ones(self.num_envs, device=self.device) * self.base_speed
        prof_t = self._profiled_progress()

        init_prev_mask = ~self.has_prev
        if init_prev_mask.any():
            self.prev_ee_pos[init_prev_mask] = ee_pos[init_prev_mask]
            self.prev_cube_pos[init_prev_mask] = cube_pos[init_prev_mask]
            self.has_prev[init_prev_mask] = True

        ee_delta = ee_pos - self.prev_ee_pos
        cube_delta = cube_pos - self.prev_cube_pos
        
        for s in range(self.NUM_STATES):
            mask = self.sm_state == s
            if not mask.any():
                continue
            
            if s == self.STATE_INIT:
                self.des_ee_pose[mask, :3] = ee_pos[mask]
                self.des_ee_pose[mask, 3:7] = self.default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    target = cube_pos.clone()
                    target[:, 2] += self.pre_approach_height
                    self._start_motion_phase(trans, self.STATE_PRE_APPROACH_CUBE, ee_pos, target)
                    self.approach_lateral[trans] = (
                        (torch.rand_like(self.approach_lateral[trans]) * 2.0 - 1.0)
                        * self.approach_lateral_jitter
                    )
                    self.target_grasp_quat[trans] = self._compute_target_grasp_quat(
                        cube_pose[trans, 3:7], self.default_quat[trans]
                    )
                
            elif s == self.STATE_PRE_APPROACH_CUBE:
                self.interp_progress[mask] += (
                    self.dt / self.pre_approach_duration
                ) * self.progress_gain[mask] / self.duration_scale[mask].clamp(min=0.2)
                prof_t = self._profiled_progress()
                interp_pos = self._arc_interpolate(
                    self.interp_start,
                    self.interp_target,
                    prof_t,
                    arc_height=0.012,
                    lateral_scale=self.approach_lateral,
                )
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = self._blended_quat(mask, prof_t)
                self.des_gripper_state[mask] = 1.0
                des_speed[mask] = self.base_speed * 0.95
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    target = cube_pos.clone()
                    target[:, 2] += self.align_height
                    self._start_motion_phase(trans, self.STATE_ALIGN_ABOVE_CUBE, interp_pos, target)

            elif s == self.STATE_ALIGN_ABOVE_CUBE:
                self.interp_progress[mask] += (
                    self.dt / self.align_duration
                ) * self.progress_gain[mask] / self.duration_scale[mask].clamp(min=0.2)
                prof_t = self._profiled_progress()
                interp_pos = self._arc_interpolate(
                    self.interp_start,
                    self.interp_target,
                    prof_t,
                    arc_height=0.006,
                    lateral_scale=0.5 * self.approach_lateral,
                )
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = self._blended_quat(mask, prof_t)
                self.des_gripper_state[mask] = 1.0
                des_speed[mask] = self.base_speed * 0.9
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    target = self._grasp_target_from_cube(cube_pos)
                    offset = (
                        (torch.rand((self.num_envs, 2), device=self.device) * 2.0 - 1.0)
                        * self.grasp_entry_xy_offset
                    )
                    self.grasp_entry_offset[trans] = offset[trans]
                    target[:, :2] = target[:, :2] + self.grasp_entry_offset
                    self._start_motion_phase(trans, self.STATE_DESCEND_CUBE, interp_pos, target)
                
            elif s == self.STATE_DESCEND_CUBE:
                self.interp_progress[mask] += (
                    self.dt / self.descend_duration
                ) * self.progress_gain[mask] / self.duration_scale[mask].clamp(min=0.2)
                prof_t = self._profiled_progress()
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, prof_t)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = self._blended_quat(mask, prof_t)
                self.des_gripper_state[mask] = 1.0
                des_speed[mask] = self.near_contact_speed
                trans = mask & (self.interp_progress >= 1.0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_PRE_GRASP_PAUSE
                    self.sm_wait_time[trans] = self.pre_grasp_pause
                
            elif s == self.STATE_PRE_GRASP_PAUSE:
                target = self._grasp_target_from_cube(cube_pos)
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = self.target_grasp_quat[mask]
                self.des_gripper_state[mask] = 1.0
                des_speed[mask] = self.near_contact_speed
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_GRASP
                    self.sm_wait_time[trans] = self.grasp_duration
                    self.grasp_stable_count[trans] = 0
                
            elif s == self.STATE_GRASP:
                target = self._grasp_target_from_cube(cube_pos)
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = self.target_grasp_quat[mask]
                self.des_gripper_state[mask] = -1.0
                des_speed[mask] = self.near_contact_speed
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_GRASP_HOLD
                    self.sm_wait_time[trans] = self.grasp_hold_min
                    self.grasp_stable_count[trans] = 0

            elif s == self.STATE_GRASP_HOLD:
                target = self._grasp_target_from_cube(cube_pos)
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = self.target_grasp_quat[mask]
                self.des_gripper_state[mask] = -1.0
                des_speed[mask] = self.near_contact_speed

                ee_cube_dist = torch.linalg.norm(cube_pos - ee_pos, dim=-1)
                coupling_err = torch.linalg.norm(cube_delta - ee_delta, dim=-1)
                if gripper_qpos is not None:
                    gripper_closed = torch.mean(gripper_qpos, dim=-1) < 0.026
                else:
                    gripper_closed = torch.ones_like(ee_cube_dist, dtype=torch.bool)
                stable = (
                    (ee_cube_dist < self.grasp_dist_threshold)
                    & (coupling_err < self.grasp_motion_coupling_threshold)
                    & gripper_closed
                )
                self.grasp_stable_count[mask] = torch.where(
                    stable[mask],
                    self.grasp_stable_count[mask] + 1,
                    torch.zeros_like(self.grasp_stable_count[mask]),
                )

                trans = (
                    mask
                    & (self.sm_wait_time <= 0)
                    & (self.grasp_stable_count >= self.grasp_stable_required)
                )
                if trans.any():
                    lift_target = cube_pos.clone()
                    lift_target[:, 2] = self.lift_clear_height
                    self._start_motion_phase(trans, self.STATE_LIFT_CLEAR, ee_pos, lift_target)

            elif s == self.STATE_LIFT_CLEAR:
                self.interp_progress[mask] += (
                    self.dt / self.lift_clear_duration
                ) * self.progress_gain[mask] / self.duration_scale[mask].clamp(min=0.2)
                prof_t = self._profiled_progress()
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, prof_t)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = self.target_grasp_quat[mask]
                self.des_gripper_state[mask] = -1.0
                des_speed[mask] = self.near_contact_speed
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    lift_target = cube_pos.clone()
                    lift_target[:, 2] = self.lift_height
                    lift_target[:, :2] = 0.82 * cube_pos[:, :2] + 0.18 * basket_pos[:, :2]
                    self.lift_ref_cube_z[trans] = cube_pos[trans, 2]
                    self._start_motion_phase(trans, self.STATE_LIFT_ASCEND, interp_pos, lift_target)
                
            elif s == self.STATE_LIFT_ASCEND:
                self.interp_progress[mask] += (
                    self.dt / self.lift_duration
                ) * self.progress_gain[mask] / self.duration_scale[mask].clamp(min=0.2)
                prof_t = self._profiled_progress()
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, prof_t)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = self.target_grasp_quat[mask]
                self.des_gripper_state[mask] = -1.0
                des_speed[mask] = self.near_contact_speed
                lift_delta_ok = cube_pos[:, 2] > (self.lift_ref_cube_z + self.min_lift_delta_z)
                fallback_ok = self.interp_progress >= 1.0
                clearance_ok = lift_delta_ok | fallback_ok
                trans = mask & (self.interp_progress >= self.blend_threshold) & clearance_ok
                if trans.any():
                    mid_1 = 0.45 * cube_pos + 0.55 * basket_pos
                    mid_1[:, 2] = self.carry_height + (
                        (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0)
                        * self.midpoint_z_jitter
                    )
                    xy_jitter = (torch.rand(self.num_envs, 2, device=self.device) * 2.0 - 1.0) * self.midpoint_xy_jitter
                    mid_1[:, :2] += xy_jitter
                    self.carry_lateral_1[trans] = (
                        (torch.rand_like(self.carry_lateral_1[trans]) * 2.0 - 1.0)
                        * (0.8 * self.midpoint_xy_jitter)
                    )
                    self.carry_mid_1[trans] = mid_1[trans]
                    self._start_motion_phase(trans, self.STATE_CARRY_MID_1, interp_pos, self.carry_mid_1)
                
            elif s == self.STATE_CARRY_MID_1:
                self.interp_progress[mask] += (
                    self.dt / self.carry_mid_1_duration
                ) * self.progress_gain[mask] / self.duration_scale[mask].clamp(min=0.2)
                prof_t = self._profiled_progress()
                arc_extra = (self.carry_height - self.lift_height) * 0.7
                interp_pos = self._arc_interpolate(
                    self.interp_start,
                    self.interp_target,
                    prof_t,
                    arc_extra,
                    lateral_scale=self.carry_lateral_1,
                )
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = self.target_grasp_quat[mask]
                self.des_gripper_state[mask] = -1.0
                des_speed[mask] = self.carry_speed
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    mid_2 = 0.2 * cube_pos + 0.8 * basket_pos
                    mid_2[:, 2] = self.carry_height + (
                        (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0)
                        * self.midpoint_z_jitter
                    )
                    xy_jitter = (torch.rand(self.num_envs, 2, device=self.device) * 2.0 - 1.0) * self.midpoint_xy_jitter
                    mid_2[:, :2] += xy_jitter * 0.75
                    self.carry_lateral_2[trans] = (
                        (torch.rand_like(self.carry_lateral_2[trans]) * 2.0 - 1.0)
                        * (0.6 * self.midpoint_xy_jitter)
                    )
                    self.carry_mid_2[trans] = mid_2[trans]
                    self._start_motion_phase(trans, self.STATE_CARRY_MID_2, interp_pos, self.carry_mid_2)

            elif s == self.STATE_CARRY_MID_2:
                self.interp_progress[mask] += (
                    self.dt / self.carry_mid_2_duration
                ) * self.progress_gain[mask] / self.duration_scale[mask].clamp(min=0.2)
                prof_t = self._profiled_progress()
                arc_extra = (self.carry_height - self.lift_height) * 0.55
                interp_pos = self._arc_interpolate(
                    self.interp_start,
                    self.interp_target,
                    prof_t,
                    arc_extra,
                    lateral_scale=self.carry_lateral_2,
                )
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = self.target_grasp_quat[mask]
                self.des_gripper_state[mask] = -1.0
                des_speed[mask] = self.carry_speed * 0.95
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    hover_target = basket_pos.clone()
                    hover_target[:, 2] = self.pre_place_height
                    hover_jitter = (torch.rand(self.num_envs, 2, device=self.device) * 2.0 - 1.0) * self.pre_place_xy_jitter
                    hover_target[:, :2] += hover_jitter
                    self.pre_place_hover[trans] = hover_target[trans]
                    self.hover_align_count[trans] = 0
                    self._start_motion_phase(trans, self.STATE_PRE_PLACE_HOVER, interp_pos, self.pre_place_hover)

            elif s == self.STATE_PRE_PLACE_HOVER:
                self.interp_progress[mask] += (
                    self.dt / self.pre_place_hover_duration
                ) * self.progress_gain[mask] / self.duration_scale[mask].clamp(min=0.2)
                prof_t = self._profiled_progress()
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, prof_t)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = self.target_grasp_quat[mask]
                self.des_gripper_state[mask] = -1.0
                des_speed[mask] = self.near_contact_speed * 1.05
                align_err = torch.linalg.norm(ee_pos[:, :2] - self.pre_place_hover[:, :2], dim=-1)
                align_ok = align_err < self.hover_xy_threshold
                self.hover_align_count[mask] = torch.where(
                    align_ok[mask],
                    self.hover_align_count[mask] + 1,
                    torch.zeros_like(self.hover_align_count[mask]),
                )
                trans = (
                    mask
                    & (self.interp_progress >= self.blend_threshold)
                    & (self.hover_align_count >= self.hover_align_required)
                )
                if trans.any():
                    target = self.pre_place_hover.clone()
                    target[:, 2] = basket_pos[:, 2] + self.basket_drop_height
                    self._start_motion_phase(trans, self.STATE_DESCEND_BASKET, interp_pos, target)
                
            elif s == self.STATE_DESCEND_BASKET:
                self.interp_progress[mask] += (
                    self.dt / self.basket_descend_duration
                ) * self.progress_gain[mask] / self.duration_scale[mask].clamp(min=0.2)
                prof_t = self._profiled_progress()
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, prof_t)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = self.target_grasp_quat[mask]
                self.des_gripper_state[mask] = -1.0
                des_speed[mask] = self.near_contact_speed
                trans = mask & (self.interp_progress >= 1.0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_RELEASE
                    self.sm_wait_time[trans] = self.release_duration
                
            elif s == self.STATE_RELEASE:
                target = self.interp_target.clone()
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = self.default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                des_speed[mask] = self.near_contact_speed
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    self.sm_state[trans] = self.STATE_POST_RELEASE_PAUSE
                    self.sm_wait_time[trans] = self.post_release_pause

            elif s == self.STATE_POST_RELEASE_PAUSE:
                target = self.interp_target.clone()
                self.des_ee_pose[mask, :3] = target[mask]
                self.des_ee_pose[mask, 3:7] = self.default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                trans = mask & (self.sm_wait_time <= 0)
                if trans.any():
                    retreat_target = basket_pos.clone()
                    retreat_target[:, 2] = self.retreat_height
                    self._start_motion_phase(trans, self.STATE_RETREAT, ee_pos, retreat_target)
                
            elif s == self.STATE_RETREAT:
                self.interp_progress[mask] += (
                    self.dt / self.retreat_duration
                ) * self.progress_gain[mask] / self.duration_scale[mask].clamp(min=0.2)
                prof_t = self._profiled_progress()
                interp_pos = self._linear_interpolate(self.interp_start, self.interp_target, prof_t)
                self.des_ee_pose[mask, :3] = interp_pos[mask]
                self.des_ee_pose[mask, 3:7] = self.default_quat[mask]
                self.des_gripper_state[mask] = 1.0
                des_speed[mask] = self.base_speed * 0.95
                trans = mask & (self.interp_progress >= self.blend_threshold)
                if trans.any():
                    self.sm_state[trans] = self.STATE_DONE
                
            elif s == self.STATE_DONE:
                self.des_ee_pose[mask, :3] = ee_pos[mask]
                self.des_ee_pose[mask, 3:7] = self.default_quat[mask]
                self.des_gripper_state[mask] = 1.0

        self.prev_ee_pos.copy_(ee_pos)
        self.prev_cube_pos.copy_(cube_pos)

        # Distance-aware fine-motion: reduce speed and step cap when close to
        # cube during approach/descend/grasp phases.
        ee_cube_dist = torch.linalg.norm(ee_pos - cube_pos, dim=-1)
        proximity_phases = (
            (self.sm_state == self.STATE_PRE_APPROACH_CUBE)
            | (self.sm_state == self.STATE_ALIGN_ABOVE_CUBE)
            | (self.sm_state == self.STATE_DESCEND_CUBE)
            | (self.sm_state == self.STATE_PRE_GRASP_PAUSE)
            | (self.sm_state == self.STATE_GRASP)
            | (self.sm_state == self.STATE_GRASP_HOLD)
        )
        prox_t = ((ee_cube_dist - self.near_cube_dist) / max(self.far_cube_dist - self.near_cube_dist, 1e-6)).clamp(0.0, 1.0)
        near_speed = self.near_cube_speed + (des_speed - self.near_cube_speed) * prox_t
        des_speed = torch.where(proximity_phases, near_speed, des_speed)
        near_step = self.max_pos_step_near + (self.max_pos_step - self.max_pos_step_near) * prox_t
        self.effective_max_pos_step = torch.where(proximity_phases, near_step, torch.full_like(near_step, self.max_pos_step))

        if (~self.filter_initialized).any():
            init_mask = ~self.filter_initialized
            self.filtered_pos[init_mask] = self.des_ee_pose[init_mask, :3]
            self.filtered_quat[init_mask] = _quat_norm_wxyz(self.des_ee_pose[init_mask, 3:7])
            self.filtered_speed[init_mask] = des_speed[init_mask]
            self.filter_initialized[init_mask] = True

        # Position smoothing with hard per-step cap and alpha blend.
        pos_error = self.des_ee_pose[:, :3] - self.filtered_pos
        pos_error = pos_error * self.pos_filter_alpha
        pos_norm = torch.linalg.norm(pos_error, dim=-1, keepdim=True).clamp(min=1e-8)
        step_cap = self.effective_max_pos_step.unsqueeze(-1)
        pos_scale = torch.minimum(torch.ones_like(pos_norm), step_cap / pos_norm)
        self.filtered_pos = self.filtered_pos + pos_error * pos_scale

        # Quaternion smoothing with slerp.
        quat_t = torch.full((self.num_envs,), self.quat_filter_alpha, device=self.device)
        self.filtered_quat = _quat_slerp_wxyz(self.filtered_quat, self.des_ee_pose[:, 3:7], quat_t)

        # Smooth speed to avoid abrupt acceleration spikes at state transitions.
        speed_err = (des_speed - self.filtered_speed) * self.speed_filter_alpha
        speed_err = torch.clamp(speed_err, -self.max_speed_step, self.max_speed_step)
        self.filtered_speed = self.filtered_speed + speed_err

        out_pose = self.des_ee_pose.clone()
        out_pose[:, :3] = self.filtered_pos
        out_pose[:, 3:7] = self.filtered_quat
        return out_pose, self.des_gripper_state.clone(), self.filtered_speed.clone()


def _check_success_from_buffer(buf: dict) -> bool:
    """Robust geometric success over the tail of the episode.

    We avoid requiring an exact final-step gripper opening because that created
    false negatives for visually successful demos.
    """
    if len(buf["cube_pos"]) == 0 or len(buf["basket_pos"]) == 0:
        return False

    cube_pos = np.asarray(buf["cube_pos"], dtype=np.float32)
    basket_pos = np.asarray(buf["basket_pos"], dtype=np.float32)
    num_steps = min(cube_pos.shape[0], basket_pos.shape[0])
    if num_steps == 0:
        return False

    # Evaluate on the final segment of the trajectory to tolerate tiny post-release
    # oscillations while still ensuring terminal success.
    tail = min(12, num_steps)
    cube_tail = cube_pos[-tail:]
    basket_tail = basket_pos[-tail:]
    xy_dist = np.linalg.norm(cube_tail[:, :2] - basket_tail[:, :2], axis=1)
    height_dist = cube_tail[:, 2] - basket_tail[:, 2]

    # Slightly relaxed bounds relative to the old strict check.
    in_basket = (xy_dist < 0.09) & (height_dist > -0.005) & (height_dist < 0.085)
    return bool(np.any(in_basket))


class DataRecorder:
    """Records video, joints, tactile, and actions for imitation learning."""
    
    def __init__(
        self,
        output_dir: str,
        num_envs: int,
        planner_params: dict,
        save_failed_videos: bool = False,
        policy_env_name: str = "Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0",
        teacher_env_name: str = "Isaac-Pick-Place-Basket-Franka-IK-Rel-TacEx-v0",
    ):
        self.num_envs = num_envs
        self.planner_params = planner_params
        self.save_failed_videos = bool(save_failed_videos)
        self.buffers = {i: self._empty_buffer() for i in range(num_envs)}
        self.demo_count = 0
        self.attempt_count = 0
        self.fail_count = 0
        self.output_dir = output_dir
        self.output_file = os.path.join(self.output_dir, "data.hdf5")
        self.video_dir = os.path.join(self.output_dir, "successful_videos")
        self.failed_video_dir = os.path.join(self.output_dir, "unsuccessful_videos")
        self.meta_dir = os.path.join(self.output_dir, "metadata")
        self.policy_env_name = policy_env_name
        self.teacher_env_name = teacher_env_name
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        if self.save_failed_videos:
            os.makedirs(self.failed_video_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        
        with h5py.File(self.output_file, "w") as f:
            f.create_group("data")
            f.attrs["format"] = "pick_place_basket_tacex"
            # Store explicit schema metadata so conversion/eval cannot silently mix control modes.
            f.attrs["env"] = self.policy_env_name
            f.attrs["teacher_env"] = self.teacher_env_name
            f.attrs["state_schema"] = "joint_state_9d:[arm_joint_pos(7),gripper_qpos(2)]"
            f.attrs["action_schema"] = "joint_action_8d:[arm_joint_pos_target_abs(7),gripper_cmd(1)]"
            f.attrs["teacher_control_mode"] = "ik_rel"
    
    def _empty_buffer(self):
        return {
            "actions": [],
            "teacher_actions_ik": [],
            "joint_pos": [],
            "joint_vel": [],
            "ee_pos": [],
            "ee_quat": [],
            "base_pos": [],
            "base_quat": [],
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
            "phase_id": [],
        }
    
    def add_step(self, env_id: int, data: dict):
        for key, val in data.items():
            if val is not None and key in self.buffers[env_id]:
                self.buffers[env_id][key].append(val)

    def _validate_temporal_consistency(self, buf: dict) -> Tuple[bool, str]:
        # Enforce obs_t/action_t recording consistency via equal per-step buffer lengths.
        required_keys = [
            "actions",
            "joint_pos",
            "joint_vel",
            "ee_pos",
            "ee_quat",
            "base_pos",
            "base_quat",
            "gripper_pos",
            "cube_pos",
            "basket_pos",
            "phase_id",
        ]
        lengths = {k: len(buf[k]) for k in required_keys}
        base = lengths["actions"]
        for k, v in lengths.items():
            if v != base:
                return False, f"length_mismatch:{k}={v},actions={base}"
        return True, "ok"

    def _write_episode_video(self, demo_id: int, buf: dict, is_success: bool) -> Optional[str]:
        # Build side-by-side visualization directly from recorded camera frames
        # (left: table camera, right: wrist camera) without geometric resizing.
        table_frames = buf["rgb_table"]
        wrist_frames = buf["rgb_wrist"]
        num_frames = max(len(table_frames), len(wrist_frames))
        if num_frames == 0:
            return None

        first_table = np.asarray(table_frames[0]) if len(table_frames) > 0 else None
        first_wrist = np.asarray(wrist_frames[0]) if len(wrist_frames) > 0 else None
        first = first_table if first_table is not None else first_wrist
        if first is None or first.ndim != 3:
            return None

        cam_height = int(first.shape[0])
        cam_width = int(first.shape[1])
        out_height = cam_height
        out_width = cam_width * 2

        def _normalize_rgb_frame(frame: np.ndarray) -> np.ndarray:
            img = frame
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim != 3:
                return np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
            if img.shape[-1] == 4:
                img = img[..., :3]
            if img.shape[-1] != 3:
                return np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
            return img

        def _get_frame(frames: list, idx: int) -> np.ndarray:
            if idx < len(frames):
                img = _normalize_rgb_frame(np.asarray(frames[idx]))
            else:
                img = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
            # Keep output geometry fixed. If camera dimensions drift, fall back to
            # a centered crop/pad with nearest-neighbor behavior.
            if img.shape[0] != cam_height or img.shape[1] != cam_width:
                canvas = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
                h = min(cam_height, img.shape[0])
                w = min(cam_width, img.shape[1])
                canvas[:h, :w] = img[:h, :w]
                img = canvas
            return img

        if is_success:
            out_dir = self.video_dir
        else:
            if not self.save_failed_videos:
                return None
            out_dir = self.failed_video_dir
        out_path = os.path.join(out_dir, f"demo_{demo_id:05d}.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (out_width, out_height),
        )
        if not writer.isOpened():
            return None
        try:
            for i in range(num_frames):
                left = _get_frame(table_frames, i)
                right = _get_frame(wrist_frames, i)
                side_by_side = np.concatenate([left, right], axis=1)
                writer.write(side_by_side[..., ::-1])  # RGB -> BGR
        finally:
            writer.release()
        return out_path

    def save_episode(self, env_id: int, background_texture: str, seed: Optional[int]) -> bool:
        buf = self.buffers[env_id]
        self.attempt_count += 1
        attempt_id = self.attempt_count
        if len(buf["actions"]) < 10:
            if self.save_failed_videos:
                self._write_episode_video(demo_id=attempt_id, buf=buf, is_success=False)
            self.buffers[env_id] = self._empty_buffer()
            self.fail_count += 1
            return False

        is_consistent, reason = self._validate_temporal_consistency(buf)
        if not is_consistent:
            print(f"[WARN] Skipping env {env_id} episode due to temporal consistency check: {reason}")
            if self.save_failed_videos:
                self._write_episode_video(demo_id=attempt_id, buf=buf, is_success=False)
            self.buffers[env_id] = self._empty_buffer()
            self.fail_count += 1
            return False

        is_success = _check_success_from_buffer(buf)
        if not is_success:
            if len(buf["cube_pos"]) > 0 and len(buf["basket_pos"]) > 0:
                cube_last = np.asarray(buf["cube_pos"][-1], dtype=np.float32)
                basket_last = np.asarray(buf["basket_pos"][-1], dtype=np.float32)
                xy_last = float(np.linalg.norm(cube_last[:2] - basket_last[:2]))
                h_last = float(cube_last[2] - basket_last[2])
                print(
                    f"[INFO] Episode filtered (env={env_id}): success_check_failed "
                    f"(xy_last={xy_last:.4f}, h_last={h_last:.4f})"
                )
            if self.save_failed_videos:
                self._write_episode_video(demo_id=attempt_id, buf=buf, is_success=False)
            self.buffers[env_id] = self._empty_buffer()
            self.fail_count += 1
            return False
        
        demo_id = self.demo_count
        with h5py.File(self.output_file, "a") as f:
            g = f["data"].create_group(f"demo_{demo_id}")
            
            for key, data in buf.items():
                if len(data) > 0:
                    arr = np.stack(data)
                    if "rgb" in key or "tactile" in key:
                        g.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
                    else:
                        g.create_dataset(key, data=arr)
            
            g.attrs["num_steps"] = len(buf["actions"])
            g.attrs["success"] = bool(is_success)
            g.attrs["background_texture"] = background_texture
            g.attrs["policy_env"] = self.policy_env_name
            g.attrs["teacher_env"] = self.teacher_env_name
            g.attrs["state_schema"] = "joint_state_9d:[arm_joint_pos(7),gripper_qpos(2)]"
            g.attrs["action_schema"] = "joint_action_8d:[arm_joint_pos_target_abs(7),gripper_cmd(1)]"
            g.attrs["phase_name_map"] = json.dumps(PickPlaceBasketStateMachine.PHASE_NAMES, sort_keys=True)
            g.attrs["seed"] = int(seed) if seed is not None else -1
            g.attrs["planner_params"] = json.dumps(self.planner_params, sort_keys=True)
            # Store planner metadata for reproducibility.
            phase_values = np.asarray(buf["phase_id"], dtype=np.int32)
            unique, counts = np.unique(phase_values, return_counts=True)
            phase_hist = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
            g.attrs["phase_histogram"] = json.dumps(phase_hist, sort_keys=True)

        video_path = self._write_episode_video(demo_id=demo_id, buf=buf, is_success=True)
        metadata = {
            "demo_id": int(demo_id),
            "success": bool(is_success),
            "num_steps": int(len(buf["actions"])),
            "hdf5_group": f"data/demo_{demo_id}",
            "hdf5_keys": sorted([k for k, v in buf.items() if len(v) > 0]),
            "phase_histogram": phase_hist,
            "background_texture": background_texture,
            "policy_env": self.policy_env_name,
            "teacher_env": self.teacher_env_name,
            "state_schema": "joint_state_9d:[arm_joint_pos(7),gripper_qpos(2)]",
            "action_schema": "joint_action_8d:[arm_joint_pos_target_abs(7),gripper_cmd(1)]",
            "seed": int(seed) if seed is not None else None,
            "planner_params": self.planner_params,
            "video_path": video_path,
        }
        meta_path = os.path.join(self.meta_dir, f"demo_{demo_id:05d}.json")
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=True)

        self.demo_count += 1
        self.buffers[env_id] = self._empty_buffer()
        return True
    
    def finalize(self):
        with h5py.File(self.output_file, "a") as f:
            f.attrs["total_demos"] = self.demo_count
            f.attrs["attempted_episodes"] = self.attempt_count
            f.attrs["failed_or_filtered_episodes"] = self.fail_count


def main():
    # Teacher rollout env: keep IK-relative control for stable scripted motion,
    # but record joint-space policy targets (8D absolute joints + gripper).
    try:
        teacher_env_name = "Isaac-Pick-Place-Basket-Franka-IK-Rel-TacEx-v0"
        policy_env_name = "Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0"
        env_cfg = parse_env_cfg(teacher_env_name, device=args_cli.device, num_envs=args_cli.num_envs)
        has_tacex = True
    except Exception:
        teacher_env_name = "Isaac-Pick-Place-Basket-Franka-IK-Rel-v0"
        policy_env_name = "Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0"
        env_cfg = parse_env_cfg(teacher_env_name, device=args_cli.device, num_envs=args_cli.num_envs)
        has_tacex = False
        print("[WARN] TacEx env not found, using standard env (no tactile)")
    
    env_cfg.terminations.time_out = None
    # Match rendering cadence with control cadence to reduce temporal artifacts/flicker.
    env_cfg.sim.render_interval = env_cfg.decimation
    env = gym.make(teacher_env_name, cfg=env_cfg).unwrapped
    selected_background_texture = _configure_background_texture(
        env,
        mode=args_cli.background_mode,
        fixed_texture=args_cli.background_texture,
    )
    
    sm = PickPlaceBasketStateMachine(env_cfg.sim.dt * env_cfg.decimation, env.num_envs, env.device)
    run_seed = getattr(args_cli, "seed", None)
    max_attempt_steps = 400
    planner_params = {
        "pre_approach_duration": sm.pre_approach_duration,
        "align_duration": sm.align_duration,
        "descend_duration": sm.descend_duration,
        "grasp_duration": sm.grasp_duration,
        "grasp_hold_min": sm.grasp_hold_min,
        "lift_clear_duration": sm.lift_clear_duration,
        "lift_duration": sm.lift_duration,
        "carry_mid_1_duration": sm.carry_mid_1_duration,
        "carry_mid_2_duration": sm.carry_mid_2_duration,
        "pre_place_hover_duration": sm.pre_place_hover_duration,
        "basket_descend_duration": sm.basket_descend_duration,
        "pre_grasp_pause": sm.pre_grasp_pause,
        "near_contact_speed": sm.near_contact_speed,
        "carry_speed": sm.carry_speed,
        "profile_gain_range": [sm.profile_gain_min, sm.profile_gain_max],
        "profile_shape_range": [sm.profile_shape_min, sm.profile_shape_max],
        "timing_jitter": sm.timing_jitter,
        "midpoint_xy_jitter": sm.midpoint_xy_jitter,
        "midpoint_z_jitter": sm.midpoint_z_jitter,
        "pre_place_xy_jitter": sm.pre_place_xy_jitter,
        "approach_lateral_jitter": sm.approach_lateral_jitter,
        "grasp_entry_xy_offset": sm.grasp_entry_xy_offset,
        "grasp_height": sm.grasp_height,
        "min_grasp_target_z": sm.min_grasp_target_z,
        "max_pos_step": sm.max_pos_step,
        "max_pos_step_near": sm.max_pos_step_near,
        "near_cube_dist": sm.near_cube_dist,
        "far_cube_dist": sm.far_cube_dist,
        "near_cube_speed": sm.near_cube_speed,
        "pos_filter_alpha": sm.pos_filter_alpha,
        "quat_filter_alpha": sm.quat_filter_alpha,
        "speed_filter_alpha": sm.speed_filter_alpha,
        "max_speed_step": sm.max_speed_step,
        "grasp_dist_threshold": sm.grasp_dist_threshold,
        "grasp_motion_coupling_threshold": sm.grasp_motion_coupling_threshold,
        "grasp_stable_required": int(sm.grasp_stable_required),
        "clearance_margin": sm.clearance_margin,
        "min_lift_delta_z": sm.min_lift_delta_z,
        "hover_xy_threshold": sm.hover_xy_threshold,
        "hover_align_required": int(sm.hover_align_required),
        "joint_action_label_source": "ik_controller_joint_pos_target",
        "max_attempt_steps": int(max_attempt_steps),
        "yaw_slack_deg": float(np.rad2deg(sm.yaw_slack)),
        "roll_slack_deg": float(np.rad2deg(sm.roll_slack)),
        "pitch_slack_deg": float(np.rad2deg(sm.pitch_slack)),
    }
    
    # Setup recorder
    recorder = (
        DataRecorder(
            args_cli.output_dir,
            env.num_envs,
            planner_params,
            save_failed_videos=args_cli.save_failed_videos,
            policy_env_name=policy_env_name,
            teacher_env_name=teacher_env_name,
        )
        if args_cli.save_demos
        else None
    )
    
    # Check available sensors
    has_wrist_cam = "wrist_cam" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    has_table_cam = "table_cam" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    has_gsmini_left = "gsmini_left" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    has_gsmini_right = "gsmini_right" in env.scene.sensors if hasattr(env.scene, "sensors") else False
    
    print(f"[INFO] Teacher environment: {teacher_env_name}")
    print(f"[INFO] Policy/eval environment: {policy_env_name}")
    print(f"[INFO] Background texture: {selected_background_texture}")
    print(f"[INFO] Seed: {run_seed}")
    print(f"[INFO] Sensors: wrist_cam={has_wrist_cam}, table_cam={has_table_cam}, "
          f"gsmini_left={has_gsmini_left}, gsmini_right={has_gsmini_right}")
    
    obs, _ = env.reset()
    ee_frame = env.scene["ee_frame"]
    initial_ee_pos = ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins
    initial_ee_quat = ee_frame.data.target_quat_w[:, 0]
    sm.reset(ee_pos=initial_ee_pos, ee_quat=initial_ee_quat)
    episode_step_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    
    demo_count = 0
    
    print(f"[INFO] Running {args_cli.num_demos} demos with {args_cli.num_envs} parallel environments")
    print(f"[INFO] Saving to: {args_cli.output_dir}" if args_cli.save_demos else "[INFO] Not saving")
    if recorder:
        print(f"[INFO] HDF5 path: {recorder.output_file}")
        print(f"[INFO] Successful videos dir: {recorder.video_dir}")
        if args_cli.save_failed_videos:
            print(f"[INFO] Failed videos dir: {recorder.failed_video_dir}")
    
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
            
            # Compute IK teacher action for scripted rollout.
            des_pose, grip, speed = sm.compute(
                ee_pose,
                cube_pose,
                basket_pose,
                gripper_qpos=robot.data.joint_pos[:, -2:],
            )
            active_mask = sm.sm_state < sm.STATE_DONE
            episode_step_count[active_mask] += 1
            delta = (des_pose[:, :3] - ee_pose[:, :3]) * speed.unsqueeze(-1)
            q_err = _quat_mul_wxyz(des_pose[:, 3:7], _quat_conj_wxyz(ee_pose[:, 3:7]))
            rotvec = _quat_to_rotvec_wxyz(q_err).clamp(min=-0.35, max=0.35)
            ik_actions = torch.cat([delta, rotvec, grip.unsqueeze(-1)], -1)

            # Cache obs_t metadata. We attach action labels after env.step from
            # the controller's actual joint-position target for this same step.
            pending_steps: dict[int, tuple[dict, float]] = {}
            if recorder:
                for i in range(env.num_envs):
                    if sm.sm_state[i] < sm.STATE_DONE:
                        step_data = {
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
                            "phase_id": int(sm.sm_state[i].item()),
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

                        pending_steps[i] = (step_data, float(grip[i].item()))
            
            # Step environment
            obs, _, terminated, truncated, _ = env.step(ik_actions)

            # Finalize per-step recording for obs_t/action_t.
            # The action label is taken from the IK controller's joint target used
            # for this step (arm) plus the scalar gripper command.
            if recorder and pending_steps:
                for env_id, (step_data, gripper_cmd) in pending_steps.items():
                    arm_target = robot.data.joint_pos_target[env_id, :7].detach().cpu().numpy().astype(np.float32)
                    joint_action = np.concatenate(
                        [arm_target, np.array([gripper_cmd], dtype=np.float32)],
                        axis=0,
                    )
                    step_data["actions"] = joint_action
                    step_data["teacher_actions_ik"] = ik_actions[env_id].cpu().numpy()
                    recorder.add_step(env_id, step_data)

            # Hard-stop attempts that exceed max step budget so stalled runs are recorded as failures.
            timeout_ids = (episode_step_count > max_attempt_steps).nonzero(as_tuple=False).squeeze(-1)
            if len(timeout_ids) > 0:
                timeout_list = timeout_ids.tolist()
                for env_id in timeout_list:
                    if recorder:
                        saved_ok = recorder.save_episode(env_id, selected_background_texture, run_seed)
                        if saved_ok:
                            demo_count += 1
                            print(f"[INFO] Saved demo {demo_count}/{args_cli.num_demos} (timeout fallback)")
                        else:
                            print(
                                f"[INFO] Marked failed attempt due to step timeout "
                                f"(env={env_id}, steps={int(episode_step_count[env_id].item())})"
                            )
                # Reset only timed-out environments and restart state machines there.
                env.reset(env_ids=timeout_ids)
                ee_pos_timeout = ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins
                ee_quat_timeout = ee_frame.data.target_quat_w[:, 0]
                sm.reset(timeout_ids, ee_pos_timeout, ee_quat_timeout)
                episode_step_count[timeout_ids] = 0
                # Prevent duplicate processing in reset branch below for these envs.
                terminated[timeout_ids] = False
                truncated[timeout_ids] = False
            
            # Check for environment resets
            env_reset_mask = terminated | truncated
            env_reset_ids = env_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if len(env_reset_ids) > 0:
                ee_pos_fresh = ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins
                
                for env_id in env_reset_ids.tolist():
                    if sm.sm_state[env_id] >= sm.STATE_RELEASE:
                        if recorder and recorder.save_episode(env_id, selected_background_texture, run_seed):
                            demo_count += 1
                            print(f"[INFO] Saved demo {demo_count}/{args_cli.num_demos}")
                        elif not recorder:
                            demo_count += 1
                            print(f"[INFO] Demo {demo_count}/{args_cli.num_demos} completed")
                
                ee_quat_fresh = ee_frame.data.target_quat_w[:, 0]
                sm.reset(env_reset_ids.tolist(), ee_pos_fresh, ee_quat_fresh)
                episode_step_count[env_reset_ids] = 0
            
            # Check for state machine DONE state
            done_envs = (sm.sm_state == sm.STATE_DONE).nonzero(as_tuple=False).squeeze(-1)
            
            if len(done_envs) > 0:
                ee_pos_current = ee_frame.data.target_pos_w[:, 0] - env.scene.env_origins
                
                for env_id in done_envs.tolist():
                    if not env_reset_mask[env_id]:
                        if recorder and recorder.save_episode(env_id, selected_background_texture, run_seed):
                            demo_count += 1
                            print(f"[INFO] Saved demo {demo_count}/{args_cli.num_demos}")
                        elif not recorder:
                            demo_count += 1
                            print(f"[INFO] Demo {demo_count}/{args_cli.num_demos} completed")
                
                ee_quat_current = ee_frame.data.target_quat_w[:, 0]
                sm.reset(done_envs.tolist(), ee_pos_current, ee_quat_current)
                episode_step_count[done_envs] = 0
            
            if demo_count >= args_cli.num_demos:
                break
    
    if recorder:
        recorder.finalize()
        success_rate = (100.0 * recorder.demo_count / max(recorder.attempt_count, 1))
        print(
            f"[INFO] Done! Saved {demo_count} successful demos to {recorder.output_file} "
            f"(attempted={recorder.attempt_count}, filtered={recorder.fail_count}, success_rate={success_rate:.1f}%)"
        )
    else:
        print(f"[INFO] Done! {demo_count} demos completed")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

