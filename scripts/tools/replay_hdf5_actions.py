#!/usr/bin/env python3
"""Replay actions from HDF5 dataset by stepping through IsaacLab environment."""

from __future__ import annotations

import argparse
from pathlib import Path

# Parse args BEFORE AppLauncher
parser = argparse.ArgumentParser(
    description="Replay HDF5 actions in IsaacLab environment and save video.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset", required=True, help="Path to HDF5 file.")
parser.add_argument("--demo", default=None, help="Demo name/index to replay. If omitted, replays first demo.")
parser.add_argument("--env", type=str, default=None, help="Environment name (auto-detected from HDF5 if not provided).")
parser.add_argument("--start", type=int, default=0, help="Start frame index (inclusive).")
parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive).")
parser.add_argument("--lag", type=float, default=0.0, help="Seconds to wait between frames (for throttling).")
parser.add_argument("--rgb-table-key", default="rgb_table", help="HDF5 dataset key for table camera.")
parser.add_argument("--rgb-wrist-key", default="rgb_wrist", help="HDF5 dataset key for wrist camera.")
parser.add_argument("--actions-key", default="actions", help="HDF5 dataset key for actions.")
parser.add_argument(
    "--restore-state",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Restore cube/basket/robot joint state from HDF5 before replaying actions.",
)
parser.add_argument(
    "--state-t",
    type=int,
    default=None,
    help="Timestep index to restore state from (default: uses --start).",
)
parser.add_argument(
    "--output",
    default=None,
    help="Output video path (.mp4).",
)
parser.add_argument(
    "--output-dir",
    default=None,
    help="Output directory for videos.",
)
parser.add_argument(
    "--fps",
    type=float,
    default=30.0,
    help="Output video FPS.",
)
parser.add_argument(
    "--img-height",
    type=int,
    default=224,
    help="Camera image height.",
)
parser.add_argument(
    "--img-width",
    type=int,
    default=224,
    help="Camera image width.",
)

# Add AppLauncher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# Imports that require simulator running
# ============================================================================

import time

import cv2
import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _as_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image to HWC uint8 format."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype in (np.float32, np.float64):
        if np.nanmax(img) <= 1.5:
            img = img * 255.0
        img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    else:
        img = np.clip(img.astype(np.int32), 0, 255).astype(np.uint8)
    
    if img.shape[0] != args_cli.img_height or img.shape[1] != args_cli.img_width:
        img = cv2.resize(img, (args_cli.img_width, args_cli.img_height), interpolation=cv2.INTER_LINEAR)
    
    return img


def _format_action(action: np.ndarray) -> str:
    """Format action as string."""
    if action.ndim > 1:
        action = action.reshape(-1)
    return "action: [" + ", ".join(f"{x:+.3f}" for x in action) + "]"


def _quat_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Best-effort conversion to IsaacLab's expected (w, x, y, z) ordering."""
    q = np.asarray(q, dtype=np.float32).reshape(4)
    # Heuristic: for near-identity quats, w has the largest magnitude (~1.0).
    if abs(float(q[0])) >= abs(float(q[3])):
        # Already wxyz
        return q
    # xyzw -> wxyz
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def _restore_demo_state(
    env,
    *,
    joint_pos: np.ndarray | None,
    joint_vel: np.ndarray | None,
    cube_pos: np.ndarray | None,
    cube_quat: np.ndarray | None,
    basket_pos: np.ndarray | None,
    basket_quat: np.ndarray | None,
) -> None:
    """Restore robot joints and cube/basket root poses from recorded demo state."""
    # NOTE: This script assumes num_envs=1.
    env_ids = torch.tensor([0], device=env.device)
    origins = env.scene.env_origins[env_ids, :3]

    if joint_pos is not None:
        robot = env.scene["robot"]
        jp = torch.tensor(joint_pos, dtype=torch.float32, device=env.device).unsqueeze(0)
        if joint_vel is None:
            jv = torch.zeros_like(jp)
        else:
            jv = torch.tensor(joint_vel, dtype=torch.float32, device=env.device).unsqueeze(0)

        # Write state + targets to minimize controller transients.
        robot.set_joint_position_target(jp, env_ids=env_ids)
        robot.set_joint_velocity_target(jv, env_ids=env_ids)
        robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

    def _set_rigid(name: str, pos: np.ndarray | None, quat: np.ndarray | None) -> None:
        if pos is None or quat is None:
            return
        obj = env.scene[name]
        p = torch.tensor(pos, dtype=torch.float32, device=env.device).unsqueeze(0) + origins
        q = torch.tensor(_quat_to_wxyz(quat), dtype=torch.float32, device=env.device).unsqueeze(0)
        obj.write_root_pose_to_sim(torch.cat([p, q], dim=-1), env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=env.device), env_ids=env_ids)

    _set_rigid("cube", cube_pos, cube_quat)
    _set_rigid("basket", basket_pos, basket_quat)


def _draw_text_block(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    font,
    font_scale: float,
    thickness: int,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    padding: int = 6,
) -> None:
    """Draw text with black background that wraps to fit max_width."""
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        trial = (current + " " + word).strip()
        (w, h), _ = cv2.getTextSize(trial, font, font_scale, thickness)
        if w <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_height = max(h for _, h in line_sizes) if line_sizes else 0
    block_height = line_height * len(lines) + padding * 2 + max(0, len(lines) - 1) * 4
    block_width = min(max(w for w, _ in line_sizes) + padding * 2, max_width + padding * 2)

    x, y = origin
    x2 = x + block_width
    y2 = y + block_height
    cv2.rectangle(img, (x, y), (x2, y2), bg_color, thickness=-1)

    y_text = y + padding + line_height
    for line in lines:
        cv2.putText(
            img,
            line,
            (x + padding, y_text),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        y_text += line_height + 4


def main():
    """Replay HDF5 actions through IsaacLab environment and save video."""
    
    if args_cli.output and args_cli.output_dir:
        raise ValueError("Cannot specify both --output and --output-dir. Choose one.")
    
    # Load HDF5 dataset
    if not Path(args_cli.dataset).exists():
        raise FileNotFoundError(f"Dataset file not found: {args_cli.dataset}")
    
    with h5py.File(args_cli.dataset, "r") as f:
        if "data" not in f:
            raise KeyError("HDF5 file missing 'data' group.")
        
        data_group = f["data"]
        demo_names = list(data_group.keys())
        if not demo_names:
            raise ValueError("No demos found in HDF5 file.")
        
        # Select demo
        if args_cli.demo:
            # Try as name first, then as index
            if args_cli.demo in demo_names:
                demo_name = args_cli.demo
            else:
                try:
                    idx = int(args_cli.demo)
                    demo_name = demo_names[idx]
                except (ValueError, IndexError):
                    raise ValueError(f"Demo '{args_cli.demo}' not found. Available: {demo_names}")
        else:
            demo_name = demo_names[0]
        
        demo = data_group[demo_name]
        
        # Get environment name
        env_name = args_cli.env
        if env_name is None:
            if "env" in f.attrs:
                env_name = f.attrs["env"]
            elif "env_name" in demo.attrs:
                env_name = demo.attrs["env_name"]
            else:
                raise ValueError("Environment name not found. Specify with --env.")
        
        # Load actions
        if args_cli.actions_key not in demo:
            raise KeyError(f"Missing '{args_cli.actions_key}' in demo '{demo_name}'.")
        
        actions = demo[args_cli.actions_key][:]
        num_frames = len(actions)
        
        start = max(0, args_cli.start)
        end = args_cli.end if args_cli.end is not None else num_frames
        end = min(end, num_frames)
        
        if start >= end:
            raise ValueError(f"Invalid range: start={start}, end={end}, num_frames={num_frames}")

        # Load state needed to restore initial conditions (optional but recommended).
        state_t = start if args_cli.state_t is None else int(args_cli.state_t)
        if state_t < 0 or state_t >= num_frames:
            raise ValueError(f"--state-t out of range: {state_t} (valid: 0..{num_frames-1})")

        joint_pos_t = demo["joint_pos"][state_t].astype(np.float32) if "joint_pos" in demo else None
        joint_vel_t = demo["joint_vel"][state_t].astype(np.float32) if "joint_vel" in demo else None
        cube_pos_t = demo["cube_pos"][state_t].astype(np.float32) if "cube_pos" in demo else None
        cube_quat_t = demo["cube_quat"][state_t].astype(np.float32) if "cube_quat" in demo else None
        basket_pos_t = demo["basket_pos"][state_t].astype(np.float32) if "basket_pos" in demo else None
        basket_quat_t = demo["basket_quat"][state_t].astype(np.float32) if "basket_quat" in demo else None
        
        print(f"Replaying demo '{demo_name}' frames {start}:{end} ({end - start} frames)")
        print(f"Environment: {env_name}")
        if args_cli.restore_state:
            print(f"Restoring initial state from HDF5 at t={state_t}")
        
        # Determine output path
        if args_cli.output_dir:
            output_dir = Path(args_cli.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"replay_{Path(demo_name).name}.mp4"
        elif args_cli.output:
            output_path = Path(args_cli.output)
        else:
            output_path = Path(f"replay_{Path(demo_name).name}.mp4")
    
    # Create environment
    print("Creating IsaacLab environment...")
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)

    # Disable termination on timeout during replay; we want to run the recorded horizon.
    try:
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    except Exception:
        pass
    
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    
    # Detect cameras
    sensors = getattr(env.scene, "sensors", {})
    has_table_cam = "table_cam" in sensors
    has_wrist_cam = "wrist_cam" in sensors
    
    print(f"Cameras: table={has_table_cam}, wrist={has_wrist_cam}")
    
    if not has_table_cam and not has_wrist_cam:
        raise RuntimeError("No cameras found in environment. Cannot create video.")
    
    # Reset environment
    env.reset()

    # Restore recorded initial state (cube/basket/robot joints) for faithful replay.
    if args_cli.restore_state:
        _restore_demo_state(
            env,
            joint_pos=joint_pos_t,
            joint_vel=joint_vel_t,
            cube_pos=cube_pos_t,
            cube_quat=cube_quat_t,
            basket_pos=basket_pos_t,
            basket_quat=basket_quat_t,
        )
    
    # Step once to initialize sensors
    zero_action = torch.zeros((1, actions.shape[1]), dtype=torch.float32, device=args_cli.device)
    if zero_action.shape[1] >= 7:
        # Keep gripper open (positive -> open in IsaacLab BinaryJointAction).
        zero_action[0, 6] = 1.0
    env.step(zero_action)
    
    # Initialize video writer
    writer = None
    
    print(f"Stepping through {end - start} frames...")
    
    try:
        for t in range(start, end):
            # Get action
            action = actions[t].astype(np.float32)
            action_tensor = torch.from_numpy(action).unsqueeze(0).to(args_cli.device)
            
            # Step environment
            env.step(action_tensor)
            
            # Capture camera frames
            img_table = np.zeros((args_cli.img_height, args_cli.img_width, 3), dtype=np.uint8)
            img_wrist = np.zeros((args_cli.img_height, args_cli.img_width, 3), dtype=np.uint8)
            
            if has_table_cam:
                rgb = env.scene.sensors["table_cam"].data.output.get("rgb")
                if rgb is not None and rgb.numel() > 0:
                    img_table = _as_hwc_uint8(rgb[0])
            
            if has_wrist_cam:
                rgb = env.scene.sensors["wrist_cam"].data.output.get("rgb")
                if rgb is not None and rgb.numel() > 0:
                    img_wrist = _as_hwc_uint8(rgb[0])
            
            # Combine cameras side-by-side
            combined = np.concatenate([img_table, img_wrist], axis=1)
            
            # Add action text overlay
            text = f"t={t} | {_format_action(action)}"
            margin = 10
            max_width = combined.shape[1] - margin * 2
            _draw_text_block(
                combined,
                text,
                (margin, margin),
                max_width=max_width,
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=0.6,
                thickness=2,
            )
            
            # Initialize writer on first frame
            if writer is None:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, args_cli.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer at '{output_path}'.")
                print(f"Writing video to: {output_path}")
            
            writer.write(combined)
            
            if (t - start) % 10 == 0:
                print(f"  Frame {t}/{end-1}: {action.tolist()}")
            
            if args_cli.lag > 0:
                time.sleep(args_cli.lag)
        
        print(f"\n✓ Video saved to: {output_path}")
        
    finally:
        if writer is not None:
            writer.release()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
