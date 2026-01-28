#!/usr/bin/env python
"""
Convert IsaacLab TacEx HDF5 demonstration data to LeRobot dataset format.

This script converts HDF5 files containing pick-place demonstrations with tactile sensing
into the LeRobot v3 dataset format for training SmolVLA with tactile integration.

Canonical Policy Convention (IsaacLab Tactile):
    State (11D): [eef_pos_b(3), eef_rot6d_b(6), gripper_qpos(2)]
    Action (7D): [Δpos_b(3), Δaxis_angle_b(3), gripper(1)]
    
All pose data is expressed in robot base frame for frame-invariance.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.isaaclab_tactile.policy_io import (
    encode_action_isaaclab_to_policy,
    encode_state_isaaclab_to_policy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert IsaacLab HDF5 to LeRobot dataset format (canonical policy convention)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input HDF5 file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="/home/radu/IsaacLab-Tactile/lerobot/datasets",
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository ID for the dataset (default: derived from input filename)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second of the demonstrations",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Pick and place the cube into the basket",
        help="Task description to include in the dataset",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="isaaclab_tactile",
        help="Robot type identifier",
    )
    parser.add_argument(
        "--force-grid-height",
        type=int,
        default=10,
        help="Height of synthetic force grid",
    )
    parser.add_argument(
        "--force-grid-width",
        type=int,
        default=12,
        help="Width of synthetic force grid",
    )
    parser.add_argument(
        "--num-fingertips",
        type=int,
        default=2,
        help="Number of fingertips (gripper sensors)",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        help="Save images as videos instead of individual PNGs",
    )
    parser.add_argument(
        "--store-debug-fields",
        action="store_true",
        help="Store raw world-frame fields for debugging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Derive repo_id from input filename if not provided
    if args.repo_id is None:
        input_path = Path(args.input)
        args.repo_id = input_path.stem + "_lerobot"

    # Canonical state dimension: 11D = [eef_pos_b(3), eef_rot6d_b(6), gripper_qpos(2)]
    STATE_DIM = 11
    # Canonical action dimension: 7D = [Δpos_b(3), Δaxis_angle_b(3), gripper(1)]
    ACTION_DIM = 7

    print(f"Converting {args.input} -> {args.output_dir}/{args.repo_id}")
    print(f"  FPS: {args.fps}")
    print(f"  Task: {args.task}")
    print(f"  State dim: {STATE_DIM} (canonical policy format)")
    print(f"  Action dim: {ACTION_DIM} (canonical policy format)")
    print(f"  Force grid: {args.num_fingertips} x {args.force_grid_height} x {args.force_grid_width} x 3")

    # Define dataset features
features = {
    "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
        # Image features with names for dataset_to_policy_features
        "observation.images.camera1": {
            "dtype": "image",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        },
        "observation.images.camera2": {
            "dtype": "image",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        },
        # Tactile force grid
        "observation.tactile.force_grid": {
            "dtype": "float32",
            "shape": (args.num_fingertips, args.force_grid_height, args.force_grid_width, 3),
            "names": None,
        },
    }

    # Optionally add debug fields
    if args.store_debug_fields:
        features["debug.eef_pos_w"] = {"dtype": "float32", "shape": (3,), "names": None}
        features["debug.eef_quat_w"] = {"dtype": "float32", "shape": (4,), "names": None}
        features["debug.base_pos_w"] = {"dtype": "float32", "shape": (3,), "names": None}
        features["debug.base_quat_w"] = {"dtype": "float32", "shape": (4,), "names": None}

    # Create dataset
ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=f"{args.output_dir}/{args.repo_id}",
        fps=args.fps,
    features=features,
        robot_type=args.robot_type,
        use_videos=args.use_videos,
)

    H, W = args.force_grid_height, args.force_grid_width
scale = 1.0 / (H * W)

    with h5py.File(args.input, "r") as f:
        demos = sorted(f["data"].keys())
        total_demos = len(demos)
        print(f"\nProcessing {total_demos} demonstrations...")

        for idx, demo_name in enumerate(demos, 1):
        demo = f["data"][demo_name]
        T = demo["actions"].shape[0]

            print(f"  [{idx}/{total_demos}] {demo_name}: {T} steps", end="\r")

        for t in range(T):
                # Extract raw data from HDF5
                ee_pos_w = demo["ee_pos"][t].astype(np.float32)  # (3,)
                ee_quat_w = demo["ee_quat"][t].astype(np.float32)  # (4,) - (x,y,z,w)
                gripper_qpos = demo["gripper_pos"][t].astype(np.float32)  # (2,)

                # Get base pose (if available, otherwise use identity)
                if "base_pos" in demo:
                    base_pos_w = demo["base_pos"][t].astype(np.float32)
                    base_quat_w = demo["base_quat"][t].astype(np.float32)
                else:
                    # Default to identity for fixed base robots
                    base_pos_w = np.zeros(3, dtype=np.float32)
                    base_quat_w = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

                # Encode state using the shared adapter (11D)
                state = encode_state_isaaclab_to_policy(
                    eef_pos_w=ee_pos_w,
                    eef_quat_w=ee_quat_w,
                    gripper_qpos=gripper_qpos,
                    base_pos_w=base_pos_w,
                    base_quat_w=base_quat_w,
                )

                # Get raw action from HDF5 (7D: [delta_pos(3), delta_rot(3), gripper(1)])
                raw_action = demo["actions"][t].astype(np.float32)

                # Transform action from world frame to base frame (7D)
                delta_pos_w = raw_action[:3]
                delta_rot_w = raw_action[3:6]
                gripper_cmd = raw_action[6:7]

                action = encode_action_isaaclab_to_policy(
                    delta_pos_w=delta_pos_w,
                    delta_rot_w=delta_rot_w,
                    gripper_cmd=gripper_cmd,
                    base_quat_w=base_quat_w,
                )

                # Build synthetic force grid from resultant forces
                fL = demo["force_geometric_left"][t].astype(np.float32)  # (3,)
            fR = demo["force_geometric_right"][t].astype(np.float32)  # (3,)
            grid = np.stack([fL, fR], axis=0)[:, None, None, :] * scale
                grid = np.broadcast_to(grid, (args.num_fingertips, H, W, 3)).copy().astype(np.float32)

                # Build frame dict
            frame = {
                    "task": args.task,
                    "observation.state": state.astype(np.float32),
                    "action": action.astype(np.float32),
                    # Map table->camera1, wrist->camera2 to match SmolVLA policy expectations
                    "observation.images.camera1": demo["rgb_table"][t],
                    "observation.images.camera2": demo["rgb_wrist"][t],
                "observation.tactile.force_grid": grid,
            }

                # Optionally store debug fields
                if args.store_debug_fields:
                    frame["debug.eef_pos_w"] = ee_pos_w
                    frame["debug.eef_quat_w"] = ee_quat_w
                    frame["debug.base_pos_w"] = base_pos_w
                    frame["debug.base_quat_w"] = base_quat_w

            ds.add_frame(frame)

        ds.save_episode()

    print(f"\n\nFinalizing dataset...")
ds.finalize()
    print(f"✓ Successfully wrote LeRobot dataset to: {ds.root}")
    print(f"  Total episodes: {len(demos)}")
    print(f"  Total frames: {len(ds)}")
    print(f"\nDataset uses canonical policy convention:")
    print(f"  State (11D): [eef_pos_b(3), eef_rot6d_b(6), gripper_qpos(2)]")
    print(f"  Action (7D): [Δpos_b(3), Δaxis_angle_b(3), gripper(1)]")


if __name__ == "__main__":
    main()
