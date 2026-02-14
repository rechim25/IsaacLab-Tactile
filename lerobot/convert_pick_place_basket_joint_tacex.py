#!/usr/bin/env python
"""
Convert IsaacLab TacEx HDF5 data to LeRobot format for joint-space control.

Joint policy convention used by this converter:
    State (9D):  [arm_joint_pos(7), gripper_qpos(2)]
    Action (8D): [arm_joint_pos_target_abs(7), gripper_cmd(1)]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert IsaacLab joint-space HDF5 demos to LeRobot dataset format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input HDF5 file")
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
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of demonstrations")
    parser.add_argument(
        "--task",
        type=str,
        default="Pick and place the cube into the basket",
        help="Task description included in dataset",
    )
    parser.add_argument("--robot-type", type=str, default="isaaclab_tactile", help="Robot type identifier")
    parser.add_argument("--force-grid-height", type=int, default=10, help="Synthetic force-grid height")
    parser.add_argument("--force-grid-width", type=int, default=12, help="Synthetic force-grid width")
    parser.add_argument("--num-fingertips", type=int, default=2, help="Number of fingertips")
    parser.add_argument("--use-videos", action="store_true", help="Store images as videos")
    return parser.parse_args()


def _extract_gripper_qpos(demo: h5py.Group, t: int, joint_pos_t: np.ndarray) -> np.ndarray:
    if "gripper_pos" in demo:
        grip = np.asarray(demo["gripper_pos"][t], dtype=np.float32).reshape(-1)
    else:
        grip = np.asarray(joint_pos_t[-2:], dtype=np.float32).reshape(-1)

    if grip.size >= 2:
        return grip[:2].astype(np.float32)
    if grip.size == 1:
        return np.array([grip[0], grip[0]], dtype=np.float32)
    return np.zeros((2,), dtype=np.float32)


def _extract_image(demo: h5py.Group, key: str, t: int, default_h: int = 224, default_w: int = 224) -> np.ndarray:
    if key not in demo:
        return np.zeros((default_h, default_w, 3), dtype=np.uint8)
    img = np.asarray(demo[key][t])
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3:
        return np.zeros((default_h, default_w, 3), dtype=np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] != 3:
        return np.zeros((default_h, default_w, 3), dtype=np.uint8)
    return img


def _extract_force_grid(demo: h5py.Group, t: int, num_fingertips: int, h: int, w: int) -> np.ndarray:
    scale = 1.0 / float(h * w)
    grid = np.zeros((num_fingertips, h, w, 3), dtype=np.float32)

    if "force_geometric_left" in demo:
        f_left = np.asarray(demo["force_geometric_left"][t], dtype=np.float32).reshape(3)
        grid[0] = np.broadcast_to(f_left[None, None, :] * scale, (h, w, 3)).astype(np.float32)
    if num_fingertips > 1 and "force_geometric_right" in demo:
        f_right = np.asarray(demo["force_geometric_right"][t], dtype=np.float32).reshape(3)
        grid[1] = np.broadcast_to(f_right[None, None, :] * scale, (h, w, 3)).astype(np.float32)

    return grid


def main():
    args = parse_args()

    if args.repo_id is None:
        args.repo_id = Path(args.input).stem + "_lerobot"

    state_dim = 9
    action_dim = 8
    print(f"Converting {args.input} -> {args.output_dir}/{args.repo_id}")
    print(f"  State dim: {state_dim} (joint policy state)")
    print(f"  Action dim: {action_dim} (absolute joint targets)")

    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
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
        "observation.tactile.force_grid": {
            "dtype": "float32",
            "shape": (args.num_fingertips, args.force_grid_height, args.force_grid_width, 3),
            "names": None,
        },
    }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=f"{args.output_dir}/{args.repo_id}",
        fps=args.fps,
        features=features,
        robot_type=args.robot_type,
        use_videos=args.use_videos,
    )

    with h5py.File(args.input, "r") as f:
        if "data" not in f:
            raise KeyError("Expected top-level 'data' group in HDF5.")
        demos = sorted(f["data"].keys())
        print(f"Processing {len(demos)} demonstrations...")

        for idx, demo_name in enumerate(demos, 1):
            demo = f["data"][demo_name]
            if "actions" not in demo:
                raise KeyError(f"Demo '{demo_name}' missing 'actions'.")
            if "joint_pos" not in demo:
                raise KeyError(f"Demo '{demo_name}' missing 'joint_pos'.")

            actions = np.asarray(demo["actions"])
            if actions.ndim != 2 or actions.shape[1] != action_dim:
                raise ValueError(
                    f"Demo '{demo_name}' has action shape {actions.shape}, expected (*, {action_dim})."
                )

            T = int(actions.shape[0])
            print(f"  [{idx}/{len(demos)}] {demo_name}: {T} steps", end="\r")

            for t in range(T):
                joint_pos_t = np.asarray(demo["joint_pos"][t], dtype=np.float32).reshape(-1)
                if joint_pos_t.size < 7:
                    raise ValueError(f"Demo '{demo_name}' step {t}: joint_pos has size {joint_pos_t.size}, expected >= 7")
                arm_joint_pos = joint_pos_t[:7].astype(np.float32)
                gripper_qpos = _extract_gripper_qpos(demo, t, joint_pos_t)
                state = np.concatenate([arm_joint_pos, gripper_qpos], axis=0).astype(np.float32)

                action = actions[t].astype(np.float32)
                force_grid = _extract_force_grid(
                    demo,
                    t,
                    num_fingertips=args.num_fingertips,
                    h=args.force_grid_height,
                    w=args.force_grid_width,
                )

                frame = {
                    "task": args.task,
                    "observation.state": state,
                    "action": action,
                    "observation.images.camera1": _extract_image(demo, "rgb_table", t),
                    "observation.images.camera2": _extract_image(demo, "rgb_wrist", t),
                    "observation.tactile.force_grid": force_grid,
                }
                ds.add_frame(frame)

            ds.save_episode()

    print("\nFinalizing dataset...")
    ds.finalize()
    print(f"Successfully wrote LeRobot dataset to: {ds.root}")
    print(f"  Total episodes: {len(demos)}")
    print(f"  Total frames: {len(ds)}")
    print("Dataset convention:")
    print("  State (9D): [arm_joint_pos(7), gripper_qpos(2)]")
    print("  Action (8D): [arm_joint_pos_target_abs(7), gripper(1)]")


if __name__ == "__main__":
    main()
