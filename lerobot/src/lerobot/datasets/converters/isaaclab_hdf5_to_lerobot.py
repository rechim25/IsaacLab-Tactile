#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert IsaacLab HDF5 demonstration files to LeRobot dataset format.

This converter handles tactile sensing data from TacEx (GelSight) sensors,
including:
- Tactile RGB images
- Force grids (if available)
- Resultant/pseudo force vectors
- Robot state and actions

Usage:
    python isaaclab_hdf5_to_lerobot.py input.hdf5 --output-dir ./lerobot_dataset

The output is a LeRobot-compatible dataset directory that can be used for
training SmolVLA with tactile sensing.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Default key mappings from IsaacLab HDF5 to LeRobot format
DEFAULT_KEY_MAPPINGS = {
    # Camera images
    "rgb_table": "observation.images.table",
    "rgb_wrist": "observation.images.wrist",
    # Tactile images (GelSight)
    "tactile_left": "observation.tactile.image_left",
    "tactile_right": "observation.tactile.image_right",
    # Force data
    "force_geometric_left": "observation.tactile.force_left",
    "force_geometric_right": "observation.tactile.force_right",
    # Robot state
    "joint_pos": "observation.state.joint_positions",
    "ee_pos": "observation.state.end_effector_position",
    "ee_quat": "observation.state.end_effector_quaternion",
    "gripper_pos": "observation.state.gripper_position",
    # Actions
    "actions": "action",
}


def create_state_vector(
    demo: h5py.Group,
    state_keys: list[str],
    frame_idx: int,
) -> np.ndarray | None:
    """Create a state vector by concatenating specified keys.

    Args:
        demo: HDF5 demo group
        state_keys: List of keys to concatenate into state
        frame_idx: Frame index

    Returns:
        Concatenated state vector or None if no keys found
    """
    state_parts = []
    for key in state_keys:
        if key in demo:
            data = demo[key][frame_idx]
            if data.ndim == 0:
                data = np.array([data])
            state_parts.append(data.flatten())

    if state_parts:
        return np.concatenate(state_parts).astype(np.float32)
    return None


def create_force_grid(
    force_left: np.ndarray | None,
    force_right: np.ndarray | None,
    grid_shape: tuple[int, int] = (10, 12),
) -> np.ndarray | None:
    """Create a force grid tensor from individual finger forces.

    For now, this creates a simple force grid by expanding the resultant
    force vector into a uniform grid. If actual per-taxel force data is
    available, this should be updated to use that instead.

    Args:
        force_left: Left finger force (3,)
        force_right: Right finger force (3,)
        grid_shape: Target grid shape (H, W)

    Returns:
        Force grid of shape (num_fingertips, H, W, 3) or None
    """
    forces = []
    if force_left is not None:
        forces.append(force_left)
    if force_right is not None:
        forces.append(force_right)

    if not forces:
        return None

    # Stack into (N, 3)
    force_stack = np.stack(forces, axis=0)  # (N, 3)
    N = force_stack.shape[0]

    # For now, create uniform force distribution across grid
    # Each cell gets force / (H*W) to maintain total force
    H, W = grid_shape
    scale = 1.0 / (H * W)

    # Broadcast to (N, H, W, 3)
    force_grid = force_stack[:, None, None, :] * scale
    force_grid = np.broadcast_to(force_grid, (N, H, W, 3)).copy()

    return force_grid.astype(np.float32)


def save_image(
    data: np.ndarray,
    output_path: Path,
) -> None:
    """Save image data to file.

    Args:
        data: Image array (H, W, C) or (H, W)
        output_path: Path to save image
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle different formats
    if data.ndim == 2:
        img = Image.fromarray(data.astype(np.uint8), mode="L")
    elif data.shape[-1] == 4:
        img = Image.fromarray(data[..., :3].astype(np.uint8), mode="RGB")
    else:
        img = Image.fromarray(data.astype(np.uint8), mode="RGB")

    img.save(output_path)


def convert_hdf5_to_lerobot(
    input_path: Path,
    output_dir: Path,
    key_mappings: dict[str, str] | None = None,
    state_keys: list[str] | None = None,
    task_description: str = "manipulation task",
    fps: int = 30,
    force_grid_shape: tuple[int, int] = (10, 12),
    include_tactile_images: bool = True,
    include_force_grid: bool = True,
) -> dict[str, Any]:
    """Convert an IsaacLab HDF5 file to LeRobot dataset format.

    Args:
        input_path: Path to input HDF5 file
        output_dir: Output directory for LeRobot dataset
        key_mappings: Custom key mappings from HDF5 to LeRobot
        state_keys: Keys to combine into observation.state vector
        task_description: Task description string
        fps: Dataset FPS
        force_grid_shape: Shape of force grid (H, W)
        include_tactile_images: Whether to include tactile RGB images
        include_force_grid: Whether to generate force grids

    Returns:
        Dataset metadata dict
    """
    if key_mappings is None:
        key_mappings = DEFAULT_KEY_MAPPINGS.copy()

    if state_keys is None:
        state_keys = ["joint_pos", "ee_pos", "gripper_pos"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    images_dir = output_dir / "images"
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Open HDF5 file
    with h5py.File(input_path, "r") as f:
        if "data" not in f:
            raise ValueError(f"No 'data' group found in {input_path}")

        demos = list(f["data"].keys())
        logger.info(f"Found {len(demos)} demos in {input_path}")

        # Collect all frames into episodes
        all_episodes = []
        all_frames = []
        frame_idx = 0

        for demo_idx, demo_name in enumerate(tqdm(demos, desc="Processing demos")):
            demo = f["data"][demo_name]

            # Get demo attributes
            demo_attrs = dict(demo.attrs)
            num_steps = demo_attrs.get("num_samples", None)

            # Determine number of steps from data
            if num_steps is None:
                for key in demo.keys():
                    if isinstance(demo[key], h5py.Dataset):
                        num_steps = demo[key].shape[0]
                        break

            if num_steps is None:
                logger.warning(f"Could not determine num_steps for {demo_name}, skipping")
                continue

            episode_start = frame_idx

            # Process each frame
            for step in range(num_steps):
                frame_data = {
                    "episode_index": demo_idx,
                    "frame_index": step,
                    "timestamp": step / fps,
                    "index": frame_idx,
                    "task": task_description,
                }

                # Process actions
                if "actions" in demo:
                    frame_data["action"] = demo["actions"][step].astype(np.float32).tolist()

                # Create state vector
                state = create_state_vector(demo, state_keys, step)
                if state is not None:
                    frame_data["observation.state"] = state.tolist()

                # Process camera images
                for hdf5_key, lerobot_key in key_mappings.items():
                    if hdf5_key not in demo:
                        continue

                    data = demo[hdf5_key][step]

                    if "images" in lerobot_key or (include_tactile_images and "tactile.image" in lerobot_key):
                        # Save as image file
                        img_path = images_dir / lerobot_key.replace(".", "/") / f"{frame_idx:06d}.png"
                        save_image(data, img_path)
                        frame_data[lerobot_key] = str(img_path.relative_to(output_dir))

                    elif "force" in lerobot_key and "force_grid" not in lerobot_key:
                        # Store force vector directly
                        frame_data[lerobot_key] = data.astype(np.float32).tolist()

                # Create force grid from individual forces
                if include_force_grid:
                    force_left = demo["force_geometric_left"][step] if "force_geometric_left" in demo else None
                    force_right = demo["force_geometric_right"][step] if "force_geometric_right" in demo else None

                    force_grid = create_force_grid(force_left, force_right, force_grid_shape)
                    if force_grid is not None:
                        # Store as nested list for JSON serialization
                        frame_data["observation.tactile.force_grid"] = force_grid.tolist()

                all_frames.append(frame_data)
                frame_idx += 1

            # Episode metadata
            episode_data = {
                "episode_index": demo_idx,
                "tasks": [task_description],
                "length": num_steps,
            }
            all_episodes.append(episode_data)

    # Save data as JSON (for simplicity; can be converted to parquet later)
    logger.info(f"Saving {len(all_frames)} frames to {output_dir}")

    # Save frames (chunked for large datasets)
    chunk_size = 10000
    for chunk_idx in range(0, len(all_frames), chunk_size):
        chunk = all_frames[chunk_idx : chunk_idx + chunk_size]
        chunk_path = data_dir / f"chunk_{chunk_idx // chunk_size:04d}.json"
        with open(chunk_path, "w") as f_out:
            json.dump(chunk, f_out)

    # Create metadata
    # Infer shapes from first frame
    sample_frame = all_frames[0]

    features = {}
    if "action" in sample_frame:
        features["action"] = {
            "dtype": "float32",
            "shape": [len(sample_frame["action"])],
        }
    if "observation.state" in sample_frame:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [len(sample_frame["observation.state"])],
        }
    if "observation.tactile.force_grid" in sample_frame:
        grid = np.array(sample_frame["observation.tactile.force_grid"])
        features["observation.tactile.force_grid"] = {
            "dtype": "float32",
            "shape": list(grid.shape),
        }

    # Add image features
    for key in sample_frame:
        if key.startswith("observation.images") or key.startswith("observation.tactile.image"):
            features[key] = {
                "dtype": "image",
                "shape": None,  # Determined at load time
            }
        elif key.startswith("observation.tactile.force") and "force_grid" not in key:
            val = sample_frame[key]
            features[key] = {
                "dtype": "float32",
                "shape": [len(val)] if isinstance(val, list) else [1],
            }

    metadata = {
        "codebase_version": "v3.0",
        "fps": fps,
        "total_episodes": len(all_episodes),
        "total_frames": len(all_frames),
        "features": features,
        "tasks": [task_description],
        "robot_type": "isaaclab_tactile",
        "episodes": all_episodes,
    }

    # Save metadata
    with open(output_dir / "meta" / "info.json", "w") as f_out:
        (output_dir / "meta").mkdir(exist_ok=True)
        json.dump(metadata, f_out, indent=2)

    with open(output_dir / "meta" / "episodes.json", "w") as f_out:
        json.dump(all_episodes, f_out, indent=2)

    logger.info(f"Conversion complete: {len(all_episodes)} episodes, {len(all_frames)} frames")
    return metadata


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert IsaacLab HDF5 demos to LeRobot dataset format"
    )
    parser.add_argument("input", type=Path, help="Input HDF5 file path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./lerobot_dataset"),
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="manipulation task",
        help="Task description",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Dataset FPS",
    )
    parser.add_argument(
        "--force-grid-shape",
        type=str,
        default="10,12",
        help="Force grid shape as 'H,W'",
    )
    parser.add_argument(
        "--no-tactile-images",
        action="store_true",
        help="Exclude tactile RGB images",
    )
    parser.add_argument(
        "--no-force-grid",
        action="store_true",
        help="Exclude force grid generation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    force_grid_shape = tuple(map(int, args.force_grid_shape.split(",")))

    convert_hdf5_to_lerobot(
        input_path=args.input,
        output_dir=args.output_dir,
        task_description=args.task,
        fps=args.fps,
        force_grid_shape=force_grid_shape,
        include_tactile_images=not args.no_tactile_images,
        include_force_grid=not args.no_force_grid,
    )


if __name__ == "__main__":
    main()
