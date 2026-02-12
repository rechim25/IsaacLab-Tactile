#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Export side-by-side camera visualization videos directly from raw HDF5 frames.

This script is intended for datasets produced by:
`scripts/environments/state_machine/pick_place_basket_tacex_sm.py`

Expected HDF5 structure:
    data/demo_*/rgb_table   (T, H, W, C)
    data/demo_*/rgb_wrist   (T, H, W, C)

The exporter performs no geometric resizing by default, so the output video
reflects camera-frame quality as stored in HDF5.
"""

import argparse
import os
from typing import List

import cv2
import h5py
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw camera videos from HDF5 demos.")
    parser.add_argument(
        "--input_hdf5",
        type=str,
        required=True,
        help="Path to input HDF5 file (e.g., <output_dir>/data.hdf5).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for videos. Defaults to <hdf5_dir>/raw_hdf5_videos.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video framerate.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec (default: mp4v). Example alternatives: MJPG, XVID.",
    )
    parser.add_argument(
        "--include_single_camera",
        action="store_true",
        help="If set, export demos that only contain one of the two cameras.",
    )
    parser.add_argument(
        "--lossless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable truly lossless export using FFV1 codec in MKV container. "
            "Enabled by default. Use --no-lossless to disable. "
            "When enabled, --codec is ignored."
        ),
    )
    return parser.parse_args()


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    keys = list(data_group.keys())

    def _idx(k: str) -> int:
        if k.startswith("demo_"):
            try:
                return int(k.split("_", maxsplit=1)[1])
            except Exception:
                return 10**9
        return 10**9

    return sorted(keys, key=_idx)


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    img = np.asarray(frame)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3:
        raise ValueError(f"Expected 3D frame (H,W,C), got shape={img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] != 3:
        raise ValueError(f"Expected RGB/RGBA channels, got shape={img.shape}")
    return img


def _pad_or_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h = min(target_h, img.shape[0])
    w = min(target_w, img.shape[1])
    out[:h, :w] = img[:h, :w]
    return out


def _write_demo_video(
    demo_group: h5py.Group,
    out_path: str,
    fps: float,
    codec: str,
    include_single_camera: bool,
) -> bool:
    has_table = "rgb_table" in demo_group
    has_wrist = "rgb_wrist" in demo_group
    if not has_table and not has_wrist:
        return False
    if (not has_table or not has_wrist) and not include_single_camera:
        return False

    table_ds = demo_group["rgb_table"] if has_table else None
    wrist_ds = demo_group["rgb_wrist"] if has_wrist else None
    table_len = int(table_ds.shape[0]) if table_ds is not None else 0
    wrist_len = int(wrist_ds.shape[0]) if wrist_ds is not None else 0
    num_frames = max(table_len, wrist_len)
    if num_frames <= 0:
        return False

    ref = table_ds[0] if table_ds is not None else wrist_ds[0]
    ref_img = _to_uint8_rgb(ref)
    cam_h, cam_w = int(ref_img.shape[0]), int(ref_img.shape[1])
    out_h, out_w = cam_h, cam_w * 2

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {out_path}")

    try:
        for t in range(num_frames):
            if table_ds is not None and t < table_len:
                left = _to_uint8_rgb(table_ds[t])
            else:
                left = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

            if wrist_ds is not None and t < wrist_len:
                right = _to_uint8_rgb(wrist_ds[t])
            else:
                right = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

            if left.shape[0] != cam_h or left.shape[1] != cam_w:
                left = _pad_or_crop(left, cam_h, cam_w)
            if right.shape[0] != cam_h or right.shape[1] != cam_w:
                right = _pad_or_crop(right, cam_h, cam_w)

            side_by_side = np.concatenate([left, right], axis=1)
            writer.write(side_by_side[..., ::-1])  # RGB -> BGR
    finally:
        writer.release()

    return True


def main() -> None:
    args = _parse_args()
    input_hdf5 = args.input_hdf5
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_hdf5), "raw_hdf5_videos")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    codec = "FFV1" if args.lossless else args.codec
    ext = ".mkv" if args.lossless else ".mp4"
    if args.lossless:
        print("[INFO] Lossless mode enabled: codec=FFV1 container=MKV")

    with h5py.File(input_hdf5, "r") as f:
        if "data" not in f:
            raise KeyError("Expected top-level 'data' group in HDF5.")
        demo_keys = _sorted_demo_keys(f["data"])
        print(f"[INFO] Found {len(demo_keys)} demos in {input_hdf5}")
        written = 0
        skipped = 0
        for demo_key in demo_keys:
            demo_group = f["data"][demo_key]
            out_path = os.path.join(output_dir, f"{demo_key}{ext}")
            ok = _write_demo_video(
                demo_group=demo_group,
                out_path=out_path,
                fps=args.fps,
                codec=codec,
                include_single_camera=args.include_single_camera,
            )
            if ok:
                written += 1
            else:
                skipped += 1
        print(f"[INFO] Wrote {written} videos to {output_dir} (skipped={skipped})")


if __name__ == "__main__":
    main()
