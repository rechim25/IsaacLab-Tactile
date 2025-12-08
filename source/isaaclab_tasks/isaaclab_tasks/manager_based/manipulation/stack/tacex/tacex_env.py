# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacEx-enabled Manager-Based RL Environment.

This module provides a custom environment class that integrates TacEx
tactile simulation with Isaac Lab's manager-based environment framework.

References:
- TacEx Paper: https://arxiv.org/pdf/2411.04776
- TacEx Site: https://sites.google.com/view/tacex
"""

from __future__ import annotations

import torch
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from .tacex_manager import TacExManager, TacExSensorConfig


class TacExManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-Based RL Environment with TacEx tactile simulation.

    This environment extends the standard ManagerBasedRLEnv to add
    TacEx GelSight tactile sensor simulation. It:

    1. Initializes TacEx sensors on the robot fingertips
    2. Steps TacEx after each physics step
    3. Resets TacEx on environment reset

    The TacEx simulation pipeline follows:
    - PhysX step (handled by Isaac Lab)
    - GIPC soft body step (gelpad deformation)
    - Tactile image generation (Taxim RGB + FOTS markers)

    Example usage:
        ```python
        import gymnasium as gym

        env = gym.make(
            "Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0",
            num_envs=16,
        )

        obs, info = env.reset()
        # obs["tactile"] contains tactile RGB images
        # obs["tactile_state"] contains contact forces

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ```
    """

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the TacEx-enabled environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for the environment.
            **kwargs: Additional arguments passed to parent.
        """
        # Initialize parent environment
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize TacEx manager
        self._init_tacex()

    def _init_tacex(self):
        """Initialize the TacEx tactile simulation manager."""
        # Check if TacEx is enabled in config
        if not getattr(self.cfg, "tacex_enabled", True):
            self.tacex = None
            print("[TacExEnv] TacEx disabled in config.")
            return

        # Get sensor configurations from config
        sensor_cfg_dicts = getattr(self.cfg, "tacex_sensor_cfgs", [])

        if not sensor_cfg_dicts:
            # Default sensor configuration for Franka parallel gripper
            sensor_cfg_dicts = [
                {
                    "name": "left_finger_tactile",
                    "sensor_case_prim": "{ENV_REGEX_NS}/Robot/panda_leftfinger/GelSightMini_Case",
                    "gelpad_prim": "{ENV_REGEX_NS}/Robot/panda_leftfinger/GelSightMini_Gelpad",
                },
                {
                    "name": "right_finger_tactile",
                    "sensor_case_prim": "{ENV_REGEX_NS}/Robot/panda_rightfinger/GelSightMini_Case",
                    "gelpad_prim": "{ENV_REGEX_NS}/Robot/panda_rightfinger/GelSightMini_Gelpad",
                },
            ]

        # Convert dict configs to TacExSensorConfig objects
        sensor_cfgs = []
        image_size = getattr(self.cfg, "tacex_image_size", 64)
        for cfg_dict in sensor_cfg_dicts:
            sensor_cfgs.append(
                TacExSensorConfig(
                    name=cfg_dict.get("name", "tactile_sensor"),
                    sensor_case_prim=cfg_dict.get("sensor_case_prim", ""),
                    gelpad_prim=cfg_dict.get("gelpad_prim", ""),
                    image_height=image_size,
                    image_width=image_size,
                )
            )

        # Create TacEx manager
        self.tacex = TacExManager(
            sim=self.sim,
            scene=self.scene,
            sensor_cfgs=sensor_cfgs,
            num_envs=self.num_envs,
            device=self.device,
        )

        print(f"[TacExEnv] Initialized TacEx with {len(sensor_cfgs)} sensors:")
        for cfg in sensor_cfgs:
            print(f"  - {cfg.name}")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-physics step processing.

        Args:
            actions: Actions from the policy.
        """
        super()._pre_physics_step(actions)

    def _post_physics_step(self) -> None:
        """Post-physics step processing including TacEx update."""
        # Step TacEx simulation after PhysX
        if self.tacex is not None:
            self.tacex.step(self.physics_dt)

        # Call parent post-physics processing
        super()._post_physics_step()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset environments by indices.

        Args:
            env_ids: Indices of environments to reset.
        """
        # Reset parent environment
        super()._reset_idx(env_ids)

        # Reset TacEx for specified environments
        if self.tacex is not None:
            self.tacex.reset(env_ids)

    def close(self) -> None:
        """Close the environment and cleanup TacEx resources."""
        # Cleanup TacEx if needed
        if self.tacex is not None:
            # TacEx cleanup would go here
            pass

        # Close parent environment
        super().close()

    def get_tactile_observations(self) -> dict[str, torch.Tensor]:
        """Get all tactile observations as a dictionary.

        Returns:
            Dictionary containing:
            - "tactile_rgb_left": Left finger tactile RGB (N, H, W, 3)
            - "tactile_rgb_right": Right finger tactile RGB (N, H, W, 3)
            - "marker_flow_left": Left finger marker flow (N, H, W, 2)
            - "marker_flow_right": Right finger marker flow (N, H, W, 2)
            - "contact_force_left": Left finger contact force (N, 3)
            - "contact_force_right": Right finger contact force (N, 3)
        """
        if self.tacex is None:
            # Return zeros if TacEx not initialized
            img_size = getattr(self.cfg, "tacex_image_size", 64)
            return {
                "tactile_rgb_left": torch.zeros(
                    (self.num_envs, img_size, img_size, 3), device=self.device
                ),
                "tactile_rgb_right": torch.zeros(
                    (self.num_envs, img_size, img_size, 3), device=self.device
                ),
                "marker_flow_left": torch.zeros(
                    (self.num_envs, img_size, img_size, 2), device=self.device
                ),
                "marker_flow_right": torch.zeros(
                    (self.num_envs, img_size, img_size, 2), device=self.device
                ),
                "contact_force_left": torch.zeros((self.num_envs, 3), device=self.device),
                "contact_force_right": torch.zeros((self.num_envs, 3), device=self.device),
            }

        return {
            "tactile_rgb_left": self.tacex.get_tactile_rgb("left_finger_tactile"),
            "tactile_rgb_right": self.tacex.get_tactile_rgb("right_finger_tactile"),
            "marker_flow_left": self.tacex.get_marker_flow("left_finger_tactile"),
            "marker_flow_right": self.tacex.get_marker_flow("right_finger_tactile"),
            "contact_force_left": self.tacex.get_contact_force("left_finger_tactile"),
            "contact_force_right": self.tacex.get_contact_force("right_finger_tactile"),
        }



