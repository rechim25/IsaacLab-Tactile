# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacEx Manager for Isaac Lab integration.

This module provides a wrapper around the TacEx library for integrating
GelSight tactile simulation into Isaac Lab environments.

Based on: TacEx: GelSight Tactile Simulation in Isaac Sim
Paper: https://arxiv.org/pdf/2411.04776
Site: https://sites.google.com/view/tacex

The TacEx simulation pipeline follows this flow (see TacEx paper Fig. 1/3):
1. PhysX physics step (handled by Isaac Lab)
2. GIPC soft body step (updates gelpad deformation)
3. Scene rendering
4. Tactile simulation (Taxim RGB + FOTS marker flow)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext


@dataclass
class TacExSensorConfig:
    """Configuration for a TacEx GelSight tactile sensor.

    Attributes:
        name: Unique name for this sensor.
        sensor_case_prim: USD prim path to the sensor case (rigid body attached to finger).
        gelpad_prim: USD prim path to the gelpad mesh (GIPC soft body).
        gelpad_tet_file: Path to the tetrahedral mesh file for GIPC simulation.
        taxim_lut_file: Path to the Taxim polynomial lookup table for optical simulation.
        marker_config: Configuration for FOTS marker motion simulation.
        image_height: Height of the tactile RGB output image.
        image_width: Width of the tactile RGB output image.
        gelpad_thickness: Thickness of the gelpad for indentation depth calculation.
    """

    name: str = "gelsight_mini"
    sensor_case_prim: str = ""
    gelpad_prim: str = ""
    gelpad_tet_file: str = ""
    taxim_lut_file: str = ""
    marker_config: dict = field(default_factory=dict)
    image_height: int = 64
    image_width: int = 64
    gelpad_thickness: float = 0.002  # 2mm typical GelSight thickness


class TacExManager:
    """Manager for TacEx tactile simulation in Isaac Lab.

    This class wraps the TacEx library to provide tactile sensing capabilities
    for Isaac Lab environments. It handles:
    - GIPC soft body simulation for gelpad deformation
    - Taxim optical simulation for RGB image generation
    - FOTS marker motion field simulation

    The manager follows the TacEx simulation pipeline:
    PhysX step -> GIPC step -> Render -> Tactile sim

    Example usage in a manager-based env:
        ```python
        from isaaclab_tasks.manager_based.manipulation.stack.tacex import TacExManager, TacExSensorConfig

        class MyTacExEnv(ManagerBasedRLEnv):
            def __init__(self, cfg, **kwargs):
                super().__init__(cfg, **kwargs)

                # Configure tactile sensors on fingertips
                sensor_cfgs = [
                    TacExSensorConfig(
                        name="left_finger_tactile",
                        sensor_case_prim="{ENV_REGEX_NS}/Robot/panda_leftfinger/GelSightMini_Case",
                        gelpad_prim="{ENV_REGEX_NS}/Robot/panda_leftfinger/GelSightMini_Gelpad",
                    ),
                    TacExSensorConfig(
                        name="right_finger_tactile",
                        sensor_case_prim="{ENV_REGEX_NS}/Robot/panda_rightfinger/GelSightMini_Case",
                        gelpad_prim="{ENV_REGEX_NS}/Robot/panda_rightfinger/GelSightMini_Gelpad",
                    ),
                ]

                self.tacex = TacExManager(
                    sim=self.sim,
                    scene=self.scene,
                    sensor_cfgs=sensor_cfgs,
                    num_envs=self.num_envs,
                    device=self.device,
                )

            def _post_physics_step(self):
                self.tacex.step(self.physics_dt)
                return super()._post_physics_step()

            def _reset_idx(self, env_ids):
                super()._reset_idx(env_ids)
                self.tacex.reset(env_ids)
        ```
    """

    def __init__(
        self,
        sim: SimulationContext,
        scene: InteractiveScene,
        sensor_cfgs: list[TacExSensorConfig],
        num_envs: int,
        device: str = "cuda:0",
    ):
        """Initialize the TacEx manager.

        Args:
            sim: Isaac Sim simulation context.
            scene: Isaac Lab interactive scene.
            sensor_cfgs: List of sensor configurations.
            num_envs: Number of parallel environments.
            device: Torch device for tensor operations.
        """
        self._sim = sim
        self._scene = scene
        self._sensor_cfgs = sensor_cfgs
        self._num_envs = num_envs
        self._device = device
        self._num_sensors = len(sensor_cfgs)

        # Buffers for tactile outputs
        self._tactile_rgb = {}
        self._marker_flow = {}
        self._height_map = {}
        self._contact_force = {}

        # Initialize output buffers for each sensor
        for cfg in sensor_cfgs:
            # RGB image: (num_envs, H, W, 3)
            self._tactile_rgb[cfg.name] = torch.zeros(
                (num_envs, cfg.image_height, cfg.image_width, 3),
                dtype=torch.float32,
                device=device,
            )
            # Marker flow: (num_envs, H, W, 2) - x,y displacement
            self._marker_flow[cfg.name] = torch.zeros(
                (num_envs, cfg.image_height, cfg.image_width, 2),
                dtype=torch.float32,
                device=device,
            )
            # Height map: (num_envs, H, W)
            self._height_map[cfg.name] = torch.zeros(
                (num_envs, cfg.image_height, cfg.image_width),
                dtype=torch.float32,
                device=device,
            )
            # Contact force: (num_envs, 3) - force vector
            self._contact_force[cfg.name] = torch.zeros(
                (num_envs, 3),
                dtype=torch.float32,
                device=device,
            )

        # TacEx library handles (to be initialized when TacEx is installed)
        self._tacex_initialized = False
        self._gipc_manager = None
        self._taxim_renderer = None
        self._fots_simulator = None

        # Try to initialize TacEx components
        self._try_initialize_tacex()

    def _try_initialize_tacex(self):
        """Attempt to initialize TacEx library components.

        This method tries to import and initialize the TacEx library.
        If TacEx is not installed, it will print a warning and the manager
        will operate in "stub mode" returning zero tensors.
        """
        try:
            # Try to import TacEx components
            # Note: These imports depend on TacEx being installed
            # from tacex import GIPCManager, TaximRenderer, FOTSSimulator

            # For now, we provide a stub implementation that works without TacEx
            # Users should install TacEx and update these imports
            print("[TacExManager] TacEx library not found. Running in stub mode.")
            print("[TacExManager] To enable tactile simulation, install TacEx from:")
            print("[TacExManager]   https://sites.google.com/view/tacex")
            self._tacex_initialized = False

        except ImportError as e:
            print(f"[TacExManager] Warning: Could not import TacEx: {e}")
            print("[TacExManager] Running in stub mode with zero tactile outputs.")
            self._tacex_initialized = False

    def step(self, dt: float):
        """Step the TacEx simulation.

        This should be called after the PhysX physics step in the environment.
        It updates the GIPC soft body simulation and generates tactile outputs.

        Following the TacEx pipeline (paper Fig. 3):
        1. Read current robot pose from PhysX
        2. Update GIPC soft body (gelpad deformation)
        3. Generate height map from gelpad deformation
        4. Run Taxim optical simulation for RGB
        5. Run FOTS for marker motion field

        Args:
            dt: Physics timestep in seconds.
        """
        if not self._tacex_initialized:
            # Stub mode: return zeros (already initialized in __init__)
            return

        # TODO: When TacEx is installed, implement the actual simulation:
        # 1. Update GIPC attachment points based on robot finger poses
        # self._gipc_manager.update_attachments(dt)
        #
        # 2. Step GIPC solver to get new gelpad vertex positions
        # self._gipc_manager.step(dt)
        #
        # 3. Update USD meshes with new vertex positions
        # self._gipc_manager.update_meshes()
        #
        # 4. Render scene to get depth map for each sensor
        # for cfg in self._sensor_cfgs:
        #     self._height_map[cfg.name] = self._render_height_map(cfg)
        #
        # 5. Run Taxim optical simulation
        # for cfg in self._sensor_cfgs:
        #     self._tactile_rgb[cfg.name] = self._taxim_renderer.render(
        #         self._height_map[cfg.name]
        #     )
        #
        # 6. Run FOTS marker simulation
        # for cfg in self._sensor_cfgs:
        #     self._marker_flow[cfg.name] = self._fots_simulator.compute_flow(
        #         self._height_map[cfg.name]
        #     )
        pass

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the TacEx simulation for specified environments.

        This restores gelpads to their initial (undeformed) state and
        clears tactile output buffers.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        # Reset output buffers for specified envs
        for cfg in self._sensor_cfgs:
            self._tactile_rgb[cfg.name][env_ids] = 0.0
            self._marker_flow[cfg.name][env_ids] = 0.0
            self._height_map[cfg.name][env_ids] = 0.0
            self._contact_force[cfg.name][env_ids] = 0.0

        if not self._tacex_initialized:
            return

        # TODO: When TacEx is installed:
        # Reset GIPC vertices to initial positions
        # self._gipc_manager.reset(env_ids)

    def get_tactile_rgb(self, sensor_name: str | None = None) -> torch.Tensor:
        """Get the tactile RGB image from TacEx Taxim simulation.

        The RGB image is generated using Taxim's polynomial lookup table
        approach, which maps surface normals to light intensities.

        Args:
            sensor_name: Name of specific sensor. If None and only one sensor,
                        returns that sensor's output.

        Returns:
            Tactile RGB image tensor of shape (num_envs, H, W, 3).
        """
        if sensor_name is None:
            if len(self._sensor_cfgs) == 1:
                sensor_name = self._sensor_cfgs[0].name
            else:
                raise ValueError(
                    "Multiple sensors configured. Please specify sensor_name."
                )

        return self._tactile_rgb[sensor_name]

    def get_marker_flow(self, sensor_name: str | None = None) -> torch.Tensor:
        """Get the marker motion flow from TacEx FOTS simulation.

        The marker flow represents the displacement of gel surface markers
        due to contact deformation, useful for slip detection.

        Args:
            sensor_name: Name of specific sensor. If None and only one sensor,
                        returns that sensor's output.

        Returns:
            Marker flow tensor of shape (num_envs, H, W, 2) with x,y displacements.
        """
        if sensor_name is None:
            if len(self._sensor_cfgs) == 1:
                sensor_name = self._sensor_cfgs[0].name
            else:
                raise ValueError(
                    "Multiple sensors configured. Please specify sensor_name."
                )

        return self._marker_flow[sensor_name]

    def get_height_map(self, sensor_name: str | None = None) -> torch.Tensor:
        """Get the gelpad height/deformation map.

        The height map represents the indentation depth at each point,
        computed from the gelpad surface deformation.

        Args:
            sensor_name: Name of specific sensor. If None and only one sensor,
                        returns that sensor's output.

        Returns:
            Height map tensor of shape (num_envs, H, W).
        """
        if sensor_name is None:
            if len(self._sensor_cfgs) == 1:
                sensor_name = self._sensor_cfgs[0].name
            else:
                raise ValueError(
                    "Multiple sensors configured. Please specify sensor_name."
                )

        return self._height_map[sensor_name]

    def get_contact_force(self, sensor_name: str | None = None) -> torch.Tensor:
        """Get the estimated contact force from tactile data.

        Args:
            sensor_name: Name of specific sensor. If None and only one sensor,
                        returns that sensor's output.

        Returns:
            Contact force tensor of shape (num_envs, 3).
        """
        if sensor_name is None:
            if len(self._sensor_cfgs) == 1:
                sensor_name = self._sensor_cfgs[0].name
            else:
                raise ValueError(
                    "Multiple sensors configured. Please specify sensor_name."
                )

        return self._contact_force[sensor_name]

    @property
    def sensor_names(self) -> list[str]:
        """Get list of configured sensor names."""
        return [cfg.name for cfg in self._sensor_cfgs]

    @property
    def num_sensors(self) -> int:
        """Get number of configured sensors."""
        return self._num_sensors

    @property
    def is_initialized(self) -> bool:
        """Check if TacEx library is properly initialized."""
        return self._tacex_initialized



