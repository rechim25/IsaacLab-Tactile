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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE, OBS_STR

from .pipeline import ActionProcessorStep, ObservationProcessorStep, ProcessorStepRegistry, TransitionKey


@dataclass
@ProcessorStepRegistry.register(name="libero_processor")
class LiberoProcessorStep(ObservationProcessorStep):
    """
    Processes LIBERO observations into the LeRobot format.

    This step handles the specific observation structure from LIBERO environments,
    which includes nested robot_state dictionaries and image observations.

    **State Processing:**
    -   Processes the `robot_state` dictionary which contains nested end-effector,
        gripper, and joint information.
    -   Extracts and concatenates:
        - End-effector position (3D)
        - End-effector quaternion converted to axis-angle (3D)
        - Gripper joint positions (2D)
    -   Maps the concatenated state to `"observation.state"`.

    **Image Processing:**
    -   Rotates images by 180 degrees by flipping both height and width dimensions.
    -   This accounts for the HuggingFaceVLA/libero camera orientation convention.
    """

    def _process_observation(self, observation):
        """
        Processes both image and robot_state observations from LIBERO.
        """
        processed_obs = observation.copy()
        for key in list(processed_obs.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                img = processed_obs[key]

                # Flip both H and W
                img = torch.flip(img, dims=[2, 3])

                processed_obs[key] = img
        # Process robot_state into a flat state vector
        observation_robot_state_str = OBS_PREFIX + "robot_state"
        if observation_robot_state_str in processed_obs:
            robot_state = processed_obs.pop(observation_robot_state_str)

            # Extract components
            eef_pos = robot_state["eef"]["pos"]  # (B, 3,)
            eef_quat = robot_state["eef"]["quat"]  # (B, 4,)
            gripper_qpos = robot_state["gripper"]["qpos"]  # (B, 2,)

            # Convert quaternion to axis-angle
            eef_axisangle = self._quat2axisangle(eef_quat)  # (B, 3)
            # Concatenate into a single state vector
            state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)

            # ensure float32
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            processed_obs[OBS_STATE] = state
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transforms feature keys from the LIBERO format to the LeRobot standard.
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}

        # copy over non-STATE features
        for ft, feats in features.items():
            if ft != PipelineFeatureType.STATE:
                new_features[ft] = feats.copy()

        # rebuild STATE features
        state_feats = {}

        # add our new flattened state
        state_feats[OBS_STATE] = PolicyFeature(
            key=OBS_STATE,
            shape=(8,),  # [eef_pos(3), axis_angle(3), gripper(2)]
            dtype="float32",
            description=("Concatenated end-effector position (3), axis-angle (3), and gripper qpos (2)."),
        )

        new_features[PipelineFeatureType.STATE] = state_feats

        return new_features

    def observation(self, observation):
        return self._process_observation(observation)

    def _quat2axisangle(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert batched quaternions to axis-angle format.
        Only accepts torch tensors of shape (B, 4).

        Args:
            quat (Tensor): (B, 4) tensor of quaternions in (x, y, z, w) format

        Returns:
            Tensor: (B, 3) axis-angle vectors

        Raises:
            TypeError: if input is not a torch tensor
            ValueError: if shape is not (B, 4)
        """

        if not isinstance(quat, torch.Tensor):
            raise TypeError(f"_quat2axisangle expected a torch.Tensor, got {type(quat)}")

        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError(f"_quat2axisangle expected shape (B, 4), got {tuple(quat.shape)}")

        quat = quat.to(dtype=torch.float32)
        device = quat.device
        batch_size = quat.shape[0]

        w = quat[:, 3].clamp(-1.0, 1.0)

        den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

        result = torch.zeros((batch_size, 3), device=device)

        mask = den > 1e-10

        if mask.any():
            angle = 2.0 * torch.acos(w[mask])  # (M,)
            axis = quat[mask, :3] / den[mask].unsqueeze(1)
            result[mask] = axis * angle.unsqueeze(1)

        return result


@dataclass
@ProcessorStepRegistry.register(name="isaaclab_arena_processor")
class IsaaclabArenaProcessorStep(ObservationProcessorStep):
    """
    Processes IsaacLab Arena observations into LeRobot format.

    **State Processing:**
    - Extracts state components from obs["policy"] based on `state_keys`.
    - Concatenates into a flat vector mapped to "observation.state".

    **Image Processing:**
    - Extracts images from obs["camera_obs"] based on `camera_keys`.
    - Converts from (B, H, W, C) uint8 to (B, C, H, W) float32 [0, 1].
    - Maps to "observation.images.<camera_name>".
    """

    # Configurable from IsaacLabEnv config / cli args: --env.state_keys="robot_joint_pos,left_eef_pos"
    state_keys: tuple[str, ...]

    # Configurable from IsaacLabEnv config / cli args: --env.camera_keys="robot_pov_cam_rgb"
    camera_keys: tuple[str, ...]

    def _process_observation(self, observation):
        """
        Processes both image and policy state observations from IsaacLab Arena.
        """
        processed_obs = {}

        if f"{OBS_STR}.camera_obs" in observation:
            camera_obs = observation[f"{OBS_STR}.camera_obs"]

            for cam_name, img in camera_obs.items():
                if cam_name not in self.camera_keys:
                    continue

                img = img.permute(0, 3, 1, 2).contiguous()
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                elif img.dtype != torch.float32:
                    img = img.float()

                processed_obs[f"{OBS_IMAGES}.{cam_name}"] = img

        # Process policy state -> observation.state
        if f"{OBS_STR}.policy" in observation:
            policy_obs = observation[f"{OBS_STR}.policy"]

            # Collect state components in order
            state_components = []
            for key in self.state_keys:
                if key in policy_obs:
                    component = policy_obs[key]
                    # Flatten extra dims: (B, N, M) -> (B, N*M)
                    if component.dim() > 2:
                        batch_size = component.shape[0]
                        component = component.view(batch_size, -1)
                    state_components.append(component)

            if state_components:
                state = torch.cat(state_components, dim=-1)
                state = state.float()
                processed_obs[OBS_STATE] = state

        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Not used for policy evaluation."""
        return features

    def observation(self, observation):
        return self._process_observation(observation)


# =============================================================================
# IsaacLab Tactile Policy Processors
# =============================================================================


@dataclass
@ProcessorStepRegistry.register(name="isaaclab_tactile_policy_obs")
class IsaacLabTactilePolicyObservationProcessorStep(ObservationProcessorStep):
    """
    Processes IsaacLab tactile observations into the canonical policy format.

    This processor transforms raw IsaacLab observations (in world frame) into
    the base-frame policy convention used by SmolVLA with tactile integration.

    **State Convention (11D):**
        [eef_pos_b(3), eef_rot6d_b(6), gripper_qpos(2)]
        - eef_pos_b: EE position in robot base frame
        - eef_rot6d_b: EE orientation in base frame as Rot6D
        - gripper_qpos: 2D gripper joint positions

    **Required Raw Observation Keys:**
        - eef_pos_w or eef_pos: EE position (world or env-local frame)
        - eef_quat_w or eef_quat: EE quaternion (x,y,z,w)
        - base_pos_w or base_pos: Base/root position (world or env-local frame)
        - base_quat_w or base_quat: Base/root quaternion (x,y,z,w)
        - gripper_qpos or gripper_pos: Gripper joint positions (2D)

    **Optional Tactile Keys:**
        - tactile_force_grid: Force grid tensor (N, H, W, 3) or (B, N, H, W, 3)
        - tactile_resultant_force: Resultant force (N, 3) or (B, N, 3)
    """

    # Keys for finding raw observation data
    eef_pos_key: str = "eef_pos"
    eef_quat_key: str = "eef_quat"
    base_pos_key: str = "base_pos"
    base_quat_key: str = "base_quat"
    gripper_qpos_key: str = "gripper_qpos"

    # Tactile keys
    tactile_force_grid_key: str = "tactile_force_grid"
    tactile_resultant_force_key: str | None = None

    # Camera keys (comma-separated)
    camera_keys: str = ""

    # Output tactile key names
    output_tactile_force_grid_key: str = "observation.tactile.force_grid"
    output_tactile_resultant_force_key: str = "observation.tactile.resultant_force"

    def _process_observation(self, observation: dict) -> dict:
        """
        Transform raw IsaacLab observations to policy format.
        """
        from lerobot.isaaclab_tactile.policy_io import encode_state_isaaclab_to_policy

        processed_obs = {}

        # Find and extract raw observation components
        raw_obs = self._extract_raw_obs(observation)

        # Check we have all required fields
        eef_pos = raw_obs.get("eef_pos")
        eef_quat = raw_obs.get("eef_quat")
        base_pos = raw_obs.get("base_pos")
        base_quat = raw_obs.get("base_quat")
        gripper_qpos = raw_obs.get("gripper_qpos")

        if eef_pos is not None and eef_quat is not None and gripper_qpos is not None:
            # Default base pose to identity if not provided (fixed base assumption)
            if base_pos is None:
                base_pos = torch.zeros(3) if isinstance(eef_pos, torch.Tensor) else np.zeros(3)
            if base_quat is None:
                base_quat = (
                    torch.tensor([0.0, 0.0, 0.0, 1.0])
                    if isinstance(eef_pos, torch.Tensor)
                    else np.array([0.0, 0.0, 0.0, 1.0])
                )

            # Convert to numpy for processing
            eef_pos_np = self._to_numpy(eef_pos)
            eef_quat_np = self._to_numpy(eef_quat)
            base_pos_np = self._to_numpy(base_pos)
            base_quat_np = self._to_numpy(base_quat)
            gripper_qpos_np = self._to_numpy(gripper_qpos)

            # Encode to policy state (11D)
            state_np = encode_state_isaaclab_to_policy(
                eef_pos_w=eef_pos_np,
                eef_quat_w=eef_quat_np,
                gripper_qpos=gripper_qpos_np,
                base_pos_w=base_pos_np,
                base_quat_w=base_quat_np,
            )

            # Convert back to tensor
            state = torch.from_numpy(state_np).float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            processed_obs[OBS_STATE] = state

        # Process images
        self._process_images(observation, processed_obs)

        # Process tactile data
        self._process_tactile(raw_obs, processed_obs)

        return processed_obs

    def _extract_raw_obs(self, observation: dict) -> dict:
        """Extract raw observation values from various possible key locations."""
        raw_obs = {}

        # Helper to find a value in nested observation dict
        def find_value(keys: list[str]) -> Any:
            for key in keys:
                # Try direct key
                if key in observation:
                    return observation[key]
                # Try with observation. prefix
                obs_key = f"{OBS_STR}.{key}"
                if obs_key in observation:
                    return observation[obs_key]
                # Try in robot_state dict
                robot_state_key = f"{OBS_PREFIX}robot_state"
                if robot_state_key in observation:
                    rs = observation[robot_state_key]
                    if isinstance(rs, dict) and key in rs:
                        return rs[key]
            return None

        # Extract each component with fallback keys
        raw_obs["eef_pos"] = find_value([self.eef_pos_key, "eef_pos_w", "ee_pos", "eef_pos"])
        raw_obs["eef_quat"] = find_value([self.eef_quat_key, "eef_quat_w", "ee_quat", "eef_quat"])
        raw_obs["base_pos"] = find_value([self.base_pos_key, "base_pos_w", "root_pos_w"])
        raw_obs["base_quat"] = find_value([self.base_quat_key, "base_quat_w", "root_quat_w"])
        raw_obs["gripper_qpos"] = find_value(
            [self.gripper_qpos_key, "gripper_pos", "gripper_qpos", "gripper_joint_pos"]
        )

        # Tactile data
        raw_obs["tactile_force_grid"] = find_value([self.tactile_force_grid_key, "force_grid"])
        if self.tactile_resultant_force_key:
            raw_obs["tactile_resultant_force"] = find_value(
                [self.tactile_resultant_force_key, "resultant_force"]
            )

        return raw_obs

    def _process_images(self, observation: dict, processed_obs: dict) -> None:
        """Process and copy image observations."""
        # Copy existing image keys
        for key in list(observation.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                processed_obs[key] = observation[key]

        # Process camera_keys if specified
        if self.camera_keys:
            camera_keys_list = [k.strip() for k in self.camera_keys.split(",") if k.strip()]
            camera_obs_key = f"{OBS_STR}.camera_obs"
            if camera_obs_key in observation:
                camera_obs = observation[camera_obs_key]
                for cam_name in camera_keys_list:
                    if cam_name in camera_obs:
                        img = camera_obs[cam_name]
                        # Convert BHWC -> BCHW if needed
                        if img.dim() == 4 and img.shape[-1] in (1, 3, 4):
                            img = img.permute(0, 3, 1, 2).contiguous()
                        if img.dtype == torch.uint8:
                            img = img.float() / 255.0
                        processed_obs[f"{OBS_IMAGES}.{cam_name}"] = img

    def _process_tactile(self, raw_obs: dict, processed_obs: dict) -> None:
        """Process tactile observations."""
        force_grid = raw_obs.get("tactile_force_grid")
        if force_grid is not None:
            if isinstance(force_grid, np.ndarray):
                force_grid = torch.from_numpy(force_grid)
            force_grid = force_grid.float()
            processed_obs[self.output_tactile_force_grid_key] = force_grid

        resultant_force = raw_obs.get("tactile_resultant_force")
        if resultant_force is not None:
            if isinstance(resultant_force, np.ndarray):
                resultant_force = torch.from_numpy(resultant_force)
            resultant_force = resultant_force.float()
            processed_obs[self.output_tactile_resultant_force_key] = resultant_force

    def _to_numpy(self, x: Any) -> np.ndarray:
        """Convert tensor or array to numpy."""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    def observation(self, observation: dict) -> dict:
        return self._process_observation(observation)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Transform features to reflect 11D state output."""
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}

        # Copy over non-STATE features
        for ft, feats in features.items():
            if ft != PipelineFeatureType.STATE:
                new_features[ft] = feats.copy()

        # Rebuild STATE features with 11D state
        state_feats = {}
        state_feats[OBS_STATE] = PolicyFeature(
            key=OBS_STATE,
            shape=(11,),  # [eef_pos_b(3), eef_rot6d_b(6), gripper_qpos(2)]
            dtype="float32",
            description=(
                "IsaacLab tactile policy state: EE position in base frame (3), "
                "EE Rot6D in base frame (6), gripper qpos (2)."
            ),
        )
        new_features[PipelineFeatureType.STATE] = state_feats

        return new_features


@dataclass
@ProcessorStepRegistry.register(name="isaaclab_tactile_policy_action")
class IsaacLabTactilePolicyActionProcessorStep(ActionProcessorStep):
    """
    Converts policy actions from base frame to world frame for IsaacLab.

    This processor is used as an env_postprocessor during evaluation to transform
    the policy's output actions (in robot base frame) to the world frame that
    IsaacLab controllers expect.

    **Policy Action Convention (7D):**
        [Δpos_b(3), Δaxis_angle_b(3), gripper(1)]
        - Δpos_b: EE translation delta in robot base frame
        - Δaxis_angle_b: EE rotation delta in robot base frame (axis-angle)
        - gripper: Gripper command scalar

    **IsaacLab Action Output (7D):**
        [Δpos_w(3), Δrot_w(3), gripper(1)]
        - Deltas transformed to world frame for the controller

    The processor requires access to the current base pose to perform the
    frame transformation. This can be provided via:
        - Storing base_quat_w in the transition's complementary data
        - Assuming identity base pose (fixed base robot)
    """

    # Key in complementary_data where base quaternion is stored
    base_quat_key: str = "base_quat_w"

    # If True, assume identity base pose (no rotation needed)
    assume_fixed_base: bool = True

    def action(self, action: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Transform action from base frame to world frame.

        Note: This simplified version assumes a fixed base (identity rotation).
        For mobile bases, the base_quat_w must be passed via complementary_data.
        """
        if self.assume_fixed_base:
            # No transformation needed for fixed base
            return action

        # For non-fixed base, this would need the base quaternion
        # which must be provided in the transition's complementary_data
        # This is handled in __call__ for full transition processing
        return action

    def __call__(self, transition: dict) -> dict:
        """
        Process the full transition, transforming action from base to world frame.
        """
        from lerobot.isaaclab_tactile.policy_io import decode_action_policy_to_isaaclab_7d

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is None:
            return new_transition

        # Get base quaternion
        base_quat_w = None
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp_data and self.base_quat_key in comp_data:
            base_quat_w = comp_data[self.base_quat_key]

        if base_quat_w is None:
            if self.assume_fixed_base:
                # Use identity quaternion
                if isinstance(action, torch.Tensor):
                    base_quat_w = torch.tensor([0.0, 0.0, 0.0, 1.0], device=action.device)
                else:
                    base_quat_w = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                raise ValueError(
                    f"Base quaternion not found in complementary_data['{self.base_quat_key}'] "
                    "and assume_fixed_base=False"
                )

        # Convert to numpy for processing
        is_tensor = isinstance(action, torch.Tensor)
        device = action.device if is_tensor else None
        dtype = action.dtype if is_tensor else None

        action_np = action.cpu().numpy() if is_tensor else np.asarray(action)
        base_quat_np = (
            base_quat_w.cpu().numpy() if isinstance(base_quat_w, torch.Tensor) else np.asarray(base_quat_w)
        )

        # Transform action to world frame
        action_world_np = decode_action_policy_to_isaaclab_7d(action_np, base_quat_np)

        # Convert back to original format
        if is_tensor:
            action_world = torch.from_numpy(action_world_np).to(device=device, dtype=dtype)
        else:
            action_world = action_world_np

        new_transition[TransitionKey.ACTION] = action_world
        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "base_quat_key": self.base_quat_key,
            "assume_fixed_base": self.assume_fixed_base,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step only changes action values (frame transform), not the schema.
        """
        return features
