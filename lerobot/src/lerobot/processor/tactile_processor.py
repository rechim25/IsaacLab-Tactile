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
Tactile sensor data processing steps for SmolVLA.

These processor steps handle the transformation of raw tactile force grid data
into the format expected by the TactileEmbedding module:
- Padding from raw grid size (e.g., 10x12) to 16x16
- Channel transposition to (C, H, W) format
- Resultant force computation from force grids
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry, RobotObservation


@dataclass
@ProcessorStepRegistry.register(name="tactile_force_grid_to_image")
class TactileForceGridToImageProcessor(ProcessorStep):
    """Processor step to convert raw tactile force grids to padded tactile images.

    Takes raw force grid tensors of shape (N, H, W, 3) where N is number of fingertips,
    H and W are spatial dimensions (e.g., 10x12), and 3 is force components (x, y, z).

    Outputs:
    - Pads spatially to (N, 3, target_size, target_size) format
    - Transposes channels to (N, C, H, W) PyTorch format

    Args:
        input_key: Key for the input force grid in observation dict
        output_key: Key for the output tactile image (if None, replaces input)
        input_shape: Expected shape of input (H, W) - default (10, 12)
        target_size: Target spatial size after padding - default 16
        device: Target device for processing
    """

    input_key: str = "observation.tactile.force_grid"
    output_key: str | None = None  # If None, replaces input_key
    input_shape: tuple[int, int] = (10, 12)
    target_size: int = 16
    device: torch.device | str | None = None

    def __post_init__(self):
        if self.output_key is None:
            self.output_key = self.input_key

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        if self.input_key not in observation:
            return new_transition

        force_grid = observation[self.input_key]

        # Convert to tensor if needed
        if not isinstance(force_grid, Tensor):
            force_grid = torch.as_tensor(force_grid)

        if self.device is not None:
            force_grid = force_grid.to(self.device)

        # Process the force grid
        # Expected input shape: (N, H, W, 3) or (B, N, H, W, 3)
        has_batch = force_grid.ndim == 5

        if has_batch:
            B, N, H, W, C = force_grid.shape
            force_grid = force_grid.view(B * N, H, W, C)
        else:
            N, H, W, C = force_grid.shape

        # Transpose to (N, C, H, W) or (B*N, C, H, W)
        force_grid = force_grid.permute(0, 3, 1, 2)

        # Compute padding
        pad_h = self.target_size - H
        pad_w = self.target_size - W

        # Symmetric padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Pad: (left, right, top, bottom)
        tactile_image = F.pad(
            force_grid, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
        )

        if has_batch:
            tactile_image = tactile_image.view(B, N, C, self.target_size, self.target_size)
        else:
            tactile_image = tactile_image.view(N, C, self.target_size, self.target_size)

        # Update observation
        new_observation = dict(observation)
        new_observation[self.output_key] = tactile_image
        new_transition[TransitionKey.OBSERVATION] = new_observation

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Update feature shapes if the key is being transformed."""
        if PipelineFeatureType.OBSERVATION not in features:
            return features

        obs_features = features[PipelineFeatureType.OBSERVATION]
        if self.input_key in obs_features:
            # Get original feature
            orig_feature = obs_features[self.input_key]

            # Create new feature with transformed shape
            # Original: (N, H, W, 3) -> (N, 3, target_size, target_size)
            if len(orig_feature.shape) == 4:
                N = orig_feature.shape[0]
                new_shape = (N, 3, self.target_size, self.target_size)
            else:
                new_shape = (3, self.target_size, self.target_size)

            new_obs_features = dict(obs_features)
            new_obs_features[self.output_key] = PolicyFeature(
                type=orig_feature.type, shape=new_shape
            )

            features = dict(features)
            features[PipelineFeatureType.OBSERVATION] = new_obs_features

        return features

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> TactileForceGridToImageProcessor:
        if device is not None:
            self.device = device
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {
            "input_key": self.input_key,
            "output_key": self.output_key,
            "input_shape": self.input_shape,
            "target_size": self.target_size,
        }


@dataclass
@ProcessorStepRegistry.register(name="tactile_resultant_force")
class TactileResultantForceProcessor(ProcessorStep):
    """Processor step to compute resultant force from force grids.

    Takes force grid tensors and computes the sum over spatial dimensions
    to get resultant force vectors per fingertip.

    Input: (N, H, W, 3) or (N, 3, H, W)
    Output: (N, 3)

    Args:
        input_key: Key for the input force grid in observation dict
        output_key: Key for the output resultant force
        input_is_image_format: If True, input is (N, C, H, W); if False, (N, H, W, C)
        device: Target device for processing
    """

    input_key: str = "observation.tactile.force_grid"
    output_key: str = "observation.tactile.resultant_force"
    input_is_image_format: bool = False  # (N, H, W, C) by default
    device: torch.device | str | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        if self.input_key not in observation:
            return new_transition

        force_grid = observation[self.input_key]

        # Convert to tensor if needed
        if not isinstance(force_grid, Tensor):
            force_grid = torch.as_tensor(force_grid)

        if self.device is not None:
            force_grid = force_grid.to(self.device)

        # Determine format and sum over spatial dims
        has_batch = force_grid.ndim == 5

        if has_batch:
            if self.input_is_image_format:
                # (B, N, C, H, W) -> sum over H, W
                resultant = force_grid.sum(dim=(3, 4))  # (B, N, C)
            else:
                # (B, N, H, W, C) -> sum over H, W
                resultant = force_grid.sum(dim=(2, 3))  # (B, N, C)
        else:
            if self.input_is_image_format:
                # (N, C, H, W) -> sum over H, W
                resultant = force_grid.sum(dim=(2, 3))  # (N, C)
            else:
                # (N, H, W, C) -> sum over H, W
                resultant = force_grid.sum(dim=(1, 2))  # (N, C)

        # Update observation
        new_observation = dict(observation)
        new_observation[self.output_key] = resultant
        new_transition[TransitionKey.OBSERVATION] = new_observation

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Add resultant force feature based on input feature."""
        if PipelineFeatureType.OBSERVATION not in features:
            return features

        obs_features = features[PipelineFeatureType.OBSERVATION]
        if self.input_key in obs_features:
            orig_feature = obs_features[self.input_key]

            # Determine number of fingertips from shape
            if len(orig_feature.shape) == 4:
                N = orig_feature.shape[0]
                new_shape = (N, 3)
            else:
                new_shape = (3,)

            new_obs_features = dict(obs_features)
            new_obs_features[self.output_key] = PolicyFeature(
                type=orig_feature.type, shape=new_shape
            )

            features = dict(features)
            features[PipelineFeatureType.OBSERVATION] = new_obs_features

        return features

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> TactileResultantForceProcessor:
        if device is not None:
            self.device = device
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {
            "input_key": self.input_key,
            "output_key": self.output_key,
            "input_is_image_format": self.input_is_image_format,
        }


@dataclass
@ProcessorStepRegistry.register(name="tactile_normalize")
class TactileNormalizeProcessor(ProcessorStep):
    """Processor step to normalize tactile force grids.

    Applies per-channel normalization to tactile force data using
    provided statistics (mean/std or min/max).

    Args:
        input_key: Key for the tactile data in observation dict
        stats: Dict with 'mean' and 'std' or 'min' and 'max' tensors
        mode: 'mean_std' or 'min_max'
        device: Target device for processing
    """

    input_key: str = "observation.tactile.force_grid"
    stats: dict[str, Any] = field(default_factory=dict)
    mode: str = "mean_std"  # or "min_max"
    eps: float = 1e-8
    device: torch.device | str | None = None

    _tensor_stats: dict[str, Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.stats:
            self._tensor_stats = {
                k: torch.as_tensor(v, device=self.device)
                for k, v in self.stats.items()
            }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        if self.input_key not in observation:
            return new_transition

        if not self._tensor_stats:
            return new_transition

        tactile = observation[self.input_key]

        if not isinstance(tactile, Tensor):
            tactile = torch.as_tensor(tactile)

        if self.device is not None:
            tactile = tactile.to(self.device)

        # Apply normalization
        if self.mode == "mean_std":
            mean = self._tensor_stats.get("mean")
            std = self._tensor_stats.get("std")
            if mean is not None and std is not None:
                # Ensure broadcasting works correctly
                if mean.device != tactile.device:
                    mean = mean.to(tactile.device)
                    std = std.to(tactile.device)
                tactile = (tactile - mean) / (std + self.eps)
        elif self.mode == "min_max":
            min_val = self._tensor_stats.get("min")
            max_val = self._tensor_stats.get("max")
            if min_val is not None and max_val is not None:
                if min_val.device != tactile.device:
                    min_val = min_val.to(tactile.device)
                    max_val = max_val.to(tactile.device)
                denom = max_val - min_val
                denom = torch.where(denom == 0, torch.tensor(self.eps), denom)
                tactile = 2 * (tactile - min_val) / denom - 1

        new_observation = dict(observation)
        new_observation[self.input_key] = tactile
        new_transition[TransitionKey.OBSERVATION] = new_observation

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> TactileNormalizeProcessor:
        if device is not None:
            self.device = device
            self._tensor_stats = {
                k: v.to(device) for k, v in self._tensor_stats.items()
            }
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return {k: v.cpu() for k, v in self._tensor_stats.items()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self._tensor_stats = {
            k: v.to(self.device) if self.device else v
            for k, v in state.items()
        }

    def get_config(self) -> dict[str, Any]:
        return {
            "input_key": self.input_key,
            "mode": self.mode,
            "eps": self.eps,
        }
