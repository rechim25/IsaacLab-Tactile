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
from typing import Literal

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "TACTILE": NormalizationMode.MEAN_STD,  # For tactile force grids
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to True in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    # ==================== Tactile Sensing Configuration ====================
    # Enable tactile sensing integration (DexGrasp-VLA paper arXiv:2511.00139v1)
    use_tactile: bool = False

    # Number of fingertips/tactile sensors (default 2 for gripper)
    num_fingertips: int = 2

    # Key for tactile force grid in observation dict (shape: N x 10 x 12 x 3)
    tactile_force_grid_key: str = "observation.tactile.force_grid"

    # Optional key for pre-computed resultant force (shape: N x 3)
    # If None, resultant force is computed from force_grid in the model
    tactile_resultant_force_key: str | None = None

    # How to organize tactile tokens in prefix
    # - "per_fingertip_per_branch": 2*N tokens (one force + one spatial per fingertip)
    # - "per_branch_pooled": 2 tokens (pooled force + pooled spatial across fingertips)
    tactile_token_mode: Literal["per_fingertip_per_branch", "per_branch_pooled"] = "per_fingertip_per_branch"

    # CAE latent dimension (paper uses 128)
    tactile_latent_dim: int = 128

    # Path to pretrained CAE weights (optional, for using a separately trained CAE)
    pretrained_cae_path: str | None = None

    # Whether to train the CAE encoder (if False, freeze CAE weights)
    train_tactile_cae: bool = True

    # ==================== Arm-Hand Feature Enhancement ====================
    # Enable Arm-Hand Feature Enhancement module (paper Sec. 3.4.1)
    use_arm_hand_feature_enhancement: bool = False

    # Indices of action dimensions corresponding to arm control
    # Default: first 6 dimensions (e.g., dx, dy, dz, drx, dry, drz)
    arm_indices: list[int] = field(default_factory=lambda: list(range(6)))

    # Indices of action dimensions corresponding to hand/gripper control
    # Default: 7th dimension (gripper)
    hand_indices: list[int] = field(default_factory=lambda: [6])

    # Weight for auxiliary losses (lambda in Eq. 13: L_total = L_main + lambda * (L_arm + L_hand))
    aux_loss_lambda: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

        # Tactile validation
        if self.use_tactile:
            if self.num_fingertips < 1:
                raise ValueError(f"num_fingertips must be >= 1, got {self.num_fingertips}")
            if self.tactile_token_mode not in ["per_fingertip_per_branch", "per_branch_pooled"]:
                raise ValueError(
                    f"tactile_token_mode must be 'per_fingertip_per_branch' or 'per_branch_pooled', "
                    f"got {self.tactile_token_mode}"
                )

        # Arm-Hand Feature Enhancement validation
        if self.use_arm_hand_feature_enhancement:
            if len(self.arm_indices) == 0 and len(self.hand_indices) == 0:
                raise ValueError(
                    "When using arm_hand_feature_enhancement, at least one of arm_indices or hand_indices must be non-empty"
                )
            # Check for overlap
            overlap = set(self.arm_indices) & set(self.hand_indices)
            if overlap:
                raise ValueError(
                    f"arm_indices and hand_indices must not overlap, but found common indices: {overlap}"
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def tactile_features(self) -> dict[str, PolicyFeature]:
        """Get tactile features from input_features."""
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.TACTILE}
