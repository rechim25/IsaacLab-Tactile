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
Tactile sensing modules for SmolVLA integration.

Implements the tactile feature extraction from DexGrasp-VLA (arXiv:2511.00139v1):
- TactileCAE: Convolutional autoencoder for spatial tactile embeddings
- TactileEmbedding: Full tactile feature extraction (resultant force + CAE latent)

The CAE architecture follows Sec. 3.2.2:
- Encoder: 3 conv layers (3x3, stride 2), 32/64/128 filters, BN + ReLU
- Latent: 2x2x128 -> flatten -> 128-d per fingertip
- Decoder: symmetric transposed convolutions for reconstruction
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TactileCAE(nn.Module):
    """Convolutional Autoencoder for tactile spatial embeddings.

    Takes a 16x16x3 tactile image (force components x, y, z) and encodes it
    into a 128-dimensional latent vector per fingertip.

    Architecture (from paper Sec. 3.2.2):
    - Encoder: 3 conv layers with 3x3 kernels, stride 2, filters 32/64/128
    - Each layer followed by BatchNorm and ReLU
    - Final 2x2x128 feature map flattened and projected to 128-d latent
    - Decoder mirrors encoder with transposed convolutions

    Args:
        in_channels: Number of input channels (default 3 for x,y,z forces)
        latent_dim: Dimension of latent vector (default 128)
        img_size: Input image size (default 16, assumes square input)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        img_size: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size

        # After 3 stride-2 convolutions: 16 -> 8 -> 4 -> 2
        self.feature_map_size = img_size // 8  # = 2 for img_size=16
        self.flatten_dim = 128 * self.feature_map_size * self.feature_map_size  # = 512

        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: 16x16x3 -> 8x8x32
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Layer 2: 8x8x32 -> 4x4x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 3: 4x4x64 -> 2x2x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Latent projection
        self.fc_encode = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder projection
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: 2x2x128 -> 4x4x64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 2: 4x4x64 -> 8x8x32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Layer 3: 8x8x32 -> 16x16x3
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encode tactile image to latent vector.

        Args:
            x: Tactile image of shape (B, C, H, W) or (B, N, C, H, W) for N fingertips

        Returns:
            Latent vector of shape (B, latent_dim) or (B, N, latent_dim)
        """
        # Handle batched fingertips
        has_fingertip_dim = x.ndim == 5
        if has_fingertip_dim:
            B, N, C, H, W = x.shape
            x = x.reshape(B * N, C, H, W)

        # Encode
        features = self.encoder(x)
        features = features.reshape(features.size(0), -1)
        latent = self.fc_encode(features)

        if has_fingertip_dim:
            latent = latent.reshape(B, N, -1)

        return latent

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector to tactile image.

        Args:
            z: Latent vector of shape (B, latent_dim) or (B, N, latent_dim)

        Returns:
            Reconstructed tactile image of shape (B, C, H, W) or (B, N, C, H, W)
        """
        has_fingertip_dim = z.ndim == 3
        if has_fingertip_dim:
            B, N, D = z.shape
            z = z.reshape(B * N, D)

        # Decode
        features = self.fc_decode(z)
        features = features.reshape(-1, 128, self.feature_map_size, self.feature_map_size)
        recon = self.decoder(features)

        if has_fingertip_dim:
            recon = recon.reshape(B, N, self.in_channels, self.img_size, self.img_size)

        return recon

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through encoder and decoder.

        Args:
            x: Tactile image of shape (B, C, H, W) or (B, N, C, H, W)

        Returns:
            Tuple of (latent, reconstruction)
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return latent, recon

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Compute reconstruction loss (Eq. 3 from paper).

        L_recon = (1 / 3HW) * sum over c,i,j of (F_{c,ij} - F_hat_{c,ij})^2

        Args:
            x: Original tactile image
            x_recon: Reconstructed tactile image

        Returns:
            Scalar reconstruction loss
        """
        # MSE averaged over all dimensions
        return F.mse_loss(x_recon, x, reduction="mean")


class TactileEmbedding(nn.Module):
    """Full tactile feature extraction module for SmolVLA.

    Extracts two complementary tactile features per fingertip:
    1. Resultant force vector f_t^{tac-f}: sum of forces over spatial grid -> R^{N x 3}
    2. Spatial tactile embedding f_t^{tac-s}: CAE latent -> R^{N x 128}

    These are then projected via MLPs to embeddings z_t^{tac-f} and z_t^{tac-s}
    that can be fused with other modalities in the VLA.

    Args:
        num_fingertips: Number of fingertips (default 2 for gripper)
        hidden_size: Output embedding dimension (should match VLM hidden size)
        latent_dim: CAE latent dimension (default 128)
        force_grid_shape: Shape of raw force grid (H, W) before padding (default (10, 12))
        pretrained_cae_path: Optional path to pretrained CAE weights
    """

    def __init__(
        self,
        num_fingertips: int = 2,
        hidden_size: int = 576,  # SmolVLM hidden size
        latent_dim: int = 128,
        force_grid_shape: tuple[int, int] = (10, 12),
        pretrained_cae_path: str | None = None,
    ):
        super().__init__()
        self.num_fingertips = num_fingertips
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.force_grid_shape = force_grid_shape

        # CAE for spatial tactile encoding
        self.cae = TactileCAE(in_channels=3, latent_dim=latent_dim, img_size=16)

        if pretrained_cae_path is not None:
            self.cae.load_state_dict(torch.load(pretrained_cae_path, map_location="cpu"))

        # MLP for resultant force projection (f_t^{tac-f} -> z_t^{tac-f})
        # Per-fingertip: 3 -> hidden_size
        self.force_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_size),
        )

        # MLP for spatial tactile projection (f_t^{tac-s} -> z_t^{tac-s})
        # Per-fingertip: latent_dim -> hidden_size
        self.spatial_mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, hidden_size),
        )

    def pad_force_grid_to_image(self, force_grid: Tensor) -> Tensor:
        """Pad raw force grid to 16x16 tactile image.

        Args:
            force_grid: Raw force grid of shape (B, N, H, W, 3) where H=10, W=12

        Returns:
            Padded tactile image of shape (B, N, 3, 16, 16)
        """
        B, N, H, W, C = force_grid.shape

        # Transpose to (B, N, C, H, W)
        force_grid = force_grid.permute(0, 1, 4, 2, 3)

        # Compute padding to reach 16x16
        pad_h = 16 - H  # = 6
        pad_w = 16 - W  # = 4

        # Pad (left, right, top, bottom) - zero padding
        # We pad symmetrically: top=3, bottom=3, left=2, right=2
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Reshape for F.pad: (B*N, C, H, W)
        force_grid = force_grid.reshape(B * N, C, H, W)
        padded = F.pad(force_grid, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        padded = padded.reshape(B, N, C, 16, 16).contiguous()

        return padded

    def compute_resultant_force(self, force_grid: Tensor) -> Tensor:
        """Compute resultant force by summing over spatial grid.

        Args:
            force_grid: Force grid of shape (B, N, H, W, 3) or (B, N, 3, H, W)

        Returns:
            Resultant force of shape (B, N, 3)
        """
        # Ensure shape is (B, N, H, W, 3)
        if force_grid.shape[-1] != 3 and force_grid.shape[2] == 3:
            # Shape is (B, N, 3, H, W), transpose
            force_grid = force_grid.permute(0, 1, 3, 4, 2)

        # Sum over spatial dimensions (H, W)
        resultant = force_grid.sum(dim=(2, 3))  # (B, N, 3)
        return resultant

    def forward(
        self,
        force_grid: Tensor | None = None,
        tactile_image: Tensor | None = None,
        resultant_force: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Extract tactile embeddings.

        Args:
            force_grid: Raw force grid (B, N, 10, 12, 3). If provided, will be padded to 16x16.
            tactile_image: Pre-padded tactile image (B, N, 3, 16, 16). Alternative to force_grid.
            resultant_force: Pre-computed resultant force (B, N, 3). If None, computed from grid.

        Returns:
            Tuple of:
            - z_tac_f: Force embedding (B, N, hidden_size)
            - z_tac_s: Spatial embedding (B, N, hidden_size)
            - cae_recon: CAE reconstruction (B, N, 3, 16, 16) or None if not training
        """
        # Get tactile image
        if tactile_image is not None:
            tac_img = tactile_image
        elif force_grid is not None:
            tac_img = self.pad_force_grid_to_image(force_grid)
        else:
            raise ValueError("Either force_grid or tactile_image must be provided")

        B, N = tac_img.shape[:2]

        # Compute resultant force if not provided
        if resultant_force is None:
            if force_grid is not None:
                resultant_force = self.compute_resultant_force(force_grid)
            else:
                # Compute from padded image (sum over spatial dims)
                # tac_img is (B, N, 3, 16, 16)
                resultant_force = tac_img.sum(dim=(3, 4))  # (B, N, 3)

        # Get CAE latent
        latent, recon = self.cae(tac_img)  # latent: (B, N, latent_dim)

        # Project to embeddings
        # Process per-fingertip
        z_tac_f = self.force_mlp(resultant_force)  # (B, N, hidden_size)
        z_tac_s = self.spatial_mlp(latent)  # (B, N, hidden_size)

        return z_tac_f, z_tac_s, recon

    def get_num_tokens(self) -> int:
        """Return number of tactile tokens added to prefix.

        Returns 2 * num_fingertips (one for force, one for spatial per fingertip).
        """
        return 2 * self.num_fingertips


class ArmHandFeatureEnhancement(nn.Module):
    """Arm-Hand Feature Enhancement module from paper Sec. 3.4.1.

    Takes shared representation z_share and produces:
    - z_arm: arm-specific features via 2-layer MLP with Mish
    - z_hand: hand-specific features via 2-layer MLP with Mish

    Also includes auxiliary prediction heads H_arm and H_hand for the
    auxiliary losses (Eq. 11-12).

    Args:
        shared_dim: Dimension of shared representation (expert_hidden_size)
        action_dim: Dimension of action space (max_action_dim)
        arm_indices: Indices of arm action dimensions (default 0..5)
        hand_indices: Indices of hand action dimensions (default [6])
    """

    def __init__(
        self,
        shared_dim: int,
        action_dim: int,
        arm_indices: list[int] | None = None,
        hand_indices: list[int] | None = None,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.action_dim = action_dim
        self.arm_indices = arm_indices if arm_indices is not None else list(range(6))
        self.hand_indices = hand_indices if hand_indices is not None else [6]

        # Feature dimension is half of shared dim
        self.feature_dim = shared_dim // 2

        # Arm encoder E_arm: 2-layer MLP with Mish
        self.E_arm = nn.Sequential(
            nn.Linear(shared_dim, self.feature_dim),
            nn.Mish(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Mish(inplace=True),
        )

        # Hand encoder E_hand: 2-layer MLP with Mish
        self.E_hand = nn.Sequential(
            nn.Linear(shared_dim, self.feature_dim),
            nn.Mish(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Mish(inplace=True),
        )

        # Auxiliary prediction heads (single linear layer each)
        self.H_arm = nn.Linear(self.feature_dim, action_dim)
        self.H_hand = nn.Linear(self.feature_dim, action_dim)

        # Main head takes concatenation of [z_share, z_arm, z_hand]
        # Input dim = shared_dim + 2 * feature_dim = shared_dim + shared_dim = 2 * shared_dim
        self.H_main = nn.Linear(shared_dim + 2 * self.feature_dim, action_dim)

    def forward(self, z_share: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            z_share: Shared representation (B, T, shared_dim) where T is chunk_size

        Returns:
            Tuple of:
            - z_arm: Arm features (B, T, feature_dim)
            - z_hand: Hand features (B, T, feature_dim)
            - v_main: Main action prediction (B, T, action_dim)
            - v_arm: Arm auxiliary prediction (B, T, action_dim)
            - v_hand: Hand auxiliary prediction (B, T, action_dim)
        """
        z_arm = self.E_arm(z_share)  # (B, T, feature_dim)
        z_hand = self.E_hand(z_share)  # (B, T, feature_dim)

        # Fused representation for main head
        z_fused = torch.cat([z_share, z_arm, z_hand], dim=-1)  # (B, T, shared_dim + 2*feature_dim)

        # Predictions
        v_main = self.H_main(z_fused)  # (B, T, action_dim)
        v_arm = self.H_arm(z_arm)  # (B, T, action_dim)
        v_hand = self.H_hand(z_hand)  # (B, T, action_dim)

        return z_arm, z_hand, v_main, v_arm, v_hand

    def compute_auxiliary_losses(
        self,
        u_t: Tensor,
        v_arm: Tensor,
        v_hand: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute auxiliary losses with selective supervision.

        Args:
            u_t: Target vector field (B, T, action_dim)
            v_arm: Arm prediction (B, T, action_dim)
            v_hand: Hand prediction (B, T, action_dim)

        Returns:
            Tuple of (L_arm, L_hand) losses
        """
        # Create masks for selective supervision
        arm_mask = torch.zeros(self.action_dim, device=u_t.device)
        arm_mask[self.arm_indices] = 1.0

        hand_mask = torch.zeros(self.action_dim, device=u_t.device)
        hand_mask[self.hand_indices] = 1.0

        # Compute masked losses
        # Only supervise on relevant action dimensions
        arm_diff = (v_arm - u_t) ** 2 * arm_mask.unsqueeze(0).unsqueeze(0)
        hand_diff = (v_hand - u_t) ** 2 * hand_mask.unsqueeze(0).unsqueeze(0)

        # Average over all dimensions (masked dims contribute 0)
        L_arm = arm_diff.mean()
        L_hand = hand_diff.mean()

        return L_arm, L_hand
