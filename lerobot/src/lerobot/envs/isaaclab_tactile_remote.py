#!/usr/bin/env python
"""
Remote IsaacLab Tactile Environment Client.

This module provides a gym.Env wrapper that communicates with an IsaacLab server
running in a separate process/conda environment over ZeroMQ.

Architecture:
    [SmolVLA Client (smolvla env)]  <--ZeroMQ-->  [IsaacLab Server (isaaclab env)]
    
The server returns raw observations (world-frame poses, tactile, images) and
the client applies the canonical policy convention via env pre/post processors.

Protocol (ZeroMQ REQ/REP):
    - Client sends: {"cmd": "reset"} or {"cmd": "step", "action": [...]}
    - Server responds: {"obs": {...}, "reward": float, "done": bool, "info": {...}}
    
Observation dict from server should contain:
    - eef_pos: (3,) EE position in env-local/world frame
    - eef_quat: (4,) EE quaternion (x,y,z,w)
    - base_pos: (3,) Robot base position
    - base_quat: (4,) Robot base quaternion
    - gripper_qpos: (2,) Gripper joint positions
    - tactile_force_grid: (N, H, W, 3) Force grid
    - rgb_table: (H, W, 3) Table camera image
    - rgb_wrist: (H, W, 3) Wrist camera image
    (Additional keys may be present and will be passed through)
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class IsaacLabTactileRemoteEnv(gym.Env):
    """
    Gym environment wrapper for remote IsaacLab tactile server.
    
    Uses ZeroMQ REQ/REP pattern for request-response communication.
    The server runs IsaacLab physics and returns raw observations.
    
    Example usage:
        >>> env = IsaacLabTactileRemoteEnv(
        ...     server_host="localhost",
        ...     server_port=5555,
        ...     observation_space_config={...},
        ...     action_space_config={...},
        ... )
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 5555,
        timeout_ms: int = 30000,
        observation_height: int = 224,
        observation_width: int = 224,
        tactile_shape: tuple[int, ...] = (2, 10, 12, 3),
        max_episode_steps: int = 300,
        render_mode: str | None = "rgb_array",
    ):
        """
        Initialize remote environment client.
        
        Args:
            server_host: Hostname/IP of IsaacLab server
            server_port: Port of IsaacLab server
            timeout_ms: Timeout for ZeroMQ operations in milliseconds
            observation_height: Height of camera images
            observation_width: Width of camera images
            tactile_shape: Shape of tactile force grid (N_fingertips, H, W, 3)
            max_episode_steps: Maximum steps per episode
            render_mode: Render mode ("rgb_array" or None)
        """
        super().__init__()
        
        self.server_host = server_host
        self.server_port = server_port
        self.timeout_ms = timeout_ms
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.tactile_shape = tactile_shape
        # Gym/Gymnasium convention: many wrappers/tools look for `_max_episode_steps`.
        # LeRobot's eval rollout calls `env.call("_max_episode_steps")` on vector envs.
        self.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        self._step_count = 0
        self._context = None
        self._socket = None
        self._connected = False
        
        # Cache last RGB frame for rendering
        self._last_rgb: np.ndarray | None = None
        
        # Task description for language-conditioned policies
        # (empty string means no language instruction)
        self._task_description = "pick and place"
        
        # Define action space (7D: delta_pos, delta_rot, gripper)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # Define observation space as dict
        # Note: This is the raw observation space before processing
        self.observation_space = spaces.Dict({
            "eef_pos": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            "eef_quat": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            "base_pos": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            "base_quat": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            "gripper_qpos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "tactile_force_grid": spaces.Box(
                low=-100, high=100, shape=tactile_shape, dtype=np.float32
            ),
            "rgb_table": spaces.Box(
                low=0, high=255, 
                shape=(observation_height, observation_width, 3), 
                dtype=np.uint8
            ),
            "rgb_wrist": spaces.Box(
                low=0, high=255,
                shape=(observation_height, observation_width, 3),
                dtype=np.uint8
            ),
        })
    
    @property
    def task(self) -> str:
        """Return task name for language-conditioned policies."""
        return self._task_description
    
    @property
    def task_description(self) -> str:
        """Return task description for language-conditioned policies."""
        return self._task_description
    
    def _connect(self) -> None:
        """Establish ZeroMQ connection to server."""
        if self._connected:
            return
            
        try:
            import zmq
        except ImportError:
            raise ImportError(
                "ZeroMQ (pyzmq) is required for remote environment. "
                "Install with: pip install pyzmq"
            )
        
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        
        endpoint = f"tcp://{self.server_host}:{self.server_port}"
        logger.info(f"Connecting to IsaacLab server at {endpoint}")
        self._socket.connect(endpoint)
        self._connected = True
    
    def _disconnect(self) -> None:
        """Close ZeroMQ connection."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._connected = False
    
    def _send_request(self, request: dict) -> dict:
        """
        Send request to server and receive response.
        
        Args:
            request: Dictionary to send as JSON
            
        Returns:
            Response dictionary from server
        """
        import zmq
        
        self._connect()
        
        try:
            # Serialize request
            request_bytes = json.dumps(request).encode("utf-8")
            self._socket.send(request_bytes)
            
            # Receive response
            response_bytes = self._socket.recv()
            
            # Check for binary response (msgpack or custom format)
            if response_bytes[:4] == b"BINR":
                # Binary response format: "BINR" + json_len (4 bytes) + json + binary_data
                json_len = struct.unpack("<I", response_bytes[4:8])[0]
                json_part = response_bytes[8:8+json_len].decode("utf-8")
                response = json.loads(json_part)
                
                # Parse binary arrays
                binary_data = response_bytes[8+json_len:]
                response = self._parse_binary_arrays(response, binary_data)
            else:
                # Plain JSON response
                response = json.loads(response_bytes.decode("utf-8"))
            
            return response
            
        except zmq.error.Again:
            raise TimeoutError(
                f"Timeout waiting for response from IsaacLab server "
                f"(timeout={self.timeout_ms}ms)"
            )
    
    def _parse_binary_arrays(self, response: dict, binary_data: bytes) -> dict:
        """Parse binary array data from response."""
        obs = response.get("obs", {})
        offset = 0
        
        # Array format metadata should be in response["_arrays"]
        array_meta = response.get("_arrays", {})
        
        for key, meta in array_meta.items():
            dtype = np.dtype(meta["dtype"])
            shape = tuple(meta["shape"])
            nbytes = meta["nbytes"]
            
            arr = np.frombuffer(binary_data[offset:offset+nbytes], dtype=dtype)
            arr = arr.reshape(shape)
            obs[key] = arr
            offset += nbytes
        
        response["obs"] = obs
        return response
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed (forwarded to server)
            options: Additional options (forwarded to server)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        self._step_count = 0
        
        request = {"cmd": "reset"}
        if seed is not None:
            request["seed"] = seed
        if options is not None:
            request["options"] = options
        
        response = self._send_request(request)
        
        obs = self._process_observation(response.get("obs", {}))
        info = response.get("info", {})
        
        # Cache rgb_table for render()
        self._last_rgb = obs.get("rgb_table", None)
        
        return obs, info
    
    def step(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: 7D action array [delta_pos(3), delta_rot(3), gripper(1)]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        
        # Ensure action is numpy array and convert to list for JSON
        action = np.asarray(action, dtype=np.float32)
        
        request = {
            "cmd": "step",
            "action": action.tolist(),
        }
        
        response = self._send_request(request)
        
        obs = self._process_observation(response.get("obs", {}))
        reward = float(response.get("reward", 0.0))
        terminated = bool(response.get("terminated", False) or response.get("done", False))
        truncated = self._step_count >= self.max_episode_steps
        info = response.get("info", {})
        
        # Cache rgb_table for render()
        self._last_rgb = obs.get("rgb_table", None)
        
        return obs, reward, terminated, truncated, info
    
    def _process_observation(self, raw_obs: dict) -> dict:
        """
        Process raw observation from server.
        
        Ensures all arrays are numpy with correct dtype.
        """
        def _squeeze_leading_ones(arr: np.ndarray) -> np.ndarray:
            # Common IsaacLab pattern: (1, ...) for single-env/batched outputs.
            while arr.ndim > 0 and arr.shape[0] == 1:
                arr = arr[0]
            return arr

        def _pad_or_crop_hw(img_hwc: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
            """Center-crop or zero-pad an HWC image to (target_h, target_w, C) without resizing."""
            h, w = img_hwc.shape[:2]
            c = img_hwc.shape[2]
            out = np.zeros((target_h, target_w, c), dtype=img_hwc.dtype)

            # Compute src crop window
            src_y0 = max(0, (h - target_h) // 2)
            src_x0 = max(0, (w - target_w) // 2)
            src_y1 = min(h, src_y0 + target_h)
            src_x1 = min(w, src_x0 + target_w)

            # Compute dst paste window
            dst_y0 = max(0, (target_h - h) // 2)
            dst_x0 = max(0, (target_w - w) // 2)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            dst_x1 = dst_x0 + (src_x1 - src_x0)

            out[dst_y0:dst_y1, dst_x0:dst_x1] = img_hwc[src_y0:src_y1, src_x0:src_x1]
            return out

        def _coerce_vector(x: Any, expected_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
            arr = np.asarray(x, dtype=dtype)
            arr = _squeeze_leading_ones(arr)
            # If still has extra leading dims, flatten to expected
            if arr.shape != expected_shape:
                arr = arr.reshape(-1)
                if arr.size != int(np.prod(expected_shape)):
                    # fallback: zeros
                    return np.zeros(expected_shape, dtype=dtype)
                arr = arr.reshape(expected_shape)
            return arr

        def _coerce_tactile(x: Any) -> np.ndarray:
            expected = tuple(self.tactile_shape)
            arr = np.asarray(x, dtype=np.float32)
            arr = _squeeze_leading_ones(arr)
            # Accept either (N,H,W,3) or (B,N,H,W,3) with B=1 already squeezed above
            if arr.shape != expected:
                # If channels are first: (N,3,H,W) -> (N,H,W,3)
                if arr.ndim == 4 and arr.shape[1] == 3 and expected[-1] == 3:
                    arr = np.transpose(arr, (0, 2, 3, 1))
                # If still mismatch but size matches, reshape
                if arr.shape != expected and arr.size == int(np.prod(expected)):
                    arr = arr.reshape(expected)
            if arr.shape != expected:
                return np.zeros(expected, dtype=np.float32)
            return arr

        def _coerce_image(x: Any) -> np.ndarray:
            target_h, target_w = self.observation_height, self.observation_width
            img = np.asarray(x, dtype=np.uint8)
            img = _squeeze_leading_ones(img)

            # Accept CHW -> HWC
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                img = np.transpose(img, (1, 2, 0))

            if img.ndim != 3:
                return np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # Drop alpha if present
            if img.shape[2] == 4:
                img = img[:, :, :3]
            # If grayscale, expand to 3 channels
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            if img.shape[2] != 3:
                return np.zeros((target_h, target_w, 3), dtype=np.uint8)

            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = _pad_or_crop_hw(img, target_h, target_w)
            return img

        # Start from defaults so keys/shapes are always consistent with observation_space
        obs: dict[str, Any] = {}
        obs["eef_pos"] = np.zeros((3,), dtype=np.float32)
        obs["eef_quat"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs["base_pos"] = np.zeros((3,), dtype=np.float32)
        obs["base_quat"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs["gripper_qpos"] = np.zeros((2,), dtype=np.float32)
        obs["tactile_force_grid"] = np.zeros(tuple(self.tactile_shape), dtype=np.float32)
        obs["rgb_table"] = np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)
        obs["rgb_wrist"] = np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)

        # Overwrite with server-provided values (coerced to expected shapes)
        if "eef_pos" in raw_obs:
            obs["eef_pos"] = _coerce_vector(raw_obs["eef_pos"], (3,), np.float32)
        if "eef_quat" in raw_obs:
            obs["eef_quat"] = _coerce_vector(raw_obs["eef_quat"], (4,), np.float32)
        if "base_pos" in raw_obs:
            obs["base_pos"] = _coerce_vector(raw_obs["base_pos"], (3,), np.float32)
        if "base_quat" in raw_obs:
            obs["base_quat"] = _coerce_vector(raw_obs["base_quat"], (4,), np.float32)
        if "gripper_qpos" in raw_obs:
            obs["gripper_qpos"] = _coerce_vector(raw_obs["gripper_qpos"], (2,), np.float32)
        if "tactile_force_grid" in raw_obs:
            obs["tactile_force_grid"] = _coerce_tactile(raw_obs["tactile_force_grid"])
        if "rgb_table" in raw_obs:
            obs["rgb_table"] = _coerce_image(raw_obs["rgb_table"])
        if "rgb_wrist" in raw_obs:
            obs["rgb_wrist"] = _coerce_image(raw_obs["rgb_wrist"])

        return obs
    
    def render(self) -> np.ndarray | None:
        """Render the environment by returning the cached rgb_table frame."""
        if self.render_mode != "rgb_array":
            return None
        
        # Return cached rgb_table from last step/reset
        if isinstance(self._last_rgb, np.ndarray) and self._last_rgb.ndim == 3:
            return self._last_rgb
        
        # Fallback: return a black frame
        return np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)
    
    def close(self) -> None:
        """Close the environment and disconnect from server."""
        try:
            request = {"cmd": "close"}
            self._send_request(request)
        except Exception:
            pass  # Ignore errors on close
        
        self._disconnect()


def make_isaaclab_tactile_remote_env(
    server_host: str = "localhost",
    server_port: int = 5555,
    timeout_ms: int = 30000,
    **kwargs,
) -> IsaacLabTactileRemoteEnv:
    """
    Factory function to create IsaacLab tactile remote environment.
    
    Args:
        server_host: Hostname/IP of IsaacLab server
        server_port: Port of IsaacLab server
        timeout_ms: Timeout for operations
        **kwargs: Additional arguments passed to env constructor
        
    Returns:
        IsaacLabTactileRemoteEnv instance
    """
    return IsaacLabTactileRemoteEnv(
        server_host=server_host,
        server_port=server_port,
        timeout_ms=timeout_ms,
        **kwargs,
    )
