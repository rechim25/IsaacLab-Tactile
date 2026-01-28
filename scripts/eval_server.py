#!/usr/bin/env python
"""
IsaacLab Tactile Evaluation Server.

This script runs an IsaacLab environment as a ZeroMQ server, allowing
remote evaluation from a separate conda environment (e.g., SmolVLA).

Usage (in isaaclab conda env):
    ./isaaclab.sh -p scripts/eval_server.py --port 5555 --headless

Protocol (ZeroMQ REP):
    - Receives: {"cmd": "reset"} or {"cmd": "step", "action": [...]}
    - Sends: {"obs": {...}, "reward": float, "terminated": bool, "info": {...}}

The server returns RAW observations in world frame. The client (LeRobot)
applies the canonical policy convention via env pre/post processors.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import struct
from typing import Any

import numpy as np

# ============================================================================
# Parse args BEFORE AppLauncher (required by IsaacLab)
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="IsaacLab Tactile Evaluation Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ server port")
    parser.add_argument("--host", type=str, default="*", help="Bind address (* for all interfaces)")
    parser.add_argument(
        "--env",
        type=str,
        default="Isaac-Pick-Place-Basket-Franka-IK-Rel-TacEx-v0",
        help="IsaacLab environment name",
    )
    parser.add_argument("--img_height", type=int, default=224, help="Camera image height")
    parser.add_argument("--img_width", type=int, default=224, help="Camera image width")
    
    # AppLauncher args are added here
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    
    args = parser.parse_args()
    return args


# Parse args and launch simulator
args_cli = parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# Imports that require simulator to be running
# ============================================================================

import gymnasium as gym
import torch
import cv2

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class IsaacLabEnvWrapper:
    """
    Wrapper for IsaacLab pick-place-basket environment with tactile sensing.
    
    Extracts raw observations that match the training data format:
    - eef_pos, eef_quat: End-effector pose (world frame, relative to env origin)
    - base_pos, base_quat: Robot base pose (for base-relative transforms)
    - gripper_qpos: Gripper joint positions
    - rgb_table, rgb_wrist: Camera images (uint8, HWC, 224x224x3)
    - tactile_force_grid: Pseudo-force grid (float32, 2x10x12x3)
    """
    
    def __init__(
        self, 
        env_name: str = "Isaac-Pick-Place-Basket-Franka-IK-Rel-TacEx-v0",
        device: str = "cuda:0",
        img_height: int = 224,
        img_width: int = 224,
    ):
        self.env_name = env_name
        self.device = device
        self.img_height = img_height
        self.img_width = img_width
        self.env = None
        self._step_count = 0
        self._max_episode_steps = 300
        
        logger.info(f"Initializing IsaacLab environment: {env_name}")
        self._setup_env()
    
    def _setup_env(self):
        """Set up the IsaacLab environment."""
        # Try TacEx environment first, fall back to non-tactile
        try:
            env_cfg = parse_env_cfg(self.env_name, device=self.device, num_envs=1)
            self.has_tacex = "TacEx" in self.env_name
        except Exception as e:
            logger.warning(f"Failed to load {self.env_name}: {e}")
            fallback_name = "Isaac-Pick-Place-Basket-Franka-IK-Rel-v0"
            logger.info(f"Trying fallback: {fallback_name}")
            env_cfg = parse_env_cfg(fallback_name, device=self.device, num_envs=1)
            self.env_name = fallback_name
            self.has_tacex = False
        
        # Disable timeout termination for evaluation
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        
        self.env = gym.make(self.env_name, cfg=env_cfg).unwrapped
        
        # Detect available sensors
        sensors = getattr(self.env.scene, "sensors", {})
        self.has_wrist_cam = "wrist_cam" in sensors
        self.has_table_cam = "table_cam" in sensors
        self.has_gsmini_left = "gsmini_left" in sensors
        self.has_gsmini_right = "gsmini_right" in sensors
        
        logger.info(f"Environment: {self.env_name}")
        logger.info(f"Sensors: wrist_cam={self.has_wrist_cam}, table_cam={self.has_table_cam}, "
                    f"gsmini_left={self.has_gsmini_left}, gsmini_right={self.has_gsmini_right}")
    
    def _extract_observation(self) -> dict:
        """Extract observation dict from current environment state."""
        robot = self.env.scene["robot"]
        ee_frame = self.env.scene["ee_frame"]
        
        # End-effector pose (relative to env origin)
        eef_pos = (ee_frame.data.target_pos_w[:, 0] - self.env.scene.env_origins)[0]
        eef_quat = ee_frame.data.target_quat_w[:, 0][0]
        
        # Robot base pose (relative to env origin)
        base_pos = (robot.data.root_pos_w - self.env.scene.env_origins)[0]
        base_quat = robot.data.root_quat_w[0]
        
        # Gripper joint positions (last 2 joints)
        gripper_qpos = robot.data.joint_pos[0, -2:]
        
        obs = {
            "eef_pos": eef_pos.cpu().numpy().astype(np.float32),
            "eef_quat": eef_quat.cpu().numpy().astype(np.float32),
            "base_pos": base_pos.cpu().numpy().astype(np.float32),
            "base_quat": base_quat.cpu().numpy().astype(np.float32),
            "gripper_qpos": gripper_qpos.cpu().numpy().astype(np.float32),
        }
        
        # Camera images
        if self.has_table_cam:
            rgb = self.env.scene.sensors["table_cam"].data.output.get("rgb")
            if rgb is not None and rgb.numel() > 0:
                obs["rgb_table"] = self._process_image(rgb[0])
        
        if self.has_wrist_cam:
            rgb = self.env.scene.sensors["wrist_cam"].data.output.get("rgb")
            if rgb is not None and rgb.numel() > 0:
                obs["rgb_wrist"] = self._process_image(rgb[0])
        
        # Tactile pseudo-force grid (2 fingertips x 10 x 12 x 3)
        # We compute this from height maps when available
        force_grid = np.zeros((2, 10, 12, 3), dtype=np.float32)
        
        if self.has_gsmini_left:
            hmap = self.env.scene.sensors["gsmini_left"].data.output.get("height_map")
            if hmap is not None and hmap.numel() > 0:
                force_grid[0] = self._compute_force_grid(hmap[0])
        
        if self.has_gsmini_right:
            hmap = self.env.scene.sensors["gsmini_right"].data.output.get("height_map")
            if hmap is not None and hmap.numel() > 0:
                force_grid[1] = self._compute_force_grid(hmap[0])
        
        obs["tactile_force_grid"] = force_grid
        
        # Add placeholder images if cameras not available
        if "rgb_table" not in obs:
            obs["rgb_table"] = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        if "rgb_wrist" not in obs:
            obs["rgb_wrist"] = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        return obs
    
    def _process_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Process camera image to uint8 HWC at target resolution."""
        img = img_tensor.detach().cpu().numpy()
        
        # Handle CHW -> HWC conversion if needed
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
        
        # Drop alpha if present
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        
        # Convert float -> uint8 safely with proper clipping
        if img.dtype in (np.float32, np.float64):
            # If looks normalized [0,1], scale up
            if np.nanmax(img) <= 1.5:
                img = img * 255.0
            img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        else:
            img = np.clip(img.astype(np.int32), 0, 255).astype(np.uint8)
        
        # Resize to target resolution
        if img.shape[0] != self.img_height or img.shape[1] != self.img_width:
            img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        
        return img
    
    def _compute_force_grid(self, height_map: torch.Tensor) -> np.ndarray:
        """
        Convert height map to pseudo-force grid (10x12x3).
        
        Creates a 3-channel force representation:
        - Channel 0 (Fx): Local x-gradient (shear)
        - Channel 1 (Fy): Local y-gradient (shear)
        - Channel 2 (Fz): Deformation depth (normal force)
        """
        hmap = height_map.cpu().numpy()
        H, W = hmap.shape
        
        # Compute baseline and deformation
        baseline = hmap.max()
        deformation = np.clip(baseline - hmap, 0, None)
        
        # Compute gradients for shear force
        grad_y, grad_x = np.gradient(deformation)
        
        # Normalize to [-1, 1] range
        grad_x = np.clip(grad_x / (np.abs(grad_x).max() + 1e-6), -1, 1)
        grad_y = np.clip(grad_y / (np.abs(grad_y).max() + 1e-6), -1, 1)
        
        # Normalize deformation to [0, 1]
        fz = deformation / (deformation.max() + 1e-6)
        
        # Resize to 10x12 grid
        force_grid = np.stack([
            cv2.resize(grad_x, (12, 10), interpolation=cv2.INTER_LINEAR),
            cv2.resize(grad_y, (12, 10), interpolation=cv2.INTER_LINEAR),
            cv2.resize(fz, (12, 10), interpolation=cv2.INTER_LINEAR),
        ], axis=-1)
        
        return force_grid.astype(np.float32)
    
    def reset(self, seed: int | None = None) -> dict:
        """Reset environment and return observation."""
        self._step_count = 0
        
        # Reset the environment
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()
        
        # Run one step to ensure sensors are updated
        action = np.zeros(7, dtype=np.float32)
        self.env.step(torch.from_numpy(action).unsqueeze(0).to(self.device))
        
        return self._extract_observation()
    
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Step the environment.
        
        Args:
            action: 7D action [delta_pos(3), delta_rot(3), gripper(1)]
            
        Returns:
            Tuple of (obs, reward, terminated, info)
        """
        self._step_count += 1
        
        # Convert action to tensor and step
        action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
        _, reward, terminated, truncated, info = self.env.step(action_tensor)
        
        # Extract scalar values
        reward_val = float(reward[0].item()) if torch.is_tensor(reward) else float(reward)
        done = bool(terminated[0].item()) if torch.is_tensor(terminated) else bool(terminated)
        done = done or (bool(truncated[0].item()) if torch.is_tensor(truncated) else bool(truncated))
        
        # Check for max steps
        if self._step_count >= self._max_episode_steps:
            done = True
        
        obs = self._extract_observation()
        
        # Check success (cube in basket)
        is_success = self._check_success()
        
        info_dict = {"step": self._step_count, "is_success": is_success}
        
        return obs, reward_val, done, info_dict
    
    def _check_success(self) -> bool:
        """
        Check if task is successful: cube is inside the basket.
        
        Success criteria:
        - Cube XY position is within basket XY bounds (with margin)
        - Cube Z is below basket rim but above basket bottom
        """
        try:
            # Get cube and basket from scene
            cube = self.env.scene.get("cube", None)
            basket = self.env.scene.get("basket", None)
            
            if cube is None or basket is None:
                # Try alternative names
                cube = self.env.scene.get("object", None)
                basket = self.env.scene.get("goal", None)
            
            if cube is None or basket is None:
                logger.debug("Could not find cube or basket in scene")
                return False
            
            # Get positions relative to env origin
            env_origin = self.env.scene.env_origins[0]
            
            cube_pos = cube.data.root_pos_w[0] - env_origin
            basket_pos = basket.data.root_pos_w[0] - env_origin
            
            cube_pos = cube_pos.cpu().numpy()
            basket_pos = basket_pos.cpu().numpy()
            
            # Check if cube is within basket bounds
            # Typical basket dimensions: ~0.15m radius, ~0.10m height
            basket_radius = 0.12  # Allow some margin
            basket_height = 0.15
            
            # XY distance from basket center
            xy_dist = np.sqrt((cube_pos[0] - basket_pos[0])**2 + 
                              (cube_pos[1] - basket_pos[1])**2)
            
            # Z: cube should be above basket bottom but below rim + margin
            z_in_basket = (cube_pos[2] > basket_pos[2] - 0.02 and 
                          cube_pos[2] < basket_pos[2] + basket_height + 0.05)
            
            is_success = (xy_dist < basket_radius) and z_in_basket
            
            return bool(is_success)
            
        except Exception as e:
            logger.debug(f"Success check failed: {e}")
            return False
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
        logger.info("Environment closed")


class EvalServer:
    """ZeroMQ server for IsaacLab evaluation."""
    
    def __init__(self, env: IsaacLabEnvWrapper, host: str = "*", port: int = 5555):
        self.env = env
        self.host = host
        self.port = port
        self.socket = None
        self.context = None
        self._running = False
    
    def start(self):
        """Start the server."""
        try:
            import zmq
        except ImportError:
            raise ImportError("ZeroMQ (pyzmq) required. Install with: pip install pyzmq")
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        endpoint = f"tcp://{self.host}:{self.port}"
        self.socket.bind(endpoint)
        logger.info(f"Server listening on {endpoint}")
        
        self._running = True
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._run_loop()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self._running = False
    
    def _run_loop(self):
        """Main server loop."""
        import zmq
        
        while self._running:
            try:
                # Wait for request with timeout to allow checking _running
                if self.socket.poll(timeout=1000):  # 1 second timeout
                    request_bytes = self.socket.recv()
                    request = json.loads(request_bytes.decode("utf-8"))
                    
                    response = self._handle_request(request)
                    
                    # Send response
                    response_bytes = self._serialize_response(response)
                    self.socket.send(response_bytes)
                    
            except zmq.error.ZMQError as e:
                if self._running:
                    logger.error(f"ZMQ error: {e}")
            except Exception as e:
                logger.error(f"Error handling request: {e}")
                # Send error response
                try:
                    error_response = {"error": str(e)}
                    self.socket.send(json.dumps(error_response).encode("utf-8"))
                except Exception:
                    pass
        
        self._cleanup()
    
    def _handle_request(self, request: dict) -> dict:
        """Handle incoming request."""
        cmd = request.get("cmd", "")
        
        if cmd == "reset":
            seed = request.get("seed")
            obs = self.env.reset(seed=seed)
            return {"obs": obs, "info": {}}
        
        elif cmd == "step":
            action = np.array(request["action"], dtype=np.float32)
            obs, reward, terminated, info = self.env.step(action)
            return {
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "info": info,
            }
        
        elif cmd == "close":
            self._running = False
            return {"status": "closing"}
        
        elif cmd == "render":
            # Return current frame if available
            return {"frame": None}
        
        else:
            return {"error": f"Unknown command: {cmd}"}
    
    def _serialize_response(self, response: dict) -> bytes:
        """
        Serialize response, handling numpy arrays efficiently.
        
        Uses binary format for large arrays (images, tactile).
        """
        obs = response.get("obs", {})
        
        # Identify large arrays to send as binary
        arrays_to_binary = {}
        json_obs = {}
        
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.nbytes > 1000:
                # Send large arrays as binary
                arrays_to_binary[key] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "nbytes": value.nbytes,
                }
            else:
                # Convert small arrays to lists for JSON
                if isinstance(value, np.ndarray):
                    json_obs[key] = value.tolist()
                else:
                    json_obs[key] = value
        
        if arrays_to_binary:
            # Use binary format
            response_copy = response.copy()
            response_copy["obs"] = json_obs
            response_copy["_arrays"] = arrays_to_binary
            
            json_bytes = json.dumps(response_copy).encode("utf-8")
            
            # Build binary data
            binary_parts = []
            for key in arrays_to_binary:
                arr = obs[key]
                binary_parts.append(arr.tobytes())
            binary_data = b"".join(binary_parts)
            
            # Format: "BINR" + json_len (4 bytes LE) + json + binary
            header = b"BINR" + struct.pack("<I", len(json_bytes))
            return header + json_bytes + binary_data
        else:
            # Plain JSON
            response_copy = response.copy()
            response_copy["obs"] = json_obs
            return json.dumps(response_copy).encode("utf-8")
    
    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.env.close()
        
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()
        
        logger.info("Server stopped")


def main():
    # Initialize environment
    env = IsaacLabEnvWrapper(
        env_name=args_cli.env,
        device=args_cli.device,
        img_height=args_cli.img_height,
        img_width=args_cli.img_width,
    )
    
    # Start server
    server = EvalServer(env, host=args_cli.host, port=args_cli.port)
    server.start()


if __name__ == "__main__":
    main()
    simulation_app.close()
