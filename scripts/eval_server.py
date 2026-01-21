#!/usr/bin/env python
"""
IsaacLab Tactile Evaluation Server.

This script runs an IsaacLab environment as a ZeroMQ server, allowing
remote evaluation from a separate conda environment (e.g., SmolVLA).

Usage (in isaaclab conda env):
    python scripts/eval_server.py --port 5555 --env pick_place_basket

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
import sys
from typing import Any

import numpy as np

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
        default="pick_place_basket",
        help="Environment name to load",
    )
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device")
    return parser.parse_args()


class IsaacLabEnvWrapper:
    """
    Wrapper for IsaacLab environment that provides a simple interface.
    
    This is a placeholder that should be replaced with actual IsaacLab
    environment initialization for your specific setup.
    """
    
    def __init__(self, env_name: str, headless: bool = True, device: str = "cuda:0"):
        self.env_name = env_name
        self.headless = headless
        self.device = device
        self.env = None
        self.scene = None
        
        logger.info(f"Initializing IsaacLab environment: {env_name}")
        self._setup_env()
    
    def _setup_env(self):
        """
        Set up the IsaacLab environment.
        
        TODO: Replace this with your actual IsaacLab environment setup.
        This is a placeholder showing the expected interface.
        """
        # Example placeholder - replace with actual IsaacLab initialization
        # from omni.isaac.lab_tasks.manager_based import ... 
        logger.warning(
            "IsaacLabEnvWrapper._setup_env() is a placeholder. "
            "Replace with your actual IsaacLab environment initialization."
        )
        
        # Mock env for demonstration
        self._obs_dim = {
            "eef_pos": 3,
            "eef_quat": 4,
            "base_pos": 3,
            "base_quat": 4,
            "gripper_qpos": 2,
        }
        self._action_dim = 7
        self._step_count = 0
    
    def reset(self, seed: int | None = None) -> dict:
        """Reset environment and return observation."""
        self._step_count = 0
        
        # TODO: Replace with actual env.reset()
        # obs = self.env.reset()
        
        # Mock observation for demonstration
        obs = {
            "eef_pos": np.random.randn(3).astype(np.float32),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "base_pos": np.zeros(3, dtype=np.float32),
            "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "gripper_qpos": np.zeros(2, dtype=np.float32),
            "tactile_force_grid": np.zeros((2, 10, 12, 3), dtype=np.float32),
            "rgb_table": np.zeros((224, 224, 3), dtype=np.uint8),
            "rgb_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        }
        
        return obs
    
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Step the environment.
        
        Args:
            action: 7D action [delta_pos(3), delta_rot(3), gripper(1)]
            
        Returns:
            Tuple of (obs, reward, terminated, info)
        """
        self._step_count += 1
        
        # TODO: Replace with actual env.step()
        # obs, reward, done, info = self.env.step(action)
        
        # Mock response for demonstration
        obs = {
            "eef_pos": np.random.randn(3).astype(np.float32),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "base_pos": np.zeros(3, dtype=np.float32),
            "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "gripper_qpos": np.array([action[6], action[6]], dtype=np.float32),
            "tactile_force_grid": np.random.randn(2, 10, 12, 3).astype(np.float32) * 0.1,
            "rgb_table": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "rgb_wrist": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        }
        
        reward = 0.0
        terminated = self._step_count >= 300
        info = {"step": self._step_count}
        
        return obs, reward, terminated, info
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            # self.env.close()
            pass
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
    args = parse_args()
    
    # Initialize environment
    env = IsaacLabEnvWrapper(
        env_name=args.env,
        headless=args.headless,
        device=args.device,
    )
    
    # Start server
    server = EvalServer(env, host=args.host, port=args.port)
    server.start()


if __name__ == "__main__":
    main()
