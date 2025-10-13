#!/usr/bin/env python3
"""
Print raw contact forces from Isaac Lab while running a task (default: Peg Insert).
No modifications to Isaac Lab sources required.

Examples:
  ./isaaclab.sh -p scripts/tools/print_contacts.py \
      --task Isaac-Factory-PegInsert-Direct-v0 --num_envs 1

Useful flags:
  --peg-only           # print only contacts on bodies that belong to the 'peg' actor
  --only-nonzero       # suppress near-zero forces
  --thresh 1e-3        # magnitude threshold for "nonzero"
  --steps 1000         # number of sim steps
  --headless           # run without GUI
"""

import argparse
import math
import sys

import numpy as np

# 1) Bring up the app/sim ------------------------------------------------------
from isaaclab.app import AppLauncher

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--peg-only", action="store_true", help="Print contacts only for bodies under the peg actor")
    p.add_argument("--only-nonzero", action="store_true", help="Print only contacts above |F| > thresh")
    p.add_argument("--thresh", type=float, default=1e-3, help="Threshold for nonzero filtering (in sim force units)")
    p.add_argument("--random-actions", action="store_true", help="Apply random actions each step (default: zeros)")
    return p.parse_args()

args = parse_args()
launcher = AppLauncher(headless=args.headless)
launcher.initialize()

# 2) Build env -----------------------------------------------------------------
from omni.isaac.lab.envs import make

render_mode = "human" if not args.headless else "headless"
env = make(task_name=args.task, num_envs=args.num_envs, render_mode=render_mode)

obs, _ = env.reset()

# Try to find a "peg" actor in the scene so we can filter to it if requested
peg_body_ids = None
peg_actor_name = None
for name, entity in env.scene.items():
    try:
        if "peg" in name.lower():
            peg_actor_name = name
            # Works for Articulation; if it's a rigid object, adapt accordingly
            if hasattr(entity, "articulation") and hasattr(entity.articulation, "body_ids"):
                peg_body_ids = np.array(entity.articulation.body_ids, dtype=np.int32)
            elif hasattr(entity, "rigid_body_ids"):
                peg_body_ids = np.array(entity.rigid_body_ids, dtype=np.int32)
            break
    except Exception:
        pass

if args.peg-only and peg_body_ids is None:
    print("[WARN] --peg-only was set, but no actor with 'peg' in its name was found. Printing all bodies instead.")

# 3) Helpers -------------------------------------------------------------------
def to_numpy(t):
    """Safely convert torch/warp/tensor-like -> numpy."""
    try:
        import torch
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(t)
    except Exception:
        return None

def get_contact_forces(sim):
    """
    Try a few API names across Isaac Lab versions. Returns numpy array of shape:
        [num_envs, num_bodies, 3]  (or compatible)
    """
    candidates = ["get_contact_force_tensor", "get_rigid_contact_force_tensor", "get_contact_forces"]
    for attr in candidates:
        if hasattr(sim, attr):
            data = getattr(sim, attr)()
            arr = to_numpy(data)
            if arr is not None:
                return arr
    return None

printed_api_warn = False

# 4) Run loop ------------------------------------------------------------------
for step in range(args.steps):
    # Choose an action
    if args.random_actions:
        actions = env.action_space.sample()
    else:
        # zeros shaped like action space
        try:
            import torch
            if hasattr(env.action_space, "shape"):
                actions = torch.zeros((env.num_envs,) + env.action_space.shape, device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                actions = env.action_space.sample()  # fallback
        except Exception:
            actions = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(actions)

    cf = get_contact_forces(env.sim)
    if cf is None:
        if not printed_api_warn:
            printed_api_warn = True
            print("[ERROR] Could not obtain contact forces from sim. "
                  "API not found on env.sim. Tried: get_contact_force_tensor / get_rigid_contact_force_tensor / get_contact_forces")
        continue

    # Expect shape [num_envs, num_bodies, 3]; try to coerce if needed
    if cf.ndim == 2 and cf.shape[-1] == 3:
        # maybe missing env dimension: add it
        cf = cf[None, ...]
    if cf.ndim != 3 or cf.shape[-1] != 3:
        print(f"[WARN] Unexpected contact force shape {cf.shape}; expecting [E, B, 3]. Printing raw array.")
        print(cf)
        continue

    # Optional: restrict to peg bodies
    if args.peg-only and peg_body_ids is not None:
        try:
            cf = cf[:, peg_body_ids, :]
        except Exception:
            print("[WARN] Couldn't slice peg body ids; printing all bodies instead.")

    # Compute magnitudes for filtering / summary
    mags = np.linalg.norm(cf, axis=-1)  # [E, B]
    if args.only-nonzero:
        # boolean mask of entries above threshold
        mask = mags > args.thresh
    else:
        mask = np.ones_like(mags, dtype=bool)

    # Print a concise summary per step
    total_contacts = int(mask.sum())
    max_mag = float(mags[mask].max()) if total_contacts > 0 else 0.0
    label = f" (peg: {peg_actor_name})" if args.peg-only and peg_actor_name else ""
    print(f"[step {step}] contacts{label}: {total_contacts} above {args.thresh:.1e}; max |F| = {max_mag:.6f}")

    # (Optional) Uncomment to print the actual vectors for the contacts above threshold
    # rows = np.argwhere(mask)
    # for (e, b) in rows:
    #     print(f"  env {e}, body {b}: F = {cf[e, b]}  |F|={mags[e, b]}")

    # Render one frame if not headless
    if render_mode == "human":
        env.render()

print("Done.")
