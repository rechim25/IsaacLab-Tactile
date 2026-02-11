# Option B (Long-term Fix): Make quaternion conventions consistent (and retrain)

## Why this is needed

IsaacLab typically stores quaternions as **(w, x, y, z)** (scalar-first).  
Your LeRobot adapter code in `lerobot/src/lerobot/isaaclab_tactile/policy_io.py` assumes quaternions are **(x, y, z, w)** (vector-first).

If the HDF5 demos were recorded in **wxyz** but later treated as **xyzw** during conversion, then:
- `observation.state` (Rot6D) will be computed from the wrong quaternion ordering,
- the learned policy will be trained on a different convention than evaluation uses,
- resulting in consistently “almost correct but always misses” behavior.

Option A (quick workaround) makes evaluation mimic the convention used during training.  
Option B fixes the root cause by correcting the dataset conversion and retraining.

---

## Step 0 — Confirm the HDF5 quaternion convention (quick check)

Inspect one stored quaternion in the raw HDF5 demo:
- If identity looks like `[1, 0, 0, 0]`, it’s **wxyz**.
- If identity looks like `[0, 0, 0, 1]`, it’s **xyzw**.

In your server logs you already saw IsaacLab identity near `[1, 0, 0, 0]` → likely **wxyz**.

---

## Step 1 — Fix quaternion ordering in dataset conversion

Edit `lerobot/convert_pick_place_basket_tacex.py` so the quaternions are converted to **xyzw** *before* calling the shared adapter:

```python
def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    # (w, x, y, z) -> (x, y, z, w)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)

# ...
ee_quat_raw = demo["ee_quat"][t].astype(np.float32)
ee_quat_w = wxyz_to_xyzw(ee_quat_raw)

if "base_quat" in demo:
    base_quat_raw = demo["base_quat"][t].astype(np.float32)
    base_quat_w = wxyz_to_xyzw(base_quat_raw)
else:
    base_quat_w = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
```

Do the same conversion for any other stored quaternions you pass into `encode_state_isaaclab_to_policy(...)` and `encode_action_isaaclab_to_policy(...)`.

---

## Step 2 — Regenerate the LeRobot dataset

Delete the old converted dataset directory and re-run conversion.

Make sure you regenerate the exact dataset you train on (e.g. `pick_place_basket_tacex_100_basepose_lerobot`).

---

## Step 3 — Retrain the baseline policy

Re-run `lerobot-train` using the regenerated dataset.

---

## Step 4 — Evaluate with the *correct* quaternion convention

After retraining on the fixed dataset, evaluation should use the **correct** convention end-to-end:

- Server sends IsaacLab quats as **wxyz**
- Client or adapter converts them to **xyzw** before policy IO / Rot6D encoding

At that point, you can restore the eval server’s quaternion conversion (or better: centralize it in one place, e.g. the env preprocessor) so both training conversion and eval preprocessing share the same convention.

---

## Recommended hardening

- Add a one-time assert/log in the conversion script:
  - detect if quats look like wxyz identity near `[1,0,0,0]` and automatically convert.
- Add a small unit/regression test:
  - verify that converting identity `wxyz=[1,0,0,0]` produces `xyzw=[0,0,0,1]` and Rot6D is sensible.

