# SmolVLA + Tactile + Arm/Hand Aux Losses (IsaacLab-Tactile)

This repo adds **tactile sensing** and the **Arm-Hand Feature Enhancement** module (with auxiliary losses) to the SmolVLA policy inside `lerobot/`.

## Policy I/O Conventions

### Canonical State (Policy Input): 11D

```
observation.state = [eef_pos_b(3), eef_rot6d_b(6), gripper_qpos(2)]
```

| Component       | Dims | Description                                           |
|-----------------|------|-------------------------------------------------------|
| `eef_pos_b`     | 3    | EE position in robot **base frame**                   |
| `eef_rot6d_b`   | 6    | EE orientation in base frame (Rot6D = first 2 cols of rotation matrix) |
| `gripper_qpos`  | 2    | Gripper joint positions                                |

### Canonical Action (Policy Output): 7D

```
action = [Δpos_b(3), Δaxis_angle_b(3), gripper(1)]
```

| Component        | Dims | Description                                          |
|------------------|------|------------------------------------------------------|
| `Δpos_b`         | 3    | EE translation delta in robot **base frame**         |
| `Δaxis_angle_b`  | 3    | EE rotation delta in base frame (axis-angle = axis × angle) |
| `gripper`        | 1    | Gripper command (-1 to 1)                             |

### Why Base Frame?

Using the robot base frame (not world frame) provides:
- **Frame invariance**: Policy transfers across environments with different world coordinates
- **Consistency**: Same convention as LIBERO but without world-frame dependency
- **Mobile robot compatibility**: Works for both fixed and mobile bases

### Shared Adapter Module

All state/action encoding is implemented in a single shared module:
- `lerobot/src/lerobot/isaaclab_tactile/policy_io.py`

This module is used by:
1. **Dataset conversion** (`convert_pick_place_basket_tacex.py`)
2. **Env preprocessor** (observation → policy input)
3. **Env postprocessor** (policy output → env action)

## Data Flow

### Data flow (recording → training)

- **IsaacLab HDF5 demos** (e.g. `datasets/pick_place_basket_tacex_100.hdf5`)
  - contain: `rgb_table`, `rgb_wrist`, `joint_pos`, `joint_vel`, `gripper_pos`, `ee_pos`, `ee_quat`, `base_pos`, `base_quat`, `actions`, and per-finger forces (`force_geometric_left/right`).
- **Conversion** creates a **LeRobot v3 dataset** (Parquet + meta) at:
  - `lerobot/datasets/<repo_id>/`
  - includes:
    - `observation.images.camera1` (from `rgb_table`)
    - `observation.images.camera2` (from `rgb_wrist`)
    - `observation.state` (**11-d** canonical format)
    - `action` (**7-d** canonical format)
    - `observation.tactile.force_grid` (synthetic grid: `(num_fingertips, 10, 12, 3)`)

### Model flow (SmolVLA)

- **Images** → embedded by the VLM image encoder (SigLIP / SmolVLM).
- **Language task** → tokenized and embedded.
- **State** → projected by `state_proj`.
- **Tactile force grid** → processed by `TactileEmbedding`:
  - resultant force branch: sum over grid → `(N, 3)` → MLP → token(s)
  - spatial branch: pad grid to `16x16`, CAE encoder → `(N, 128)` → MLP → token(s)
  - tokens are appended to the prefix (configurable tokenization mode).
- **Action expert** predicts flow-matching vector field for actions.
- **Arm-Hand Feature Enhancement** (optional) adds:
  - `E_arm`, `E_hand` MLPs from shared expert hidden state
  - `H_main` head and auxiliary `H_arm`, `H_hand` heads
  - total loss: `L_total = L_main + λ (L_arm + L_hand)` with selective supervision via indices:
    - arm indices default: `[0,1,2,3,4,5]`
    - hand indices default: `[6]`

## How Tactile is Normalized

SmolVLA uses LeRobot's `NormalizerProcessorStep` with `policy.normalization_mapping`.

Defaults in `SmolVLAConfig`:

- `VISUAL`: identity
- `STATE`: mean/std
- `ACTION`: mean/std
- `TACTILE`: **mean/std**

Important detail: tactile keys under `observation.tactile.*` are classified as `FeatureType.TACTILE` (so they use the `TACTILE` normalization rule).

## Conversion (HDF5 → LeRobot v3)

### 0) Activate environment

```bash
conda activate smolvla
```

### 1) Delete previous output (avoids "File exists")

```bash
rm -rf /home/radu/IsaacLab-Tactile/lerobot/datasets/pick_place_basket_tacex_100_lerobot
```

### 2) Convert

```bash
cd /home/radu/IsaacLab-Tactile/lerobot
python convert_pick_place_basket_tacex.py \
  --input /home/radu/IsaacLab-Tactile/datasets/pick_place_basket_tacex_100.hdf5 \
  --output-dir /home/radu/IsaacLab-Tactile/lerobot/datasets \
  --repo-id pick_place_basket_tacex_100_lerobot
```

The converter uses the shared `lerobot.isaaclab_tactile.policy_io` module to:
- Convert EE pose from world frame to base frame
- Encode orientation as Rot6D (11D state)
- Transform action deltas from world frame to base frame (7D action)

Optional flags:
- `--store-debug-fields`: Also store raw world-frame poses for debugging

After conversion, the dataset root should exist:

- `/home/radu/IsaacLab-Tactile/lerobot/datasets/pick_place_basket_tacex_100_lerobot/meta/info.json`
- `/home/radu/IsaacLab-Tactile/lerobot/datasets/pick_place_basket_tacex_100_lerobot/meta/stats.json`
- `/home/radu/IsaacLab-Tactile/lerobot/datasets/pick_place_basket_tacex_100_lerobot/data/...`

## Training (with WandB)

### 0) Activate environment

```bash
conda activate smolvla
```

### 1) (Once) login to WandB

```bash
wandb login
```

### 2) Run training

```bash
cd /home/radu/IsaacLab-Tactile/lerobot
lerobot-train \
  --dataset.repo_id=pick_place_basket_tacex_100_lerobot \
  --dataset.root=/home/radu/IsaacLab-Tactile/lerobot/datasets/pick_place_basket_tacex_100_lerobot \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.push_to_hub=false \
  --policy.use_tactile=true \
  --policy.num_fingertips=2 \
  --policy.use_arm_hand_feature_enhancement=true \
  --policy.arm_indices='[0,1,2,3,4,5]' \
  --policy.hand_indices='[6]' \
  --policy.aux_loss_lambda=1.0 \
  --policy.empty_cameras=1 \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/smolvla_tactile_armhand \
  --wandb.enable=true \
  --wandb.project=smolvla-tactile \
  --wandb.entity=YOUR_WANDB_ENTITY
```

Notes:

- If you run OOM, lower `--batch_size` (e.g. `4`).
- If you don't want to set entity, you can omit `--wandb.entity=...` and WandB will use your default.
- State dimension is now **11** (not 27), action dimension remains **7**.

## Evaluation (IsaacLab in separate environment)

For evaluation, IsaacLab runs as a server in its own conda environment (with GPU/physics), and the SmolVLA policy runs as a client in the `smolvla` environment.

### Server (IsaacLab side)

```bash
conda activate isaaclab
python scripts/eval_server.py --port 5555 --env pick_place_basket
```

### Client (SmolVLA side)

```bash
conda activate smolvla
lerobot-eval \
  --policy.path=outputs/smolvla_tactile_armhand \
  --env.type=isaaclab_tactile_remote \
  --env.server_host=localhost \
  --env.server_port=5555 \
  --env.task=pick_place \
  --eval.n_episodes=10
```

The client applies env preprocessor (`IsaacLabTactilePolicyObservationProcessorStep`) and postprocessor (`IsaacLabTactilePolicyActionProcessorStep`) using the same shared adapter.

## Where Outputs are Saved

### If you pass `--output_dir=...`

Outputs go exactly there, e.g.:

- `lerobot/outputs/smolvla_tactile_armhand/`

### If you do NOT pass `--output_dir`

Training defaults to:

- `lerobot/outputs/train/YYYY-MM-DD/HH-MM-SS_<job_name>/`

### What's inside

The run directory includes:

- `train_config.json`: exact config used for the run
- checkpoints and logs (and a local `wandb/` cache directory when WandB is enabled)

## Testing the Adapter Math

Unit tests for the shared adapter are in:
- `lerobot/tests/test_isaaclab_tactile_policy_io.py`

Run tests:
```bash
cd /home/radu/IsaacLab-Tactile/lerobot
pytest tests/test_isaaclab_tactile_policy_io.py -v
```

