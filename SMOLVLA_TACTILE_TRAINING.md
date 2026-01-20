# SmolVLA + Tactile + Arm/Hand Aux Losses (IsaacLab-Tactile)

This repo adds **tactile sensing** and the **Arm-Hand Feature Enhancement** module (with auxiliary losses) to the SmolVLA policy inside `lerobot/`.

## What gets trained (high-level)

### Data flow

- **IsaacLab HDF5 demos** (e.g. `datasets/pick_place_basket_tacex_100.hdf5`)
  - contain: `rgb_table`, `rgb_wrist`, `joint_pos`, `joint_vel`, `gripper_pos`, `ee_pos`, `ee_quat`, `actions`, and per-finger forces (`force_geometric_left/right`).
- **Conversion** creates a **LeRobot v3 dataset** (Parquet + meta) at:
  - `lerobot/datasets/<repo_id>/`
  - includes:
    - `observation.images.camera1` (from `rgb_table`)
    - `observation.images.camera2` (from `rgb_wrist`)
    - `observation.state` (27-d concatenated vector)
    - `action` (7-d)
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

## How tactile is normalized

SmolVLA uses LeRobot’s `NormalizerProcessorStep` with `policy.normalization_mapping`.

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

### 1) Delete previous output (avoids “File exists”)

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
- If you don’t want to set entity, you can omit `--wandb.entity=...` and WandB will use your default.

## Where outputs are saved

### If you pass `--output_dir=...`

Outputs go exactly there, e.g.:

- `lerobot/outputs/smolvla_tactile_armhand/`

### If you do NOT pass `--output_dir`

Training defaults to:

- `lerobot/outputs/train/YYYY-MM-DD/HH-MM-SS_<job_name>/`

### What’s inside

The run directory includes:

- `train_config.json`: exact config used for the run
- checkpoints and logs (and a local `wandb/` cache directory when WandB is enabled)

