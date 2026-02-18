Record data:

```sh
conda activate env_isaaclab

./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_tacex_sm.py \
  --num_envs 4 \
  --num_demos 100 \
  --enable_cameras \
  --save_demos \
  --output_dir ./datasets/pick_place_basket_joint_tacex_100 \
  --background_mode fixed \
  --background_texture small_empty_house_4k.hdr \
  --headless
```

Optional background randomization:

```sh
--background_mode random
```

Notes:
- HDF5 data is saved to `<output_dir>/data.hdf5`.
- Collector executes scripted rollouts with an IK teacher, but writes joint-policy labels:
  - `state = [arm_joint_pos(7), gripper_qpos(2)]`
  - `action = [arm_joint_pos_target_abs(7), gripper_cmd(1)]`
- Collector defaults to `rendering_mode=balanced` (more stable temporal rendering at 224x224).
- Collector forces `render_interval == decimation` to reduce temporal flicker artifacts.
- Basket asset defaults to a material-stable bowl to avoid unresolved texture references during rendering.
- Only successful episodes are saved to HDF5.
- Successful episode videos are written to `<output_dir>/successful_videos/`.
- Optional failed episode videos are written to `<output_dir>/unsuccessful_videos/` by adding
  `--save_failed_videos`.
- Per-episode metadata JSON is written to `<output_dir>/metadata/`.
- Per-step phase labels are saved in HDF5 as `phase_id` (with `phase_name_map` attr).
- Planner now uses multi-waypoint motion with slower contact phases:
  - `pre_approach -> align_above_cube -> descend -> grasp_hold -> lift_clear -> carry_mid_1 -> carry_mid_2 -> pre_place_hover -> descend_place`.
- Recorded joint actions are taken from the IK controller's actual arm joint targets (`joint_pos_target`) for
  the same timestep, so `obs_t` is paired with the controller command used at `t` (not post-step realized joints).
- Planner metadata includes timing/gating/diversity/smoothing parameters in episode attrs and JSON.
- Expected successful demo length is typically higher than before (roughly 140-220 steps).
- Approach wobble reduction: lateral jitter (0.004 m), grasp entry offset (0.003 m), orientation slack
  (yaw ±5°, roll/pitch ±2°), and timing/profile randomization are tightened compared to earlier versions.
  Transport phases keep their original diversity parameters.
- Distance-aware fine motion: when EE is within 0.12 m of the cube during approach/descend/grasp phases,
  speed and position-step cap are linearly reduced; at 0.04 m or closer the step cap drops to 0.005 m
  (from 0.012 m) and speed to 0.45 (from baseline). This produces visibly smaller, smoother increments
  near contact without slowing the far approach.
- Orientation blending starts at 72% progress (up from 45%) so the wrist rotates toward the grasp
  orientation only in the final phase of approach, avoiding early left-right rotation artifacts.

Convert data to LeRobot:

```sh
conda activate smolvla

cd /home/radu/IsaacLab-Tactile/lerobot

python convert_pick_place_basket_joint_tacex.py \
  --input /home/radu/IsaacLab-Tactile/datasets/pick_place_basket_joint_tacex_100/data.hdf5 \
  --output-dir /home/radu/IsaacLab-Tactile/lerobot/datasets \
  --repo-id pick_place_basket_joint_tacex_100_lerobot \
  --task "Pick and place the cube into the basket"
```

Train SmolVLA:

```sh
conda activate smolvla

lerobot-train \
  --dataset.repo_id=pick_place_basket_joint_tacex_200_lerobot_v2 \
  --dataset.root=/home/radu/IsaacLab-Tactile/lerobot/datasets/pick_place_basket_joint_tacex_200_lerobot_v2 \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.push_to_hub=false \
  --policy.use_tactile=false \
  --policy.empty_cameras=1 \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/smolvla_joint_pick_place_basket_200_v2 \
  --wandb.enable=true \
  --wandb.project=smolvla-tactile \
  --job_name joint_200_no_tactile_v2_run1
```

Evaluation:

```sh
conda activate env_isaaclab

./isaaclab.sh -p scripts/eval_server.py \
  --env Isaac-Pick-Place-Basket-Franka-Joint-TacEx-v0 \
  --port 5555 \
  --enable_cameras \
  --headless
```

```sh
conda activate smolvla

lerobot-eval \
  --policy.path=/home/radu/IsaacLab-Tactile/lerobot/outputs/smolvla_joint_pick_place_basket_200_v2/checkpoints/last/pretrained_model \
  --env.type=isaaclab_tactile_remote_joint \
  --env.server_host=localhost \
  --env.server_port=5555 \
  --env.task="Pick and place the cube into the basket" \
  --env.observation_height=224 \
  --env.observation_width=224 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --rename_map='{"observation.images.rgb_table": "observation.images.camera1", "observation.images.rgb_wrist": "observation.images.camera2"}'
```

Visualize lerobot dataset:
```sh
lerobot-dataset-viz \
  --repo-id pick_place_basket_joint_tacex_200_lerobot \
  --root /home/radu/IsaacLab-Tactile/lerobot/datasets/pick_place_basket_joint_tacex_200_lerobot \
  --mode local \
  --episode-index 0 \
  --display-compressed-images True \
  --save 1 \
  --output-dir /home/radu/IsaacLab-Tactile/lerobot/outputs/viz_rrd
```

then on a local machine run:
```sh
rerun /home/radu/IsaacLab-Tactile/lerobot/outputs/viz_rrd/pick_place_basket_joint_tacex_200_lerobot_episode_0.rrd
```
