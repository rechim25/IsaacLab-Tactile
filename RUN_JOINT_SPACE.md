Record data:

```sh
conda activate env_isaaclab

./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_joint_tacex_sm.py   --num_envs 4 --num_demos 100 --enable_cameras --save_demos   --output_file ./datasets/pick_place_basket_joint_tacex_100.hdf5 --headless
```

Convert data to LeRobot:

```sh
conda activate smolvla

cd /home/radu/IsaacLab-Tactile/lerobot

python convert_pick_place_basket_joint_tacex.py \
  --input /home/radu/IsaacLab-Tactile/datasets/pick_place_basket_joint_tacex_100.hdf5 \
  --output-dir /home/radu/IsaacLab-Tactile/lerobot/datasets \
  --repo-id pick_place_basket_joint_tacex_100_lerobot \
  --task "Pick and place the cube into the basket"
```

Train SmolVLA:

```sh
conda activate smolvla

lerobot-train \
  --dataset.repo_id=pick_place_basket_joint_tacex_100_lerobot \
  --dataset.root=/home/radu/IsaacLab-Tactile/lerobot/datasets/pick_place_basket_joint_tacex_100_lerobot \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.push_to_hub=false \
  --policy.use_tactile=false \
  --policy.empty_cameras=1 \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/smolvla_joint_pick_place_basket \
  --wandb.enable=true \
  --wandb.project=smolvla-tactile
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
  --policy.path=/home/radu/IsaacLab-Tactile/lerobot/outputs/smolvla_joint_pick_place_basket/checkpoints/last/pretrained_model \
  --env.type=isaaclab_tactile_remote_joint \
  --env.server_host=localhost \
  --env.server_port=5555 \
  --env.observation_height=224 \
  --env.observation_width=224 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --rename_map='{"observation.images.rgb_table": "observation.images.camera1", "observation.images.rgb_wrist": "observation.images.camera2"}'
```