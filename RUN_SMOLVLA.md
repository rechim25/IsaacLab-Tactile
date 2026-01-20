cd /home/radu/IsaacLab-Tactile/lerobot
conda run -n smolvla python convert_pick_place_basket_tacex.py \
  --input /home/radu/IsaacLab-Tactile/datasets/pick_place_basket_tacex_100.hdf5 \
  --output-dir /home/radu/IsaacLab-Tactile/lerobot/datasets \
  --repo-id pick_place_basket_tacex_100_lerobot

cd /home/radu/IsaacLab-Tactile/lerobot
conda run -n smolvla lerobot-train \
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
  --output_dir=outputs/smolvla_tactile_armhand