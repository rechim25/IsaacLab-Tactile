cd /home/rechim/IsaacLab

# Run with TacEx tactile sensing

```sh
./isaaclab.sh -p scripts/environments/random_agent.py\
--task Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0\
--num_envs 4\
--enable_cameras

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py\
--task Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0\
--num_envs 1\
--enable_cameras\
--device spacemouse
```

# NEW

Generate first demos:

```sh
# First, update stack_cube_tacex_sm.py with the same fixes (switch to agent mode)
# Then run:
./isaaclab.sh -p scripts/environments/state_machine/stack_cube_tacex_sm.py \
    --num_envs 8 --num_demos 10 --enable_cameras \
    --save_demos --output_file ./datasets/tacex_source.hdf5
```

Use Mimic to generate more demos:

```sh
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
    --input_file ./datasets/tacex_source.hdf5 \
    --output_file ./datasets/tacex_augmented.hdf5 \
    --num_envs 4 --generation_num_trials 100
```

Or use SkillGen (motion planning):

```sh
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 \
    --input_file ./datasets/tacex_source.hdf5 \
    --output_file ./datasets/tacex_skillgen.hdf5 \
    --num_envs 4 --generation_num_trials 100 --use_skillgen
```
