cd /home/rechim/IsaacLab

# Run with TacEx tactile sensing

./isaaclab.sh -p scripts/environments/random_agent.py\
--task Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0\
--num_envs 4\
--enable_cameras

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py\
--task Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0\
--num_envs 1\
--enable_cameras\
--device spacemouse
