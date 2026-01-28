# Installation - Data Collection

Clone this repository first:
```sh
git clone --recurse-submodules https://github.com/rechim25/IsaacLab-Tactile.git
```

## Install Isaac Lab + TacEx (Isaac Sim 4.5) and run TacEx data collection

This guide is the **most reliable** setup for TacEx: **Isaac Sim 4.5 (Python
3.10)** + **Isaac Lab** + **TacEx** using **conda**.

> **Important compatibility**
>
> - **TacEx requires Isaac Sim 4.5** (Python **3.10**)
> - Isaac Sim 5.x uses Python **3.11** → TacEx install will fail with
>   wheel/platform mismatches.

---

## Prerequisites (Ubuntu)

- Ubuntu 22.04 recommended
- NVIDIA GPU + drivers working (`nvidia-smi` should work)
- `git-lfs` installed (needed for USD assets in some repos)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git git-lfs build-essential cmake unzip
git lfs install
```

---

## 1) Install Isaac Sim 4.5 (standalone zip)

Download **Isaac Sim 4.5.0 standalone** zip from NVIDIA/Omniverse.

### Extract correctly (the zip extracts “flat”)

The file `isaac-sim-standalone-4.5.0-linux-x86_64.zip` does **NOT** contain a
top-level folder. So you must create one and unzip into it.

```bash
mkdir -p ~/.local/share/ov/pkg/isaac-sim-4.5.0
cd ~/Downloads
unzip isaac-sim-standalone-4.5.0-linux-x86_64.zip -d ~/.local/share/ov/pkg/isaac-sim-4.5.0
```

### Verify version + first-run reset

```bash
cat ~/.local/share/ov/pkg/isaac-sim-4.5.0/VERSION
~/.local/share/ov/pkg/isaac-sim-4.5.0/isaac-sim.sh --reset-user
~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh -c "print('Isaac Sim OK')"
```

---

## 2) Install Miniconda (recommended)

If you don’t already have conda:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

---

## 3) Clone Isaac Lab and link Isaac Sim 4.5

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Create the standard Isaac Lab symlink to your Isaac Sim install
ln -s ~/.local/share/ov/pkg/isaac-sim-4.5.0 _isaac_sim
```

---

## 4) Create the conda env (Python 3.10) + install Isaac Lab

```bash
cd ~/IsaacLab

# Create env (default name: env_isaaclab)
./isaaclab.sh --conda

conda activate env_isaaclab
python --version  # MUST be 3.10.x for Isaac Sim 4.5

# Install/build Isaac Lab into the env
./isaaclab.sh --install
```

If you see Python 3.11 here, you’re not actually using Isaac Sim 4.5 / the setup
is mis-detected.

---

## 5) Install TacEx (WITH submodules)

```bash
cd ~/IsaacLab
git clone --recurse-submodules https://github.com/DH-Ng/TacEx

cd ~/IsaacLab/TacEx
./tacex.sh --install
```

---

## 6) Quick TacEx sanity test (must enable cameras)

```bash
cd ~/IsaacLab
conda activate env_isaaclab

./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-TacEx-v0 \
  --num_envs 1 \
  --enable_cameras
```

---

## Troubleshooting

- **CUDA error 999 / `cudaErrorUnknown`**
  - First try closing any Isaac Sim processes and re-run.
  - If the NVIDIA kernel modules are “in use” and can’t be reloaded, **reboot**
    is the most reliable fix.

- **Cameras error**
  - Always run TacEx scripts with **`--enable_cameras`**.

- **Version check**
  - Isaac Sim binary install version:
    ```bash
    cat ~/.local/share/ov/pkg/isaac-sim-4.5.0/VERSION
    ```
  - Python version (TacEx requires 3.10):
    ```bash
    conda activate env_isaaclab
    python --version
    ```

--- 

# Run - Data Collection

In this tutorial, we will run data collection with tactile sensing for a custom `pick_place_basket` task.

### Generate Demonstrations

To generate 100 demos (4 parallel envs) headless and save data to `./datasets/pick_place_basket_tacex.hdf5`:
```sh
cd /home/rechim/IsaacLab

./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_tacex_sm.py --num_envs 4 --num_demos 100 --headless --enable_cameras --rendering_mode quality --save_demos --output_file ./datasets/pick_place_basket_tacex.hdf5
```

### Inspect Data

Create plots:
```sh
python scripts/tools/inspect_hdf5.py ./datasets/pick_place_basket_tacex.hdf5 --samples 1 --plot --trajectory --forces
```

Create video:
```sh
python scripts/tools/inspect_hdf5.py ./datasets/pick_place_basket_tacex.hdf5 --video --demo-idx 0   --video-output demo_video.mp4 --fps 30
```

---

![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows,
such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html),
it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real
transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based
cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and
computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks.
Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.


## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:

- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.


## Getting Started

### Documentation

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including
detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Isaac Sim Version Dependency

Isaac Lab is built on top of Isaac Sim and requires specific versions of Isaac Sim that are compatible with each
release of Isaac Lab. Below, we outline the recent Isaac Lab releases and GitHub branches and their corresponding
dependency versions for Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version   |
| ----------------------------- | ------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0 |
| `v2.1.X`                      | Isaac Sim 4.5       |
| `v2.0.X`                      | Isaac Sim 4.5       |


## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Inspiration

We encourage you to utilize our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell)
area in the `Discussions` section of this repository. This space is designed for you to:

* Share the tutorials you've created
* Showcase your learning content
* Present exciting projects you've developed

By sharing your work, you'll inspire others and contribute to the collective knowledge
of our community. Your contributions can spark new ideas and collaborations, fostering
innovation in robotics and simulation.

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas,
  asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of
  work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features,
  or general updates.

## Connect with the NVIDIA Omniverse Community

Do you have a project or resource you'd like to share more widely? We'd love to hear from you!
Reach out to the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to explore opportunities
to spotlight your work.

You can also join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to
connect with other developers, share your projects, and help grow a vibrant, collaborative ecosystem
where creativity and technology intersect. Your contributions can make a meaningful impact on the Isaac Lab
community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its
corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its
dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

Note that Isaac Lab requires Isaac Sim, which includes components under proprietary licensing terms. Please see the [Isaac Sim license](docs/licenses/dependencies/isaacsim-license.txt) for information on Isaac Sim licensing.

Note that the `isaaclab_mimic` extension requires cuRobo, which has proprietary licensing terms that can be found in [`docs/licenses/dependencies/cuRobo-license.txt`](docs/licenses/dependencies/cuRobo-license.txt).

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if
you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
