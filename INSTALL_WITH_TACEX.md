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

If you see: `A camera was spawned without the --enable_cameras flag.` → add
`--enable_cameras`.

---

## 7) Add / transfer the Pick-Place-Basket TacEx environment + state machine

If you developed `pick_place_basket` on another machine, copy these into this
repo:

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/`
  (entire folder)
- `scripts/environments/state_machine/pick_place_basket_sm.py`
- `scripts/environments/state_machine/pick_place_basket_tacex_sm.py`

After copying, verify the tasks register (example task IDs):

- `Isaac-Pick-Place-Basket-Franka-v0`
- `Isaac-Pick-Place-Basket-Franka-IK-Rel-v0`
- `Isaac-Pick-Place-Basket-Franka-IK-Rel-TacEx-v0`

---

## 8) Record TacEx demonstrations (HDF5)

### Record 1 demo to `datasets/test_vis.hdf5`

```bash
cd ~/IsaacLab
conda activate env_isaaclab
mkdir -p datasets

./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_tacex_sm.py \
  --num_envs 1 \
  --num_demos 1 \
  --enable_cameras \
  --save_demos \
  --output_file ./datasets/test_vis.hdf5
```

### Record many demos (parallel envs)

```bash
./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_tacex_sm.py \
  --num_envs 4 \
  --num_demos 100 \
  --enable_cameras \
  --save_demos \
  --output_file ./datasets/pick_place_basket_tacex.hdf5
```

---

## 9) Visualize the dataset with TacEx inspector

```bash
cd ~/IsaacLab
conda activate env_isaaclab

./isaaclab.sh -p TacEx/tools/inspect_hdf5.py \
  --hdf5_file ./datasets/test_vis.hdf5 \
  --save_video \
  --save_plots
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
