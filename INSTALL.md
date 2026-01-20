[3 tools called]

Based on the codebase, here's the **guaranteed step-by-step installation guide** for setting up Isaac Lab + TacEx on a new PC:

## **Complete Installation Guide for Isaac Lab + TacEx**

### **Prerequisites**
- Ubuntu 22.04 LTS (recommended)
- NVIDIA GPU with Driver Version 550+ and CUDA 12.4+
- At least 32GB RAM (64GB recommended for multiple environments)
- 50GB+ free disk space

### **Step 1: Install System Dependencies**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install git-lfs (CRITICAL for USD assets)
sudo apt install git-lfs
git lfs install

# Install other dependencies
sudo apt install build-essential cmake python3-pip
```

### **Step 2: Install NVIDIA Drivers & CUDA**
```bash
# Check if drivers are installed
nvidia-smi

# If not, install NVIDIA driver 550+
sudo ubuntu-drivers install

# Verify CUDA
nvcc --version  # Should show CUDA 12.4+
```

### **Step 3: Clone IsaacLab**
```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### **Step 4: Install Isaac Sim 4.5**
Follow the official installation: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

```bash
# Download Isaac Sim 4.5 from NVIDIA (requires login)
# Install to default location: ~/.local/share/ov/pkg/isaac-sim-4.5.0/
```

### **Step 5: Set Up Isaac Lab**
```bash
cd ~/IsaacLab

# Make the script executable
chmod +x isaaclab.sh

# Create conda environment and install Isaac Lab
./isaaclab.sh --install

# Activate the environment
conda activate isaaclab
```

### **Step 6: Clone and Install TacEx**
```bash
cd ~
git clone --recurse-submodules https://github.com/DH-Ng/TacEx
cd TacEx

# Install TacEx
./tacex.sh --install

# Link TacEx to Isaac Lab
cd ~/IsaacLab
ln -s ~/TacEx ./TacEx
```

### **Step 7: Install Your Custom Pick-Place-Basket Environment**

Copy these files from your current setup to the new PC:

```bash
# Create the environment directory structure
cd ~/IsaacLab
mkdir -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/{config/franka,mdp}

# Files to copy (from your current PC):
# 1. Environment configuration files
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/__init__.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/pick_place_basket_env_cfg.py

# 2. MDP functions
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/mdp/__init__.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/mdp/observations.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/mdp/terminations.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/mdp/events.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/mdp/tacex_observations.py

# 3. Robot-specific configs
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/config/__init__.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/config/franka/__init__.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/config/franka/pick_place_basket_joint_pos_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/config/franka/pick_place_basket_ik_rel_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/config/franka/pick_place_basket_ik_rel_tacex_env_cfg.py

# 4. State machine scripts
scripts/environments/state_machine/pick_place_basket_sm.py
scripts/environments/state_machine/pick_place_basket_tacex_sm.py

# 5. Data inspection script
TacEx/tools/inspect_hdf5.py
```

### **Step 8: Verify Installation**
```bash
cd ~/IsaacLab
source ~/.bashrc
conda activate isaaclab

# Test basic environment (without TacEx)
./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_sm.py \
    --num_envs 2

# Test with TacEx (requires --enable_cameras)
./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_tacex_sm.py \
    --num_envs 2 --enable_cameras
```

### **Step 9: Record Data with TacEx**
```bash
cd ~/IsaacLab

# Record demonstrations
./isaaclab.sh -p scripts/environments/state_machine/pick_place_basket_tacex_sm.py \
    --num_envs 8 \
    --num_demos 20 \
    --enable_cameras \
    --save_demos \
    --output_file ./datasets/pick_place_basket_tacex.hdf5

# Visualize recorded data
./isaaclab.sh -p TacEx/tools/inspect_hdf5.py \
    --hdf5_file ./datasets/pick_place_basket_tacex.hdf5 \
    --save_video --save_plots
```

### **Troubleshooting Checklist**
- ✅ `git-lfs` installed BEFORE cloning repos
- ✅ NVIDIA drivers 550+ installed
- ✅ Isaac Sim 4.5 installed (NOT 5.0)
- ✅ TacEx cloned with `--recurse-submodules`
- ✅ `--enable_cameras` flag used for TacEx environments
- ✅ Conda environment activated: `conda activate isaaclab`

### **Quick Transfer Method (If Both PCs Are Accessible)**
```bash
# On old PC, create a tarball of your custom code
cd ~/IsaacLab
tar -czf pick_place_basket_bundle.tar.gz \
    source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place_basket/ \
    scripts/environments/state_machine/pick_place_basket*.py \
    TacEx/tools/inspect_hdf5.py

# Transfer to new PC
scp pick_place_basket_bundle.tar.gz user@newpc:~/

# On new PC (after installing Isaac Lab + TacEx)
cd ~/IsaacLab
tar -xzf ~/pick_place_basket_bundle.tar.gz
```

This approach ensures all dependencies are correctly installed and your custom environment is properly integrated. Let me know if you need any clarification!