# SimpleVLA-RL Installation Guide

This guide provides step-by-step instructions for setting up the SimpleVLA-RL environment. Our project builds upon [veRL](https://github.com/volcengine/verl), and the installation process involves three main components: veRL, simulation benchmarks, and the VLA model (OpenVLA-OFT).

## Installation Options

### Option 1: Running RL on LIBERO Benchmark

#### Step 1: Install veRL

> **Note:** We recommend veRL version 0.2 or 0.3. Latest versions may have library conflicts.

Follow the official [veRL installation guide](https://verl.readthedocs.io/en/v0.3.x/start/install.html):

```bash
# Create and activate conda environment
conda create -n simplevla python==3.10
conda activate simplevla

# Install PyTorch
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Clone veRL (recommended to place at the same level as simplevla-rl, not inside the simplevla-rl folder)
git clone -b v0.2.x https://github.com/volcengine/verl.git
cd verl
pip3 install -e .
cd ..
```

#### Step 2: Install LIBERO and OpenVLA-OFT

Follow the official [OpenVLA-OFT installation guide](https://github.com/moojink/openvla-oft):

```bash
conda activate simplevla
pip3 install torch torchvision

# Clone OpenVLA-OFT (place at the same level as simplevla-rl, not inside the simplevla-rl folder)
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training
# If you encounter issues, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Should return exit code "0"
pip3 install flash-attn --no-build-isolation

cd ..

# Install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
cd openvla-oft
pip install -r experiments/robot/libero/libero_requirements.txt
```

### Option 2: Running RL on RoboTwin 2.0 Benchmark

#### Step 1: Install veRL

Same as Option 1, Step 1.

#### Step 2: Install RoboTwin 2.0

Follow the official [RoboTwin 2.0 Installation Guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html#1-dependencies):

```bash
# Install system dependencies
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

conda activate simplevla

# Clone and install RoboTwin
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
bash script/_install.sh

# Download RoboTwin assets
bash script/_download_assets.sh
cd ..
```

#### Step 3: Install OpenVLA-OFT

```bash
conda activate simplevla
pip3 install torch torchvision

# Clone OpenVLA-OFT (place at the same level as simplevla-rl, not inside the simplevla-rl folder)
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2
pip install packaging ninja
ninja --version; echo $?  # Should return exit code "0"
pip3 install flash-attn --no-build-isolation
cd ..
```

#### Step 4: Configure RoboTwin for SimpleVLA-RL

Apply necessary modifications to RoboTwin:

```bash
git clone https://github.com/PRIME-RL/SimpleVLA-RL.git
cd SimpleVLA-RL

# Apply RoboTwin modifications
bash copy_overwrite_robotwin2.sh <your_robotwin_path> <your_simplevlarl_path>
# Example: bash copy_overwrite_robotwin2.sh /mnt/petrelfs/SimpleVLA-RL /mnt/petrelfs/RoboTwin
```

## Troubleshooting

- If you encounter issues with RoboTwin 2.0 installation, refer to the [RoboTwin documentation](https://robotwin-platform.github.io/doc/) or check their GitHub issues
- All repositories (veRL, OpenVLA-OFT, RoboTwin) are recommended to clone at the same directory level as SimpleVLA-RL

## Directory Structure

After installation, your directory structure should look like:
```
your_workspace/
├── SimpleVLA-RL/
├── verl/
├── openvla-oft/
├── LIBERO/          (for Option 1)
└── RoboTwin/        (for Option 2)
```


## Supporting Additional Tasks in RoboTwin 2.0 (Optional) 

### Step 1: Collect Feasible Seeds

RoboTwin 2.0 tasks may have infeasible seeds (e.g., objects beyond arm reach). To optimize RL training, we pre-collect feasible seeds to avoid repeated validation during training epochs.

**Collection Process:**

1. Update `DATASET_NAME` in `pre_collect_robotwin2_seed.sh` with your target task name
2. Run the collection script:
   ```bash
   sh pre_collect_robotwin2_seed.sh
   ```
3. This generates `robotwin2_train_seeds.json` in the SimpleVLA-RL directory
4. Add the JSON content to:
   ```
   SimpleVLA-RL/verl/utils/envs/robotwin2/seeds/robotwin2_train_seeds.json
   ```

### Step 2: Register New Tasks

1. Add task name in `SimpleVLA-RL/verl/utils/dataset/rob_dataset.py`
2. Add task name and corresponding max steps in `SimpleVLA-RL/verl/workers/rollout/rob_rollout.py`

### Step 3: Implement Task-Specific Functions

Add a `get_info()` function in the corresponding task file under `SimpleVLA-RL/verl/utils/envs/robotwin2/envs/task_name.py`. 

For implementation reference, see:
```
SimpleVLA-RL/modified_codes/robotwin2/envs/handover_block.py
```
