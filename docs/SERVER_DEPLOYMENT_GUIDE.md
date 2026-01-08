# Bench2Drive Server Deployment Guide

A comprehensive guide for deploying Bench2Drive on **headless Linux servers** (no display). Based on real-world deployment experience - covers all common pitfalls.

**Estimated setup time:**
- Expert user: ~15 minutes
- Normal user: ~30-45 minutes
- First-time user: ~60 minutes

---

## Table of Contents

1. [Quick Start (TL;DR)](#1-quick-start-tldr)
2. [Requirements](#2-requirements)
3. [CARLA Installation](#3-carla-installation)
4. [Critical: Running CARLA as Non-Root User](#4-critical-running-carla-as-non-root-user)
5. [Python Environment Setup](#5-python-environment-setup)
6. [Verification & Testing](#6-verification--testing)
7. [Running Evaluations](#7-running-evaluations)
8. [Troubleshooting](#8-troubleshooting)
9. [Quick Reference](#9-quick-reference)

---

## 1. Quick Start (TL;DR)

For experienced users who just need the commands:

```bash
# 1. Download and extract CARLA
cd /workspace && mkdir carla && cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz && rm CARLA_0.9.15.tar.gz

# 2. Install system dependencies
apt-get update && apt-get install -y vulkan-tools libxext6 libx11-6 xdg-user-dirs

# 3. Create non-root user (CARLA refuses to run as root!)
useradd -m -s /bin/bash carla
chown -R carla:carla /workspace/carla

# 4. Create startup script
cat > /workspace/carla/run_carla.sh << 'EOF'
#!/bin/bash
export HOME=/home/carla
export XDG_RUNTIME_DIR=/tmp/runtime-carla
mkdir -p $XDG_RUNTIME_DIR
cd /workspace/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 -nosound
EOF
chmod +x /workspace/carla/run_carla.sh
chown carla:carla /workspace/carla/run_carla.sh

# 5. Start CARLA
su carla -s /bin/bash -c 'nohup /workspace/carla/run_carla.sh > /home/carla/carla.log 2>&1 &'

# 6. Setup Python environment
conda create -n b2d python=3.7 -y
conda activate b2d
pip install /workspace/carla/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
pip install torch torchvision numpy opencv-python pillow scipy py-trees==0.8.3 networkx==2.2 psutil shapely xmlschema ephem tabulate

# 7. Verify connection
python -c "import carla; c=carla.Client('localhost',2000); print(c.get_server_version())"
```

---

## 2. Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 6GB VRAM | 8GB+ VRAM |
| CPU | 4 cores | 8+ cores |
| RAM | 16GB | 32GB+ |
| Disk | 50GB | 100GB+ (with datasets) |

### GPU Memory Estimates

| Configuration | Expected VRAM Usage |
|---------------|---------------------|
| CARLA (Low quality headless) | 3-4 GB |
| CARLA + ADMLP | 4-5 GB |
| CARLA + TCP | 5-7 GB |
| CARLA + UniAD/VAD | 12-16 GB |

### Software Requirements

- Ubuntu 18.04 / 20.04 / 22.04
- NVIDIA Driver >= 470 (470.x most stable, 535.x works, **avoid 550.x**)
- Vulkan support (required for CARLA 0.9.12+)
- Python 3.7 or 3.8
- Conda (recommended)

### Cloud Platform Notes

| Platform | Root User | Special Considerations |
|----------|-----------|----------------------|
| Vast.ai | Yes (default) | Must create non-root user |
| RunPod | Yes (default) | Must create non-root user |
| Lambda Labs | No | Works directly |
| AWS EC2 | No (ubuntu) | Works directly |
| Google Cloud | No | Works directly |

---

## 3. CARLA Installation

### 3.1 Install System Dependencies

```bash
# Update package list
apt-get update

# Install Vulkan (required for CARLA 0.9.12+)
apt-get install -y vulkan-tools libvulkan1

# Install X11 libraries (required even for headless mode!)
apt-get install -y libxext6 libx11-6 libxrender1 libgl1-mesa-glx

# Install XDG utilities
apt-get install -y xdg-user-dirs

# Network tools (for debugging)
apt-get install -y lsof net-tools
```

### 3.2 Verify Vulkan Installation

```bash
# Test Vulkan - should show your NVIDIA GPU
vulkaninfo --summary 2>&1 | head -50
```

**Expected output:**
```
==========
VULKANINFO
==========

Vulkan Instance Version: 1.3.xxx

...

Devices:
========
GPU0:
    deviceName         = NVIDIA GeForce RTX xxxx
    driverName         = NVIDIA
    driverInfo         = 535.xxx.xx
```

**If you see errors:**
```bash
# Error: "libXext.so.6: cannot open shared object file"
apt-get install -y libxext6

# Error: "lavapipe" warning (software rendering)
# This means Vulkan isn't using your GPU - usually a driver issue
```

### 3.3 Download CARLA 0.9.15

```bash
# Set install location
export CARLA_ROOT=/workspace/carla
mkdir -p $CARLA_ROOT && cd $CARLA_ROOT

# Download CARLA (~8GB download, ~20GB extracted)
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz

# Extract
tar -xzf CARLA_0.9.15.tar.gz

# Remove archive to save space
rm CARLA_0.9.15.tar.gz

# Verify installation
ls -la $CARLA_ROOT/CarlaUE4.sh
ls -la $CARLA_ROOT/PythonAPI/carla/dist/
```

### 3.4 (Optional) Additional Maps

Only needed for full 220-route benchmark (requires Town12):

```bash
cd $CARLA_ROOT/Import
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd $CARLA_ROOT
bash ImportAssets.sh
```

**Note:** Additional maps require ~7GB extra space. Skip for initial testing.

---

## 4. Critical: Running CARLA as Non-Root User

### The Problem

**CARLA refuses to run as root user.** This is a security feature built into Unreal Engine. You'll see this output and CARLA will exit immediately:

```
4.26.2-0+++UE4+Release-4.26 522 0
Disabling core dumps.
(process exits)
```

### The Solution

Create a non-root user and run CARLA under that user:

```bash
# Create 'carla' user
useradd -m -s /bin/bash carla

# Give carla user ownership of CARLA installation
chown -R carla:carla $CARLA_ROOT

# Add carla to video group (GPU access)
usermod -aG video,render carla 2>/dev/null || true
```

### Create CARLA Startup Script

```bash
cat > $CARLA_ROOT/run_carla.sh << 'EOF'
#!/bin/bash
# CARLA Headless Startup Script
# Must be run as non-root user!

export HOME=/home/carla
export XDG_RUNTIME_DIR=/tmp/runtime-carla
mkdir -p $XDG_RUNTIME_DIR

cd /workspace/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 -nosound
EOF

chmod +x $CARLA_ROOT/run_carla.sh
chown carla:carla $CARLA_ROOT/run_carla.sh
```

### Startup Script Parameters Explained

| Parameter | Purpose |
|-----------|---------|
| `-RenderOffScreen` | Headless mode (no display required) |
| `-quality-level=Low` | Reduce VRAM usage (~3-4GB vs ~6-8GB) |
| `-carla-port=2000` | TCP port for Python client |
| `-nosound` | Disable audio (avoids ALSA errors) |

### Start CARLA

```bash
# Start CARLA as 'carla' user in background
su carla -s /bin/bash -c 'nohup /workspace/carla/run_carla.sh > /home/carla/carla.log 2>&1 &'

# Wait for startup (30-60 seconds first time)
sleep 40

# Verify it's running
ps aux | grep CarlaUE4 | grep -v grep
```

**Expected output:**
```
carla    12345  50.0  2.0 10323092 3821412 ?    Sl   14:10   0:30 .../CarlaUE4-Linux-Shipping ...
```

### Stop CARLA

```bash
# Kill CARLA process
pkill -u carla CarlaUE4

# Or kill all CARLA processes
pkill -9 CarlaUE4
```

---

## 5. Python Environment Setup

### 5.1 Create Conda Environment

```bash
# Create Python 3.7 environment (required for CARLA 0.9.15)
conda create -n b2d python=3.7 -y
conda activate b2d
```

### 5.2 Install CARLA Python API

```bash
# Option 1: Install wheel (recommended)
pip install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl

# Option 2: Use egg file via .pth
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> \
    $CONDA_PREFIX/lib/python3.7/site-packages/carla.pth
```

### 5.3 Install Bench2Drive Dependencies

```bash
# Core dependencies
pip install torch torchvision  # Auto-selects CUDA version
pip install numpy opencv-python pillow scipy
pip install py-trees==0.8.3 networkx==2.2 six
pip install psutil shapely xmlschema ephem tabulate
```

### 5.4 Verify Installation

```bash
conda activate b2d

python << 'EOF'
import carla
import torch

print(f"CARLA Python API: OK")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test CARLA connection
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    print(f"CARLA Server: {client.get_server_version()}")
    print(f"Current Map: {client.get_world().get_map().name}")
    print("Connection: SUCCESS")
except Exception as e:
    print(f"Connection: FAILED - {e}")
EOF
```

**Expected output:**
```
CARLA Python API: OK
PyTorch: 1.13.1+cu117
CUDA available: True
CARLA Server: 0.9.15
Current Map: Carla/Maps/Town10HD_Opt
Connection: SUCCESS
```

---

## 6. Verification & Testing

### 6.1 Environment Checklist

Run these commands to verify your setup:

```bash
# 1. NVIDIA Driver
nvidia-smi --query-gpu=name,driver_version --format=csv
# Expected: Your GPU name and driver version

# 2. Vulkan
vulkaninfo --summary 2>&1 | grep "deviceName"
# Expected: NVIDIA GeForce RTX xxxx

# 3. CARLA Process
ps aux | grep CarlaUE4 | grep -v grep
# Expected: Running process

# 4. CARLA Port
lsof -i:2000
# Expected: CarlaUE4 process

# 5. Python Connection
python -c "import carla; c=carla.Client('localhost',2000); print(c.get_server_version())"
# Expected: 0.9.15
```

### 6.2 NpcAgent Test

Test the full evaluation pipeline with the built-in NPC agent (no ML model required):

```bash
cd /workspace/Bench2Drive
conda activate b2d

# Set environment
export CARLA_ROOT=/workspace/carla

# Run single route test
python leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=leaderboard/data/drivetransformer_bench2drive_dev10.xml \
    --routes-subset=0 \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint=results/npc_test.json \
    --agent=leaderboard/leaderboard/autoagents/npc_agent.py \
    --agent-config="" \
    --debug=0 \
    --record="" \
    --resume=False \
    --port=2000 \
    --timeout=600
```

---

## 7. Running Evaluations

### 7.1 Quick Evaluation Commands

```bash
# Start CARLA (if not running)
su carla -s /bin/bash -c '/workspace/carla/run_carla.sh &'
sleep 40

# Activate environment
conda activate b2d
cd /workspace/Bench2Drive

# Set common variables
export CARLA_ROOT=/workspace/carla
export PYTHONPATH=$PYTHONPATH:/workspace/Bench2DriveZoo
export IS_BENCH2DRIVE=1

# Run evaluation
python leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=leaderboard/data/drivetransformer_bench2drive_dev10.xml \
    --routes-subset=0 \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint=results/test.json \
    --agent=$TEAM_AGENT \
    --agent-config=$TEAM_CONFIG \
    --port=2000 \
    --timeout=600
```

### 7.2 Available Routes

| Route File | Routes | Purpose |
|------------|--------|---------|
| `bench2drive220.xml` | 220 | Full benchmark (requires additional maps) |
| `drivetransformer_bench2drive_dev10.xml` | 10 | Quick testing |

### 7.3 Merge Results

```bash
# Merge multiple route results
python tools/merge_route_json.py -f results/your_model/

# Compute driving scores
python tools/ability_benchmark.py -r results/merge.json
```

---

## 8. Troubleshooting

### 8.1 CARLA Won't Start

#### Symptom: "Refusing to run with root privileges"

**Cause:** Running CARLA as root user.

**Solution:** Create non-root user (see Section 4).

```bash
useradd -m -s /bin/bash carla
chown -R carla:carla /workspace/carla
su carla -s /bin/bash -c '/workspace/carla/run_carla.sh &'
```

#### Symptom: Shows version then exits immediately

```
4.26.2-0+++UE4+Release-4.26 522 0
Disabling core dumps.
(exits)
```

**Cause:** Missing XDG_RUNTIME_DIR or Vulkan issue.

**Solution:**
```bash
# Ensure XDG_RUNTIME_DIR is set in startup script
export XDG_RUNTIME_DIR=/tmp/runtime-carla
mkdir -p $XDG_RUNTIME_DIR

# Verify Vulkan works
vulkaninfo --summary
```

#### Symptom: "libXext.so.6: cannot open shared object file"

**Cause:** Missing X11 libraries.

**Solution:**
```bash
apt-get install -y libxext6 libx11-6 libxrender1
```

#### Symptom: "Cannot find a compatible Vulkan driver"

**Cause:** Vulkan not properly configured.

**Solution:**
```bash
apt-get install -y vulkan-tools libvulkan1

# If still failing, set ICD path explicitly
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### 8.2 Connection Issues

#### Symptom: Python can't connect to CARLA

```python
RuntimeError: time-out of 10000ms while waiting for the simulator
```

**Solutions:**

1. Check CARLA is running:
```bash
ps aux | grep CarlaUE4 | grep -v grep
```

2. Check port is open:
```bash
lsof -i:2000
```

3. Check firewall isn't blocking:
```bash
# Allow port 2000
iptables -I INPUT -p tcp --dport 2000 -j ACCEPT
```

4. Increase timeout:
```python
client = carla.Client('localhost', 2000)
client.set_timeout(30.0)  # 30 seconds instead of 10
```

### 8.3 GPU/Memory Issues

#### Symptom: "CUDA out of memory"

**Solutions:**
```bash
# Use low quality mode
./CarlaUE4.sh -RenderOffScreen -quality-level=Low ...

# Check GPU memory before starting
nvidia-smi

# Kill zombie CARLA processes
pkill -9 CarlaUE4
```

#### Symptom: Wrong GPU selected

```bash
# List available GPUs
nvidia-smi -L

# Select specific GPU for CARLA (not CUDA device number!)
./CarlaUE4.sh -RenderOffScreen -graphicsadapter=0
```

### 8.4 Cleanup Commands

```bash
# Kill all CARLA processes
pkill -9 CarlaUE4
pkill -9 CarlaUE4-Linux-Shipping

# Kill processes on specific port
kill $(lsof -t -i:2000) 2>/dev/null

# Clean up all CARLA remnants
ps aux | grep -i carla | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Verify cleanup
ps aux | grep CarlaUE4
lsof -i:2000
nvidia-smi  # Check VRAM is released
```

---

## 9. Quick Reference

### Essential Commands

```bash
# Start CARLA
su carla -s /bin/bash -c 'nohup /workspace/carla/run_carla.sh > /home/carla/carla.log 2>&1 &'

# Stop CARLA
pkill -u carla CarlaUE4

# Check CARLA status
ps aux | grep CarlaUE4 | grep -v grep

# View CARLA logs
tail -f /home/carla/carla.log

# Activate Python environment
conda activate b2d

# Test connection
python -c "import carla; c=carla.Client('localhost',2000); print(c.get_server_version())"
```

### Environment Variables

```bash
export CARLA_ROOT=/workspace/carla
export PYTHONPATH=$PYTHONPATH:/workspace/Bench2DriveZoo
export IS_BENCH2DRIVE=1
export SAVE_PATH=results/          # Save sensor data
export PLANNER_TYPE=only_traj      # For TCP model
```

### CARLA Command Line Options

| Option | Description |
|--------|-------------|
| `-RenderOffScreen` | Headless mode (required for servers) |
| `-quality-level=Low` | Low quality (saves ~3GB VRAM) |
| `-quality-level=Epic` | Full quality (default) |
| `-carla-port=PORT` | TCP port (default: 2000) |
| `-graphicsadapter=N` | GPU index (not CUDA device!) |
| `-nosound` | Disable audio |
| `-benchmark -fps=20` | Fixed framerate mode |

### Directory Structure

```
/workspace/
├── carla/                    # CARLA installation
│   ├── CarlaUE4.sh          # Main startup script
│   ├── run_carla.sh         # Custom headless startup
│   └── PythonAPI/carla/dist/ # Python wheels/eggs
├── Bench2Drive/              # Benchmark code
│   ├── leaderboard/         # Evaluation framework
│   ├── scenario_runner/     # Scenario execution
│   └── tools/               # Analysis scripts
├── Bench2DriveZoo/          # Model implementations
└── data/                    # Datasets
```

---

## Appendix A: Complete Setup Script

Save this as `setup_bench2drive.sh`:

```bash
#!/bin/bash
set -e

echo "=== Bench2Drive Setup Script ==="

# Configuration
CARLA_ROOT=/workspace/carla
B2D_ROOT=/workspace/Bench2Drive

# 1. Install system dependencies
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq vulkan-tools libvulkan1 libxext6 libx11-6 xdg-user-dirs lsof

# 2. Download CARLA
echo "[2/6] Downloading CARLA 0.9.15..."
mkdir -p $CARLA_ROOT && cd $CARLA_ROOT
if [ ! -f "CarlaUE4.sh" ]; then
    wget -q --show-progress https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    tar -xzf CARLA_0.9.15.tar.gz
    rm CARLA_0.9.15.tar.gz
fi

# 3. Create non-root user
echo "[3/6] Creating carla user..."
id carla &>/dev/null || useradd -m -s /bin/bash carla
chown -R carla:carla $CARLA_ROOT

# 4. Create startup script
echo "[4/6] Creating startup script..."
cat > $CARLA_ROOT/run_carla.sh << 'EOF'
#!/bin/bash
export HOME=/home/carla
export XDG_RUNTIME_DIR=/tmp/runtime-carla
mkdir -p $XDG_RUNTIME_DIR
cd /workspace/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 -nosound
EOF
chmod +x $CARLA_ROOT/run_carla.sh
chown carla:carla $CARLA_ROOT/run_carla.sh

# 5. Setup Python environment
echo "[5/6] Setting up Python environment..."
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n b2d python=3.7 -y || true
conda activate b2d
pip install -q $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
pip install -q torch torchvision numpy opencv-python pillow scipy
pip install -q py-trees==0.8.3 networkx==2.2 six psutil shapely xmlschema ephem tabulate

# 6. Start CARLA
echo "[6/6] Starting CARLA..."
su carla -s /bin/bash -c 'nohup /workspace/carla/run_carla.sh > /home/carla/carla.log 2>&1 &'
sleep 40

# Verify
echo ""
echo "=== Verification ==="
ps aux | grep CarlaUE4 | grep -v grep && echo "CARLA: Running" || echo "CARLA: NOT RUNNING"
conda activate b2d
python -c "import carla; c=carla.Client('localhost',2000); print(f'Connection: {c.get_server_version()}')" 2>/dev/null || echo "Connection: FAILED"

echo ""
echo "=== Setup Complete ==="
echo "Start CARLA:  su carla -s /bin/bash -c '/workspace/carla/run_carla.sh &'"
echo "Stop CARLA:   pkill -u carla CarlaUE4"
echo "Activate env: conda activate b2d"
```

---

## Appendix B: Version Information

| Component | Version | Notes |
|-----------|---------|-------|
| CARLA | 0.9.15 | Latest stable |
| Unreal Engine | 4.26 | Bundled with CARLA |
| Python | 3.7 / 3.8 | 3.7 recommended |
| NVIDIA Driver | 470.x / 535.x | Avoid 550.x |
| Ubuntu | 18.04 / 20.04 / 22.04 | All supported |

---

## Appendix C: References

### Official Documentation
- [CARLA Documentation](https://carla.readthedocs.io/en/0.9.15/)
- [CARLA Headless Mode](https://carla.readthedocs.io/en/latest/adv_rendering_options/)

### Related Issues
- [CARLA Root Privileges Issue #9049](https://github.com/carla-simulator/carla/issues/9049)
- [Bench2Drive Root Issue #37](https://github.com/Thinklab-SJTU/Bench2Drive/issues/37)
- [CARLA Headless Server #3943](https://github.com/carla-simulator/carla/issues/3943)

### Resources
- [Bench2Drive GitHub](https://github.com/Thinklab-SJTU/Bench2Drive)
- [Bench2DriveZoo Models](https://github.com/Thinklab-SJTU/Bench2DriveZoo)
- [Model Checkpoints](https://huggingface.co/rethinklab/Bench2DriveZoo)

---

**Last Updated:** January 2025
