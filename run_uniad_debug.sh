#!/bin/bash

# UniAD Evaluation Debug Script for Bench2Drive
# Runs a single route for quick testing

set -e

# Environment setup
export CARLA_ROOT=/workspace/carla
export IS_BENCH2DRIVE=True
export CUDA_HOME=/root/miniconda3/envs/b2d_zoo
export PATH=$CUDA_HOME/bin:$PATH

# PYTHONPATH - all required paths
export PYTHONPATH="/workspace/Bench2Drive"
export PYTHONPATH="${PYTHONPATH}:/workspace/Bench2Drive/scenario_runner"
export PYTHONPATH="${PYTHONPATH}:/workspace/Bench2Drive/leaderboard"
export PYTHONPATH="${PYTHONPATH}:/workspace/Bench2Drive/leaderboard/team_code"
export PYTHONPATH="${PYTHONPATH}:/workspace/Bench2DriveZoo"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla"

# Configuration
CARLA_PORT=2000
ROUTES=/workspace/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml
TEAM_AGENT=/workspace/Bench2Drive/leaderboard/team_code/uniad_b2d_agent.py
TEAM_CONFIG="Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+/workspace/Bench2DriveZoo/ckpts/uniad_base_b2d.pth"
CHECKPOINT_ENDPOINT=/workspace/Bench2Drive/results/uniad_debug.json
SAVE_PATH=/workspace/Bench2Drive/results/uniad_debug/

cd /workspace/Bench2Drive
mkdir -p results

echo "=========================================="
echo "UniAD Evaluation - Single Route Debug"
echo "=========================================="

# Check if CARLA is running
if pgrep -x "CarlaUE4-Linux" > /dev/null; then
    echo "[OK] CARLA is already running"
else
    echo "[!] CARLA not running. Starting CARLA..."
    su carla -s /bin/bash -c "nohup ${CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=${CARLA_PORT} -quality-level=Low > /home/carla/carla.log 2>&1 &"
    echo "Waiting 30s for CARLA to initialize..."
    sleep 30
fi

# Verify CARLA connection
echo "Verifying CARLA connection on port ${CARLA_PORT}..."
python -c "import carla; c=carla.Client('localhost', ${CARLA_PORT}); c.set_timeout(10); print('CARLA version:', c.get_server_version())"

if [ $? -ne 0 ]; then
    echo "[ERROR] Cannot connect to CARLA on port ${CARLA_PORT}"
    exit 1
fi

echo "=========================================="
echo "Starting evaluation..."
echo "Routes: $ROUTES"
echo "Agent: $TEAM_AGENT"
echo "=========================================="

# Run evaluation - use first route ID from dev10 file (3514)
python /workspace/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=${ROUTES} \
    --routes-subset=3514 \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=0 \
    --record="" \
    --resume=False \
    --port=${CARLA_PORT} \
    --timeout=1200

echo "=========================================="
echo "Evaluation complete!"
echo "Results: ${CHECKPOINT_ENDPOINT}"
echo "=========================================="
