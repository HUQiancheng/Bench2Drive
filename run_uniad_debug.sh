#!/bin/bash
# UniAD Closed-Loop Evaluation Script for Bench2Drive
# Based on official leaderboard/scripts/run_evaluation.sh

# Must set CARLA_ROOT
export CARLA_ROOT=/workspace/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export SCENARIO_RUNNER_ROOT=scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=True
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=traj

# Port configuration
export PORT=2000
export TM_PORT=2500
export GPU_RANK=0

# Route configuration - using dev10 with Town03 route (available in base CARLA)
export ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10.xml
export ROUTES_SUBSET=25378  # Town03 route, available without Additional Maps

# Agent configuration
export TEAM_AGENT=leaderboard/team_code/uniad_b2d_agent.py
export TEAM_CONFIG="Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+/workspace/Bench2DriveZoo/ckpts/uniad_base_b2d.pth"

# Output configuration
export CHECKPOINT_ENDPOINT=results/uniad_debug.json
export SAVE_PATH=results/uniad_debug/

cd /workspace/Bench2Drive
mkdir -p results

echo "=========================================="
echo "UniAD Closed-Loop Evaluation (Official)"
echo "=========================================="
echo "CARLA_ROOT: $CARLA_ROOT"
echo "ROUTES: $ROUTES"
echo "ROUTES_SUBSET: $ROUTES_SUBSET"
echo "TEAM_AGENT: $TEAM_AGENT"
echo "SAVE_PATH: $SAVE_PATH"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --routes=${ROUTES} \
    --routes-subset=${ROUTES_SUBSET} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --record="" \
    --resume=${RESUME} \
    --port=${PORT} \
    --traffic-manager-port=${TM_PORT} \
    --gpu-rank=${GPU_RANK} \
    --timeout=1200

echo "=========================================="
echo "Evaluation complete!"
echo "Results: ${CHECKPOINT_ENDPOINT}"
echo "=========================================="
