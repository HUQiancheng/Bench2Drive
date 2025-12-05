# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bench2Drive is a closed-loop autonomous driving benchmark built on CARLA 0.9.15 for evaluating end-to-end driving agents across 220 diverse routes. Published at NeurIPS 2024 Datasets and Benchmarks Track.

## Essential Commands

### CARLA Setup
```bash
export CARLA_ROOT=/path/to/carla
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> $CONDA_PREFIX/lib/python3.7/site-packages/carla.pth
```

### Evaluation
```bash
# Debug single route (set GPU_RANK, TEAM_AGENT, TEAM_CONFIG first)
bash leaderboard/scripts/run_evaluation_debug.sh

# Multi-GPU parallel evaluation (set TASK_NUM, GPU_RANK_LIST, TASK_LIST, TEAM_AGENT, TEAM_CONFIG)
bash leaderboard/scripts/run_evaluation_multi_uniad.sh
```

### Metrics
```bash
# Merge 220 route results and compute driving score
python tools/merge_route_json.py -f your_json_folder/

# Multi-ability breakdown
python tools/ability_benchmark.py -r merge.json

# Efficiency and smoothness metrics
python tools/efficiency_smoothness_benchmark.py -f merge.json -m your_metric_folder/
```

### Utilities
```bash
# Kill stuck CARLA processes (run multiple times)
bash tools/clean_carla.sh

# Generate debug video
python tools/generate_video.py -f your_rgb_folder/
```

## Architecture

### Core Components

**leaderboard/** - Evaluation framework
- `leaderboard/leaderboard_evaluator.py`: Main orchestrator - launches CARLA, loads agents, runs routes, collects statistics
- `leaderboard/autoagents/autonomous_agent.py`: Base class all agents inherit from
- `leaderboard/utils/statistics_manager.py`: Tracks infractions, calculates scores
- `team_code/`: Where custom agents are placed

**scenario_runner/** - CARLA ScenarioRunner integration
- `srunner/scenariomanager/`: Scenario execution with py_trees behavior trees
- `srunner/scenarios/`: 40+ driving scenarios (PedestrianCrossing, VehicleTurning, LaneChange, etc.)
- `srunner/scenariomanager/scenarioatomics/`: Atomic behaviors and criteria (CollisionTest, RedLightTest, etc.)

**tools/** - Post-processing and analysis
- `merge_route_json.py`: Aggregates all route results
- `ability_benchmark.py`: Multi-ability score calculation
- `efficiency_smoothness_benchmark.py`: Driving comfort metrics
- `data_collect.py`: Rich sensor data collection during routes

**leaderboard/data/** - Route definitions
- `bench2drive220.xml`: Main 220 evaluation routes
- `drivetransformer_bench2drive_dev10.xml`: 10-route dev set for quick ablations

### Evaluation Flow

1. `run_evaluation.sh` sets environment and calls `leaderboard_evaluator.py`
2. LeaderboardEvaluator connects to CARLA, loads agent dynamically
3. For each route: load world → initialize scenario → run agent loop → track infractions → save JSON
4. Post-processing scripts aggregate results across all 220 routes

### Agent Interface

Agents inherit from `AutonomousAgent` and implement:
- `setup(path_to_conf_file)`: Initialize model, set track (SENSORS or MAP)
- `sensors()`: Return sensor configuration list
- `run_step(input_data, timestamp)`: Return `carla.VehicleControl`
- `destroy()`: Cleanup

Sensor limits for SENSORS track: 8 cameras max, 2 LiDAR, 4 radar.

## CARLA-Specific Issues

- **GPU selection**: Use `-graphicsadapter=N` flag, not `CUDA_VISIBLE_DEVICES`
- **Port conflicts**: Use ports >10000, check with `lsof -i:PORT`
- **Vulkan required**: Test with `/usr/bin/vulkaninfo | head -n 5`
- **Driver compatibility**: NVIDIA 470 is stable; 550 has bugs
- **Sleep times**: Increase sleep in evaluation scripts for slower machines
- **Back camera extrinsic**: Known bug in UniAD/VAD team_code (fixed in training, wrong in eval)

## Key Metrics

- **Driving Score**: Route completion with infraction penalties (collision 0.5-0.65, traffic violation 0.7-0.8)
- **Success Rate**: Percentage of routes completed without infractions
- **Multi-Ability**: Breakdown by scenario type (Overtaking, Merging, Emergency_Brake, Give_Way, Traffic_Signs)
- **Efficiency/Smoothness**: Jerk, acceleration, yaw metrics

## Dataset Structure

Three splits collected by Think2Drive RL expert:
- Mini: 10 clips (4GB) - representative scenes
- Base: 1000 clips (400GB)
- Full: 13638 clips (4TB)

See `docs/anno.md` for annotation format details.
