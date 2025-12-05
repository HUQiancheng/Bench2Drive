# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bench2Drive is a closed-loop autonomous driving benchmark built on CARLA 0.9.15 for evaluating end-to-end driving agents across 220 diverse routes. Published at NeurIPS 2024 Datasets and Benchmarks Track.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Leaderboard Evaluator                        │
│    (leaderboard_evaluator.py - orchestrates entire evaluation)      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         SCENARIO RUNNER                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ CarlaDataProvider│  │ ScenarioManager  │  │    Scenarios      │  │
│  │  (world access)  │  │ (execution loop) │  │ (40+ behaviors)   │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          CARLA Simulator                            │
│         (physics, rendering, traffic lights, weather)               │
└─────────────────────────────────────────────────────────────────────┘
```

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

# Resume crashed evaluation
# Set RESUME=True in script, then re-run same command
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

# Check port availability
lsof -i:30000
```

## Key Files Reference

### Leaderboard Module (`leaderboard/leaderboard/`)

| File | Lines | Purpose |
|------|-------|---------|
| `leaderboard_evaluator.py` | ~565 | Main orchestrator - launches CARLA, loads agents, runs routes |
| `autoagents/autonomous_agent.py` | ~161 | Base class all agents inherit from |
| `scenarios/route_scenario.py` | ~495 | Route + scenario container, spawns ego and scenarios |
| `scenarios/scenario_manager.py` | ~300 | Main tick loop (20 Hz), calls agent.run_step() |
| `utils/statistics_manager.py` | ~603 | Score calculation, penalty tracking, JSON output |
| `utils/route_parser.py` | - | XML → RouteConfiguration parsing |

### Scenario Runner Module (`scenario_runner/srunner/`)

| File | Lines | Purpose |
|------|-------|---------|
| `scenariomanager/carla_data_provider.py` | ~800 | **CRITICAL SINGLETON** - all CARLA world access |
| `scenariomanager/timer.py` | - | GameTime singleton for simulation time |
| `scenariomanager/watchdog.py` | - | Timeout detection for agent hangs |
| `scenariomanager/traffic_events.py` | - | TrafficEventType enum for infractions |
| `scenariomanager/scenarioatomics/atomic_behaviors.py` | ~2500 | 40+ behavior primitives |
| `scenariomanager/scenarioatomics/atomic_criteria.py` | ~1200 | 15+ evaluation criteria |
| `scenariomanager/scenarioatomics/atomic_trigger_conditions.py` | ~1350 | 25+ trigger conditions |
| `scenarios/basic_scenario.py` | ~300 | Base class for all 40+ scenarios |

## Critical Patterns

### CarlaDataProvider Singleton

**Every component uses this** for CARLA access. Located at `scenario_runner/srunner/scenariomanager/carla_data_provider.py`.

```python
# World/Map Access
CarlaDataProvider.get_world()           # Returns carla.World
CarlaDataProvider.get_map()             # Returns carla.Map (cached)
CarlaDataProvider.get_client()          # Returns carla.Client

# Actor Information (cached per-tick for performance)
CarlaDataProvider.get_location(actor)   # Returns carla.Location
CarlaDataProvider.get_velocity(actor)   # Returns float (m/s scalar)
CarlaDataProvider.get_transform(actor)  # Returns carla.Transform

# Actor Spawning (with collision retry logic)
CarlaDataProvider.request_new_actor(model, spawn_point, rolename, autopilot)

# Cleanup
CarlaDataProvider.cleanup()             # Destroy all tracked actors
```

### Agent Interface

Agents inherit from `AutonomousAgent` and implement:

```python
class MyAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Called once. Load model, set self.track = Track.SENSORS or Track.MAP"""
        pass

    def sensors(self):
        """Return list of sensor configs. Limits: 8 cameras, 2 LiDAR, 4 radar"""
        return [{'type': 'sensor.camera.rgb', 'id': 'CAM_FRONT', ...}]

    def run_step(self, input_data, timestamp):
        """Called every tick (20 Hz). Return carla.VehicleControl"""
        return carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)

    def destroy(self):
        """Cleanup when agent is destroyed"""
        pass
```

### Scenario Building Blocks (py_trees)

Scenarios compose atomic behaviors using behavior trees:

```python
# SEQUENCE: Children execute in order, stops on first FAILURE
sequence = py_trees.composites.Sequence("MySequence")
sequence.add_child(trigger)   # Wait for condition
sequence.add_child(action)    # Then perform action

# PARALLEL: Children execute simultaneously
parallel = py_trees.composites.Parallel("MyParallel",
    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
```

**Key Atomic Behaviors:**
- `KeepVelocity` - Maintain constant speed
- `WaypointFollower` - Follow waypoint route
- `LaneChange` - Change to adjacent lane
- `ActorTransformSetter` - Teleport actor
- `ActorDestroy` - Remove actor

**Key Atomic Criteria:**
- `CollisionTest` - Detects collisions (penalty: 0.50-0.65)
- `RunningRedLightTest` - Red light violations (penalty: 0.70)
- `RunningStopTest` - Stop sign violations (penalty: 0.80)
- `RouteCompletionTest` - Tracks route progress %
- `ActorBlockedTest` - Detects stuck vehicles (>60s)

**Key Trigger Conditions:**
- `InTriggerDistanceToVehicle` - Actor within distance of another
- `InTriggerDistanceToLocation` - Actor within distance of point
- `DriveDistance` - Actor has driven N meters
- `StandStill` - Actor velocity = 0 for duration

## Scoring System

### Driving Score Formula
```
Driving Score = Route Completion (%) × Penalty Factor
```

### Penalty Multipliers (multiplicative)

| Infraction | Penalty | Effect |
|------------|---------|--------|
| Collision with pedestrian | 0.50 | Score × 0.50 |
| Collision with vehicle | 0.60 | Score × 0.60 |
| Collision with static object | 0.65 | Score × 0.65 |
| Running red light | 0.70 | Score × 0.70 |
| Ignoring stop sign | 0.80 | Score × 0.80 |
| Scenario timeout | 0.70 | Score × 0.70 |

### Example
```
Route: 100% completion
Infractions: 1 vehicle collision + 1 red light
Penalty = 0.60 × 0.70 = 0.42
Driving Score = 100 × 0.42 = 42.0
```

## Evaluation Flow (Detailed)

```
1. run_evaluation.sh
   └─► LeaderboardEvaluator.__init__()
       ├─► _setup_simulation()
       │   ├─► find_free_port()
       │   ├─► subprocess.Popen(CarlaUE4.sh -graphicsadapter=N)
       │   ├─► time.sleep(30)  # Wait for CARLA startup
       │   └─► carla.Client() with retry logic
       └─► importlib.import_module(agent)

2. LeaderboardEvaluator.run()
   └─► FOR EACH route_config:
       ├─► _load_and_wait_for_world(town)
       ├─► RouteScenario(config)
       │   ├─► interpolate_trajectory()  # Sparse keypoints → dense route
       │   ├─► _spawn_ego_vehicle()
       │   └─► build_scenarios()  # Init scenarios within 500m
       ├─► Agent Setup
       │   ├─► agent.setup(config)
       │   ├─► agent.sensors()
       │   └─► validate_sensor_configuration()
       └─► ScenarioManager.run_scenario()
           └─► TICK LOOP (20 Hz, max 4000 ticks):
               ├─► world.tick()
               ├─► agent() → run_step(input_data) → VehicleControl
               ├─► ego.apply_control()
               ├─► behavior_tree.tick()  # Scenario logic
               └─► Check criteria  # Track infractions

3. Post-processing
   └─► compute_route_statistics() → JSON output
```

## CARLA-Specific Issues

- **GPU selection**: Use `-graphicsadapter=N` flag, **NOT** `CUDA_VISIBLE_DEVICES`
  - GPU mapping may skip indices (GPU1 might be `-graphicsadapter=2`)
- **Port conflicts**: Use ports >10000, check with `lsof -i:PORT`
  - Multi-GPU: increment by 150 (30000, 30150, 30300...)
- **Vulkan required**: Test with `/usr/bin/vulkaninfo | head -n 5`
- **Driver compatibility**: NVIDIA 470 is stable; 550 has bugs
- **Sleep times**: Increase sleep in evaluation scripts for slower machines
  - `leaderboard_evaluator.py:207` - 30s → 60s for slow startup
- **Synchronous mode**: Simulation runs at 20 Hz (`fixed_delta_seconds=0.05`)

## Cross-Module Dependencies

```
leaderboard/                          scenario_runner/
    │                                     │
    ├─► leaderboard_evaluator.py ────────►│
    │       import CarlaDataProvider      │ carla_data_provider.py
    │       import GameTime               │ timer.py
    │       import Watchdog               │ watchdog.py
    │                                     │
    ├─► route_scenario.py ───────────────►│
    │       extends BasicScenario         │ basic_scenario.py
    │       uses AtomicBehaviors          │ atomic_behaviors.py
    │       uses AtomicCriteria           │ atomic_criteria.py
    │       loads scenario classes        │ scenarios/*.py
    │                                     │
    └─► statistics_manager.py ◄──────────│
            uses TrafficEventType         │ traffic_events.py
```

## Debugging Tips

### Scenario Issues
```python
# Print behavior tree structure
py_trees.display.print_ascii_tree(scenario.scenario_tree, show_status=True)

# Draw debug point at actor location
world = CarlaDataProvider.get_world()
world.debug.draw_point(location, size=0.1, color=carla.Color(255,0,0))
```

### Common Problems

| Issue | Cause | Solution |
|-------|-------|----------|
| Scenario never triggers | Ego too far from trigger | Check trigger_point coordinates in XML |
| NPC doesn't move | Physics disabled | `actor.set_simulate_physics(True)` |
| Agent timeout | Model loading slow | Increase `--timeout` (default 600s) |
| `SensorReceivedNoData` | Sensor sync timeout | Increase `_queue_timeout` in sensor_interface.py |
| Missing routes in results | Crashed during eval | Use `--resume=True` |

## Route Definitions

**leaderboard/data/** - Route definitions
- `bench2drive220.xml`: Main 220 evaluation routes
- `drivetransformer_bench2drive_dev10.xml`: 10-route dev set for quick ablations

Route XML structure:
```xml
<route id="0" town="Town12">
    <waypoints>
        <position x="-30.0" y="20.5" z="0.0"/>
        <position x="50.0" y="100.0" z="0.0"/>
    </waypoints>
    <scenarios>
        <scenario name="VehicleTurningRoute" type="VehicleTurningRoute">
            <trigger_point x="-10.0" y="60.0" z="0.0" yaw="90.0"/>
        </scenario>
    </scenarios>
</route>
```

## Dataset Structure

Three splits collected by Think2Drive RL expert:
- Mini: 10 clips (4GB) - representative scenes
- Base: 1000 clips (400GB)
- Full: 13638 clips (4TB)

See `docs/anno.md` for annotation format details.

## Documentation

Comprehensive documentation available in `.llm/docs/`:
- `docs_bench2drive_leaderboard_docs.md` - Full leaderboard module documentation
- `docs_bench2drive_scenario_runner_comprehensive.md` - Full scenario_runner documentation
