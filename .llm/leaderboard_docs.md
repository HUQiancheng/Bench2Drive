# Bench2Drive Leaderboard Module - Comprehensive Documentation

This document provides a comprehensive guide to the `leaderboard/` module in Bench2Drive, covering its architecture, how to run evaluations, implementation details, and cross-module connections.

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Directory Structure](#2-directory-structure)
3. [How to Run Evaluations](#3-how-to-run-evaluations)
4. [Component Deep Dive](#4-component-deep-dive)
5. [Data Flow](#5-data-flow)
6. [Scoring System](#6-scoring-system)
7. [CARLA Integration](#7-carla-integration)
8. [Cross-Module Connections](#8-cross-module-connections)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Module Overview

### Purpose

The `leaderboard/` module is the **evaluation framework** for autonomous driving agents in Bench2Drive. It serves as the bridge between:
- **Agent implementations** (neural network models like UniAD, VAD, TCP)
- **CARLA simulator** (world, traffic, sensors)
- **Scenario definitions** (44 interactive scenarios across 220 routes)

### Position in Bench2Drive Ecosystem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Bench2Drive Ecosystem                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐   │
│   │  Your Agent  │────►│   LEADERBOARD    │────►│ CARLA Simulator  │   │
│   │  (team_code) │     │   (evaluation)   │     │   (0.9.15)       │   │
│   └──────────────┘     └──────────────────┘     └──────────────────┘   │
│          │                     │                        │               │
│          │                     ▼                        │               │
│          │             ┌──────────────┐                 │               │
│          │             │scenario_runner│◄────────────────┘               │
│          │             │  (behaviors) │                                 │
│          │             └──────────────┘                                 │
│          │                     │                                        │
│          │                     ▼                                        │
│          │             ┌──────────────┐                                 │
│          └────────────►│    tools/    │                                 │
│                        │  (metrics)   │                                 │
│                        └──────────────┘                                 │
│                                │                                        │
│                                ▼                                        │
│                        ┌──────────────┐                                 │
│                        │   Results    │                                 │
│                        │ (DS, SR, %)  │                                 │
│                        └──────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Metrics Produced

| Metric | Description | Formula |
|--------|-------------|---------|
| **Driving Score (DS)** | Main evaluation metric | `route_completion% × penalty_factor` |
| **Success Rate (SR)** | Percentage of perfect routes | `routes_without_infractions / 220` |
| **Multi-Ability** | Breakdown by scenario type | Overtaking, Merging, Emergency Brake, Give Way, Traffic Signs |
| **Efficiency** | Speed relative to limit | Penalizes too slow driving |
| **Smoothness** | Driving comfort | Jerk, acceleration, yaw metrics |

---

## 2. Directory Structure

```
leaderboard/
├── leaderboard/                          # Main Python package
│   ├── __init__.py
│   ├── leaderboard_evaluator.py          # [565 lines] Main orchestrator
│   │
│   ├── autoagents/                       # Agent interface layer
│   │   ├── autonomous_agent.py           # [161 lines] Base class for all agents
│   │   ├── agent_wrapper.py              # [~400 lines] Sensor validation & setup
│   │   ├── sensor_interface.py           # Sensor data synchronization
│   │   ├── dummy_agent.py                # Example: all sensors, no action
│   │   ├── npc_agent.py                  # Example: CARLA BasicAgent
│   │   ├── human_agent.py                # Manual keyboard control
│   │   ├── ros_base_agent.py             # ROS1/ROS2 bridge base
│   │   ├── ros1_agent.py                 # ROS1 implementation
│   │   └── ros2_agent.py                 # ROS2 implementation
│   │
│   ├── scenarios/                        # Scenario management
│   │   ├── route_scenario.py             # [495 lines] Route + scenario container
│   │   └── scenario_manager.py           # [~300 lines] Execution loop manager
│   │
│   ├── envs/                             # Environment interface
│   │   └── sensor_interface.py           # Sensor data handling
│   │
│   └── utils/                            # Support utilities
│       ├── statistics_manager.py         # [603 lines] Score calculation
│       ├── route_indexer.py              # Route iteration & resume
│       ├── route_parser.py               # XML → RouteConfiguration
│       ├── route_manipulation.py         # Route interpolation
│       ├── checkpoint_tools.py           # JSON save/load utilities
│       ├── result_writer.py              # Terminal output formatting
│       └── parked_vehicles.py            # [3.2MB] Parking slot database
│
├── scripts/                              # Shell scripts & helpers
│   ├── run_evaluation.sh                 # [44 lines] Core evaluation script
│   ├── run_evaluation_debug.sh           # Single route debug mode
│   ├── run_evaluation_multi_uniad.sh     # Multi-GPU parallel (UniAD)
│   ├── run_evaluation_multi_vad.sh       # Multi-GPU parallel (VAD)
│   ├── merge_statistics.py               # Combine multiple checkpoints
│   ├── route_summarizer.py               # Route info extraction
│   └── pretty_print_json.py              # JSON formatting
│
├── data/                                 # Route definitions
│   ├── bench2drive220.xml                # 220 official evaluation routes
│   ├── drivetransformer_bench2drive_dev10.xml  # 10 dev routes for quick iteration
│   ├── routes_training.xml               # Training routes
│   ├── routes_validation.xml             # Validation routes
│   └── weather.xml                       # Weather presets (18 conditions)
│
└── team_code/                            # Custom agent implementations
    └── your_agent.py                     # Place your agent here
```

---

## 3. How to Run Evaluations

### 3.1 Prerequisites

**Environment Setup**:
```bash
# 1. Install CARLA 0.9.15
mkdir carla && cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh

# 2. Set environment variables
export CARLA_ROOT=/path/to/carla
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> \
    $CONDA_PREFIX/lib/python3.7/site-packages/carla.pth

# 3. Install Python dependencies
pip install py-trees==0.8.3 networkx pygame opencv-python dictor
```

**Required Environment Variables** (set in run_evaluation.sh):
```bash
export CARLA_ROOT=YOUR_CARLA_PATH
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export SCENARIO_RUNNER_ROOT=scenario_runner
```

### 3.2 Debug Mode (Single Route)

For testing your agent on a single route:

```bash
# Edit run_evaluation_debug.sh first:
GPU_RANK=0                                    # Which GPU to use
TEAM_AGENT=leaderboard/team_code/your_agent.py
TEAM_CONFIG=your_model.pth                    # For TCP/ADMLP
# TEAM_CONFIG=config.py+checkpoint.pth        # For UniAD/VAD

# Run
bash leaderboard/scripts/run_evaluation_debug.sh
```

**What happens**:
1. Launches CARLA on specified GPU
2. Loads your agent
3. Runs all 220 routes (use `--routes-subset` to limit)
4. Saves results to `eval.json`

### 3.3 Production Mode (Multi-GPU Parallel)

For full evaluation across 220 routes:

```bash
# Edit run_evaluation_multi_uniad.sh:
GPU_RANK_LIST=(0 1 2 3 4 5 6 7)    # Available GPUs
TASK_LIST=(0 1 2 3 4 5 6 7)        # Task IDs (1:1 with GPUs)
TEAM_AGENT=team_code/uniad_b2d_agent.py
TEAM_CONFIG=path/to/config.py+path/to/checkpoint.pth

# Run
bash leaderboard/scripts/run_evaluation_multi_uniad.sh
```

**What happens**:
1. Splits `bench2drive220.xml` into N chunks (via `tools/split_xml.py`)
2. Launches N CARLA instances on different GPUs/ports
3. Each instance evaluates its subset of routes
4. Results saved to `{algo}_b2d_{planner}/eval_bench2drive220_{i}.json`

### 3.4 Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--routes` | XML file with route definitions | Required |
| `--agent` | Path to your agent Python file | Required |
| `--agent-config` | Model config/checkpoint path | Required |
| `--checkpoint` | Output JSON path for results | `./simulation_results.json` |
| `--port` | CARLA server port | 2000 |
| `--traffic-manager-port` | Traffic manager port | 8000 |
| `--gpu-rank` | GPU for CARLA (`-graphicsadapter`) | 0 |
| `--timeout` | Agent/scenario timeout in seconds | 600 |
| `--resume` | Resume from checkpoint | False |
| `--track` | Competition track (SENSORS/MAP) | SENSORS |
| `--repetitions` | Run each route N times | 1 |
| `--routes-subset` | Subset like "0-10" or "5,7,9" | "" (all) |

### 3.5 Resume from Checkpoint

If evaluation crashes (CARLA is unstable), you can resume:

```bash
# In run_evaluation.sh, set:
RESUME=True

# Re-run the same command
bash leaderboard/scripts/run_evaluation.sh ...
```

The system will:
1. Validate checkpoint matches route configuration
2. Skip already-completed routes
3. Repeat the last crashed route
4. Continue from there

### 3.6 Post-Processing Results

After evaluation completes:

```bash
# 1. Merge all route results (expects 220 routes)
python tools/merge_route_json.py -f your_json_folder/
# Output: merge.json with Driving Score and Success Rate

# 2. Get multi-ability breakdown
python tools/ability_benchmark.py -r merge.json
# Output: Scores by scenario type (Overtaking, Merging, etc.)

# 3. Get driving efficiency and smoothness
python tools/efficiency_smoothness_benchmark.py -f merge.json -m your_metric_folder/
# Output: Comfort metrics (jerk, acceleration, yaw)
```

---

## 4. Component Deep Dive

### 4.1 LeaderboardEvaluator

**File**: `leaderboard/leaderboard/leaderboard_evaluator.py` (565 lines)

The main orchestrator that coordinates the entire evaluation process.

#### Key Constants

```python
client_timeout = 300.0  # CARLA client timeout (seconds)
frame_rate = 20.0       # Simulation runs at 20 Hz
```

#### Initialization Flow

```python
class LeaderboardEvaluator:
    def __init__(self, args, statistics_manager):
        # 1. Setup CARLA simulation
        self.client, self.client_timeout, self.traffic_manager = self._setup_simulation(args)

        # 2. Validate CARLA version
        if LooseVersion(dist.version) < LooseVersion('0.9.10'):
            raise ImportError("CARLA version 0.9.10.1 or newer required")

        # 3. Dynamic agent import
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # 4. Create scenario manager
        self.manager = ScenarioManager(args.timeout, self.statistics_manager, args.debug)
```

#### CARLA Server Launch (`_setup_simulation`, lines 197-247)

```python
def _setup_simulation(self, args):
    # Find free port (avoid conflicts)
    args.port = find_free_port(args.port)

    # Launch CARLA server as subprocess
    cmd = f"{self.carla_path}/CarlaUE4.sh -RenderOffScreen -nosound " \
          f"-carla-rpc-port={args.port} -graphicsadapter={args.gpu_rank}"
    self.server = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

    # Register cleanup on exit
    atexit.register(os.killpg, self.server.pid, signal.SIGKILL)
    time.sleep(30)  # Wait for CARLA to start

    # Connect client with retry logic (up to 20 attempts)
    attempts = 0
    while attempts < 20:
        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(client_timeout)

            # Configure synchronous mode
            settings = carla.WorldSettings(
                synchronous_mode=True,
                fixed_delta_seconds=1.0 / 20.0,  # 20 Hz
                deterministic_ragdolls=True,
                spectator_as_ego=False
            )
            client.get_world().apply_settings(settings)
            break
        except Exception:
            attempts += 1
            time.sleep(5)

    # Setup traffic manager
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_hybrid_physics_mode(True)

    return client, client_timeout, traffic_manager
```

#### Main Evaluation Loop (`run`, lines 453-505)

```python
def run(self, args):
    # 1. Create route indexer (parses XML, handles subsets)
    route_indexer = RouteIndexer(args.routes, args.repetitions, args.routes_subset)

    # 2. Handle checkpoint resume
    if args.resume:
        resume = route_indexer.validate_and_resume(args.checkpoint)
        if resume:
            self.statistics_manager.add_file_records(args.checkpoint)

    # 3. Iterate through all routes
    crashed = False
    while route_indexer.peek() and not crashed:
        config = route_indexer.get_next_config()
        crashed = self._load_and_run_scenario(args, config)

        # Save progress after each route
        self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
        self.statistics_manager.write_statistics()

    # 4. Compute final statistics
    if not crashed:
        self.statistics_manager.compute_global_statistics()
        self.statistics_manager.validate_and_write_statistics(...)

    return crashed
```

#### Per-Route Execution (`_load_and_run_scenario`, lines 306-451)

```python
def _load_and_run_scenario(self, args, config):
    entry_status = "Started"

    # Phase 1: Prepare route metadata
    route_name = f"{config.name}_rep{config.repetition_index}"
    scenario_name = config.scenario_configs[0].name
    self.statistics_manager.create_route_data(...)

    # Phase 2: Load world
    try:
        self._load_and_wait_for_world(args, config.town)
        self.route_scenario = RouteScenario(world=self.world, config=config)
    except Exception:
        return True  # Crash, stop evaluation

    # Phase 3: Setup agent (with timeout watchdog)
    try:
        self._agent_watchdog = Watchdog(args.timeout)
        self._agent_watchdog.start()

        # Get agent class and instantiate
        agent_class_name = getattr(self.module_agent, 'get_entry_point')()
        agent_class = getattr(self.module_agent, agent_class_name)
        self.agent_instance = agent_class(args.host, args.port, args.debug)

        # Set route and config
        self.agent_instance.set_global_plan(self.route_scenario.gps_route,
                                            self.route_scenario.route)
        self.agent_instance.setup(args.agent_config)

        # Validate sensors
        self.sensors = self.agent_instance.sensors()
        validate_sensor_configuration(self.sensors, self.agent_instance.track, args.track)

        self._agent_watchdog.stop()
    except Exception:
        return False  # Agent error, continue to next route

    # Phase 4: Run scenario
    try:
        self.manager.load_scenario(self.route_scenario, self.agent_instance, ...)
        self.manager.run_scenario()  # Main loop
    except AgentError:
        entry_status = "Started"
        crash_message = "Agent crashed"
    except TickRuntimeError:
        # tick_count > 4000, safety limit
        entry_status = "Started"
        crash_message = "TickRuntime"

    # Phase 5: Cleanup
    self.manager.stop_scenario()
    self._register_statistics(config.index, entry_status, crash_message)
    self._cleanup()

    return crash_message == "Simulation crashed"
```

### 4.2 Agent Interface

**Files**: `leaderboard/leaderboard/autoagents/`

#### AutonomousAgent Base Class

**File**: `autonomous_agent.py` (161 lines)

All custom agents must inherit from this class:

```python
class Track(Enum):
    SENSORS = 'SENSORS'           # Cameras, LiDAR, Radar, GPS, IMU only
    MAP = 'MAP'                   # Above + OpenDRIVE map access
    SENSORS_QUALIFIER = 'SENSORS_QUALIFIER'  # Stricter sensor limits
    MAP_QUALIFIER = 'MAP_QUALIFIER'

class AutonomousAgent:
    def __init__(self, carla_host, carla_port, debug=False):
        self.track = Track.SENSORS  # Default track
        self._global_plan = None    # GPS route
        self._global_plan_world_coord = None  # World coordinates
        self.sensor_interface = SensorInterface()

    def setup(self, path_to_conf_file):
        """
        Initialize your agent. Called once before evaluation starts.

        Args:
            path_to_conf_file: Config path (e.g., "config.py+checkpoint.pth")

        Must set self.track to Track.SENSORS or Track.MAP
        """
        pass

    def sensors(self):
        """
        Define your sensor suite.

        Returns:
            List of sensor dictionaries, e.g.:
            [
                {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                 'width': 1600, 'height': 900, 'fov': 100, 'id': 'CAM_FRONT'},
                {'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': 2.0,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'id': 'LIDAR_TOP'},
            ]
        """
        return []

    def run_step(self, input_data, timestamp):
        """
        Execute one navigation step.

        Args:
            input_data: Dict of {sensor_id: (frame, data)}
            timestamp: Current game time

        Returns:
            carla.VehicleControl with throttle, steer, brake
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.steer = 0.0
        control.brake = 0.0
        return control

    def destroy(self):
        """Cleanup when agent is destroyed."""
        pass

    def __call__(self):
        """Called each tick to get control."""
        input_data = self.sensor_interface.get_data(GameTime.get_frame())
        timestamp = GameTime.get_time()
        control = self.run_step(input_data, timestamp)
        return control
```

#### Sensor Validation Rules

**File**: `agent_wrapper.py`

```python
# Sensor limits for different tracks
SENSORS_LIMITS = {
    'sensor.camera.rgb': 8,          # Max 8 RGB cameras
    'sensor.lidar.ray_cast': 2,      # Max 2 LiDARs
    'sensor.other.radar': 4,         # Max 4 radars
    'sensor.other.gnss': 1,          # Max 1 GPS
    'sensor.other.imu': 1,           # Max 1 IMU
    'sensor.opendrive_map': 1,       # Max 1 map reader (MAP track only)
    'sensor.speedometer': 1          # Max 1 speedometer
}

# For Bench2Drive: 100m radius allowed (vs 3m for standard leaderboard)
MAX_ALLOWED_RADIUS_SENSOR = 100.0  # When SAVE_PATH env var is set

def validate_sensor_configuration(sensors, agent_track, selected_track):
    """
    Validates:
    1. No duplicate sensor IDs
    2. Track restrictions (SENSORS can't use opendrive_map)
    3. All sensors within allowed types
    4. Position within radius: sqrt(x² + y² + z²) <= 100m
    5. Count within limits per type
    """
    ...
```

#### SensorInterface

**File**: `envs/sensor_interface.py`

Handles synchronized data delivery from all sensors:

```python
class SensorInterface:
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = Queue()  # Thread-safe queue
        self._queue_timeout = 300     # 300 second timeout

    def get_data(self, frame):
        """
        Block until all sensors have data for this frame.

        Returns:
            Dict of {sensor_id: (frame, data)}

        Raises:
            SensorReceivedNoData if timeout
        """
        data_dict = {}
        while len(data_dict) < len(self._sensors_objects):
            sensor_data = self._data_buffers.get(True, self._queue_timeout)
            if sensor_data[1] == frame:  # Match frame number
                data_dict[sensor_data[0]] = (sensor_data[1], sensor_data[2])
        return data_dict

class CallBack:
    """Sensor callback that parses CARLA sensor data."""

    def _parse_image(self, sensor_data):
        # CARLA image → numpy (H, W, 4) RGBA
        array = np.frombuffer(sensor_data.raw_data, dtype=np.uint8)
        return array.reshape((sensor_data.height, sensor_data.width, 4))

    def _parse_lidar(self, sensor_data):
        # CARLA LiDAR → numpy (N, 4) [x, y, z, intensity]
        points = np.frombuffer(sensor_data.raw_data, dtype=np.float32)
        return points.reshape((-1, 4))
```

### 4.3 RouteScenario

**File**: `leaderboard/leaderboard/scenarios/route_scenario.py` (495 lines)

Manages the route and dynamically spawns scenarios as the ego vehicle approaches.

#### Initialization

```python
class RouteScenario(BasicScenario):
    INIT_THRESHOLD = 500  # Initialize scenarios within 500m of ego

    def __init__(self, world, config, debug_mode=0):
        # 1. Interpolate route from XML keypoints
        self.gps_route, self.route = interpolate_trajectory(config.keypoints)

        # 2. Filter scenarios close to route (within 2m)
        scenario_configs = self._filter_scenarios(config.scenario_configs)
        self.missing_scenario_configurations = scenario_configs.copy()

        # 3. Spawn ego vehicle at first waypoint
        ego_vehicle = self._spawn_ego_vehicle()

        # 4. Get parking slots near route
        self._get_parking_slots()

        # 5. Initialize parent (creates behavior tree)
        super().__init__(config.name, [ego_vehicle], config, world, ...)

        # 6. Build initial scenarios within threshold
        self.build_scenarios(ego_vehicle, debug=debug_mode > 0)
```

#### Dynamic Scenario Building

Called periodically in background thread:

```python
def build_scenarios(self, ego_vehicle, debug=False):
    """Initialize scenarios when ego gets close."""
    new_scenarios = []

    for scenario_config in self.missing_scenario_configurations:
        trigger_location = scenario_config.trigger_points[0].location
        ego_location = CarlaDataProvider.get_location(ego_vehicle)

        # Only init if within 500m
        if trigger_location.distance(ego_location) < self.INIT_THRESHOLD:
            # Instantiate scenario class
            scenario_class = self.all_scenario_classes[scenario_config.type]
            scenario_instance = scenario_class(self.world, [ego_vehicle],
                                               scenario_config, timeout=self.timeout)

            # Add to tracking
            self.list_scenarios.append(scenario_instance)
            new_scenarios.append(scenario_instance)
            self.missing_scenario_configurations.remove(scenario_config)

    # Add behaviors to tree
    for scenario in new_scenarios:
        if scenario.behavior_tree is not None:
            self.behavior_node.add_child(scenario.behavior_tree)
            self.scenario_triggerer.add_blackboard([
                scenario.config.route_var_name,
                scenario.config.trigger_points[0].location
            ])
```

#### Criteria System

Always-active criteria that run throughout the route:

```python
def _create_test_criteria(self):
    criteria = py_trees.composites.Parallel(
        name="Criteria",
        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
    )

    # End condition: route completion
    criteria.add_child(RouteCompletionTest(self.ego_vehicles[0], route=self.route))

    # Infraction detection
    criteria.add_child(OutsideRouteLanesTest(self.ego_vehicles[0], route=self.route))
    criteria.add_child(CollisionTest(self.ego_vehicles[0], name="CollisionTest"))
    criteria.add_child(RunningRedLightTest(self.ego_vehicles[0]))
    criteria.add_child(RunningStopTest(self.ego_vehicles[0]))
    criteria.add_child(MinimumSpeedRouteTest(self.ego_vehicles[0], self.route,
                                              checkpoints=20, name="MinSpeedTest"))

    # Early termination conditions
    criteria.add_child(InRouteTest(self.ego_vehicles[0], route=self.route,
                                   offroad_max=30, terminate_on_failure=True))
    criteria.add_child(ActorBlockedTest(self.ego_vehicles[0], min_speed=0.1,
                                        max_time=60.0, terminate_on_failure=True))

    return criteria
```

### 4.4 ScenarioManager

**File**: `leaderboard/leaderboard/scenarios/scenario_manager.py` (~300 lines)

Manages the main simulation loop.

#### Main Tick Loop

```python
def _tick_scenario(self):
    """Called repeatedly until scenario ends."""
    while self._running:
        # 1. Tick CARLA world (advance simulation)
        CarlaDataProvider.get_world().tick(self._timeout)

        # 2. Update timing and actor info
        self._watchdog.update()
        self._agent_watchdog.resume()

        # 3. Get agent action
        ego_action = self._agent_wrapper()  # Calls agent.run_step()

        # 4. Apply control to ego vehicle
        self.ego_vehicles[0].apply_control(ego_action)

        # 5. Tick behavior tree (scenario logic)
        self.scenario_tree.tick_once()

        # 6. Check termination
        if self.scenario_tree.status != py_trees.common.Status.RUNNING:
            self._running = False

        # 7. Safety limit: prevent infinite loops
        self.tick_count += 1
        if self.tick_count > 4000:
            raise TickRuntimeError("tick_count > 4000")
```

#### Background Scenario Thread

```python
def build_scenarios_loop(self, debug):
    """Runs in background thread."""
    while self._running:
        # Periodically check for new scenarios to initialize
        self.scenario.build_scenarios(self.ego_vehicles[0], debug=debug)

        # Spawn parked vehicles near ego
        self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])

        time.sleep(1)  # Check every second
```

### 4.5 StatisticsManager

**File**: `leaderboard/leaderboard/utils/statistics_manager.py` (603 lines)

Handles all score calculation and result tracking.

#### Penalty System

```python
# Fixed penalties (multiplicative)
PENALTY_VALUE_DICT = {
    TrafficEventType.COLLISION_PEDESTRIAN: 0.5,    # 50% of score
    TrafficEventType.COLLISION_VEHICLE: 0.6,       # 60% of score
    TrafficEventType.COLLISION_STATIC: 0.65,       # 65% of score
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 0.7,# 70% of score
    TrafficEventType.STOP_INFRACTION: 0.8,         # 80% of score
    TrafficEventType.SCENARIO_TIMEOUT: 0.7,        # 70% of score
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 0.7,
}

# Variable penalties (currently disabled)
PENALTY_PERC_DICT = {
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: [0, 'increases'],  # Disabled
    TrafficEventType.MIN_SPEED_INFRACTION: [0.7, 'unused'],             # Disabled
}

# Infraction names for JSON output
PENALTY_NAME_DICT = {
    TrafficEventType.COLLISION_STATIC: 'collisions_layout',
    TrafficEventType.COLLISION_PEDESTRIAN: 'collisions_pedestrian',
    TrafficEventType.COLLISION_VEHICLE: 'collisions_vehicle',
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 'red_light',
    TrafficEventType.STOP_INFRACTION: 'stop_infraction',
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: 'outside_route_lanes',
    TrafficEventType.MIN_SPEED_INFRACTION: 'min_speed_infractions',
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 'yield_emergency_vehicle_infractions',
    TrafficEventType.SCENARIO_TIMEOUT: 'scenario_timeouts',
    TrafficEventType.ROUTE_DEVIATION: 'route_dev',
    TrafficEventType.VEHICLE_BLOCKED: 'vehicle_blocked',
}
```

#### Score Calculation

```python
def compute_route_statistics(self, route_index, duration_system, duration_game, failure_message=""):
    """Compute scores for a single route."""
    route_record = self._results.checkpoint.records[route_index]

    score_penalty = 1.0  # Start at 100%
    score_route = 0.0    # Route completion %

    # Process all criteria events
    for node in self._scenario.get_criteria():
        for event in node.events:
            # Fixed penalties (multiply)
            if event.get_type() in PENALTY_VALUE_DICT:
                score_penalty *= PENALTY_VALUE_DICT[event.get_type()]

            # Route completion event
            elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                score_route = event.get_dict()['route_completed']

    # Final scores
    route_record.scores['score_route'] = score_route
    route_record.scores['score_penalty'] = score_penalty
    route_record.scores['score_composed'] = max(score_route * score_penalty, 0.0)

    # Determine status
    if score_route >= 100:
        route_record.status = 'Perfect' if score_penalty == 1.0 else 'Completed'
    else:
        route_record.status = 'Failed - ' + failure_message
```

#### Checkpoint Format

```python
class Results:
    def to_json(self):
        return {
            '_checkpoint': {
                'global_record': {...},  # Aggregated stats
                'progress': [current_route, total_routes],
                'records': [...]  # Per-route records
            },
            'entry_status': 'Finished',  # Started/Finished/Crashed/Invalid
            'eligible': True,
            'sensors': ['carla_camera', 'carla_lidar', ...],
            'values': [...],  # Global metrics
            'labels': [...]   # Metric names
        }

class RouteRecord:
    def to_json(self):
        return {
            'index': 0,
            'route_id': 'RouteScenario_0_rep0',
            'scenario_name': 'VehicleTurningRoute',
            'weather_id': 'ClearNoon',
            'status': 'Completed',
            'scores': {
                'score_route': 100.0,
                'score_penalty': 0.7,
                'score_composed': 70.0
            },
            'infractions': {
                'collisions_pedestrian': [],
                'collisions_vehicle': ['Collision with vehicle...'],
                'red_light': [],
                ...
            },
            'meta': {
                'route_length': 1234.5,
                'duration_game': 45.2,
                'duration_system': 90.1
            }
        }
```

### 4.6 Route System

#### RouteParser

**File**: `leaderboard/leaderboard/utils/route_parser.py`

Parses XML route definitions:

```xml
<!-- Example route in bench2drive220.xml -->
<route id="0" town="Town12">
    <waypoints>
        <position x="-30.0" y="20.5" z="0.0"/>
        <position x="-30.0" y="100.0" z="0.0"/>
        <position x="50.0" y="100.0" z="0.0"/>
    </waypoints>
    <scenarios>
        <scenario name="VehicleTurningRoute" type="VehicleTurningRoute">
            <trigger_point x="-10.0" y="60.0" z="0.0" yaw="90.0"/>
        </scenario>
    </scenarios>
    <weathers>
        <weather cloudiness="50" precipitation="0" sun_altitude_angle="70"
                 route_percentage="0"/>
    </weathers>
</route>
```

```python
def parse_routes_file(route_filename, scenario_file=None):
    """Parse XML to RouteScenarioConfiguration objects."""
    tree = ET.parse(route_filename)

    routes = []
    for route in tree.iter('route'):
        config = RouteScenarioConfiguration()
        config.town = route.attrib['town']
        config.name = f"RouteScenario_{route.attrib['id']}"

        # Parse waypoints
        config.keypoints = []
        for position in route.find('waypoints').iter('position'):
            config.keypoints.append(carla.Location(
                float(position.attrib['x']),
                float(position.attrib['y']),
                float(position.attrib['z'])
            ))

        # Parse scenarios
        config.scenario_configs = parse_scenarios(route.find('scenarios'))

        # Parse weather
        config.weather = parse_weather(route.find('weathers'))

        routes.append(config)

    return routes
```

#### Route Interpolation

**File**: `leaderboard/leaderboard/utils/route_manipulation.py`

Converts sparse keypoints to dense trajectory:

```python
def interpolate_trajectory(keypoints, hop_resolution=1.0):
    """
    Interpolate sparse keypoints to dense trajectory.

    Args:
        keypoints: List of carla.Location (sparse, from XML)
        hop_resolution: Distance between waypoints in meters (default 1m)

    Returns:
        gps_route: [(lat, lon, RoadOption), ...]
        route: [(carla.Transform, RoadOption), ...]
    """
    grp = GlobalRoutePlanner(CarlaDataProvider.get_map(), hop_resolution)

    route = []
    for i in range(len(keypoints) - 1):
        start = keypoints[i]
        end = keypoints[i + 1]

        # Trace route on road network
        segment = grp.trace_route(start, end)
        route.extend(segment)

    # Convert to GPS coordinates
    gps_route = []
    for waypoint, road_option in route:
        lat, lon = _location_to_gps(waypoint.transform.location)
        gps_route.append((lat, lon, road_option))

    return gps_route, route
```

---

## 5. Data Flow

### 5.1 Complete Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Evaluation Execution Flow                         │
└─────────────────────────────────────────────────────────────────────────┘

run_evaluation.sh
    │
    ▼
LeaderboardEvaluator.__init__()
    ├─► _setup_simulation()
    │       ├─► find_free_port() ─────────────────────► Available port
    │       ├─► subprocess.Popen(CarlaUE4.sh) ────────► CARLA server
    │       ├─► time.sleep(30) ───────────────────────► Wait for startup
    │       ├─► carla.Client() ───────────────────────► Connection
    │       └─► TrafficManager setup ─────────────────► NPC traffic control
    │
    ├─► importlib.import_module(agent) ───────────────► Your agent class
    │
    └─► ScenarioManager() ────────────────────────────► Execution manager
    │
    ▼
LeaderboardEvaluator.run()
    │
    ├─► RouteIndexer(routes.xml)
    │       ├─► RouteParser.parse_routes_file() ──────► List[RouteConfig]
    │       └─► Handle --routes-subset ───────────────► Filtered configs
    │
    ├─► Check --resume flag
    │       └─► validate_and_resume() ────────────────► Skip completed routes
    │
    └─► FOR EACH route_config:
            │
            ▼
        _load_and_run_scenario()
            │
            ├─► _load_and_wait_for_world(town)
            │       ├─► client.load_world(town) ──────► Load map
            │       └─► CarlaDataProvider.set_world()
            │
            ├─► RouteScenario(config)
            │       ├─► interpolate_trajectory() ─────► Dense route
            │       ├─► _filter_scenarios() ──────────► Near-route scenarios
            │       ├─► _spawn_ego_vehicle() ─────────► Lincoln MKZ at start
            │       └─► build_scenarios() ────────────► Init nearby scenarios
            │
            ├─► Agent Setup (with Watchdog)
            │       ├─► agent.setup(config) ──────────► Load model
            │       ├─► agent.sensors() ──────────────► Sensor definitions
            │       ├─► validate_sensor_configuration()
            │       └─► AgentWrapper.setup_sensors() ─► Spawn CARLA sensors
            │
            ├─► ScenarioManager.run_scenario()
            │       │
            │       └─► TICK LOOP (20 Hz):
            │               ├─► world.tick() ─────────► Advance simulation
            │               ├─► agent() ──────────────► Get VehicleControl
            │               │       └─► sensor_interface.get_data()
            │               │       └─► agent.run_step(input_data)
            │               ├─► ego.apply_control() ──► Move vehicle
            │               ├─► behavior_tree.tick() ─► Scenario logic
            │               └─► Check criteria ───────► Track infractions
            │
            ├─► compute_route_statistics()
            │       └─► Calculate scores, record infractions
            │
            └─► write_statistics()
                    └─► Save to checkpoint JSON
            │
            ▼
    compute_global_statistics()
        └─► Aggregate across all routes
            │
            ▼
    Output: simulation_results.json
```

### 5.2 Agent Data Flow

```
                    ┌──────────────────────────────┐
                    │        CARLA World           │
                    │  ┌────────────────────────┐  │
                    │  │   RGB Cameras (×8)     │  │
                    │  │   LiDAR (×2)           │──┼──► Raw sensor data
                    │  │   Radar (×4)           │  │
                    │  │   GPS, IMU             │  │
                    │  └────────────────────────┘  │
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      SensorInterface         │
                    │  ┌────────────────────────┐  │
                    │  │   CallBack parsers     │  │
                    │  │   Frame synchronization│──┼──► Synchronized data dict
                    │  │   Thread-safe queue    │  │    {sensor_id: (frame, data)}
                    │  └────────────────────────┘  │
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Your Agent            │
                    │  ┌────────────────────────┐  │
                    │  │   Perception           │  │
                    │  │   Planning             │──┼──► carla.VehicleControl
                    │  │   Control              │  │    (throttle, steer, brake)
                    │  └────────────────────────┘  │
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Ego Vehicle           │
                    │  ┌────────────────────────┐  │
                    │  │   apply_control()      │  │
                    │  │   Physics simulation   │──┼──► Vehicle movement
                    │  └────────────────────────┘  │
                    └──────────────────────────────┘
```

---

## 6. Scoring System

### 6.1 Driving Score Formula

```
Driving Score = Route Completion (%) × Penalty Factor

Where:
- Route Completion: Percentage of route waypoints reached (0-100%)
- Penalty Factor: Product of all infraction penalties (0-1.0)
```

### 6.2 Penalty Multipliers

| Infraction Type | Penalty | Effect |
|-----------------|---------|--------|
| Collision with pedestrian | 0.50 | Score × 0.50 |
| Collision with vehicle | 0.60 | Score × 0.60 |
| Collision with static object | 0.65 | Score × 0.65 |
| Running red light | 0.70 | Score × 0.70 |
| Ignoring stop sign | 0.80 | Score × 0.80 |
| Scenario timeout | 0.70 | Score × 0.70 |
| Not yielding to emergency vehicle | 0.70 | Score × 0.70 |

### 6.3 Example Calculation

```
Route: 100% completion
Infractions: 1 vehicle collision + 1 red light

Penalty = 0.60 × 0.70 = 0.42
Driving Score = 100 × 0.42 = 42.0

Status: "Completed" (route finished but with infractions)
```

### 6.4 Route Status Values

| Status | Meaning |
|--------|---------|
| `Perfect` | 100% completion, no infractions |
| `Completed` | 100% completion, has infractions |
| `Failed - Agent timed out` | Route timeout |
| `Failed - Agent got blocked` | Stuck for >60 seconds |
| `Failed - Agent deviated` | >30m off route |
| `Started` | Agent crashed during execution |
| `Crashed` | Simulation crashed |

### 6.5 Global Statistics

```python
# Computed across all 220 routes
global_record = {
    'scores_mean': {
        'score_composed': 45.5,    # Average driving score
        'score_route': 82.3,       # Average route completion
        'score_penalty': 0.55      # Average penalty factor
    },
    'scores_std_dev': {
        'score_composed': 25.1,
        'score_route': 20.5,
        'score_penalty': 0.15
    },
    'infractions': {  # Per kilometer driven
        'collisions_pedestrian': 0.02,
        'collisions_vehicle': 0.15,
        'red_light': 0.08,
        ...
    }
}
```

---

## 7. CARLA Integration

### 7.1 GPU Selection

**Important**: CARLA ignores `CUDA_VISIBLE_DEVICES`. Use `-graphicsadapter` instead:

```bash
# WRONG: CARLA will still use GPU 0
CUDA_VISIBLE_DEVICES=1 ./CarlaUE4.sh

# CORRECT: Use graphicsadapter flag
./CarlaUE4.sh -graphicsadapter=1
```

**Note**: GPU mapping may vary by machine. Sometimes `-graphicsadapter=1` is unavailable:
```
GPU0 → -graphicsadapter=0
GPU1 → -graphicsadapter=2  # Skip 1!
GPU2 → -graphicsadapter=3
GPU3 → -graphicsadapter=4
```

### 7.2 Port Management

For multi-GPU parallel evaluation:

```bash
# Each CARLA instance needs unique ports
# Instance 0: port=30000, tm_port=50000
# Instance 1: port=30150, tm_port=50150
# Instance 2: port=30300, tm_port=50300
# ...

# Check port availability
lsof -i:30000
```

### 7.3 Synchronous Mode

Configured in `_setup_simulation`:

```python
settings = carla.WorldSettings(
    synchronous_mode=True,           # Simulation waits for tick()
    fixed_delta_seconds=1.0/20.0,    # 20 Hz = 50ms per tick
    deterministic_ragdolls=True,     # Reproducible physics
    spectator_as_ego=False           # Don't follow spectator
)
```

### 7.4 Traffic Manager

```python
traffic_manager = client.get_trafficmanager(tm_port)
traffic_manager.set_synchronous_mode(True)   # Sync with simulation
traffic_manager.set_hybrid_physics_mode(True) # Simplified physics for distant NPCs
traffic_manager.set_random_device_seed(seed)  # Reproducibility
```

---

## 8. Cross-Module Connections

### 8.1 Connection to scenario_runner/

The leaderboard module heavily depends on scenario_runner:

| scenario_runner File | Used By | Purpose |
|---------------------|---------|---------|
| `srunner/scenariomanager/carla_data_provider.py` | All | Singleton for CARLA world/client access |
| `srunner/scenariomanager/timer.py` | leaderboard_evaluator.py | GameTime for synchronized timing |
| `srunner/scenariomanager/watchdog.py` | leaderboard_evaluator.py | Timeout protection for agent |
| `srunner/scenariomanager/traffic_events.py` | statistics_manager.py | TrafficEventType enum |
| `srunner/scenariomanager/scenarioatomics/atomic_behaviors.py` | route_scenario.py | ScenarioTriggerer, Idle |
| `srunner/scenariomanager/scenarioatomics/atomic_criteria.py` | route_scenario.py | All test criteria |
| `srunner/scenarios/*.py` | route_scenario.py | 40+ scenario implementations |

### 8.2 Connection to tools/

Post-processing scripts that consume leaderboard output:

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `tools/merge_route_json.py` | `**/eval*.json` | `merge.json` | Aggregate 220 routes |
| `tools/ability_benchmark.py` | `merge.json` | Console | Multi-ability scores |
| `tools/efficiency_smoothness_benchmark.py` | `merge.json` + metrics | Console | Comfort metrics |
| `tools/clean_carla.sh` | - | - | Kill stuck processes |
| `tools/split_xml.py` | `bench2drive220.xml` | `*_0.xml`, `*_1.xml`, ... | Split for parallel eval |

### 8.3 Data Flow Between Modules

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  leaderboard/                           scenario_runner/               │
│      │                                       │                         │
│      ├─► leaderboard_evaluator.py ──────────►│                         │
│      │       │  import CarlaDataProvider     │ carla_data_provider.py  │
│      │       │  import GameTime              │ timer.py                │
│      │       │  import Watchdog              │ watchdog.py             │
│      │       ▼                               │                         │
│      ├─► route_scenario.py ─────────────────►│                         │
│      │       │  extends BasicScenario        │ basic_scenario.py       │
│      │       │  uses AtomicBehaviors         │ atomic_behaviors.py     │
│      │       │  uses AtomicCriteria          │ atomic_criteria.py      │
│      │       │  loads scenario classes       │ scenarios/*.py          │
│      │       ▼                               │                         │
│      └─► statistics_manager.py ◄────────────│                         │
│              │  uses TrafficEventType        │ traffic_events.py       │
│              ▼                               │                         │
│         simulation_results.json              │                         │
│              │                               │                         │
└──────────────┼───────────────────────────────┴─────────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│            tools/                     │
│                                       │
│   merge_route_json.py ◄──────────────┤
│         │                             │
│         ▼                             │
│   merge.json                          │
│         │                             │
│         ├────► ability_benchmark.py   │
│         │           │                 │
│         │           ▼                 │
│         │      Multi-ability scores   │
│         │                             │
│         └────► efficiency_smoothness_ │
│                benchmark.py           │
│                    │                  │
│                    ▼                  │
│               Comfort metrics         │
└───────────────────────────────────────┘
```

---

## 9. Troubleshooting

### 9.1 CARLA Won't Start

**Symptom**: CARLA exits immediately or shows Vulkan errors.

**Solution**:
```bash
# Check Vulkan installation
/usr/bin/vulkaninfo | head -n 5

# Reinstall if needed
sudo apt install vulkan-tools vulkan-utils

# Good output: "Vulkan Instance Version: 1.x"
# Bad output: "WARNING: lavapipe is not a conformant vulkan implementation"
```

**NVIDIA Driver**: Version 470 is most stable. 550 has bugs.

### 9.2 Port Conflicts

**Symptom**: "Address already in use" or connection timeouts.

**Solution**:
```bash
# Check what's using the port
lsof -i:30000

# Kill CARLA processes
bash tools/clean_carla.sh

# Run multiple times if needed
bash tools/clean_carla.sh
bash tools/clean_carla.sh
```

### 9.3 Agent Timeout

**Symptom**: "Timeout: Agent took longer than 600s to setup"

**Solutions**:
1. Increase timeout: `--timeout 1200`
2. Check model loading time
3. Ensure GPU memory is sufficient

### 9.4 Simulation Crashes

**Symptom**: Random crashes during evaluation.

**Solutions**:
1. Use `--resume=True` to continue from checkpoint
2. Increase sleep times in scripts for slower machines:
   - `leaderboard_evaluator.py:207` - 30s → 60s
   - `run_evaluation_multi_*.sh:58` - 5s → 15s
3. Run `bash tools/clean_carla.sh` before starting

### 9.5 Missing Routes in Results

**Symptom**: `merge_route_json.py` reports less than 220 routes.

**Solutions**:
1. Check all JSON files in output folder
2. Look for crashed routes: `grep -r "Crashed" *.json`
3. Re-run with `--resume=True` to complete missing routes

### 9.6 Sensor Data Missing

**Symptom**: `SensorReceivedNoData` exception.

**Solutions**:
1. Check sensor configuration matches CARLA version
2. Increase `_queue_timeout` in `sensor_interface.py` (default 300s)
3. Verify no duplicate sensor IDs

---

## Appendix A: Creating a Custom Agent

```python
# team_code/my_agent.py

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
import carla

def get_entry_point():
    """Required: Returns your agent class name."""
    return 'MyAgent'

class MyAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Called once at start. Load your model here."""
        self.track = Track.SENSORS  # or Track.MAP

        # Parse config: "config.py+checkpoint.pth+save_name"
        config_path, ckpt_path, save_name = path_to_conf_file.split('+')

        # Load your model
        self.model = load_model(ckpt_path)

    def sensors(self):
        """Define your sensor suite."""
        return [
            # Front camera
            {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1600, 'height': 900, 'fov': 100, 'id': 'CAM_FRONT'},

            # LiDAR
            {'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': 2.5,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'id': 'LIDAR_TOP'},

            # GPS
            {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 2.0,
             'id': 'GPS'},

            # IMU
            {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 2.0,
             'id': 'IMU'},

            # Speedometer (pseudo-sensor)
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'SPEED'},
        ]

    def run_step(self, input_data, timestamp):
        """Called every tick. Return vehicle control."""
        # Get sensor data
        rgb = input_data['CAM_FRONT'][1]       # (H, W, 4) RGBA
        lidar = input_data['LIDAR_TOP'][1]     # (N, 4) [x,y,z,intensity]
        gps = input_data['GPS'][1]             # [lat, lon, alt]
        imu = input_data['IMU'][1]             # [acc, gyro, compass]
        speed = input_data['SPEED'][1]['speed'] # m/s

        # Your model inference
        throttle, steer, brake = self.model.predict(rgb, lidar, gps, imu, speed)

        # Return control
        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.steer = float(steer)
        control.brake = float(brake)
        return control

    def destroy(self):
        """Cleanup when agent is destroyed."""
        del self.model
```

---

## Appendix B: Quick Reference

### Essential Commands

```bash
# Debug single route
bash leaderboard/scripts/run_evaluation_debug.sh

# Full 220-route evaluation (8 GPUs)
bash leaderboard/scripts/run_evaluation_multi_uniad.sh

# Kill stuck CARLA (run multiple times)
bash tools/clean_carla.sh

# Merge results
python tools/merge_route_json.py -f results_folder/

# Get multi-ability scores
python tools/ability_benchmark.py -r merge.json

# Get efficiency/smoothness
python tools/efficiency_smoothness_benchmark.py -f merge.json -m metrics_folder/
```

### Important File Paths

| File | Purpose |
|------|---------|
| `leaderboard/leaderboard/leaderboard_evaluator.py` | Main orchestrator |
| `leaderboard/leaderboard/autoagents/autonomous_agent.py` | Agent base class |
| `leaderboard/leaderboard/utils/statistics_manager.py` | Score calculation |
| `leaderboard/data/bench2drive220.xml` | 220 evaluation routes |
| `leaderboard/scripts/run_evaluation.sh` | Core eval script |

### Default Ports

| Port | Purpose |
|------|---------|
| 30000+ | CARLA server (increment by 150 for parallel) |
| 50000+ | Traffic manager (increment by 150 for parallel) |

---

*Documentation generated for Bench2Drive leaderboard module. Last updated based on code analysis.*
