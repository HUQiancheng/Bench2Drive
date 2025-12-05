# Scenario Runner Module - Comprehensive Documentation

## Overview

The `scenario_runner` module is the core engine that orchestrates **traffic scenario execution** in Bench2Drive. It manages NPC actors, defines complex driving scenarios using behavior trees, and evaluates agent performance through criteria tracking.

**Role in the Bench2Drive System:**
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
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  ActorControls  │  │  ScenarioAtomics │  │   Traffic Events  │  │
│  │  (NPC control)  │  │ (behavior atoms) │  │   (infractions)   │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          CARLA Simulator                            │
│         (physics, rendering, traffic lights, weather)               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
scenario_runner/
└── srunner/
    ├── scenariomanager/           # Core execution engine
    │   ├── carla_data_provider.py # Singleton for CARLA world access
    │   ├── timer.py               # Game time tracking
    │   ├── watchdog.py            # Timeout detection
    │   ├── traffic_events.py      # Infraction event types
    │   ├── scenarioatomics/       # Building blocks for behaviors
    │   │   ├── atomic_behaviors.py      # 40+ behavior primitives
    │   │   ├── atomic_criteria.py       # 10+ evaluation criteria
    │   │   └── atomic_trigger_conditions.py  # 25+ trigger conditions
    │   └── actorcontrols/         # NPC actor controllers
    │       ├── basic_control.py
    │       ├── simple_vehicle_control.py
    │       ├── npc_vehicle_control.py
    │       ├── pedestrian_control.py
    │       └── ...
    ├── scenarios/                 # 40+ scenario implementations
    │   ├── basic_scenario.py      # Base class for all scenarios
    │   ├── pedestrian_crossing.py
    │   ├── cut_in.py
    │   ├── object_crash_vehicle.py
    │   └── ...
    └── tools/                     # Helper utilities
        ├── scenario_helper.py
        ├── route_manipulation.py
        └── background_manager.py
```

---

## Core Components

### 1. CarlaDataProvider (`carla_data_provider.py`)

The **central singleton** that manages all CARLA world access. Every component uses this class to interact with the simulator.

**Location:** `scenario_runner/srunner/scenariomanager/carla_data_provider.py`
**Lines:** ~800

#### Key Responsibilities:
- Maintain references to CARLA world, map, and blueprint library
- Track all spawned actors (ego vehicles, NPCs, sensors)
- Provide cached access to actor locations, velocities, transforms
- Manage random seed for reproducibility
- Handle actor spawning with collision retry logic

#### Critical Methods:

```python
# World/Map Access
CarlaDataProvider.get_world()           # Returns carla.World
CarlaDataProvider.get_map()             # Returns carla.Map (cached)
CarlaDataProvider.get_client()          # Returns carla.Client

# Actor Information (cached for performance)
CarlaDataProvider.get_location(actor)   # Returns carla.Location
CarlaDataProvider.get_velocity(actor)   # Returns float (m/s scalar)
CarlaDataProvider.get_transform(actor)  # Returns carla.Transform

# Actor Spawning
CarlaDataProvider.request_new_actor(    # Spawn with collision retry
    model='vehicle.tesla.model3',
    spawn_point=carla.Transform(...),
    rolename='hero',
    autopilot=False,
    random_location=False
)

# Batch Operations
CarlaDataProvider.register_actors(actors)    # Track multiple actors
CarlaDataProvider.cleanup()                  # Destroy all tracked actors

# Traffic Management
CarlaDataProvider.set_ego_vehicle_route(route)
CarlaDataProvider.get_global_route_planner()
```

#### Internal Actor Tracking:

```python
# These dictionaries are updated every tick via on_carla_tick()
_actor_velocity_map = {}      # actor.id -> carla.Vector3D
_actor_location_map = {}      # actor.id -> carla.Location
_actor_transform_map = {}     # actor.id -> carla.Transform
```

#### Usage in Scenarios:

```python
class MyScenario(BasicScenario):
    def _initialize_actors(self, config):
        # Spawn an NPC vehicle
        vehicle = CarlaDataProvider.request_new_actor(
            'vehicle.audi.a2',
            spawn_transform
        )
        self.other_actors.append(vehicle)

    def _create_behavior(self):
        # Get ego location dynamically
        ego_loc = CarlaDataProvider.get_location(self.ego_vehicles[0])
```

---

### 2. BasicScenario (`basic_scenario.py`)

The **base class** that all 40+ scenarios inherit from. Defines the contract for scenario implementation.

**Location:** `scenario_runner/srunner/scenarios/basic_scenario.py`
**Lines:** ~300

#### Abstract Methods to Implement:

```python
class BasicScenario:
    def _initialize_actors(self, config):
        """
        Spawn NPC actors for this scenario.
        Called during __init__ before behavior tree construction.

        Args:
            config: ScenarioConfiguration with trigger_points, other_actors, etc.

        Must populate: self.other_actors list
        """
        raise NotImplementedError

    def _create_behavior(self):
        """
        Build and return the py_trees behavior tree root.

        Returns:
            py_trees.behaviour.Behaviour: Root of scenario behavior tree
        """
        raise NotImplementedError

    def _create_test_criteria(self):
        """
        Create list of criteria to evaluate agent performance.

        Returns:
            list[Criterion]: List of atomic criteria (CollisionTest, etc.)
        """
        raise NotImplementedError
```

#### Key Attributes:

```python
class BasicScenario:
    ego_vehicles = []       # List of ego vehicle actors
    other_actors = []       # List of NPC actors spawned by scenario
    scenario_tree = None    # Root py_trees behavior tree
    criteria_tree = None    # Parallel criteria evaluation tree
    timeout = 60            # Scenario timeout in seconds
    route_mode = False      # True when running as part of route evaluation
```

#### Behavior Tree Structure:

Every scenario builds a tree with this structure:
```
                    ┌─────────────────┐
                    │  Scenario Root  │
                    │   (Parallel)    │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Scenario    │  │   Criteria   │  │   Timeout    │
    │  Behavior    │  │  (Parallel)  │  │   Handler    │
    └──────────────┘  └──────────────┘  └──────────────┘
```

---

### 3. ScenarioAtomics - The Building Blocks

The `scenarioatomics` module provides **reusable primitives** that scenarios compose into complex behaviors.

#### 3.1 Atomic Behaviors (`atomic_behaviors.py`)

**Location:** `scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py`
**Lines:** ~2500
**Classes:** 40+

These are **actions** that actors perform. All inherit from `py_trees.behaviour.Behaviour`.

##### Movement Behaviors:

| Behavior | Description | Key Parameters |
|----------|-------------|----------------|
| `KeepVelocity` | Maintain constant speed | `actor`, `target_velocity`, `duration`, `distance` |
| `WaypointFollower` | Follow waypoint route | `actor`, `target_velocity`, `plan` (waypoint list) |
| `LaneChange` | Change to adjacent lane | `actor`, `direction` ('left'/'right'), `distance_same_lane` |
| `AccelerateToCatchUp` | Speed up to reach another actor | `actor`, `other_actor`, `throttle_value`, `delta_velocity` |
| `BasicAgentBehavior` | Use CARLA BasicAgent | `actor`, `target_location`, `target_speed` |

##### Actor Management Behaviors:

| Behavior | Description | Key Parameters |
|----------|-------------|----------------|
| `ActorTransformSetter` | Teleport actor to position | `actor`, `transform`, `physics` (bool) |
| `ActorDestroy` | Remove actor from world | `actor` |
| `ActorSource` | Spawn actors from pool | `actor_type_list`, `transform`, `threshold` |
| `ActorSink` | Destroy actors in region | `location`, `threshold` |

##### Control Behaviors:

| Behavior | Description | Key Parameters |
|----------|-------------|----------------|
| `StopVehicle` | Apply brake | `actor`, `brake_value` |
| `SyncArrival` | Synchronize two actors' arrival | `actor`, `master_actor`, `target_location` |
| `SetTrafficLightState` | Control traffic light | `actor`, `state` (carla.TrafficLightState) |
| `TrafficLightFreezer` | Hold traffic light state | `traffic_lights_dict` |

##### Wait/Timing Behaviors:

| Behavior | Description | Key Parameters |
|----------|-------------|----------------|
| `Idle` | Do nothing for duration | `duration` |
| `WaitForever` | Never complete | - |

##### Example - KeepVelocity Implementation:

```python
class KeepVelocity(AtomicBehavior):
    """
    Keep actor at constant velocity for duration/distance.

    Returns SUCCESS when duration or distance exceeded.
    Returns RUNNING while maintaining velocity.
    """

    def __init__(self, actor, target_velocity,
                 avoid_collision=False, duration=float("inf"),
                 distance=float("inf"), name="KeepVelocity"):
        self._actor = actor
        self._target_velocity = target_velocity
        self._duration = duration
        self._distance = distance
        self._control = carla.VehicleControl()

    def update(self):
        # Calculate throttle to reach target velocity
        current_velocity = CarlaDataProvider.get_velocity(self._actor)

        if current_velocity < self._target_velocity:
            self._control.throttle = 0.8
        else:
            self._control.throttle = 0.0

        self._actor.apply_control(self._control)

        # Check termination conditions
        if GameTime.get_time() - self._start_time > self._duration:
            return py_trees.common.Status.SUCCESS
        if self._driven_distance > self._distance:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING
```

---

#### 3.2 Atomic Criteria (`atomic_criteria.py`)

**Location:** `scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py`
**Lines:** ~1200
**Classes:** 15+

Criteria **evaluate agent performance** and generate TrafficEvents for scoring.

##### Available Criteria:

| Criterion | What it Detects | Penalty (Driving Score) |
|-----------|-----------------|------------------------|
| `CollisionTest` | Any collision | 0.50 (pedestrian), 0.60 (vehicle), 0.65 (static) |
| `DrivenDistanceTest` | Distance driven | - |
| `AverageVelocityTest` | Speed below minimum | - |
| `KeepLaneTest` | Lane departure | 0.90 |
| `ReachedRegionTest` | Reached target area | - |
| `OnSidewalkTest` | Driving on sidewalk | 0.80 |
| `WrongLaneTest` | Wrong lane driving | 0.70 |
| `InRadiusRegionTest` | Within radius of point | - |
| `InRouteTest` | Following route | 0.80 |
| `RouteCompletionTest` | Route progress % | - |
| `OutsideRouteLanesTest` | Driving off-route | 0.90 |
| `RunningRedLightTest` | Red light violation | 0.70 |
| `RunningStopTest` | Stop sign violation | 0.80 |
| `ActorSpeedAboveThresholdTest` | Minimum speed | - |
| `MinimumSpeedRouteTest` | Min speed on route | - |

##### Example - CollisionTest Implementation:

```python
class CollisionTest(Criterion):
    """
    Register collisions and assign penalties based on collision type.

    Uses CARLA's collision sensor attached to ego vehicle.
    """

    COLLISION_PENALTIES = {
        'pedestrian': 0.50,   # Worst penalty - hitting pedestrians
        'vehicle': 0.60,      # Hitting other vehicles
        'static': 0.65,       # Hitting static objects
    }

    def __init__(self, actor, terminate_on_failure=False, name="CollisionTest"):
        self._collision_sensor = None
        # Spawn collision sensor attached to ego
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=actor)
        self._collision_sensor.listen(self._on_collision)

    def _on_collision(self, event):
        """Callback when collision detected"""
        actor_type = self._classify_actor(event.other_actor)

        # Create traffic event for scoring
        collision_event = TrafficEvent(
            event_type=TrafficEventType.COLLISION,
            message=f"Collision with {actor_type}",
            dictionary={'type': actor_type}
        )
        self.list_traffic_events.append(collision_event)

    def update(self):
        # Criteria run continuously in parallel
        return py_trees.common.Status.RUNNING
```

---

#### 3.3 Atomic Trigger Conditions (`atomic_trigger_conditions.py`)

**Location:** `scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py`
**Lines:** ~1350
**Classes:** 25+

Trigger conditions **wait for specific states** before allowing behavior to proceed.

##### Distance-Based Triggers:

| Trigger | Condition for SUCCESS |
|---------|----------------------|
| `InTriggerDistanceToVehicle` | Actor within distance of another actor |
| `InTriggerDistanceToLocation` | Actor within distance of fixed location |
| `InTriggerDistanceToNextIntersection` | Actor within distance of intersection |
| `InTriggerDistanceToLocationAlongRoute` | Distance measured along route |
| `InTriggerRegion` | Actor inside rectangular region |

##### Time-Based Triggers:

| Trigger | Condition for SUCCESS |
|---------|----------------------|
| `InTimeToArrivalToLocation` | Actor can reach location within time |
| `InTimeToArrivalToVehicle` | Actor can catch up to another actor |
| `StandStill` | Actor velocity = 0 for duration |

##### Velocity/State Triggers:

| Trigger | Condition for SUCCESS |
|---------|----------------------|
| `TriggerVelocity` | Actor velocity meets threshold |
| `TriggerAcceleration` | Actor acceleration meets threshold |
| `RelativeVelocityToOtherActor` | Relative speed between actors |
| `WaitForTrafficLightState` | Traffic light in specific state |
| `DriveDistance` | Actor has driven distance |
| `WaitEndIntersection` | Actor exits junction |
| `WaitUntilInFront` | Actor passes another actor |

##### Example - InTriggerDistanceToVehicle:

```python
class InTriggerDistanceToVehicle(AtomicCondition):
    """
    Wait until actor is within trigger distance of reference actor.

    Returns RUNNING until distance condition met.
    Returns SUCCESS when close enough.
    """

    def __init__(self, reference_actor, actor, distance,
                 comparison_operator=operator.lt,  # Less than
                 distance_type="cartesianDistance",
                 freespace=False,
                 name="TriggerDistanceToVehicle"):
        self._reference_actor = reference_actor
        self._actor = actor
        self._distance = distance
        self._comparison_operator = comparison_operator

    def update(self):
        location = CarlaDataProvider.get_location(self._actor)
        reference_location = CarlaDataProvider.get_location(self._reference_actor)

        distance = location.distance(reference_location)

        if self._comparison_operator(distance, self._distance):
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING
```

---

### 4. Actor Controls (`actorcontrols/`)

Controllers that drive NPC actors during scenarios.

**Location:** `scenario_runner/srunner/scenariomanager/actorcontrols/`

#### Controller Hierarchy:

```
BasicControl (abstract base)
    │
    ├── SimpleVehicleControl      # Direct velocity control (bypasses CARLA physics)
    ├── NpcVehicleControl         # Uses CARLA's traffic manager
    ├── CarlaAutoPilotControl     # Uses CARLA's autopilot
    ├── PedestrianControl         # Walker AI controller
    ├── VehicleLongitudinalControl # PID-based longitudinal control
    └── ExternalControl           # External control interface
```

#### SimpleVehicleControl Features:

```python
class SimpleVehicleControl(BasicControl):
    """
    Controller that directly sets vehicle velocity, bypassing CARLA physics.

    Advantages:
    - Precise speed control
    - Deterministic behavior for test scenarios

    Limitations:
    - Unrealistic cornering
    - Ignores traffic regulations by default

    Optional features (enabled via args):
    - consider_obstacles: Use obstacle sensor to avoid collisions
    - consider_trafficlights: Respect red lights
    - proximity_threshold: Distance for obstacle detection
    - max_deceleration: Realistic braking
    - max_acceleration: Realistic acceleration
    """

    def run_step(self, target_speed, waypoints):
        """
        Execute one control step.

        1. Calculate direction to next waypoint
        2. Set velocity vector directly
        3. Check for obstacles/traffic lights if enabled
        """
        # Get target waypoint
        if waypoints:
            target_wp = waypoints[0]
            direction = target_wp.location - self._actor.get_location()
            direction = direction.make_unit_vector()

        # Set velocity directly (bypasses physics engine)
        velocity = carla.Vector3D(
            x=direction.x * target_speed,
            y=direction.y * target_speed,
            z=0
        )
        self._actor.set_target_velocity(velocity)
```

---

### 5. Traffic Events (`traffic_events.py`)

Defines infraction types for scoring.

**Location:** `scenario_runner/srunner/scenariomanager/traffic_events.py`

```python
class TrafficEventType(Enum):
    """
    All possible infraction types tracked during evaluation.
    """
    COLLISION_STATIC = 'collision_static'
    COLLISION_VEHICLE = 'collision_vehicle'
    COLLISION_PEDESTRIAN = 'collision_pedestrian'
    ROUTE_DEVIATION = 'route_deviation'
    ROUTE_COMPLETION = 'route_completion'
    TRAFFIC_LIGHT_INFRACTION = 'traffic_light_infraction'
    WRONG_WAY_INFRACTION = 'wrong_way_infraction'
    ON_SIDEWALK_INFRACTION = 'on_sidewalk_infraction'
    STOP_INFRACTION = 'stop_infraction'
    OUTSIDE_LANE_INFRACTION = 'outside_lane_infraction'
    OUTSIDE_ROUTE_LANES_INFRACTION = 'outside_route_lanes_infraction'
    VEHICLE_BLOCKED = 'vehicle_blocked'
    MIN_SPEED_INFRACTION = 'min_speed_infraction'

class TrafficEvent:
    """
    A single traffic event/infraction.

    Attributes:
        event_type: TrafficEventType enum
        message: Human-readable description
        dictionary: Additional metadata
        frame: Simulation frame when event occurred
    """
    def __init__(self, event_type, message="", dictionary=None, frame=-1):
        self.event_type = event_type
        self.message = message
        self.dictionary = dictionary or {}
        self.frame = frame
```

---

### 6. Timer and Watchdog

#### GameTime (`timer.py`)

Tracks simulation time (not wall-clock time).

```python
class GameTime:
    """
    Singleton tracking CARLA simulation time.
    """
    _current_game_time = 0.0      # Current simulation time
    _carla_time = 0.0             # Raw CARLA timestamp
    _last_frame = 0               # Last processed frame
    _platform_timestamp = 0.0     # Wall-clock reference

    @staticmethod
    def get_time():
        """Get current game time in seconds."""
        return GameTime._current_game_time

    @staticmethod
    def on_carla_tick(timestamp):
        """
        Called every CARLA tick to update time.

        Args:
            timestamp: carla.Timestamp with frame, elapsed_seconds, etc.
        """
        GameTime._current_game_time = timestamp.elapsed_seconds
        GameTime._last_frame = timestamp.frame
```

#### Watchdog (`watchdog.py`)

Detects simulation hangs/timeouts.

```python
class Watchdog:
    """
    Timeout watchdog that triggers callback if not reset.

    Used to detect when CARLA simulation hangs.
    """

    def __init__(self, timeout, callback):
        self._timeout = timeout      # Seconds before timeout
        self._callback = callback    # Function to call on timeout
        self._timer = None

    def start(self):
        """Start the watchdog timer."""
        self._timer = threading.Timer(self._timeout, self._callback)
        self._timer.start()

    def update(self):
        """Reset the timer (call every tick)."""
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self._timeout, self._callback)
        self._timer.start()

    def stop(self):
        """Stop the watchdog."""
        if self._timer:
            self._timer.cancel()
```

---

## Scenario Implementation Guide

### Complete Example: Cut-In Scenario

**Location:** `scenario_runner/srunner/scenarios/cut_in.py`

This scenario simulates a vehicle cutting in front of the ego on a highway.

```python
class CutIn(BasicScenario):
    """
    Highway cut-in scenario.

    Setup:
    1. NPC vehicle driving in adjacent lane
    2. NPC accelerates to catch up with ego
    3. NPC changes lane in front of ego
    4. Ego must brake to avoid collision
    """

    timeout = 1200  # 20 minutes max

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=600):

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(
            config.trigger_points[0].location
        )

        # Scenario parameters
        self._velocity = 40           # NPC initial speed (m/s)
        self._delta_velocity = 10     # Speed difference to catch up
        self._trigger_distance = 30   # Distance to trigger cut-in

        # Get direction from config name (e.g., "CutInFromLeft")
        self._direction = None
        self._config = config

        super().__init__("CutIn", ego_vehicles, config, world,
                        debug_mode, criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """Spawn the cut-in vehicle."""

        # Determine cut-in direction from scenario name
        if 'LEFT' in self._config.name.upper():
            self._direction = 'left'
        elif 'RIGHT' in self._config.name.upper():
            self._direction = 'right'

        # Spawn vehicles from XML config
        for actor in config.other_actors:
            vehicle = CarlaDataProvider.request_new_actor(
                actor.model,
                actor.transform
            )
            self.other_actors.append(vehicle)
            vehicle.set_simulate_physics(enabled=False)  # Teleport mode

    def _create_behavior(self):
        """
        Build behavior tree for cut-in sequence.

        Tree structure:
        Sequence("CutIn")
        ├── ActorTransformSetter     # Position NPC vehicle
        ├── Parallel("DrivingStraight")
        │   ├── WaypointFollower     # NPC drives straight
        │   └── InTriggerDistanceToVehicle  # Wait for ego approach
        ├── AccelerateToCatchUp      # NPC speeds up
        └── LaneChange               # NPC cuts in front
        """

        # Create sequence
        behaviour = py_trees.composites.Sequence(
            f"CarOn_{self._direction}_Lane"
        )

        # Step 1: Position the NPC vehicle
        car_visible = ActorTransformSetter(
            self.other_actors[0],
            self._transform_visible
        )
        behaviour.add_child(car_visible)

        # Step 2: Drive straight until ego approaches
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        # NPC follows waypoints
        car_driving = WaypointFollower(
            self.other_actors[0],
            self._velocity
        )
        just_drive.add_child(car_driving)

        # Trigger: ego within 30m
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0],
            self.ego_vehicles[0],
            self._trigger_distance
        )
        just_drive.add_child(trigger_distance)

        behaviour.add_child(just_drive)

        # Step 3: Accelerate to catch up
        accelerate = AccelerateToCatchUp(
            self.other_actors[0],
            self.ego_vehicles[0],
            throttle_value=1,
            delta_velocity=self._delta_velocity,
            trigger_distance=5,
            max_distance=500
        )
        behaviour.add_child(accelerate)

        # Step 4: Lane change (opposite of starting lane)
        change_direction = 'right' if self._direction == 'left' else 'left'
        lane_change = LaneChange(
            self.other_actors[0],
            speed=None,
            direction=change_direction,
            distance_same_lane=5,
            distance_other_lane=300
        )
        behaviour.add_child(lane_change)

        # End condition
        endcondition = DriveDistance(self.other_actors[0], 200)

        root = py_trees.composites.Sequence("Behavior")
        root.add_child(behaviour)
        root.add_child(endcondition)

        return root

    def _create_test_criteria(self):
        """Only collision test for this scenario."""
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """Cleanup spawned actors."""
        self.remove_all_actors()
```

---

## py_trees Behavior Tree Primer

Scenarios use the [py_trees](https://github.com/splintered-reality/py_trees) library for behavior composition.

### Key Composites:

```python
import py_trees

# SEQUENCE: Children execute in order, stops on first FAILURE
sequence = py_trees.composites.Sequence("MySequence")
sequence.add_child(behavior1)
sequence.add_child(behavior2)  # Only runs if behavior1 succeeds

# PARALLEL: Children execute simultaneously
parallel = py_trees.composites.Parallel(
    "MyParallel",
    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE  # or SUCCESS_ON_ALL
)
parallel.add_child(behavior1)  # Both run at same time
parallel.add_child(behavior2)

# SELECTOR: First child to succeed wins
selector = py_trees.composites.Selector("MySelector")
selector.add_child(behavior1)
selector.add_child(behavior2)  # Only runs if behavior1 fails
```

### Status Values:

```python
py_trees.common.Status.RUNNING   # Still executing
py_trees.common.Status.SUCCESS   # Completed successfully
py_trees.common.Status.FAILURE   # Failed
py_trees.common.Status.INVALID   # Not yet ticked
```

### Common Pattern - Trigger then Action:

```python
# Wait for trigger, then execute action
trigger_and_action = py_trees.composites.Sequence("TriggerAction")

# Trigger: wait for ego to approach
trigger = InTriggerDistanceToVehicle(
    reference_actor=self.ego_vehicles[0],
    actor=self.other_actors[0],
    distance=30
)
trigger_and_action.add_child(trigger)

# Action: NPC performs lane change
action = LaneChange(self.other_actors[0], direction='left')
trigger_and_action.add_child(action)
```

---

## Available Scenarios (40+)

### By Category:

**Intersection Scenarios:**
- `SignalizedJunctionLeftTurn` - Left turn at traffic light
- `SignalizedJunctionRightTurn` - Right turn at traffic light
- `NoSignalJunctionCrossing` - Crossing unsignalized intersection
- `OppositeVehicleRunningRedLight` - Oncoming car runs red light

**Highway Scenarios:**
- `CutIn` - Vehicle cuts in front
- `HighwayExit` - Take highway exit
- `HighwayCutIn` - Highway cut-in with high speed

**Pedestrian Scenarios:**
- `PedestrianCrossing` - Pedestrians cross at crosswalk
- `DynamicObjectCrossing` - Sudden pedestrian appearance
- `Walker` - Single pedestrian interaction

**Vehicle Interaction:**
- `VehicleTurnLeft/Right` - Vehicle turns in front
- `FollowLeadingVehicle` - Follow vehicle ahead
- `LeadingVehicleSlowsDown` - Lead vehicle brakes
- `ControlLoss` - Loss of vehicle control

**Obstacle Scenarios:**
- `ObjectCrashVehicle` - Static obstacle crash
- `ConstructionObstacle` - Construction zone
- `ParkedVehicle` - Parked car obstacle
- `StaticCutIn` - Parked car in lane

**Emergency:**
- `EmergencyBrake` - Emergency braking required
- `HardBreak` - Sudden hard brake

---

## How Scenarios Are Loaded and Run

### 1. Route XML Definition

Routes define sequences of scenarios along a path:

```xml
<!-- From bench2drive220.xml -->
<route id="0" map="Town01">
    <waypoint x="100.0" y="200.0" z="0.0"/>
    <waypoint x="150.0" y="200.0" z="0.0"/>

    <scenario name="PedestrianCrossing" type="Pedestrian">
        <trigger_point x="120.0" y="200.0" z="0.0"/>
    </scenario>
</route>
```

### 2. Scenario Loading in Leaderboard

```python
# In leaderboard_evaluator.py
from srunner.scenarios.pedestrian_crossing import PedestrianCrossing

# ScenarioConfiguration created from XML
config = ScenarioConfiguration()
config.trigger_points = [carla.Transform(...)]
config.other_actors = [ActorConfiguration(...)]

# Scenario instantiated
scenario = PedestrianCrossing(
    world=self.world,
    ego_vehicles=[self.ego_vehicle],
    config=config,
    criteria_enable=True
)
```

### 3. Execution Loop

```python
# Simplified from scenario_manager.py
while scenario.running:
    # 1. Tick CARLA world
    world.tick()

    # 2. Update time
    GameTime.on_carla_tick(timestamp)
    CarlaDataProvider.on_carla_tick()

    # 3. Tick behavior tree
    scenario.scenario_tree.tick_once()

    # 4. Check criteria
    for criterion in scenario.criteria_tree.children:
        criterion.update()

    # 5. Check termination
    if scenario.scenario_tree.status == SUCCESS:
        break
```

---

## Debugging Scenarios

### 1. Enable Debug Mode

```python
scenario = MyScenario(
    world, ego_vehicles, config,
    debug_mode=True  # Enables visual debugging
)
```

### 2. Print Behavior Tree Status

```python
# Print tree structure
py_trees.display.print_ascii_tree(scenario.scenario_tree)

# Print current status
py_trees.display.print_ascii_tree(
    scenario.scenario_tree,
    show_status=True
)
```

### 3. Visualize Actor Positions

```python
# In scenario behavior
def update(self):
    # Draw debug point at actor location
    world = CarlaDataProvider.get_world()
    loc = CarlaDataProvider.get_location(self._actor)
    world.debug.draw_point(loc, size=0.1, color=carla.Color(255,0,0))
```

### 4. Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Scenario never triggers | Ego too far from trigger point | Check trigger_point coordinates |
| NPC doesn't move | Physics disabled | Call `actor.set_simulate_physics(True)` |
| NPC clips through objects | Using direct velocity control | Use `NpcVehicleControl` instead |
| Criteria not recording | Criteria tree not ticked | Ensure `criteria_enable=True` |

---

## Creating Custom Scenarios

### Step-by-Step Template:

```python
#!/usr/bin/env python
"""
Custom scenario: VehicleOvertake
An NPC vehicle overtakes the ego from behind.
"""

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
    WaypointFollower,
    LaneChange,
    ActorDestroy
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToVehicle,
    DriveDistance
)
from srunner.scenarios.basic_scenario import BasicScenario


class VehicleOvertake(BasicScenario):
    """
    NPC vehicle approaches from behind and overtakes ego.

    Scenario flow:
    1. NPC spawns behind ego
    2. NPC accelerates toward ego
    3. NPC changes to left lane
    4. NPC passes ego
    5. NPC changes back to right lane
    """

    timeout = 120

    def __init__(self, world, ego_vehicles, config,
                 debug_mode=False, criteria_enable=True):

        self._map = CarlaDataProvider.get_map()
        self._npc_speed = 15  # m/s (faster than typical ego)

        super().__init__(
            "VehicleOvertake",
            ego_vehicles, config, world,
            debug_mode, criteria_enable=criteria_enable
        )

    def _initialize_actors(self, config):
        """Spawn NPC vehicle behind ego."""

        # Get spawn point behind ego
        ego_location = self.ego_vehicles[0].get_location()
        ego_waypoint = self._map.get_waypoint(ego_location)

        # 50m behind ego
        spawn_waypoint = ego_waypoint.previous(50)[0]
        spawn_transform = spawn_waypoint.transform
        spawn_transform.location.z += 0.5

        # Spawn vehicle
        vehicle = CarlaDataProvider.request_new_actor(
            'vehicle.audi.tt',
            spawn_transform
        )
        self.other_actors.append(vehicle)

    def _create_behavior(self):
        """Build overtake behavior tree."""

        root = py_trees.composites.Sequence("VehicleOvertake")

        # Phase 1: Approach ego from behind
        approach = py_trees.composites.Parallel(
            "Approach",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        approach.add_child(WaypointFollower(
            self.other_actors[0],
            self._npc_speed
        ))
        approach.add_child(InTriggerDistanceToVehicle(
            self.other_actors[0],
            self.ego_vehicles[0],
            distance=15
        ))
        root.add_child(approach)

        # Phase 2: Change to left lane
        root.add_child(LaneChange(
            self.other_actors[0],
            direction='left',
            distance_same_lane=3,
            distance_other_lane=30
        ))

        # Phase 3: Pass ego
        pass_ego = py_trees.composites.Parallel(
            "PassEgo",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        pass_ego.add_child(WaypointFollower(
            self.other_actors[0],
            self._npc_speed
        ))
        pass_ego.add_child(DriveDistance(
            self.other_actors[0],
            distance=40
        ))
        root.add_child(pass_ego)

        # Phase 4: Return to right lane
        root.add_child(LaneChange(
            self.other_actors[0],
            direction='right',
            distance_same_lane=3,
            distance_other_lane=20
        ))

        # Cleanup
        root.add_child(ActorDestroy(self.other_actors[0]))

        return root

    def _create_test_criteria(self):
        """Collision is the only criterion."""
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        self.remove_all_actors()
```

### Register Custom Scenario:

Add to `scenario_runner/srunner/scenarios/__init__.py`:
```python
from srunner.scenarios.vehicle_overtake import VehicleOvertake
```

---

## Performance Considerations

### 1. CarlaDataProvider Caching

Locations/velocities are cached per-tick. Multiple calls within same tick are fast:

```python
# Fast - uses cache
loc1 = CarlaDataProvider.get_location(actor)
loc2 = CarlaDataProvider.get_location(actor)  # Cache hit

# Cache invalidated on next tick
world.tick()
loc3 = CarlaDataProvider.get_location(actor)  # Fresh query
```

### 2. Sensor Overhead

Each criterion spawns sensors. Minimize by:
- Using route mode (shares sensors across scenarios)
- Only enabling necessary criteria

### 3. Behavior Tree Efficiency

- Avoid deep nesting (increases tick overhead)
- Use `SUCCESS_ON_ONE` for parallel early exit
- Clean up finished behaviors with `ActorDestroy`

---

## Integration Points

### With Leaderboard:

```python
# leaderboard_evaluator.py calls:
scenario = ScenarioClass(world, ego_vehicles, config)
scenario.scenario_tree.tick_once()  # Every frame
scenario.get_running_status()       # Check if done
scenario.scenario_tree.status       # Get final status
```

### With Agent:

```python
# Agent receives sensor data, scenario controls NPCs
# No direct communication - interaction through physics

class MyAgent(AutonomousAgent):
    def run_step(self, input_data, timestamp):
        # React to scenario-controlled NPCs via sensors
        camera = input_data['front_camera']
        # Scenario's NPC vehicle visible in camera
        return control
```

---

## Summary

The `scenario_runner` module is the **behavior engine** of Bench2Drive:

1. **CarlaDataProvider** - Central access point for all CARLA world data
2. **BasicScenario** - Template for scenario implementations
3. **ScenarioAtomics** - 80+ reusable behavior/criteria/trigger primitives
4. **ActorControls** - Controllers for NPC movement
5. **py_trees** - Behavior tree library for composing complex scenarios

Understanding this module is essential for:
- Debugging evaluation failures
- Creating custom scenarios
- Understanding how agents are challenged
- Extending the benchmark

---

*Document generated from analysis of Bench2Drive scenario_runner module*
*Last updated: December 2024*
