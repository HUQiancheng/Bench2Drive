# 🎓 Bench2Drive 系统性学习路径

> **目标**: 从零开始理解CARLA闭环测试，掌握Bench2Drive评估框架
> **预计时间**: 3-5天深度学习

---

## 📋 学习路线总览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  阶段1: CARLA 基础                                                           │
│  ├── 核心概念 (World, Actor, Sensor)                                         │
│  ├── Python API                                                              │
│  └── 同步模式与数据流                                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│  阶段2: 闭环测试原理                                                          │
│  ├── 开环 vs 闭环对比                                                        │
│  ├── 传感器数据获取                                                           │
│  └── 实时控制回路                                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  阶段3: Scenario Runner                                                      │
│  ├── 场景定义与执行                                                           │
│  ├── py_trees 行为树                                                         │
│  └── 评估标准 (Criteria)                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  阶段4: Bench2Drive 框架                                                     │
│  ├── Leaderboard 评估器                                                      │
│  ├── Agent 接口                                                              │
│  └── 评分系统                                                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  阶段5: Bench2DriveZoo 实践                                                  │
│  ├── TCP 模型实现                                                            │
│  ├── 模型集成流程                                                            │
│  └── 完整评估实践                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔵 阶段1: CARLA 基础 (建议1天)

### 1.1 核心概念

**官方文档 (必读)**:

| 顺序 | 文档 | 链接 | 重点 |
|------|------|------|------|
| 1 | Core Concepts | [carla.readthedocs.io/en/0.9.15/core_concepts](https://carla.readthedocs.io/en/0.9.15/core_concepts/) | World, Actor, Blueprint, Map |
| 2 | 1st. World and client | [carla.readthedocs.io/en/0.9.15/core_world](https://carla.readthedocs.io/en/0.9.15/core_world/) | 连接CARLA，获取World |
| 3 | 2nd. Actors and blueprints | [carla.readthedocs.io/en/0.9.15/core_actors](https://carla.readthedocs.io/en/0.9.15/core_actors/) | 车辆/行人生成与控制 |
| 4 | 3rd. Maps and navigation | [carla.readthedocs.io/en/0.9.15/core_map](https://carla.readthedocs.io/en/0.9.15/core_map/) | 地图、路点、车道 |
| 5 | 4th. Sensors and data | [carla.readthedocs.io/en/0.9.15/core_sensors](https://carla.readthedocs.io/en/0.9.15/core_sensors/) | 传感器原理与数据流 |

**需要理解的关键概念**:

| 概念 | 说明 | 为什么重要 |
|------|------|------------|
| **World** | CARLA模拟世界的容器 | 所有操作都通过World进行 |
| **Actor** | 场景中的实体(车辆、行人、传感器) | 你的Agent就是一个Actor |
| **Blueprint** | Actor的模板/配置 | 定义传感器参数 |
| **Transform** | 位置+旋转 | 控制物体在3D空间的位姿 |
| **Tick** | 模拟时间步 | 闭环测试的核心节拍(20Hz) |

### 1.2 传感器系统 (重点!)

**官方文档 (必读)**:

| 文档 | 链接 | 内容 |
|------|------|------|
| Sensors reference | [carla.readthedocs.io/en/0.9.15/ref_sensors](https://carla.readthedocs.io/en/0.9.15/ref_sensors/) | 所有传感器详细参数 |
| Camera RGB | [ref_sensors/#rgb-camera](https://carla.readthedocs.io/en/0.9.15/ref_sensors/#rgb-camera) | RGB相机配置 |
| Depth camera | [ref_sensors/#depth-camera](https://carla.readthedocs.io/en/0.9.15/ref_sensors/#depth-camera) | 深度相机 |
| LiDAR | [ref_sensors/#lidar-sensor](https://carla.readthedocs.io/en/0.9.15/ref_sensors/#lidar-raycast-sensor) | 激光雷达 |
| GNSS | [ref_sensors/#gnss-sensor](https://carla.readthedocs.io/en/0.9.15/ref_sensors/#gnss-sensor) | GPS定位 |
| IMU | [ref_sensors/#imu-sensor](https://carla.readthedocs.io/en/0.9.15/ref_sensors/#imu-sensor) | 惯性测量 |

**Bench2Drive 常用传感器**:

| 传感器类型 | 用途 | 输出格式 |
|-----------|------|----------|
| `sensor.camera.rgb` | 可见光相机 | BGRA uint8 array |
| `sensor.camera.depth` | 深度相机 | 深度图 (logarithmic) |
| `sensor.camera.semantic_segmentation` | 语义分割 | 分类mask |
| `sensor.lidar.ray_cast` | 激光雷达 | 点云 (x,y,z,intensity) |
| `sensor.other.gnss` | GPS | 经纬度高度 |
| `sensor.other.imu` | 惯性测量 | 加速度/角速度 |

### 1.3 同步模式 (关键!)

**官方文档 (必读)**:

| 文档 | 链接 |
|------|------|
| Synchrony and time-step | [carla.readthedocs.io/en/0.9.15/adv_synchrony_timestep](https://carla.readthedocs.io/en/0.9.15/adv_synchrony_timestep/) |

**为什么重要**: Bench2Drive使用同步模式确保评估一致性

```python
# 同步模式设置
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 Hz
world.apply_settings(settings)

# 主循环 - 每次tick推进模拟
while running:
    world.tick()  # 推进模拟一步
    # 此时所有传感器数据已更新
    sensor_data = get_sensor_data()
    control = model.predict(sensor_data)
    vehicle.apply_control(control)
```

### 1.4 Python API 参考

**官方文档 (查阅用)**:

| 文档 | 链接 | 用途 |
|------|------|------|
| Python API reference | [carla.readthedocs.io/en/0.9.15/python_api](https://carla.readthedocs.io/en/0.9.15/python_api/) | 完整API |
| carla.Client | [python_api/#carlaclient](https://carla.readthedocs.io/en/0.9.15/python_api/#carlaclient) | 连接服务器 |
| carla.World | [python_api/#carlaworld](https://carla.readthedocs.io/en/0.9.15/python_api/#carlaworld) | 世界操作 |
| carla.Vehicle | [python_api/#carlavehicle](https://carla.readthedocs.io/en/0.9.15/python_api/#carlavehicle) | 车辆Actor |
| carla.VehicleControl | [python_api/#carlavehiclecontrol](https://carla.readthedocs.io/en/0.9.15/python_api/#carlavehiclecontrol) | 控制命令 |
| carla.Transform | [python_api/#carlatransform](https://carla.readthedocs.io/en/0.9.15/python_api/#carlatransform) | 位姿 |

**最常用的类**:
```python
carla.Client          # 连接CARLA服务器
carla.World           # 模拟世界
carla.Vehicle         # 车辆Actor
carla.VehicleControl  # 车辆控制命令 (throttle, steer, brake)
carla.Transform       # 位姿
carla.Location        # 位置
carla.Rotation        # 旋转
```

### 1.5 CARLA 教程 (可选)

| 教程 | 链接 | 说明 |
|------|------|------|
| Getting started | [carla.readthedocs.io/en/0.9.15/start_quickstart](https://carla.readthedocs.io/en/0.9.15/start_quickstart/) | 安装与第一次运行 |
| First steps | [carla.readthedocs.io/en/0.9.15/tuto_first_steps](https://carla.readthedocs.io/en/0.9.15/tuto_first_steps/) | 基本操作教程 |
| Retrieve simulation data | [carla.readthedocs.io/en/0.9.15/tuto_G_retrieve_data](https://carla.readthedocs.io/en/0.9.15/tuto_G_retrieve_data/) | 获取传感器数据 |

---

## 🟢 阶段2: 闭环测试原理 (建议0.5天)

### 2.1 开环 vs 闭环对比

**这是理解Bench2Drive的核心!**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        开环评估 (Open-Loop)                         │
│                                                                     │
│   预录数据 ──► 模型 ──► 预测轨迹 ──► 与Ground Truth比较              │
│                                                                     │
│   问题: 模型的错误不会影响后续输入，无法评估真实驾驶能力             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        闭环评估 (Closed-Loop)                       │
│                           ┌──────────┐                              │
│   ┌──────┐    感知数据    │          │    控制命令    ┌──────────┐  │
│   │CARLA ├───────────────►│  Agent   ├───────────────►│ 车辆执行 │  │
│   │      │◄───────────────┤  (模型)  │◄───────────────┤          │  │
│   └──────┘    世界更新    │          │    状态反馈    └──────────┘  │
│                           └──────────┘                              │
│                                                                     │
│   关键: 模型的每个决策都会改变世界状态，影响后续输入                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 推荐阅读

| 资源 | 链接 | 说明 |
|------|------|------|
| Bench2Drive 论文 | [arxiv.org/abs/2406.03877](https://arxiv.org/abs/2406.03877) | Section 3: Benchmark Design |
| What is Bench2Drive | [.llm/docs/what_is_bench2drive.md](./../.llm/docs/what_is_bench2drive.md) | 本地概览文档 |
| 闭环评估重要性 | [CARLA Garage: Common Mistakes](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/common_mistakes_in_benchmarking_ad.md) | 为什么开环评估有问题 |

### 2.3 闭环测试的核心循环

```python
# 每个时间步 (20 Hz = 50ms/步)
while not route_finished:
    # 1. CARLA 推进物理模拟
    world.tick()

    # 2. 获取传感器数据 (相机、LiDAR等)
    input_data = sensor_interface.get_data()

    # 3. Agent 处理数据，生成控制命令
    control = agent.run_step(input_data, timestamp)

    # 4. 执行控制命令
    ego_vehicle.apply_control(control)

    # 5. 评估: 检查碰撞、闯红灯等
    criteria.update()
```

### 2.4 为什么闭环测试更难?

| 挑战 | 说明 |
|------|------|
| **误差累积** | 一个小错误会导致车辆偏离，后续感知完全不同 |
| **实时性** | 必须在50ms内完成推理，否则模拟不同步 |
| **分布偏移** | 训练数据(专家轨迹)与测试时(自己驾驶)分布不同 |
| **长尾场景** | 必须处理各种罕见情况(行人突然冲出等) |

---

## 🟡 阶段3: Scenario Runner (建议1天)

### 3.1 什么是 Scenario Runner?

Scenario Runner 是 CARLA 官方的场景执行框架，Bench2Drive 基于它构建。

**本项目文档 (按顺序阅读)**:

| 顺序 | 文档 | 路径 | 内容 |
|------|------|------|------|
| 1 | Getting ScenarioRunner | [scenario_runner/Docs/getting_scenariorunner.md](../scenario_runner/Docs/getting_scenariorunner.md) | 安装与配置 |
| 2 | Getting Started | [scenario_runner/Docs/getting_started.md](../scenario_runner/Docs/getting_started.md) | 运行第一个场景 |
| 3 | Creating New Scenarios | [scenario_runner/Docs/creating_new_scenario.md](../scenario_runner/Docs/creating_new_scenario.md) | 理解场景结构 |
| 4 | Agent Evaluation | [scenario_runner/Docs/agent_evaluation.md](../scenario_runner/Docs/agent_evaluation.md) | Agent评估机制 |
| 5 | List of Scenarios | [scenario_runner/Docs/list_of_scenarios.md](../scenario_runner/Docs/list_of_scenarios.md) | 所有可用场景 |
| 6 | Metrics Module | [scenario_runner/Docs/metrics_module.md](../scenario_runner/Docs/metrics_module.md) | 评估指标 |
| 7 | FAQ | [scenario_runner/Docs/FAQ.md](../scenario_runner/Docs/FAQ.md) | 常见问题 |

### 3.2 核心组件文件

```
scenario_runner/srunner/
├── scenariomanager/
│   ├── carla_data_provider.py   ← 关键! CARLA交互的单例
│   ├── timer.py                 ← 模拟时间管理
│   ├── scenario_manager.py      ← 场景执行主循环
│   ├── traffic_events.py        ← 违规事件定义
│   └── scenarioatomics/
│       ├── atomic_behaviors.py      ← 行为原语 (驾驶动作)
│       ├── atomic_criteria.py       ← 评估标准 (碰撞、闯红灯)
│       └── atomic_trigger_conditions.py ← 触发条件
└── scenarios/
    ├── basic_scenario.py        ← 场景基类
    └── ... (40+ 具体场景)
```

### 3.3 CarlaDataProvider - 最重要的类

**文件**: [scenario_runner/srunner/scenariomanager/carla_data_provider.py](../scenario_runner/srunner/scenariomanager/carla_data_provider.py)

| 方法 | 行号 | 功能 |
|------|------|------|
| `class CarlaDataProvider` | [L34](../scenario_runner/srunner/scenariomanager/carla_data_provider.py#L34) | 类定义 |
| `get_velocity(actor)` | [L149](../scenario_runner/srunner/scenariomanager/carla_data_provider.py#L149) | 获取速度 (m/s) |
| `get_location(actor)` | [L163](../scenario_runner/srunner/scenariomanager/carla_data_provider.py#L163) | 获取位置 |
| `get_world()` | [L218](../scenario_runner/srunner/scenariomanager/carla_data_provider.py#L218) | 获取World对象 |
| `get_map()` | [L225](../scenario_runner/srunner/scenariomanager/carla_data_provider.py#L225) | 获取Map对象 |
| `request_new_actor()` | [L585](../scenario_runner/srunner/scenariomanager/carla_data_provider.py#L585) | 生成新Actor |

```python
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# 获取世界/地图
world = CarlaDataProvider.get_world()
map = CarlaDataProvider.get_map()

# 获取Actor信息 (有缓存，高效)
location = CarlaDataProvider.get_location(actor)
velocity = CarlaDataProvider.get_velocity(actor)  # m/s

# 生成Actor
actor = CarlaDataProvider.request_new_actor(model, spawn_point)
```

### 3.4 违规事件类型

**文件**: [scenario_runner/srunner/scenariomanager/traffic_events.py](../scenario_runner/srunner/scenariomanager/traffic_events.py)

| 事件类型 | 行号 | 说明 |
|----------|------|------|
| `class TrafficEventType` | [L13](../scenario_runner/srunner/scenariomanager/traffic_events.py#L13) | 枚举定义 |
| `COLLISION_STATIC` | [L20](../scenario_runner/srunner/scenariomanager/traffic_events.py#L20) | 撞静物 |
| `COLLISION_VEHICLE` | [L21](../scenario_runner/srunner/scenariomanager/traffic_events.py#L21) | 撞车辆 |
| `COLLISION_PEDESTRIAN` | [L22](../scenario_runner/srunner/scenariomanager/traffic_events.py#L22) | 撞行人 |
| `TRAFFIC_LIGHT_INFRACTION` | [L26](../scenario_runner/srunner/scenariomanager/traffic_events.py#L26) | 闯红灯 |
| `STOP_INFRACTION` | [L29](../scenario_runner/srunner/scenariomanager/traffic_events.py#L29) | 无视停止标志 |
| `VEHICLE_BLOCKED` | [L32](../scenario_runner/srunner/scenariomanager/traffic_events.py#L32) | 车辆阻塞 |
| `SCENARIO_TIMEOUT` | [L35](../scenario_runner/srunner/scenariomanager/traffic_events.py#L35) | 场景超时 |

### 3.5 原子行为 (Atomic Behaviors)

**文件**: [scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py)

| 行为类 | 行号 | 功能 |
|--------|------|------|
| `AtomicBehavior` (基类) | [L90](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py#L90) | 所有行为的基类 |
| `KeepVelocity` | [L1393](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py#L1393) | 保持恒定速度 |
| `WaypointFollower` | [L2226](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py#L2226) | 沿路点行驶 |
| `ActorTransformSetter` | [L2630](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py#L2630) | 瞬移Actor |
| `ActorDestroy` | [L2601](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py#L2601) | 销毁Actor |
| `StopVehicle` | [L1567](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py#L1567) | 停止车辆 |
| `ChangeAutoPilot` | [L1502](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py#L1502) | 切换自动驾驶 |

### 3.6 原子评估标准 (Atomic Criteria)

**文件**: [scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py)

| 标准类 | 行号 | 功能 |
|--------|------|------|
| `CollisionTest` | [L281](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py#L281) | 碰撞检测 |
| `RunningRedLightTest` | [L1620](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py#L1620) | 闯红灯检测 |

### 3.7 触发条件 (Trigger Conditions)

**文件**: [scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py)

| 条件类 | 行号 | 功能 |
|--------|------|------|
| `AtomicCondition` (基类) | [L41](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py#L41) | 条件基类 |
| `InTriggerDistanceToVehicle` | [L556](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py#L556) | 距离车辆触发 |
| `InTriggerDistanceToLocation` | [L616](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py#L616) | 距离位置触发 |
| `DriveDistance` | [L1112](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py#L1112) | 行驶距离触发 |
| `StandStill` | [L223](../scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py#L223) | 静止触发 |

### 3.8 py_trees 行为树

**外部文档**: [py-trees.readthedocs.io](https://py-trees.readthedocs.io/en/devel/)

Scenario Runner 使用 py_trees 库定义场景逻辑:

```python
import py_trees

# 顺序执行: 一个接一个
sequence = py_trees.composites.Sequence("MySequence")
sequence.add_child(wait_trigger)  # 先等待触发
sequence.add_child(do_action)     # 再执行动作

# 并行执行: 同时进行
parallel = py_trees.composites.Parallel("MyParallel")
parallel.add_child(npc_drive)     # NPC驾驶
parallel.add_child(monitor_ego)   # 同时监控主车
```

### 3.9 场景基类

**文件**: [scenario_runner/srunner/scenarios/basic_scenario.py#L28](../scenario_runner/srunner/scenarios/basic_scenario.py#L28)

---

## 🟠 阶段4: Bench2Drive 框架 (建议1天)

### 4.1 项目结构

```
Bench2Drive/
├── leaderboard/                 ← 评估框架主目录
│   ├── leaderboard/
│   │   ├── leaderboard_evaluator.py  ← 主控制器 ⭐
│   │   ├── autoagents/
│   │   │   └── autonomous_agent.py   ← Agent基类 ⭐
│   │   ├── scenarios/
│   │   │   ├── route_scenario.py     ← 路线场景
│   │   │   └── scenario_manager.py   ← 场景管理
│   │   └── utils/
│   │       └── statistics_manager.py ← 评分统计
│   ├── scripts/
│   │   ├── run_evaluation.sh         ← 单GPU评估
│   │   └── run_evaluation_multi_*.sh ← 多GPU评估
│   ├── data/
│   │   ├── bench2drive220.xml        ← 220条测试路线
│   │   └── drivetransformer_bench2drive_dev10.xml ← 10条开发路线
│   └── team_code/                    ← 放置你的Agent代码
├── scenario_runner/             ← 场景执行器 (阶段3已学)
├── tools/                       ← 工具脚本
│   ├── merge_route_json.py      ← 合并结果
│   ├── ability_benchmark.py     ← 能力评估
│   └── clean_carla.sh           ← 清理进程
└── docs/
    └── SERVER_DEPLOYMENT_GUIDE.md ← 部署指南 ⭐
```

### 4.2 核心文件阅读顺序

#### 第一步: 整体部署指南

| 文档 | 路径 | 内容 |
|------|------|------|
| 服务器部署指南 | [docs/SERVER_DEPLOYMENT_GUIDE.md](../docs/SERVER_DEPLOYMENT_GUIDE.md) | 完整部署流程 |
| 项目概览 | [CLAUDE.md](../CLAUDE.md) | 架构与关键模式 |
| README | [README.md](../README.md) | 快速参考 |

#### 第二步: 理解评估主流程

**文件**: [leaderboard/leaderboard/leaderboard_evaluator.py](../leaderboard/leaderboard/leaderboard_evaluator.py)

| 内容 | 行号 | 说明 |
|------|------|------|
| `class LeaderboardEvaluator` | [L80](../leaderboard/leaderboard/leaderboard_evaluator.py#L80) | 主控制器类 |
| `_setup_simulation()` | [L197](../leaderboard/leaderboard/leaderboard_evaluator.py#L197) | 启动CARLA服务器 |

重点函数:
```python
class LeaderboardEvaluator:
    def __init__(self):
        # 初始化CARLA连接
        self._setup_simulation()  # 看 L197: 如何启动CARLA

    def run(self, args):
        # 主循环: 遍历每条路线
        for route_config in route_list:
            self._load_world(route_config.town)
            self._prepare_ego_vehicle()
            self._run_route(route_config)  # 执行单条路线
            self._compute_statistics()     # 计算得分
```

#### 第三步: 理解Agent接口 (最重要!)

**文件**: [leaderboard/leaderboard/autoagents/autonomous_agent.py](../leaderboard/leaderboard/autoagents/autonomous_agent.py)

| 内容 | 行号 | 说明 |
|------|------|------|
| `class AutonomousAgent` | [L32](../leaderboard/leaderboard/autoagents/autonomous_agent.py#L32) | Agent基类定义 |
| `setup()` | [L51](../leaderboard/leaderboard/autoagents/autonomous_agent.py#L51) | 初始化方法 |
| `sensors()` | [L59](../leaderboard/leaderboard/autoagents/autonomous_agent.py#L59) | 定义传感器 |
| `run_step()` | [L81](../leaderboard/leaderboard/autoagents/autonomous_agent.py#L81) | 核心决策方法 (20Hz) |

这是你需要继承的基类:
```python
class AutonomousAgent:
    def setup(self, path_to_conf_file):       # L51
        """初始化: 加载模型权重"""
        pass

    def sensors(self):                         # L59
        """定义需要的传感器"""
        return [
            {'type': 'sensor.camera.rgb', 'id': 'CAM_FRONT', ...},
            {'type': 'sensor.lidar.ray_cast', 'id': 'LIDAR', ...},
        ]

    def run_step(self, input_data, timestamp): # L81
        """核心方法: 每帧调用一次 (20Hz)

        Args:
            input_data: dict, 传感器数据
            timestamp: 当前模拟时间

        Returns:
            carla.VehicleControl: 控制命令
        """
        # 1. 处理传感器数据
        camera_image = input_data['CAM_FRONT'][1]

        # 2. 模型推理
        prediction = self.model(camera_image)

        # 3. 生成控制命令
        return carla.VehicleControl(
            throttle=prediction.throttle,
            steer=prediction.steer,
            brake=prediction.brake
        )

    def destroy(self):
        """清理资源"""
        pass
```

#### 第四步: 参考实现

| Agent | 文件 | 说明 |
|-------|------|------|
| NPC Agent | [leaderboard/leaderboard/autoagents/npc_agent.py](../leaderboard/leaderboard/autoagents/npc_agent.py) | 使用CARLA内置AI的简单Agent |
| Dummy Agent | [leaderboard/leaderboard/autoagents/dummy_agent.py](../leaderboard/leaderboard/autoagents/dummy_agent.py) | 最简实现示例 |
| Human Agent | [leaderboard/leaderboard/autoagents/human_agent.py](../leaderboard/leaderboard/autoagents/human_agent.py) | 人工控制Agent |

**NPC Agent 关键行号**:
- `setup()`: [L30](../leaderboard/leaderboard/autoagents/npc_agent.py#L30)
- `sensors()`: [L38](../leaderboard/leaderboard/autoagents/npc_agent.py#L38)
- `run_step()`: [L63](../leaderboard/leaderboard/autoagents/npc_agent.py#L63)

#### 第五步: 理解评分系统

**文件**: [leaderboard/leaderboard/utils/statistics_manager.py](../leaderboard/leaderboard/utils/statistics_manager.py)

| 内容 | 行号 | 说明 |
|------|------|------|
| `PENALTY_VALUE_DICT` | [L21](../leaderboard/leaderboard/utils/statistics_manager.py#L21) | 惩罚系数定义 |
| `PENALTY_PERC_DICT` | [L31](../leaderboard/leaderboard/utils/statistics_manager.py#L31) | 百分比惩罚 |
| `class StatisticsManager` | [L193](../leaderboard/leaderboard/utils/statistics_manager.py#L193) | 统计管理器 |
| `compute_route_statistics()` | [L342](../leaderboard/leaderboard/utils/statistics_manager.py#L342) | 计算路线得分 |

```
Driving Score = Route Completion (%) × Penalty Factor

Penalty 示例:
- 撞行人: ×0.50
- 撞车辆: ×0.60
- 撞静物: ×0.65
- 闯红灯: ×0.70
- 无视停止标志: ×0.80
```

#### 第六步: 场景相关

| 文件 | 关键行 | 说明 |
|------|--------|------|
| [leaderboard/leaderboard/scenarios/route_scenario.py](../leaderboard/leaderboard/scenarios/route_scenario.py) | [L52](../leaderboard/leaderboard/scenarios/route_scenario.py#L52) | 路线场景类 |
| [leaderboard/leaderboard/scenarios/scenario_manager.py](../leaderboard/leaderboard/scenarios/scenario_manager.py) | [L31](../leaderboard/leaderboard/scenarios/scenario_manager.py#L31) | 场景管理器 |
| 同上 | [L139](../leaderboard/leaderboard/scenarios/scenario_manager.py#L139) | `run_scenario()` 主循环 |

### 4.3 路线文件

| 文件 | 路径 | 用途 |
|------|------|------|
| 220条完整评估路线 | [leaderboard/data/bench2drive220.xml](../leaderboard/data/bench2drive220.xml) | 正式评估 |
| 10条开发路线 | [leaderboard/data/drivetransformer_bench2drive_dev10.xml](../leaderboard/data/drivetransformer_bench2drive_dev10.xml) | 快速调试 |

### 4.4 评估脚本

| 脚本 | 路径 | 用途 |
|------|------|------|
| 调试模式 | [leaderboard/scripts/run_evaluation_debug.sh](../leaderboard/scripts/run_evaluation_debug.sh) | 单路线调试 |
| 多GPU评估 | [leaderboard/scripts/run_evaluation_multi_uniad.sh](../leaderboard/scripts/run_evaluation_multi_uniad.sh) | 并行评估 |

### 4.5 工具脚本

| 工具 | 路径 | 用途 |
|------|------|------|
| 合并结果 | [tools/merge_route_json.py](../tools/merge_route_json.py) | 合并220路线JSON |
| 能力评估 | [tools/ability_benchmark.py](../tools/ability_benchmark.py) | 多能力分解 |
| 清理CARLA | [tools/clean_carla.sh](../tools/clean_carla.sh) | 清理僵尸进程 |
| 生成视频 | [tools/generate_video.py](../tools/generate_video.py) | 调试可视化 |

### 4.6 评估流程图

```
run_evaluation.sh
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ LeaderboardEvaluator.__init__()                     │
│   ├── 找可用端口                                    │
│   ├── 启动 CARLA 服务器 (subprocess)                │
│   ├── 等待 CARLA 就绪 (sleep 30s)                   │
│   └── 连接 CARLA (carla.Client)                     │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ FOR EACH route in 220 routes:                       │
│   │                                                 │
│   ├── 加载地图 (Town01-Town12)                      │
│   │                                                 │
│   ├── RouteScenario(route_config)                   │
│   │     ├── 插值生成密集路点                        │
│   │     ├── 生成 Ego 车辆                           │
│   │     └── 初始化沿途场景 (行人、其他车辆等)       │
│   │                                                 │
│   ├── Agent 设置                                    │
│   │     ├── agent.setup(config)                     │
│   │     ├── agent.sensors() → 生成传感器            │
│   │     └── 验证传感器配置                          │
│   │                                                 │
│   └── ScenarioManager.run_scenario()                │
│         │                                           │
│         └── TICK LOOP (20 Hz, max 4000 ticks):      │
│               ├── world.tick()                      │
│               ├── input_data = get_sensor_data()    │
│               ├── control = agent.run_step(...)     │
│               ├── ego.apply_control(control)        │
│               ├── behavior_tree.tick()              │
│               └── criteria.update()                 │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ 输出 JSON 结果 → merge_route_json.py → 最终得分     │
└─────────────────────────────────────────────────────┘
```

---

## 🔴 阶段5: Bench2DriveZoo 实践 (建议1天)

### 5.1 Bench2DriveZoo 是什么?

官方提供的模型实现库，用于:
1. 理解如何实现 Agent
2. 作为新模型的模板
3. 验证环境配置正确

**仓库**: [github.com/Thinklab-SJTU/Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo)

### 5.2 可用分支

| 分支 | 模型 | VRAM需求 | 推荐程度 | 链接 |
|------|------|----------|----------|------|
| `tcp/admlp` | TCP, ADMLP | 6-8GB | ⭐⭐⭐ 推荐新手 | [tcp/admlp](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/tcp/admlp) |
| `uniad/vad` | UniAD, VAD | 12-16GB | 适合有经验者 | [uniad/vad](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/uniad/vad) |

### 5.3 TCP 模型学习路径

**推荐从 TCP 开始**，因为:
- 使用简单的 ResNet backbone
- 直接处理相机图像
- 代码相对简洁
- VRAM 需求低

```bash
# 克隆并切换分支
git clone https://github.com/Thinklab-SJTU/Bench2DriveZoo
cd Bench2DriveZoo
git checkout tcp/admlp
```

### 5.4 核心文件阅读 (Bench2DriveZoo)

**注意**: 以下路径在 Bench2DriveZoo 仓库中

```
Bench2DriveZoo/
├── team_code/
│   └── tcp_agent.py          ← Agent实现 ⭐⭐⭐
├── TCP/
│   ├── model.py              ← 模型定义
│   ├── data.py               ← 数据加载
│   └── config.py             ← 配置
└── scripts/
    └── eval_tcp.sh           ← 评估脚本
```

### 5.5 tcp_agent.py 结构分析

```python
class TCPAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        # 1. 加载配置
        self.config = load_config(path_to_conf_file)

        # 2. 初始化模型
        self.model = TCP(self.config)
        self.model.load_state_dict(torch.load(checkpoint))
        self.model.eval()
        self.model.cuda()

    def sensors(self):
        # 定义7个相机 (前、左前、右前、左、右、左后、右后)
        return [
            {'type': 'sensor.camera.rgb', 'id': 'CAM_FRONT', ...},
            {'type': 'sensor.camera.rgb', 'id': 'CAM_FRONT_LEFT', ...},
            # ...
        ]

    def run_step(self, input_data, timestamp):
        # 1. 提取相机图像
        images = self._get_camera_images(input_data)

        # 2. 获取车辆状态 (速度、位置等)
        ego_state = self._get_ego_state()

        # 3. 获取导航信息 (目标点)
        target_point = self._get_target_point()

        # 4. 模型推理
        with torch.no_grad():
            output = self.model(images, ego_state, target_point)

        # 5. 转换为控制命令
        return self._output_to_control(output)
```

### 5.6 实践练习

1. **运行官方模型**
   - 下载 TCP 预训练权重
   - 配置环境
   - 在 Dev10 (10条路线) 上测试

2. **理解数据流**
   - 打印 `input_data` 的结构
   - 可视化相机图像
   - 记录模型输入输出

3. **尝试修改**
   - 添加新的传感器
   - 修改控制逻辑
   - 观察效果变化

---

## 📝 学习检查清单

### 阶段1完成标志 ✓
- [ ] 能解释 World, Actor, Sensor 的关系
- [ ] 理解同步模式 vs 异步模式
- [ ] 能手写代码生成车辆并控制移动
- [ ] 理解传感器回调机制

### 阶段2完成标志 ✓
- [ ] 能清晰解释开环 vs 闭环的区别
- [ ] 理解闭环测试的主循环
- [ ] 知道为什么闭环测试更具挑战性

### 阶段3完成标志 ✓
- [ ] 理解 CarlaDataProvider 的作用
- [ ] 能读懂 py_trees 行为树代码
- [ ] 知道常用的原子行为和评估标准

### 阶段4完成标志 ✓
- [ ] 理解 Leaderboard Evaluator 的执行流程
- [ ] 能实现一个简单的 AutonomousAgent
- [ ] 理解评分系统 (Driving Score)
- [ ] 能在本地运行评估脚本

### 阶段5完成标志 ✓
- [ ] 成功运行 TCP 模型评估
- [ ] 理解 tcp_agent.py 的代码结构
- [ ] 知道如何添加自己的模型

---

## 🔧 常见问题速查

### Q: CARLA 启动失败?
```bash
# 检查 Vulkan
/usr/bin/vulkaninfo | head -n 5

# 检查端口
lsof -i:2000

# 清理旧进程
bash tools/clean_carla.sh
```

### Q: 评估中途崩溃?
```bash
# 设置 RESUME=True 继续
# 或重新运行，已完成的路线会跳过
```

### Q: 模型推理太慢?
- 检查是否使用了 GPU
- 减少传感器数量
- 降低图像分辨率

---

## 📚 参考资源汇总

### 官方文档

| 资源 | 链接 |
|------|------|
| CARLA 0.9.15 文档 | [carla.readthedocs.io/en/0.9.15](https://carla.readthedocs.io/en/0.9.15/) |
| CARLA Python API | [carla.readthedocs.io/en/0.9.15/python_api](https://carla.readthedocs.io/en/0.9.15/python_api/) |
| Bench2Drive 论文 | [arxiv.org/abs/2406.03877](https://arxiv.org/abs/2406.03877) |
| Bench2DriveZoo | [github.com/Thinklab-SJTU/Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) |
| py_trees 文档 | [py-trees.readthedocs.io](https://py-trees.readthedocs.io/en/devel/) |

### 本项目关键文件快速跳转

| 文件 | 用途 |
|------|------|
| [docs/SERVER_DEPLOYMENT_GUIDE.md](../docs/SERVER_DEPLOYMENT_GUIDE.md) | 部署指南 |
| [CLAUDE.md](../CLAUDE.md) | 架构概览 |
| [leaderboard/leaderboard/autoagents/autonomous_agent.py#L32](../leaderboard/leaderboard/autoagents/autonomous_agent.py#L32) | Agent 基类 |
| [leaderboard/leaderboard/leaderboard_evaluator.py#L80](../leaderboard/leaderboard/leaderboard_evaluator.py#L80) | 评估器 |
| [scenario_runner/srunner/scenariomanager/carla_data_provider.py#L34](../scenario_runner/srunner/scenariomanager/carla_data_provider.py#L34) | CARLA交互单例 |
| [leaderboard/leaderboard/utils/statistics_manager.py#L21](../leaderboard/leaderboard/utils/statistics_manager.py#L21) | 惩罚系数 |

### 视频教程
- Bench2Drive 演示: [youtube.com/watch?v=-osdzJJs2g0](https://www.youtube.com/watch?v=-osdzJJs2g0)

---

*最后更新: 2025-12-21*
