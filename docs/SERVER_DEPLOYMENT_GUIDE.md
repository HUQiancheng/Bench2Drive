# Bench2Drive 服务器部署完整指南

本指南详细说明如何在**无显示器的Linux服务器**上完整运行Bench2Drive评估流程。

## 目录

1. [环境要求](#1-环境要求)
2. [CARLA 安装与配置](#2-carla-安装与配置)
3. [Bench2Drive 环境配置](#3-bench2drive-环境配置)
4. [验证环境 - NpcAgent测试](#4-验证环境---npcagent测试)
5. [运行视觉模型 - TCP测试](#5-运行视觉模型---tcp测试)
6. [接入自定义模型](#6-接入自定义模型)
7. [结果分析与可视化](#7-结果分析与可视化)
8. [常见问题排查](#8-常见问题排查)

---

## 1. 环境要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | 6GB VRAM | 8GB+ VRAM |
| CPU | 4核 | 8核+ |
| 内存 | 16GB | 32GB+ |
| 磁盘 | 50GB | 100GB+ (含数据集) |

### 显存估算

| 配置 | 预计显存占用 |
|------|-------------|
| CARLA (低质量无头模式) | 3-4 GB |
| CARLA + ADMLP | 4-5 GB |
| CARLA + TCP | 5-7 GB |
| CARLA + UniAD/VAD | 12-16 GB |

### 软件要求

- Ubuntu 18.04/20.04/22.04
- NVIDIA Driver >= 470 (推荐470，515可用，**避免550有bug**)
- Vulkan 支持 (CARLA 0.9.12+ 必需)
- Python 3.7 或 3.8
- Conda (推荐)

---

## 2. CARLA 安装与配置

> **重要背景**: CARLA 0.9.12+ 基于 Unreal Engine 4.26，**只支持 Vulkan 图形API**，不再支持 OpenGL。
> 这意味着必须正确配置 Vulkan 才能在无头服务器上运行。

### 2.1 安装系统依赖 (首先执行!)

在安装 CARLA 之前，必须确保系统依赖正确安装。

#### Step 1: 检查 NVIDIA 驱动

```bash
# 检查驱动版本
nvidia-smi

# 预期输出示例:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 470.xxx    Driver Version: 470.xxx    CUDA Version: 11.x        |
# +-----------------------------------------------------------------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# +-----------------------------------------------------------------------------+

# 如果没有安装驱动，安装推荐版本:
# sudo apt install nvidia-driver-470
```

**驱动版本建议:**
- ✅ 470.x - 最稳定，推荐使用
- ⚠️ 515.x - 有些问题但可用
- ❌ 550.x - 已知有很多bug，避免使用

#### Step 2: 安装 Vulkan 依赖

```bash
# 安装 Vulkan 工具和库
sudo apt update
sudo apt install -y vulkan-tools vulkan-utils libvulkan1 libvulkan-dev

# 对于 NVIDIA GPU，确保安装 libnvidia-gl
sudo apt install -y libnvidia-gl-470  # 版本号匹配你的驱动

# 安装其他可能需要的库
sudo apt install -y libsdl2-2.0-0 libsdl2-dev
```

#### Step 3: 验证 Vulkan 安装 (关键步骤!)

```bash
# 运行 vulkaninfo
vulkaninfo | head -20

# ✅ 正确输出示例:
# ==========
# VULKANINFO
# ==========
# Vulkan Instance Version: 1.3.xxx
#
# Instance Extensions: count = xx
# ...
# GPU0: NVIDIA GeForce RTX xxxx (或其他NVIDIA GPU)

# ❌ 错误输出示例:
# WARNING: lavapipe is not a conformant vulkan implementation, testing use only.
# (这表示使用的是软件渲染，不是GPU渲染!)
```

**如果 vulkaninfo 显示 lavapipe 警告:**

```bash
# 设置 Vulkan ICD 路径，强制使用 NVIDIA
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
# 或者
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# 找到正确的路径
find /usr -name "nvidia_icd.json" 2>/dev/null
find /etc -name "nvidia_icd.json" 2>/dev/null

# 再次验证
vulkaninfo | head -20
```

#### Step 4: 安装其他依赖

```bash
# 图形相关库
sudo apt install -y libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

# OpenGL 库 (虽然 CARLA 用 Vulkan，但有些组件可能需要)
sudo apt install -y freeglut3-dev mesa-utils

# 网络工具
sudo apt install -y lsof net-tools
```

### 2.2 下载 CARLA 0.9.15

```bash
# 创建目录
mkdir -p ~/carla && cd ~/carla

# 下载 CARLA 主程序 (~13GB)
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz

# 解压 (需要约20GB空间)
tar -xzf CARLA_0.9.15.tar.gz

# 下载额外地图 (Bench2Drive需要Town12等地图)
cd Import
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd ..

# 导入额外地图
bash ImportAssets.sh

# 设置环境变量
export CARLA_ROOT=~/carla
echo 'export CARLA_ROOT=~/carla' >> ~/.bashrc
source ~/.bashrc
```

**验证下载完整性:**
```bash
# 检查关键文件是否存在
ls -la $CARLA_ROOT/CarlaUE4.sh
ls -la $CARLA_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping

# 检查 Python API
ls -la $CARLA_ROOT/PythonAPI/carla/dist/
# 应该看到: carla-0.9.15-py3.7-linux-x86_64.egg
```

### 2.3 配置 Python CARLA 模块

```bash
# 方法1: 使用 .pth 文件 (推荐，永久生效)
# 首先确认 conda 环境已激活
conda activate your_env_name

# 创建 .pth 文件
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> \
    $CONDA_PREFIX/lib/python3.7/site-packages/carla.pth

# 方法2: 使用 pip 安装 wheel (替代方案)
pip install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-linux_x86_64.whl

# 方法3: 使用环境变量 (临时，每次需要设置)
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
```

**验证 Python 模块:**
```bash
python3 -c "import carla; print(f'CARLA Python API version: {carla.__file__}')"
# 应该输出 egg 文件的路径，没有报错
```

### 2.4 无头模式启动 CARLA

> **关键概念**: CARLA 0.9.12+ 使用 `-RenderOffScreen` 参数实现无头渲染。
> 这与旧版本使用的 `SDL_VIDEODRIVER=offscreen` 或 `-opengl` 不同!

#### 基础启动命令

```bash
cd $CARLA_ROOT

# 基础无头模式
./CarlaUE4.sh -RenderOffScreen

# 指定端口 (推荐使用大端口号避免冲突)
./CarlaUE4.sh -RenderOffScreen -carla-port=2000

# 低质量模式 (节省显存，推荐用于评估)
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000

# 指定GPU (多GPU服务器)
./CarlaUE4.sh -RenderOffScreen -carla-port=2000 -graphicsadapter=0
```

#### 完整参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `-RenderOffScreen` | 无头模式，不需要显示器 | 必需 |
| `-quality-level=Low` | 低画质，节省显存，绘制距离50m | 推荐 |
| `-quality-level=Epic` | 高画质，完整渲染 (默认) | 需要更多显存 |
| `-carla-port=PORT` | 指定TCP端口 | `-carla-port=2000` |
| `-graphicsadapter=N` | 指定GPU编号 | `-graphicsadapter=0` |
| `-carla-server` | 显式服务器模式 | 可选 |
| `-benchmark -fps=10` | 固定帧率模式 | 调试用 |

**关于 `-graphicsadapter` 的重要说明:**
- 这不是 CUDA 设备编号!
- GPU编号可能跳跃，例如4个GPU可能是: 0, 2, 3, 4 (跳过1)
- 需要测试确定正确的映射关系

#### 后台运行

```bash
# 使用 nohup 后台运行
cd $CARLA_ROOT
nohup ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 > carla.log 2>&1 &

# 检查是否启动成功
sleep 30
ps aux | grep CarlaUE4

# 查看日志
tail -f carla.log
```

### 2.5 验证 CARLA 安装 (关键步骤!)

按顺序执行以下验证步骤:

#### Step 1: 启动 CARLA 服务器

```bash
cd $CARLA_ROOT
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 &

# 等待启动 (第一次可能需要更长时间)
echo "Waiting for CARLA to start..."
sleep 30
```

**预期输出 (正常):**
```
4.26.2-0+++UE4+Release-4.26 522 0
Disabling core dumps.
```
如果只显示这两行然后卡住，这是**正常的**! CARLA 正在运行。

**错误输出示例:**
```
# 如果立即退出，可能是 Vulkan 问题
WARNING: lavapipe is not a conformant vulkan implementation, testing use only.
# 解决: 检查 VK_ICD_FILENAMES 环境变量

# 如果显示 GPU 错误
Cannot find a compatible Vulkan installable client driver (ICD)
# 解决: 重新安装 NVIDIA 驱动和 Vulkan
```

#### Step 2: 测试 Python 连接

```bash
# 打开新终端，运行测试脚本
python3 << 'EOF'
import carla
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    print(f"CARLA Version: {client.get_server_version()}")
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
EOF
```

预期输出：
```
CARLA Version: 0.9.15
Connection successful!
```

---

## 3. Bench2Drive 环境配置

### 3.1 克隆仓库

```bash
cd ~
git clone https://github.com/Thinklab-SJTU/Bench2Drive.git
cd Bench2Drive

# 克隆模型仓库
git clone https://github.com/Thinklab-SJTU/Bench2DriveZoo.git

# 或者创建软链接 (如果已在其他位置克隆)
# ln -s /path/to/Bench2DriveZoo ./Bench2DriveZoo
```

### 3.2 创建 Conda 环境

```bash
conda create -n b2d python=3.7 -y
conda activate b2d

# 安装基础依赖
pip install torch torchvision  # 根据CUDA版本选择
pip install numpy opencv-python pillow scipy
pip install py-trees==0.8.3 networkx==2.2 six
pip install psutil shapely xmlschema ephem tabulate

# 配置 CARLA Python 模块
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> \
    $CONDA_PREFIX/lib/python3.7/site-packages/carla.pth
```

### 3.3 切换 Bench2DriveZoo 分支

根据你要测试的模型选择分支：

```bash
cd Bench2DriveZoo

# 如果要用 TCP/ADMLP (轻量模型)
git remote add upstream https://github.com/Thinklab-SJTU/Bench2DriveZoo.git
git fetch upstream
git checkout -b tcp/admlp upstream/tcp/admlp

# 如果要用 UniAD/VAD (大模型)
git checkout -b uniad/vad upstream/uniad/vad
```

### 3.4 创建 team_code 目录

```bash
cd ~/Bench2Drive
mkdir -p leaderboard/team_code

# 链接 Bench2DriveZoo 的 agent 文件
ln -s ../../Bench2DriveZoo/team_code/* leaderboard/team_code/
```

目录结构应如下：
```
Bench2Drive/
├── Bench2DriveZoo/          # 模型仓库 (软链接或克隆)
│   ├── TCP/                 # TCP 模型代码
│   ├── ADMLP/               # ADMLP 模型代码
│   └── team_code/           # Agent 实现
├── leaderboard/
│   ├── team_code/           # -> 链接到 Bench2DriveZoo/team_code
│   ├── scripts/             # 评估脚本
│   └── data/                # 路线定义 XML
└── scenario_runner/
```

---

## 4. 验证环境 - NpcAgent测试

在测试自己的模型之前，先用内置的 NpcAgent 验证整个流程是否通畅。

### 4.1 NpcAgent 说明

NpcAgent 使用 CARLA 内置的 BasicAgent 自动驾驶，**不需要任何深度学习模型**，只需要 CARLA 能正常运行即可。

### 4.2 启动 CARLA

```bash
cd $CARLA_ROOT
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 &
sleep 30
```

### 4.3 运行 NpcAgent 评估

```bash
cd ~/Bench2Drive

# 设置环境变量
export CARLA_ROOT=~/carla
export ROUTES=leaderboard/data/bench2drive220.xml  # 或使用 dev10 快速测试
export ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10.xml  # 10条路线，快速测试
export TEAM_AGENT=leaderboard/leaderboard/autoagents/npc_agent.py
export TEAM_CONFIG=""  # NpcAgent 不需要配置文件
export CHECKPOINT_ENDPOINT=results/npc_test.json
export SAVE_PATH=results/npc_test/

# 运行单条路线测试
python leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=$ROUTES \
    --routes-subset=0 \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint=$CHECKPOINT_ENDPOINT \
    --agent=$TEAM_AGENT \
    --agent-config=$TEAM_CONFIG \
    --debug=0 \
    --record="" \
    --resume=False \
    --port=2000 \
    --timeout=600
```

### 4.4 检查结果

```bash
# 查看 JSON 结果
cat results/npc_test.json

# 应该能看到类似输出:
# {
#   "records": [...],
#   "progress": [1, 1],
#   ...
# }
```

**如果 NpcAgent 能成功运行，说明 CARLA + Bench2Drive 环境正常！**

---

## 5. 运行视觉模型 - TCP测试

TCP 是一个使用图像输入的视觉模型，适合作为自定义视觉模型的测试模板。

### 5.1 下载 TCP Checkpoint

```bash
# 从 Hugging Face 下载
cd ~/Bench2Drive
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/rethinklab/Bench2DriveZoo/resolve/main/tcp_b2d.ckpt
```

或者从百度云下载: https://pan.baidu.com/s/1CgYscY2esIJLRepkO3FBvQ?pwd=1234

### 5.2 安装 TCP 依赖

```bash
cd ~/Bench2Drive/Bench2DriveZoo
pip install -r requirements.txt  # 如果有的话

# TCP 需要的额外依赖
pip install imgaug
```

### 5.3 运行 TCP 评估

```bash
cd ~/Bench2Drive

# 确保 CARLA 在运行
# ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 &

# 设置环境变量
export CARLA_ROOT=~/carla
export PYTHONPATH=$PYTHONPATH:~/Bench2Drive/Bench2DriveZoo
export ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10.xml
export TEAM_AGENT=leaderboard/team_code/tcp_b2d_agent.py
export TEAM_CONFIG=checkpoints/tcp_b2d.ckpt
export PLANNER_TYPE=only_traj  # 可选: only_ctrl, only_traj, merge_ctrl_traj
export IS_BENCH2DRIVE=1
export CHECKPOINT_ENDPOINT=results/tcp_test.json
export SAVE_PATH=results/tcp_test/

# 运行评估
python leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=$ROUTES \
    --routes-subset=0 \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint=$CHECKPOINT_ENDPOINT \
    --agent=$TEAM_AGENT \
    --agent-config=$TEAM_CONFIG \
    --debug=0 \
    --record="" \
    --resume=False \
    --port=2000 \
    --timeout=600
```

### 5.4 TCP Agent 核心代码解析

理解 TCP agent 的结构对于接入自己的模型很重要：

```python
# tcp_b2d_agent.py 核心结构

class TCPAgent(autonomous_agent.AutonomousAgent):

    def setup(self, path_to_conf_file):
        """初始化模型"""
        self.config = GlobalConfig()
        self.net = TCP(self.config)
        ckpt = torch.load(path_to_conf_file)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.cuda()
        self.net.eval()

    def sensors(self):
        """定义需要的传感器"""
        return [
            {'type': 'sensor.camera.rgb', 'id': 'CAM_FRONT', ...},  # 前方相机
            {'type': 'sensor.camera.rgb', 'id': 'CAM_FRONT_LEFT', ...},
            {'type': 'sensor.camera.rgb', 'id': 'CAM_FRONT_RIGHT', ...},
            {'type': 'sensor.other.imu', 'id': 'IMU', ...},
            {'type': 'sensor.other.gnss', 'id': 'GPS', ...},
            {'type': 'sensor.speedometer', 'id': 'SPEED', ...},
        ]

    def run_step(self, input_data, timestamp):
        """每帧调用，返回控制指令"""
        # 1. 预处理图像
        rgb = self._im_transform(input_data['CAM_FRONT'][1]).to('cuda')

        # 2. 获取车辆状态
        speed = input_data['SPEED'][1]['speed']

        # 3. 模型推理
        pred = self.net(rgb, state, target_point)

        # 4. 生成控制指令
        control = carla.VehicleControl()
        control.steer = pred['steer']
        control.throttle = pred['throttle']
        control.brake = pred['brake']
        return control
```

---

## 6. 接入自定义模型

### 6.1 Agent 模板

创建 `leaderboard/team_code/your_model_agent.py`:

```python
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
from leaderboard.autoagents import autonomous_agent

# 导入你的模型
from your_model_package import YourModel, YourConfig

def get_entry_point():
    return 'YourModelAgent'

class YourModelAgent(autonomous_agent.AutonomousAgent):

    def setup(self, path_to_conf_file):
        """初始化 - 加载模型"""
        self.track = autonomous_agent.Track.SENSORS

        # 图像预处理
        self._im_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

        # 加载配置和模型
        config_path, ckpt_path = path_to_conf_file.split('+')
        self.config = YourConfig.from_file(config_path)
        self.model = YourModel(self.config)

        # 加载权重
        ckpt = torch.load(ckpt_path, map_location='cuda')
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.cuda()
        self.model.eval()

        self.step = 0

    def sensors(self):
        """定义传感器配置"""
        sensors = [
            # 前方相机
            {
                'type': 'sensor.camera.rgb',
                'x': 0.80, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT'
            },
            # 可以添加更多相机...
            # IMU
            {
                'type': 'sensor.other.imu',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'IMU'
            },
            # GPS
            {
                'type': 'sensor.other.gnss',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'sensor_tick': 0.01,
                'id': 'GPS'
            },
            # 速度计
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'SPEED'
            },
        ]
        return sensors

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        """每帧推理"""
        self.step += 1

        # 1. 获取传感器数据
        # 图像格式: (H, W, 4) BGRA
        rgb_front = input_data['CAM_FRONT'][1][:, :, :3]  # 去掉Alpha通道
        rgb_front = rgb_front[:, :, ::-1]  # BGR -> RGB

        # GPS: [lat, lon, alt]
        gps = input_data['GPS'][1][:2]

        # IMU: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, compass]
        imu = input_data['IMU'][1]
        compass = imu[-1]  # 航向角

        # 速度
        speed = input_data['SPEED'][1]['speed']

        # 2. 预处理
        rgb_tensor = self._im_transform(Image.fromarray(rgb_front))
        rgb_tensor = rgb_tensor.unsqueeze(0).cuda()

        # 3. 模型推理
        with torch.no_grad():
            output = self.model(rgb_tensor, speed, compass, ...)

        # 4. 后处理 - 获取控制指令
        # 假设模型输出轨迹点，需要用PID转换为控制
        waypoints = output['trajectory'].cpu().numpy()
        steer, throttle, brake = self.trajectory_to_control(waypoints, speed)

        # 5. 返回控制指令
        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.throttle = float(np.clip(throttle, 0.0, 0.75))
        control.brake = float(np.clip(brake, 0.0, 1.0))

        return control

    def trajectory_to_control(self, waypoints, speed):
        """将轨迹点转换为控制指令 (简化版PID)"""
        # 这里需要实现你的控制逻辑
        # 可以参考 TCP 或 ADMLP 的 control_pid 方法
        aim_point = waypoints[2]  # 选择一个目标点
        angle = np.arctan2(aim_point[1], aim_point[0])
        steer = angle / np.pi  # 归一化到 [-1, 1]

        desired_speed = np.linalg.norm(waypoints[1] - waypoints[0]) * 20  # 估算期望速度
        if speed < desired_speed:
            throttle = 0.5
            brake = 0.0
        else:
            throttle = 0.0
            brake = 0.3

        return steer, throttle, brake

    def destroy(self):
        """清理资源"""
        del self.model
        torch.cuda.empty_cache()
```

### 6.2 传感器配置说明

Bench2Drive 支持的传感器类型：

| 类型 | 限制 | 数据格式 |
|------|------|----------|
| `sensor.camera.rgb` | 最多8个 | (H, W, 4) BGRA uint8 |
| `sensor.lidar.ray_cast` | 最多2个 | (N, 4) x,y,z,intensity |
| `sensor.other.radar` | 最多4个 | 点云数据 |
| `sensor.other.gnss` | 1个 | [lat, lon, alt] |
| `sensor.other.imu` | 1个 | [acc(3), gyro(3), compass] |
| `sensor.speedometer` | 1个 | {'speed': float} |

### 6.3 运行自定义模型

```bash
export TEAM_AGENT=leaderboard/team_code/your_model_agent.py
export TEAM_CONFIG=path/to/config.py+path/to/checkpoint.pth

python leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=$ROUTES \
    --routes-subset=0 \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint=results/your_model.json \
    --agent=$TEAM_AGENT \
    --agent-config=$TEAM_CONFIG \
    --port=2000
```

---

## 7. 结果分析与可视化

### 7.1 合并结果

```bash
# 合并多个路线的 JSON 结果
python tools/merge_route_json.py -f results/your_model/

# 输出 merge.json，包含驾驶分数等指标
```

### 7.2 多能力评估

```bash
python tools/ability_benchmark.py -r results/merge.json
```

### 7.3 效率和平滑度

```bash
python tools/efficiency_smoothness_benchmark.py \
    -f results/merge.json \
    -m results/your_model/  # metric_info.json 所在目录
```

### 7.4 生成可视化视频

如果在评估时保存了传感器数据：

```bash
python tools/generate_video.py -f results/your_model/rgb_front/
```

### 7.5 评分系统说明

**驾驶分数计算：**
```
Driving Score = Route Completion (%) × Penalty Factor
```

**惩罚因子 (累乘)：**

| 违规类型 | 惩罚系数 |
|----------|----------|
| 撞行人 | 0.50 |
| 撞车辆 | 0.60 |
| 撞静态物体 | 0.65 |
| 闯红灯 | 0.70 |
| 无视停止标志 | 0.80 |
| 场景超时 | 0.70 |

---

## 8. 常见问题排查

### 8.1 CARLA 启动问题

#### 问题：CARLA 启动后立即退出

**症状**: 运行 `./CarlaUE4.sh` 后只显示版本信息就退出，没有保持运行。

**诊断步骤:**
```bash
# 1. 检查 Vulkan 是否正确配置
vulkaninfo | head -20

# 如果看到 "lavapipe" 警告，说明没有使用 GPU
# WARNING: lavapipe is not a conformant vulkan implementation

# 2. 检查 NVIDIA ICD 文件
ls -la /usr/share/vulkan/icd.d/
ls -la /etc/vulkan/icd.d/
# 应该看到 nvidia_icd.json

# 3. 设置正确的 Vulkan ICD
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
# 或
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# 4. 再次验证
vulkaninfo | head -20
# 应该看到 NVIDIA GPU，而不是 lavapipe
```

**解决方案:**
```bash
# 方案1: 重新安装 Vulkan 和 NVIDIA GL
sudo apt install --reinstall vulkan-tools libvulkan1
sudo apt install --reinstall libnvidia-gl-470  # 版本号匹配驱动

# 方案2: 将 VK_ICD_FILENAMES 添加到环境变量
echo 'export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json' >> ~/.bashrc
source ~/.bashrc
```

#### 问题：显示 "Cannot find a compatible Vulkan driver"

**原因**: NVIDIA 驱动未正确安装，或 Vulkan ICD 未配置。

```bash
# 检查驱动
nvidia-smi
# 如果失败，需要重新安装 NVIDIA 驱动

# 重装驱动 (谨慎操作)
sudo apt purge nvidia-*
sudo apt install nvidia-driver-470
sudo reboot
```

#### 问题：CARLA 启动卡住不动

**症状**: 显示版本信息后长时间无响应。

```bash
# 这通常是正常的! CARLA 正在后台运行
# 验证方法:
ps aux | grep CarlaUE4

# 测试连接:
python3 -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(5); print(c.get_server_version())"
```

#### 问题：显存不足 (Out of Memory)

```bash
# 方案1: 使用低质量模式
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000

# 方案2: 减少传感器分辨率 (修改 agent 的 sensors() 方法)
# 将 width: 1600, height: 900 改为 width: 800, height: 450

# 方案3: 使用更小的模型 (ADMLP 几乎不占显存)
```

#### 问题：清理残留 CARLA 进程

CARLA 崩溃后可能留下僵尸进程，导致端口占用或GPU显存泄漏。

```bash
# 方法1: 使用 Bench2Drive 提供的脚本 (可能需要多次运行)
bash tools/clean_carla.sh
bash tools/clean_carla.sh
bash tools/clean_carla.sh

# 方法2: 手动清理
pkill -9 CarlaUE4
pkill -9 CarlaUE4-Linux

# 方法3: 查找并杀死所有相关进程
ps aux | grep -i carla | awk '{print $2}' | xargs -r kill -9

# 验证清理完成
ps aux | grep CarlaUE4
lsof -i:2000  # 应该无输出
nvidia-smi   # 检查显存是否释放
```

### 8.2 评估脚本问题

**问题：端口被占用**
```bash
# 检查端口
lsof -i:2000

# 使用其他端口
./CarlaUE4.sh -RenderOffScreen -carla-port=3000
# 评估时也要改 --port=3000
```

**问题：评估中途崩溃**
```bash
# 使用 resume 模式继续
python leaderboard/leaderboard/leaderboard_evaluator.py \
    ... \
    --resume=True
```

**问题：Agent 超时**
```bash
# 增加超时时间
--timeout=1200  # 默认600秒
```

### 8.3 模型推理问题

**问题：CUDA out of memory**
```bash
# 减小 batch size (如果模型支持)
# 使用混合精度推理
# 或者使用更小的模型 (ADMLP)
```

**问题：传感器数据格式错误**
```python
# 图像是 BGRA 格式，需要转换
rgb = input_data['CAM_FRONT'][1][:, :, :3]  # 去掉 Alpha
rgb = rgb[:, :, ::-1]  # BGR -> RGB
```

### 8.4 多GPU并行评估

```bash
# 在不同GPU上运行不同路线
# GPU 0: 路线 0-54
CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh -RenderOffScreen -carla-port=2000 -graphicsadapter=0 &
# 注意：-graphicsadapter 可能需要调整，不一定和 CUDA 编号一致

# GPU 1: 路线 55-109
./CarlaUE4.sh -RenderOffScreen -carla-port=2150 -graphicsadapter=1 &

# 分别运行评估脚本，指定不同的 --port 和 --routes-subset
```

---

## 快速参考

### 常用命令

```bash
# 启动 CARLA (无头)
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 &

# 杀死 CARLA
pkill -9 CarlaUE4

# 运行评估
python leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=leaderboard/data/drivetransformer_bench2drive_dev10.xml \
    --routes-subset=0 \
    --track=SENSORS \
    --checkpoint=results/test.json \
    --agent=leaderboard/team_code/your_agent.py \
    --agent-config=path/to/config+checkpoint.pth \
    --port=2000

# 合并结果
python tools/merge_route_json.py -f results/
```

### 环境变量

```bash
export CARLA_ROOT=~/carla
export PYTHONPATH=$PYTHONPATH:~/Bench2Drive/Bench2DriveZoo
export IS_BENCH2DRIVE=1
export SAVE_PATH=results/   # 保存传感器数据
export PLANNER_TYPE=only_traj  # TCP专用
```

---

## 附录A：完整测试流程检查清单

按顺序完成以下步骤，每步都有对应的验证命令：

### 阶段1: 系统依赖

- [ ] **NVIDIA 驱动安装正确**
  ```bash
  nvidia-smi  # 应显示 GPU 信息和驱动版本
  ```

- [ ] **Vulkan 配置正确**
  ```bash
  vulkaninfo | head -20  # 应显示 NVIDIA GPU，不是 lavapipe
  ```

- [ ] **Vulkan ICD 设置** (如需要)
  ```bash
  echo $VK_ICD_FILENAMES  # 应指向 nvidia_icd.json
  ```

### 阶段2: CARLA 安装

- [ ] **CARLA 文件完整**
  ```bash
  ls $CARLA_ROOT/CarlaUE4.sh  # 应存在
  ls $CARLA_ROOT/PythonAPI/carla/dist/  # 应有 .egg 或 .whl 文件
  ```

- [ ] **CARLA 可以启动**
  ```bash
  cd $CARLA_ROOT && ./CarlaUE4.sh -RenderOffScreen -carla-port=2000 &
  sleep 30 && ps aux | grep CarlaUE4  # 应看到进程
  ```

- [ ] **Python 可以连接**
  ```bash
  python3 -c "import carla; c=carla.Client('localhost',2000); print(c.get_server_version())"
  # 应输出: 0.9.15
  ```

### 阶段3: Bench2Drive 环境

- [ ] **Python 模块可导入**
  ```bash
  python3 -c "import carla; print('carla OK')"
  python3 -c "import torch; print('torch OK')"
  ```

- [ ] **目录结构正确**
  ```bash
  ls ~/Bench2Drive/leaderboard/team_code/  # 应有 agent 文件
  ls ~/Bench2Drive/Bench2DriveZoo/  # 应有模型代码
  ```

### 阶段4: 评估测试

- [ ] **NpcAgent 测试通过**
  ```bash
  # 运行评估，检查输出 JSON
  cat results/npc_test.json | head -20
  ```

- [ ] **TCP/ADMLP 测试通过** (可选)
  ```bash
  # 运行评估，检查驾驶分数
  python tools/merge_route_json.py -f results/tcp_test/
  ```

---

## 附录B：参考资料

### 官方文档

- [CARLA Documentation](https://carla.readthedocs.io/en/0.9.15/)
- [CARLA Quick Start](https://carla.readthedocs.io/en/latest/start_quickstart/)
- [CARLA Rendering Options](https://carla.readthedocs.io/en/latest/adv_rendering_options/)

### GitHub Issues (常见问题)

- [Running CARLA on Headless Server](https://github.com/carla-simulator/carla/issues/3943)
- [Vulkan Issues with Docker](https://github.com/carla-simulator/carla/issues/6234)
- [Running CARLA Offscreen](https://github.com/carla-simulator/carla/issues/3671)

### Bench2Drive 资源

- [Bench2Drive GitHub](https://github.com/Thinklab-SJTU/Bench2Drive)
- [Bench2DriveZoo (Models)](https://github.com/Thinklab-SJTU/Bench2DriveZoo)
- [Checkpoints on HuggingFace](https://huggingface.co/rethinklab/Bench2DriveZoo)

### 教程

- [CARLA Headless Tutorial](https://arijitray.com/CARLA_tutorial/) - 远程服务器数据采集教程

---

## 附录C：版本信息

本文档针对以下版本编写：

| 组件 | 版本 |
|------|------|
| CARLA | 0.9.15 |
| Unreal Engine | 4.26 |
| Python | 3.7 / 3.8 |
| 推荐 NVIDIA Driver | 470.x |
| Ubuntu | 18.04 / 20.04 / 22.04 |

**最后更新**: 2024年12月
