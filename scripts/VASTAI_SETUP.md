# Vast.ai CARLA 环境配置指南

## 关键要求

**必须选择 NVIDIA Driver 470-535 的机器！**

- Driver 550+ 与 CARLA 0.9.15 不兼容（Vulkan 渲染崩溃）
- 在 Vast.ai 筛选时设置 `Driver Version: 535` 或 `470-535`

---

## 遇到的问题及解决方案

### 问题1: CARLA 拒绝 root 用户运行
```
Refusing to run with the root privileges.
```
**原因**：Unreal Engine 4 安全限制，禁止 root 运行
**解决**：创建非 root 用户运行 CARLA

### 问题2: Vulkan 无法识别 NVIDIA GPU
```
vulkaninfo 显示 llvmpipe (CPU软渲染) 而非 NVIDIA GPU
```
**原因**：缺少 NVIDIA Vulkan ICD 配置文件
**解决**：手动创建 `/usr/share/vulkan/icd.d/nvidia_icd.json`

### 问题3: 驱动版本过新导致渲染崩溃
```
GameThread timed out waiting for RenderThread after 60.00 secs
Segmentation fault
```
**原因**：NVIDIA Driver 550+/570+/580+ 与 CARLA 0.9.15 不兼容
**解决**：选择 Driver 535 或更低版本的机器

### 问题4: Python 模块缺失
```
ModuleNotFoundError: No module named 'srunner'
ModuleNotFoundError: No module named 'six'
ModuleNotFoundError: No module named 'agents'
```
**原因**：未设置 PYTHONPATH 和缺少依赖包
**解决**：设置环境变量并安装依赖（见下方配置步骤）

### 问题5: leaderboard_evaluator 自动启动 CARLA 失败
```
/root/qch_ws/carla/CarlaUE4.sh ... None
Refusing to run with the root privileges.
```
**原因**：评估脚本以 root 运行，自动启动的 CARLA 拒绝 root
**解决**：修改 `leaderboard_evaluator.py`，root 时用 `su - carla -c` 启动 CARLA（已合并到仓库）

### 问题6: 路由 ID 不匹配
```
ValueError: Couldn't find the route with id '0' inside the given routes file
```
**原因**：`drivetransformer_bench2drive_dev10.xml` 中路由 id 从 3514 开始，不是 0
**解决**：去掉 `--routes-subset` 参数，或使用正确的 id（如 `--routes-subset=3514`）

### 问题7: 缺少额外地图
```
Town13 等地图不存在
```
**原因**：Bench2Drive 需要 Town12、Town13 等额外地图，默认 CARLA 不包含
**解决**：下载并导入 AdditionalMaps（见下方配置步骤）

---

## 完整配置步骤

### 1. 选择机器（重要！）
在 Vast.ai Machine Options 中设置：
- **Driver Version**: `535` 或 `470`
- **Min Cuda Version**: `11`
- **Ubuntu Version**: `20.04`

### 2. 验证驱动版本
```bash
nvidia-smi
# 确认 Driver Version 是 535.x 或更低
```

### 3. 安装 Vulkan 工具
```bash
apt update && apt install vulkan-tools libxext6 libx11-6 -y
```

### 4. 创建 NVIDIA Vulkan ICD（如果缺失）
```bash
# 检查是否存在
ls /usr/share/vulkan/icd.d/nvidia*.json

# 如果不存在，手动创建
cat > /usr/share/vulkan/icd.d/nvidia_icd.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.194"
    }
}
EOF

# 验证 Vulkan 识别 NVIDIA GPU
vulkaninfo 2>&1 | grep -E "deviceName" | head -5
# 应该看到 "NVIDIA GeForce RTX 3090" 或类似
```

### 5. 安装 Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init
source ~/.bashrc

# 接受条款（如果提示）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 6. 下载 CARLA
```bash
mkdir -p ~/qch_ws/carla && cd ~/qch_ws/carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
```

### 6.1 下载额外地图（Bench2Drive 需要 Town12/Town13）
```bash
cd ~/qch_ws/carla/Import
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd ..
bash ImportAssets.sh

# 验证地图已导入
ls CarlaUE4/Content/Carla/Maps/ | grep -i town1
# 应该看到 Town10HD, Town11, Town12, Town13 等
```

### 7. 创建 Python 3.7 环境
```bash
conda create -n carla37 python=3.7 -y
conda activate carla37
pip install ~/qch_ws/carla/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
pip install numpy pygame
```

### 8. 创建非 root 用户
```bash
useradd -m -s /bin/bash carla
chmod 755 /root /root/qch_ws /root/qch_ws/carla
chmod -R 755 ~/qch_ws/carla
```

### 9. 启动 CARLA
```bash
# 后台启动
su - carla -c 'cd /root/qch_ws/carla && ./CarlaUE4.sh -RenderOffScreen -nosound' &

# 等待启动
sleep 30

# 测试连接
conda activate carla37
python -c "import carla; c = carla.Client('localhost', 2000); c.set_timeout(10); print('SUCCESS:', c.get_server_version())"
```

### 10. 停止 CARLA
```bash
pkill -9 CarlaUE4
```

---

## 快速脚本

### setup_vastai.sh
一次性环境设置脚本，见 `scripts/setup_vastai.sh`

### run_carla_vastai.sh
启动 CARLA 并测试连接，见 `scripts/run_carla_vastai.sh`

---

## 常见警告（可忽略）

```
sh: 1: xdg-user-dir: not found
```
无影响，只是找不到用户目录命令。

```
WARNING: Running pip as the 'root' user...
```
警告而已，conda 环境是隔离的。

---

## Vast.ai 特有提示

### 禁用自动 tmux
```bash
touch ~/.no_auto_tmux
```

### tmux 滚动
- `Ctrl+b` 然后 `[` 进入滚动模式
- 方向键或 PageUp/PageDown 滚动
- `q` 退出

---

## 验证成功的输出

### 进程检查
```bash
$ ps aux | grep CarlaUE4
root        6984  0.0  0.0   6752  3248 pts/5    S    18:24   0:00 su - carla -c cd /root/qch_ws/carla && ./CarlaUE4.sh -RenderOffScreen -nosound
carla       6986  0.0  0.0   2616   592 ?        Ss   18:24   0:00 /bin/sh ./CarlaUE4.sh -RenderOffScreen -nosound
carla       6995  285  6.0 12347628 3982060 ?    Sl   18:24  13:19 /root/qch_ws/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -RenderOffScreen -nosound
```

### Python 连接测试
```bash
$ python -c "import carla; c = carla.Client('localhost', 2000); c.set_timeout(10); print('SUCCESS:', c.get_server_version())"
SUCCESS: 0.9.15
```

---

## 确认可用的配置

| 项目 | 值 |
|------|-----|
| 平台 | Vast.ai |
| GPU | 2x NVIDIA GeForce RTX 3090 |
| Driver | **535.54.03** |
| CUDA | 12.2 |
| Ubuntu | 20.04 |
| CARLA | 0.9.15 |
| Python | 3.7 (conda env: carla37) |

---

## 当前目录结构

```
~/qch_ws/
├── carla/                          # CARLA 0.9.15
│   ├── CarlaUE4.sh                 # 启动脚本
│   ├── CarlaUE4/                   # 主程序
│   ├── PythonAPI/                  # Python API
│   └── CARLA_0.9.15.tar.gz         # 下载的压缩包
├── Bench2Drive/                    # 主仓库
│   ├── CARLA -> ../carla/          # symlink
│   ├── Bench2DriveZoo -> ../Bench2DriveZoo/  # symlink
│   ├── leaderboard/                # 评估代码
│   ├── scenario_runner/            # 场景运行器
│   ├── scripts/                    # 脚本
│   └── tools/                      # 工具
└── Bench2DriveZoo/                 # 模型仓库
    ├── adzoo/                      # 模型代码
    └── team_code/                  # Agent 代码
```

---

## 快速命令参考

```bash
# 启动 CARLA (后台)
su - carla -c 'cd /root/qch_ws/carla && ./CarlaUE4.sh -RenderOffScreen -nosound' &

# 检查进程
ps aux | grep CarlaUE4

# 测试连接
conda activate carla37
python -c "import carla; c = carla.Client('localhost', 2000); c.set_timeout(10); print('SUCCESS:', c.get_server_version())"

# 停止 CARLA
pkill -9 CarlaUE4
```

---

## 参考链接

- [CARLA Issue #9049 - Root privileges](https://github.com/carla-simulator/carla/issues/9049)
- [CARLA Issue #8369 - RenderThread timeout](https://github.com/carla-simulator/carla/issues/8369)
- [Bench2Drive Issue #37 - Root user](https://github.com/Thinklab-SJTU/Bench2Drive/issues/37)
