#!/bin/bash
# CARLA 启动和连接测试脚本
# 在 AutoDL 服务器上运行

set -e

export CARLA_ROOT=/root/autodl-tmp/carla
export SDL_VIDEODRIVER=offscreen

echo "=== Step 1: 清理残留进程 ==="
pkill -9 CarlaUE4 2>/dev/null || true
pkill -9 CarlaUE4-Linux 2>/dev/null || true
sleep 2

echo ""
echo "=== Step 2: 检查环境 ==="
echo "CARLA_ROOT: $CARLA_ROOT"
echo "Python版本:"
python3 --version

echo ""
echo "=== Step 3: 检查 Vulkan ==="
if command -v vulkaninfo &> /dev/null; then
    vulkaninfo 2>&1 | head -10 || echo "Vulkan 检测失败"
else
    echo "vulkaninfo 未安装，尝试安装..."
    apt-get update && apt-get install -y vulkan-tools mesa-vulkan-drivers 2>/dev/null || true
    vulkaninfo 2>&1 | head -10 || echo "Vulkan 仍然失败"
fi

echo ""
echo "=== Step 4: 检查 GPU ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv

echo ""
echo "=== Step 5: 启动 CARLA (无头模式, root用户) ==="
cd $CARLA_ROOT

# 直接以 root 运行 (在容器环境中通常是可以的)
chmod +x CarlaUE4.sh
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 -nosound &
CARLA_PID=$!
echo "CARLA PID: $CARLA_PID"

echo ""
echo "=== Step 6: 等待 CARLA 启动 (60秒) ==="
for i in {1..60}; do
    echo -n "."
    sleep 1
    # 每10秒检查一次进程
    if [ $((i % 10)) -eq 0 ]; then
        if ! ps -p $CARLA_PID > /dev/null 2>&1; then
            echo ""
            echo "警告: CARLA 进程已退出!"
            echo "查看可能的错误日志..."
            dmesg | tail -20 2>/dev/null || true
            break
        fi
    fi
done
echo ""

echo ""
echo "=== Step 7: 检查进程状态 ==="
ps aux | grep -E "CarlaUE4|carla" | grep -v grep || echo "CARLA 进程未找到"

echo ""
echo "=== Step 8: 检查端口 ==="
netstat -tlnp 2>/dev/null | grep 2000 || ss -tlnp | grep 2000 || echo "端口 2000 未监听"

echo ""
echo "=== Step 9: 测试 Python 连接 ==="
# 使用 pip 安装的 carla 而不是 egg
python3 << 'EOF'
import sys
print(f"Python: {sys.version}")
print(f"Path: {sys.path[:3]}...")

try:
    import carla
    print(f"carla 模块加载成功")
except ImportError as e:
    print(f"carla 导入失败: {e}")
    print("尝试安装 carla wheel...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install",
                   "/root/autodl-tmp/carla/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl"],
                   check=True)
    import carla

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    version = client.get_server_version()
    print(f"SUCCESS! CARLA Version: {version}")

    world = client.get_world()
    map_name = world.get_map().name
    print(f"Current Map: {map_name}")

except Exception as e:
    print(f"连接失败: {e}")
    print("CARLA 可能未成功启动，请检查上面的日志")
EOF

echo ""
echo "=== 测试完成 ==="
