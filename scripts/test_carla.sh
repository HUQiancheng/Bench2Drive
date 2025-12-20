#!/bin/bash
# CARLA 启动和连接测试脚本
# 在 AutoDL 服务器上运行

set -e

export CARLA_ROOT=/root/autodl-tmp/carla
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:$PYTHONPATH

echo "=== Step 1: 清理残留进程 ==="
pkill -9 CarlaUE4 2>/dev/null || true
pkill -9 CarlaUE4-Linux 2>/dev/null || true
sleep 2

echo ""
echo "=== Step 2: 启动 CARLA (无头模式) ==="
cd $CARLA_ROOT
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 &
CARLA_PID=$!
echo "CARLA PID: $CARLA_PID"

echo ""
echo "=== Step 3: 等待 CARLA 启动 (30秒) ==="
for i in {1..30}; do
    echo -n "."
    sleep 1
done
echo ""

echo ""
echo "=== Step 4: 检查进程 ==="
ps aux | grep CarlaUE4 | grep -v grep || echo "警告: CARLA 进程未找到!"

echo ""
echo "=== Step 5: 测试 Python 连接 ==="
python3 << 'EOF'
import carla
import sys

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    version = client.get_server_version()
    print(f"SUCCESS! CARLA Version: {version}")

    # 获取世界信息
    world = client.get_world()
    map_name = world.get_map().name
    print(f"Current Map: {map_name}")

    sys.exit(0)
except Exception as e:
    print(f"FAILED! Error: {e}")
    sys.exit(1)
EOF

echo ""
echo "=== 测试完成 ==="
echo "如果看到 'SUCCESS!'，CARLA 运行正常"
echo "下一步运行: bash scripts/test_npc_agent.sh"
