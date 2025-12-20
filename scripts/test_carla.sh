#!/bin/bash
# CARLA 启动和连接测试脚本
# 前置条件: 先运行 fix_carla_env.sh

set -e

CARLA_USER_HOME=/home/carla

echo "=========================================="
echo "  CARLA 启动测试"
echo "=========================================="

# ============================================
# Step 1: 清理
# ============================================
echo ""
echo "=== Step 1: 清理残留进程 ==="
pkill -9 CarlaUE4 2>/dev/null || true
sleep 2

# ============================================
# Step 2: 检查前置条件
# ============================================
echo ""
echo "=== Step 2: 检查环境 ==="

if [ ! -f "$CARLA_USER_HOME/start_carla.sh" ]; then
    echo "错误: 请先运行 bash scripts/fix_carla_env.sh"
    exit 1
fi

if ! conda env list | grep -q "carla37"; then
    echo "错误: carla37 环境不存在，请先运行 bash scripts/fix_carla_env.sh"
    exit 1
fi

echo "前置条件检查通过"

# ============================================
# Step 3: 启动 CARLA (以 carla 用户)
# ============================================
echo ""
echo "=== Step 3: 启动 CARLA ==="
su - carla -c "bash start_carla.sh" &
CARLA_PID=$!
echo "后台启动 CARLA, PID: $CARLA_PID"

# ============================================
# Step 4: 等待启动
# ============================================
echo ""
echo "=== Step 4: 等待 CARLA 启动 (90秒) ==="
echo "首次启动需要加载shader，可能较慢..."

for i in {1..90}; do
    echo -n "."
    sleep 1

    # 每15秒检查
    if [ $((i % 15)) -eq 0 ]; then
        echo ""
        if pgrep -u carla CarlaUE4 > /dev/null 2>&1; then
            echo "[${i}s] CARLA 进程运行中..."
        else
            echo "[${i}s] 警告: CARLA 进程未找到"
        fi
    fi
done
echo ""

# ============================================
# Step 5: 检查状态
# ============================================
echo ""
echo "=== Step 5: 检查进程状态 ==="
ps aux | grep -E "CarlaUE4" | grep -v grep || echo "未找到 CARLA 进程"

echo ""
echo "=== Step 6: 检查端口 ==="
netstat -tlnp 2>/dev/null | grep 2000 || echo "端口 2000 未监听 (可能还在启动)"

# ============================================
# Step 7: Python 连接测试
# ============================================
echo ""
echo "=== Step 7: Python 连接测试 ==="

# 激活 carla37 环境运行测试
source $(conda info --base)/etc/profile.d/conda.sh
conda activate carla37

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

import carla
print("carla 模块加载成功")

print("尝试连接 CARLA...")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)  # 首次连接给更长时间
    version = client.get_server_version()
    print(f"")
    print(f"========================================")
    print(f"  SUCCESS! CARLA 连接成功!")
    print(f"  Server Version: {version}")
    world = client.get_world()
    print(f"  Current Map: {world.get_map().name}")
    print(f"========================================")
except Exception as e:
    print(f"")
    print(f"连接失败: {e}")
    print(f"")
    print(f"可能原因:")
    print(f"  1. CARLA 还在启动中 (shader编译)")
    print(f"  2. Vulkan/GPU 驱动问题")
    print(f"")
    print(f"调试步骤:")
    print(f"  - 检查进程: ps aux | grep CarlaUE4")
    print(f"  - 查看日志: 在另一个终端运行 CARLA 看输出")
EOF

conda deactivate

echo ""
echo "=========================================="
echo "  测试完成"
echo "=========================================="
echo ""
echo "如果连接成功，下一步运行: bash scripts/test_npc_agent.sh"
echo "如果失败，尝试手动启动看日志: su - carla -c 'bash start_carla.sh'"
