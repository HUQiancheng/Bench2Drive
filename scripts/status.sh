#!/bin/bash
# 服务器状态检查脚本
# 快速查看当前环境状态

echo "=========================================="
echo "       Bench2Drive 服务器状态检查"
echo "=========================================="
echo ""

echo "=== GPU 状态 ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""

echo "=== CARLA 进程 ==="
if ps aux | grep -v grep | grep CarlaUE4 > /dev/null; then
    ps aux | grep CarlaUE4 | grep -v grep
    echo "状态: ✓ CARLA 运行中"
else
    echo "状态: ✗ CARLA 未运行"
fi
echo ""

echo "=== 端口占用 ==="
lsof -i:2000 2>/dev/null || echo "端口 2000: 空闲"
echo ""

echo "=== 文件检查 ==="
echo -n "CARLA: "
if [ -f "/root/autodl-tmp/carla/CarlaUE4.sh" ]; then
    echo "✓ 已安装"
else
    echo "✗ 未安装或未解压"
fi

echo -n "Bench2Drive: "
if [ -d "/root/autodl-tmp/Bench2Drive" ]; then
    echo "✓ 已克隆"
else
    echo "✗ 未克隆"
fi

echo -n "Bench2DriveZoo: "
if [ -d "/root/autodl-tmp/Bench2Drive/Bench2DriveZoo" ]; then
    cd /root/autodl-tmp/Bench2Drive/Bench2DriveZoo
    branch=$(git branch --show-current 2>/dev/null)
    echo "✓ 已克隆 (分支: $branch)"
else
    echo "✗ 未克隆"
fi
echo ""

echo "=== Screen 会话 ==="
screen -ls 2>/dev/null || echo "无 screen 会话"
echo ""

echo "=== Vulkan 检查 ==="
vulkaninfo 2>/dev/null | head -5 || echo "vulkaninfo 不可用"
echo ""

echo "=========================================="
