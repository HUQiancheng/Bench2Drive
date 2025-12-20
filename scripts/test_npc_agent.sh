#!/bin/bash
# NpcAgent 评估测试脚本
# 在 AutoDL 服务器上运行

set -e

# 环境变量
export CARLA_ROOT=/root/autodl-tmp/carla
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:$PYTHONPATH

# Bench2Drive 路径 (根据实际位置调整)
B2D_ROOT=/root/autodl-tmp/Bench2Drive
cd $B2D_ROOT

echo "=== NpcAgent 评估测试 ==="
echo "Bench2Drive: $B2D_ROOT"
echo "CARLA: $CARLA_ROOT"
echo ""

# 确保 CARLA 在运行
if ! ps aux | grep -v grep | grep CarlaUE4 > /dev/null; then
    echo "CARLA 未运行，先启动..."
    cd $CARLA_ROOT
    ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 &
    echo "等待 CARLA 启动..."
    sleep 30
    cd $B2D_ROOT
fi

echo "开始评估..."
mkdir -p results

python leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=leaderboard/data/drivetransformer_bench2drive_dev10.xml \
    --routes-subset=0 \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint=results/npc_test.json \
    --agent=leaderboard/leaderboard/autoagents/npc_agent.py \
    --agent-config="" \
    --port=2000 \
    --timeout=600

echo ""
echo "=== 评估完成 ==="
echo "结果文件: results/npc_test.json"
cat results/npc_test.json | head -30
