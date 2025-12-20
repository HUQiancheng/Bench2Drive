#!/bin/bash
# NpcAgent 评估测试
# 前提: CARLA 已运行 (bash scripts/run_carla.sh)

set -e

CARLA_ROOT=/root/autodl-tmp/carla
B2D_ROOT=/root/autodl-tmp/Bench2Drive

echo "========================================"
echo "  NpcAgent 评估测试"
echo "========================================"

# 检查 CARLA
echo ""
echo "[1/2] 检查 CARLA..."
if ! pgrep -f CarlaUE4 > /dev/null; then
    echo "CARLA 未运行，请先执行: bash scripts/run_carla.sh"
    exit 1
fi
echo "CARLA 运行中 ✓"

# 运行评估
echo ""
echo "[2/2] 运行 NpcAgent 评估..."
cd $B2D_ROOT

source $(conda info --base)/etc/profile.d/conda.sh
conda activate carla37

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
echo "========================================"
echo "  评估完成"
echo "========================================"
echo "结果: results/npc_test.json"
