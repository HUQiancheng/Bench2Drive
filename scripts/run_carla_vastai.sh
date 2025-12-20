#!/bin/bash
# Vast.ai CARLA 启动脚本

echo "========================================"
echo "  启动 CARLA"
echo "========================================"

# 清理旧进程
echo ""
echo "[1/3] 清理旧进程..."
pkill -9 CarlaUE4 2>/dev/null || true
sleep 2

# 启动 CARLA
echo ""
echo "[2/3] 启动 CARLA..."
su - carla -c 'cd /root/qch_ws/carla && ./CarlaUE4.sh -RenderOffScreen -nosound -carla-port=2000' &

echo "等待启动 (60秒)..."
for i in {1..60}; do
    sleep 1
    echo -n "."
    if [ $((i % 20)) -eq 0 ]; then
        echo " ${i}s"
    fi
done
echo ""

# 检查状态
echo ""
echo "[3/3] 检查状态..."

echo "进程:"
if pgrep -f CarlaUE4 > /dev/null; then
    echo "  ✓ CARLA 运行中"
else
    echo "  ✗ CARLA 未运行"
    exit 1
fi

echo ""
echo "Python 连接测试:"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate carla37 2>/dev/null

python -c "
import carla
try:
    c = carla.Client('localhost', 2000)
    c.set_timeout(10)
    print('  ✓ 连接成功! Version:', c.get_server_version())
    print('  Map:', c.get_world().get_map().name)
except Exception as e:
    print('  ✗ 连接失败:', e)
"

echo ""
echo "========================================"
echo "CARLA 在后台运行"
echo "停止: pkill -9 CarlaUE4"
echo "========================================"
