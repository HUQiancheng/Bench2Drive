#!/bin/bash
# 启动 CARLA 服务器并测试连接

echo "========================================"
echo "  启动 CARLA"
echo "========================================"

# 1. 清理
echo ""
echo "[1/3] 清理旧进程..."
pkill -9 CarlaUE4 2>/dev/null || true
sleep 2

# 2. 启动
echo ""
echo "[2/3] 启动 CARLA (opengl 模式)..."
su - carla -c 'bash /home/carla/start_carla.sh' &

echo "等待启动 (约60秒)..."
for i in {1..60}; do
    sleep 1
    echo -n "."
    if [ $((i % 20)) -eq 0 ]; then
        echo " ${i}s"
    fi
done
echo ""

# 3. 检查
echo ""
echo "[3/3] 检查状态..."

echo "进程:"
if pgrep -f CarlaUE4 > /dev/null; then
    echo "  ✓ CARLA 运行中"
else
    echo "  ✗ CARLA 未运行"
    echo ""
    echo "可能原因: 查看前台输出 su - carla -c 'bash /home/carla/start_carla.sh'"
    exit 1
fi

echo ""
echo "端口:"
if ss -tlnp 2>/dev/null | grep -q ":2000 "; then
    echo "  ✓ 端口 2000 已开放"
else
    echo "  ? 端口 2000 未检测到 (可能还在初始化)"
fi

echo ""
echo "Python 连接测试:"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate carla37 2>/dev/null
python3 -c "
import carla
try:
    c = carla.Client('localhost', 2000)
    c.set_timeout(10)
    print('  ✓ 连接成功! Version:', c.get_server_version())
except Exception as e:
    print('  ✗ 连接失败:', e)
" 2>/dev/null || echo "  (需要 carla37 环境)"

echo ""
echo "========================================"
echo "CARLA 保持在后台运行"
echo "停止命令: pkill -9 CarlaUE4"
echo "========================================"
