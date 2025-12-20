#!/bin/bash
# CARLA 解压和环境配置脚本
# 在 AutoDL 服务器上运行

set -e

echo "=== Step 1: 解压 CARLA ==="
cd /root/autodl-tmp/carla
if [ ! -f "CarlaUE4.sh" ]; then
    echo "解压 CARLA..."
    tar -xzf CARLA_0.9.15.tar.gz
else
    echo "CARLA 已解压"
fi
ls -la CarlaUE4.sh

echo ""
echo "=== Step 2: 设置环境变量 ==="
export CARLA_ROOT=/root/autodl-tmp/carla
echo "CARLA_ROOT=$CARLA_ROOT"

# 添加到 bashrc (如果还没有)
if ! grep -q "CARLA_ROOT" ~/.bashrc; then
    echo 'export CARLA_ROOT=/root/autodl-tmp/carla' >> ~/.bashrc
    echo 'export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:$PYTHONPATH' >> ~/.bashrc
    echo "已添加到 ~/.bashrc"
fi

echo ""
echo "=== Step 3: 检查 Vulkan ==="
vulkaninfo 2>/dev/null | head -20 || echo "vulkaninfo 不可用，可能需要安装: apt install vulkan-tools"

echo ""
echo "=== Step 4: 检查 CARLA 文件 ==="
ls -la $CARLA_ROOT/CarlaUE4.sh
ls $CARLA_ROOT/PythonAPI/carla/dist/

echo ""
echo "=== 完成! ==="
echo "下一步运行: bash scripts/test_carla.sh"
