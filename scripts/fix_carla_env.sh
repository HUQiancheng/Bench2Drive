#!/bin/bash
# 修复 CARLA 运行环境
# 解决: root权限问题 + Python版本问题 + Vulkan环境

set -e

CARLA_ROOT=/root/autodl-tmp/carla
CARLA_USER_HOME=/home/carla
CARLA_USER_DIR=$CARLA_USER_HOME/carla

echo "=========================================="
echo "  CARLA 环境修复脚本"
echo "=========================================="

# ============================================
# Step 1: 创建 carla 用户
# ============================================
echo ""
echo "=== Step 1: 创建 carla 用户 ==="
if ! id "carla" &>/dev/null; then
    useradd -m -s /bin/bash carla
    echo "创建用户 carla"
else
    echo "用户 carla 已存在"
fi

# ============================================
# Step 2: 复制 CARLA 到 carla 用户目录
# ============================================
echo ""
echo "=== Step 2: 复制 CARLA 到 carla 用户目录 ==="
if [ ! -d "$CARLA_USER_DIR" ]; then
    echo "复制 CARLA... (这需要几分钟)"
    cp -r $CARLA_ROOT $CARLA_USER_DIR
    chown -R carla:carla $CARLA_USER_DIR
    echo "完成"
else
    echo "CARLA 已在 $CARLA_USER_DIR"
fi

# ============================================
# Step 3: 设置 Vulkan 环境变量
# ============================================
echo ""
echo "=== Step 3: 配置环境变量 ==="
cat > $CARLA_USER_HOME/.carla_env << 'ENVEOF'
export CARLA_ROOT=/home/carla/carla
export XDG_RUNTIME_DIR=/tmp/runtime-carla
export SDL_VIDEODRIVER=offscreen
export DISPLAY=
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
ENVEOF
chown carla:carla $CARLA_USER_HOME/.carla_env
echo "环境变量已写入 $CARLA_USER_HOME/.carla_env"

# ============================================
# Step 4: 创建 Python 3.7 环境 (用于 CARLA)
# ============================================
echo ""
echo "=== Step 4: 创建 Python 3.7 conda 环境 ==="
if ! conda env list | grep -q "carla37"; then
    echo "创建 carla37 环境..."
    conda create -n carla37 python=3.7 -y
    echo "安装 carla wheel..."

    # 激活环境并安装
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate carla37
    pip install $CARLA_USER_DIR/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
    pip install numpy pygame
    conda deactivate
else
    echo "carla37 环境已存在"
fi

# ============================================
# Step 5: 创建启动脚本
# ============================================
echo ""
echo "=== Step 5: 创建启动脚本 ==="

# CARLA 服务器启动脚本 (以 carla 用户运行)
cat > $CARLA_USER_HOME/start_carla.sh << 'SCRIPTEOF'
#!/bin/bash
source ~/.carla_env
cd $CARLA_ROOT
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000 -nosound "$@"
SCRIPTEOF
chmod +x $CARLA_USER_HOME/start_carla.sh
chown carla:carla $CARLA_USER_HOME/start_carla.sh

echo "启动脚本已创建: $CARLA_USER_HOME/start_carla.sh"

# ============================================
# Step 6: 验证
# ============================================
echo ""
echo "=== Step 6: 验证配置 ==="
echo "CARLA 目录:"
ls -la $CARLA_USER_DIR/CarlaUE4.sh

echo ""
echo "Python wheels:"
ls $CARLA_USER_DIR/PythonAPI/carla/dist/*.whl

echo ""
echo "Conda 环境:"
conda env list | grep carla37 || echo "carla37 环境未找到"

echo ""
echo "=========================================="
echo "  配置完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  1. 启动 CARLA:  su - carla -c 'bash start_carla.sh' &"
echo "  2. 测试连接:    conda activate carla37 && python -c 'import carla; print(carla)'"
echo ""
echo "下一步运行: bash scripts/test_carla.sh"
