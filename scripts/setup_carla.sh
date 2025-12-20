#!/bin/bash
# CARLA 环境一次性设置
# 功能: 解压CARLA、创建用户、创建Python环境

set -e
CARLA_ROOT=/root/autodl-tmp/carla

echo "========================================"
echo "  CARLA 环境设置"
echo "========================================"

# 1. 解压 CARLA（如果需要）
echo ""
echo "[1/4] 检查 CARLA..."
if [ ! -f "$CARLA_ROOT/CarlaUE4.sh" ]; then
    if [ -f "$CARLA_ROOT/CARLA_0.9.15.tar.gz" ]; then
        echo "解压 CARLA..."
        cd $CARLA_ROOT && tar -xzf CARLA_0.9.15.tar.gz
    else
        echo "错误: 找不到 CARLA 文件"
        exit 1
    fi
else
    echo "CARLA 已就绪"
fi

# 2. 创建 carla 用户
echo ""
echo "[2/4] 设置用户..."
if ! id "carla" &>/dev/null; then
    useradd -m -s /bin/bash carla
    echo "已创建用户 carla"
else
    echo "用户 carla 已存在"
fi

# 修改权限让 carla 用户可访问
chmod 755 /root /root/autodl-tmp
chmod -R 755 $CARLA_ROOT

# 3. 创建 Python 3.7 环境
echo ""
echo "[3/4] 设置 Python 环境..."
if ! conda env list 2>/dev/null | grep -q carla37; then
    echo "创建 carla37 环境..."
    conda create -n carla37 python=3.7 -y

    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate carla37
    pip install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
    pip install numpy pygame
    conda deactivate
else
    echo "carla37 环境已存在"
fi

# 4. 创建启动脚本
echo ""
echo "[4/4] 创建启动脚本..."
cat > /home/carla/start_carla.sh << 'EOF'
#!/bin/bash
cd /root/autodl-tmp/carla
./CarlaUE4.sh -RenderOffScreen -opengl -nosound -carla-port=2000 "$@"
EOF
chmod +x /home/carla/start_carla.sh
chown carla:carla /home/carla/start_carla.sh

echo ""
echo "========================================"
echo "  设置完成!"
echo "========================================"
echo ""
echo "下一步: bash scripts/run_carla.sh"
