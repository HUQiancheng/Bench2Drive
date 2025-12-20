#!/bin/bash
# Vast.ai CARLA 环境一次性设置脚本
# 前提：选择了 Driver 535 或更低的机器

set -e

echo "========================================"
echo "  Vast.ai CARLA 环境设置"
echo "========================================"

# 1. 检查驱动版本
echo ""
echo "[1/8] 检查驱动版本..."
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Driver Version: $DRIVER_VERSION"

MAJOR_VERSION=$(echo $DRIVER_VERSION | cut -d'.' -f1)
if [ "$MAJOR_VERSION" -gt 545 ]; then
    echo "警告: 驱动版本 $DRIVER_VERSION 可能与 CARLA 不兼容！"
    echo "建议选择 Driver 535 或更低的机器"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. 安装依赖
echo ""
echo "[2/8] 安装依赖..."
apt update && apt install -y vulkan-tools libxext6 libx11-6

# 3. 配置 NVIDIA Vulkan ICD
echo ""
echo "[3/8] 配置 Vulkan..."
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.194"
    }
}
EOF
    echo "已创建 NVIDIA Vulkan ICD"
else
    echo "NVIDIA Vulkan ICD 已存在"
fi

# 验证 Vulkan
echo "Vulkan GPU 检测:"
vulkaninfo 2>&1 | grep -E "deviceName" | head -3 || echo "Vulkan 检测失败"

# 4. 安装 Miniconda
echo ""
echo "[4/8] 安装 Miniconda..."
if ! command -v conda &> /dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init
    rm /tmp/miniconda.sh
    echo "Miniconda 已安装"
else
    echo "Conda 已存在"
fi

# 重新加载
source ~/.bashrc 2>/dev/null || true
eval "$($HOME/miniconda3/bin/conda shell.bash hook)" 2>/dev/null || true

# 接受条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# 5. 下载 CARLA
echo ""
echo "[5/8] 下载 CARLA..."
CARLA_DIR=~/qch_ws/carla
mkdir -p $CARLA_DIR
cd $CARLA_DIR

if [ ! -f "CarlaUE4.sh" ]; then
    if [ ! -f "CARLA_0.9.15.tar.gz" ]; then
        echo "下载 CARLA 0.9.15 (约 8GB)..."
        wget -q --show-progress https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    fi
    echo "解压 CARLA..."
    tar -xzf CARLA_0.9.15.tar.gz
else
    echo "CARLA 已存在"
fi

# 6. 创建 Python 环境
echo ""
echo "[6/8] 创建 Python 3.7 环境..."
if ! conda env list | grep -q carla37; then
    conda create -n carla37 python=3.7 -y
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda activate carla37
    pip install $CARLA_DIR/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
    pip install numpy pygame
else
    echo "carla37 环境已存在"
fi

# 7. 创建非 root 用户
echo ""
echo "[7/8] 创建 carla 用户..."
if ! id "carla" &>/dev/null; then
    useradd -m -s /bin/bash carla
    echo "已创建用户 carla"
else
    echo "用户 carla 已存在"
fi

chmod 755 /root /root/qch_ws /root/qch_ws/carla
chmod -R 755 ~/qch_ws/carla 2>/dev/null || true

# 8. 创建启动脚本
echo ""
echo "[8/8] 创建启动脚本..."
cat > ~/start_carla.sh << 'EOF'
#!/bin/bash
su - carla -c 'cd /root/qch_ws/carla && ./CarlaUE4.sh -RenderOffScreen -nosound -carla-port=2000'
EOF
chmod +x ~/start_carla.sh

echo ""
echo "========================================"
echo "  设置完成!"
echo "========================================"
echo ""
echo "启动 CARLA:  ~/start_carla.sh &"
echo "测试连接:    conda activate carla37 && python -c \"import carla; print(carla.Client('localhost',2000).get_server_version())\""
echo "停止 CARLA:  pkill -9 CarlaUE4"
