# 任务跟踪

## 当前状态

**日期**: 2024-12-21
**阶段**: CARLA 环境验证成功，准备测试评估

---

## 已完成 ✅

### 基础设施
- [x] ~~AutoDL 服务器~~ (Driver 580 不兼容，已放弃)
- [x] **Vast.ai 服务器配置成功**
  - 2x RTX 3090, Driver 535.54.03, CUDA 12.2, Ubuntu 20.04
- [x] CARLA 0.9.15 下载并运行成功
- [x] Python 3.7 环境 (carla37) 创建成功
- [x] CARLA Python 连接测试通过 (`SUCCESS: 0.9.15`)

### 文档
- [x] 创建 `scripts/VASTAI_SETUP.md` - Vast.ai 完整配置指南
- [x] 创建 `scripts/setup_vastai.sh` - 一次性设置脚本
- [x] 创建 `scripts/run_carla_vastai.sh` - CARLA 启动脚本
- [x] 更新 `.tasks/ENVIRONMENT.md` - 环境说明

### 问题解决记录
- [x] 解决 "Refusing to run with root privileges" - 创建非 root 用户
- [x] 解决 "GameThread timed out waiting for RenderThread" - 选择 Driver 535
- [x] 解决 "Vulkan 无法识别 NVIDIA GPU" - 手动创建 nvidia_icd.json

---

## 进行中 🔄

### Bench2Drive 测试
- [ ] 克隆 Bench2Drive 到 Vast.ai 服务器
- [ ] 克隆 Bench2DriveZoo (模型仓库)
- [ ] 运行 NpcAgent 评估测试

---

## 待完成 📋

### 环境验证 (优先级: 高)
- [ ] NpcAgent 评估测试
- [ ] 确认整个评估流程通畅

### 模型测试 (优先级: 中)
- [ ] 下载 TCP checkpoint
- [ ] 切换 Bench2DriveZoo 到 `tcp/admlp` 分支
- [ ] 配置 TCP agent 环境
- [ ] 运行 TCP 评估测试

### 自定义模型接入 (优先级: 后续)
- [ ] 理解 TCP agent 代码结构
- [ ] 编写自定义模型 agent 模板
- [ ] 测试自定义模型评估

---

## 关键发现 ⚠️

### NVIDIA Driver 兼容性
| Driver 版本 | CARLA 0.9.15 兼容性 |
|-------------|---------------------|
| 470.x | ✓ 兼容 |
| 535.x | ✓ **已验证可用** |
| 550+ | ✗ 不兼容 (RenderThread 崩溃) |
| 570+ | ✗ 不兼容 |
| 580+ | ✗ 不兼容 |

### 平台选择
- **AutoDL**: Driver 580，不兼容 CARLA，已放弃
- **Vast.ai**: 可选择 Driver 版本，**推荐使用**

---

## 当前服务器信息

**Vast.ai Instance**
```
GPU: 2x NVIDIA GeForce RTX 3090
Driver: 535.54.03
CUDA: 12.2
Ubuntu: 20.04
工作目录: ~/qch_ws/
CARLA: ~/qch_ws/carla/
```

---

## 下次会话待办

1. 克隆 Bench2Drive 和 Bench2DriveZoo
2. 运行 NpcAgent 评估测试
3. 测试 TCP 模型
