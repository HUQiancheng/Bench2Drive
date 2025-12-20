# 任务跟踪

## 当前状态

**日期**: 2024-12-20
**阶段**: 环境配置中

---

## 已完成 ✅

### 基础设施
- [x] AutoDL 服务器开通 (2x RTX 3090)
- [x] CARLA 0.9.15 下载完成 (`/root/autodl-tmp/carla/CARLA_0.9.15.tar.gz`)
- [x] Bench2Drive 仓库克隆完成
- [x] Bench2DriveZoo 仓库克隆完成

### 文档
- [x] 创建 `docs/SERVER_DEPLOYMENT_GUIDE.md` - 完整服务器部署指南
- [x] 更新 `CLAUDE.md` - 添加文档引用
- [x] 创建 `.tasks/ENVIRONMENT.md` - 环境说明
- [x] 创建测试脚本 (`scripts/setup_carla.sh`, `test_carla.sh`, `test_npc_agent.sh`)

### 本地研究
- [x] 分析 Bench2Drive 代码结构
- [x] 分析 Agent 接口 (AutonomousAgent)
- [x] 对比 UniAD/VAD/TCP/ADMLP 模型差异
- [x] 确定 TCP 作为视觉模型测试模板

---

## 进行中 🔄

### 服务器环境配置
- [ ] 解压 CARLA (`bash scripts/setup_carla.sh`)
- [ ] 验证 Vulkan 配置
- [ ] 测试 CARLA 启动 (`bash scripts/test_carla.sh`)

---

## 待完成 📋

### 环境验证 (优先级: 高)
- [ ] CARLA Python 连接测试
- [ ] NpcAgent 评估测试 (`bash scripts/test_npc_agent.sh`)
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

## 阻塞问题 ⚠️

### 当前无阻塞

---

## 下次会话待办

1. 确认服务器上脚本执行结果
2. 根据输出调试问题
3. 继续推进 NpcAgent 测试
4. 下载 TCP checkpoint 并测试

---

## 笔记

### CARLA 注意事项
- 驱动 580 比较新，可能有兼容性问题，需要测试
- 无头模式必须用 `-RenderOffScreen`
- GPU 选择用 `-graphicsadapter=N`，不是 `CUDA_VISIBLE_DEVICES`

### 模型选择
- **ADMLP**: 最轻量，但不用图像，不适合视觉模型测试
- **TCP**: 用 ResNet 处理图像，适合作为视觉模型模板
- **UniAD/VAD**: 太大，需要大量显存

### 评估路线
- `bench2drive220.xml`: 完整 220 条路线
- `drivetransformer_bench2drive_dev10.xml`: 10 条路线，用于快速测试
