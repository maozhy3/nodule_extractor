# 肺结节尺寸提取工具

基于 llama.cpp 和 Qwen 医疗模型的肺结节尺寸自动提取工具，支持从影像报告中智能提取病灶最大直径。

> **平台支持**：目前仅提供 Windows 平台支持。Linux/Mac 用户可以从源码运行，但打包的 exe 版本仅适用于 Windows。

## 功能特性

- 🖥️ 图形化界面（GUI）和命令行两种使用方式
- 🔍 智能文本预处理和数值提取
- 🚀 支持单进程/多进程并行推理
- 💾 断点续传（支持中断后继续处理）
- 📊 批量处理 Excel 数据
- ⚙️ 灵活的配置管理
- 📦 支持打包成独立 exe

---

## 快速开始

### 1. 安装 VC++ 运行库（Windows 必需）

下载并安装：https://aka.ms/vs/17/release/vc_redist.x64.exe

或放置到 `_vcredist/` 目录，程序会自动安装。

### 2. 安装 Python 依赖

```bash
# 创建虚拟环境（推荐）
conda create -n lung-nodule python=3.10
conda activate lung-nodule

# 安装依赖
pip install -r requirements.txt
```

### 3. 下载模型

> **注意**：当前 Release 版本不包含模型文件，需要单独下载。

```bash
python download_model.py
```

推荐模型：
- **Qwen2.5-3B** (~2GB) - 快速，适合测试
- **Qwen2.5-7B** (~4.7GB) - 效果更好，推荐使用

### 4. 运行程序

**图形界面（推荐）**：
```bash
python gui.py
```

**命令行**：
```bash
python main.py
```

---

## 配置说明

编辑 `config.py` 修改配置：

```python
# 输入输出
EXCEL_PATH = "tests/test.xlsx"           # 输入文件
OUTPUT_PATH = "tests/test_results.xlsx"  # 输出文件

# 模型路径
MODEL_PATHS = ["models/qwen2.5-7b-instruct-q4_k_m.gguf"]

# 性能配置
LLAMA_N_THREADS = 8               # CPU 线程数
LLAMA_N_GPU_LAYERS = 0            # GPU 层数（0=纯CPU）
PROCESS_POOL_MAX_WORKERS = 1      # 进程数（1=单进程）
CHECKPOINT_SAVE_INTERVAL = 5000   # 检查点保存间隔
```

**性能建议**：
- 低配机器（4核8GB）：`PROCESS_POOL_MAX_WORKERS = 1, LLAMA_N_THREADS = 4`
- 中配机器（8核16GB）：`PROCESS_POOL_MAX_WORKERS = 2, LLAMA_N_THREADS = 4`
- 高配机器（16核32GB）：`PROCESS_POOL_MAX_WORKERS = 4, LLAMA_N_THREADS = 4`

---

## 使用说明

### 数据格式

Excel 文件需包含 `yxbx` 列（影像表现）：

| yxbx |
|------|
| 右肺上叶见结节影，大小约8mm |
| 左肺下叶磨玻璃影，直径1.2cm |

### 结果输出

新增 `pred_模型名` 列：

| yxbx | pred_qwen2.5-7b |
|------|-----------------|
| 右肺上叶见结节影，大小约8mm | 8 |
| 左肺下叶磨玻璃影，直径1.2cm | 12 |

### 断点续传

- 按 `Ctrl+C` 中断，检查点自动保存到 `checkpoints/`
- 再次运行会从上次中断处继续
- 完成后检查点自动清除

---

## 打包部署

> **注意**：打包功能仅支持 Windows 平台。

### 打包成 exe

```bash
# 运行打包脚本（仅 Windows）
build_gui.bat
```

打包后目录结构：
```
dist/医疗影像报告预测工具/
├── 医疗影像报告预测工具.exe
├── _internal/              # 依赖库
├── models/                 # 模型文件（需手动复制）
└── _vcredist/              # VC++运行库（需手动复制）
```

### 部署到离线电脑

> **重要**：
> - Release 版本不包含模型文件，需要手动下载并放入 `models/` 目录
> - 仅支持 Windows 平台

1. 将 `dist/医疗影像报告预测工具/` 整个文件夹复制到目标 Windows 电脑
2. **手动下载模型文件**并放入 `models/` 目录（Release 不包含）
3. 确保 `_vcredist/vc_redist.x64.exe` 存在
4. 双击 `医疗影像报告预测工具.exe` 运行

### 外部配置

在 exe 同级目录创建 `config.py` 可覆盖默认配置，无需重新打包。

---

## 开发指南

### 安装开发依赖

```bash
pip install -r requirements-dev.txt
```

### 运行测试

```bash
# Windows
cd tests
run_tests.bat

# Linux/Mac
cd tests
./run_tests.sh

# 或直接使用 pytest
pytest tests/ -v
```

### 代码格式化

```bash
# 自动格式化
format_code.bat

# 或手动
black *.py tests/
ruff check . --fix
```

### 项目结构

```
.
├── main.py              # 命令行入口
├── gui.py               # GUI 入口
├── core.py              # 核心逻辑
├── config.py            # 配置文件
├── config_loader.py     # 配置加载
├── download_model.py    # 模型下载
├── requirements.txt     # 核心依赖
├── requirements-dev.txt # 开发依赖
├── tests/               # 测试文件（单元测试、测试数据、测试脚本）
├── models/              # 模型目录
└── checkpoints/         # 检查点目录
```

---

## 常见问题

**Q: 安装 llama-cpp-python 失败？**  
A: 确保已安装 VC++ 运行库

**Q: 内存不足？**  
A: 设置 `PROCESS_POOL_MAX_WORKERS = 1`

**Q: 如何使用 GPU？**  
A: 安装 CUDA 版本的 llama-cpp-python，设置 `LLAMA_N_GPU_LAYERS = 35`

**Q: 检查点文件可以删除吗？**  
A: 任务完成后会自动清除，也可手动删除 `checkpoints/` 目录

**Q: 如何验证安装？**  
A: 运行 `python verify_setup.py`

---

## 更新日志

### v1.1.0 (2025-11-10)
- ✅ 添加标准依赖管理（requirements.txt）
- ✅ 规范化代码风格和类型注解
- ✅ 重构配置加载逻辑
- ✅ 添加完整单元测试（52个测试用例）
- ✅ 改进文档和开发工具

### v1.0.0 (2025-11-06)
- ✅ 初始版本发布
- ✅ GUI 和命令行界面
- ✅ 批量预测和断点续传
- ✅ 多进程并行处理

---

## 许可证

本项目仅供学习和研究使用。
