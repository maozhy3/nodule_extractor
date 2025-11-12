# 肺结节尺寸提取工具

中文 | [English](README_EN.md)

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

### 配置文件优先级

本项目支持多层配置覆盖机制，优先级从低到高：

```
config.py (默认配置)
    ↓ 覆盖
config_private.py (私有配置) ⭐ 推荐
    ↓ 覆盖
外部 config.py (打包后)
```

### 配置文件说明

#### 1. config.py（默认配置）
- **位置**：项目根目录
- **用途**：项目的默认配置，提交到 git
- **适用场景**：团队共享的默认设置

#### 2. config_private.py（私有配置）⭐ 推荐
- **位置**：项目根目录
- **用途**：个人的私有配置，**不会提交到 git**
- **适用场景**：
  - 个人的模型路径
  - 个人的硬件配置（CPU线程数、GPU层数）
  - 测试用的特殊配置

#### 3. 外部 config.py（打包后）
- **位置**：exe 文件同级目录
- **用途**：打包后的外部配置
- **适用场景**：分发给用户后，用户可以修改配置而不需要重新打包

### 快速开始：创建私有配置

**推荐使用 `config_private.py` 进行个人配置**，避免修改 `config.py` 导致 git 冲突。

1. **复制示例文件**：
   ```bash
   copy config_private.py.example config_private.py
   ```

2. **编辑 config_private.py**（只需要写你想覆盖的配置项）：
   ```python
   # 启用特征提取
   ENABLE_FEATURE_EXTRACTION = True
   
   # 使用你的模型路径
   MODEL_PATHS = [
       str(_ROOT / "models" / "qwen-medical-lora-251106-q4_k_m.gguf")
   ]
   
   # 你的硬件配置
   LLAMA_N_THREADS = 16
   LLAMA_N_GPU_LAYERS = 33
   ```

3. **运行程序**：
   ```bash
   python main.py
   ```
   
   程序会自动加载你的私有配置，并显示：
   ```
   ✓ 已加载私有配置: config_private.py
   ```

### 主要配置项

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

# 特征提取配置（新增）
ENABLE_FEATURE_EXTRACTION = False  # 是否启用特征提取
FEATURE_EXTRACTION_MODEL_PATH = None  # 特征提取使用的模型
```

### 性能建议

- 低配机器（4核8GB）：`PROCESS_POOL_MAX_WORKERS = 1, LLAMA_N_THREADS = 4`
- 中配机器（8核16GB）：`PROCESS_POOL_MAX_WORKERS = 2, LLAMA_N_THREADS = 4`
- 高配机器（16核32GB）：`PROCESS_POOL_MAX_WORKERS = 4, LLAMA_N_THREADS = 4`

### 配置示例

#### 示例1：开发者的个人配置
```python
# config_private.py

# 我的测试数据
EXCEL_PATH = _ROOT / "my_test.xlsx"

# 我的 GPU 配置
LLAMA_N_THREADS = 4
LLAMA_N_GPU_LAYERS = 33

# 启用特征提取
ENABLE_FEATURE_EXTRACTION = True
```

#### 示例2：快速测试配置
```python
# config_private.py

# 小数据集测试
EXCEL_PATH = _ROOT / "test_5_samples.xlsx"

# 快速检查点
CHECKPOINT_SAVE_INTERVAL = 5

# 详细日志
LLAMA_VERBOSE = True
```

### 注意事项

✅ **推荐做法**：
- 使用 `config_private.py` 存储个人配置
- 只在 `config_private.py` 中写需要覆盖的配置项
- 定期更新 `config_private.py.example` 作为模板

❌ **不推荐做法**：
- 直接修改 `config.py` 进行个人配置（会影响 git 提交）
- 将 `config_private.py` 提交到 git（已在 .gitignore 中排除）

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

### 前置要求

1. **Windows 操作系统**（打包功能仅支持 Windows）
2. **Conda 环境**：已安装并激活 `node_extractor` 环境
3. **PyInstaller**：已在环境中安装
4. **VC++ 运行库**：`_vcredist/vc_redist.x64.exe`（首次运行时自动安装）

### 打包方式

#### 方式一：GUI 版本（推荐）

GUI 版本提供图形界面，适合大多数用户。

```bash
# 运行打包脚本
build_gui.bat
```

**输出目录**：`dist/医疗影像报告预测工具/`

**包含文件**：
- `医疗影像报告预测工具.exe` - 主程序（GUI）
- `_internal/` - 依赖库
- `models/` - 模型文件目录（需手动复制模型）
- `_vcredist/` - VC++ 运行库
- `test.xlsx` - 示例数据文件
- `config_example.py` - 配置示例
- `config_private.py.example` - 私有配置模板

#### 方式二：命令行版本

命令行版本适合自动化脚本或服务器环境。

```bash
# 运行打包脚本
build_main.bat
```

**输出目录**：`dist/医疗影像报告预测工具-CLI/`

### 打包配置文件

#### gui.spec

GUI 版本的 PyInstaller 配置文件。

**关键配置**：
- `console=False` - 不显示控制台窗口
- 包含 `config.py`、`config_loader.py`、`core.py` 等核心模块
- 自动收集 llama_cpp 的 DLL 文件

#### main.spec

命令行版本的 PyInstaller 配置文件。

**关键配置**：
- `console=True` - 显示控制台窗口
- 包含所有核心模块和依赖

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

打包后的程序支持三层配置覆盖：

```
内置 config.py (打包时包含)
    ↓ 覆盖
config_private.py (如果存在)
    ↓ 覆盖
外部 config.py (exe 同级目录)
```

### 打包后测试

#### 测试清单

- [ ] 程序能正常启动
- [ ] 能加载模型文件
- [ ] 能读取 Excel 文件
- [ ] 能正常预测并保存结果
- [ ] 停止功能正常工作
- [ ] 检查点功能正常（中断后继续）
- [ ] 特征提取功能正常（如果启用）
- [ ] 外部配置文件能正常加载

#### 测试步骤

1. **基础功能测试**：
   - 双击运行 exe
   - 选择输入文件（test.xlsx）
   - 选择模型文件
   - 点击"开始预测"
   - 等待完成，检查输出文件

2. **停止和继续测试**：
   - 开始预测
   - 点击"停止"按钮
   - 检查是否保存了检查点
   - 再次运行，确认从断点继续

3. **外部配置测试**：
   - 在 exe 同级目录创建 config.py
   - 修改配置（如线程数）
   - 运行程序，确认配置生效

### 常见打包问题

#### 1. 打包失败：找不到模块

**解决方案**：检查 `.spec` 文件中的 `hiddenimports` 是否包含所有必要模块。

当前必需模块：`pandas`、`openpyxl`、`llama_cpp`、`tqdm`、`tkinter`、`config_loader`、`core`

#### 2. 运行时找不到 DLL

**解决方案**：
- 确保 `llama_cpp` 的 DLL 文件被正确收集
- 检查 `.spec` 文件中的 `binaries` 配置
- 确保 VC++ 运行库已安装

#### 3. 配置文件不生效

**解决方案**：
- 检查配置文件名是否正确（`config.py` 或 `config_private.py`）
- 检查配置文件位置（exe 同级目录）
- 查看程序启动日志，确认是否加载了配置

#### 4. 模型加载失败

**解决方案**：
- 确保模型文件在 `models/` 目录下
- 检查模型文件路径配置
- 确认模型文件格式正确（.gguf）

---

## 开发指南

### 安装开发依赖

```bash
pip install -r requirements-dev.txt
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

### v1.2.0 (2024-11-12)
- ✅ 优化打包流程：打包时不再自动复制所有模型文件
- ✅ 改进用户体验：用户可根据需要手动选择要部署的模型
- ✅ 减小发布包体积：避免打包不必要的大型模型文件

### v1.1.0 (2024-11-10)
- ✅ 添加标准依赖管理（requirements.txt）
- ✅ 规范化代码风格和类型注解
- ✅ 重构配置加载逻辑
- ✅ 添加完整单元测试（52个测试用例）
- ✅ 改进文档和开发工具

### v1.0.0 (2024-11-06)
- ✅ 初始版本发布
- ✅ GUI 和命令行界面
- ✅ 批量预测和断点续传
- ✅ 多进程并行处理

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

**免责声明**：本工具仅供学习和研究使用，不应用于临床诊断。使用者需自行承担使用本工具的风险和责任。
