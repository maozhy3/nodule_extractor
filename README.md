# 肺结节尺寸提取工具

基于 llama.cpp 和 Qwen 医疗模型的肺结节尺寸自动提取工具，支持从影像报告中智能提取病灶最大直径。

## 功能特性

- 🖥️ **图形化界面**（GUI）和命令行两种使用方式
- 🔍 智能文本预处理和数值提取
- 🚀 支持单进程/多进程并行推理
- 💾 断点续传（支持中断后继续处理）
- 📊 批量处理 Excel 数据
- ⚙️ 灵活的配置管理（支持外部配置覆盖）
- 📦 支持打包成独立exe，无需Python环境

## 环境准备

### 1. VC++ 运行库（Windows 必需，优先安装）

⚠️ **重要：必须先安装 VC++ 运行库，否则 llama-cpp-python 无法安装！**

**方式一：自动安装（推荐）**
1. 下载：https://aka.ms/vs/17/release/vc_redist.x64.exe
2. 放置到项目根目录的 `_vcredist/` 文件夹
3. 程序会在首次运行时自动静默安装

目录结构：
```
项目根目录/
├── _vcredist/
│   └── VC_redist.x64.exe  ← 下载后放这里
├── main.py
└── ...
```

**方式二：手动安装**
1. 下载：https://aka.ms/vs/17/release/vc_redist.x64.exe
2. 双击运行安装
3. 安装完成后再进行后续步骤

### 2. Python 环境

推荐使用 Anaconda 或 Miniconda：

```bash
# 创建虚拟环境
conda create -n lung-nodule python=3.10
conda activate lung-nodule

# 安装依赖
pip install pandas openpyxl llama-cpp-python tqdm
```

如果 `llama-cpp-python` 安装失败，请确认已安装 VC++ 运行库。

### 3. 模型文件

将 GGUF 格式的模型文件放置到 `models/` 目录。

**推荐使用模型：**

可以使用 Qwen2.5 系列的 GGUF 量化版本：

| 模型 | 文件名 | 大小 | 下载地址 |
|------|--------|------|----------|
| Qwen2.5-3B-Instruct | `qwen2.5-3b-instruct-q4_k_m.gguf` | ~2GB | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) |
| Qwen2.5-7B-Instruct | `qwen2.5-7b-instruct-q4_k_m.gguf` | ~4.7GB | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) |

**快速下载：** 运行下载脚本自动下载模型
```bash
python download_model.py
```

```
models/
├── qwen2.5-3b-instruct-q4_k_m.gguf
└── qwen2.5-7b-instruct-q4_k_m.gguf
```

**说明：** 当前开发环境中的模型（`qwen-medical-lora-251106-*.gguf`）是作者基于医疗数据微调的版本，由于涉及隐私数据不便公开。使用 Qwen2.5 基础模型同样可以完成肺结节尺寸提取任务，推荐使用 7B 模型以获得更好的效果。

## 配置说明

### 基础配置（config.py）

```python
# 输入输出
EXCEL_PATH = "test.xlsx"           # 待预测的 Excel 文件
OUTPUT_PATH = "test_results.xlsx"  # 结果保存路径

# 模型路径
MODEL_PATHS = [
    "models/qwen2.5-3b-instruct-q4_k_m.gguf"
]

# 推理参数
LLAMA_N_CTX = 2048          # 上下文长度
LLAMA_N_THREADS = 8         # CPU 线程数
LLAMA_N_GPU_LAYERS = 0      # GPU 层数（0=纯CPU）

# 并行配置
PROCESS_POOL_MAX_WORKERS = 1  # 进程数（1=单进程，2-4=多进程）

# 检查点配置
CHECKPOINT_SAVE_INTERVAL = 5000  # 每处理多少条保存一次检查点
```

### 外部配置覆盖

可在打包后的 exe 同级目录创建 `config.py` 来覆盖默认配置，无需重新打包。

## 使用方法

### 方式一：图形化界面（推荐）

```bash
python gui.py
```

或双击 `run.bat`（如果已重命名）

界面操作：
1. 选择输入Excel文件
2. 选择输出路径
3. 添加/删除模型文件
4. 点击"开始预测"
5. 查看实时日志输出

### 方式二：命令行

```bash
python main.py
```

### 数据格式

Excel 文件需包含 `yxbx` 列（影像表现）：

| yxbx |
|------|
| 右肺上叶见结节影，大小约8mm... |
| 左肺下叶磨玻璃影，直径1.2cm... |

### 结果输出

结果会保存到指定的Excel文件，新增 `pred_模型名` 列：

| yxbx | pred_qwen2.5-3b-instruct-q4_k_m |
|------|----------------------------------|
| 右肺上叶见结节影，大小约8mm... | 8 |
| 左肺下叶磨玻璃影，直径1.2cm... | 12 |

## 断点续传

程序支持中断后继续处理：

1. 按 `Ctrl+C` 中断运行
2. 检查点自动保存到 `checkpoints/` 目录
3. 再次运行 `python main.py` 会从上次中断处继续

检查点文件：`checkpoints/{模型名}_checkpoint.pkl`

## 性能优化

### 单进程 vs 多进程

**单进程模式**（推荐内存有限时）：
```python
PROCESS_POOL_MAX_WORKERS = 1
LLAMA_N_THREADS = 8  # 使用所有 CPU 核心
```

**多进程模式**（推荐多核 CPU）：
```python
PROCESS_POOL_MAX_WORKERS = 4  # 4个进程
LLAMA_N_THREADS = 2           # 每进程2线程
# 总线程数 = 4 × 2 = 8
# 内存占用 = 模型大小 × 4
```

### GPU 加速

如果有 NVIDIA GPU 且安装了 CUDA 版本的 llama-cpp-python：

```python
LLAMA_N_GPU_LAYERS = 35  # 根据显存调整
```

## 常见问题

**Q: 提示缺少 DLL 文件？**  
A: 确保已下载并放置 `vc_redist.x64.exe` 到 `_vcredist/` 目录

**Q: 内存不足？**  
A: 设置 `PROCESS_POOL_MAX_WORKERS = 1` 使用单进程模式

**Q: 如何中断后继续？**  
A: 直接按 `Ctrl+C`，检查点会自动保存，再次运行即可继续

**Q: 检查点文件可以删除吗？**  
A: 任务完成后会自动清除，也可手动删除 `checkpoints/` 目录

## 打包部署

将程序打包成独立exe，部署到没有Python的离线Windows电脑：

### 快速打包

```bash
# 运行打包脚本
build_gui.bat
```

### 部署方式

1. **ZIP压缩包**：压缩 `dist/医疗影像报告预测工具/` 文件夹
2. **安装包**：使用 Inno Setup 编译 `installer.iss`

详细说明请查看：
- `DEPLOYMENT.md` - 完整部署指南
- `打包部署说明.txt` - 快速参考

## 项目结构

```
.
├── gui.py                  # GUI界面程序
├── main.py                 # 命令行程序
├── core.py                 # 核心业务逻辑
├── config.py               # 默认配置
├── download_model.py       # 模型下载脚本
├── gui.spec                # GUI打包配置
├── main.spec               # 命令行打包配置
├── build_gui.bat           # 打包脚本
├── installer.iss           # 安装包制作脚本
├── test.xlsx               # 输入数据
├── test_results.xlsx       # 输出结果
├── models/                 # 模型文件目录
├── checkpoints/            # 检查点目录（自动生成）
└── _vcredist/              # VC++ 运行库目录
    └── vc_redist.x64.exe   # 需手动下载放置
```

## 技术栈

- llama-cpp-python：模型推理引擎
- pandas：数据处理
- tqdm：进度显示
- multiprocessing：并行处理

## 许可证

本项目仅供学习和研究使用。