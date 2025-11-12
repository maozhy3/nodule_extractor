# config.py
import os
import sys
from pathlib import Path


def _get_base_path() -> Path:
    """获取基础路径，兼容开发和打包环境"""
    if getattr(sys, 'frozen', False):
        # 打包后的环境
        return Path(sys.executable).parent
    else:
        # 开发环境
        return Path(__file__).parent


# ===== 输入输出 =====
_ROOT = _get_base_path()
EXCEL_PATH = _ROOT / "test_results1.xlsx"  # 待预测文件
OUTPUT_PATH = _ROOT / "test_results2.xlsx"  # 结果保存

# ===== 模型列表 =====
# 支持单个字符串或列表；也可使用通配符，由脚本自动展开
MODEL_PATHS = [
    str(_ROOT / "models" / "qwen-medical-lora-251106-q4_k_m.gguf")
]
# ===== llama.cpp 推理参数 =====
LLAMA_N_CTX        = 2048   # 上下文长度
LLAMA_N_THREADS    = 8      # 每个进程内的线程数（建议：总核心数 / 进程数）
LLAMA_N_GPU_LAYERS = 0      # 0 表示纯 CPU；>0 表示 offload 到 GPU 的层数
LLAMA_VERBOSE      = False  # 是否打印 llama.cpp 的 debug 信息

# ===== 并行配置 =====
# 设置为 1 使用单进程模式（适合内存有限的情况）
# 设置为 2-4 使用多进程模式（每个进程独立加载模型，内存占用 = 模型大小 × 进程数）
# 建议：进程数 × 每进程线程数 ≈ CPU 核心数
PROCESS_POOL_MAX_WORKERS = 1  # 进程数（每个进程会独立加载一份模型）

# ===== 检查点配置 =====
CHECKPOINT_SAVE_INTERVAL = 5000  # 每处理多少条数据保存一次检查点（建议 5-20）

PROMPT_TEMPLATE = """<|im_start|>system
你是医疗信息提取助手。我需要你从影像表现的报告中，找到肺部病灶（包括肺上叶中叶下叶的结节、磨玻璃结节、团块或局灶影，不包括空洞，不包括纵隔肿物）的最大直径，可能是“长径”、“直径”、“大小”中的最大一项。如果有多个病灶，只需要最长的。返回的结果以mm为单位，如果是cm你需要进行转换，1cm=10mm。忽略CT值（HU）。  报告中其他部位和系统（如肝，肾，脾）的病灶请无视。如果报告中没有肺部病灶，或者没有具体的尺寸信息，请输出0。你的输出结果只需要输出最终的数字，不需要任何的单位或者前置描述。<|im_end|>
<|im_start|>user
影像报告：{processed_input}

请提取报告中最大的病灶尺寸（mm）：<|im_end|>
<|im_start|>assistant
"""

# ===== 特征提取配置（新增） =====
# 是否启用特征提取功能（设置为 True 启用，False 仅提取尺寸）
ENABLE_FEATURE_EXTRACTION = True

# 特征提取使用的模型路径（用于LLM兜底判断）
# 如果为 None，则使用 MODEL_PATHS 中的模型
# 如果指定了路径，则特征提取会使用该模型，而不是 MODEL_PATHS 中的模型
FEATURE_EXTRACTION_MODEL_PATH = None
# 示例：使用不同的模型进行特征提取
# FEATURE_EXTRACTION_MODEL_PATH = str(_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf")

# 是否在结果中保存目标句子（用于调试）
SAVE_TARGET_SENTENCE = False

# ===== 特征提取 Prompt 模板 =====

# 位置判断 Prompt
PROMPT_UPPER_LOBE = """<|im_start|>system
你是医疗信息提取助手。请判断以下描述中的病灶是否位于肺上叶（左上叶或右上叶）。
如果是上叶，输出"是"；如果是中叶或下叶，输出"否"；如果无法判断，输出"未知"。<|im_end|>
<|im_start|>user
{processed_input}<|im_end|>
<|im_start|>assistant
"""

# 毛刺征 Prompt
PROMPT_SPICULATION = """<|im_start|>system
你是医疗信息提取助手。请判断以下描述中的病灶是否有毛刺征。
如果有毛刺，输出"有"；如果没有或边缘光滑，输出"无"；如果无法判断，输出"未知"。<|im_end|>
<|im_start|>user
{processed_input}<|im_end|>
<|im_start|>assistant
"""

# 钙化 Prompt
PROMPT_CALCIFICATION = """<|im_start|>system
你是医疗信息提取助手。请判断以下描述中的病灶是否有钙化。
如果有钙化，输出"有"；如果没有钙化，输出"无"；如果无法判断，输出"未知"。<|im_end|>
<|im_start|>user
{processed_input}<|im_end|>
<|im_start|>assistant
"""

# 边界 Prompt
PROMPT_BOUNDARY = """<|im_start|>system
你是医疗信息提取助手。请判断以下描述中的病灶边界是否清晰。
如果边界清晰，输出"清晰"；如果边界不清或模糊，输出"不清晰"；如果无法判断，输出"未知"。<|im_end|>
<|im_start|>user
{processed_input}<|im_end|>
<|im_start|>assistant
"""

# 分叶征 Prompt
PROMPT_LOBULATION = """<|im_start|>system
你是医疗信息提取助手。请判断以下描述中的病灶是否有分叶征。
如果有分叶，输出"有"；如果没有分叶，输出"无"；如果无法判断，输出"未知"。<|im_end|>
<|im_start|>user
{processed_input}<|im_end|>
<|im_start|>assistant
"""
