# config.py
import os
from pathlib import Path

# ===== 输入输出 =====
EXCEL_PATH      = Path(__file__).with_name("test_results6.xlsx")      # 待预测文件
OUTPUT_PATH     = Path(__file__).with_name("test_results7.xlsx")  # 结果保存

# ===== 模型列表 =====
# 支持单个字符串或列表；也可使用通配符，由脚本自动展开
_ROOT = Path(__file__).parent
MODEL_PATHS = [
    (_ROOT / "models" / "qwen-medical-lora-251106-f16.gguf").as_posix()
]
# ===== llama.cpp 推理参数 =====
LLAMA_N_CTX        = 2048   # 上下文长度
LLAMA_N_THREADS = 8
THREAD_POOL_MAX_WORKERS = 1
LLAMA_N_GPU_LAYERS = 0      # 0 表示纯 CPU；>0 表示 offload 到 GPU 的层数
LLAMA_VERBOSE      = False  # 是否打印 llama.cpp 的 debug 信息

PROMPT_TEMPLATE = """<|im_start|>system
你是医疗信息提取助手。我需要你从影像表现的报告中，找到肺部病灶（包括肺上叶中叶下叶的结节、磨玻璃结节、团块或局灶影，不包括空洞，不包括纵隔肿物）的最大直径，可能是“长径”、“直径”、“大小”中的最大一项。如果有多个病灶，只需要最长的。返回的结果以mm为单位，如果是cm你需要进行转换，1cm=10mm。忽略CT值（HU）。  报告中其他部位和系统（如肝，肾，脾）的病灶请无视。如果报告中没有肺部病灶，或者没有具体的尺寸信息，请输出0。你的输出结果只需要输出最终的数字，不需要任何的单位或者前置描述。<|im_end|>
<|im_start|>user
影像报告：{processed_input}

请提取报告中最大的病灶尺寸（mm）：<|im_end|>
<|im_start|>assistant
"""