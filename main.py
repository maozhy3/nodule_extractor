#!/usr/bin/env python3
import sys, subprocess, os, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import re
import time
import pandas as pd
from tqdm import tqdm
from typing import Optional
from llama_cpp import Llama
from concurrent.futures import ProcessPoolExecutor, as_completed

# 默认配置（内层兜底）
from config import *

# 尝试加载外层配置（如果存在）
outer = os.path.join(os.path.dirname(sys.executable), 'config.py')
if os.path.exists(outer):
    spec = importlib.util.spec_from_file_location("external_config", outer)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    # 用外层配置覆盖内层
    for key in dir(cfg):
        if not key.startswith('_'):
            globals()[key] = getattr(cfg, key)


# 首次运行处理vc
bundle = Path(__file__).parent
flag = bundle / '_vcredist' / '.done'
vc    = bundle / '_vcredist' / 'vc_redist.x64.exe'
if vc.exists() and not flag.exists():
    subprocess.check_call([str(vc), '/quiet', '/norestart'])
    flag.touch()

def filter_segments(text: str) -> str:
    """
    将输入段落按"。"、"；"或";"分割，保留分隔符；
    仅保留同时包含("肺"|"膈"|"肋"|"气")和数字的片段；
    最后按原顺序拼接并返回。
    """
    parts = re.split(r'([。；;])', text)
    segments = [parts[i] + parts[i+1] for i in range(0, len(parts)-1, 2)]
    if len(parts) % 2 == 1:
        segments.append(parts[-1])
    kept = []
    for seg in segments:
        if re.search(r'[肺膈肋气]', seg) and re.search(r'\d', seg):
            kept.append(seg)
    return ''.join(kept)

def remove_img_tags(text: str) -> str:
    """
    删除中英文括号内包含 lm/im/img（不区分大小写）的片段，
    但仅当整个括号片段长度 ≤ 20 字符。
    """
    if not text:
        return text

    def replacer(match: re.Match) -> str:
        # match.group(0) 是整个括号片段，如 "(IMG79/80)" 或 "（img123）"
        if len(match.group(0)) <= 20:
            return ""
        return match.group(0)  # 太长，保留原样

    # 匹配最内层或外层括号均可，这里用非贪婪匹配括号内容
    pattern = r'[（(][^)）]*?(?:lm|im|img)[^)）]*?[)）]'
    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)

def preprocessing(input: str) -> str:
    """预处理输入文本"""
    pattern = re.compile(r'(\d+(?:\.\d+)?)×(\d+(?:\.\d+)?)(mm|cm)')
    input = input.replace(" ", "").replace("\n", "").replace("(brn)", "").replace("（brn）", "")
    input = re.sub(r'[xX*\-~]', '×', input)
    def _repl(m: re.Match) -> str:
        n1, n2, unit = m.groups()
        return f"{n1}{unit}×{n2}{unit}"
    return pattern.sub(_repl, input)

import re
from typing import Optional

# 预编译正则：把"数字+可选空白+单位"整体抓出来
_NUM_UNIT_RE = re.compile(r'(\d+\.?\d*)\s*(cm|mm|um|μm|m)?', flags=re.I)

def _normalize(text: str) -> str:
    """统一中英文符号、去掉常见噪音"""
    return (text
            .replace('（', '(').replace('）', ')')
            .replace('×', 'x').replace('X', 'x')
            .replace('~', '-').replace('*', '-'))

def extract_max_value(text: str, raw_text: str = "") -> Optional[float]:
    """提取最大数值，cm×10→mm。若最终结果<2，则乘10。"""
    text = _normalize(text).replace("(brn)", "")
    numbers = []

    # 1) 带单位
    for val_str, unit in _NUM_UNIT_RE.findall(text):
        try:
            val = float(val_str)
        except ValueError:
            continue
        if unit and unit.lower() == "cm":
            val *= 10
        numbers.append(val)

    # 2) 兜底纯数字
    if not numbers:
        numbers = [float(m) for m in re.findall(r"\d+\.?\d*", text)]

    if not numbers:
        return None

    max_val = max(numbers)

    # 3) cm或mm最终校验
    has_cm = bool(re.search(r'cm', raw_text, flags=re.I)) or '厘米' in raw_text
    has_mm = bool(re.search(r'mm', raw_text, flags=re.I)) or '毫米' in raw_text

    if max_val < 3 and has_cm:                        
        max_val *= 10
    elif 3 <= max_val < 7 and has_cm and not has_mm:  
        max_val *= 10


    return int(max_val) if max_val.is_integer() else max_val


def predict_single(llm: Llama, input_text: str) -> tuple[Optional[float], float]:
    """
    对单个输入进行预测
    
    Returns:
        (预测结果, 推理耗时)
    """
    # 预处理
    processed_input = input_text
    processed_input = preprocessing(processed_input)
    processed_input = remove_img_tags(processed_input)
    processed_input = filter_segments(processed_input)
    
    # 构建prompt
    prompt = PROMPT_TEMPLATE.format(processed_input=processed_input)
    
    # 推理
    t0 = time.perf_counter()
    try:
        response = llm(
            prompt,
            max_tokens=10,
            temperature=0.1,
            top_p=0.9,
            stop=["<|im_end|>", "\n", "cm", "。"],
            echo=False,
        )
        t1 = time.perf_counter()
        
        raw_result = response['choices'][0]['text'].strip()
        final_result = extract_max_value(raw_result, input_text)
        
        return final_result, t1 - t0
    except Exception as e:
        print(f"预测失败: {e}")
        return None, 0.0



def _worker_init(model_path: str):
    """进程初始化：每个进程加载一次模型"""
    global _worker_llm
    _worker_llm = Llama(
        model_path=model_path,
        n_ctx=LLAMA_N_CTX,
        n_threads=LLAMA_N_THREADS,
        n_gpu_layers=LLAMA_N_GPU_LAYERS,
        verbose=False,  # 多进程时关闭日志避免混乱
    )


def _worker_predict(args):
    """工作进程：使用已加载的模型进行预测"""
    idx, input_text = args
    pred_value, infer_time = predict_single(_worker_llm, input_text)
    return idx, pred_value, infer_time


def batch_predict(df: pd.DataFrame, model_path: str) -> tuple[list, float, str]:
    """
    Args:
        df: 包含测试数据的DataFrame
        model_path: 模型文件路径
    Returns:
        (预测结果列表, 总耗时, 模型名称)
    """
    # 获取模型名称
    model_name = Path(model_path).stem
    
    print(f"\n{'=' * 80}")
    print(f"模型: {model_name}")
    print(f"{'=' * 80}")
    
    # 获取进程数配置
    try:
        from config import PROCESS_POOL_MAX_WORKERS
        max_workers = PROCESS_POOL_MAX_WORKERS
    except ImportError:
        max_workers = 1  # 默认单进程
    
    if max_workers <= 1:
        # 单进程模式
        print("正在加载模型...")
        try:
            llm = Llama(
                model_path=model_path,
                n_ctx=LLAMA_N_CTX,
                n_threads=LLAMA_N_THREADS,
                n_gpu_layers=LLAMA_N_GPU_LAYERS,
                verbose=LLAMA_VERBOSE,
            )
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None, 0.0, model_name
        
        print(f"正在进行批量预测（共 {len(df)} 条）...")
        print("-" * 80)
        
        predictions = []
        total_time = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="预测进度"):
            input_text = str(row['yxbx'])
            pred_value, infer_time = predict_single(llm, input_text)
            predictions.append(pred_value)
            total_time += infer_time
    else:
        # 多进程模式
        print(f"正在启动 {max_workers} 个进程并加载模型...")
        print("-" * 80)
        
        predictions = [None] * len(df)
        total_time = 0
        
        # 准备任务列表
        tasks = [(idx, str(row['yxbx'])) for idx, row in df.iterrows()]
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(model_path,)
        ) as executor:
            futures = {executor.submit(_worker_predict, task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(df), desc="预测进度"):
                idx, pred_value, infer_time = future.result()
                predictions[idx] = pred_value
                total_time += infer_time

    avg_time = total_time / len(df)
    print("-" * 80)
    print(f"✓预测完成！总耗时: {total_time:.2f}s，平均耗时: {avg_time:.3f}s")
        
    return predictions, total_time, model_name


def batch_run(excel_path: str, model_paths: list[str], output_path: Optional[str] = None) -> bool:
    # 保存结果
    print(f"\n正在保存结果到 {output_path}...")
    try:
        df.to_excel(output_path, index=False)
        print(f"✓ 结果已保存到: {output_path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("推理完成！")
    print("=" * 80)
    
    return True


if __name__ == "__main__":

    df = pd.read_excel(EXCEL_PATH)

    for model_path in MODEL_PATHS:
        preds, total_time, model_name = batch_predict(df, model_path)

        col_name = f"pred_{model_name}"
        df[col_name] = preds


    df.to_excel(OUTPUT_PATH, index=False)
    print(f"结果已保存至：{OUTPUT_PATH}")
    input("按任意键结束")