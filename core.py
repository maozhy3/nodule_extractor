"""
核心业务逻辑模块
包含文本预处理、数值提取、模型预测等功能
"""
# 标准库
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Set, Tuple, Callable

# 第三方库
import pandas as pd
from llama_cpp import Llama
from tqdm import tqdm


# ==================== 全局停止标志 ====================
_stop_flag = False


def set_stop_flag(value: bool = True) -> None:
    """设置停止标志"""
    global _stop_flag
    _stop_flag = value


def should_stop() -> bool:
    """检查是否应该停止"""
    global _stop_flag
    return _stop_flag


# ==================== 文本预处理 ====================

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
        if len(match.group(0)) <= 20:
            return ""
        return match.group(0)

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


# ==================== 数值提取 ====================

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


# ==================== 模型预测 ====================

def predict_single(llm: Llama, input_text: str, prompt_template: str) -> Tuple[Optional[float], float]:
    """
    对单个输入进行预测
    
    Args:
        llm: 已加载的模型实例
        input_text: 原始输入文本
        prompt_template: prompt模板
    
    Returns:
        (预测结果, 推理耗时)
    """
    # 预处理
    processed_input = input_text
    processed_input = preprocessing(processed_input)
    processed_input = remove_img_tags(processed_input)
    processed_input = filter_segments(processed_input)
    
    # 构建prompt
    prompt = prompt_template.format(processed_input=processed_input)
    
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


# ==================== 多进程支持 ====================

def _worker_init(model_path: str, n_ctx: int, n_threads: int, n_gpu_layers: int) -> None:
    """进程初始化：每个进程加载一次模型"""
    global _worker_llm, _worker_prompt_template
    _worker_llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def _worker_predict(args: Tuple[int, str, str]) -> Tuple[int, Optional[float], float]:
    """工作进程：使用已加载的模型进行预测"""
    idx, input_text, prompt_template = args
    pred_value, infer_time = predict_single(_worker_llm, input_text, prompt_template)
    return idx, pred_value, infer_time


# ==================== 检查点管理 ====================

class CheckpointManager:
    """管理预测进度的检查点"""
    
    def __init__(self, checkpoint_dir: Path = None, save_interval: int = 10):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            save_interval: 每处理多少条数据保存一次检查点
        """
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_interval = save_interval
    
    def get_checkpoint_path(self, model_name: str) -> Path:
        """获取指定模型的检查点文件路径"""
        return self.checkpoint_dir / f"{model_name}_checkpoint.pkl"
    
    def save_checkpoint(
        self, 
        model_name: str, 
        predictions: list, 
        processed_indices: Set[int], 
        total_time: float
    ) -> None:
        """保存检查点"""
        checkpoint_path = self.get_checkpoint_path(model_name)
        checkpoint_data = {
            'predictions': predictions,
            'processed_indices': processed_indices,
            'total_time': total_time,
            'timestamp': time.time()
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except Exception as e:
            print(f"⚠ 保存检查点失败: {e}")
    
    def load_checkpoint(self, model_name: str) -> Optional[dict[str, Any]]:
        """加载检查点"""
        checkpoint_path = self.get_checkpoint_path(model_name)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            return checkpoint_data
        except Exception as e:
            print(f"⚠ 加载检查点失败: {e}")
            return None
    
    def clear_checkpoint(self, model_name: str) -> None:
        """清除检查点文件"""
        checkpoint_path = self.get_checkpoint_path(model_name)
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except Exception as e:
                print(f"⚠ 删除检查点失败: {e}")


# ==================== 批量预测 ====================

def batch_predict(df: pd.DataFrame, model_path: str, config: Any) -> Tuple[list, float, str]:
    """
    批量预测（支持断点续传）
    
    Args:
        df: 包含测试数据的DataFrame
        model_path: 模型文件路径
        config: 配置模块
    
    Returns:
        (预测结果列表, 总耗时, 模型名称)
    """
    model_name = Path(model_path).stem
    
    print(f"\n{'=' * 80}")
    print(f"模型: {model_name}")
    print(f"{'=' * 80}")
    
    # 初始化检查点管理器
    save_interval = getattr(config, 'CHECKPOINT_SAVE_INTERVAL', 10)
    checkpoint_manager = CheckpointManager(save_interval=save_interval)
    
    # 尝试加载检查点
    checkpoint = checkpoint_manager.load_checkpoint(model_name)
    if checkpoint:
        predictions = checkpoint['predictions']
        processed_indices = checkpoint['processed_indices']
        total_time = checkpoint['total_time']
        remaining = len(df) - len(processed_indices)
        print(f"✓ 检测到检查点，已完成 {len(processed_indices)}/{len(df)} 条，继续处理剩余 {remaining} 条")
    else:
        predictions = [None] * len(df)
        processed_indices = set()
        total_time = 0
        print(f"开始新的预测任务（共 {len(df)} 条）")
    
    max_workers = getattr(config, 'PROCESS_POOL_MAX_WORKERS', 1)
    
    if max_workers <= 1:
        # 单进程模式
        print("正在加载模型...")
        try:
            llm = Llama(
                model_path=model_path,
                n_ctx=config.LLAMA_N_CTX,
                n_threads=config.LLAMA_N_THREADS,
                n_gpu_layers=config.LLAMA_N_GPU_LAYERS,
                verbose=config.LLAMA_VERBOSE,
            )
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None, 0.0, model_name
        
        print(f"正在进行批量预测...")
        print("-" * 80)
        
        processed_count = 0
        stopped_by_user = False
        try:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="预测进度", initial=len(processed_indices)):
                # 检查停止标志
                if should_stop():
                    print("\n⚠ 检测到停止信号，正在保存检查点...")
                    stopped_by_user = True
                    checkpoint_manager.save_checkpoint(
                        model_name, predictions, processed_indices, total_time
                    )
                    print("✓ 检查点已保存，可以稍后继续运行")
                    break
                
                # 跳过已处理的数据
                if idx in processed_indices:
                    continue
                
                input_text = str(row['yxbx'])
                pred_value, infer_time = predict_single(llm, input_text, config.PROMPT_TEMPLATE)
                predictions[idx] = pred_value
                total_time += infer_time
                processed_indices.add(idx)
                processed_count += 1
                
                # 周期性保存检查点
                if processed_count % save_interval == 0:
                    checkpoint_manager.save_checkpoint(
                        model_name, predictions, processed_indices, total_time
                    )
        except KeyboardInterrupt:
            print("\n⚠ 检测到中断信号，正在保存检查点...")
            checkpoint_manager.save_checkpoint(
                model_name, predictions, processed_indices, total_time
            )
            print("✓ 检查点已保存，可以稍后继续运行")
            raise
        except Exception as e:
            print(f"\n❌ 预测过程出错: {e}")
            print("正在保存检查点...")
            checkpoint_manager.save_checkpoint(
                model_name, predictions, processed_indices, total_time
            )
            print("✓ 检查点已保存")
            raise
    else:
        # 多进程模式
        print(f"正在启动 {max_workers} 个进程并加载模型...")
        print("-" * 80)
        
        # 只处理未完成的任务
        tasks = [(idx, str(row['yxbx']), config.PROMPT_TEMPLATE) 
                 for idx, row in df.iterrows() if idx not in processed_indices]
        
        if not tasks:
            print("✓ 所有数据已处理完成")
        else:
            processed_count = 0
            try:
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_worker_init,
                    initargs=(model_path, config.LLAMA_N_CTX, config.LLAMA_N_THREADS, config.LLAMA_N_GPU_LAYERS)
                ) as executor:
                    futures = {executor.submit(_worker_predict, task): task for task in tasks}
                    
                    for future in tqdm(as_completed(futures), total=len(tasks), desc="预测进度", initial=len(processed_indices)):
                        # 检查停止标志
                        if should_stop():
                            print("\n⚠ 检测到停止信号，正在保存检查点...")
                            stopped_by_user = True
                            checkpoint_manager.save_checkpoint(
                                model_name, predictions, processed_indices, total_time
                            )
                            print("✓ 检查点已保存，可以稍后继续运行")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        
                        idx, pred_value, infer_time = future.result()
                        predictions[idx] = pred_value
                        total_time += infer_time
                        processed_indices.add(idx)
                        processed_count += 1
                        
                        # 周期性保存检查点
                        if processed_count % save_interval == 0:
                            checkpoint_manager.save_checkpoint(
                                model_name, predictions, processed_indices, total_time
                            )
            except KeyboardInterrupt:
                print("\n⚠ 检测到中断信号，正在保存检查点...")
                checkpoint_manager.save_checkpoint(
                    model_name, predictions, processed_indices, total_time
                )
                print("✓ 检查点已保存，可以稍后继续运行")
                raise
            except Exception as e:
                print(f"\n❌ 预测过程出错: {e}")
                print("正在保存检查点...")
                checkpoint_manager.save_checkpoint(
                    model_name, predictions, processed_indices, total_time
                )
                print("✓ 检查点已保存")
                raise

    # 最终保存检查点（仅在未被用户停止时）
    if not stopped_by_user:
        checkpoint_manager.save_checkpoint(
            model_name, predictions, processed_indices, total_time
        )
    
    avg_time = total_time / len(df) if len(df) > 0 else 0
    print("-" * 80)
    
    if stopped_by_user:
        print(f"⚠ 预测已停止！已处理: {len(processed_indices)}/{len(df)} 条")
    else:
        print(f"✓ 预测完成！总耗时: {total_time:.2f}s，平均耗时: {avg_time:.3f}s")
    
    # 完成后清除检查点（仅在完全完成时）
    if not stopped_by_user and len(processed_indices) == len(df):
        checkpoint_manager.clear_checkpoint(model_name)
        
    return predictions, total_time, model_name


# ==================== 特征提取功能（新增） ====================

def locate_lesion_sentence(text: str, max_size: float) -> str:
    """
    根据最大病灶尺寸，定位描述该病灶的句子
    
    Args:
        text: 原始报告文本
        max_size: 已提取的最大病灶尺寸（mm，已统一单位）
    
    Returns:
        包含该病灶的句子
    """
    if not max_size or max_size <= 0:
        return ""
    
    # 按句号、分号分割
    segments = re.split(r'[。；;]', text)
    
    matched_segments = []
    
    for seg in segments:
        # 第一次查找：mm 单位
        # 查找是否同时存在 ("mm" 或 "毫米") 和 max_size
        has_mm_unit = bool(re.search(r'mm|毫米', seg, re.IGNORECASE))
        has_mm_value = str(int(max_size)) in seg if max_size == int(max_size) else str(max_size) in seg
        
        if has_mm_unit and has_mm_value:
            matched_segments.append(seg)
            continue
        
        # 第二次查找：cm 单位
        # 查找是否同时存在 ("cm" 或 "厘米") 和 max_size/10
        cm_value = max_size / 10
        has_cm_unit = bool(re.search(r'cm|厘米', seg, re.IGNORECASE))
        has_cm_value = str(int(cm_value)) in seg if cm_value == int(cm_value) else str(cm_value) in seg
        
        if has_cm_unit and has_cm_value:
            matched_segments.append(seg)
    
    # 如果没有匹配，返回空
    if not matched_segments:
        return ""
    
    # 如果只有一个匹配，直接返回
    if len(matched_segments) == 1:
        return matched_segments[0]
    
    # 如果有多个匹配，进一步过滤
    # 排除不包含"肺""肋""膈""气""管"中任意一个的片段
    lung_keywords = ['肺', '肋', '膈', '气', '管']
    filtered_segments = [
        seg for seg in matched_segments 
        if any(kw in seg for kw in lung_keywords)
    ]
    
    # 返回过滤后的第一个，如果过滤后为空则返回原匹配的第一个
    return filtered_segments[0] if filtered_segments else matched_segments[0]


class FeatureExtractor:
    """基于规则+LLM的特征提取器"""
    
    # 关键词字典
    KEYWORDS = {
        'upper_lobe': {
            'positive': ['左上叶', '右上叶', '上叶', 'LUL', 'RUL', '左肺上叶', '右肺上叶'],
            'negative': ['左下叶', '右下叶', '下叶', '中叶', 'LLL', 'RLL', 'RML', '左肺下叶', '右肺下叶', '右肺中叶']
        },
        'spiculation': {
            'positive': ['毛刺', '短毛刺', '长毛刺', '毛刺征', '毛糙', '边缘毛糙', '毛刺样',
                        '可见毛刺', '伴毛刺', '见毛刺', '呈毛刺', '细毛刺'],
            'negative': ['无毛刺', '边缘光滑', '光滑']
        },
        'calcification': {
            'positive': ['钙化', '钙化灶', '钙化点', '钙化影', '钙化斑'],
            'negative': ['无钙化', '未见钙化']
        },
        'boundary': {
            'clear': ['边界清晰', '边界清', '界清', '轮廓清晰', '边缘清晰', '边清', 
                     '尚清', '边界尚清', '边缘尚清', '界尚清', '清楚', '边界清楚', '边缘清楚', 
                     '界限清', '界限清楚', '界限清晰'],
            'unclear': ['边界不清', '边界模糊', '界不清', '边缘模糊', '边不清',
                       '欠清', '边界欠清', '边缘欠清', '界欠清', '欠清晰',
                       '不清楚', '边界不清楚', '边缘不清楚', '界限不清', '界限模糊', '模糊']
        },
        'lobulation': {
            'positive': ['分叶', '分叶征', '分叶状', '浅分叶', '深分叶',
                        '可见分叶', '伴分叶', '见分叶', '呈分叶', '呈分叶状', '分叶样'],
            'negative': ['无分叶', '未见分叶', '未见明显分叶', '无明显分叶', '不伴分叶']
        },
        'pleural_indentation': {
            'positive': ['胸膜凹陷征', '胸膜牵拉', '胸膜凹陷', '胸膜牵拉征',
                        '可见胸膜凹陷', '伴胸膜凹陷', '见胸膜凹陷'],
            'negative': ['无胸膜凹陷征', '无胸膜牵拉', '无胸膜凹陷']
        }
    }
    
    @staticmethod
    def extract_by_keywords(target_sentence: str) -> dict:
        """
        基于关键词提取特征
        
        Args:
            target_sentence: 目标句子
        
        Returns:
            dict: 特征字典，None 表示需要LLM判断
        """
        results = {}
        
        # 1. 位置判断
        has_upper = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['upper_lobe']['positive'])
        has_lower = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['upper_lobe']['negative'])
        
        if has_upper and not has_lower:
            results['upper_lobe'] = '是'
        elif has_lower and not has_upper:
            results['upper_lobe'] = '否'
        else:
            results['upper_lobe'] = None
        
        # 2. 毛刺征（negative优先，因为"无毛刺"包含"毛刺"）
        has_spic_neg = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['spiculation']['negative'])
        has_spic_pos = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['spiculation']['positive'])
        
        if has_spic_neg:
            results['spiculation'] = '无'
        elif has_spic_pos:
            results['spiculation'] = '有'
        else:
            results['spiculation'] = None
        
        # 3. 钙化（negative优先，因为"无钙化"包含"钙化"）
        has_calc_neg = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['calcification']['negative'])
        has_calc_pos = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['calcification']['positive'])
        
        if has_calc_neg:
            results['calcification'] = '无'
        elif has_calc_pos:
            results['calcification'] = '有'
        else:
            results['calcification'] = None
        
        # 4. 边界（unclear优先，因为"边界不清"包含"界清"）
        has_unclear = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['boundary']['unclear'])
        has_clear = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['boundary']['clear'])
        
        if has_unclear:
            results['boundary'] = '不清晰'
        elif has_clear:
            results['boundary'] = '清晰'
        else:
            results['boundary'] = None
        
        # 5. 分叶征（negative优先，因为"无分叶"包含"分叶"）
        has_lob_neg = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['lobulation']['negative'])
        has_lob_pos = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['lobulation']['positive'])
        
        if has_lob_neg:
            results['lobulation'] = '无'
        elif has_lob_pos:
            results['lobulation'] = '有'
        else:
            results['lobulation'] = None
        
        # 6. 胸膜凹陷征（negative优先，因为"无胸膜凹陷征"包含"胸膜凹陷征"）
        has_pleural_neg = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['pleural_indentation']['negative'])
        has_pleural_pos = any(kw in target_sentence for kw in FeatureExtractor.KEYWORDS['pleural_indentation']['positive'])
        
        if has_pleural_neg:
            results['pleural_indentation'] = '无'
        elif has_pleural_pos:
            results['pleural_indentation'] = '有'
        else:
            results['pleural_indentation'] = '无'  # 没有提到则默认为无
        
        return results


def llm_fallback_extract(llm: Llama, target_sentence: str, feature_name: str, config: Any) -> str:
    """
    当关键词无法判断时，使用LLM进行判断
    
    Args:
        llm: 模型实例
        target_sentence: 目标句子
        feature_name: 特征名称
        config: 配置对象
    
    Returns:
        特征值
    """
    prompt_map = {
        'upper_lobe': getattr(config, 'PROMPT_UPPER_LOBE', None),
        'spiculation': getattr(config, 'PROMPT_SPICULATION', None),
        'calcification': getattr(config, 'PROMPT_CALCIFICATION', None),
        'boundary': getattr(config, 'PROMPT_BOUNDARY', None),
        'lobulation': getattr(config, 'PROMPT_LOBULATION', None)
    }
    
    prompt_template = prompt_map.get(feature_name)
    if not prompt_template:
        return '未知'
    
    prompt = prompt_template.format(processed_input=target_sentence)
    
    try:
        response = llm(
            prompt,
            max_tokens=5,
            temperature=0,
            stop=["<|im_end|>", "\n", "。"],
            echo=False,
        )
        result = response['choices'][0]['text'].strip()
        return result
    except Exception as e:
        print(f"  ⚠ LLM判断失败 ({feature_name}): {e}")
        return '未知'


def extract_all_features(llm: Llama, input_text: str, max_size: float, config: Any) -> dict:
    """
    提取所有特征（规则优先，LLM兜底）
    
    Args:
        llm: 模型实例
        input_text: 原始报告文本
        max_size: 已提取的最大病灶尺寸（mm）
        config: 配置对象
    
    Returns:
        dict: 包含所有特征的字典
    """
    # 第一步：定位目标句子
    target_sentence = locate_lesion_sentence(input_text, max_size)
    
    if not target_sentence:
        # 无法定位，返回默认值
        return {
            'upper_lobe': '未知',
            'spiculation': '未知',
            'calcification': '未知',
            'boundary': '未知',
            'lobulation': '未知',
            'pleural_indentation': '未知',
            'target_sentence': '',
            'llm_calls': 0
        }
    
    # 第二步：基于关键词提取
    features = FeatureExtractor.extract_by_keywords(target_sentence)
    
    # 第三步：LLM兜底（胸膜凹陷征不需要LLM，已经通过规则判断）
    llm_call_count = 0
    for feature_name, value in list(features.items()):
        if value is None:  # 关键词无法判断
            # 胸膜凹陷征不需要LLM兜底，已经在规则中处理
            if feature_name == 'pleural_indentation':
                features[feature_name] = '无'
            else:
                features[feature_name] = llm_fallback_extract(
                    llm, target_sentence, feature_name, config
                )
                llm_call_count += 1
    
    # 添加调试信息
    features['target_sentence'] = target_sentence
    features['llm_calls'] = llm_call_count
    
    return features


def batch_predict_with_features(df: pd.DataFrame, model_path: str, config: Any, existing_size_col: Optional[str] = None) -> Tuple[pd.DataFrame, float, str]:
    """
    批量预测（包含尺寸和特征提取）- 新增功能
    
    Args:
        df: 包含测试数据的DataFrame
        model_path: 模型文件路径（用于尺寸提取）
        config: 配置模块
        existing_size_col: 已有的尺寸列名（如果提供，则跳过尺寸提取）
    
    Returns:
        (结果DataFrame, 总耗时, 模型名称)
    """
    model_name = Path(model_path).stem
    
    print(f"\n{'=' * 80}")
    print(f"模型: {model_name} (特征提取模式)")
    print(f"{'=' * 80}")
    
    # 检查是否使用已有尺寸
    use_existing_size = existing_size_col is not None and existing_size_col in df.columns
    
    if use_existing_size:
        print(f"✓ 使用已有尺寸列: {existing_size_col}")
        print(f"  跳过尺寸提取，仅进行特征提取")
    
    # 检查是否使用单独的特征提取模型
    feature_model_path = getattr(config, 'FEATURE_EXTRACTION_MODEL_PATH', None)
    if feature_model_path and Path(feature_model_path).exists():
        actual_model_path = feature_model_path
        feature_model_name = Path(feature_model_path).stem
        print(f"✓ 特征提取使用单独模型: {feature_model_name}")
    else:
        actual_model_path = model_path
        feature_model_name = model_name
    
    # 初始化检查点管理器
    save_interval = getattr(config, 'CHECKPOINT_SAVE_INTERVAL', 10)
    checkpoint_manager = CheckpointManager(save_interval=save_interval)
    
    # 尝试加载检查点
    checkpoint_name = f"{model_name}_features"
    checkpoint = checkpoint_manager.load_checkpoint(checkpoint_name)
    if checkpoint:
        predictions = checkpoint['predictions']
        processed_indices = checkpoint['processed_indices']
        total_time = checkpoint['total_time']
        remaining = len(df) - len(processed_indices)
        print(f"✓ 检测到检查点，已完成 {len(processed_indices)}/{len(df)} 条，继续处理剩余 {remaining} 条")
    else:
        predictions = [None] * len(df)
        processed_indices = set()
        total_time = 0
        print(f"开始新的预测任务（共 {len(df)} 条）")
    
    # 单进程模式
    # 如果使用已有尺寸，只需加载特征提取模型
    # 如果不使用已有尺寸，需要加载尺寸提取模型（可能与特征提取模型相同）
    llm_size = None  # 用于尺寸提取的模型
    llm_feature = None  # 用于特征提取的模型
    
    if not use_existing_size:
        # 需要提取尺寸
        print(f"正在加载尺寸提取模型: {model_name}...")
        try:
            llm_size = Llama(
                model_path=model_path,
                n_ctx=config.LLAMA_N_CTX,
                n_threads=config.LLAMA_N_THREADS,
                n_gpu_layers=config.LLAMA_N_GPU_LAYERS,
                verbose=config.LLAMA_VERBOSE,
            )
            print("✓ 尺寸提取模型加载成功")
        except Exception as e:
            print(f"❌ 尺寸提取模型加载失败: {e}")
            return pd.DataFrame(), 0.0, model_name
    
    # 加载特征提取模型（如果与尺寸提取模型相同，则复用）
    if actual_model_path == model_path and llm_size is not None:
        llm_feature = llm_size
        print("✓ 特征提取复用尺寸提取模型")
    else:
        print(f"正在加载特征提取模型: {feature_model_name}...")
        try:
            llm_feature = Llama(
                model_path=actual_model_path,
                n_ctx=config.LLAMA_N_CTX,
                n_threads=config.LLAMA_N_THREADS,
                n_gpu_layers=config.LLAMA_N_GPU_LAYERS,
                verbose=config.LLAMA_VERBOSE,
            )
            print("✓ 特征提取模型加载成功")
        except Exception as e:
            print(f"❌ 特征提取模型加载失败: {e}")
            return pd.DataFrame(), 0.0, model_name
    
    print(f"正在进行批量预测（含特征提取）...")
    print("-" * 80)
    
    processed_count = 0
    total_llm_calls = 0
    stopped_by_user = False
    
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="预测进度", initial=len(processed_indices)):
            # 检查停止标志
            if should_stop():
                print("\n⚠ 检测到停止信号，正在保存检查点...")
                stopped_by_user = True
                checkpoint_manager.save_checkpoint(
                    checkpoint_name, predictions, processed_indices, total_time
                )
                print("✓ 检查点已保存，可以稍后继续运行")
                break
            
            # 跳过已处理的数据
            if idx in processed_indices:
                continue
            
            input_text = str(row['yxbx'])
            
            # 1. 提取最大尺寸（或使用已有尺寸）
            if use_existing_size:
                max_size = row[existing_size_col]
                # 处理可能的NaN或非数值
                if pd.isna(max_size):
                    max_size = 0
                else:
                    try:
                        max_size = float(max_size)
                    except (ValueError, TypeError):
                        max_size = 0
            else:
                max_size, infer_time = predict_single(llm_size, input_text, config.PROMPT_TEMPLATE)
                total_time += infer_time
            
            # 2. 提取其他特征（使用特征提取模型）
            if max_size and max_size > 0:
                features = extract_all_features(llm_feature, input_text, max_size, config)
                total_llm_calls += features.pop('llm_calls', 0)
                target_sent = features.pop('target_sentence', '')
            else:
                # 尺寸为0表示没有病灶，所有特征都是"无"
                features = {
                    'upper_lobe': '否',
                    'spiculation': '无',
                    'calcification': '无',
                    'boundary': '无',
                    'lobulation': '无',
                    'pleural_indentation': '无'
                }
                target_sent = ''
            
            # 3. 合并结果
            result = {
                'max_size': max_size,
                'upper_lobe': features['upper_lobe'],
                'spiculation': features['spiculation'],
                'calcification': features['calcification'],
                'boundary': features['boundary'],
                'lobulation': features['lobulation'],
                'pleural_indentation': features['pleural_indentation'],
            }
            
            # 可选：保存目标句子用于调试
            if getattr(config, 'SAVE_TARGET_SENTENCE', False):
                result['target_sentence'] = target_sent
            
            predictions[idx] = result
            processed_indices.add(idx)
            processed_count += 1
            
            # 周期性保存检查点
            if processed_count % save_interval == 0:
                checkpoint_manager.save_checkpoint(
                    checkpoint_name, predictions, processed_indices, total_time
                )
    except KeyboardInterrupt:
        print("\n⚠ 检测到中断信号，正在保存检查点...")
        checkpoint_manager.save_checkpoint(
            checkpoint_name, predictions, processed_indices, total_time
        )
        print("✓ 检查点已保存，可以稍后继续运行")
        raise
    except Exception as e:
        print(f"\n❌ 预测过程出错: {e}")
        print("正在保存检查点...")
        checkpoint_manager.save_checkpoint(
            checkpoint_name, predictions, processed_indices, total_time
        )
        print("✓ 检查点已保存")
        raise
    
    # 最终保存检查点（仅在未被用户停止时）
    if not stopped_by_user:
        checkpoint_manager.save_checkpoint(
            checkpoint_name, predictions, processed_indices, total_time
        )
    
    # 统计信息
    avg_time = total_time / len(df) if len(df) > 0 else 0
    avg_llm_calls = total_llm_calls / len(df) if len(df) > 0 else 0
    # 5个需要LLM兜底的特征（胸膜凹陷征不需要LLM）
    rule_coverage = (1 - avg_llm_calls / 5) * 100 if avg_llm_calls < 5 else 0
    
    print("-" * 80)
    
    if stopped_by_user:
        print(f"⚠ 预测已停止！已处理: {len(processed_indices)}/{len(df)} 条")
    else:
        print(f"✓ 预测完成！总耗时: {total_time:.2f}s，平均耗时: {avg_time:.3f}s")
        print(f"  特征提取统计：平均每条额外调用LLM {avg_llm_calls:.2f} 次")
        print(f"  规则覆盖率：{rule_coverage:.1f}%")
    
    # 完成后清除检查点（仅在完全完成时）
    if not stopped_by_user and len(processed_indices) == len(df):
        checkpoint_manager.clear_checkpoint(checkpoint_name)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(predictions)
    
    return results_df, total_time, model_name
