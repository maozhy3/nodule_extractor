#!/usr/bin/env python3
"""
命令行版本 - 医疗影像报告批量预测工具
"""
# 标准库
import os
import subprocess
import sys
from pathlib import Path

# 第三方库
import pandas as pd

# 本地模块
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_config
from core import batch_predict

# 加载配置
config = load_config()


# 首次运行处理vc
bundle = Path(__file__).parent
flag = bundle / '_vcredist' / '.done'
vc    = bundle / '_vcredist' / 'vc_redist.x64.exe'
if vc.exists() and not flag.exists():
    try:
        subprocess.check_call([str(vc), '/quiet', '/norestart'])
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
    except Exception as e:
        print(f"警告：VC++ 运行库安装失败: {e}")


def main() -> None:
    """主函数"""
    try:
        df = pd.read_excel(config.EXCEL_PATH)
        print(f"✓ 成功读取输入文件: {config.EXCEL_PATH}")
        print(f"共 {len(df)} 条数据\n")

        # 检查是否启用特征提取功能
        enable_features = getattr(config, 'ENABLE_FEATURE_EXTRACTION', False)
        
        if enable_features:
            print("📋 特征提取模式已启用")
            print("   将提取：最大尺寸、位置、毛刺征、钙化、边界清晰度、分叶征、胸膜凹陷征\n")
        else:
            print("📏 仅提取最大尺寸模式\n")

        for model_path in config.MODEL_PATHS:
            if enable_features:
                # 使用特征提取模式
                from core import batch_predict_with_features
                
                # 检查是否已有该模型的尺寸结果
                model_name = Path(model_path).stem
                existing_size_col = None
                
                # 检查已知的模型列名
                known_models = [
                    "qwen-medical-lora-251106-f16",
                    "qwen-medical-lora-251106-q4_k_m",
                    "qwen2.5-3b-instruct-q4_k_m"
                ]
                
                for known_model in known_models:
                    pred_col = f"pred_{known_model}"
                    if pred_col in df.columns:
                        existing_size_col = pred_col
                        print(f"✓ 检测到已有尺寸结果列: {pred_col}")
                        print(f"  将跳过尺寸提取，直接使用已有结果进行特征提取\n")
                        break
                
                results_df, total_time, model_name = batch_predict_with_features(
                    df, model_path, config, existing_size_col
                )
                
                # 将结果列合并到原始df
                for col in results_df.columns:
                    col_name = f"{col}_{model_name}" if col != 'max_size' else f"pred_{model_name}"
                    df[col_name] = results_df[col]
            else:
                # 使用原有的仅提取尺寸模式
                preds, total_time, model_name = batch_predict(df, model_path, config)
                col_name = f"pred_{model_name}"
                df[col_name] = preds

        df.to_excel(config.OUTPUT_PATH, index=False)
        print(f"\n✓ 结果已保存至：{config.OUTPUT_PATH}")
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        input("\n按任意键结束...")


if __name__ == "__main__":
    main()