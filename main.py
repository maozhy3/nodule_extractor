#!/usr/bin/env python3
import sys, subprocess, os, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd

# 导入核心业务逻辑
from core import batch_predict

# 默认配置（内层兜底）
import config

# 尝试加载外层配置（如果存在）
outer = os.path.join(os.path.dirname(sys.executable), 'config.py')
if os.path.exists(outer):
    spec = importlib.util.spec_from_file_location("external_config", outer)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    
    # 用外层配置覆盖内层
    for key in dir(cfg):
        if not key.startswith('_'):
            setattr(config, key, getattr(cfg, key))


# 首次运行处理vc
bundle = Path(__file__).parent
flag = bundle / '_vcredist' / '.done'
vc    = bundle / '_vcredist' / 'vc_redist.x64.exe'
if vc.exists() and not flag.exists():
    subprocess.check_call([str(vc), '/quiet', '/norestart'])
    flag.touch()


if __name__ == "__main__":
    df = pd.read_excel(config.EXCEL_PATH)

    for model_path in config.MODEL_PATHS:
        preds, total_time, model_name = batch_predict(df, model_path, config)
        col_name = f"pred_{model_name}"
        df[col_name] = preds

    df.to_excel(config.OUTPUT_PATH, index=False)
    print(f"结果已保存至：{config.OUTPUT_PATH}")
    input("按任意键结束")