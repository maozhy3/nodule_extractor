#!/usr/bin/env python3
"""
检查点功能测试脚本
演示如何使用断点续传功能
"""
import pandas as pd
from pathlib import Path
from core import CheckpointManager

def show_checkpoint_status():
    """显示当前检查点状态"""
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print("没有找到检查点目录")
        return
    
    checkpoint_files = list(checkpoint_dir.glob("*_checkpoint.pkl"))
    
    if not checkpoint_files:
        print("没有找到检查点文件")
        return
    
    print("\n当前检查点状态：")
    print("=" * 60)
    
    manager = CheckpointManager()
    for checkpoint_file in checkpoint_files:
        model_name = checkpoint_file.stem.replace("_checkpoint", "")
        checkpoint = manager.load_checkpoint(model_name)
        
        if checkpoint:
            processed = len(checkpoint['processed_indices'])
            total_time = checkpoint['total_time']
            timestamp = checkpoint['timestamp']
            
            import time
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
            
            print(f"模型: {model_name}")
            print(f"  已处理: {processed} 条")
            print(f"  累计耗时: {total_time:.2f}s")
            print(f"  保存时间: {time_str}")
            print("-" * 60)

def clear_all_checkpoints():
    """清除所有检查点"""
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print("没有找到检查点目录")
        return
    
    checkpoint_files = list(checkpoint_dir.glob("*_checkpoint.pkl"))
    
    if not checkpoint_files:
        print("没有找到检查点文件")
        return
    
    manager = CheckpointManager()
    for checkpoint_file in checkpoint_files:
        model_name = checkpoint_file.stem.replace("_checkpoint", "")
        manager.clear_checkpoint(model_name)
        print(f"✓ 已清除 {model_name} 的检查点")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        clear_all_checkpoints()
    else:
        show_checkpoint_status()
        print("\n提示：运行 'python test_checkpoint.py clear' 可清除所有检查点")
