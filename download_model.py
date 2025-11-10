#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5 模型下载脚本
支持从 Hugging Face 下载 GGUF 格式的量化模型
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError


# 模型配置
MODELS = {
    "1": {
        "name": "Qwen2.5-3B-Instruct (Q4_K_M)",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "size": "~2GB"
    },
    "2": {
        "name": "Qwen2.5-7B-Instruct (Q4_K_M)",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        "size": "~4.7GB"
    }
}

# 模型保存目录
MODELS_DIR = Path("models")


def show_progress(block_num, block_size, total_size):
    """显示下载进度"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100.0 / total_size, 100)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        bar_length = 50
        filled_length = int(bar_length * downloaded / total_size)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\r下载进度: |{bar}| {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)')
        sys.stdout.flush()


def download_model(model_info):
    """下载模型文件"""
    filename = model_info["filename"]
    url = model_info["url"]
    
    # 创建模型目录
    MODELS_DIR.mkdir(exist_ok=True)
    
    # 目标文件路径
    target_path = MODELS_DIR / filename
    
    # 检查文件是否已存在
    if target_path.exists():
        print(f"\n✓ 模型文件已存在: {target_path}")
        overwrite = input("是否重新下载？(y/N): ").strip().lower()
        if overwrite != 'y':
            print("跳过下载")
            return True
    
    print(f"\n开始下载: {model_info['name']}")
    print(f"文件大小: {model_info['size']}")
    print(f"下载地址: {url}")
    print(f"保存路径: {target_path}\n")
    
    try:
        # 下载文件
        urlretrieve(url, target_path, reporthook=show_progress)
        print(f"\n\n✓ 下载完成: {target_path}")
        return True
    
    except HTTPError as e:
        print(f"\n\n✗ HTTP 错误: {e.code} - {e.reason}")
        print("提示: 如果遇到网络问题，可以尝试:")
        print("  1. 使用代理或 VPN")
        print("  2. 手动从 Hugging Face 下载")
        return False
    
    except URLError as e:
        print(f"\n\n✗ 网络错误: {e.reason}")
        print("请检查网络连接")
        return False
    
    except KeyboardInterrupt:
        print("\n\n下载已取消")
        # 删除未完成的文件
        if target_path.exists():
            target_path.unlink()
            print(f"已删除未完成的文件: {target_path}")
        return False
    
    except Exception as e:
        print(f"\n\n✗ 下载失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("Qwen2.5 模型下载工具")
    print("=" * 60)
    
    # 显示模型列表
    print("\n可用模型:")
    for key, model in MODELS.items():
        print(f"  [{key}] {model['name']} - {model['size']}")
    print("  [3] 下载全部模型")
    print("  [0] 退出")
    
    # 用户选择
    choice = input("\n请选择要下载的模型 (输入编号): ").strip()
    
    if choice == "0":
        print("退出程序")
        return
    
    elif choice == "3":
        # 下载全部模型
        print("\n准备下载全部模型...")
        success_count = 0
        for model_info in MODELS.values():
            if download_model(model_info):
                success_count += 1
        
        print("\n" + "=" * 60)
        print(f"下载完成: {success_count}/{len(MODELS)} 个模型成功")
        print("=" * 60)
    
    elif choice in MODELS:
        # 下载单个模型
        model_info = MODELS[choice]
        if download_model(model_info):
            print("\n" + "=" * 60)
            print("下载成功!")
            print("=" * 60)
            print(f"\n模型已保存到: {MODELS_DIR / model_info['filename']}")
            print("\n使用方法:")
            print(f"  1. 在 config.py 中设置模型路径")
            print(f"  2. 或在 GUI 中添加模型文件")
    
    else:
        print("无效的选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
        sys.exit(0)
