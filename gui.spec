# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# 获取 conda 环境路径
conda_env = r'C:\Users\User\miniconda3\envs\node_extractor'
llama_cpp_lib = os.path.join(conda_env, 'Lib', 'site-packages', 'llama_cpp', 'lib')

# 收集 llama_cpp 的 DLL 文件
llama_binaries = []
if os.path.exists(llama_cpp_lib):
    for file in os.listdir(llama_cpp_lib):
        if file.endswith('.dll'):
            llama_binaries.append((os.path.join(llama_cpp_lib, file), 'llama_cpp/lib'))

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=llama_binaries,
    datas=[
        ('config.py', '.'),
        ('config_loader.py', '.'),
        ('core.py', '.'),
    ],
    hiddenimports=['pandas', 'openpyxl', 'llama_cpp', 'tqdm', 'tkinter', 'config_loader', 'core'],
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='医疗影像报告预测工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI程序不显示控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='医疗影像报告预测工具',
)
