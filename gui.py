#!/usr/bin/env python3
"""
GUI界面 - 医疗影像报告批量预测工具
Version: 1.2.0
"""

__version__ = "1.2.0"
# 标准库
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from io import StringIO
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

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

# 第三方库
import pandas as pd

# 本地模块
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_config
from core import batch_predict, batch_predict_with_features, set_stop_flag

# 加载配置
config = load_config()


class RedirectText:
    """重定向print输出到GUI文本框，智能处理tqdm进度条"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.current_line = ""
        self.is_progress_line = False
        
    def write(self, string):
        import re
        
        # 清理ANSI转义序列
        clean_string = self._clean_ansi(string)
        
        # 处理回车符（tqdm使用\r来更新同一行）
        if '\r' in clean_string:
            # 分割字符串
            parts = clean_string.split('\r')
            
            for i, part in enumerate(parts):
                if i == 0 and self.is_progress_line:
                    # 删除上一个进度行
                    try:
                        self.text_widget.delete("end-2c linestart", "end-1c")
                    except:
                        pass
                
                if part.strip():
                    # 检测是否是进度条行（包含百分比或it/s）
                    if '%' in part or 'it/s' in part or '/s' in part:
                        self.text_widget.insert(tk.END, part.strip() + '\n')
                        self.is_progress_line = True
                    else:
                        self.text_widget.insert(tk.END, part)
                        self.is_progress_line = False
        else:
            # 普通输出
            self.text_widget.insert(tk.END, clean_string)
            if '\n' in clean_string:
                self.is_progress_line = False
            
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()
        
    def _clean_ansi(self, text):
        """清理ANSI转义序列和其他控制字符"""
        import re
        # 移除ANSI颜色代码和控制序列
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        # 移除其他控制字符（保留\n, \r, \t）
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        return text
        
    def flush(self):
        pass


class MedicalPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"医疗影像报告批量预测工具 v{__version__}")
        self.root.geometry("900x700")
        
        # 变量
        self.excel_path = tk.StringVar(value=str(config.EXCEL_PATH))
        self.output_path = tk.StringVar(value=str(config.OUTPUT_PATH))
        self.model_path = tk.StringVar(value=str(config.MODEL_PATHS[0]) if config.MODEL_PATHS else "")
        self.is_running = False
        
        # 配置参数变量
        self.n_threads = tk.IntVar(value=config.LLAMA_N_THREADS)
        self.n_gpu_layers = tk.IntVar(value=config.LLAMA_N_GPU_LAYERS)
        self.max_workers = tk.IntVar(value=config.PROCESS_POOL_MAX_WORKERS)
        self.checkpoint_interval = tk.IntVar(value=config.CHECKPOINT_SAVE_INTERVAL)
        
        # 特征提取相关变量
        self.enable_features = tk.BooleanVar(value=getattr(config, 'ENABLE_FEATURE_EXTRACTION', False))
        self.save_target_sentence = tk.BooleanVar(value=getattr(config, 'SAVE_TARGET_SENTENCE', False))
        self.feature_model_path = tk.StringVar(value=str(getattr(config, 'FEATURE_EXTRACTION_MODEL_PATH', '') or ''))
        
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="医疗影像报告批量预测工具", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 15))
        
        # ===== 文件和模型配置区域 =====
        row = 1
        io_frame = ttk.LabelFrame(main_frame, text="文件和模型配置", padding="10")
        io_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        io_frame.columnconfigure(1, weight=1)
        
        # 输入文件
        ttk.Label(io_frame, text="输入文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(io_frame, textvariable=self.excel_path, width=50).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(io_frame, text="浏览...", command=self.browse_input).grid(
            row=0, column=2, padx=5)
        
        # 输出文件
        ttk.Label(io_frame, text="输出文件:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(io_frame, textvariable=self.output_path, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(io_frame, text="浏览...", command=self.browse_output).grid(
            row=1, column=2, padx=5)
        
        # 模型文件
        ttk.Label(io_frame, text="模型文件:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(io_frame, textvariable=self.model_path, width=50).grid(
            row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(io_frame, text="浏览...", command=self.browse_model).grid(
            row=2, column=2, padx=5)
        
        # ===== 性能参数配置区域 =====
        row += 1
        config_frame = ttk.LabelFrame(main_frame, text="性能参数", padding="10")
        config_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        config_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(3, weight=1)
        
        # 第一行：线程数和GPU层数
        ttk.Label(config_frame, text="CPU线程数:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Spinbox(config_frame, from_=1, to=32, textvariable=self.n_threads, 
                   width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(config_frame, text="GPU层数:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        ttk.Spinbox(config_frame, from_=0, to=100, textvariable=self.n_gpu_layers, 
                   width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(config_frame, text="(0=纯CPU)").grid(row=0, column=4, sticky=tk.W, padx=5)
        
        # 第二行：进程数和检查点间隔
        ttk.Label(config_frame, text="并行进程数:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        ttk.Spinbox(config_frame, from_=1, to=16, textvariable=self.max_workers, 
                   width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=(10, 0))
        
        ttk.Label(config_frame, text="检查点间隔:").grid(row=1, column=2, sticky=tk.W, padx=(20, 5), pady=(10, 0))
        ttk.Spinbox(config_frame, from_=1, to=10000, textvariable=self.checkpoint_interval, 
                   width=10).grid(row=1, column=3, sticky=tk.W, padx=5, pady=(10, 0))
        
        ttk.Label(config_frame, text="(条/次)").grid(row=1, column=4, sticky=tk.W, padx=5, pady=(10, 0))
        
        # 说明文字
        help_text = "提示：进程数×线程数 ≈ CPU核心数；多进程会增加内存占用"
        ttk.Label(config_frame, text=help_text, foreground="gray", 
                 font=("Arial", 8)).grid(row=2, column=0, columnspan=5, sticky=tk.W, pady=(5, 0))
        
        # ===== 特征提取配置区域 =====
        row += 1
        feature_frame = ttk.LabelFrame(main_frame, text="特征提取配置", padding="10")
        feature_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        feature_frame.columnconfigure(1, weight=1)
        
        # 启用特征提取
        ttk.Checkbutton(feature_frame, text="启用特征提取（提取位置、毛刺征、钙化、边界、分叶征、胸膜凹陷征）", 
                       variable=self.enable_features).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 特征提取模型路径
        ttk.Label(feature_frame, text="特征提取模型:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(feature_frame, textvariable=self.feature_model_path, width=40).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(feature_frame, text="浏览...", command=self.browse_feature_model).grid(
            row=1, column=2, padx=5)
        
        ttk.Label(feature_frame, text="(留空则使用主模型)", foreground="gray", 
                 font=("Arial", 8)).grid(row=2, column=1, sticky=tk.W, pady=(0, 5))
        
        # 保存目标句子选项
        ttk.Checkbutton(feature_frame, text="保存目标句子（用于调试）", 
                       variable=self.save_target_sentence).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # ===== 控制按钮 =====
        row += 1
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, pady=15)
        
        self.start_btn = ttk.Button(btn_frame, text="开始预测", 
                                    command=self.start_prediction, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="停止", 
                                   command=self.stop_prediction, 
                                   state=tk.DISABLED, width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # ===== 日志输出 =====
        row += 1
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="5")
        log_frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(row, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, 
                                                  wrap=tk.WORD, state=tk.NORMAL)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ===== 状态栏 =====
        row += 1
        self.status_label = ttk.Label(main_frame, text="就绪", 
                                     relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def browse_input(self):
        """浏览输入文件"""
        filename = filedialog.askopenfilename(
            title="选择输入Excel文件",
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("所有文件", "*.*")]
        )
        if filename:
            self.excel_path.set(filename)
    
    def browse_output(self):
        """浏览输出文件"""
        filename = filedialog.asksaveasfilename(
            title="选择输出Excel文件",
            defaultextension=".xlsx",
            filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
    
    def browse_model(self):
        """浏览模型文件"""
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("GGUF模型", "*.gguf"), ("所有文件", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
    
    def browse_feature_model(self):
        """浏览特征提取模型文件"""
        filename = filedialog.askopenfilename(
            title="选择特征提取模型文件",
            filetypes=[("GGUF模型", "*.gguf"), ("所有文件", "*.*")]
        )
        if filename:
            self.feature_model_path.set(filename)
    
    def start_prediction(self):
        """开始预测"""
        # 验证输入
        if not Path(self.excel_path.get()).exists():
            messagebox.showerror("错误", "输入文件不存在！")
            return
        
        if not self.model_path.get() or not Path(self.model_path.get()).exists():
            messagebox.showerror("错误", "请选择有效的模型文件！")
            return
        
        # 更新配置参数
        config.LLAMA_N_THREADS = self.n_threads.get()
        config.LLAMA_N_GPU_LAYERS = self.n_gpu_layers.get()
        config.PROCESS_POOL_MAX_WORKERS = self.max_workers.get()
        config.CHECKPOINT_SAVE_INTERVAL = self.checkpoint_interval.get()
        
        # 更新特征提取配置
        config.ENABLE_FEATURE_EXTRACTION = self.enable_features.get()
        config.SAVE_TARGET_SENTENCE = self.save_target_sentence.get()
        feature_model = self.feature_model_path.get().strip()
        config.FEATURE_EXTRACTION_MODEL_PATH = feature_model if feature_model else None
        
        # 重置停止标志
        set_stop_flag(False)
        
        # 更新UI状态
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="正在运行...")
        self.log_text.delete(1.0, tk.END)
        
        # 重定向输出
        sys.stdout = RedirectText(self.log_text)
        sys.stderr = RedirectText(self.log_text)
        
        # 在新线程中运行预测
        thread = threading.Thread(target=self.run_prediction, daemon=True)
        thread.start()
    
    def run_prediction(self):
        """运行预测任务"""
        try:
            # 读取数据
            df = pd.read_excel(self.excel_path.get())
            print(f"✓ 成功读取输入文件: {self.excel_path.get()}")
            print(f"共 {len(df)} 条数据\n")
            
            # 检查是否启用特征提取功能
            enable_features = getattr(config, 'ENABLE_FEATURE_EXTRACTION', False)
            
            if enable_features:
                print("📋 特征提取模式已启用")
                print("   将提取：最大尺寸、位置、毛刺征、钙化、边界清晰度、分叶征、胸膜凹陷征\n")
            else:
                print("📏 仅提取最大尺寸模式\n")
            
            # 获取模型路径
            model_path = self.model_path.get()
            
            if not self.is_running:
                print("\n⚠ 预测已停止")
                return
            
            if enable_features:
                # 使用特征提取模式
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
            
            # 保存结果
            if self.is_running:
                df.to_excel(self.output_path.get(), index=False)
                print(f"\n✓ 结果已保存至：{self.output_path.get()}")
                self.root.after(0, lambda: messagebox.showinfo(
                    "完成", f"预测完成！\n结果已保存至：{self.output_path.get()}"))
            else:
                # 用户停止了预测，保存部分结果
                df.to_excel(self.output_path.get(), index=False)
                print(f"\n✓ 部分结果已保存至：{self.output_path.get()}")
                print("✓ 检查点已保存，下次运行将从断点继续")
                self.root.after(0, lambda: messagebox.showinfo(
                    "已停止", f"预测已停止！\n部分结果已保存至：{self.output_path.get()}\n下次运行将从断点继续"))
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        
        finally:
            # 恢复UI状态
            self.root.after(0, self.reset_ui)
    
    def stop_prediction(self):
        """停止预测"""
        self.is_running = False
        set_stop_flag(True)
        self.status_label.config(text="正在停止...")
        print("\n⚠ 用户请求停止，正在保存检查点...")
    
    def reset_ui(self):
        """重置UI状态"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="就绪")
        
        # 恢复标准输出
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def main():
    root = tk.Tk()
    app = MedicalPredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
