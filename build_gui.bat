@echo off
chcp 65001 >nul

REM 激活 conda 环境
echo ========================================
echo 激活 conda 环境: node_extractor
echo ========================================
call C:\Users\User\miniconda3\Scripts\activate.bat C:\Users\User\miniconda3
call conda activate node_extractor
if errorlevel 1 (
    echo.
    echo ❌ 激活 conda 环境失败！
    pause
    exit /b 1
)
echo ✓ 环境激活成功
echo.

echo ========================================
echo 开始打包 GUI 版本
echo ========================================
echo.

echo [1/3] 清理旧的构建文件...
if exist "dist\医疗影像报告预测工具" rmdir /s /q "dist\医疗影像报告预测工具"
if exist "build" rmdir /s /q "build"
echo 完成
echo.

echo [2/3] 使用 PyInstaller 打包...
pyinstaller gui.spec --clean
if errorlevel 1 (
    echo.
    echo ❌ 打包失败！请检查错误信息
    pause
    exit /b 1
)
echo 完成
echo.

echo [3/3] 复制必要文件到发布目录...
if not exist "dist\医疗影像报告预测工具\models" mkdir "dist\医疗影像报告预测工具\models"
if not exist "dist\医疗影像报告预测工具\_vcredist" mkdir "dist\医疗影像报告预测工具\_vcredist"

REM 复制示例文件
if exist "test.xlsx" copy "test.xlsx" "dist\医疗影像报告预测工具\"
if exist "config.py" copy "config.py" "dist\医疗影像报告预测工具\config_example.py"
if exist "config_private.py.example" copy "config_private.py.example" "dist\医疗影像报告预测工具\"

REM 复制VC++运行库
if exist "_vcredist\vc_redist.x64.exe" (
    copy "_vcredist\vc_redist.x64.exe" "dist\医疗影像报告预测工具\_vcredist\"
    echo ✓ 已复制 VC++ 运行库
) else (
    echo ⚠ 警告: 未找到 VC++ 运行库，请手动下载并放置到 _vcredist 目录
)

REM 复制模型文件（已禁用自动复制）
REM if exist "models\*.gguf" (
REM     copy "models\*.gguf" "dist\医疗影像报告预测工具\models\"
REM     echo ✓ 已复制模型文件
REM ) else (
REM     echo ⚠ 警告: 未找到模型文件，请手动复制到 models 目录
REM )
echo ⚠ 注意: 模型文件需要手动复制到 models 目录

echo 完成
echo.

echo ========================================
echo ✓ 打包完成！
echo ========================================
echo.
echo 发布目录: dist\医疗影像报告预测工具\
echo.
echo 下一步:
echo 1. 将模型文件复制到 dist\医疗影像报告预测工具\models\
echo 2. 确保 _vcredist\vc_redist.x64.exe 已复制
echo 3. 压缩整个文件夹为 ZIP 或使用安装包制作工具
echo.
pause
