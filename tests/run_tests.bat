@echo off
chcp 65001 >nul
echo ========================================
echo 运行单元测试
echo ========================================
echo.

REM 检查是否安装了 pytest
python -c "import pytest" 2>nul
if errorlevel 1 (
    echo ❌ 未安装 pytest，正在安装...
    pip install pytest pytest-cov
    if errorlevel 1 (
        echo ❌ 安装失败，请手动运行: pip install pytest pytest-cov
        pause
        exit /b 1
    )
)

echo [1/2] 运行所有测试...
echo.
pytest tests/ -v

if errorlevel 1 (
    echo.
    echo ❌ 部分测试失败
    pause
    exit /b 1
)

echo.
echo [2/2] 生成测试覆盖率报告...
echo.
pytest tests/ --cov=. --cov-report=html --cov-report=term

echo.
echo ========================================
echo ✓ 测试完成！
echo ========================================
echo.
echo 覆盖率报告已生成到: htmlcov/index.html
echo.
pause
