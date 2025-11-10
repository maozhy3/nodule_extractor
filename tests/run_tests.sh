#!/bin/bash
# 运行单元测试脚本

echo "========================================"
echo "运行单元测试"
echo "========================================"
echo

# 检查是否安装了 pytest
if ! python -c "import pytest" 2>/dev/null; then
    echo "❌ 未安装 pytest，正在安装..."
    pip install pytest pytest-cov
    if [ $? -ne 0 ]; then
        echo "❌ 安装失败，请手动运行: pip install pytest pytest-cov"
        exit 1
    fi
fi

echo "[1/2] 运行所有测试..."
echo
pytest tests/ -v

if [ $? -ne 0 ]; then
    echo
    echo "❌ 部分测试失败"
    exit 1
fi

echo
echo "[2/2] 生成测试覆盖率报告..."
echo
pytest tests/ --cov=. --cov-report=html --cov-report=term

echo
echo "========================================"
echo "✓ 测试完成！"
echo "========================================"
echo
echo "覆盖率报告已生成到: htmlcov/index.html"
echo
