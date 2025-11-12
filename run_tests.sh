#!/usr/bin/env bash
# run_tests.sh - 便捷测试运行脚本
# 支持不同测试场景，自动检查依赖，生成几何图片

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== pyZeeTom 测试运行器 ===${NC}\n"

# 检查虚拟环境
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -d ".venv" ]]; then
        echo -e "${YELLOW}检测到 .venv 但未激活，尝试自动激活...${NC}"
        source .venv/bin/activate || {
            echo -e "${RED}激活失败，请手动运行: source .venv/bin/activate${NC}"
            exit 1
        }
    else
        echo -e "${YELLOW}警告: 未检测到虚拟环境，建议先运行:${NC}"
        echo "  python -m venv .venv && source .venv/bin/activate"
    fi
fi

# 检查并安装依赖
echo -e "${GREEN}[1/3] 检查依赖...${NC}"
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}pytest 未安装，正在安装开发依赖...${NC}"
    pip install -e .[dev] || {
        echo -e "${RED}依赖安装失败，请检查 pyproject.toml${NC}"
        exit 1
    }
fi

# 检查可选的 matplotlib（用于几何图生成）
MATPLOTLIB_AVAILABLE=false
if python -c "import matplotlib" 2>/dev/null; then
    MATPLOTLIB_AVAILABLE=true
    echo "  ✓ matplotlib 可用，将生成几何图片"
else
    echo -e "${YELLOW}  ⚠ matplotlib 未安装，几何图生成测试将跳过${NC}"
    echo "    安装命令: pip install matplotlib"
fi

# 运行测试
echo -e "\n${GREEN}[2/3] 运行测试套件...${NC}"

# 解析命令行参数
TEST_PATH="test/"
PYTEST_ARGS="-v"
COVERAGE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--quiet)
            PYTEST_ARGS="-q"
            shift
            ;;
        -vv|--verbose)
            PYTEST_ARGS="-vv"
            shift
            ;;
        --cov|--coverage)
            COVERAGE_MODE=true
            shift
            ;;
        --grid-only)
            TEST_PATH="test/test_grid_tom.py"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -q, --quiet        简洁输出"
            echo "  -vv, --verbose     详细输出"
            echo "  --cov, --coverage  启用代码覆盖率"
            echo "  --grid-only        仅运行几何网格测试"
            echo "  -h, --help         显示此帮助信息"
            exit 0
            ;;
        *)
            TEST_PATH="$1"
            shift
            ;;
    esac
done

if $COVERAGE_MODE; then
    if python -c "import pytest_cov" 2>/dev/null; then
        pytest $PYTEST_ARGS --cov=core --cov=pyzeetom --cov-report=term-missing $TEST_PATH
    else
        echo -e "${YELLOW}pytest-cov 未安装，运行普通测试...${NC}"
        pytest $PYTEST_ARGS $TEST_PATH
    fi
else
    pytest $PYTEST_ARGS $TEST_PATH
fi

TEST_EXIT_CODE=$?

# 总结
echo -e "\n${GREEN}[3/3] 测试总结${NC}"
if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ 所有测试通过${NC}"
    
    if $MATPLOTLIB_AVAILABLE; then
        echo -e "\n${GREEN}提示：几何图片已在测试期间生成（临时目录）${NC}"
        echo "若需保留示例图片，可运行:"
        echo "  python -c \"from core.grid_tom import visualize_grid; visualize_grid(nr=20, r_in=0, r_out=200, save_path='grid_example.png', show=False)\""
    fi
else
    echo -e "${RED}✗ 部分测试失败 (exit code: $TEST_EXIT_CODE)${NC}"
    exit $TEST_EXIT_CODE
fi
