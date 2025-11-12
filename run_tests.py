#!/usr/bin/env python3
"""
run_tests.py - Python 测试运行脚本
跨平台兼容，支持 Windows/macOS/Linux
"""
import sys
import subprocess
from pathlib import Path


class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

    @staticmethod
    def strip_if_windows():
        if sys.platform == 'win32':
            Colors.GREEN = Colors.YELLOW = Colors.RED = Colors.NC = ''


Colors.strip_if_windows()


def print_header(msg):
    print(f"\n{Colors.GREEN}{msg}{Colors.NC}")


def print_warning(msg):
    print(f"{Colors.YELLOW}{msg}{Colors.NC}")


def print_error(msg):
    print(f"{Colors.RED}{msg}{Colors.NC}")


def check_venv():
    """检查并提示虚拟环境"""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and
                                                sys.base_prefix != sys.prefix):
        venv_path = Path('.venv')
        if venv_path.exists():
            print_warning("警告: 检测到 .venv 但未激活")
            if sys.platform == 'win32':
                print("  请运行: .venv\\Scripts\\activate")
            else:
                print("  请运行: source .venv/bin/activate")
        else:
            print_warning("警告: 未检测到虚拟环境")
            print("  建议创建: python -m venv .venv")


def check_dependencies():
    """检查并安装必要依赖"""
    print_header("[1/3] 检查依赖...")

    # 检查 pytest
    try:
        import pytest  # noqa: F401
        print("  ✓ pytest 已安装")
    except ImportError:
        print_warning("  pytest 未安装，正在安装开发依赖...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-e', '.[dev]'],
                check=True,
                cwd=Path(__file__).parent)
        except subprocess.CalledProcessError:
            print_error("依赖安装失败，请检查 pyproject.toml")
            sys.exit(1)

    # 检查 matplotlib（可选）
    try:
        import matplotlib  # noqa: F401
        print("  ✓ matplotlib 可用，将生成几何图片")
        return True
    except ImportError:
        print_warning("  ⚠ matplotlib 未安装，几何图生成测试将跳过")
        print("    安装命令: pip install matplotlib")
        return False


def run_tests(args):
    """运行测试套件"""
    print_header("[2/3] 运行测试套件...")

    # 默认参数
    pytest_args = ['-v']
    test_path = 'test/'
    coverage_mode = False

    # 解析参数
    i = 1
    while i < len(args):
        arg = args[i]
        if arg in ('-q', '--quiet'):
            pytest_args = ['-q']
        elif arg in ('-vv', '--verbose'):
            pytest_args = ['-vv']
        elif arg in ('--cov', '--coverage'):
            coverage_mode = True
        elif arg == '--grid-only':
            test_path = 'test/test_grid_tom.py'
        elif arg in ('-h', '--help'):
            print_help()
            sys.exit(0)
        else:
            test_path = arg
        i += 1

    # 构建命令
    cmd = [sys.executable, '-m', 'pytest'] + pytest_args

    if coverage_mode:
        try:
            import pytest_cov  # noqa: F401
            cmd.extend(
                ['--cov=core', '--cov=pyzeetom', '--cov-report=term-missing'])
        except ImportError:
            print_warning("pytest-cov 未安装，运行普通测试...")

    cmd.append(test_path)

    # 运行测试
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def print_help():
    print("""
用法: python run_tests.py [选项]

选项:
  -q, --quiet        简洁输出
  -vv, --verbose     详细输出
  --cov, --coverage  启用代码覆盖率
  --grid-only        仅运行几何网格测试
  -h, --help         显示此帮助信息
  
示例:
  python run_tests.py              # 运行所有测试（默认详细模式）
  python run_tests.py -q           # 简洁模式
  python run_tests.py --grid-only  # 仅运行网格测试
  python run_tests.py --cov        # 带覆盖率报告
""")


def main():
    print(f"{Colors.GREEN}=== pyZeeTom 测试运行器 ==={Colors.NC}")

    check_venv()
    matplotlib_available = check_dependencies()
    exit_code = run_tests(sys.argv)

    # 总结
    print_header("[3/3] 测试总结")
    if exit_code == 0:
        print(f"{Colors.GREEN}✓ 所有测试通过{Colors.NC}")
        if matplotlib_available:
            print(f"\n{Colors.GREEN}提示：几何图片已在测试期间生成（临时目录）{Colors.NC}")
            print("若需保留示例图片，可运行:")
            print(
                '  python -c "from core.grid_tom import visualize_grid; '
                'visualize_grid(nr=20, r_in=0, r_out=200, save_path=\'grid_example.png\', show=False)"'
            )
    else:
        print_error(f"✗ 部分测试失败 (exit code: {exit_code})")

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
