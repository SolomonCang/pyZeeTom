"""
测试几何模型可视化工具
------------------
生成测试用的几何模型并可视化
"""
import sys
from pathlib import Path
import numpy as np

# 添加项目路径
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "core"))


def create_test_geomodel():
    """创建一个测试用的几何模型"""
    print("=== 测试1: 创建测试几何模型 ===")

    from grid_tom import diskGrid

    # 创建网格
    inclination = 60.0
    nrings = 30
    grid = diskGrid(nr=nrings, r_in=0.5, r_out=2.0, verbose=0)

    print(f"  网格参数: {len(grid.r)} 个节点, {nrings} 个环")

    # 直接创建一个简单的数据结构用于保存
    n = len(grid.r)

    # 亮度：两个暗斑（低纬度和高纬度）
    brightness = np.ones(n)
    for i in range(n):
        # 低纬度暗斑（phi ~ 0）
        if 0.8 < grid.r[i] < 1.2 and -np.pi / 4 < grid.phi[i] < np.pi / 4:
            brightness[i] = 0.7
        # 高纬度暗斑（phi ~ pi）
        if 1.5 < grid.r[i] < 1.8 and 3 * np.pi / 4 < grid.phi[
                i] < 5 * np.pi / 4:
            brightness[i] = 0.6

    # Blos：径向磁场结构（极性翻转）
    Blos = np.zeros(n)
    for i in range(n):
        # 正极在上半平面，负极在下半平面
        if grid.phi[i] < np.pi:
            Blos[i] = 300 * (1.0 + 0.5 * np.sin(3 * grid.phi[i]))
        else:
            Blos[i] = -300 * (1.0 + 0.5 * np.sin(3 * grid.phi[i]))
        # 随半径衰减
        Blos[i] *= np.exp(-(grid.r[i] - 1.0)**2 / 0.5)

    # Bperp：环向磁场结构
    Bperp = np.zeros(n)
    chi = np.zeros(n)
    for i in range(n):
        # 强度随半径和方位角变化
        Bperp[i] = 200 * np.sin(2 * grid.phi[i])**2 * np.exp(
            -(grid.r[i] - 1.5)**2 / 0.5)
        chi[i] = grid.phi[i] + np.pi / 2  # 环向

    # 手动构造geomodel文件
    output_dir = _root / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "test_geomodel.tomog"

    # 写入文件
    with open(model_file, 'w') as f:
        # Header
        f.write("# Geometric Model Export\n")
        f.write(f"# Inclination: {inclination}\n")
        f.write(f"# Vsini: 25.0\n")
        f.write(f"# Phase: 0.0\n")
        f.write(f"# Num_nodes: {n}\n")
        f.write("# Columns: r phi area brightness Blos Bperp chi\n")
        f.write("#\n")

        # Data
        for i in range(n):
            f.write(
                f"{grid.r[i]:.6f} {grid.phi[i]:.6f} {grid.area[i]:.6e} "
                f"{brightness[i]:.6f} {Blos[i]:.3f} {Bperp[i]:.3f} {chi[i]:.6f}\n"
            )

    print(f"✓ 已保存测试模型至 {model_file}")
    print(f"  亮度范围: [{brightness.min():.3f}, {brightness.max():.3f}]")
    print(f"  Blos 范围: [{Blos.min():.1f}, {Blos.max():.1f}] G")
    print(f"  Bperp 范围: [{Bperp.min():.1f}, {Bperp.max():.1f}] G")
    print()

    return model_file


def test_visualize_polar():
    """测试极坐标投影"""
    print("=== 测试2: 极坐标投影可视化 ===")

    import subprocess

    model_file = _root / "test_output" / "test_geomodel.tomog"
    out_fig = _root / "test_output" / "test_geomodel_polar.png"
    script_path = _root / "utils" / "visualize_geomodel.py"

    cmd = [
        sys.executable,
        str(script_path), "--model",
        str(model_file), "--out",
        str(out_fig), "--projection", "polar"
    ]

    print(f"  执行命令: {' '.join(cmd[-4:])}")

    try:
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                timeout=30)

        if result.returncode == 0:
            print(f"✓ 极坐标可视化成功")
            if out_fig.exists():
                size = out_fig.stat().st_size / 1024
                print(f"✓ 已生成图片: {out_fig.name} ({size:.1f} KB)")
            else:
                print(f"✗ 未找到输出图片")
            if result.stdout:
                print(f"  输出:\n{result.stdout}")
        else:
            print(f"✗ 可视化失败 (返回码 {result.returncode})")
            if result.stderr:
                print(f"  错误信息:\n{result.stderr}")

    except subprocess.TimeoutExpired:
        print("✗ 执行超时")
    except Exception as e:
        print(f"✗ 执行出错: {e}")

    print()


def test_visualize_cart():
    """测试笛卡尔投影"""
    print("=== 测试3: 笛卡尔投影可视化 ===")

    import subprocess

    model_file = _root / "test_output" / "test_geomodel.tomog"
    out_fig = _root / "test_output" / "test_geomodel_cart.png"
    script_path = _root / "utils" / "visualize_geomodel.py"

    cmd = [
        sys.executable,
        str(script_path), "--model",
        str(model_file), "--out",
        str(out_fig), "--projection", "cart"
    ]

    print(f"  执行命令: {' '.join(cmd[-4:])}")

    try:
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                timeout=30)

        if result.returncode == 0:
            print(f"✓ 笛卡尔可视化成功")
            if out_fig.exists():
                size = out_fig.stat().st_size / 1024
                print(f"✓ 已生成图片: {out_fig.name} ({size:.1f} KB)")
            else:
                print(f"✗ 未找到输出图片")
            if result.stdout:
                print(f"  输出:\n{result.stdout}")
        else:
            print(f"✗ 可视化失败 (返回码 {result.returncode})")
            if result.stderr:
                print(f"  错误信息:\n{result.stderr}")

    except subprocess.TimeoutExpired:
        print("✗ 执行超时")
    except Exception as e:
        print(f"✗ 执行出错: {e}")

    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("几何模型可视化工具测试套件")
    print("=" * 60 + "\n")

    try:
        model_file = create_test_geomodel()
        test_visualize_polar()
        test_visualize_cart()

        print("=" * 60)
        print("测试完成！")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
