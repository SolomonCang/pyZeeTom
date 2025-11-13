"""
测试：径向排列的混合spot链
----------------------------
设置10个spot延固定方位角（phi=0）从0.5R到5R排列
内圈5个为吸收线spot，外圈5个为发射线spot
恒星参数：vsini=50 km/s，差速转动（pOmega=-0.1）
测试完整流程：参数文件 -> 模型 -> 光谱 -> 可视化
"""
import sys
from pathlib import Path
import numpy as np
import subprocess

# 添加项目路径
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "core"))


def create_radial_spot_chain_model():
    """创建沿径向排列的spot链模型"""
    print("=== 步骤1: 创建径向spot链几何模型 ===")

    from grid_tom import diskGrid

    # 创建网格（更密集以便清晰显示spot）
    nrings = 50
    grid = diskGrid(nr=nrings, r_in=0.3, r_out=5.5, verbose=0)

    print(f"  网格: {len(grid.r)} 个节点, {nrings} 个环")
    print(f"  r 范围: [{grid.r.min():.2f}, {grid.r.max():.2f}]")

    n = len(grid.r)

    # 恒星参数
    vsini = 50.0  # km/s
    inclination = 60.0  # deg
    pOmega = -0.1  # 差速转动（用于测试）

    # 初始化物理场
    brightness = np.ones(n)  # 默认背景亮度
    Blos = np.zeros(n)
    Bperp = np.zeros(n)
    chi = np.zeros(n)

    # 创建10个吸收线spot沿固定方位角排列
    spot_phi = 0.0  # 方位角固定在0度（向上）
    spot_radii = np.linspace(0.5, 5.0, 10)  # 10个spot的半径位置
    # 使用极坐标度量的等尺寸spot：在 (Δr)^2 + (r*Δφ)^2 度量下半径固定
    spot_metric_radius = 0.25  # 等效"圆斑"半径（单位与r一致），与半径无关地保持视觉等大小

    spot_brightness_abs = 0.5  # 吸收线：亮度降低
    spot_brightness_emi = 1.5  # 发射线：亮度增强
    spot_Blos = 800.0  # 强磁场

    print("\n  创建10个径向排列的spot（内5个吸收，外5个发射）:")
    print(f"    方位角: {np.degrees(spot_phi):.1f}°")
    print(f"    半径位置: {spot_radii}")
    print(
        f"    吸收亮度: {spot_brightness_abs:.2f}, 发射亮度: {spot_brightness_emi:.2f}"
    )
    print(f"    Blos: {spot_Blos:.1f} G")

    # 应用spot
    for idx, spot_r in enumerate(spot_radii):
        # 前5个为吸收，后5个为发射
        spot_brightness = spot_brightness_abs if idx < 5 else spot_brightness_emi
        for i in range(n):
            # 规范化方位角差至 [-pi, pi]
            dphi = ((grid.phi[i] - spot_phi + np.pi) % (2 * np.pi)) - np.pi
            # 极坐标下的等距度量（以局部半径 spot_r 缩放角向距离）
            dist = np.sqrt((grid.r[i] - spot_r)**2 + (spot_r * dphi)**2)
            if dist <= 3 * spot_metric_radius:
                # 以 metric 半径为尺度的高斯权重，产生等尺寸“圆形”spot
                weight = np.exp(-0.5 * (dist / spot_metric_radius)**2)
                brightness[i] += (spot_brightness - 1.0) * weight
                Blos[i] += spot_Blos * weight

    # 保存模型
    output_dir = _root / "test_output" / "radial_spots"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "geomodel.tomog"

    with open(model_file, 'w') as f:
        f.write("# Geometric Model: Radial Spot Chain\n")
        f.write(f"# Inclination: {inclination}\n")
        f.write(f"# Vsini: {vsini}\n")
        f.write("# Phase: 0.0\n")
        f.write(f"# pOmega: {pOmega}\n")
        f.write(f"# Num_nodes: {n}\n")
        f.write("# Num_spots: 10\n")
        f.write(f"# Spot_radii: {','.join(map(str, spot_radii))}\n")
        f.write("# Columns: r phi area brightness Blos Bperp chi\n")
        f.write("#\n")

        for i in range(n):
            f.write(
                f"{grid.r[i]:.6f} {grid.phi[i]:.6f} {grid.area[i]:.6e} "
                f"{brightness[i]:.6f} {Blos[i]:.3f} {Bperp[i]:.3f} {chi[i]:.6f}\n"
            )

    print(f"\n✓ 已保存模型至 {model_file}")
    print(f"  亮度范围: [{brightness.min():.3f}, {brightness.max():.3f}]")
    print(f"  Blos 范围: [{Blos.min():.1f}, {Blos.max():.1f}] G")

    return model_file, vsini, inclination, pOmega


def create_parameter_file(vsini, inclination, pOmega):
    """创建参数文件"""
    print("\n=== 步骤2: 创建参数文件 ===")

    output_dir = _root / "test_output" / "radial_spots"
    param_file = output_dir / "params_tomog.txt"

    # 生成8个相位的观测
    n_phases = 8
    period = 1.0  # 天
    jd_ref = 0.0
    jds = np.linspace(jd_ref, jd_ref + period, n_phases, endpoint=False)

    with open(param_file, 'w') as f:
        f.write("# Radial Spot Chain Test - Parameter File\n")
        f.write("# 10 emission spots along phi=0, r=0.5-5.0\n")
        f.write("#\n")
        f.write("#0 inclination vsini period pOmega\n")
        f.write(f"{inclination:.1f} {vsini:.1f} {period:.1f} {pOmega:.2f}\n")
        f.write("#1 mass radius\n")
        f.write("1.0 1.0\n")
        f.write("#2 nRingsStellarGrid\n")
        f.write("50\n")
        f.write("#3 targetForm targetValue numIterations\n")
        f.write("C 1.0 0\n")
        f.write("#4 test_aim\n")
        f.write("1e-3\n")
        f.write("#5 lineAmpConst k_QU enableV enableQU\n")
        f.write("0.5 1.0 1 0\n")  # 发射线：正振幅
        f.write("#6 initMagFromFile initMagGeomFile\n")
        f.write("0 none\n")
        f.write("#7 fitBri chiScaleI brightEntScale\n")
        f.write("0 1.0 1.0\n")
        f.write("#8 fEntropyBright defaultBright maximumBright\n")
        f.write("1 1.0 2.0\n")
        f.write("#9 initBrightFromFile initBrightFile\n")
        f.write("0 none\n")
        f.write("#10 estimateStrenght\n")
        f.write("0\n")
        f.write("#11 instrumentRes lineParamFile\n")
        f.write("5.0 lines.txt\n")
        f.write("#12 velStart velEnd obsFileType\n")
        f.write("-150.0 150.0 lsd_i\n")
        f.write("#13 jDateRef\n")
        f.write(f"{jd_ref:.1f}\n")
        f.write("#14+ observation entries: filename JD velR\n")

        for i, jd in enumerate(jds):
            fname = f"obs/phase{i:03d}.lsd"
            f.write(f"{fname} {jd:.6f} 0.0\n")

    print(f"  参数文件: {param_file}")
    print(f"  观测相位数: {n_phases}")
    print(f"  恒星参数: vsini={vsini} km/s, inc={inclination}°, pOmega={pOmega}")

    return param_file


def visualize_model(model_file):
    """可视化几何模型"""
    print("\n=== 步骤3: 可视化几何模型 ===")

    output_dir = model_file.parent
    fig_polar = output_dir / "model_polar.png"
    fig_cart = output_dir / "model_cart.png"
    script_path = _root / "utils" / "visualize_geomodel.py"

    # 极坐标投影
    cmd = [
        sys.executable,
        str(script_path), "--model",
        str(model_file), "--out",
        str(fig_polar), "--projection", "polar"
    ]

    print("  生成极坐标投影...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        size = fig_polar.stat().st_size / 1024
        print(f"  ✓ 极坐标图: {fig_polar.name} ({size:.1f} KB)")
    else:
        print("  ✗ 极坐标可视化失败")
        if result.stderr:
            print(f"    {result.stderr}")

    # 笛卡尔投影
    cmd = [
        sys.executable,
        str(script_path), "--model",
        str(model_file), "--out",
        str(fig_cart), "--projection", "cart"
    ]

    print("  生成笛卡尔投影...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        size = fig_cart.stat().st_size / 1024
        print(f"  ✓ 笛卡尔图: {fig_cart.name} ({size:.1f} KB)")
    else:
        print("  ✗ 笛卡尔可视化失败")


def generate_synthetic_spectra(param_file, model_file, pOmega=-0.1, r0=None):
    """生成合成光谱（使用tomography正演）"""
    print("\n=== 步骤4: 生成合成光谱 ===")

    output_dir = model_file.parent

    # 创建lines.txt
    lines_file = output_dir / "lines.txt"
    with open(lines_file, 'w') as f:
        f.write("# wl0(nm) sigWl(nm) g\n")
        f.write("656.28 0.02 1.0\n")  # H-alpha

    print(f"  谱线参数文件: {lines_file}")

    # 创建obs目录（存放生成的光谱）
    obs_dir = output_dir / "obs"
    obs_dir.mkdir(exist_ok=True)

    # 生成简单的合成光谱（手动模拟，因为正演需要完整的tomography流程）
    # 这里我们创建基于spot位置的简化光谱
    print("  生成8个相位的合成光谱...")

    n_phases = 8
    n_vel = 300
    velocities = np.linspace(-150, 150, n_vel)

    # 读取spot位置
    spot_radii = np.linspace(0.5, 5.0, 10)
    vsini = 50.0
    inclination = 60.0
    # 参考半径 r0：若未提供，从模型点的最大半径估计
    if r0 is None:
        try:
            arr = np.loadtxt(model_file, comments='#', usecols=[0])
            if arr.size > 0:
                r0 = float(np.max(arr))
            else:
                r0 = max(spot_radii)
        except Exception:
            r0 = max(spot_radii)

    for phase_idx in range(n_phases):
        phase = phase_idx / n_phases  # 0-1
        rotation_angle = 2 * np.pi * phase

        # 计算每个spot的投影速度
        spectrum_I = np.ones(n_vel)
        spectrum_V = np.zeros(n_vel)

        for idx, spot_r in enumerate(spot_radii):
            # 差速转动：Ω(r) ~ (r/r0)^pOmega，phase 转为角位移
            scale = (spot_r / r0)**(pOmega) if r0 else 1.0
            spot_phi = rotation_angle * scale
            v_los = vsini * np.sin(spot_phi) * np.sin(np.radians(inclination))

            # 前5个吸收，后5个发射
            sigma_v = 10.0  # km/s
            if idx < 5:
                # 吸收线：负高斯轮廓
                line_profile = -0.3 * np.exp(-0.5 * (
                    (velocities - v_los) / sigma_v)**2)
            else:
                # 发射线：正高斯轮廓
                line_profile = 0.3 * np.exp(-0.5 * (
                    (velocities - v_los) / sigma_v)**2)

            spectrum_I += line_profile

            # Stokes V: 反对称
            spectrum_V += 0.05 * line_profile * np.sign(velocities - v_los)

        # 保存光谱（LSD格式）
        spec_file = obs_dir / f"phase{phase_idx:03d}.lsd"
        with open(spec_file, 'w') as f:
            f.write(f"# Synthetic spectrum, phase {phase:.3f}\n")
            f.write(f"{n_vel} 6\n")
            for j in range(n_vel):
                f.write(f"{velocities[j]:.3f} {spectrum_I[j]:.6e} 1e-4 "
                        f"{spectrum_V[j]:.6e} 1e-5 0.0 1e-5\n")

        print(f"    phase{phase_idx:03d}.lsd: v_max={v_los:.1f} km/s")

    print(f"  ✓ 已生成 {n_phases} 个光谱文件至 {obs_dir}")


def visualize_dynamic_spectrum():
    """生成动态谱"""
    print("\n=== 步骤5: 生成动态谱 ===")

    output_dir = _root / "test_output" / "radial_spots"
    spec_dir = output_dir / "obs"
    param_file = output_dir / "params_tomog.txt"
    dynspec_file = output_dir / "dynamic_spectrum.png"

    script_path = _root / "utils" / "dynamic_spec_plot.py"

    cmd = [
        sys.executable,
        str(script_path), "--spec_dir",
        str(spec_dir), "--param_file",
        str(param_file), "--out",
        str(dynspec_file), "--file_type", "lsd_pol"
    ]

    print("  执行动态谱绘制...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        if dynspec_file.exists():
            size = dynspec_file.stat().st_size / 1024
            print(f"  ✓ 动态谱: {dynspec_file.name} ({size:.1f} KB)")
        else:
            print("  ✗ 未生成动态谱文件")
    else:
        print("  ✗ 动态谱生成失败")
        if result.stderr:
            print(f"    错误: {result.stderr[:500]}")


def summary_results():
    """总结输出结果"""
    print("\n" + "=" * 60)
    print("测试完成！生成的文件:")
    print("=" * 60)

    output_dir = _root / "test_output" / "radial_spots"

    files = [
        ("几何模型", "geomodel.tomog"),
        ("参数文件", "params_tomog.txt"),
        ("谱线参数", "lines.txt"),
        ("模型可视化(极坐标)", "model_polar.png"),
        ("模型可视化(笛卡尔)", "model_cart.png"),
        ("动态谱", "dynamic_spectrum.png"),
        ("光谱文件目录", "obs/"),
    ]

    for desc, fname in files:
        fpath = output_dir / fname
        if fpath.exists():
            if fpath.is_dir():
                n_files = len(list(fpath.glob("*.lsd")))
                print(f"  ✓ {desc:20s}: {fname} ({n_files} files)")
            else:
                size = fpath.stat().st_size / 1024
                print(f"  ✓ {desc:20s}: {fname} ({size:.1f} KB)")
        else:
            print(f"  ✗ {desc:20s}: {fname} (未生成)")

    print(f"\n所有文件位于: {output_dir}")
    print("=" * 60 + "\n")


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("径向Spot链测试")
    print("=" * 60)
    print("配置:")
    print("  - 10个混合spot沿phi=0方位角排列（内5吸收+外5发射）")
    print("  - 半径范围: 0.5R - 5.0R")
    print("  - 恒星: vsini=50 km/s, 差速转动(pOmega=-0.1)")
    print("  - 观测: 8个相位，覆盖完整自转周期")
    print("=" * 60 + "\n")

    try:
        # 1. 创建几何模型
        model_file, vsini, inclination, pOmega = create_radial_spot_chain_model(
        )

        # 2. 创建参数文件
        param_file = create_parameter_file(vsini, inclination, pOmega)

        # 3. 可视化几何模型
        visualize_model(model_file)

        # 4. 生成合成光谱
        generate_synthetic_spectra(param_file, model_file)

        # 5. 生成动态谱
        visualize_dynamic_spectrum()

        # 6. 总结
        summary_results()

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
