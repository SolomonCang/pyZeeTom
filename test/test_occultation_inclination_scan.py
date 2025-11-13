"""
遮挡效应倾角扫描测试

测试不同轨道倾角下的恒星遮挡效应：
- 倾角范围：0° - 90°（步长15°）
- 固定10个发射spot（0-2R*，径向分布）
- 生成对比图：
  1. 各倾角的遮挡区域分布
  2. 遮挡比例 vs 倾角曲线
  3. 单个相位下不同倾角的光谱对比
  4. 动态谱对比（选择性展示几个典型倾角）

物理预期：
- i=0° (face-on): 无遮挡，所有像素可见
- i=30°: 轻微遮挡，~8%像素
- i=60°: 显著遮挡，~25%像素
- i=90° (edge-on): 最大遮挡，~50%像素
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# 添加项目路径
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from core.mainFuncs import readParamsTomog
from core.grid_tom import diskGrid
from core.local_linemodel_basic import LineData, GaussianZeemanWeakLineModel
from core.spot_geometry import Spot, SpotCollection, TimeEvolvingSpotGeometry


def plot_occultation_vs_inclination(grid, inclinations, occ_fractions, phi_obs,
                                    output_dir):
    """绘制遮挡比例 vs 倾角曲线"""
    print("\n生成遮挡比例vs倾角曲线...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # 绘制数据点和曲线
    ax.plot(inclinations,
            occ_fractions * 100,
            'o-',
            color='red',
            linewidth=2.5,
            markersize=10,
            markeredgewidth=2,
            markeredgecolor='darkred',
            label='Computed')

    # 理论预期（简化模型）
    # 对于赤道盘，遮挡比例 ≈ (1 - cos(i)) / 2 * (投影效应)
    # 实际更复杂，这里仅作参考
    inc_theory = np.linspace(0, 90, 100)
    inc_rad_theory = np.deg2rad(inc_theory)
    # 简化理论：sin(i) 依赖（投影面积）
    occ_theory = 50 * np.sin(inc_rad_theory)  # 最大50%
    ax.plot(inc_theory,
            occ_theory,
            '--',
            color='gray',
            linewidth=2,
            alpha=0.7,
            label='Simplified theory: 50·sin(i)')

    ax.set_xlabel('Inclination (degrees)', fontsize=13)
    ax.set_ylabel('Occluded Fraction (%)', fontsize=13)
    ax.set_title(
        'Stellar Occultation vs Inclination\n'
        f'Observer at φ={np.degrees(phi_obs):.0f}°, R*=1.0, Disk extends to 2R*',
        fontsize=14,
        fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=11, loc='upper left')

    # 标注关键点
    for inc, frac in zip(inclinations, occ_fractions):
        ax.annotate(f'{frac*100:.1f}%',
                    xy=(inc, frac * 100),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    color='darkred',
                    fontweight='bold')

    ax.set_xlim(-5, 95)
    ax.set_ylim(-2, 52)

    plt.tight_layout()
    fname = os.path.join(output_dir, 'occultation_vs_inclination.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {fname}")


def plot_occultation_maps_grid(grid, inclinations, masks_dict, phi_obs,
                               output_dir):
    """绘制不同倾角下的遮挡区域对比（网格布局）"""
    print("\n生成遮挡区域对比图...")

    n_inc = len(inclinations)
    ncols = 3
    nrows = (n_inc + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten() if n_inc > 1 else [axes]

    x = grid.r * np.cos(grid.phi)
    y = grid.r * np.sin(grid.phi)

    for idx, inc in enumerate(inclinations):
        ax = axes[idx]
        mask = masks_dict[inc]

        # 可见像素（蓝色）
        visible = ~mask
        ax.scatter(x[visible],
                   y[visible],
                   c='skyblue',
                   s=10,
                   alpha=0.5,
                   label='Visible')

        # 遮挡像素（红色）
        ax.scatter(x[mask],
                   y[mask],
                   c='red',
                   s=10,
                   alpha=0.8,
                   label='Occluded')

        # 恒星轮廓
        theta = np.linspace(0, 2 * np.pi, 100)
        stellar_x = np.cos(theta)
        stellar_y = np.sin(theta)
        ax.plot(stellar_x, stellar_y, 'k-', linewidth=2, alpha=0.7)
        ax.fill(stellar_x, stellar_y, color='yellow', alpha=0.2)

        # 观察者方向
        obs_len = 2.3
        obs_x = obs_len * np.cos(phi_obs)
        obs_y = obs_len * np.sin(phi_obs)
        ax.arrow(0,
                 0,
                 obs_x,
                 obs_y,
                 head_width=0.12,
                 head_length=0.15,
                 fc='green',
                 ec='green',
                 linewidth=2,
                 alpha=0.8)

        ax.set_aspect('equal')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel('x (R*)', fontsize=10)
        ax.set_ylabel('y (R*)', fontsize=10)

        n_occ = np.sum(mask)
        occ_pct = n_occ / len(mask) * 100
        ax.set_title(
            f'i = {inc:.0f}°\nOccluded: {n_occ}/{len(mask)} ({occ_pct:.1f}%)',
            fontsize=11,
            fontweight='bold')
        ax.grid(True, alpha=0.2)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # 隐藏多余子图
    for idx in range(n_inc, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Occultation Region vs Inclination (Disk Coordinate Frame)',
                 fontsize=14,
                 fontweight='bold')
    plt.tight_layout()

    fname = os.path.join(output_dir, 'occultation_maps_grid.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {fname}")


def plot_spectra_comparison(inclinations, spectra_dict_I, spectra_dict_V,
                            v_range, phase, output_dir):
    """对比不同倾角下的光谱（单个相位）"""
    print(f"\n生成光谱对比图 (phase={phase:.3f})...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(inclinations)))

    # 左：Stokes I
    ax = axes[0]
    for inc, color in zip(inclinations, colors):
        I = spectra_dict_I[inc]
        ax.plot(v_range,
                I,
                linewidth=2,
                color=color,
                label=f'i={inc:.0f}°',
                alpha=0.8)

    ax.set_xlabel('Velocity (km/s)', fontsize=12)
    ax.set_ylabel('Stokes I', fontsize=12)
    ax.set_title(f'Stokes I Comparison (Phase {phase:.3f})',
                 fontsize=13,
                 fontweight='bold')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right', ncol=2)

    # 右：Stokes V
    ax = axes[1]
    for inc, color in zip(inclinations, colors):
        V = spectra_dict_V[inc]
        ax.plot(v_range,
                V,
                linewidth=2,
                color=color,
                label=f'i={inc:.0f}°',
                alpha=0.8)

    ax.set_xlabel('Velocity (km/s)', fontsize=12)
    ax.set_ylabel('Stokes V', fontsize=12)
    ax.set_title(f'Stokes V Comparison (Phase {phase:.3f})',
                 fontsize=13,
                 fontweight='bold')
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right', ncol=2)

    # 自动设置V范围
    all_V = np.concatenate([spectra_dict_V[inc] for inc in inclinations])
    vlim = np.max(np.abs(all_V))
    if vlim > 0:
        ax.set_ylim(-1.1 * vlim, 1.1 * vlim)

    plt.suptitle(
        f'Spectral Comparison at Different Inclinations\n'
        f'With Stellar Occultation (Phase={phase:.3f})',
        fontsize=14,
        fontweight='bold')
    plt.tight_layout()

    fname = os.path.join(output_dir,
                         f'spectra_comparison_phase{phase:.3f}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {fname}")


def plot_dynamic_spectra_selected(inclinations_sel, phases, v_range,
                                  spectra_I_dict, spectra_V_dict, output_dir):
    """绘制选定几个倾角的动态谱对比"""
    print("\n生成动态谱对比 (选定倾角)...")

    n_inc = len(inclinations_sel)
    fig, axes = plt.subplots(n_inc, 2, figsize=(14, 5 * n_inc))

    if n_inc == 1:
        axes = axes.reshape(1, -1)

    extent = [v_range[0], v_range[-1], phases[0], phases[-1]]

    for idx, inc in enumerate(inclinations_sel):
        spectra_I = spectra_I_dict[inc]
        spectra_V = spectra_V_dict[inc]

        # 左：Stokes I
        ax = axes[idx, 0]
        im = ax.imshow(spectra_I,
                       aspect='auto',
                       origin='lower',
                       extent=extent,
                       cmap='hot',
                       interpolation='bilinear')
        ax.set_xlabel('Velocity (km/s)', fontsize=11)
        ax.set_ylabel('Phase', fontsize=11)
        ax.set_title(f'Stokes I (i={inc:.0f}°)',
                     fontsize=12,
                     fontweight='bold')
        plt.colorbar(im, ax=ax, label='Intensity', fraction=0.046)
        ax.grid(True, alpha=0.3, color='white', linestyle='--')

        # 右：Stokes V
        ax = axes[idx, 1]
        vmax = np.max(np.abs(spectra_V))
        im = ax.imshow(spectra_V,
                       aspect='auto',
                       origin='lower',
                       extent=extent,
                       cmap='RdBu_r',
                       vmin=-vmax,
                       vmax=vmax,
                       interpolation='bilinear')
        ax.set_xlabel('Velocity (km/s)', fontsize=11)
        ax.set_ylabel('Phase', fontsize=11)
        ax.set_title(f'Stokes V (i={inc:.0f}°)',
                     fontsize=12,
                     fontweight='bold')
        plt.colorbar(im, ax=ax, label='Circular Polarization', fraction=0.046)
        ax.grid(True, alpha=0.3, color='black', linestyle='--', linewidth=0.5)

    plt.suptitle('Dynamic Spectra at Selected Inclinations (With Occultation)',
                 fontsize=14,
                 fontweight='bold',
                 y=0.995)
    plt.tight_layout()

    fname = os.path.join(output_dir,
                         'dynamic_spectra_selected_inclinations.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {fname}")


def compute_spectrum(grid, line_data, linemodel, brightness, Br, Bphi,
                     inclination_deg, vsini, pOmega, r0, phi_obs,
                     enable_occultation, stellar_radius):
    """计算光谱（支持遮挡）"""
    incl_rad = np.deg2rad(inclination_deg)

    # 计算遮挡mask
    occultation_mask = np.zeros(grid.numPoints, dtype=bool)
    if enable_occultation:
        occultation_mask = grid.compute_stellar_occultation_mask(
            phi_obs, inclination_deg, stellar_radius, verbose=0)

    # 视向磁场
    Blos = Br * np.sin(incl_rad) * np.cos(grid.phi - phi_obs)
    Bperp = np.sqrt(Br**2 + Bphi**2 - Blos**2)
    Bperp = np.maximum(Bperp, 0.0)
    chi = grid.phi

    # 环向速度（差速转动）
    v0_r0 = vsini / np.sin(incl_rad) if np.sin(incl_rad) > 1e-6 else 0.0
    v_phi = v0_r0 * (grid.r / r0)**pOmega

    # 视向速度
    v_los = v_phi * np.sin(incl_rad) * np.sin(grid.phi - phi_obs)

    # 速度网格
    v_range = np.linspace(-200, 200, 401)
    c = 2.99792458e5
    wl_obs = line_data.wl0 * (1 + v_range / c)
    wl_local = wl_obs[:, None] / (1.0 + v_los[None, :] / c)

    # 应用遮挡
    amp_local = brightness.copy()
    Ic_weight = grid.area.copy()
    if enable_occultation:
        amp_local[occultation_mask] = 0.0
        Ic_weight[occultation_mask] = 0.0

    # 计算谱线
    profiles = linemodel.compute_local_profile(wl_local,
                                               amp=amp_local,
                                               Blos=Blos,
                                               Bperp=Bperp,
                                               chi=chi,
                                               Ic_weight=Ic_weight)

    stokes_i = np.sum(profiles['I'], axis=1)
    stokes_v = np.sum(profiles['V'], axis=1)
    total_area = np.sum(Ic_weight)
    if total_area > 0:
        stokes_i = stokes_i / total_area
        stokes_v = stokes_v / total_area

    return v_range, stokes_i, stokes_v, occultation_mask


def main():
    output_dir = 'test_output/occultation_inclination_scan'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("恒星遮挡效应倾角扫描测试")
    print("=" * 70)

    # 读取参数
    par = readParamsTomog('input/params_tomog.txt', verbose=1)

    # 倾角范围（0° - 90°，步长15°）
    inclinations = np.array([0, 15, 30, 45, 60, 75, 90])
    print(f"\n=== 倾角扫描范围 ===")
    print(f"  倾角: {inclinations}°")
    print(f"  步长: 15°")
    print(f"  观察者方向: φ=0° (固定)\n")

    # 网格和谱线
    pOmega = getattr(par, 'pOmega', 0.0)
    radius = getattr(par, 'radius', 1.0)
    Vmax = getattr(par, 'Vmax', 0.0)
    velEq = getattr(par, 'velEq', getattr(par, 'vsini', 100.0))

    if abs(pOmega + 1.0) > 1e-6:
        r_out_grid = radius * (Vmax / velEq)**(1.0 / (pOmega + 1.0))
    else:
        r_out_grid = getattr(par, 'r_out', 5.0 * radius)

    line_data = LineData('input/lines.txt')
    linemodel = GaussianZeemanWeakLineModel(
        line_data,
        k_QU=getattr(par, 'lineKQU', 1.0),
        enable_V=bool(getattr(par, 'lineEnableV', 1)),
        enable_QU=bool(getattr(par, 'lineEnableQU', 1)))

    grid = diskGrid(nr=60, r_in=0.0, r_out=r_out_grid, verbose=1)

    # 创建10个发射spot
    n_spots = 10
    rs = np.linspace(0.0, 2.0, n_spots)
    phi_initial = 0.0
    spots = []
    for r in rs:
        spots.append(
            Spot(r=r,
                 phi_initial=phi_initial,
                 amplitude=2.0,
                 spot_type='emission',
                 radius=0.25,
                 B_amplitude=1000,
                 B_direction='radial'))

    spot_collection = SpotCollection(spots=spots)
    spot_collection.pOmega = par.pOmega
    spot_collection.r0 = radius
    spot_collection.period = par.period
    spot_geometry = TimeEvolvingSpotGeometry(grid, spot_collection)

    # 相位设置
    HJD0 = getattr(par, 'jDateRef', 0.5)
    period = par.period
    n_phases = 8
    HJDs = np.linspace(HJD0, HJD0 + period, n_phases, endpoint=False)
    phases = (HJDs - HJD0) / period

    phi_obs = 0.0  # 观察者方向固定

    # 第一步：计算各倾角的遮挡mask（phase=0时的spot分布）
    print("\n=== 计算各倾角的遮挡mask ===")
    brightness, Br, Bphi = spot_geometry.generate_distributions(0.0)

    masks_dict = {}
    occ_fractions = []

    for inc in inclinations:
        mask = grid.compute_stellar_occultation_mask(phi_obs,
                                                     inc,
                                                     radius,
                                                     verbose=0)
        masks_dict[inc] = mask
        n_occ = np.sum(mask)
        frac = n_occ / grid.numPoints
        occ_fractions.append(frac)
        print(
            f"  i={inc:2.0f}°: {n_occ:4d}/{grid.numPoints} ({frac*100:5.2f}%)")

    occ_fractions = np.array(occ_fractions)

    # 生成遮挡统计图
    plot_occultation_vs_inclination(grid, inclinations, occ_fractions, phi_obs,
                                    output_dir)
    plot_occultation_maps_grid(grid, inclinations, masks_dict, phi_obs,
                               output_dir)

    # 第二步：计算各倾角、各相位的光谱（用于动态谱）
    print("\n=== 计算各倾角的时间序列光谱 ===")

    # 存储所有倾角的动态谱
    dynamic_I_dict = {inc: [] for inc in inclinations}
    dynamic_V_dict = {inc: [] for inc in inclinations}

    # 对每个相位
    for phase_idx, (phase, HJD) in enumerate(zip(phases, HJDs)):
        print(f"\n相位 {phase_idx}: phase={phase:.3f}, HJD={HJD:.3f}")
        brightness, Br, Bphi = spot_geometry.generate_distributions(phase)

        # 对每个倾角计算光谱
        for inc in inclinations:
            v_range, I, V, _ = compute_spectrum(grid,
                                                line_data,
                                                linemodel,
                                                brightness,
                                                Br,
                                                Bphi,
                                                inclination_deg=inc,
                                                vsini=par.vsini,
                                                pOmega=par.pOmega,
                                                r0=radius,
                                                phi_obs=phi_obs,
                                                enable_occultation=True,
                                                stellar_radius=radius)

            dynamic_I_dict[inc].append(I)
            dynamic_V_dict[inc].append(V)

        print(f"  已完成{len(inclinations)}个倾角的光谱计算")

    # 转换为numpy数组
    for inc in inclinations:
        dynamic_I_dict[inc] = np.array(dynamic_I_dict[inc])
        dynamic_V_dict[inc] = np.array(dynamic_V_dict[inc])

    # 第三步：生成单相位光谱对比（选择phase=0.5，spot在背面）
    print("\n=== 生成单相位光谱对比 ===")
    phase_sel = 0.5
    phase_idx_sel = np.argmin(np.abs(phases - phase_sel))
    actual_phase = phases[phase_idx_sel]

    spectra_I_single = {
        inc: dynamic_I_dict[inc][phase_idx_sel]
        for inc in inclinations
    }
    spectra_V_single = {
        inc: dynamic_V_dict[inc][phase_idx_sel]
        for inc in inclinations
    }

    plot_spectra_comparison(inclinations, spectra_I_single, spectra_V_single,
                            v_range, actual_phase, output_dir)

    # 第四步：生成选定倾角的动态谱对比
    print("\n=== 生成动态谱对比 ===")
    inclinations_sel = [0, 30, 60, 90]  # 选择4个典型倾角
    plot_dynamic_spectra_selected(inclinations_sel, phases, v_range,
                                  dynamic_I_dict, dynamic_V_dict, output_dir)

    print("\n" + "=" * 70)
    print("倾角扫描测试完成！")
    print(f"输出目录: {output_dir}")
    print(f"生成文件:")
    print(f"  - occultation_vs_inclination.png (遮挡比例曲线)")
    print(f"  - occultation_maps_grid.png (遮挡区域对比)")
    print(f"  - spectra_comparison_phase{actual_phase:.3f}.png (单相位光谱对比)")
    print(f"  - dynamic_spectra_selected_inclinations.png (动态谱对比)")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
