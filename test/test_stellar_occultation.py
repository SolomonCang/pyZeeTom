"""
恒星遮挡效果测试

对比开启和关闭恒星遮挡时的光谱差异：
- 10个发射spot沿径向分布（0-2R*）
- 倾角60°，可以看到明显的遮挡效应
- 生成四组对比图：
  1. 单相位对比（有/无遮挡）
  2. 动态谱对比
  3. 遮挡mask可视化（盘面坐标系）
  4. 遮挡区域示意图

物理图像：
- 观察者方向固定（phi_obs=0，即"向上"看）
- 盘面网格坐标固定（r, phi不随时间变化）
- 遮挡区域在盘面坐标系中固定（phi≈180°方向，恒星背后）
- spot随差速转动进出遮挡区域
- 遮挡导致可见spot数量和位置变化，影响光谱形态
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


def plot_comparison_panel(grid,
                          brightness_no_occ,
                          brightness_with_occ,
                          occultation_mask,
                          spots,
                          phase,
                          wl,
                          I_no_occ,
                          I_with_occ,
                          V_no_occ,
                          V_with_occ,
                          output_dir,
                          idx,
                          phi_obs=0.0):
    """绘制对比图：无遮挡 vs 有遮挡
    
    上半部分：无遮挡（spot分布 + Stokes I + Stokes V）
    下半部分：有遮挡（spot分布+mask + Stokes I + Stokes V）
    """
    x = grid.r * np.cos(grid.phi)
    y = grid.r * np.sin(grid.phi)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ==================== 上半部分：无遮挡 ====================
    # 左上：spot分布（无遮挡）
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(
        x,
        y,
        c=brightness_no_occ,
        s=20,
        cmap='YlOrRd',
        vmin=0,
        vmax=np.max(brightness_no_occ) if np.max(brightness_no_occ) > 0 else 3)

    # 标记spot位置
    for i, spot in enumerate(spots):
        x_c = spot.r * np.cos(spot.phi_initial)
        y_c = spot.r * np.sin(spot.phi_initial)
        ax1.plot(x_c,
                 y_c,
                 'o',
                 color='blue',
                 markersize=8,
                 markeredgewidth=1.5,
                 markeredgecolor='black')
        ax1.text(x_c,
                 y_c,
                 f'{i}',
                 color='white',
                 fontsize=7,
                 ha='center',
                 va='center',
                 fontweight='bold')

    # 观察者方向
    obs_x = 2.5 * np.cos(phi_obs)
    obs_y = 2.5 * np.sin(phi_obs)
    ax1.arrow(0,
              0,
              obs_x,
              obs_y,
              head_width=0.12,
              head_length=0.15,
              fc='green',
              ec='green',
              linewidth=2,
              alpha=0.8)
    ax1.text(obs_x * 1.15,
             obs_y * 1.15,
             'Obs',
             color='green',
             fontsize=10,
             fontweight='bold',
             ha='center')

    ax1.set_aspect('equal')
    ax1.set_xlim(-2.8, 2.8)
    ax1.set_ylim(-2.8, 2.8)
    ax1.set_xlabel('x (R*)', fontsize=11)
    ax1.set_ylabel('y (R*)', fontsize=11)
    ax1.set_title(f'Phase {phase:.3f} - No Occultation',
                  fontsize=12,
                  fontweight='bold')
    plt.colorbar(sc, ax=ax1, label='Brightness', fraction=0.046)
    ax1.grid(True, alpha=0.2)

    # 中上：Stokes I（无遮挡）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(wl,
             I_no_occ,
             color='black',
             linewidth=1.5,
             label='No Occultation')
    ax2.set_xlabel('Velocity (km/s)', fontsize=11)
    ax2.set_ylabel('Stokes I', fontsize=11)
    ax2.set_title('Stokes I - No Occultation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.legend(fontsize=9)

    # 右上：Stokes V（无遮挡）
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(wl, V_no_occ, color='red', linewidth=1.5, label='No Occultation')
    ax3.set_xlabel('Velocity (km/s)', fontsize=11)
    ax3.set_ylabel('Stokes V', fontsize=11)
    ax3.set_title('Stokes V - No Occultation', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    vlim = np.max(np.abs(V_no_occ))
    if vlim > 0:
        ax3.set_ylim(-1.1 * vlim, 1.1 * vlim)
    ax3.legend(fontsize=9)

    # ==================== 下半部分：有遮挡 ====================
    # 左下：spot分布+遮挡mask
    ax4 = fig.add_subplot(gs[1, 0])

    # 显示亮度（被遮挡的像素设为0）
    brightness_masked = brightness_with_occ.copy()
    sc2 = ax4.scatter(
        x,
        y,
        c=brightness_masked,
        s=20,
        cmap='YlOrRd',
        vmin=0,
        vmax=np.max(brightness_no_occ) if np.max(brightness_no_occ) > 0 else 3)

    # 用半透明灰色标记被遮挡区域
    if np.any(occultation_mask):
        x_occ = x[occultation_mask]
        y_occ = y[occultation_mask]
        ax4.scatter(x_occ,
                    y_occ,
                    c='gray',
                    s=15,
                    alpha=0.6,
                    marker='x',
                    linewidths=0.5)

    # 绘制恒星轮廓（半径=1R*）
    theta = np.linspace(0, 2 * np.pi, 100)
    stellar_x = np.cos(theta)
    stellar_y = np.sin(theta)
    ax4.plot(stellar_x,
             stellar_y,
             'k--',
             linewidth=1.5,
             alpha=0.5,
             label='Stellar surface')

    # 标记spot位置（区分是否被遮挡）
    for i, spot in enumerate(spots):
        x_c = spot.r * np.cos(spot.phi_initial)
        y_c = spot.r * np.sin(spot.phi_initial)

        # 判断spot中心是否被遮挡
        # 简单判断：找到距离spot中心最近的网格点
        dist = np.sqrt((grid.r * np.cos(grid.phi) - x_c)**2 +
                       (grid.r * np.sin(grid.phi) - y_c)**2)
        nearest_idx = np.argmin(dist)
        is_occluded = occultation_mask[nearest_idx]

        color = 'gray' if is_occluded else 'blue'
        alpha = 0.4 if is_occluded else 1.0
        ax4.plot(x_c,
                 y_c,
                 'o',
                 color=color,
                 markersize=8,
                 markeredgewidth=1.5,
                 markeredgecolor='black',
                 alpha=alpha)
        ax4.text(x_c,
                 y_c,
                 f'{i}',
                 color='white' if not is_occluded else 'black',
                 fontsize=7,
                 ha='center',
                 va='center',
                 fontweight='bold',
                 alpha=alpha)

    # 观察者方向
    ax4.arrow(0,
              0,
              obs_x,
              obs_y,
              head_width=0.12,
              head_length=0.15,
              fc='green',
              ec='green',
              linewidth=2,
              alpha=0.8)
    ax4.text(obs_x * 1.15,
             obs_y * 1.15,
             'Obs',
             color='green',
             fontsize=10,
             fontweight='bold',
             ha='center')

    ax4.set_aspect('equal')
    ax4.set_xlim(-2.8, 2.8)
    ax4.set_ylim(-2.8, 2.8)
    ax4.set_xlabel('x (R*)', fontsize=11)
    ax4.set_ylabel('y (R*)', fontsize=11)
    ax4.set_title(f'Phase {phase:.3f} - With Occultation',
                  fontsize=12,
                  fontweight='bold',
                  color='red')
    plt.colorbar(sc2, ax=ax4, label='Brightness', fraction=0.046)
    ax4.grid(True, alpha=0.2)
    ax4.legend(fontsize=8, loc='upper right')

    # 中下：Stokes I对比
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(wl,
             I_no_occ,
             color='black',
             linewidth=1.5,
             label='No Occultation',
             alpha=0.5)
    ax5.plot(wl,
             I_with_occ,
             color='red',
             linewidth=1.8,
             label='With Occultation')
    ax5.set_xlabel('Velocity (km/s)', fontsize=11)
    ax5.set_ylabel('Stokes I', fontsize=11)
    ax5.set_title('Stokes I - Comparison', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax5.legend(fontsize=9)

    # 右下：Stokes V对比
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(wl,
             V_no_occ,
             color='black',
             linewidth=1.5,
             label='No Occultation',
             alpha=0.5)
    ax6.plot(wl,
             V_with_occ,
             color='red',
             linewidth=1.8,
             label='With Occultation')
    ax6.set_xlabel('Velocity (km/s)', fontsize=11)
    ax6.set_ylabel('Stokes V', fontsize=11)
    ax6.set_title('Stokes V - Comparison', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    vlim_max = max(np.max(np.abs(V_no_occ)), np.max(np.abs(V_with_occ)))
    if vlim_max > 0:
        ax6.set_ylim(-1.1 * vlim_max, 1.1 * vlim_max)
    ax6.legend(fontsize=9)

    # 统计遮挡信息
    n_occluded = np.sum(occultation_mask)
    n_total = len(occultation_mask)
    occ_fraction = n_occluded / n_total * 100 if n_total > 0 else 0

    fig.suptitle(
        f'Stellar Occultation Test - Phase {phase:.3f}\n'
        f'Occluded pixels: {n_occluded}/{n_total} ({occ_fraction:.1f}%)',
        fontsize=14,
        fontweight='bold')

    fname = os.path.join(output_dir,
                         f'occultation_comparison_phase_{idx:02d}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存对比图: {fname}")


def plot_dynamic_spectrum_comparison(phases, velocities, spectra_I_no_occ,
                                     spectra_I_with_occ, spectra_V_no_occ,
                                     spectra_V_with_occ, output_dir):
    """绘制动态谱对比（2x2布局）"""
    print("\n生成动态谱对比图...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    extent = [velocities[0], velocities[-1], phases[0], phases[-1]]

    # 左上：Stokes I 无遮挡
    ax = axes[0, 0]
    im = ax.imshow(spectra_I_no_occ,
                   aspect='auto',
                   origin='lower',
                   extent=extent,
                   cmap='hot',
                   interpolation='bilinear')
    ax.set_xlabel('Velocity (km/s)', fontsize=11)
    ax.set_ylabel('Phase', fontsize=11)
    ax.set_title('Stokes I - No Occultation', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.grid(True, alpha=0.3, color='white', linestyle='--')

    # 右上：Stokes I 有遮挡
    ax = axes[0, 1]
    im = ax.imshow(spectra_I_with_occ,
                   aspect='auto',
                   origin='lower',
                   extent=extent,
                   cmap='hot',
                   interpolation='bilinear')
    ax.set_xlabel('Velocity (km/s)', fontsize=11)
    ax.set_ylabel('Phase', fontsize=11)
    ax.set_title('Stokes I - With Occultation',
                 fontsize=12,
                 fontweight='bold',
                 color='red')
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.grid(True, alpha=0.3, color='white', linestyle='--')

    # 左下：Stokes V 无遮挡
    ax = axes[1, 0]
    vmax = max(np.max(np.abs(spectra_V_no_occ)),
               np.max(np.abs(spectra_V_with_occ)))
    im = ax.imshow(spectra_V_no_occ,
                   aspect='auto',
                   origin='lower',
                   extent=extent,
                   cmap='RdBu_r',
                   vmin=-vmax,
                   vmax=vmax,
                   interpolation='bilinear')
    ax.set_xlabel('Velocity (km/s)', fontsize=11)
    ax.set_ylabel('Phase', fontsize=11)
    ax.set_title('Stokes V - No Occultation', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Circular Polarization')
    ax.grid(True, alpha=0.3, color='black', linestyle='--', linewidth=0.5)

    # 右下：Stokes V 有遮挡
    ax = axes[1, 1]
    im = ax.imshow(spectra_V_with_occ,
                   aspect='auto',
                   origin='lower',
                   extent=extent,
                   cmap='RdBu_r',
                   vmin=-vmax,
                   vmax=vmax,
                   interpolation='bilinear')
    ax.set_xlabel('Velocity (km/s)', fontsize=11)
    ax.set_ylabel('Phase', fontsize=11)
    ax.set_title('Stokes V - With Occultation',
                 fontsize=12,
                 fontweight='bold',
                 color='red')
    plt.colorbar(im, ax=ax, label='Circular Polarization')
    ax.grid(True, alpha=0.3, color='black', linestyle='--', linewidth=0.5)

    plt.suptitle('Dynamic Spectrum Comparison: Occultation Effect',
                 fontsize=14,
                 fontweight='bold',
                 y=0.995)
    plt.tight_layout()

    fname = os.path.join(output_dir, 'dynamic_spectrum_comparison.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存动态谱对比: {fname}")


def plot_occultation_geometry(grid, occultation_mask, phi_obs, inclination_deg,
                              output_dir):
    """绘制遮挡区域几何示意图（盘面坐标系）
    
    展示遮挡区域在盘面坐标系中的固定位置
    """
    print("\n生成遮挡几何示意图...")

    x = grid.r * np.cos(grid.phi)
    y = grid.r * np.sin(grid.phi)

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    # 可见像素（蓝色）
    visible = ~occultation_mask
    ax.scatter(x[visible],
               y[visible],
               c='skyblue',
               s=15,
               alpha=0.6,
               label='Visible pixels')

    # 遮挡像素（红色）
    ax.scatter(x[occultation_mask],
               y[occultation_mask],
               c='red',
               s=15,
               alpha=0.8,
               label='Occluded pixels')

    # 恒星轮廓
    theta = np.linspace(0, 2 * np.pi, 100)
    stellar_x = np.cos(theta)
    stellar_y = np.sin(theta)
    ax.plot(stellar_x,
            stellar_y,
            'k-',
            linewidth=2.5,
            label='Stellar surface (R=1R*)')
    ax.fill(stellar_x, stellar_y, color='yellow', alpha=0.3)

    # 观察者方向箭头
    obs_arrow_len = 2.5
    obs_x = obs_arrow_len * np.cos(phi_obs)
    obs_y = obs_arrow_len * np.sin(phi_obs)
    ax.arrow(0,
             0,
             obs_x,
             obs_y,
             head_width=0.15,
             head_length=0.2,
             fc='green',
             ec='green',
             linewidth=3,
             alpha=0.9,
             zorder=10)
    ax.text(obs_x * 1.2,
            obs_y * 1.2,
            'Observer\n(φ=0°, Fixed)',
            color='green',
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 标注遮挡区域方向
    occ_arrow_len = 2.3
    occ_phi = np.pi  # 背向观察者（φ=180°）
    occ_x = occ_arrow_len * np.cos(occ_phi)
    occ_y = occ_arrow_len * np.sin(occ_phi)
    ax.arrow(0,
             0,
             occ_x,
             occ_y,
             head_width=0.12,
             head_length=0.15,
             fc='red',
             ec='red',
             linewidth=2.5,
             alpha=0.7,
             linestyle='--',
             zorder=9)
    ax.text(occ_x * 1.2,
            occ_y * 1.2,
            'Occluded\nregion\n(φ≈180°)',
            color='red',
            fontsize=11,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_aspect('equal')
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.set_xlabel('x (R*)', fontsize=13)
    ax.set_ylabel('y (R*)', fontsize=13)
    ax.set_title(
        f'Stellar Occultation Geometry (Disk Frame)\n'
        f'Inclination = {inclination_deg:.1f}°, Observer at φ={np.degrees(phi_obs):.0f}°',
        fontsize=14,
        fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # 统计信息
    n_occ = np.sum(occultation_mask)
    n_total = len(occultation_mask)
    occ_pct = n_occ / n_total * 100 if n_total > 0 else 0
    ax.text(0.02,
            0.98,
            f'Occluded: {n_occ}/{n_total} pixels ({occ_pct:.1f}%)',
            transform=ax.transAxes,
            fontsize=11,
            va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    fname = os.path.join(output_dir, 'occultation_geometry.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存遮挡几何图: {fname}")


def compute_spectrum(grid,
                     line_data,
                     linemodel,
                     brightness,
                     Br,
                     Bphi,
                     phase,
                     inclination_deg,
                     vsini,
                     pOmega,
                     r0,
                     phi_obs=0.0,
                     enable_occultation=False,
                     stellar_radius=1.0):
    """计算光谱，支持遮挡开关
    
    返回:
    - v_range, stokes_i, stokes_v, occultation_mask
    """
    incl_rad = np.deg2rad(inclination_deg)

    # 计算遮挡mask（观察者方向固定，不依赖phase）
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
    v0_r0 = vsini / np.sin(incl_rad)
    v_phi = v0_r0 * (grid.r / r0)**pOmega

    # 视向速度
    v_los = v_phi * np.sin(incl_rad) * np.sin(grid.phi - phi_obs)

    # 速度网格
    v_range = np.linspace(-200, 200, 401)
    c = 2.99792458e5
    wl_obs = line_data.wl0 * (1 + v_range / c)
    wl_local = wl_obs[:, None] / (1.0 + v_los[None, :] / c)

    # 亮度和权重（应用遮挡）
    amp_local = brightness.copy()
    Ic_weight = grid.area.copy()

    if enable_occultation:
        amp_local[occultation_mask] = 0.0
        Ic_weight[occultation_mask] = 0.0

    # 计算局部谱线
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
    output_dir = 'test_output/stellar_occultation'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("恒星遮挡效果测试")
    print("=" * 70)

    # 读取参数
    par = readParamsTomog('input/params_tomog.txt', verbose=1)

    # 计算网格外半径
    pOmega = getattr(par, 'pOmega', 0.0)
    radius = getattr(par, 'radius', 1.0)
    Vmax = getattr(par, 'Vmax', 0.0)
    velEq = getattr(par, 'velEq', getattr(par, 'vsini', 100.0))

    if abs(pOmega + 1.0) > 1e-6:
        r_out_grid = radius * (Vmax / velEq)**(1.0 / (pOmega + 1.0))
    else:
        r_out_grid = getattr(par, 'r_out', 5.0 * radius)

    print(f"\n=== 网格参数 ===")
    print(f"  radius (R*): {radius:.2f}")
    print(f"  r_out: {par.r_out:.2f} R*")
    print(f"  Vmax: {Vmax:.2f} km/s")
    print(f"  r_out_grid: {r_out_grid:.3f} R*")
    print(f"  pOmega: {pOmega:.3f}")
    print(f"  inclination: {par.inclination:.1f}°")
    print(f"  vsini: {par.vsini:.2f} km/s\n")

    # 谱线和网格
    line_data = LineData('input/lines.txt')
    linemodel = GaussianZeemanWeakLineModel(
        line_data,
        k_QU=getattr(par, 'lineKQU', 1.0),
        enable_V=bool(getattr(par, 'lineEnableV', 1)),
        enable_QU=bool(getattr(par, 'lineEnableQU', 1)))

    grid = diskGrid(nr=getattr(par, 'nRingsStellarGrid', 60),
                    r_in=0.0,
                    r_out=r_out_grid,
                    verbose=1)

    # 时间和相位设置
    HJD0 = getattr(par, 'jDateRef', 0.5)
    period = par.period
    n_phases = 8
    HJDs = np.linspace(HJD0, HJD0 + period, n_phases, endpoint=False)
    phases = (HJDs - HJD0) / period

    print(f"\n=== 时间和相位 ===")
    print(f"  周期: {period} 天")
    print(f"  相位数: {n_phases}")
    print(f"  相位范围: {phases[0]:.3f} - {phases[-1]:.3f}\n")

    # 创建10个发射spot
    n_spots = 10
    r_min, r_max = 0.0, 2.0
    phi_initial = 0.0
    amp = 2.0
    spot_radius = 0.25
    B_amp = 1000

    print(f"=== 创建{n_spots}个发射spot ===")
    print(f"  半径范围: {r_min:.1f} - {r_max:.1f} R*")
    print(f"  初始方位角: {np.degrees(phi_initial):.1f}°")
    print(f"  振幅: {amp:.1f} (发射)")
    print(f"  磁场: {B_amp:.0f} G\n")

    rs = np.linspace(r_min, r_max, n_spots)
    spots = []
    for i, r in enumerate(rs):
        spots.append(
            Spot(r=r,
                 phi_initial=phi_initial,
                 amplitude=amp,
                 spot_type='emission',
                 radius=spot_radius,
                 B_amplitude=B_amp,
                 B_direction='radial'))
        print(f"  Spot {i}: r={r:.3f} R*, phi={np.degrees(phi_initial):.1f}°")

    spot_collection = SpotCollection(spots=spots)
    spot_collection.pOmega = par.pOmega
    spot_collection.r0 = par.radius
    spot_collection.period = period
    spot_geometry = TimeEvolvingSpotGeometry(grid, spot_collection)

    phi_obs = 0.0  # 观察者固定方向

    # 存储光谱用于动态谱
    v_range = None
    spectra_I_no_occ = []
    spectra_V_no_occ = []
    spectra_I_with_occ = []
    spectra_V_with_occ = []

    print("\n" + "=" * 70)
    print("计算各相位光谱（对比有/无遮挡）")
    print("=" * 70)

    for i, (phase, HJD) in enumerate(zip(phases, HJDs)):
        print(f"\n相位 {i}: phase={phase:.3f}, HJD={HJD:.3f}")

        brightness, Br, Bphi = spot_geometry.generate_distributions(phase)
        spots_evolved = spot_collection.get_spots_at_phase(phase)

        # 无遮挡
        v_range, I_no, V_no, _ = compute_spectrum(
            grid,
            line_data,
            linemodel,
            brightness,
            Br,
            Bphi,
            phase,
            inclination_deg=par.inclination,
            vsini=par.vsini,
            pOmega=par.pOmega,
            r0=par.radius,
            phi_obs=phi_obs,
            enable_occultation=False)

        # 有遮挡
        _, I_with, V_with, occ_mask = compute_spectrum(
            grid,
            line_data,
            linemodel,
            brightness,
            Br,
            Bphi,
            phase,
            inclination_deg=par.inclination,
            vsini=par.vsini,
            pOmega=par.pOmega,
            r0=par.radius,
            phi_obs=phi_obs,
            enable_occultation=True,
            stellar_radius=radius)

        n_occ = np.sum(occ_mask)
        occ_pct = n_occ / len(occ_mask) * 100 if len(occ_mask) > 0 else 0
        print(f"  遮挡像素: {n_occ}/{len(occ_mask)} ({occ_pct:.1f}%)")

        spectra_I_no_occ.append(I_no)
        spectra_V_no_occ.append(V_no)
        spectra_I_with_occ.append(I_with)
        spectra_V_with_occ.append(V_with)

        # 绘制对比图
        brightness_with_occ = brightness.copy()
        brightness_with_occ[occ_mask] = 0.0

        plot_comparison_panel(grid,
                              brightness,
                              brightness_with_occ,
                              occ_mask,
                              spots_evolved,
                              phase,
                              v_range,
                              I_no,
                              I_with,
                              V_no,
                              V_with,
                              output_dir,
                              i,
                              phi_obs=phi_obs)

        # 保存第一个相位的遮挡mask用于几何示意图
        if i == 0:
            first_occ_mask = occ_mask

    # 生成动态谱对比
    spectra_I_no_occ = np.array(spectra_I_no_occ)
    spectra_V_no_occ = np.array(spectra_V_no_occ)
    spectra_I_with_occ = np.array(spectra_I_with_occ)
    spectra_V_with_occ = np.array(spectra_V_with_occ)

    plot_dynamic_spectrum_comparison(phases, v_range, spectra_I_no_occ,
                                     spectra_I_with_occ, spectra_V_no_occ,
                                     spectra_V_with_occ, output_dir)

    # 生成遮挡几何示意图
    plot_occultation_geometry(grid, first_occ_mask, phi_obs, par.inclination,
                              output_dir)

    print("\n" + "=" * 70)
    print("测试完成！")
    print(f"输出目录: {output_dir}")
    print(f"生成文件: {n_phases}张对比图 + 1张动态谱对比 + 1张遮挡几何图")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
