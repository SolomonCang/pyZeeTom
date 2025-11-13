"""
正演测试：10个发射团块，径向分布从0到2R*，固定观察者方向
测试新参数结构（r_out=2R*，不考虑遮挡）
物理约定：
- 观察者方向固定（phi_obs = 0，即"向上"方向）
- 所有spot初始在同一方位角（phi_0 = 0）
- 随时间演化，spot因差速转动而相对观察者移动
- 可以处理r=0的spot（中心不转动）
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


def plot_phase_panel(grid,
                     brightness,
                     Br,
                     Bphi,
                     spots,
                     phase,
                     wl,
                     I,
                     V,
                     output_dir,
                     idx,
                     HJD=None,
                     phi_obs=0.0):
    """绘制单个相位的三合一图：spot分布 + Stokes I + Stokes V
    
    参数:
    - phi_obs: 观察者方向（弧度），固定值，默认0（向上）
    """
    x = grid.r * np.cos(grid.phi)
    y = grid.r * np.sin(grid.phi)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 左：团块分布
    ax = axes[0]
    sc = ax.scatter(x,
                    y,
                    c=brightness,
                    s=20,
                    cmap='YlOrRd',
                    vmin=0,
                    vmax=np.max(brightness) if np.max(brightness) > 0 else 3)

    # 标记spot位置
    for spot in spots:
        x_c = spot.r * np.cos(spot.phi_initial)
        y_c = spot.r * np.sin(spot.phi_initial)
        ax.plot(x_c,
                y_c,
                'o',
                color='blue',
                markersize=10,
                markeredgewidth=1.5,
                markeredgecolor='black')
        # 标注spot编号
        ax.text(x_c,
                y_c,
                f'{spots.index(spot)}',
                color='white',
                fontsize=8,
                ha='center',
                va='center',
                fontweight='bold')

    # 固定观测者方向（向phi_obs方向，默认向上）
    obs_x = 2.5 * np.cos(phi_obs)
    obs_y = 2.5 * np.sin(phi_obs)
    ax.arrow(0,
             0,
             obs_x,
             obs_y,
             head_width=0.15,
             head_length=0.2,
             fc='green',
             ec='green',
             linewidth=2.5,
             alpha=0.8)
    ax.text(obs_x * 1.2,
            obs_y * 1.2,
            'Observer\n(Fixed)',
            color='green',
            fontsize=11,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_aspect('equal')
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.set_xlabel('x (R*)', fontsize=12)
    ax.set_ylabel('y (R*)', fontsize=12)

    title_str = f'Phase {phase:.3f} Spot Distribution'
    if HJD is not None:
        title_str += f'\nHJD={HJD:.3f}'
    ax.set_title(title_str, fontsize=13, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Brightness')

    # 中：Stokes I
    axI = axes[1]
    axI.plot(wl, I, color='black', linewidth=1.5)
    axI.set_xlabel('Velocity (km/s)', fontsize=12)
    axI.set_ylabel('Stokes I', fontsize=12)
    axI.set_title(f'Phase {phase:.3f}  Stokes I',
                  fontsize=13,
                  fontweight='bold')
    axI.grid(True, alpha=0.3)
    axI.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # 右：Stokes V
    axV = axes[2]
    axV.plot(wl, V, color='red', linewidth=1.5)
    axV.set_xlabel('Velocity (km/s)', fontsize=12)
    axV.set_ylabel('Stokes V', fontsize=12)
    axV.set_title(f'Phase {phase:.3f}  Stokes V',
                  fontsize=13,
                  fontweight='bold')
    axV.grid(True, alpha=0.3)
    axV.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # 自动设置V的y轴范围
    vlim = np.max(np.abs(V))
    if vlim > 0:
        axV.set_ylim(-1.1 * vlim, 1.1 * vlim)

    plt.tight_layout()
    fname = os.path.join(output_dir, f'emission_spots_phase_{idx:02d}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {fname}")


def plot_dynamic_spectrum(phases, velocities, spectra_I, spectra_V,
                          output_dir):
    """绘制动态谱（2D热图：相位 vs 速度）"""
    print("\n生成动态谱...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Stokes I 动态谱
    ax = axes[0]
    extent = [velocities[0], velocities[-1], phases[0], phases[-1]]
    im = ax.imshow(spectra_I,
                   aspect='auto',
                   origin='lower',
                   extent=extent,
                   cmap='hot',
                   interpolation='bilinear')
    ax.set_xlabel('Velocity (km/s)', fontsize=12)
    ax.set_ylabel('Phase', fontsize=12)
    ax.set_title('Dynamic Spectrum: Stokes I', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.grid(True, alpha=0.3, color='white', linestyle='--')

    # Stokes V 动态谱
    ax = axes[1]
    vmax = np.max(np.abs(spectra_V))
    im = ax.imshow(spectra_V,
                   aspect='auto',
                   origin='lower',
                   extent=extent,
                   cmap='RdBu_r',
                   vmin=-vmax,
                   vmax=vmax,
                   interpolation='bilinear')
    ax.set_xlabel('Velocity (km/s)', fontsize=12)
    ax.set_ylabel('Phase', fontsize=12)
    ax.set_title('Dynamic Spectrum: Stokes V', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Circular Polarization')
    ax.grid(True, alpha=0.3, color='black', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    fname = os.path.join(output_dir, 'dynamic_spectrum.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存动态谱: {fname}")


def plot_all_phases_overview(phases, spots_list, output_dir):
    """绘制所有相位的spot位置总览"""
    print("\n生成所有相位总览图...")

    n_phases = len(phases)
    ncols = 4
    nrows = (n_phases + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten() if n_phases > 1 else [axes]

    for idx, (phase, spots) in enumerate(zip(phases, spots_list)):
        ax = axes[idx]

        # 绘制spot轨迹（连接同一spot在不同相位的位置）
        for i in range(10):  # 10个spot
            rs = [spots_list[p][i].r for p in range(n_phases)]
            phis = [spots_list[p][i].phi_initial for p in range(n_phases)]
            xs = [r * np.cos(phi) for r, phi in zip(rs, phis)]
            ys = [r * np.sin(phi) for r, phi in zip(rs, phis)]
            ax.plot(xs, ys, 'o-', alpha=0.3, markersize=3, linewidth=0.5)

        # 当前相位的spot位置
        for spot in spots:
            x_c = spot.r * np.cos(spot.phi_initial)
            y_c = spot.r * np.sin(spot.phi_initial)
            ax.plot(x_c,
                    y_c,
                    'o',
                    color='red',
                    markersize=8,
                    markeredgewidth=1,
                    markeredgecolor='black')

        # 观察者方向
        ax.arrow(0,
                 0,
                 0,
                 2.3,
                 head_width=0.1,
                 head_length=0.15,
                 fc='green',
                 ec='green',
                 linewidth=2,
                 alpha=0.7)

        ax.set_aspect('equal')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(f'Phase {phase:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_phases, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('All Phases: Spot Evolution (Observer Fixed at φ=0)',
                 fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(output_dir, 'all_phases_overview.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存总览图: {fname}")


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
                     phi_obs=0.0):
    """计算光谱，观察者方向固定
    
    参数:
    - phi_obs: 观察者方位角（弧度），固定值，默认0（向上）
    - phase: 当前相位（0-1），用于确定spot位置，不影响观察者方向
    """
    incl_rad = np.deg2rad(inclination_deg)

    # 视向磁场：观察者方向固定在phi_obs
    Blos = Br * np.sin(incl_rad) * np.cos(grid.phi - phi_obs)
    Bperp = np.sqrt(Br**2 + Bphi**2 - Blos**2)
    Bperp = np.maximum(Bperp, 0.0)
    chi = grid.phi

    # 环向速度（差速转动）
    v0_r0 = vsini / np.sin(incl_rad)
    v_phi = v0_r0 * (grid.r / r0)**pOmega

    # 视向速度：观察者固定，spot随盘转动
    # v_los = v_phi * sin(i) * sin(phi - phi_obs)
    v_los = v_phi * np.sin(incl_rad) * np.sin(grid.phi - phi_obs)

    v_range = np.linspace(-200, 200, 401)
    c = 2.99792458e5
    wl_obs = line_data.wl0 * (1 + v_range / c)
    wl_local = wl_obs[:, None] / (1.0 + v_los[None, :] / c)
    amp_local = brightness

    profiles = linemodel.compute_local_profile(wl_local,
                                               amp=amp_local,
                                               Blos=Blos,
                                               Bperp=Bperp,
                                               chi=chi,
                                               Ic_weight=grid.area)
    stokes_i = np.sum(profiles['I'], axis=1)
    stokes_v = np.sum(profiles['V'], axis=1)
    total_area = np.sum(grid.area)
    if total_area > 0:
        stokes_i = stokes_i / total_area
        stokes_v = stokes_v / total_area

    return v_range, stokes_i, stokes_v


def main():
    output_dir = 'test_output/emission_spots_fixed_obs'
    os.makedirs(output_dir, exist_ok=True)

    # 读取参数（包含新结构的Vmax/r_out逻辑）
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
    print(f"  Vmax: {Vmax:.2f} km/s (computed: {par.Vmax:.2f})")
    print(f"  r_out_grid: {r_out_grid:.3f} R*")
    print(f"  pOmega: {pOmega:.3f}")
    print(f"  velEq: {velEq:.2f} km/s\n")

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

    print(f"\n=== 时间和相位设置 ===")
    print(f"  参考时间 HJD0: {HJD0}")
    print(f"  周期: {period} 天")
    print(f"  观测时间: {HJDs[0]:.3f} - {HJDs[-1]:.3f} HJD")
    print(f"  相位: {phases[0]:.3f} - {phases[-1]:.3f}\n")

    # 10个发射团块：沿径向分布从0到2R*，初始都在phi=0（同一方位角）
    n_spots = 10
    r_min, r_max = 0.0, 2.0
    phi_initial = 0.0  # 所有spot初始方位角相同（向上）
    amp = 2.0  # 发射振幅
    spot_radius = 0.25
    B_amp = 1000

    print(f"=== 创建{n_spots}个发射spot ===")
    print(f"  半径范围: {r_min:.1f} - {r_max:.1f} R*")
    print(f"  初始方位角: {np.degrees(phi_initial):.1f}° (所有spot)")
    print(f"  振幅: {amp:.1f} (发射)")
    print(f"  磁场: {B_amp:.0f} G\n")

    rs = np.linspace(r_min, r_max, n_spots)
    spots = []
    for i, r in enumerate(rs):
        spots.append(
            Spot(
                r=r,
                phi_initial=phi_initial,  # 所有spot初始在同一方位角
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

    # 观察者方向固定
    phi_obs = 0.0  # 固定向上（phi=0方向）

    # 存储光谱用于动态谱
    v_range = None
    spectra_I_all = []
    spectra_V_all = []
    spots_list_all = []

    print("\n=== 计算各相位光谱 ===")
    for i, (phase, HJD) in enumerate(zip(phases, HJDs)):
        print(f"处理相位 {i}: phase={phase:.3f}, HJD={HJD:.3f}")
        brightness, Br, Bphi = spot_geometry.generate_distributions(phase)
        spots_evolved = spot_collection.get_spots_at_phase(phase)
        spots_list_all.append(spots_evolved)

        v_range, I, V = compute_spectrum(grid,
                                         line_data,
                                         linemodel,
                                         brightness,
                                         Br,
                                         Bphi,
                                         phase=phase,
                                         inclination_deg=par.inclination,
                                         vsini=par.vsini,
                                         pOmega=par.pOmega,
                                         r0=par.radius,
                                         phi_obs=phi_obs)

        spectra_I_all.append(I)
        spectra_V_all.append(V)

        plot_phase_panel(grid,
                         brightness,
                         Br,
                         Bphi,
                         spots_evolved,
                         phase,
                         v_range,
                         I,
                         V,
                         output_dir,
                         i,
                         HJD=HJD,
                         phi_obs=phi_obs)

    # 生成动态谱
    spectra_I_all = np.array(spectra_I_all)
    spectra_V_all = np.array(spectra_V_all)
    plot_dynamic_spectrum(phases, v_range, spectra_I_all, spectra_V_all,
                          output_dir)

    # 生成所有相位总览
    plot_all_phases_overview(phases, spots_list_all, output_dir)

    print("\n" + "=" * 60)
    print("正演测试完成！")
    print(f"输出目录: {output_dir}")
    print(f"团块范围: {r_min:.1f} - {r_max:.1f} R* (包含r=0)")
    print(f"网格范围: 0 - {r_out_grid:.3f} R*")
    print(f"观察者方向: φ={np.degrees(phi_obs):.0f}° (固定)")
    print(f"相位范围: {phases[0]:.3f} - {phases[-1]:.3f}")
    print(f"时间范围: {HJDs[0]:.3f} - {HJDs[-1]:.3f} HJD")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
