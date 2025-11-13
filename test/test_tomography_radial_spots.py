"""
正演测试：10个发射团块，径向分布从0到2R*，8个相位，输出每相位团块分布和Stokes I/V谱线。
测试新参数结构（r_out=2R*，不考虑遮挡）。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from core.mainFuncs import readParamsTomog
from core.grid_tom import diskGrid
from core.local_linemodel_basic import LineData, GaussianZeemanWeakLineModel
from core.spot_geometry import Spot, SpotCollection, TimeEvolvingSpotGeometry


def make_radial_spots(n_spots,
                      r_min,
                      r_max,
                      phi,
                      amp,
                      spot_radius,
                      B_amp=1000):
    rs = np.linspace(r_min, r_max, n_spots)
    spots = []
    for r in rs:
        spots.append(
            Spot(r=r,
                 phi_initial=phi,
                 amplitude=amp,
                 spot_type='emission',
                 radius=spot_radius,
                 B_amplitude=B_amp,
                 B_direction='radial'))
    return SpotCollection(spots=spots)


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
                     HJD=None):
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
                    vmax=np.max(brightness))
    for spot in spots:
        x_c = spot.r * np.cos(spot.phi_initial)
        y_c = spot.r * np.sin(spot.phi_initial)
        ax.plot(x_c,
                y_c,
                'o',
                color='blue',
                markersize=12,
                markeredgewidth=1.5,
                markeredgecolor='black')

    # 标注观测者方向（观测者方位角 = phase * 2π）
    phi_obs = phase * 2 * np.pi
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
             linewidth=2,
             alpha=0.7)
    ax.text(obs_x * 1.15,
            obs_y * 1.15,
            'Observer',
            color='green',
            fontsize=10,
            fontweight='bold',
            ha='center')

    ax.set_aspect('equal')
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    title_str = f'Phase {phase:.3f} Spot Distribution\n(Observer at φ={phase * 360:.0f}°)'
    if HJD is not None:
        title_str += f'\nHJD={HJD:.3f}'
    ax.set_title(title_str)
    plt.colorbar(sc, ax=ax, label='Brightness')
    # 中：Stokes I
    axI = axes[1]
    axI.plot(wl, I, color='black')
    axI.set_xlabel('Wavelength (nm)')
    axI.set_ylabel('Stokes I')
    axI.set_title(f'Phase {phase:.3f}  Stokes I')
    axI.grid(True, alpha=0.3)
    axI.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    # 右：Stokes V
    axV = axes[2]
    axV.plot(wl, V, color='red')
    axV.set_xlabel('Wavelength (nm)')
    axV.set_ylabel('Stokes V')
    axV.set_title(f'Phase {phase:.3f}  Stokes V')
    axV.grid(True, alpha=0.3)
    # 自动设置V的y轴范围，便于比较
    vlim = np.max(np.abs(V))
    if vlim > 0:
        axV.set_ylim(-1.1 * vlim, 1.1 * vlim)
    plt.tight_layout()
    fname = os.path.join(output_dir, f'radial_spots_phase_{idx}.png')
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"已保存: {fname}")


def compute_spectrum(grid, line_data, linemodel, brightness, Br, Bphi,
                     phase_deg, inclination_deg, vsini, pOmega, r0):
    # 观测者方位角：相位0对应观测者在phi_obs=0，相位增加时观测者方位逆时针增加
    # 物理：天体自转，观测者在惯性系中固定，等效于观测者相对天体逆时针转动
    phase_rad = np.deg2rad(phase_deg)
    phi_obs = phase_rad  # 观测者方位角
    incl_rad = np.deg2rad(inclination_deg)

    # 视向磁场：考虑观测者方位
    Blos = Br * np.sin(incl_rad) * np.cos(grid.phi - phi_obs)
    Bperp = np.sqrt(Br**2 + Bphi**2 - Blos**2)
    Bperp = np.maximum(Bperp, 0.0)
    chi = grid.phi

    # 环向速度（较差转动）
    v0_r0 = vsini / np.sin(incl_rad)
    v_phi = v0_r0 * (grid.r / r0)**pOmega

    # 视向速度：v_los = v_phi * sin(i) * sin(phi - phi_obs)
    # 当团块在观测者正前方时(phi=phi_obs)，sin=0，v_los=0
    # 当团块在左侧(phi>phi_obs)，sin>0，蓝移（v_los>0）
    # 当团块在右侧(phi<phi_obs)，sin<0，红移（v_los<0）
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
    return wl_obs, stokes_i, stokes_v


def main():
    output_dir = 'test_output/radial_spots_2Rstar'
    os.makedirs(output_dir, exist_ok=True)

    # 读取参数（包含新结构的Vmax/r_out逻辑）
    par = readParamsTomog('input/params_tomog.txt', verbose=1)

    # 计算网格外半径（与tomography.py中逻辑一致）
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
    HJD0 = getattr(par, 'jDateRef', 0.5)  # 参考时间
    period = par.period  # 周期（天）
    n_phases = 8
    # 生成8个均匀分布的观测时间（从HJD0到HJD0+period）
    HJDs = np.linspace(HJD0, HJD0 + period, n_phases, endpoint=False)
    phases = (HJDs - HJD0) / period  # 相位 = (HJD - HJD0) / period

    print(f"\n时间和相位设置:")
    print(f"  参考时间 HJD0: {HJD0}")
    print(f"  周期: {period} 天")
    print(f"  观测时间: {HJDs[0]:.3f} - {HJDs[-1]:.3f} HJD")
    print(f"  相位: {phases[0]:.3f} - {phases[-1]:.3f}\n")

    # 10个发射团块：沿径向分布从0.1到2R*，初始集中在phi=0附近（±30°范围）
    # 注意：避免r=0（差速转动时会有数值问题）
    # 这样既能看到较差转动效应，又能在不同相位看到明显的多普勒频移
    n_spots = 10
    r_min, r_max = 0.1, 2.0  # 从0.1到2R*（避免r=0）
    phi_center = 0.0  # 团块中心方位
    phi_spread = np.deg2rad(30)  # 方位展宽±30°
    amp = 2.0  # 发射振幅（正值）
    spot_radius = 0.3
    B_amp = 1000

    # 创建团块：径向均匀分布，方位角在phi_center附近随机分布
    rs = np.linspace(r_min, r_max, n_spots)
    np.random.seed(42)
    phis = phi_center + np.random.uniform(-phi_spread, phi_spread, n_spots)
    spots = []
    for r, phi in zip(rs, phis):
        spots.append(
            Spot(r=r,
                 phi_initial=phi,
                 amplitude=amp,
                 spot_type='emission',
                 radius=spot_radius,
                 B_amplitude=B_amp,
                 B_direction='radial'))
    spot_collection = SpotCollection(spots=spots)
    spot_collection.pOmega = par.pOmega
    spot_collection.r0 = par.radius  # 使用恒星半径作为参考
    spot_collection.period = period
    spot_geometry = TimeEvolvingSpotGeometry(grid, spot_collection)

    for i, (phase, HJD) in enumerate(zip(phases, HJDs)):
        print(f"处理相位 {i}: phase={phase:.3f}, HJD={HJD:.3f}")
        brightness, Br, Bphi = spot_geometry.generate_distributions(phase)
        spots_evolved = spot_collection.get_spots_at_phase(phase)
        phase_deg = phase * 360.0
        wl, I, V = compute_spectrum(grid,
                                    line_data,
                                    linemodel,
                                    brightness,
                                    Br,
                                    Bphi,
                                    phase_deg=phase_deg,
                                    inclination_deg=par.inclination,
                                    vsini=par.vsini,
                                    pOmega=par.pOmega,
                                    r0=par.radius)
        plot_phase_panel(grid,
                         brightness,
                         Br,
                         Bphi,
                         spots_evolved,
                         phase,
                         wl,
                         I,
                         V,
                         output_dir,
                         i,
                         HJD=HJD)
    print("\n正演测试完成！")
    print(f"输出目录: {output_dir}")
    print(f"团块范围: 0 - {r_max:.1f} R*")
    print(f"网格范围: 0 - {r_out_grid:.3f} R*")
    print(f"相位范围: {phases[0]:.3f} - {phases[-1]:.3f}")
    print(f"时间范围: {HJDs[0]:.3f} - {HJDs[-1]:.3f} HJD\n")


if __name__ == '__main__':
    main()
