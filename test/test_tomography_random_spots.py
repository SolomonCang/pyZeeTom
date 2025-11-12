"""
综合测试：从参数文件和谱线文件读取，构建随机团块盘模型，生成8个相位合成谱线，保存数据并可视化。

重构说明：
- 使用 core.spot_geometry 模块处理团块演化（较差转动的时间演化已在核心代码实现）
- 移除测试文件中的 evolve_spot_positions 和 regenerate_brightness_from_spots 函数
- 通过 TimeEvolvingSpotGeometry 类管理时间相关的物质分布
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from core.mainFuncs import readParamsTomog
from core.grid_tom import diskGrid
from core.local_linemodel_basic import LineData, GaussianZeemanWeakLineModel
from core.velspace_DiskIntegrator import disk_velocity_adaptive_inner_omega_sequence
from core.spot_geometry import SpotCollection, TimeEvolvingSpotGeometry


def save_spectrum(filename, wavelength, stokes_i, stokes_v):
    header = "wavelength(nm)  Stokes_I  Stokes_V"
    data = np.column_stack([wavelength, stokes_i, stokes_v])
    np.savetxt(filename, data, header=header, fmt='%12.6f %12.8e %12.8e')
    print(f"已保存: {filename}")


def plot_spot_evolution_polar(spot_geometry,
                              phases,
                              pOmega,
                              r0,
                              inclination_deg,
                              output_dir,
                              period=1.0):
    """
    绘制团块位置演化的极坐标图（8个相位）
    
    参数:
    - spot_geometry: TimeEvolvingSpotGeometry 对象
    - phases: 相位数组
    - pOmega: 较差转动幂律指数
    - r0: 参考半径
    - inclination_deg: 倾角（度）
    - output_dir: 输出目录
    - period: 周期（天）
    """
    n_cols = 4
    n_rows = 2

    fig = plt.figure(figsize=(16, 8), dpi=120)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    inc_rad = np.deg2rad(inclination_deg)

    for idx, phase in enumerate(phases):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col], projection='polar')

        # 观测者方位角（相位0对应观测者在0°方向）
        phi_obs = 0.0
        time_days = phase * period

        # 绘制盘面边界
        theta_disk = np.linspace(0, 2 * np.pi, 100)
        r_disk = np.full_like(theta_disk, 5.0)
        ax.plot(theta_disk, r_disk, 'k--', linewidth=1.5, alpha=0.3)

        # 从核心类获取演化后的团块位置
        spot_centers_evolved = spot_geometry.spot_collection.evolve_to_phase(
            phase)

        # 绘制团块
        for r_spot, phi_spot, amp, spot_type in spot_centers_evolved:
            # 投影因子（蓝移/红移）
            proj = np.sin(inc_rad) * np.sin(phi_spot - phi_obs)

            # 可见性
            sin_term = np.sin(inc_rad) * np.cos(phi_spot - phi_obs)
            cos_theta = np.abs(np.cos(inc_rad) + sin_term)
            visibility = np.clip(cos_theta, 0.0, 1.0)

            # 颜色：蓝色=朝向观测者，红色=背离
            if proj > 0:
                color = plt.cm.Blues(0.3 + 0.7 * abs(proj))
            else:
                color = plt.cm.Reds(0.3 + 0.7 * abs(proj))

            # 标记：发射=星号，吸收=圆圈
            marker = '*' if spot_type == 'emission' else 'o'
            marker_size = 100 + 200 * visibility

            ax.plot(phi_spot,
                    r_spot,
                    marker,
                    color=color,
                    markersize=np.sqrt(marker_size),
                    alpha=0.8,
                    markeredgecolor='black',
                    markeredgewidth=0.5)

        # 绘制观测者视线方向
        ax.plot([phi_obs, phi_obs], [0, 5.5], 'g-', linewidth=2, alpha=0.7)
        ax.plot(phi_obs, 5.3, 'g^', markersize=10)

        ax.set_ylim(0, 6)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_title(f'Phase {phase:.3f} (t={time_days:.3f}d)',
                     fontsize=11,
                     fontweight='bold',
                     pad=10)
        ax.grid(True, alpha=0.3)

        # 图例（只在第一个子图）
        if idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0],
                       marker='*',
                       color='w',
                       markerfacecolor='gold',
                       markersize=10,
                       label='Emission',
                       markeredgecolor='black'),
                Line2D([0], [0],
                       marker='o',
                       color='w',
                       markerfacecolor='cyan',
                       markersize=8,
                       label='Absorption',
                       markeredgecolor='black'),
                Line2D([0], [0], color='blue', linewidth=2, label='Blueshift'),
                Line2D([0], [0], color='red', linewidth=2, label='Redshift'),
            ]
            ax.legend(handles=legend_elements,
                      loc='upper left',
                      bbox_to_anchor=(0.0, 1.15),
                      fontsize=8)

    title = (f'Spot Evolution with Differential Rotation '
             f'(i={inclination_deg}°, p={pOmega:.2f}, P={period:.2f}d)')
    fig.suptitle(title, fontsize=14, fontweight='bold')

    output_path = os.path.join(output_dir, 'random_spots_evolution_polar.png')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"已保存团块演化图: {output_path}")
    plt.close()


def plot_disk_structure(grid,
                        brightness,
                        Br,
                        Bphi,
                        phase_deg,
                        inclination_deg,
                        vsini,
                        output_dir,
                        spot_centers=None,
                        fig_name_prefix='random_spots_disk_structure'):
    """绘制盘面结构（亮度、磁场、视向投影）并标记团块位置"""
    phase_rad = np.deg2rad(phase_deg)
    incl_rad = np.deg2rad(inclination_deg)
    Btotal = np.sqrt(Br**2 + Bphi**2)
    Blos = Br * np.sin(incl_rad) * np.cos(grid.phi - phase_rad)
    v_los = vsini * np.sin(incl_rad) * np.sin(grid.phi - phase_rad) * grid.r
    x = grid.r * np.cos(grid.phi)
    y = grid.r * np.sin(grid.phi)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    plot_lim = grid.r_out * 1.1

    # 亮度
    ax = axes[0, 0]
    vmin_b = np.min(brightness)
    vmax_b = np.max(brightness)
    scatter = ax.scatter(x,
                         y,
                         c=brightness,
                         s=15,
                         cmap='RdBu_r',
                         alpha=0.8,
                         edgecolors='none',
                         vmin=vmin_b,
                         vmax=vmax_b)
    # 标记团块中心
    if spot_centers is not None:
        for r_c, phi_c, amp, typ in spot_centers:
            x_c = r_c * np.cos(phi_c)
            y_c = r_c * np.sin(phi_c)
            marker = '*' if typ == 'emission' else 'X'
            color = 'yellow' if typ == 'emission' else 'cyan'
            ax.plot(x_c,
                    y_c,
                    marker,
                    markersize=14,
                    markeredgewidth=1.2,
                    markeredgecolor='black',
                    color=color)
    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    ax.set_aspect('equal')
    ax.set_title('Spot Brightness')
    plt.colorbar(scatter, ax=ax)

    # 磁场强度
    ax = axes[0, 1]
    scatter = ax.scatter(x,
                         y,
                         c=Btotal,
                         s=15,
                         cmap='plasma',
                         alpha=0.8,
                         edgecolors='none')
    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    ax.set_aspect('equal')
    ax.set_title('Magnetic Field |B|')
    plt.colorbar(scatter, ax=ax)

    # 视向磁场
    ax = axes[1, 0]
    vmax_blos = np.max(np.abs(Blos))
    scatter = ax.scatter(x,
                         y,
                         c=Blos,
                         s=15,
                         cmap='RdBu_r',
                         alpha=0.8,
                         edgecolors='none',
                         vmin=-vmax_blos,
                         vmax=vmax_blos)
    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    ax.set_aspect('equal')
    ax.set_title('Line-of-Sight B')
    plt.colorbar(scatter, ax=ax)

    # 视向速度
    ax = axes[1, 1]
    scatter = ax.scatter(x,
                         y,
                         c=v_los,
                         s=15,
                         cmap='RdBu_r',
                         alpha=0.8,
                         edgecolors='none',
                         vmin=-vsini,
                         vmax=vsini)
    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    ax.set_aspect('equal')
    ax.set_title('Line-of-Sight Velocity')
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f'{fig_name_prefix}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"已保存盘面结构图: {output_file}")
    plt.close()


def compute_spectrum(grid,
                     line_data,
                     linemodel,
                     brightness,
                     Br,
                     Bphi,
                     phase_deg,
                     inclination_deg,
                     vsini,
                     pOmega=-0.5,
                     r0=3.0):
    """
    计算考虑差异转动的光谱
    
    参数:
    - pOmega: 差异转动幂律指数（Ω ∝ r^pOmega）
    - r0: 参考半径（在此半径处 v = vsini）
    """
    phase_rad = np.deg2rad(phase_deg)
    incl_rad = np.deg2rad(inclination_deg)
    Blos = Br * np.sin(incl_rad) * np.cos(grid.phi - phase_rad)
    Bperp = np.sqrt(Br**2 + Bphi**2 - Blos**2)
    Bperp = np.maximum(Bperp, 0.0)
    chi = grid.phi

    # 使用差异转动模型计算环向速度
    v0_r0 = vsini / np.sin(incl_rad)  # 在 r0 处的赤道速度
    v_phi = disk_velocity_adaptive_inner_omega_sequence(grid,
                                                        v0_r0=v0_r0,
                                                        p=pOmega,
                                                        r0=r0,
                                                        profile='cosine',
                                                        blend_edge=True)

    # 投影到视线方向
    v_los = v_phi * np.sin(incl_rad) * np.sin(grid.phi - phase_rad)

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
    """
    主测试函数：使用核心模块的 TimeEvolvingSpotGeometry 进行较差转动演化
    """
    output_dir = 'test_output'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取参数和谱线
    par = readParamsTomog('input/params_tomog.txt')
    line_data = LineData('input/lines.txt')
    linemodel = GaussianZeemanWeakLineModel(
        line_data,
        k_QU=getattr(par, 'lineKQU', 1.0),
        enable_V=bool(getattr(par, 'lineEnableV', 1)),
        enable_QU=bool(getattr(par, 'lineEnableQU', 1)))

    # 2. 构建盘面网格
    grid = diskGrid(nr=getattr(par, 'nRingsStellarGrid', 60),
                    r_in=0.0,
                    r_out=getattr(par, 'radius', 5.0),
                    verbose=1)

    # 3. 使用核心模块生成随机团块集合
    r0 = getattr(par, 'radius', 3.0)
    spot_collection = SpotCollection.create_random_spots(
        n_emission=5,
        n_absorption=5,
        r_range=(0.5, grid.r_out * 0.8),
        amp_emission_range=(1.0, 3.0),
        amp_absorption_range=(-3.0, -1.0),
        spot_radius=0.5,
        B_range=(500, 2500),
        seed=2024)

    # 设置较差转动参数
    spot_collection.pOmega = par.pOmega
    spot_collection.r0 = r0
    spot_collection.period = 1.0

    # 创建时间演化几何
    spot_geometry = TimeEvolvingSpotGeometry(grid, spot_collection)

    # 打印团块信息
    print(f"\n生成了 {len(spot_collection.spots)} 个团块:")
    emit_count = sum(1 for s in spot_collection.spots
                     if s.spot_type == 'emission')
    abs_count = sum(1 for s in spot_collection.spots
                    if s.spot_type == 'absorption')
    print(f"  发射团块: {emit_count}")
    print(f"  吸收团块: {abs_count}")
    print("\n差异转动参数:")
    print(f"  pOmega (幂律指数): {par.pOmega}")
    print(f"  参考半径 r0: {r0}")
    print(f"  vsini: {par.vsini} km/s")
    print(f"  倾角: {par.inclination}°")
    print("\n团块将随差异转动演化：不同半径处转动速度不同")

    # 4. 计算8个相位的合成谱线
    phases = np.linspace(0, 1, 8, endpoint=False)
    spectra = []

    for i, phase in enumerate(phases):
        print(f"\n处理相位 {i}: phase={phase:.3f}")

        # 从核心类获取演化后的亮度和磁场分布
        brightness, Br, Bphi = spot_geometry.generate_distributions(phase)

        # 获取演化后的团块位置（用于可视化）
        spot_centers_evolved = spot_collection.evolve_to_phase(phase)

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
                                    r0=r0)

        # 保存光谱数据
        output_file = os.path.join(output_dir, f'random_spots_phase_{i}.dat')
        save_spectrum(output_file, wl, I, V)
        spectra.append((phase, wl, I, V))

        # 保存盘结构图，文件名带相位（显示演化后的团块位置）
        plot_disk_structure(
            grid,
            brightness,
            Br,
            Bphi,
            phase_deg=phase_deg,
            inclination_deg=par.inclination,
            vsini=par.vsini,
            output_dir=output_dir,
            spot_centers=spot_centers_evolved,
            fig_name_prefix=f'random_spots_disk_structure_phase_{i}')

        # 保存谱线轮廓图
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(wl, I, label='I', color='black')
        ax.plot(wl, V * 1000, label='V×1000', color='red', linestyle='--')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'Phase {phase:.3f} (deg={phase_deg:.1f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0,
                   color='gray',
                   linestyle='--',
                   linewidth=0.8,
                   alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                 f'random_spots_spectrum_phase_{i}.png'),
                    dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    # 5. 绘制团块位置演化的极坐标图
    print("\n生成团块演化极坐标图...")
    plot_spot_evolution_polar(spot_geometry, phases, par.pOmega, r0,
                              par.inclination, output_dir)

    print("\n综合测试完成!")
    print(f"已保存 {len(phases)} 个相位的:")
    print("  - 盘面结构图: test_output/random_spots_disk_structure_phase_*.png")
    print("  - 谱线轮廓图: test_output/random_spots_spectrum_phase_*.png")
    print("  - 光谱数据: test_output/random_spots_phase_*.dat")
    print("  - 团块演化图: test_output/random_spots_evolution_polar.png")
    print("\n团块配置:")
    print("  - 5个发射团块 + 5个吸收团块")
    print("  - 团块以外区域: 无信号 (背景=0)")
    print(f"  - 差异转动: Ω ∝ r^{par.pOmega}")
    print("  - 团块位置随相位演化（不同半径转速不同）")
    print("\n架构改进: 较差转动的时间演化已在核心代码中实现 (core/spot_geometry.py)")


if __name__ == '__main__':
    main()
