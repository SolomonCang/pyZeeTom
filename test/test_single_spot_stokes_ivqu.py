"""
test_single_spot_stokes_ivqu.py

单个spot的Stokes IVQU谱线建模与可视化测试

功能：
  1. 从 input/ 读取参数与谱线参数文件
  2. 生成一个单spot模型（距离2R*, 1000Gauss磁场）
  3. 利用核心模块合成观测空间的Stokes I/V/Q/U谱线
  4. 可视化展示结果
  
依赖：
  - core.grid_tom.diskGrid
  - core.spot_geometry.Spot, SpotCollection
  - core.local_linemodel_basic.LineData, GaussianZeemanWeakLineModel
  - core.velspace_DiskIntegrator.VelspaceDiskIntegrator
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目根路径到 PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.grid_tom import diskGrid  # noqa: E402
from core.spot_geometry import Spot, SpotCollection  # noqa: E402
from core.local_linemodel_basic import LineData, GaussianZeemanWeakLineModel  # noqa: E402
from core.mainFuncs import readParamsTomog  # noqa: E402
from core.velspace_DiskIntegrator import VelspaceDiskIntegrator  # noqa: E402


class SingleSpotModel:
    """
    单spot模型建造器与仓库
    """

    def __init__(self, params_file, lines_file):
        """
        初始化模型参数
        
        Args:
            params_file: 参数文件路径 (input/params_tomog.txt)
            lines_file: 谱线参数文件路径 (input/lines.txt)
        """
        # 读取参数
        try:
            self.params = readParamsTomog(params_file)
        except Exception as e:
            print(f"Warning: 参数读取有问题，使用默认值。Error: {e}")
            self.params = None

        # 读取谱线参数
        self.line_data = LineData(lines_file)

        # 基本观测参数（从params读取，或使用默认值）
        if self.params is not None:
            self.inclination = getattr(self.params, 'inclination', 60.0)
            self.vsini = getattr(self.params, 'vsini', 25.0)
            self.period = getattr(self.params, 'period', 1.0)
            self.pOmega = getattr(self.params, 'pOmega', -0.05)
            self.radius = getattr(self.params, 'radius', 0.5)
            self.r_out = getattr(self.params, 'r_out', 2.0)
            self.nrings = getattr(self.params, 'nRingsStellarGrid', 60)
        else:
            defaults = self._default_params()
            self.inclination = defaults['inclination']
            self.vsini = defaults['vsini']
            self.period = defaults['period']
            self.pOmega = defaults['pOmega']
            self.radius = defaults['radius']
            self.r_out = defaults['r_out']
            self.nrings = defaults['nRingsStellarGrid']

        # 盘网格
        self.grid = None
        self.make_grid()

        # Spot集合
        self.spot_collection = None
        self.make_spot_collection()

        # VelspaceDiskIntegrator（用于完整建模）
        self.integrator = None
        self.line_model = None
        self.make_integrator()

    def _default_params(self):
        """默认参数集合"""
        return {
            'inclination': 60.0,
            'vsini': 25.0,
            'period': 1.0,
            'pOmega': -0.05,
            'radius': 0.5,
            'r_out': 2.0,
            'nRingsStellarGrid': 60,
        }

    def make_grid(self):
        """生成盘网格"""
        # 将外半径（R_sun单位）转为网格单位（通常以 radius 为单位）
        r_out_normalized = self.r_out / self.radius if self.radius > 0 else 2.0

        self.grid = diskGrid(
            nr=self.nrings,
            r_in=1.0,  # 从恒星半径开始
            r_out=r_out_normalized,
            target_pixels_per_ring=16)
        print(f"✓ 盘网格创建完成: {self.nrings} 环, r_out={r_out_normalized:.2f}R*")
        print(f"  总像素数: {self.grid.numPoints}")

    def make_spot_collection(self):
        """生成单个spot的集合"""
        # Spot 参数
        r_spot = 2.0  # 2R*
        phi_initial = 0.0  # 初始方位角 (弧度)
        amplitude = 0.5  # 发射强度（相对于连续谱）
        radius_spot = 0.3  # spot半径
        B_amplitude = 1000.0  # 1000 Gauss
        B_direction = 'radial'

        spot = Spot(r=r_spot,
                    phi_initial=phi_initial,
                    amplitude=amplitude,
                    spot_type='emission',
                    radius=radius_spot,
                    B_amplitude=B_amplitude,
                    B_direction=B_direction)

        self.spot_collection = SpotCollection(spots=[spot],
                                              pOmega=self.pOmega,
                                              r0=self.radius,
                                              period=self.period)
        print("✓ Spot集合创建完成:")
        print(
            f"  Spot: r={r_spot}R*, phi_0=0, amp={amplitude}, B={B_amplitude} G"
        )

    def make_integrator(self):
        """创建 VelspaceDiskIntegrator 用于完整建模"""
        # 初始化谱线模型
        self.line_model = GaussianZeemanWeakLineModel(line_data=self.line_data,
                                                      k_QU=1.0,
                                                      enable_V=True,
                                                      enable_QU=True)

        # 简单的几何对象包装（VelspaceDiskIntegrator 需要）
        class SimpleGeometry:

            def __init__(self, grid, inclination, vsini, period, pomega,
                         radius):
                self.grid = grid
                self.inclination_rad = np.deg2rad(inclination)
                self.inclination_deg = inclination
                self.vsini = vsini
                self.period = period
                self.pomega = pomega
                self.radius = radius
                self.phi0 = 0.0

        geom = SimpleGeometry(self.grid, self.inclination, self.vsini,
                              self.period, self.pOmega, self.radius)

        # 计算速度网格
        v_grid = np.linspace(-300, 300, 1000)

        try:
            self.integrator = VelspaceDiskIntegrator(
                geom=geom,
                wl0_nm=self.line_data.wl0,
                v_grid=v_grid,
                line_model=self.line_model,
                inst_fwhm_kms=0.5,
                disk_v0_kms=self.vsini,
                disk_power_index=self.pOmega,
                disk_r0=self.radius)
            print("✓ VelspaceDiskIntegrator 已初始化")
        except Exception as e:
            print(f"⚠ VelspaceDiskIntegrator 初始化警告: {e}")
            self.integrator = None

    def get_spot_properties_at_phase(self, phase):
        """
        获取指定相位处spot的属性
        
        Args:
            phase: 相位 (0-1, 单位为周期)
            
        Returns:
            dict: 包含 spot 位置、方位角等信息
        """
        evolved = self.spot_collection.evolve_to_phase(phase)
        return evolved  # [(r, phi, amp, type), ...]

    def map_spot_to_pixels(self, phase):
        """
        将spot映射到网格像素，生成每像素的振幅与磁场
        
        Args:
            phase: 相位 (0-1)
            
        Returns:
            tuple: (amplitudes, Blos, Bperp, chi)
              - amplitudes: (Npix,) 每像素的发射/吸收振幅
              - Blos: (Npix,) 每像素的视向磁场 (G)
              - Bperp: (Npix,) 每像素的横向磁场 (G)
              - chi: (Npix,) 磁场方向角 (弧度)
        """
        evolved = self.get_spot_properties_at_phase(phase)
        Npix = self.grid.numPoints

        amplitudes = np.zeros(Npix, dtype=float)
        Blos = np.zeros(Npix, dtype=float)
        Bperp = np.zeros(Npix, dtype=float)
        chi = np.zeros(Npix, dtype=float)

        inc_rad = np.deg2rad(self.inclination)

        for spot_r, spot_phi, spot_amp, spot_type in evolved:
            # 找出在spot范围内的像素
            dr_pixel = np.abs(self.grid.r - spot_r)
            dphi_pixel = np.minimum(
                np.abs(self.grid.phi - spot_phi),
                2 * np.pi - np.abs(self.grid.phi - spot_phi))

            # 高斯分布
            spot_rad = 0.3  # spot半径 (R_sun单位)
            mask = (dr_pixel**2 + (spot_rad * dphi_pixel)**2) < (3 *
                                                                 spot_rad)**2

            if not np.any(mask):
                continue

            # 计算高斯权重
            dist_sq = dr_pixel[mask]**2 + (spot_rad * dphi_pixel[mask])**2
            weight = np.exp(-dist_sq / (spot_rad**2))

            # 更新像素属性
            amplitudes[mask] += spot_amp * weight

            # 磁场：1000 G, 径向磁场
            B_amplitude = self.spot_collection.spots[0].B_amplitude
            if self.spot_collection.spots[0].B_direction == 'radial':
                # 径向磁场 -> 视向分量随方位角变化
                Blos[mask] += B_amplitude * np.cos(spot_phi) * np.sin(
                    inc_rad) * weight
                Bperp[mask] += B_amplitude * np.sin(inc_rad) * weight
                chi[mask] = 0.0  # 假设无旋转
            else:
                # 环向磁场
                Blos[mask] += B_amplitude * np.sin(inc_rad) * weight

        # 正规化磁场（只对有非零振幅的像素）
        nonzero_mask = amplitudes > 1e-10
        B_amplitude = self.spot_collection.spots[0].B_amplitude

        # Blos 只在有spot的地方非零
        Blos = np.where(nonzero_mask, Blos, 0.0)
        # Bperp 需要至少有一个非零值，避免后续计算中的除以零
        Bperp = np.where(Bperp > 1e-10, Bperp, 1.0)

        return amplitudes, Blos, Bperp, chi


def synthesize_stokes_spectrum(model: SingleSpotModel,
                               phase: float,
                               v_grid=None,
                               inst_fwhm_kms: float = 0.5) -> dict:
    """
    合成指定相位的Stokes IVQU谱线
    
    Args:
        model: SingleSpotModel 实例
        phase: 相位 (0-1)
        v_grid: 速度网格 (km/s)，默认 [-300, 300] 共1000点
        inst_fwhm_kms: 仪器分辨率 (km/s)
        
    Returns:
        dict: {'wl': wl, 'specI': I, 'specV': V, 'specQ': Q, 'specU': U}
    """
    if v_grid is None:
        v_grid = np.linspace(-300, 300, 1000)

    # 获取spot属性映射
    amp, Blos, Bperp, chi = model.map_spot_to_pixels(phase)
    Npix = len(amp)

    # 初始化谱线模型
    line_model = GaussianZeemanWeakLineModel(line_data=model.line_data,
                                             k_QU=1.0,
                                             enable_V=True,
                                             enable_QU=True)

    # 转换速度网格为波长网格
    wl0 = model.line_data.wl0
    c_kms = 299792.458
    wl = wl0 * (1.0 + v_grid / c_kms)

    # 计算每个像素的谱线（逐像素调用）
    Nlambda = len(wl)
    specI_pix = np.zeros((Nlambda, Npix))
    specV_pix = np.zeros((Nlambda, Npix))
    specQ_pix = np.zeros((Nlambda, Npix))
    specU_pix = np.zeros((Nlambda, Npix))

    for i in range(Npix):
        # 只有在该像素有振幅时才计算
        if np.abs(amp[i]) > 1e-10:
            local_profiles = line_model.compute_local_profile(
                wl,
                amp[i],  # 标量
                Blos=Blos[i],  # 标量
                Bperp=Bperp[i],  # 标量
                chi=chi[i],  # 标量
                enable_V=True,
                enable_QU=True)
            # 结果形状: (Nlambda, 1)
            specI_pix[:, i] = local_profiles['I'][:, 0]
            specV_pix[:, i] = local_profiles['V'][:, 0]
            specQ_pix[:, i] = local_profiles['Q'][:, 0]
            specU_pix[:, i] = local_profiles['U'][:, 0]

    # 合成：对所有像素求和，再按面积加权
    area_weights = model.grid.area / np.sum(model.grid.area)

    specI = np.average(specI_pix, axis=1, weights=area_weights)
    specV = np.average(specV_pix, axis=1, weights=area_weights)
    specQ = np.average(specQ_pix, axis=1, weights=area_weights)
    specU = np.average(specU_pix, axis=1, weights=area_weights)

    # 仪器卷积 (简单高斯卷积)
    if inst_fwhm_kms > 0:
        sigma_pix = inst_fwhm_kms / np.mean(
            np.diff(v_grid)) / (2 * np.sqrt(2 * np.log(2)))
        from scipy.ndimage import gaussian_filter1d
        specI = gaussian_filter1d(specI, sigma_pix)
        specV = gaussian_filter1d(specV, sigma_pix)
        specQ = gaussian_filter1d(specQ, sigma_pix)
        specU = gaussian_filter1d(specU, sigma_pix)

    return {
        'v': v_grid,
        'wl': wl,
        'specI': specI,
        'specV': specV,
        'specQ': specQ,
        'specU': specU
    }


def plot_stokes_spectrum(spec_result: dict,
                         phase: float,
                         save_path: str = None):
    """
    可视化Stokes IVQU谱线
    
    Args:
        spec_result: synthesize_stokes_spectrum 返回的字典
        phase: 相位值（用于标题）
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Stokes IVQU Spectrum (Single Spot, Phase={phase:.2f})',
                 fontsize=14,
                 fontweight='bold')

    v = spec_result['v']

    # I 谱线
    axes[0].plot(v,
                 spec_result['specI'],
                 'b-',
                 linewidth=1.5,
                 label='Stokes I')
    axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('I / I_c', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # V 谱线
    axes[1].plot(v,
                 spec_result['specV'],
                 'r-',
                 linewidth=1.5,
                 label='Stokes V')
    axes[1].axhline(0.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('V / I_c', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # Q 谱线
    axes[2].plot(v,
                 spec_result['specQ'],
                 'g-',
                 linewidth=1.5,
                 label='Stokes Q')
    axes[2].axhline(0.0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Q / I_c', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    # U 谱线
    axes[3].plot(v,
                 spec_result['specU'],
                 'm-',
                 linewidth=1.5,
                 label='Stokes U')
    axes[3].axhline(0.0, color='gray', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('U / I_c', fontsize=11)
    axes[3].set_xlabel('Velocity (km/s)', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 谱线图已保存: {save_path}")

    return fig, axes


def save_model_diagnostics(model: SingleSpotModel, output_dir: Path):
    """
    保存模型诊断信息用于可视化检查
    
    Args:
        model: SingleSpotModel 实例
        output_dir: 输出目录
    """
    # 收集模型信息
    diagnostics = {
        'inclination': model.inclination,
        'vsini': model.vsini,
        'period': model.period,
        'pOmega': model.pOmega,
        'radius': model.radius,
        'r_out': model.r_out,
        'nrings': model.nrings,
        'numPoints': model.grid.numPoints,
        'grid_r': model.grid.r.copy(),
        'grid_phi': model.grid.phi.copy(),
        'grid_area': model.grid.area.copy(),
        'grid_dr_cell': model.grid.dr_cell.copy(),
        'grid_dphi_cell': model.grid.dphi_cell.copy(),
        'line_wl0': model.line_data.wl0,
        'line_sigWl': model.line_data.sigWl,
        'line_g': model.line_data.g,
    }

    # 如果VelspaceDiskIntegrator可用，添加更多信息
    if model.integrator is not None:
        diagnostics['integrator_available'] = True
        diagnostics['integrator_type'] = 'VelspaceDiskIntegrator'
    else:
        diagnostics['integrator_available'] = False

    # 添加多个相位的Spot位置
    phase_list = [0.0, 0.25, 0.5, 0.75]
    for phase in phase_list:
        amp, Blos, Bperp, chi = model.map_spot_to_pixels(phase)
        nonzero_idx = np.where(np.abs(amp) > 1e-10)[0]

        diagnostics[f'phase_{phase:.2f}_nonzero_count'] = len(nonzero_idx)
        diagnostics[f'phase_{phase:.2f}_amp_max'] = np.max(np.abs(amp))
        Blos_min, Blos_max = np.min(Blos), np.max(Blos)
        diagnostics[f'phase_{phase:.2f}_Blos_range'] = (Blos_min, Blos_max)
        if len(nonzero_idx) > 0:
            diagnostics[f'phase_{phase:.2f}_spot_pixels'] = nonzero_idx

    # 保存为NPZ格式
    diag_file = output_dir / "model_diagnostics.npz"
    np.savez(diag_file, **diagnostics, allow_pickle=True)
    print(f"    ✓ 模型诊断信息已保存: {diag_file}")

    # 同时保存为文本格式便于查看
    txt_file = output_dir / "model_diagnostics.txt"
    with open(txt_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Single Spot Model Diagnostics\n")
        f.write("=" * 70 + "\n\n")

        f.write("观测参数\n")
        f.write("-" * 70 + "\n")
        f.write(f"倾角: {diagnostics['inclination']} deg\n")
        f.write(f"v sin(i): {diagnostics['vsini']} km/s\n")
        f.write(f"周期: {diagnostics['period']} 天\n")
        f.write(f"差速指数: {diagnostics['pOmega']}\n")
        f.write(f"恒星半径: {diagnostics['radius']} R_sun\n")
        f.write(f"外盘半径: {diagnostics['r_out']} R_sun\n\n")

        f.write("网格信息\n")
        f.write("-" * 70 + "\n")
        f.write(f"环数: {diagnostics['nrings']}\n")
        f.write(f"总像素数: {diagnostics['numPoints']}\n")
        f.write(f"网格r范围: [{diagnostics['grid_r'].min():.4f}, "
                f"{diagnostics['grid_r'].max():.4f}]\n")
        f.write(f"网格面积总和: {diagnostics['grid_area'].sum():.6f}\n\n")

        f.write("谱线参数\n")
        f.write("-" * 70 + "\n")
        f.write(f"中心波长 λ0: {diagnostics['line_wl0']:.4f} nm\n")
        f.write(f"Doppler宽度 σ: {diagnostics['line_sigWl']:.6f} nm\n")
        f.write(f"Landé g因子: {diagnostics['line_g']:.2f}\n\n")

        f.write("多相位Spot诊断\n")
        f.write("-" * 70 + "\n")
        for phase in phase_list:
            f.write(f"\nPhase = {phase:.2f}:\n")
            key_count = f'phase_{phase:.2f}_nonzero_count'
            key_max = f'phase_{phase:.2f}_amp_max'
            key_Blos = f'phase_{phase:.2f}_Blos_range'

            if key_count in diagnostics:
                f.write(f"  非零振幅像素数: {diagnostics[key_count]}\n")
                f.write(f"  最大振幅: {diagnostics[key_max]:.6f}\n")
                Blos_min, Blos_max = diagnostics[key_Blos]
                f.write(f"  Blos范围: [{Blos_min:.2f}, {Blos_max:.2f}] G\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(
            f"VelspaceDiskIntegrator: "
            f"{'Available' if diagnostics['integrator_available'] else 'Not Available'}\n"
        )
        f.write("=" * 70 + "\n")

    print(f"    ✓ 诊断报告已保存: {txt_file}")


def main():
    """主函数：运行完整的单spot建模与可视化"""

    print("\n" + "=" * 70)
    print(" 单Spot Stokes IVQU 谱线建模与可视化测试")
    print("=" * 70 + "\n")

    # 文件路径
    input_dir = project_root / "input"
    output_dir = project_root / "test_output"
    output_dir.mkdir(exist_ok=True)

    params_file = input_dir / "params_tomog.txt"
    lines_file = input_dir / "lines.txt"

    print("[1] 读取输入文件...")
    print(f"    参数文件: {params_file}")
    print(f"    谱线文件: {lines_file}")

    # 初始化模型
    model = SingleSpotModel(str(params_file), str(lines_file))

    print("\n[2] 模型参数:")
    print(f"    倾角: {model.inclination}°")
    print(f"    v sin(i): {model.vsini} km/s")
    print(f"    周期: {model.period} 天")
    print(f"    差速指数: {model.pOmega}")
    print(f"    恒星半径: {model.radius} R_sun")
    print(f"    外盘半径: {model.r_out} R_sun")

    # 遍历多个相位合成谱线
    phases = np.array([0.0, 0.25, 0.5, 0.75])
    results = {}

    print("\n[3] 合成Stokes IVQU谱线...")
    for phase in phases:
        print(f"    处理相位: {phase:.2f}...", end='', flush=True)
        spec_result = synthesize_stokes_spectrum(model, phase)
        results[phase] = spec_result
        print(" ✓")

    # 可视化与保存
    print("\n[4] 绘制谱线图...")
    for i, phase in enumerate(phases):
        fig, axes = plot_stokes_spectrum(
            results[phase],
            phase,
            save_path=str(output_dir / f"stokes_spectrum_phase_{i:02d}.png"))
        plt.close(fig)

    # 输出统计信息
    print("\n[5] 谱线统计信息:")
    phase = phases[0]
    spec = results[phase]
    print(f"    速度范围: [{spec['v'].min():.1f}, {spec['v'].max():.1f}] km/s")
    print(f"    波长范围: [{spec['wl'].min():.4f}, {spec['wl'].max():.4f}] nm")
    print(
        f"    Stokes I 范围: [{spec['specI'].min():.6f}, {spec['specI'].max():.6f}]"
    )
    print(
        f"    Stokes V 范围: [{spec['specV'].min():.6f}, {spec['specV'].max():.6f}]"
    )
    print(
        f"    Stokes Q 范围: [{spec['specQ'].min():.6f}, {spec['specQ'].max():.6f}]"
    )
    print(
        f"    Stokes U 范围: [{spec['specU'].min():.6f}, {spec['specU'].max():.6f}]"
    )

    # 保存数据
    print("\n[6] 保存数据...")
    data_file = output_dir / "single_spot_stokes_data.npz"
    np.savez(data_file,
             phases=phases,
             **{
                 f"phase_{i:02d}": results[p]
                 for i, p in enumerate(phases)
             })
    print(f"    ✓ 数据已保存: {data_file}")

    # 保存模型信息用于可视化检查
    print("\n[7] 保存模型信息...")
    save_model_diagnostics(model, output_dir)

    # 生成 .tomog 模型文件供可视化
    print("\n[8] 生成 .tomog 几何模型文件...")
    save_geomodel_files(model, phases, output_dir)

    print("\n" + "=" * 70)
    print(" 测试完成！输出已保存到 test_output/")
    print("=" * 70 + "\n")


def save_geomodel_files(model: SingleSpotModel, phases: list,
                        output_dir: Path):
    """
    生成 .tomog 几何模型文件供 utils/visualize_geomodel.py 使用
    
    Args:
        model: SingleSpotModel 实例
        phases: 相位列表 [0.0, 0.25, ...]
        output_dir: 输出目录
    """
    print("\n[8] 生成 .tomog 模型文件...")

    for phase_idx, phase in enumerate(phases):
        # 获取该相位的Spot属性
        amp, Blos, Bperp, chi = model.map_spot_to_pixels(phase)

        # 创建几何模型数据表
        table_data = {
            'idx': np.arange(model.grid.numPoints),
            'ring_id': model.grid.ring_id,
            'phi_id': model.grid.phi_id,
            'r': model.grid.r,
            'phi': model.grid.phi,
            'area': model.grid.area,
            'Ic_weight': amp,  # 使用振幅作为权重
            'A': amp,  # 响应
            'Blos': Blos,
            'Bperp': Bperp,
            'chi': chi,
        }

        # 保存为 .tomog 文本文件
        tomog_file = output_dir / f"geomodel_phase_{phase_idx:02d}.tomog"

        with open(tomog_file, 'w') as f:
            # 写入头部
            f.write("# TOMOG_MODEL format v1.0\n")
            f.write(f"# Single Spot Model - Phase {phase:.2f}\n")
            f.write("# Generated by test_single_spot_stokes_ivqu.py\n")
            f.write("#\n")
            f.write(f"# inclination_deg = {model.inclination}\n")
            f.write(f"# vsini = {model.vsini}\n")
            f.write(f"# period = {model.period}\n")
            f.write(f"# pOmega = {model.pOmega}\n")
            f.write(f"# phase = {phase:.2f}\n")
            f.write(f"# line_wl0_nm = {model.line_data.wl0}\n")
            f.write(f"# line_g = {model.line_data.g}\n")
            f.write("#\n")

            # 写入列名
            columns = [
                'idx', 'ring_id', 'phi_id', 'r', 'phi', 'area', 'Ic_weight',
                'A', 'Blos', 'Bperp', 'chi'
            ]
            f.write("# COLUMNS: " + " ".join(columns) + "\n")

            # 写入数据行
            for i in range(model.grid.numPoints):
                row_data = [
                    table_data['idx'][i],
                    table_data['ring_id'][i],
                    table_data['phi_id'][i],
                    table_data['r'][i],
                    table_data['phi'][i],
                    table_data['area'][i],
                    table_data['Ic_weight'][i],
                    table_data['A'][i],
                    table_data['Blos'][i],
                    table_data['Bperp'][i],
                    table_data['chi'][i],
                ]
                f.write(" ".join(f"{x:.10g}" for x in row_data) + "\n")

        print(f"    ✓ {tomog_file.name}")

    print(f"    总共生成 {len(phases)} 个 .tomog 文件")


if __name__ == "__main__":
    main()
