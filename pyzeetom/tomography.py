"""pyzeetom.tomography — 简化的主程序

本版本按项目重组约定：
- 基本线模型来自 core/local_linemodel_*（弱场高斯线型等）；
- 结构/积分模型使用 core/velspace_DiskIntegrator.VelspaceDiskIntegrator；
"""
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# 添加项目根目录到路径（用于直接运行此脚本）
if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import core.mainFuncs as mf
import core.SpecIO as SpecIO
from core.grid_tom import diskGrid
from core.velspace_DiskIntegrator import VelspaceDiskIntegrator
from core.local_linemodel_basic import LineData as BasicLineData, GaussianZeemanWeakLineModel


class SimpleDiskGeometry:
    """最小化的盘几何容器，供 VelspaceDiskIntegrator 使用。
    提供：grid, area_proj, inclination_rad, phi0, pOmega, r0, period,
         enable_stellar_occultation, stellar_radius, B_los, B_perp, chi。
    """

    def __init__(self,
                 grid: diskGrid,
                 inclination_deg: float = 60.0,
                 phi0: float = 0.0,
                 pOmega: float = 0.0,
                 r0: float = 1.0,
                 period: float = 1.0,
                 enable_stellar_occultation: int = 0,
                 stellar_radius: float = 1.0,
                 B_los: Optional[np.ndarray] = None,
                 B_perp: Optional[np.ndarray] = None,
                 chi: Optional[np.ndarray] = None):
        self.grid = grid
        self.area_proj = np.asarray(grid.area)
        self.inclination_rad = np.deg2rad(float(inclination_deg))
        self.phi0 = float(phi0)
        self.pOmega = float(pOmega)
        self.r0 = float(r0)
        self.period = float(period)
        self.enable_stellar_occultation = bool(enable_stellar_occultation)
        self.stellar_radius = float(stellar_radius)
        # 添加磁场参数
        self.B_los = B_los if B_los is not None else np.zeros(grid.numPoints)
        self.B_perp = B_perp if B_perp is not None else np.zeros(
            grid.numPoints)
        self.chi = chi if chi is not None else np.zeros(grid.numPoints)


def _save_model_spectra_with_polchannel(results,
                                        basepath="model_phase",
                                        fmt="lsd"):
    """保存模型光谱，支持每个相位的pol_channel
    
    Parameters
    ----------
    results : list of tuples
        每个元组为 (v_grid, I, V, Q, U, pol_channel)
    basepath : str
        输出文件名前缀
    fmt : str
        输出格式 ('lsd' 或 'spec')
    """
    for i, result in enumerate(results):
        v_grid, I, V, Q, U, pol_channel = result
        ext = 'lsd' if fmt == 'lsd' else 'spec'
        fname = f"{basepath}_{i:02d}.{ext}"
        header = {"phase_index": str(i), "pol_channel": pol_channel}

        # 根据pol_channel选择输出的偏振分量
        if pol_channel == 'Q':
            pol_data = Q
        elif pol_channel == 'U':
            pol_data = U
        else:  # V 或其他
            pol_data = V

        # 调用write_model_spectrum，传入pol_channel以自动推断输出格式
        SpecIO.write_model_spectrum(fname,
                                    v_grid,
                                    I,
                                    V=pol_data if pol_channel == 'V' else None,
                                    Q=pol_data if pol_channel == 'Q' else None,
                                    U=pol_data if pol_channel == 'U' else None,
                                    fmt=fmt,
                                    header=header,
                                    pol_channel=pol_channel)


class ConstantAmpLineModel:
    """将需要 amp 的基本线模型包装为无需显式 amp 的接口。
    统一使用常数振幅 amp_const（<0 吸收）。
    """

    def __init__(self,
                 base_model: GaussianZeemanWeakLineModel,
                 amp_const: float = -0.5):
        self.base = base_model
        self.amp_const = float(amp_const)

    def compute_local_profile(self,
                              wl_grid,
                              amp=None,
                              Blos=None,
                              Ic_weight=None,
                              **kwargs):
        """计算局部谱线轮廓。
        
        注意：amp参数被忽略，使用self.amp_const代替。
        这样保持与BaseLineModel的接口一致性。
        """
        wl_grid = np.asarray(wl_grid)
        if wl_grid.ndim == 1:
            Npix = 1
        else:
            Npix = wl_grid.shape[1]
        amp_to_use = np.full((Npix, ), self.amp_const, dtype=float)
        return self.base.compute_local_profile(wl_grid,
                                               amp=amp_to_use,
                                               Blos=Blos,
                                               Ic_weight=Ic_weight,
                                               **kwargs)


def main(
        par=None,
        obsSet=None,
        lineData: Optional[BasicLineData] = None,
        verbose: int = 1,
        run_mem: bool = False
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """简化主入口：读取参数与观测，构建盘网格与几何，生成各相位的合成 I/V。
    
    支持两种模式：
    1. 正演模式（run_mem=False）：生成单次合成谱
    2. MEM反演模式（run_mem=True）：执行MEM迭代优化
    
    返回：[(v_grid, I_model, V_model), ...] 与观测一一对应。
    """
    # 1) 读参数
    if par is None:
        par = mf.readParamsTomog('input/params_tomog.txt')

    # 2) 读观测
    if obsSet is None:
        file_type = getattr(par, 'obsFileType', 'auto')
        pol_channels = getattr(par, 'polChannels', None)
        if pol_channels is not None:
            pol_channels = list(pol_channels)
        obsSet = SpecIO.obsProfSetInRange(list(par.fnames),
                                          par.velStart,
                                          par.velEnd,
                                          par.velRs,
                                          file_type=file_type,
                                          pol_channels=pol_channels)

    # 3) 线数据与线模型
    if lineData is None:
        line_file = getattr(par, 'lineParamFile', 'lines.txt')
        lineData = BasicLineData(line_file)

    # 计算仪器FWHM（根据光谱分辨率和谱线波长）
    par.compute_instrument_fwhm(lineData.wl0, verbose=verbose)

    # 从参数加载线模型设置
    k_qu = float(getattr(par, 'lineKQU', 1.0))
    enable_v = bool(getattr(par, 'lineEnableV', 1))
    enable_qu = bool(getattr(par, 'lineEnableQU', 1))
    base_model = GaussianZeemanWeakLineModel(lineData,
                                             k_QU=k_qu,
                                             enable_V=enable_v,
                                             enable_QU=enable_qu)
    # 常数振幅（amp<0 为吸收）
    amp_const = float(getattr(par, 'lineAmpConst', -0.5))
    line_model = ConstantAmpLineModel(base_model, amp_const=amp_const)

    # 4) 盘网格与几何（全相位复用）
    # 根据 Vmax 或 radius+r_out 确定网格外半径
    nr = getattr(par, 'nRingsStellarGrid', 60)

    # Vmax 已在 readParamsTomog 中计算（基于 r_out 和差速转动）
    # r_out 已经是恒星半径为单位，网格需要物理单位（R_sun）
    # 因此：r_out_grid = radius * r_out（将恒星半径转换为R_sun）
    pOmega = getattr(par, 'pOmega', 0.0)
    radius = getattr(par, 'radius', 1.0)  # R_sun
    Vmax = getattr(par, 'Vmax', 0.0)
    velEq = getattr(par, 'velEq', getattr(par, 'vsini', 100.0))

    if abs(pOmega + 1.0) > 1e-6:
        # 一般情况：从 Vmax 反推 r_out（恒星半径为单位）
        # v(r) = veq * (r/R*)^(pOmega+1) => r = (v/veq)^(1/(pOmega+1))
        r_out_stellar = (Vmax / velEq)**(1.0 / (pOmega + 1.0))
        r_out_grid = radius * r_out_stellar  # 转换为R_sun
    else:
        # 特殊情况：pOmega = -1（恒定角动量），v(r) = const
        # 此时任何 r_out 都给出相同速度，使用参数文件中的 r_out
        r_out_stellar = getattr(par, 'r_out', 5.0)
        r_out_grid = radius * r_out_stellar

    if verbose:
        print(
            f"[Grid] Building grid with nr={nr}, r_out={r_out_grid:.3f}R_sun ({r_out_stellar:.3f}R*), Vmax={Vmax:.2f} km/s"
        )

    grid = diskGrid(nr=nr, r_in=0.0, r_out=r_out_grid, verbose=(verbose > 0))

    # 从参数中获取差速转动信息
    period = getattr(par, 'period', 1.0)
    enable_occultation = getattr(par, 'enable_stellar_occultation', 0)

    # 初始化磁场参数
    npix = grid.numPoints
    B_los_init = np.zeros(npix)  # 默认 0 Gauss
    B_perp_init = np.zeros(npix)  # 默认 0 Gauss
    chi_init = np.zeros(npix)  # 默认 0

    geom = SimpleDiskGeometry(grid,
                              inclination_deg=par.inclination,
                              pOmega=pOmega,
                              r0=radius,
                              period=period,
                              enable_stellar_occultation=enable_occultation,
                              stellar_radius=radius,
                              B_los=B_los_init,
                              B_perp=B_perp_init,
                              chi=chi_init)

    # 5) 每个观测相位计算合成谱
    # 确保已计算相位信息
    if not hasattr(par, 'phases') or par.phases is None:
        if verbose:
            print("警告：未找到相位信息，尝试从 jDateRef 和 period 计算...")
        if hasattr(par, 'jDateRef') and hasattr(par, 'period'):
            par.phases = mf.compute_phase_from_jd(par.jDates, par.jDateRef,
                                                  par.period)
        else:
            # 无相位信息，使用 0.0（将不进行时间演化）
            par.phases = np.zeros(len(obsSet))
            if verbose:
                print("警告：无法计算相位，将不考虑时间演化")

    # 分支：正演模式或MEM反演模式
    if not run_mem:
        # 正演模式：单次生成所有相位合成谱
        results = _run_forward_mode(par, obsSet, geom, lineData, line_model,
                                    velEq, pOmega, radius, verbose)
    else:
        # MEM反演模式：执行迭代优化
        results = _run_mem_mode(par, obsSet, grid, geom, lineData, line_model,
                                velEq, pOmega, radius, verbose)

    return results


def _run_forward_mode(par, obsSet, geom, lineData, line_model, velEq, pOmega,
                      radius, verbose):
    """正演模式：生成单次合成谱"""
    results = []
    for i, obs in enumerate(obsSet):
        v_grid = np.asarray(obs.wl, dtype=float)  # LSD: 这里即速度网格

        # 获取当前观测的相位
        current_phase = float(
            par.phases[i]) if par.phases is not None else None

        inte = VelspaceDiskIntegrator(
            geom=geom,
            wl0_nm=lineData.wl0,
            v_grid=v_grid,
            line_model=line_model,
            inst_fwhm_kms=getattr(par, 'instrumentRes', 0.0),
            normalize_continuum=True,
            disk_v0_kms=velEq,
            disk_power_index=pOmega,
            disk_r0=radius,
            time_phase=current_phase,  # 传递相位信息
        )
        # 获取当前观测的偏振通道，根据通道返回相应的Stokes分量
        pol_channel = getattr(obs, 'pol_channel', 'V')
        results.append((v_grid, inte.I, inte.V, inte.Q, inte.U, pol_channel))

    # 6) 可选：保存一个简易输出以便快速查看
    with open('out_synth.txt', 'w') as f:
        for i, result in enumerate(results):
            v, I, V, Q, U, pol_ch = result
            f.write(f"# phase_index {i}, pol_channel {pol_ch}\n")
            for k in range(v.size):
                if pol_ch == 'I':
                    f.write(f"{v[k]:.6f} {I[k]:.8e}\n")
                elif pol_ch == 'Q':
                    f.write(f"{v[k]:.6f} {I[k]:.8e} {Q[k]:.8e}\n")
                elif pol_ch == 'U':
                    f.write(f"{v[k]:.6f} {I[k]:.8e} {U[k]:.8e}\n")
                else:  # V
                    f.write(f"{v[k]:.6f} {I[k]:.8e} {V[k]:.8e}\n")
            f.write("\n")

    # 7) 导出第一个相位的几何模型为 geomodel.tomog（示例）
    if verbose and len(results) > 0:
        print("[tomography] 导出第一相位几何模型到 output/geomodel_phase0.tomog ...")
        # 重新构造一个积分器以获取完整的几何对象（或从上面保存）
        v_grid = np.asarray(obsSet[0].wl, dtype=float)
        phase0 = float(par.phases[0]) if par.phases is not None else None
        inte0 = VelspaceDiskIntegrator(
            geom=geom,
            wl0_nm=lineData.wl0,
            v_grid=v_grid,
            line_model=line_model,
            inst_fwhm_kms=getattr(par, 'instrumentRes', 0.0),
            normalize_continuum=True,
            disk_v0_kms=velEq,
            disk_power_index=pOmega,
            disk_r0=radius,
            time_phase=phase0,
        )
        meta_info = {
            "target": getattr(par, "target_name", "Unknown"),
            "period": getattr(par, "period", 1.0),
            "phase0": phase0 if phase0 is not None else 0.0,
            "n_observations": len(obsSet),
        }
        inte0.write_geomodel("output/geomodel_phase0.tomog", meta=meta_info)

    # 8) 保存所有相位的模型光谱到单独文件（示例）
    if verbose:
        print("[tomography] 保存所有相位模型光谱到 output/outModel/ ...")
        # 使用save_model_spectra函数，输出到标准目录
        phase_indices = [i for i in range(len(results))]
        results_for_save = [(r[0], r[1], r[2], r[3],
                             r[4]) if len(r) >= 6 else r for r in results]
        mf.save_model_spectra(results_for_save,
                              phase_indices,
                              output_dir="output/outModel",
                              fmt="lsd",
                              prefix="phase")

    return results


def _run_mem_mode(par, obsSet, grid, geom, lineData, line_model, velEq, pOmega,
                  radius, verbose):
    """MEM反演模式：执行迭代优化"""
    from core.mem_tomography import (MEMTomographyAdapter, MagneticFieldParams)

    if verbose:
        print("\n" + "=" * 70)
        print("MEM反演模式")
        print("=" * 70)

    # 初始化磁场参数
    npix = len(grid.r)
    mag_field = MagneticFieldParams(Blos=np.zeros(npix),
                                    Bperp=np.zeros(npix),
                                    chi=np.zeros(npix))

    # 初始化MEM适配器
    adapter = MEMTomographyAdapter(fit_brightness=True,
                                   fit_magnetic=True,
                                   entropy_weights_blos=grid.area,
                                   entropy_weights_bperp=grid.area,
                                   entropy_weights_chi=grid.area * 0.1,
                                   default_blos=10.0,
                                   default_bperp=100.0,
                                   default_chi=0.0)

    # 打包初始参数
    Image, n_bright, n_blos, n_bperp, n_chi = adapter.pack_image_vector(
        mag_field=mag_field)

    # 读取迭代参数
    max_iterations = getattr(par, 'numIterations', 20)
    convergence_threshold = getattr(par, 'test_aim', 1e-3)

    if verbose:
        print("\n初始化完成:")
        print(f"  网格像素数: {npix}")
        print(f"  参数向量长度: {len(Image)}")
        print(f"  最大迭代次数: {max_iterations}")
        print(f"  收敛阈值: {convergence_threshold}")

        if max_iterations == 0:
            print("\n⚠️  numIterations=0: 仅生成初始正演模型（无MEM迭代）")
        else:
            print("\n开始MEM迭代...")
        print("-" * 70)

    # 特殊处理：0次迭代，仅生成初始模型
    if max_iterations == 0:
        if verbose:
            print("\n生成初始正演模型...")

        results_this_iter = []
        for i_phase, obs in enumerate(obsSet):
            phase = float(
                par.phases[i_phase]) if par.phases is not None else None
            v_grid = np.asarray(obs.wl, dtype=float)

            # 创建积分器并计算正演
            inte = VelspaceDiskIntegrator(geom=geom,
                                          wl0_nm=lineData.wl0,
                                          v_grid=v_grid,
                                          line_model=line_model,
                                          inst_fwhm_kms=getattr(
                                              par, 'instrumentRes', 0.0),
                                          normalize_continuum=True,
                                          disk_v0_kms=velEq,
                                          disk_power_index=pOmega,
                                          disk_r0=radius,
                                          time_phase=phase)

            results_this_iter.append((v_grid, inte.I, inte.V))

        # 保存初始模型
        if verbose:
            print("  保存初始模型光谱...")
        mf.save_model_spectra(results_this_iter,
                              range(len(obsSet)),
                              output_dir="output/outModel",
                              fmt="lsd",
                              prefix="phase")

        if verbose:
            print("  保存初始tomography模型...")
        meta_info = {
            "target": getattr(par, "target_name", "Unknown"),
            "period": getattr(par, "period", 1.0),
            "iteration": 0,
            "chi2": 0.0,
            "entropy": 0.0,
            "note": "Initial forward model (no MEM iterations)"
        }
        # 使用最后一个 integrator 保存完整模型（包含几何和速度场信息）
        mf.save_geomodel_tomog(grid,
                               mag_field=mag_field,
                               output_file="output/outGeoModel.tomog",
                               meta=meta_info,
                               geom=geom,
                               integrator=inte)

        if verbose:
            print("\n" + "=" * 70)
            print("初始模型生成完成")
            print("=" * 70)
            print("结果已保存至:")
            print("  - 模型光谱: output/outModel/phase*.lsd")
            print("  - Tomography模型: output/outGeoModel.tomog")

        return results_this_iter

    # MEM迭代循环
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n迭代 {iteration + 1}/{max_iterations}")

        # 计算所有相位的正演模型
        all_data = []
        all_model = []
        all_sig2 = []
        all_resp_parts = []

        results_this_iter = []

        for i_phase, obs in enumerate(obsSet):
            phase = float(
                par.phases[i_phase]) if par.phases is not None else None

            v_grid = np.asarray(obs.wl, dtype=float)

            # 创建积分器并计算正演
            inte = VelspaceDiskIntegrator(geom=geom,
                                          wl0_nm=lineData.wl0,
                                          v_grid=v_grid,
                                          line_model=line_model,
                                          inst_fwhm_kms=getattr(
                                              par, 'instrumentRes', 0.0),
                                          normalize_continuum=True,
                                          disk_v0_kms=velEq,
                                          disk_power_index=pOmega,
                                          disk_r0=radius,
                                          time_phase=phase)

            # 收集正演结果，包含pol_channel信息
            pol_channel = getattr(obs, 'pol_channel', 'V')
            results_this_iter.append(
                (v_grid, inte.I, inte.V, inte.Q, inte.U, pol_channel))

            # 计算导数（占位实现）
            forward = mf.compute_forward_single_phase(inte,
                                                      mag_field=mag_field,
                                                      compute_derivatives=True)

            all_data.append(obs.specI)
            all_model.append(forward['specI'])
            all_sig2.append(obs.specIsig**2)

            # 组装响应矩阵
            resp_phase = np.hstack([
                forward['dI_dBlos'], forward['dI_dBperp'],
                np.zeros((len(obs.wl), npix))
            ])
            all_resp_parts.append(resp_phase)

        # 拼接所有数据
        Data = np.concatenate(all_data)
        Fmodel = np.concatenate(all_model)
        sig2 = np.concatenate(all_sig2)
        Resp = np.vstack(all_resp_parts)

        # 计算当前χ²
        chi2_current = np.sum((Fmodel - Data)**2 / sig2)
        chi2_reduced = chi2_current / len(Data)

        if verbose:
            print(
                f"  当前 χ² = {chi2_current:.2f} (reduced = {chi2_reduced:.4f})")

        # 执行MEM迭代
        entropy, chi2, test, Image = adapter.optimizer.iterate(
            Image=Image,
            Fmodel=Fmodel,
            Data=Data,
            sig2=sig2,
            Resp=Resp,
            weights=np.ones_like(Image),
            entropy_params={
                'npix': npix,
                'n_blos': n_blos,
                'n_bperp': n_bperp,
                'n_chi': n_chi
            },
            n1=0,
            n2=0,
            ntot=len(Image),
            fixEntropy=0,
            targetAim=None)

        # 解包参数
        adapter.unpack_image_vector(Image,
                                    n_bright,
                                    n_blos,
                                    n_bperp,
                                    n_chi,
                                    mag_field=mag_field)

        if verbose:
            print(
                f"  迭代后 χ² = {chi2:.2f}, 熵 S = {entropy:.4f}, test = {test:.6f}"
            )
            print(
                f"  Blos: [{mag_field.Blos.min():.1f}, {mag_field.Blos.max():.1f}] G, "
                f"mean={mag_field.Blos.mean():.1f} G")

        # 保存本次迭代结果
        # 1. 更新模型光谱文件
        mf.save_model_spectra(results_this_iter,
                              range(len(obsSet)),
                              output_dir="output/outModel",
                              fmt="lsd",
                              prefix="phase")

        # 2. 更新tomography模型
        meta_info = {
            "target": getattr(par, "target_name", "Unknown"),
            "period": getattr(par, "period", 1.0),
            "iteration": iteration + 1,
            "chi2": chi2,
            "entropy": entropy,
        }
        # 使用最后一个 integrator 保存完整模型（包含几何和速度场信息）
        mf.save_geomodel_tomog(grid,
                               mag_field=mag_field,
                               output_file="output/outGeoModel.tomog",
                               meta=meta_info,
                               geom=geom,
                               integrator=inte)

        # 3. 追加统计摘要
        mf.save_iteration_summary(
            "output/outSummary.txt",
            iteration,
            chi2,
            entropy,
            test,
            mag_field=mag_field,
            mode='append' if iteration > 0 else 'overwrite')

        # 收敛检查
        if test < convergence_threshold:
            if verbose:
                print(f"\n✓ 收敛！(test={test:.6f} < {convergence_threshold})")
            break

    if verbose:
        print("\n" + "=" * 70)
        print("MEM反演完成")
        print("=" * 70)
        print("最终结果已保存至:")
        print("  - 模型光谱: output/outModel/phase*.lsd")
        print("  - Tomography模型: output/outGeoModel.tomog")
        print("  - 统计摘要: output/outSummary.txt")

    return results_this_iter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        "pyZeeTom Stellar Tomography - Forward Model & MEM Inversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 正演模式（默认）
  python tomography.py --params input/params_tomog.txt
  
  # MEM反演模式
  python tomography.py --params input/params_tomog.txt --mem
  
  # 使用自定义谱线参数文件
  python tomography.py --params input/params_tomog.txt --lines input/lines.txt
  
  # 指定参数和线文件
  python tomography.py -p input/params_test.txt -l input/lines.txt --mem
        """)

    parser.add_argument('-p',
                        '--params',
                        type=str,
                        default='input/params_tomog.txt',
                        help='参数文件路径 (default: input/params_tomog.txt)')

    parser.add_argument('-l',
                        '--lines',
                        type=str,
                        default='input/lines.txt',
                        help='谱线参数文件路径（若不指定，从params文件中读取）')

    parser.add_argument('--mem',
                        action='store_true',
                        help='使用MEM反演模式（默认为正演模式）')

    parser.add_argument('-v',
                        '--verbose',
                        type=int,
                        default=1,
                        choices=[0, 1, 2],
                        help='输出详细程度: 0=安静, 1=正常, 2=详细 (default: 1)')

    args = parser.parse_args()

    print("=" * 70)
    print("pyZeeTom Stellar Tomography")
    print("=" * 70)
    print("\n配置:")
    print(f"  参数文件: {args.params}")
    print(f"  谱线文件: {args.lines if args.lines else '(从参数文件读取)'}")
    print(f"  运行模式: {'MEM反演' if args.mem else '正演模型'}")
    print(f"  详细程度: {args.verbose}")
    print()

    # 读取参数
    par = mf.readParamsTomog(args.params)

    # 如果指定了谱线文件，覆盖参数中的设置
    lineData = None
    if args.lines:
        lineData = BasicLineData(args.lines)
        if args.verbose:
            print(f"[Lines] 从命令行指定的文件加载: {args.lines}")

    # 运行主程序
    try:
        results = main(par=par,
                       obsSet=None,
                       lineData=lineData,
                       verbose=args.verbose,
                       run_mem=args.mem)

        print("\n" + "=" * 70)
        print("✓ 运行完成")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ 运行失败: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
