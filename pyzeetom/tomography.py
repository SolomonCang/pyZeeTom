"""pyzeetom.tomography — 简化的主程序

本版本按项目重组约定：
- 基本线模型来自 core/local_linemodel_*（弱场高斯线型等）；
- 结构/积分模型使用 core/velspace_DiskIntegrator.VelspaceDiskIntegrator；
- 不再依赖 magneticGeom/brightnessGeom；
- 参数解析类从 readParamsZDI 更名为 readParamsTomog（在 core/mainFuncs.py 中做了别名）。
"""
from typing import List, Tuple, Optional
import numpy as np

import core.mainFuncs as mf
import core.SpecIO as SpecIO
from core.grid_tom import diskGrid
from core.velspace_DiskIntegrator import VelspaceDiskIntegrator
from core.local_linemodel_basic import LineData as BasicLineData, GaussianZeemanWeakLineModel


class SimpleDiskGeometry:
    """最小化的盘几何容器，供 VelspaceDiskIntegrator 使用。
    提供：grid, area_proj, inclination_rad, phi0, pOmega, r0, period, 
         enable_stellar_occultation, stellar_radius。
    """

    def __init__(self,
                 grid: diskGrid,
                 inclination_deg: float = 60.0,
                 phi0: float = 0.0,
                 pOmega: float = 0.0,
                 r0: float = 1.0,
                 period: float = 1.0,
                 enable_stellar_occultation: int = 0,
                 stellar_radius: float = 1.0):
        self.grid = grid
        self.area_proj = np.asarray(grid.area)
        self.inclination_rad = np.deg2rad(float(inclination_deg))
        self.phi0 = float(phi0)
        self.pOmega = float(pOmega)
        self.r0 = float(r0)
        self.period = float(period)
        self.enable_stellar_occultation = bool(enable_stellar_occultation)
        self.stellar_radius = float(stellar_radius)


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
                              Ic_weight=None,
                              Blos=None,
                              **kwargs):
        wl_grid = np.asarray(wl_grid)
        if wl_grid.ndim == 1:
            Npix = 1
        else:
            Npix = wl_grid.shape[1]
        amp = np.full((Npix, ), self.amp_const, dtype=float)
        return self.base.compute_local_profile(wl_grid,
                                               amp=amp,
                                               Blos=Blos,
                                               Ic_weight=Ic_weight,
                                               **kwargs)


def main(par=None,
         obsSet=None,
         lineData: Optional[BasicLineData] = None,
         verbose: int = 1) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """简化主入口：读取参数与观测，构建盘网格与几何，生成各相位的合成 I/V。

    返回：[(v_grid, I_model, V_model), ...] 与观测一一对应。
    """
    # 1) 读参数
    if par is None:
        par = mf.readParamsTomog('input/params_tomog.txt')

    # 2) 读观测
    if obsSet is None:
        file_type = getattr(par, 'obsFileType', 'auto')
        obsSet = SpecIO.obsProfSetInRange(list(par.fnames),
                                          par.velStart,
                                          par.velEnd,
                                          par.velRs,
                                          file_type=file_type)

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

    # Vmax 已在 readParamsTomog 中计算，现在用它确定 r_out
    # 根据差速转动公式反推：Vmax = veq * (r_out/radius)^(pOmega+1)
    # => r_out = radius * (Vmax/veq)^(1/(pOmega+1))
    pOmega = getattr(par, 'pOmega', 0.0)
    radius = getattr(par, 'radius', 1.0)
    Vmax = getattr(par, 'Vmax', 0.0)
    velEq = getattr(par, 'velEq', getattr(par, 'vsini', 100.0))

    if abs(pOmega + 1.0) > 1e-6:
        # 一般情况：v(r) = veq * (r/R*)^(pOmega+1)
        r_out_grid = radius * (Vmax / velEq)**(1.0 / (pOmega + 1.0))
    else:
        # 特殊情况：pOmega = -1（恒定角动量），v(r) = const
        # 此时任何 r_out 都给出相同速度，使用默认值
        r_out_grid = getattr(par, 'r_out', 5.0 * radius)

    if verbose:
        print(
            f"[Grid] Building grid with nr={nr}, r_out={r_out_grid:.3f}R*, Vmax={Vmax:.2f} km/s"
        )

    grid = diskGrid(nr=nr, r_in=0.0, r_out=r_out_grid, verbose=(verbose > 0))

    # 从参数中获取差速转动信息
    period = getattr(par, 'period', 1.0)
    enable_occultation = getattr(par, 'enable_stellar_occultation', 0)

    geom = SimpleDiskGeometry(grid,
                              inclination_deg=par.inclination,
                              pOmega=pOmega,
                              r0=radius,
                              period=period,
                              enable_stellar_occultation=enable_occultation,
                              stellar_radius=radius)

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
        results.append((v_grid, inte.I, inte.V))

    # 6) 可选：保存一个简易输出以便快速查看
    with open('out_synth.txt', 'w') as f:
        for i, (v, I, V) in enumerate(results):
            f.write(f"# phase_index {i}\n")
            for k in range(v.size):
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
            "period": period,
            "phase0": phase0 if phase0 is not None else 0.0,
            "n_observations": len(obsSet),
        }
        inte0.write_geomodel("output/geomodel_phase0.tomog", meta=meta_info)

    # 8) 保存所有相位的模型光谱到单独文件（示例）
    if verbose:
        print("[tomography] 保存所有相位模型光谱到 output/model_phase_*.lsd ...")
        SpecIO.save_results_series(results,
                                   basepath="output/model_phase",
                                   fmt="lsd")

    return results


if __name__ == "__main__":
    main()
