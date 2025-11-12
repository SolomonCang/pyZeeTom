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
import core.readObs as readObs
from core.grid_tom import diskGrid
from core.velspace_DiskIntegrator import VelspaceDiskIntegrator
from core.local_linemodel_basic import LineData as BasicLineData, GaussianZeemanWeakLineModel


class SimpleDiskGeometry:
    """最小化的盘几何容器，供 VelspaceDiskIntegrator 使用。
    仅提供：grid, area_proj, inclination_rad, phi0。
    """

    def __init__(self,
                 grid: diskGrid,
                 inclination_deg: float = 60.0,
                 phi0: float = 0.0):
        self.grid = grid
        self.area_proj = np.asarray(grid.area)
        self.inclination_rad = np.deg2rad(float(inclination_deg))
        self.phi0 = float(phi0)


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
        par = mf.readParamsTomog('inzdi.dat')

    # 2) 读观测
    if obsSet is None:
        file_type = getattr(par, 'obsFileType', 'auto')
        obsSet = readObs.obsProfSetInRange(list(par.fnames),
                                           par.velStart,
                                           par.velEnd,
                                           par.velRs,
                                           file_type=file_type)

    # 3) 线数据与线模型
    if lineData is None:
        line_file = getattr(par, 'lineParamFile', 'lines.txt')
        lineData = BasicLineData(line_file)
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
    nr = getattr(par, 'nRingsStellarGrid', 60)
    r_out = getattr(par, 'radius', 1.0)
    grid = diskGrid(nr=nr, r_in=0.0, r_out=r_out, verbose=0)
    geom = SimpleDiskGeometry(grid, inclination_deg=par.inclination)

    # 5) 每个观测相位计算合成谱
    results = []
    for i, obs in enumerate(obsSet):
        v_grid = np.asarray(obs.wl, dtype=float)  # LSD: 这里即速度网格
        inte = VelspaceDiskIntegrator(
            geom=geom,
            wl0_nm=lineData.wl0,
            v_grid=v_grid,
            line_model=line_model,
            inst_fwhm_kms=getattr(par, 'instrumentRes', 0.0),
            normalize_continuum=True,
            disk_v0_kms=getattr(par, 'velEq', getattr(par, 'vsini', 100.0)),
            disk_power_index=-0.5,
            disk_r0=1.0,
        )
        results.append((v_grid, inte.I, inte.V))

    # 6) 可选：保存一个简易输出以便快速查看
    with open('out_synth.txt', 'w') as f:
        for i, (v, I, V) in enumerate(results):
            f.write(f"# phase_index {i}\n")
            for k in range(v.size):
                f.write(f"{v[k]:.6f} {I[k]:.8e} {V[k]:.8e}\n")
            f.write("\n")

    return results


if __name__ == "__main__":
    main()
