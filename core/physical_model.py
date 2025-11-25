"""完整的物理模型集成模块 (physical_model.py)

整合核心物理与数值模块：
  - grid_tom.py: 盘面网格生成
  - disk_geometry.py: 盘几何与磁场参数容器
  - velspace_DiskIntegrator.py: 速度空间积分与谱线合成

该模块提供统一的物理模型初始化接口，接收 readParamsTomog 参数对象，
生成完整的物理模型容器，支持正演与反演工作流。

Classes
-------
PhysicalModelBuilder : 物理模型构建器，协调各子模块
PhysicalModel : 完整的物理模型容器

Functions
---------
create_physical_model : 便利函数，从 readParamsTomog 直接创建物理模型
"""

import numpy as np
from typing import Optional, Any, Dict
from dataclasses import dataclass, field

from core.grid_tom import diskGrid
from core.disk_geometry_integrator import SimpleDiskGeometry, VelspaceDiskIntegrator, create_disk_geometry_from_params
from core.mainFuncs import readParamsTomog

__all__ = [
    'PhysicalModel',
    'PhysicalModelBuilder',
    'create_physical_model',
]


@dataclass
class PhysicalModel:
    """完整的物理模型容器。
    
    包含盘面网格、几何参数、磁场配置、亮度分布以及速度空间积分器。
    该容器提供了执行正演或反演所需的所有物理模型组件。
    
    Attributes
    ----------
    par : readParamsTomog
        原始参数对象（保留用于追踪）
    grid : diskGrid
        盘面网格（等Δr分层）
    geometry : SimpleDiskGeometry
        盘几何容器（包含磁场参数和亮度分布）
    integrator : VelspaceDiskIntegrator or None
        速度空间积分器（延迟初始化）
    
    Parameters from par
    -------------------
    inclination_deg : float
        盘面倾角（度数）
    pOmega : float
        差速转动指数
    period : float
        自转周期（天）
    radius : float
        恒星半径 (R_sun)
    vsini : float
        视向赤道速度 (km/s)
    Vmax : float
        网格最大速度 (km/s)
    enable_stellar_occultation : int
        恒星遮挡标志
    nRingsStellarGrid : int
        网格环数
        
    Magnetic Field
    ---------------
    B_los : np.ndarray
        视向磁场分量 (Gauss)
    B_perp : np.ndarray
        垂直磁场强度 (Gauss)
    chi : np.ndarray
        磁场方向角 (rad)
    
    Spectral Amplitude (Response)
    ---------------------------
    amp : np.ndarray
        谱线振幅（响应权重）(0-1)，用于调制发射/吸收
    """

    par: readParamsTomog
    grid: diskGrid
    geometry: SimpleDiskGeometry
    integrator: Optional[VelspaceDiskIntegrator] = None

    # 缓存速度相关的参数
    _v_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    _wl0: Optional[float] = None
    _line_model: Optional[Any] = None

    def __post_init__(self):
        """验证物理模型的一致性。"""
        self.validate()

    def validate(self) -> bool:
        """验证物理模型参数的一致性和完整性。
        
        检查项：
        - grid 与 geometry 的像素数一致
        - 磁场参数维度与像素数一致
        - 亮度分布维度与像素数一致
        - 参数数值范围合理
        
        Returns
        -------
        bool
            验证通过返回 True，失败抛出 ValueError
        
        Raises
        ------
        ValueError
            当参数不一致或范围不合理时抛出
        """
        npix_grid = self.grid.numPoints
        npix_geom = self.geometry.grid.numPoints

        if npix_grid != npix_geom:
            raise ValueError(
                f"Grid pixel mismatch: grid.numPoints={npix_grid} vs "
                f"geometry.grid.numPoints={npix_geom}")

        # 磁场维度检查
        if len(self.geometry.B_los) != npix_grid:
            raise ValueError(f"B_los length ({len(self.geometry.B_los)}) != "
                             f"grid.numPoints ({npix_grid})")
        if len(self.geometry.B_perp) != npix_grid:
            raise ValueError(f"B_perp length ({len(self.geometry.B_perp)}) != "
                             f"grid.numPoints ({npix_grid})")
        if len(self.geometry.chi) != npix_grid:
            raise ValueError(f"chi length ({len(self.geometry.chi)}) != "
                             f"grid.numPoints ({npix_grid})")

        # 谱线振幅维度检查
        if len(self.geometry.amp) != npix_grid:
            raise ValueError(f"amp length ({len(self.geometry.amp)}) != "
                             f"grid.numPoints ({npix_grid})")

        # 参数范围检查
        if self.par.inclination < 0 or self.par.inclination > 90:
            raise ValueError(
                f"inclination out of range: {self.par.inclination}° "
                f"(must be in [0, 90])")

        if self.par.period <= 0:
            raise ValueError(f"period must be positive, got {self.par.period}")

        if self.par.radius <= 0:
            raise ValueError(
                f"stellar radius must be positive, got {self.par.radius}")

        if self.grid.r_out <= self.grid.r_in:
            raise ValueError(
                f"grid radial range invalid: r_in={self.grid.r_in}, "
                f"r_out={self.grid.r_out}")

        return True

    def get_summary(self) -> Dict[str, Any]:
        """获取物理模型的摘要信息。
        
        Returns
        -------
        dict
            包含模型关键参数的字典
        """
        return {
            'grid_npix': self.grid.numPoints,
            'grid_nr': self.grid.nr,
            'grid_r_range': (self.grid.r_in, self.grid.r_out),
            'inclination_deg': self.par.inclination,
            'period_day': self.par.period,
            'pOmega': self.par.pOmega,
            'vsini_kms': self.par.vsini,
            'Vmax_kms': self.par.Vmax,
            'stellar_radius_rsun': self.par.radius,
            'magnetic_field': self.geometry.get_magnetic_field_summary(),
        }

    def update_magnetic_field(self,
                              B_los: Optional[np.ndarray] = None,
                              B_perp: Optional[np.ndarray] = None,
                              chi: Optional[np.ndarray] = None) -> None:
        """更新磁场参数。
        
        Parameters
        ----------
        B_los : np.ndarray, optional
            视向磁场 (Gauss)
        B_perp : np.ndarray, optional
            垂直磁场强度 (Gauss)
        chi : np.ndarray, optional
            磁场方向角 (rad)
        
        Raises
        ------
        ValueError
            当磁场维度不匹配时抛出
        """
        npix = self.grid.numPoints

        if B_los is not None:
            B_los = np.asarray(B_los, dtype=float)
            if len(B_los) != npix:
                raise ValueError(
                    f"B_los length ({len(B_los)}) != grid.numPoints ({npix})")
            self.geometry.B_los = B_los

        if B_perp is not None:
            B_perp = np.asarray(B_perp, dtype=float)
            if len(B_perp) != npix:
                raise ValueError(
                    f"B_perp length ({len(B_perp)}) != grid.numPoints ({npix})"
                )
            self.geometry.B_perp = B_perp

        if chi is not None:
            chi = np.asarray(chi, dtype=float)
            if len(chi) != npix:
                raise ValueError(
                    f"chi length ({len(chi)}) != grid.numPoints ({npix})")
            self.geometry.chi = chi

    def update_amplitude(self, amp: np.ndarray) -> None:
        """更新谱线振幅（响应权重）。
        
        谱线振幅用于调制发射/吸收，值域为 [0, 1]。
        amp=1.0 表示最大发射，amp=0.0 表示无信号。
        
        Parameters
        ----------
        amp : np.ndarray
            谱线振幅 (归一化, 0-1)，长度应为 grid.numPoints
        
        Raises
        ------
        ValueError
            当维度不匹配时抛出
        """
        npix = self.grid.numPoints
        amp = np.asarray(amp, dtype=float)

        if len(amp) != npix:
            raise ValueError(
                f"amp length ({len(amp)}) != grid.numPoints ({npix})")

        # 裁剪到 [0, 1] 范围
        amp = np.clip(amp, 0.0, 1.0)
        self.geometry.amp = amp


class PhysicalModelBuilder:
    """物理模型构建器，协调各子模块的初始化。
    
    提供灵活的方式来构建物理模型，支持分步初始化和自定义参数。
    
    Parameters
    ----------
    par : readParamsTomog
        参数对象
    verbose : int, default=1
        详细程度 (0=安静, 1=正常, 2=详细)
    """

    def __init__(self, par: readParamsTomog, verbose: int = 1):
        """初始化构建器。"""
        if not isinstance(par, readParamsTomog):
            raise TypeError(
                f"par must be readParamsTomog instance, got {type(par)}")
        self.par = par
        self.verbose = int(verbose)
        self._grid = None
        self._geometry = None

    def build_grid(self,
                   nr: Optional[int] = None,
                   r_in: Optional[float] = None,
                   r_out: Optional[float] = None,
                   target_pixels_per_ring: Optional[Any] = None) -> diskGrid:
        """构建盘面网格。
        
        Parameters
        ----------
        nr : int, optional
            环数。如果未提供，将根据 r_out 自动调整以保持径向分辨率与 par 配置一致。
        r_in : float, optional
            内半径 (R_sun)，默认0.0
        r_out : float, optional
            外半径 (R_sun)，默认从 par.r_out
        target_pixels_per_ring : optional
            每环像素数配置，支持 int 或类数组
        
        Returns
        -------
        diskGrid
            创建的网格对象
        """
        # 1. 确定几何边界
        r_in_val = float(r_in) if r_in is not None else 0.0
        r_out_val = float(r_out) if r_out is not None else float(
            getattr(self.par, 'r_out', 5.0))

        # 2. 确定环数 nr
        if nr is not None:
            nr_val = int(nr)
        else:
            # 自动调整逻辑：保持径向分辨率
            par_nr = int(getattr(self.par, 'nRingsStellarGrid', 60))
            par_r_out = float(getattr(self.par, 'r_out', 5.0))

            # 如果 r_out 显著改变，则调整 nr 以保持 dr 不变
            # 假设 par 配置是基于 r_in=0 的（通常情况）
            if abs(r_out_val - par_r_out) > 1e-6 and par_r_out > 0:
                # base_dr = par_r_out / par_nr
                # new_nr = (r_out - r_in) / base_dr
                base_dr = par_r_out / max(par_nr, 1)
                nr_val = int(np.ceil((r_out_val - r_in_val) / base_dr))
                # 确保至少有 1 个环
                nr_val = max(1, nr_val)

                if self.verbose:
                    print(
                        f"[PhysicalModelBuilder] Auto-adjusted nr={nr_val} "
                        f"(base: nr={par_nr} @ r_out={par_r_out}) to maintain resolution"
                    )
            else:
                nr_val = par_nr

        if self.verbose:
            print(f"[PhysicalModelBuilder] Creating grid: nr={nr_val}, "
                  f"r ∈ [{r_in_val:.2f}, {r_out_val:.2f}] R_sun")

        self._grid = diskGrid(nr=nr_val,
                              r_in=r_in_val,
                              r_out=r_out_val,
                              target_pixels_per_ring=target_pixels_per_ring,
                              verbose=self.verbose)

        return self._grid

    def build_geometry(self,
                       grid: Optional[diskGrid] = None,
                       B_los: Optional[np.ndarray] = None,
                       B_perp: Optional[np.ndarray] = None,
                       chi: Optional[np.ndarray] = None,
                       amp: Optional[np.ndarray] = None) -> SimpleDiskGeometry:
        """构建盘几何容器。
        
        Parameters
        ----------
        grid : diskGrid, optional
            盘面网格（默认使用 build_grid 创建的网格）
        B_los : np.ndarray, optional
            视向磁场 (Gauss)
        B_perp : np.ndarray, optional
            垂直磁场强度 (Gauss)
        chi : np.ndarray, optional
            磁场方向角 (rad)
        amp : np.ndarray, optional
            谱线振幅分布（响应权重, >0），直接来自几何
        
        Returns
        -------
        SimpleDiskGeometry
            创建的几何容器
        """
        if grid is None:
            if self._grid is None:
                self.build_grid()
            grid = self._grid

        if self.verbose:
            print(
                "[PhysicalModelBuilder] Creating geometry from parameters...")

        self._geometry = create_disk_geometry_from_params(self.par,
                                                          grid,
                                                          B_los=B_los,
                                                          B_perp=B_perp,
                                                          chi=chi,
                                                          amp=amp,
                                                          verbose=self.verbose)

        return self._geometry

    def build_integrator(self,
                         geometry: Optional[SimpleDiskGeometry] = None,
                         wl0_nm: float = 656.3,
                         v_grid: Optional[np.ndarray] = None,
                         line_model: Optional[Any] = None,
                         **integrator_kwargs) -> VelspaceDiskIntegrator:
        """构建速度空间积分器。
        
        Parameters
        ----------
        geometry : SimpleDiskGeometry, optional
            盘几何容器（默认使用 build_geometry 创建的几何）
        wl0_nm : float, default=656.3
            谱线中心波长 (nm)
        v_grid : np.ndarray, optional
            速度网格 (km/s)
        line_model : optional
            谱线模型对象（必需参数）
        **integrator_kwargs
            其他传递给 VelspaceDiskIntegrator 的参数
        
        Returns
        -------
        VelspaceDiskIntegrator
            创建的积分器
        
        Raises
        ------
        ValueError
            当 line_model 为 None 时抛出
        """
        if geometry is None:
            if self._geometry is None:
                self.build_geometry()
            geometry = self._geometry

        if line_model is None:
            raise ValueError(
                "line_model is required for VelspaceDiskIntegrator")

        # 自动从 par 提取速度参数 (如果未在 kwargs 中提供)
        if 'disk_v0_kms' not in integrator_kwargs:
            # 计算赤道速度 v_eq = vsini / sin(i)
            vsini = float(getattr(self.par, 'vsini', 10.0))
            inc_deg = float(getattr(self.par, 'inclination', 60.0))
            inc_rad = np.deg2rad(inc_deg)
            # 避免除以零
            if abs(np.sin(inc_rad)) > 1e-6:
                v_eq = vsini / np.sin(inc_rad)
            else:
                v_eq = vsini  # Fallback for face-on (though vsini is 0 then)

            integrator_kwargs['disk_v0_kms'] = v_eq
            if self.verbose:
                print(
                    f"[PhysicalModelBuilder] Auto-set disk_v0_kms={v_eq:.2f} km/s "
                    f"(from vsini={vsini}, i={inc_deg}°)")

        if 'disk_power_index' not in integrator_kwargs:
            integrator_kwargs['disk_power_index'] = float(
                getattr(self.par, 'pOmega', 0))

        if 'disk_r0' not in integrator_kwargs:
            # 优先使用 r0_rot (如果存在于 par)，否则使用 radius
            # 注意：readParamsTomog 通常将 radius 存储为 par.radius
            integrator_kwargs['disk_r0'] = float(
                getattr(self.par, 'radius', 1.0))

        if 'normalize_continuum' not in integrator_kwargs:
            integrator_kwargs['normalize_continuum'] = bool(
                getattr(self.par, 'normalize_continuum', True))

        # 生成默认速度网格（如果未提供）
        if v_grid is None:
            Vmax = float(self.par.Vmax)
            dv = 1.0  # km/s per pixel (default)
            n_vel = int(2 * Vmax / dv) + 1
            v_grid = np.linspace(-Vmax, Vmax, n_vel)
            if self.verbose:
                print(f"[PhysicalModelBuilder] Generated velocity grid: "
                      f"v ∈ [{-Vmax:.1f}, {Vmax:.1f}] km/s, "
                      f"N_v={len(v_grid)}, dv={dv:.1f}")

        if self.verbose:
            print(f"[PhysicalModelBuilder] Creating integrator: "
                  f"wl0={wl0_nm:.1f} nm")

        integrator = VelspaceDiskIntegrator(geom=geometry,
                                            wl0_nm=wl0_nm,
                                            v_grid=v_grid,
                                            line_model=line_model,
                                            **integrator_kwargs)

        return integrator

    def build(self,
              wl0_nm: float = 656.3,
              v_grid: Optional[np.ndarray] = None,
              line_model: Optional[Any] = None,
              B_los: Optional[np.ndarray] = None,
              B_perp: Optional[np.ndarray] = None,
              chi: Optional[np.ndarray] = None,
              amp: Optional[np.ndarray] = None,
              **grid_kwargs) -> PhysicalModel:
        """完整构建物理模型（一站式接口）。
        
        Parameters
        ----------
        wl0_nm : float, default=656.3
            谱线中心波长 (nm)
        v_grid : np.ndarray, optional
            速度网格 (km/s)
        line_model : optional
            谱线模型对象（必需参数）
        B_los : np.ndarray, optional
            视向磁场 (Gauss)
        B_perp : np.ndarray, optional
            垂直磁场强度 (Gauss)
        chi : np.ndarray, optional
            磁场方向角 (rad)
        amp : np.ndarray, optional
            谱线振幅分布（响应权重, >0），直接来自几何
        **grid_kwargs
            传递给 build_grid 的关键字参数
        
        Returns
        -------
        PhysicalModel
            完整的物理模型对象
        """
        # 分步构建
        self.build_grid(**grid_kwargs)
        self.build_geometry(B_los=B_los, B_perp=B_perp, chi=chi, amp=amp)
        integrator = self.build_integrator(wl0_nm=wl0_nm,
                                           v_grid=v_grid,
                                           line_model=line_model)

        # 组装物理模型
        model = PhysicalModel(
            par=self.par,
            grid=self._grid,
            geometry=self._geometry,
            integrator=integrator,
            _v_grid=v_grid if v_grid is not None else np.linspace(
                -self.par.Vmax, self.par.Vmax,
                int(2 * self.par.Vmax) + 1),
            _wl0=float(wl0_nm),
            _line_model=line_model,
        )

        if self.verbose:
            print(f"[PhysicalModelBuilder] Physical model built successfully")
            print(f"  {model.geometry}")
            summary = model.get_summary()
            print(f"  Summary: {summary['grid_npix']} pixels, "
                  f"r ∈ [{summary['grid_r_range'][0]:.1f}, "
                  f"{summary['grid_r_range'][1]:.1f}] R_sun, "
                  f"inclination={summary['inclination_deg']:.0f}°")

        return model


def create_physical_model(par: readParamsTomog,
                          wl0_nm: float = 656.3,
                          v_grid: Optional[np.ndarray] = None,
                          line_model: Optional[Any] = None,
                          B_los: Optional[np.ndarray] = None,
                          B_perp: Optional[np.ndarray] = None,
                          chi: Optional[np.ndarray] = None,
                          amp: Optional[np.ndarray] = None,
                          verbose: int = 1,
                          **grid_kwargs) -> PhysicalModel:
    """便利函数：直接从参数对象创建完整的物理模型。
    
    Parameters
    ----------
    par : readParamsTomog
        参数对象
    wl0_nm : float, default=656.3
        谱线中心波长 (nm)
    v_grid : np.ndarray, optional
        速度网格 (km/s)
    line_model : optional
        谱线模型对象（必需参数）
    B_los : np.ndarray, optional
        视向磁场 (Gauss)
    B_perp : np.ndarray, optional
        垂直磁场强度 (Gauss)
    chi : np.ndarray, optional
        磁场方向角 (rad)
    amp : np.ndarray, optional
        谱线振幅分布（响应权重, >0），直接来自几何
    verbose : int, default=1
        详细程度
    **grid_kwargs
        传递给网格构建的关键字参数
    
    Returns
    -------
    PhysicalModel
        完整的物理模型对象
    
    Examples
    --------
    >>> from core.mainFuncs import readParamsTomog
    >>> from core.local_linemodel_basic import GaussianZeemanWeakLineModel
    >>> from core.physical_model import create_physical_model
    >>> 
    >>> # 读取参数
    >>> par = readParamsTomog('input/params_tomog.txt')
    >>> 
    >>> # 创建谱线模型
    >>> line_model = GaussianZeemanWeakLineModel()
    >>> 
    >>> # 创建物理模型
    >>> phys_model = create_physical_model(
    ...     par,
    ...     wl0_nm=656.3,
    ...     line_model=line_model,
    ...     verbose=1
    ... )
    >>> 
    >>> # 检查模型
    >>> print(phys_model.get_summary())
    >>> phys_model.validate()
    """
    # -------------------------------------------------------------------------
    # Handle initialization from .tomog file if enabled
    # -------------------------------------------------------------------------
    if getattr(par, 'initTomogFile', 0) == 1 and getattr(
            par, 'initModelPath', None):
        model_path = par.initModelPath
        if verbose:
            print(
                f"[create_physical_model] Loading initial model from: {model_path}"
            )

        try:
            # Load the model
            geom_loaded, meta, table = VelspaceDiskIntegrator.read_geomodel(
                model_path)

            # 1. Override parameters in par
            if 'inclination_deg' in meta:
                par.inclination = float(meta['inclination_deg'])
            if 'pOmega' in meta:
                par.pOmega = float(meta['pOmega'])
            if 'period' in meta:
                par.period = float(meta['period'])
            if 'r0_rot' in meta:
                par.radius = float(
                    meta['r0_rot'])  # Assuming r0_rot corresponds to radius
            if 'nr' in meta:
                par.nRingsStellarGrid = int(meta['nr'])

            # Update grid extent from loaded model
            if hasattr(geom_loaded.grid, 'r'):
                # Try to get r_in and r_out from r_edges if available
                r_min_val = np.min(geom_loaded.grid.r)
                r_max_val = np.max(geom_loaded.grid.r)

                if hasattr(geom_loaded.grid, 'r_edges') and len(
                        geom_loaded.grid.r_edges) > 0:
                    r_min_val = np.min(geom_loaded.grid.r_edges)
                    r_max_val = np.max(geom_loaded.grid.r_edges)

                par.r_out = float(r_max_val)

                # Update r_in in grid_kwargs
                grid_kwargs['r_in'] = float(r_min_val)

                if verbose:
                    print(
                        f"  ✓ Updated grid extent: r_in={r_min_val:.2f}, r_out={par.r_out:.2f}"
                    )

            if verbose:
                print(
                    f"  ✓ Parameters updated from model file (inc={par.inclination}°, nr={par.nRingsStellarGrid})"
                )

            # 2. Extract physical quantities (if not explicitly provided)
            if B_los is None and hasattr(geom_loaded, 'B_los'):
                B_los = geom_loaded.B_los
                if verbose:
                    print(f"  ✓ Loaded B_los ({len(B_los)} pixels)")

            if B_perp is None and hasattr(geom_loaded, 'B_perp'):
                B_perp = geom_loaded.B_perp
                if verbose:
                    print(f"  ✓ Loaded B_perp ({len(B_perp)} pixels)")

            if chi is None and hasattr(geom_loaded, 'chi'):
                chi = geom_loaded.chi
                if verbose:
                    print(f"  ✓ Loaded chi ({len(chi)} pixels)")

            if amp is None and hasattr(geom_loaded, 'amp'):
                amp = geom_loaded.amp
                if verbose:
                    print(f"  ✓ Loaded amp ({len(amp)} pixels)")

        except Exception as e:
            print(
                f"[create_physical_model] ⚠️  Error loading model from {model_path}: {e}"
            )
            print("  Continuing with default initialization...")

    builder = PhysicalModelBuilder(par, verbose=verbose)
    return builder.build(wl0_nm=wl0_nm,
                         v_grid=v_grid,
                         line_model=line_model,
                         B_los=B_los,
                         B_perp=B_perp,
                         chi=chi,
                         amp=amp,
                         **grid_kwargs)
