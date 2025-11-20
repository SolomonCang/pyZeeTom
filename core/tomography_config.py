"""
tomography_config.py - 正演和反演工作流的配置对象

本模块提供统一的配置容器，封装正演和反演工作流所需的所有参数。
使用配置对象而非字典，提供：
  - 类型安全和IDE自动补全
  - 内置验证逻辑
  - 清晰的文档和类型注解
  - 便利的序列化/反序列化
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
import numpy as np
from datetime import datetime

from core.local_linemodel_basic import LineData as BasicLineData
from core.disk_geometry_integrator import SimpleDiskGeometry
from core.grid_tom import diskGrid
from core.local_linemodel_basic import GaussianZeemanWeakLineModel, ConstantAmpLineModel


@dataclass
class ForwardModelConfig:
    """正演模型配置容器
    
    封装正演工作流（run_forward_synthesis）所需的所有参数，包括：
    - 观测数据集和参数对象
    - 几何与物理模型
    - 动力学参数
    - 磁场初始条件
    - 输出控制选项
    """

    # ============ 核心输入 ============

    par: Any  # readParamsTomog 返回的参数对象
    """参数对象，包含所有配置信息"""

    obsSet: List[Any]
    """观测数据集列表，每个元素为 ObservationProfile 对象"""

    lineData: Optional[BasicLineData] = None
    """谱线参数数据"""

    # ============ 物理模型 ============

    geom: Optional[SimpleDiskGeometry] = None
    """盘面几何对象，包含网格和动力学参数"""

    line_model: Optional[Any] = None
    """谱线模型对象，提供 compute_local_profile() 方法"""

    # ============ 动力学参数 ============

    velEq: float = 100.0
    """赤道速度，单位 km/s"""

    pOmega: float = 0.0
    """差速转动指数"""

    radius: float = 1.0
    """参考半径，单位 R_sun"""

    # ============ 速度场参数（外区域） ============

    disk_v0_kms: float = 200.0
    """速度场参考半径处的速度，单位 km/s"""

    disk_power_index: float = -0.5
    """外区域幂律指数"""

    disk_r0: float = 1.0
    """速度场参考半径"""

    # ============ 内层减速参数 ============

    inner_slowdown_mode: str = "adaptive"
    """内层减速模式：'adaptive' 或 'continuous'"""

    inner_profile: str = "cosine"
    """内层减速剖面类型：'cosine', 'sine' 等"""

    inner_edge_blend: bool = True
    """是否启用内外区域边界混合"""

    inner_mode: str = "poly"
    """连续模式的内层模式：'poly' 或 'lat'"""

    inner_alpha: float = 0.6
    """内层减速参数 α"""

    inner_beta: float = 2.0
    """内层减速参数 β（指数）"""

    # ============ 谱线模型参数 ============

    enable_v: bool = True
    """是否启用 Stokes V 计算"""

    enable_qu: bool = True
    """是否启用 Stokes Q/U 计算"""

    line_area: float = 1.0
    """谱线面积因子"""

    normalize_continuum: bool = True
    """是否归一化连续谱"""

    amp_init: Optional[np.ndarray] = None
    """初始幅度分布"""

    B_los_init: Optional[np.ndarray] = None
    """初始视向磁场"""

    B_perp_init: Optional[np.ndarray] = None
    """初始垂直磁场"""

    chi_init: Optional[np.ndarray] = None
    """初始方位角"""

    # ============ 输出控制 ============

    output_dir: str = "./output"
    """输出目录"""

    save_intermediate: bool = False
    """是否保存中间结果"""

    verbose: int = 0
    """详细程度：0=静默，1=正常，2=详细"""

    # ============ 内部状态 ============

    _validated: bool = field(default=False, init=False, repr=False)
    """配置是否已验证"""

    _creation_time: str = field(
        default_factory=lambda: datetime.now().isoformat(),
        init=False,
        repr=False)
    """创建时间"""

    def validate(self) -> None:
        """验证配置完整性和一致性
        
        Raises
        ------
        ValueError
            当任何必需参数为 None 或无效时
        AssertionError
            当参数维度不匹配时
        """
        if self.par is None:
            raise ValueError("参数对象 (par) 不能为 None")

        if not self.obsSet or len(self.obsSet) == 0:
            raise ValueError("观测数据集 (obsSet) 不能为空")

        if self.lineData is None:
            raise ValueError("谱线参数 (lineData) 不能为 None")

        if self.geom is None:
            raise ValueError("几何对象 (geom) 不能为 None")

        if self.line_model is None:
            raise ValueError("谱线模型 (line_model) 不能为 None")

        # 检查动力学参数
        if self.velEq <= 0:
            raise ValueError(f"赤道速度 (velEq) 必须为正数，得到 {self.velEq}")

        if self.radius <= 0:
            raise ValueError(f"参考半径 (radius) 必须为正数，得到 {self.radius}")

        # 检查振幅和磁场数组维度
        npix = self.geom.grid.numPoints

        if self.amp_init is not None:
            if len(self.amp_init) != npix:
                raise AssertionError(
                    f"amp_init 维度 ({len(self.amp_init)}) 与网格像素数 ({npix}) 不匹配")
        else:
            # 创建默认全1数组（单位振幅）
            self.amp_init = np.ones(npix)

        if self.B_los_init is not None:
            if len(self.B_los_init) != npix:
                raise AssertionError(
                    f"B_los_init 维度 ({len(self.B_los_init)}) 与网格像素数 ({npix}) 不匹配"
                )
        else:
            # 创建默认全零数组
            self.B_los_init = np.zeros(npix)

        if self.B_perp_init is not None:
            if len(self.B_perp_init) != npix:
                raise AssertionError(
                    f"B_perp_init 维度 ({len(self.B_perp_init)}) 与网格像素数 ({npix}) 不匹配"
                )
        else:
            self.B_perp_init = np.zeros(npix)

        if self.chi_init is not None:
            if len(self.chi_init) != npix:
                raise AssertionError(
                    f"chi_init 维度 ({len(self.chi_init)}) 与网格像素数 ({npix}) 不匹配")
        else:
            self.chi_init = np.zeros(npix)

        # 检查输出目录
        from pathlib import Path
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._validated = True

    def to_dict(self) -> Dict[str, Any]:
        """序列化配置为字典
        
        Returns
        -------
        dict
            配置字典，numpy数组转换为列表，对象转换为字符串表示
        """
        self.validate()

        result = {
            'par': str(type(self.par).__name__),
            'obsSet_count': len(self.obsSet),
            'lineData_wl0':
            float(self.lineData.wl0) if self.lineData else None,
            'geom_npix': self.geom.grid.numPoints,
            'line_model': str(type(self.line_model).__name__),
            'velEq': float(self.velEq),
            'pOmega': float(self.pOmega),
            'radius': float(self.radius),
            'B_los_init_stats': {
                'min': float(self.B_los_init.min()),
                'max': float(self.B_los_init.max()),
                'mean': float(self.B_los_init.mean()),
            } if self.B_los_init is not None else None,
            'B_perp_init_stats': {
                'min': float(self.B_perp_init.min()),
                'max': float(self.B_perp_init.max()),
                'mean': float(self.B_perp_init.mean()),
            } if self.B_perp_init is not None else None,
            'output_dir': self.output_dir,
            'save_intermediate': self.save_intermediate,
            'verbose': self.verbose,
            'creation_time': self._creation_time,
        }

        return result

    @classmethod
    def from_par(cls,
                 par,
                 obsSet,
                 lineData,
                 verbose=False) -> 'ForwardModelConfig':
        """从旧参数对象创建正演配置
        
        将 readParamsTomog 返回的参数对象、观测集合和谱线数据转换为
        ForwardModelConfig 对象，便于与新工作流集成。
        
        Parameters
        ----------
        par : readParamsTomog
            来自 mf.readParamsTomog() 的参数对象
        obsSet : list
            来自 SpecIO.obsProfSetInRange() 的观测集合
        lineData : LineData
            来自 BasicLineData() 的谱线参数
        verbose : bool
            详细输出
            
        Returns
        -------
        ForwardModelConfig
            新的正演配置对象
            
        Raises
        ------
        ValueError
            当必需参数缺失或无效时
        """

        if verbose:
            print("[ForwardModelConfig.from_par] 开始参数转换...")

        # 提取动力学参数
        velEq = float(getattr(par, 'velEq', getattr(par, 'vsini', 100.0)))
        pOmega = float(getattr(par, 'pOmega', 0.0))
        radius = float(getattr(par, 'radius', 1.0))
        period = float(getattr(par, 'period', 1.0))
        inclination_deg = float(getattr(par, 'inclination', 90.0))

        # 计算网格外半径
        nr = int(getattr(par, 'nRingsStellarGrid', 60))
        Vmax = float(getattr(par, 'Vmax', 0.0))

        if abs(pOmega + 1.0) > 1e-6:
            # 一般情况：从 Vmax 反推 r_out
            r_out_stellar = (Vmax /
                             velEq)**(1.0 /
                                      (pOmega + 1.0)) if Vmax > 0 else 5.0
            r_out_grid = radius * r_out_stellar
        else:
            # 特殊情况：pOmega = -1（恒定角动量）
            r_out_stellar = float(getattr(par, 'r_out', 5.0))
            r_out_grid = radius * r_out_stellar

        if verbose:
            print(f"[from_par] 网格: nr={nr}, r_out={r_out_grid:.3f} R_sun")

        # 构建网格
        grid = diskGrid(nr=nr, r_in=0.0, r_out=r_out_grid, verbose=verbose)

        # 构建几何
        enable_occultation = bool(getattr(par, 'enable_stellar_occultation',
                                          0))
        geom = SimpleDiskGeometry(
            grid=grid,
            inclination_deg=inclination_deg,
            pOmega=pOmega,
            r0=radius,
            period=period,
            enable_stellar_occultation=enable_occultation,
            stellar_radius=radius,
            B_los=np.zeros(grid.numPoints),
            B_perp=np.zeros(grid.numPoints),
            chi=np.zeros(grid.numPoints),
            amp=np.ones(grid.numPoints))

        if verbose:
            print(f"[from_par] 几何: 倾角={inclination_deg}°, 周期={period}d")

        # 构建线模型
        k_qu = float(getattr(par, 'lineKQU', 1.0))
        enable_v = bool(getattr(par, 'lineEnableV', 1))
        enable_qu = bool(getattr(par, 'lineEnableQU', 1))
        amp_const = float(getattr(par, 'lineAmpConst', -0.5))

        base_model = GaussianZeemanWeakLineModel(lineData,
                                                 k_QU=k_qu,
                                                 enable_V=enable_v,
                                                 enable_QU=enable_qu)
        line_model = ConstantAmpLineModel(base_model, amp=amp_const)

        if verbose:
            print(
                f"[from_par] 线模型: amp={amp_const}, k_QU={k_qu}, V={enable_v}, QU={enable_qu}"
            )

        # 计算仪器FWHM
        if hasattr(par, 'compute_instrument_fwhm'):
            par.compute_instrument_fwhm(lineData.wl0, verbose=False)
        instrument_res = float(getattr(par, 'instrumentFWHM', 0.1))

        # 提取速度场参数
        disk_v0_kms = float(getattr(par, 'disk_v0_kms', velEq))
        disk_power_index = float(getattr(par, 'disk_power_index', pOmega))
        disk_r0 = float(getattr(par, 'disk_r0', radius))
        inner_slowdown_mode = str(
            getattr(par, 'inner_slowdown_mode', 'adaptive'))
        inner_profile = str(getattr(par, 'inner_profile', 'cosine'))
        inner_edge_blend = bool(getattr(par, 'inner_edge_blend', True))
        inner_mode = str(getattr(par, 'inner_mode', 'poly'))
        inner_alpha = float(getattr(par, 'inner_alpha', 0.6))
        inner_beta = float(getattr(par, 'inner_beta', 2.0))

        if verbose:
            print(f"[from_par] 速度场: v0={disk_v0_kms:.1f} km/s, "
                  f"power={disk_power_index:.2f}, r0={disk_r0:.2f}")
            print(f"[from_par] 内层减速: mode={inner_slowdown_mode}, "
                  f"profile={inner_profile}, alpha={inner_alpha:.2f}")

        # 创建配置对象
        config = cls(par=par,
                     obsSet=obsSet,
                     lineData=lineData,
                     geom=geom,
                     line_model=line_model,
                     velEq=velEq,
                     pOmega=pOmega,
                     radius=radius,
                     disk_v0_kms=disk_v0_kms,
                     disk_power_index=disk_power_index,
                     disk_r0=disk_r0,
                     inner_slowdown_mode=inner_slowdown_mode,
                     inner_profile=inner_profile,
                     inner_edge_blend=inner_edge_blend,
                     inner_mode=inner_mode,
                     inner_alpha=inner_alpha,
                     inner_beta=inner_beta,
                     inst_fwhm_kms=instrument_res,
                     enable_v=enable_v,
                     enable_qu=enable_qu,
                     amp_init=np.ones(grid.numPoints),
                     B_los_init=np.zeros(grid.numPoints),
                     B_perp_init=np.zeros(grid.numPoints),
                     chi_init=np.zeros(grid.numPoints),
                     output_dir=str(getattr(par, 'outputDir', './output')),
                     save_intermediate=bool(
                         getattr(par, 'saveIntermediate', False)),
                     verbose=1 if verbose else 0)

        # 存储额外属性便于后续使用
        config._instrument_res = instrument_res
        config._nr = nr
        config._r_out_grid = r_out_grid
        config._phases = np.array(
            [float(getattr(obs, 'phase', 0.0)) for obs in obsSet])

        if verbose:
            print(f"[from_par] 转换完成：{len(obsSet)} 个观测相位")

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ForwardModelConfig':
        """从字典反序列化配置
        
        注意：本方法主要用于调试和日志记录。
        实际重建配置需要从原始数据源（参数文件、观测数据等）创建。
        """
        raise NotImplementedError("from_dict() 需要原始数据源。请使用参数文件和观测数据重新构建配置。")

    def create_summary(self) -> str:
        """生成配置摘要字符串
        
        Returns
        -------
        str
            格式化的配置摘要，适合日志输出
        """
        self.validate()

        lines = [
            "=" * 70,
            "正演配置摘要 (ForwardModelConfig)",
            "=" * 70,
            f"观测数据集: {len(self.obsSet)} 个相位",
            f"谱线参数: λ₀ = {self.lineData.wl0:.4f} nm",
            f"网格像素数: {self.geom.grid.numPoints}",
            f"动力学参数:",
            f"  - 赤道速度: {self.velEq:.1f} km/s",
            f"  - 差速指数: {self.pOmega:.3f}",
            f"  - 参考半径: {self.radius:.3f} R_sun",
            f"几何参数:",
            f"  - 倾角: {self.geom.inclination_rad * 180 / np.pi:.1f}°",
            f"  - 周期: {self.geom.period:.4f} d",
            f"磁场初值:",
            f"  - B_los: [{self.B_los_init.min():.1f}, {self.B_los_init.max():.1f}] G",
            f"  - B_perp: [{self.B_perp_init.min():.1f}, {self.B_perp_init.max():.1f}] G",
            f"  - χ: [{self.chi_init.min():.3f}, {self.chi_init.max():.3f}] rad",
            f"输出设置:",
            f"  - 目录: {self.output_dir}",
            f"  - 保存中间结果: {self.save_intermediate}",
            f"  - 详细程度: {self.verbose}",
            "=" * 70,
        ]

        return "\n".join(lines)


@dataclass
class InversionConfig(ForwardModelConfig):
    """MEM反演配置容器
    
    继承 ForwardModelConfig，添加反演特定的参数和控制选项。
    """

    # ============ 反演迭代参数 ============

    num_iterations: int = 10
    """最大迭代次数"""

    entropy_weight: float = 1.0
    """熵正则化权重 (λ_S)"""

    data_weight: float = 1.0
    """数据拟合权重 (λ_D)"""

    smoothness_weight: float = 0.1
    """平滑性正则化权重 (λ_R)"""

    # ============ 收敛标准 ============

    convergence_threshold: float = 1e-3
    """收敛判定阈值"""

    max_iterations: Optional[int] = None
    """最大迭代次数（若为 None，则使用 num_iterations）"""

    # ============ 初始模型 ============

    initial_B_los: Optional[np.ndarray] = None
    """初始 B_los 模型（若为 None，使用 B_los_init）"""

    initial_B_perp: Optional[np.ndarray] = None
    """初始 B_perp 模型（若为 None，使用 B_perp_init）"""

    initial_chi: Optional[np.ndarray] = None
    """初始 χ 模型（若为 None，使用 chi_init）"""

    # ============ 保存策略 ============

    save_every_iter: int = 1
    """每隔多少迭代保存一次检查点"""

    save_final_only: bool = False
    """是否仅保存最终结果"""

    def validate(self) -> None:
        """验证反演配置，包括父类验证
        
        Raises
        ------
        ValueError
            当反演特定参数无效时
        """
        # 先调用父类验证
        super().validate()

        # 验证反演特定参数
        if self.num_iterations < 0:
            raise ValueError(f"迭代次数必须非负，得到 {self.num_iterations}")

        if not 0 < self.entropy_weight <= 10.0:
            raise ValueError(f"熵权重应在 (0, 10] 范围内，得到 {self.entropy_weight}")

        if not 0 < self.data_weight <= 10.0:
            raise ValueError(f"数据权重应在 (0, 10] 范围内，得到 {self.data_weight}")

        if not 0 <= self.smoothness_weight <= 10.0:
            raise ValueError(f"平滑权重应在 [0, 10] 范围内，得到 {self.smoothness_weight}")

        if self.convergence_threshold <= 0:
            raise ValueError(f"收敛阈值必须为正数，得到 {self.convergence_threshold}")

        if self.save_every_iter < 1:
            raise ValueError(f"保存间隔必须 ≥ 1，得到 {self.save_every_iter}")

        # 若指定了初始模型，检查维度
        npix = self.geom.grid.numPoints
        if self.initial_B_los is not None and len(self.initial_B_los) != npix:
            raise AssertionError(
                f"initial_B_los 维度 ({len(self.initial_B_los)}) 与网格不匹配 ({npix})"
            )

        if self.initial_B_perp is not None and len(
                self.initial_B_perp) != npix:
            raise AssertionError(
                f"initial_B_perp 维度 ({len(self.initial_B_perp)}) 与网格不匹配 ({npix})"
            )

        if self.initial_chi is not None and len(self.initial_chi) != npix:
            raise AssertionError(
                f"initial_chi 维度 ({len(self.initial_chi)}) 与网格不匹配 ({npix})")

    def get_mem_adapter_config(self) -> Dict[str, Any]:
        """获取 MEMTomographyAdapter 的配置字典
        
        Returns
        -------
        dict
            MEMTomographyAdapter 所需的配置参数
        """
        self.validate()

        npix = self.geom.grid.numPoints

        return {
            'fit_brightness':
            True,
            'fit_magnetic':
            True,
            'entropy_weights_blos':
            self.geom.grid.area
            if hasattr(self.geom.grid, 'area') else np.ones(npix),
            'entropy_weights_bperp':
            self.geom.grid.area
            if hasattr(self.geom.grid, 'area') else np.ones(npix),
            'entropy_weights_chi':
            (self.geom.grid.area *
             0.1) if hasattr(self.geom.grid, 'area') else np.ones(npix) * 0.1,
            'default_blos':
            10.0,
            'default_bperp':
            100.0,
            'default_chi':
            0.0,
        }

    @classmethod
    def from_par(cls,
                 par,
                 obsSet,
                 lineData,
                 verbose=False) -> 'InversionConfig':
        """从旧参数对象创建反演配置
        
        类似 ForwardModelConfig.from_par()，但添加反演特定参数。
        
        Parameters
        ----------
        par : readParamsTomog
            来自 mf.readParamsTomog() 的参数对象
        obsSet : list
            来自 SpecIO.obsProfSetInRange() 的观测集合
        lineData : LineData
            来自 BasicLineData() 的谱线参数
        verbose : bool
            详细输出
            
        Returns
        -------
        InversionConfig
            新的反演配置对象
            
        Raises
        ------
        ValueError
            当必需参数缺失或无效时
        """
        if verbose:
            print("[InversionConfig.from_par] 开始反演参数转换...")

        # 首先使用父类方法创建基础正演配置
        forward_config = ForwardModelConfig.from_par(par,
                                                     obsSet,
                                                     lineData,
                                                     verbose=verbose)

        # 提取反演特定参数
        num_iterations = int(getattr(par, 'numIterations', 10))
        entropy_weight = float(getattr(par, 'entropyWeight', 1.0))
        data_weight = float(getattr(par, 'dataWeight', 1.0))
        smoothness_weight = float(getattr(par, 'smoothnessWeight', 0.1))
        convergence_threshold = float(
            getattr(par, 'convergenceThreshold', 1e-3))

        if verbose:
            print(
                f"[from_par] 反演参数: iterations={num_iterations}, "
                f"entropy={entropy_weight}, data={data_weight}, smooth={smoothness_weight}"
            )

        # 创建反演配置（复制所有父类字段）
        config = cls(
            par=forward_config.par,
            obsSet=forward_config.obsSet,
            lineData=forward_config.lineData,
            geom=forward_config.geom,
            line_model=forward_config.line_model,
            velEq=forward_config.velEq,
            pOmega=forward_config.pOmega,
            radius=forward_config.radius,
            # Velocity field parameters
            disk_v0_kms=forward_config.disk_v0_kms,
            disk_power_index=forward_config.disk_power_index,
            disk_r0=forward_config.disk_r0,
            # Inner slowdown parameters
            inner_slowdown_mode=forward_config.inner_slowdown_mode,
            inner_profile=forward_config.inner_profile,
            inner_edge_blend=forward_config.inner_edge_blend,
            inner_mode=forward_config.inner_mode,
            inner_alpha=forward_config.inner_alpha,
            inner_beta=forward_config.inner_beta,
            # Instrument parameters
            inst_fwhm_kms=forward_config.inst_fwhm_kms,
            line_area=forward_config.line_area,
            normalize_continuum=forward_config.normalize_continuum,
            enable_v=forward_config.enable_v,
            enable_qu=forward_config.enable_qu,
            # Initial state
            amp_init=forward_config.amp_init.copy()
            if forward_config.amp_init is not None else None,
            B_los_init=forward_config.B_los_init.copy()
            if forward_config.B_los_init is not None else None,
            B_perp_init=forward_config.B_perp_init.copy()
            if forward_config.B_perp_init is not None else None,
            chi_init=forward_config.chi_init.copy()
            if forward_config.chi_init is not None else None,
            output_dir=forward_config.output_dir,
            save_intermediate=forward_config.save_intermediate,
            verbose=forward_config.verbose,
            # Inversion-specific parameters
            num_iterations=num_iterations,
            entropy_weight=entropy_weight,
            data_weight=data_weight,
            smoothness_weight=smoothness_weight,
            convergence_threshold=convergence_threshold)

        if verbose:
            print("[from_par] 反演配置转换完成")

        return config

    def create_summary(self) -> str:
        """生成反演配置摘要
        
        Returns
        -------
        str
            格式化的反演配置摘要
        """
        parent_summary = super().create_summary()

        lines = [
            "",
            "反演参数扩展:",
            f"  - 最大迭代数: {self.num_iterations}",
            f"  - 收敛阈值: {self.convergence_threshold:.3e}",
            f"  - 熵权重: {self.entropy_weight}",
            f"  - 数据权重: {self.data_weight}",
            f"  - 平滑权重: {self.smoothness_weight}",
            f"  - 保存间隔: {self.save_every_iter}",
            f"  - 仅保存最终: {self.save_final_only}",
        ]

        # 如果指定了初始模型，添加信息
        if self.initial_B_los is not None or self.initial_B_perp is not None or self.initial_chi is not None:
            lines.append("  - 使用自定义初始模型")

        return parent_summary + "\n" + "\n".join(lines)
