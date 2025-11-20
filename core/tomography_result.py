"""
tomography_result.py - 正演和反演工作流的结果对象

本模块提供结构化的结果容器，取代函数返回的元组。
提供：
  - 类型安全的数据访问
  - 内置的统计和诊断方法
  - 方便的序列化/导出功能
  - 清晰的文档和 IDE 自动补全
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime


@dataclass
class ForwardModelResult:
    """正演模型结果容器
    
    存储单相位正演工作流的输出结果。
    
    Attributes
    ----------
    stokes_i : np.ndarray
        Stokes I 分量光谱 (形状: Nλ)
    stokes_v : np.ndarray
        Stokes V 分量光谱 (形状: Nλ)
    stokes_q : Optional[np.ndarray]
        Stokes Q 分量光谱 (形状: Nλ)，可选
    stokes_u : Optional[np.ndarray]
        Stokes U 分量光谱 (形状: Nλ)，可选
    wavelength : np.ndarray
        波长网格 (形状: Nλ)
    error : Optional[np.ndarray]
        观测误差 (形状: Nλ)
    """

    # ============ 核心结果 ============

    stokes_i: np.ndarray
    """Stokes I 光谱数组 (Nλ,)"""

    stokes_v: np.ndarray
    """Stokes V 光谱数组 (Nλ,)"""

    stokes_q: Optional[np.ndarray] = None
    """Stokes Q 光谱数组 (Nλ,)，可选"""

    stokes_u: Optional[np.ndarray] = None
    """Stokes U 光谱数组 (Nλ,)，可选"""

    wavelength: np.ndarray = field(default_factory=lambda: np.array([]))
    """波长网格 (Nλ,)"""

    error: Optional[np.ndarray] = None
    """观测误差 (Nλ,)"""

    # ============ 元数据 ============

    hjd: Optional[float] = None
    """儒略日 (HJD)"""

    phase_index: int = 0
    """观测相位索引"""

    pol_channel: str = "V"
    """偏振通道标签 (仅支持: 'I', 'V', 'Q', 'U')"""

    model_name: str = "forward_synthesis"
    """模型名称"""

    # ============ 内部状态 ============

    integrator: Any = field(default=None, repr=False, compare=False)
    """关联的积分器对象 (VelspaceDiskIntegrator)"""

    _creation_time: str = field(
        default_factory=lambda: datetime.now().isoformat(),
        init=False,
        repr=False)
    """创建时间"""

    def validate(self) -> None:
        """验证结果数据完整性和一致性
        
        Raises
        ------
        ValueError
            数据不一致或不完整
        """
        if self.stokes_i is None or len(self.stokes_i) == 0:
            raise ValueError("Stokes I 不能为空")

        if self.stokes_v is None or len(self.stokes_v) == 0:
            raise ValueError("Stokes V 不能为空")

        nl = len(self.stokes_i)

        if len(self.stokes_v) != nl:
            raise ValueError(
                f"Stokes V 长度 ({len(self.stokes_v)}) 与 I 不匹配 ({nl})")

        if self.stokes_q is not None and len(self.stokes_q) != nl:
            raise ValueError(
                f"Stokes Q 长度 ({len(self.stokes_q)}) 与 I 不匹配 ({nl})")

        if self.stokes_u is not None and len(self.stokes_u) != nl:
            raise ValueError(
                f"Stokes U 长度 ({len(self.stokes_u)}) 与 I 不匹配 ({nl})")

        if len(self.wavelength) > 0 and len(self.wavelength) != nl:
            raise ValueError(f"波长长度 ({len(self.wavelength)}) 与光谱不匹配 ({nl})")

        if self.error is not None and len(self.error) != nl:
            raise ValueError(f"误差长度 ({len(self.error)}) 与光谱不匹配 ({nl})")

    def get_chi2(self,
                 obs_spectrum: np.ndarray,
                 obs_spectrum_other: Optional[np.ndarray] = None,
                 obs_error: Optional[np.ndarray] = None,
                 obs_error_other: Optional[np.ndarray] = None) -> float:
        """计算与观测数据的 χ² 值
        
        根据 pol_channel 计算对应 Stokes 分量的 χ²。
        
        Parameters
        ----------
        obs_spectrum : np.ndarray
            观测 Stokes I 光谱 (Nλ,)
        obs_spectrum_other : Optional[np.ndarray]
            观测 Stokes V/Q/U 光谱 (Nλ,)，仅当 pol_channel != 'I' 时使用
        obs_error : Optional[np.ndarray]
            Stokes I 观测误差 (Nλ,)，默认为结果中的 error
        obs_error_other : Optional[np.ndarray]
            Stokes V/Q/U 观测误差 (Nλ,)
        
        Returns
        -------
        float
            χ² 值 = Σ((obs - model) / σ)²
            
        Notes
        -----
        - pol_channel='I': 仅计算 Stokes I 的 χ²
        - pol_channel='V'/'Q'/'U': 计算对应分量的 χ²（不包括 I）
        """
        self.validate()

        if obs_spectrum is None or len(obs_spectrum) != len(self.stokes_i):
            raise ValueError("观测光谱维度不匹配")

        pol_ch = self.pol_channel.upper()

        # 验证 pol_channel 值
        if pol_ch not in ('I', 'V', 'Q', 'U'):
            raise ValueError(
                f"pol_channel 必须为 'I', 'V', 'Q' 或 'U'，收到 '{pol_ch}'")

        # Stokes I 通道
        if pol_ch == 'I':
            sigma_i = obs_error if obs_error is not None else self.error
            if sigma_i is None or np.all(sigma_i == 0):
                residuals = (obs_spectrum - self.stokes_i)**2
            else:
                residuals = ((obs_spectrum - self.stokes_i) / sigma_i)**2
            return float(np.sum(residuals))

        # Stokes V 通道
        if pol_ch == 'V':
            if obs_spectrum_other is None:
                raise ValueError("pol_channel='V' 时必须提供 obs_spectrum_other")
            if len(obs_spectrum_other) != len(self.stokes_v):
                raise ValueError("Stokes V 观测光谱维度不匹配")
            sigma_v = obs_error_other if obs_error_other is not None else (
                self.error if self.error is not None else
                np.ones_like(self.stokes_v) * 1e-5)
            if np.all(sigma_v == 0):
                residuals = (obs_spectrum_other - self.stokes_v)**2
            else:
                residuals = ((obs_spectrum_other - self.stokes_v) / sigma_v)**2
            return float(np.sum(residuals))

        # Stokes Q 通道
        if pol_ch == 'Q':
            if obs_spectrum_other is None:
                raise ValueError("pol_channel='Q' 时必须提供 obs_spectrum_other")
            if self.stokes_q is None:
                raise ValueError("模型中不存在 Stokes Q")
            if len(obs_spectrum_other) != len(self.stokes_q):
                raise ValueError("Stokes Q 观测光谱维度不匹配")
            sigma_q = obs_error_other if obs_error_other is not None else (
                self.error if self.error is not None else
                np.ones_like(self.stokes_q) * 1e-5)
            if np.all(sigma_q == 0):
                residuals = (obs_spectrum_other - self.stokes_q)**2
            else:
                residuals = ((obs_spectrum_other - self.stokes_q) / sigma_q)**2
            return float(np.sum(residuals))

        # Stokes U 通道
        if pol_ch == 'U':
            if obs_spectrum_other is None:
                raise ValueError("pol_channel='U' 时必须提供 obs_spectrum_other")
            if self.stokes_u is None:
                raise ValueError("模型中不存在 Stokes U")
            if len(obs_spectrum_other) != len(self.stokes_u):
                raise ValueError("Stokes U 观测光谱维度不匹配")
            sigma_u = obs_error_other if obs_error_other is not None else (
                self.error if self.error is not None else
                np.ones_like(self.stokes_u) * 1e-5)
            if np.all(sigma_u == 0):
                residuals = (obs_spectrum_other - self.stokes_u)**2
            else:
                residuals = ((obs_spectrum_other - self.stokes_u) / sigma_u)**2
            return float(np.sum(residuals))

        # 这行代码不应该被执行（已在前面验证 pol_ch 值）
        raise ValueError(f"无效的 pol_channel 值：{pol_ch}")

    def get_relative_residuals(
            self,
            obs_spectrum: np.ndarray,
            obs_spectrum_other: Optional[np.ndarray] = None) -> np.ndarray:
        """计算相对残差
        
        根据 pol_channel 计算对应 Stokes 分量的相对残差。
        
        Parameters
        ----------
        obs_spectrum : np.ndarray
            观测 Stokes I 光谱
        obs_spectrum_other : Optional[np.ndarray]
            观测 Stokes V/Q/U 光谱，仅当 pol_channel != 'I' 时使用
        
        Returns
        -------
        np.ndarray
            相对残差数组 = (obs - model) / obs
            
        Notes
        -----
        - pol_channel='I': 计算 Stokes I 残差
        - pol_channel='V'/'Q'/'U': 计算对应分量的残差（不包括 I）
        """
        self.validate()

        pol_ch = self.pol_channel.upper()

        # 验证 pol_channel 值
        if pol_ch not in ('I', 'V', 'Q', 'U'):
            raise ValueError(
                f"pol_channel 必须为 'I', 'V', 'Q' 或 'U'，收到 '{pol_ch}'")

        # Stokes I 通道
        if pol_ch == 'I':
            safe_obs = np.where(
                np.abs(obs_spectrum) > 1e-10, obs_spectrum, 1e-10)
            return (obs_spectrum - self.stokes_i) / safe_obs

        # Stokes V 通道
        if pol_ch == 'V':
            if obs_spectrum_other is None:
                raise ValueError("pol_channel='V' 时必须提供 obs_spectrum_other")
            if len(obs_spectrum_other) != len(self.stokes_v):
                raise ValueError("Stokes V 观测光谱维度不匹配")
            safe_obs = np.where(
                np.abs(obs_spectrum_other) > 1e-10, obs_spectrum_other, 1e-10)
            return (obs_spectrum_other - self.stokes_v) / safe_obs

        # Stokes Q 通道
        if pol_ch == 'Q':
            if obs_spectrum_other is None:
                raise ValueError("pol_channel='Q' 时必须提供 obs_spectrum_other")
            if self.stokes_q is None:
                raise ValueError("模型中不存在 Stokes Q")
            if len(obs_spectrum_other) != len(self.stokes_q):
                raise ValueError("Stokes Q 观测光谱维度不匹配")
            safe_obs = np.where(
                np.abs(obs_spectrum_other) > 1e-10, obs_spectrum_other, 1e-10)
            return (obs_spectrum_other - self.stokes_q) / safe_obs

        # Stokes U 通道
        if pol_ch == 'U':
            if obs_spectrum_other is None:
                raise ValueError("pol_channel='U' 时必须提供 obs_spectrum_other")
            if self.stokes_u is None:
                raise ValueError("模型中不存在 Stokes U")
            if len(obs_spectrum_other) != len(self.stokes_u):
                raise ValueError("Stokes U 观测光谱维度不匹配")
            safe_obs = np.where(
                np.abs(obs_spectrum_other) > 1e-10, obs_spectrum_other, 1e-10)
            return (obs_spectrum_other - self.stokes_u) / safe_obs

        # 这行代码不应该被执行（已在前面验证 pol_ch 值）
        raise ValueError(f"无效的 pol_channel 值：{pol_ch}")

    def get_spectrum_stats(self) -> Dict[str, Any]:
        """获取光谱统计信息
        
        Returns
        -------
        dict
            包含各分量统计的字典
        """
        self.validate()

        stats = {
            'wavelength_range':
            (float(self.wavelength.min()), float(self.wavelength.max()))
            if len(self.wavelength) > 0 else None,
            'stokes_i': {
                'min': float(self.stokes_i.min()),
                'max': float(self.stokes_i.max()),
                'mean': float(self.stokes_i.mean()),
                'std': float(self.stokes_i.std()),
            },
            'stokes_v': {
                'min': float(self.stokes_v.min()),
                'max': float(self.stokes_v.max()),
                'mean': float(self.stokes_v.mean()),
                'std': float(self.stokes_v.std()),
                'amplitude': float(self.stokes_v.max() - self.stokes_v.min()),
            },
        }

        if self.stokes_q is not None:
            stats['stokes_q'] = {
                'min': float(self.stokes_q.min()),
                'max': float(self.stokes_q.max()),
                'mean': float(self.stokes_q.mean()),
                'std': float(self.stokes_q.std()),
            }

        if self.stokes_u is not None:
            stats['stokes_u'] = {
                'min': float(self.stokes_u.min()),
                'max': float(self.stokes_u.max()),
                'mean': float(self.stokes_u.mean()),
                'std': float(self.stokes_u.std()),
            }

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """序列化结果为字典（用于保存）
        
        Returns
        -------
        dict
            结果字典
        """
        self.validate()

        result = {
            'hjd': self.hjd,
            'phase_index': self.phase_index,
            'pol_channel': self.pol_channel,
            'model_name': self.model_name,
            'creation_time': self._creation_time,
        }

        return result

    def save_spectrum(self,
                      output_path: str,
                      par: Optional[Any] = None,
                      obsSet: Optional[List[Any]] = None,
                      verbose: int = 1) -> str:
        """将合成光谱保存到文件
        
        调用 mainFuncs.save_model_spectra_to_outModelSpec 以保证正确的输出格式
        和 pol_channel 参数传递。
        
        Parameters
        ----------
        output_path : str
            输出文件路径或目录
        par : Optional[readParamsTomog]
            参数对象，如果为 None 将使用结果中的元数据
        obsSet : Optional[List[ObservationProfile]]
            观测数据对象列表，如果为 None 将使用默认格式
        verbose : int
            详细程度 (0=静默, 1=正常, 2=详细)
        
        Returns
        -------
        str
            生成的输出文件路径
        """
        from pathlib import Path
        import core.mainFuncs as mf

        self.validate()

        # 构造临时 par 对象（如果未提供）
        if par is None:
            from types import SimpleNamespace
            par = SimpleNamespace()
            par.jDates = np.array(
                [self.hjd] if self.hjd is not None else [0.0])
            par.velRs = np.array([0.0])
            par.polChannels = np.array([self.pol_channel])
            par.phases = np.array([self.phase_index])

        # 构造临时 obsSet（如果未提供）
        if obsSet is None:
            from core.SpecIO import ObservationProfile
            # 创建临时观测对象，用于推断格式
            obsSet = [
                ObservationProfile(wl=self.wavelength,
                                   specI=self.stokes_i,
                                   specV=self.stokes_v,
                                   specQ=self.stokes_q,
                                   specU=self.stokes_u,
                                   profile_type="spec",
                                   pol_channel=self.pol_channel)
            ]

        # 组织结果为列表（格式：(v_grid, specI, specV, specQ, specU, pol_channel)）
        # 注意：wavelength 作为 x 轴（可能是波长或速度）
        result_tuple = (self.wavelength, self.stokes_i, self.stokes_v,
                        self.stokes_q, self.stokes_u, self.pol_channel)
        results = [result_tuple]

        # 调用 mainFuncs.save_model_spectra_to_outModelSpec
        output_files = mf.save_model_spectra_to_outModelSpec(
            par,
            results,
            obsSet,
            output_dir=str(Path(output_path).parent),
            verbose=verbose)

        if output_files:
            return output_files[0]
        else:
            raise ValueError("保存失败：未能生成输出文件")

    def save_geomodel(self,
                      output_dir: str = './output',
                      integrator=None,
                      meta: Optional[Dict[str, Any]] = None,
                      verbose: int = 1) -> str:
        """将几何模型保存到文件（geomodel.tomog 格式）
        
        调用 VelspaceDiskIntegrator.write_geomodel() 以保存当前的几何和物理模型。
        
        Parameters
        ----------
        output_dir : str
            输出目录路径
        integrator : VelspaceDiskIntegrator, optional
            积分器对象，包含几何模型信息。如果为 None，将尝试从结果中获取
        meta : Optional[Dict[str, Any]]
            额外的元数据（如观测信息），将包含在文件头中
        verbose : int
            详细程度 (0=静默, 1=正常, 2=详细)
            
        Returns
        -------
        str
            生成的输出文件路径
            
        Raises
        ------
        ValueError
            如果没有提供 integrator 且无法从结果中获取
            
        Notes
        -----
        geomodel.tomog 文件包含以下信息：
        - 速度场参数（disk_v0_kms, disk_power_index 等）
        - 几何参数（倾角、差速旋转指数等）
        - 网格定义（径向层数、网格边界等）
        - 每个像素的物理量（磁场、区域权重、振幅等）
        """
        from pathlib import Path

        if integrator is None:
            integrator = getattr(self, 'integrator', None)

        if integrator is None:
            raise ValueError(
                "必须提供 integrator 参数以保存几何模型。"
                "integrator 应该是从 ForwardModelResult 生成时使用的 VelspaceDiskIntegrator 实例。"
            )

        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 构造输出文件名
        filename = f"geomodel_phase_{self.phase_index:02d}.tomog"
        filepath = output_path / filename

        # 构造元数据
        if meta is None:
            meta = {}

        # 添加来自 ForwardModelResult 的信息
        meta.update({
            'phase_index': self.phase_index,
            'hjd': self.hjd,
            'pol_channel': self.pol_channel,
            'model_name': self.model_name,
            'creation_time': self._creation_time,
        })

        if verbose:
            print(f"[save_geomodel] 保存几何模型到: {filepath}")
            print(f"[save_geomodel] 相位索引: {self.phase_index}")
            print(f"[save_geomodel] HJD: {self.hjd}")

        try:
            # 调用 integrator 的 write_geomodel 方法
            integrator.write_geomodel(str(filepath), meta=meta)

            if verbose:
                print(f"[save_geomodel] ✓ 几何模型保存成功: {filepath}")

            return str(filepath)

        except Exception as e:
            raise ValueError(f"保存几何模型失败: {type(e).__name__}: {e}")

    def save_model_data(self,
                        output_dir: str = './output',
                        integrator=None,
                        par=None,
                        obsSet=None,
                        verbose: int = 1) -> Dict[str, str]:
        """一次性保存所有模型数据（光谱 + 几何模型）
        
        方便的包装方法，同时保存合成光谱和几何模型文件。
        
        Parameters
        ----------
        output_dir : str
            输出目录路径
        integrator : VelspaceDiskIntegrator, optional
            积分器对象（用于保存几何模型）
        par : Optional[readParamsTomog]
            参数对象（用于保存光谱）
        obsSet : Optional[List[ObservationProfile]]
            观测数据集（用于保存光谱）
        verbose : int
            详细程度
            
        Returns
        -------
        dict
            包含生成的文件路径的字典：
            {
                'spectrum': '...',  # 光谱文件
                'geomodel': '...',  # 几何模型文件（如果提供 integrator）
            }
            
        Examples
        --------
        >>> result = forward_tomography(...)
        >>> files = result[0].save_model_data(
        ...     output_dir='./output',
        ...     integrator=phys_model.integrator,
        ...     verbose=1
        ... )
        >>> print(f"光谱保存到: {files['spectrum']}")
        >>> print(f"几何模型保存到: {files['geomodel']}")
        """
        output_files = {}

        if verbose:
            print(f"[save_model_data] 保存模型数据到: {output_dir}")

        # 保存光谱
        try:
            spectrum_file = self.save_spectrum(output_dir,
                                               par=par,
                                               obsSet=obsSet,
                                               verbose=verbose)
            output_files['spectrum'] = spectrum_file
        except Exception as e:
            if verbose:
                print(f"[save_model_data] ⚠️  光谱保存失败: {e}")
            output_files['spectrum'] = None

        # 保存几何模型（如果提供 integrator）
        if integrator is not None:
            try:
                geomodel_file = self.save_geomodel(output_dir=output_dir,
                                                   integrator=integrator,
                                                   verbose=verbose)
                output_files['geomodel'] = geomodel_file
            except Exception as e:
                if verbose:
                    print(f"[save_model_data] ⚠️  几何模型保存失败: {e}")
                output_files['geomodel'] = None

        if verbose:
            print("[save_model_data] ✓ 模型数据保存完成")
            for key, path in output_files.items():
                if path:
                    print(f"  - {key}: {path}")

        return output_files

    def create_summary(self) -> str:
        """生成结果摘要字符串
        
        Returns
        -------
        str
            格式化的结果摘要
        """
        self.validate()
        stats = self.get_spectrum_stats()

        lines = [
            "=" * 70,
            f"正演结果摘要 (相位 {self.phase_index}, HJD={self.hjd})",
            "=" * 70,
            f"模型: {self.model_name}",
            f"偏振通道: {self.pol_channel}",
            f"光谱点数: {len(self.stokes_i)}",
        ]

        if stats['wavelength_range']:
            lines.append(
                f"波长范围: {stats['wavelength_range'][0]:.4f} - {stats['wavelength_range'][1]:.4f} nm"
            )

        lines.extend([
            "",
            "Stokes I 统计:",
            f"  范围: [{stats['stokes_i']['min']:.6f}, {stats['stokes_i']['max']:.6f}]",
            f"  均值: {stats['stokes_i']['mean']:.6f}",
            f"  标准差: {stats['stokes_i']['std']:.6f}",
            "",
            "Stokes V 统计:",
            f"  范围: [{stats['stokes_v']['min']:.6f}, {stats['stokes_v']['max']:.6f}]",
            f"  振幅: {stats['stokes_v']['amplitude']:.6f}",
            f"  均值: {stats['stokes_v']['mean']:.6f}",
        ])

        if 'stokes_q' in stats:
            lines.extend([
                "",
                "Stokes Q 统计:",
                f"  范围: [{stats['stokes_q']['min']:.6f}, {stats['stokes_q']['max']:.6f}]",
                f"  均值: {stats['stokes_q']['mean']:.6f}",
            ])

        if 'stokes_u' in stats:
            lines.extend([
                "",
                "Stokes U 统计:",
                f"  范围: [{stats['stokes_u']['min']:.6f}, {stats['stokes_u']['max']:.6f}]",
                f"  均值: {stats['stokes_u']['mean']:.6f}",
            ])

        lines.append("=" * 70)

        return "\n".join(lines)


@dataclass
class InversionResult:
    """MEM 反演结果容器
    
    存储 MEM 反演工作流的输出结果，包括多次迭代的演化信息。
    """

    # ============ 最终磁场解 ============

    B_los_final: np.ndarray
    """最终视向磁场 (Npix,)"""

    B_perp_final: np.ndarray
    """最终垂直磁场 (Npix,)"""

    chi_final: np.ndarray
    """最终磁场方向角 (Npix,)"""

    # ============ 迭代追踪 ============

    iterations_completed: int = 0
    """已完成的迭代次数"""

    chi2_history: List[float] = field(default_factory=list)
    """χ² 值随迭代变化"""

    entropy_history: List[float] = field(default_factory=list)
    """熵值随迭代变化"""

    regularization_history: List[float] = field(default_factory=list)
    """正则化项随迭代变化"""

    # ============ 收敛信息 ============

    converged: bool = False
    """是否收敛"""

    convergence_reason: str = ""
    """收敛原因描述"""

    final_chi2: float = 0.0
    """最终 χ² 值"""

    final_entropy: float = 0.0
    """最终熵值"""

    # ============ 统计信息 ============

    magnetic_field_stats: Optional[Dict[str, Any]] = None
    """磁场统计信息"""

    fit_quality: Dict[str, float] = field(default_factory=dict)
    """拟合质量指标 (例: 'rms_residual', 'max_residual' 等)"""

    # ============ 元数据 ============

    phase_index: int = 0
    """相位索引"""

    pol_channels: List[str] = field(default_factory=lambda: ["I+V"])
    """处理的偏振通道列表"""

    # ============ 内部状态 ============

    _creation_time: str = field(
        default_factory=lambda: datetime.now().isoformat(),
        init=False,
        repr=False)
    """创建时间"""

    def validate(self) -> None:
        """验证结果数据完整性
        
        Raises
        ------
        ValueError
            数据不一致或不完整
        """
        if self.B_los_final is None or len(self.B_los_final) == 0:
            raise ValueError("B_los_final 不能为空")

        npix = len(self.B_los_final)

        if len(self.B_perp_final) != npix:
            raise ValueError(
                f"B_perp_final 长度 ({len(self.B_perp_final)}) 与 B_los 不匹配 ({npix})"
            )

        if len(self.chi_final) != npix:
            raise ValueError(
                f"chi_final 长度 ({len(self.chi_final)}) 与 B_los 不匹配 ({npix})")

        if self.iterations_completed < 0:
            raise ValueError(f"迭代次数不能为负: {self.iterations_completed}")

        # 仅当迭代数 > 0 且存在历史记录时检查一致性
        if self.iterations_completed > 0:
            if len(self.chi2_history) > 0 and len(
                    self.chi2_history) != self.iterations_completed:
                raise ValueError(
                    f"χ² 历史长度 ({len(self.chi2_history)}) 与迭代次数不匹配 ({self.iterations_completed})"
                )

    def get_convergence_rate(self) -> Optional[float]:
        """计算收敛速率
        
        Returns
        -------
        Optional[float]
            相邻迭代间 χ² 相对变化的平均值，若历史不足则返回 None
        """
        if len(self.chi2_history) < 2:
            return None

        chi2_array = np.array(self.chi2_history)
        # 避免除零
        denom = np.where(
            np.abs(chi2_array[:-1]) > 1e-10, chi2_array[:-1], 1e-10)
        changes = np.abs(np.diff(chi2_array) / denom)

        return float(np.mean(changes))

    def get_magnetic_field_stats(self) -> Dict[str, Any]:
        """计算磁场统计信息
        
        Returns
        -------
        dict
            包含 B_los、B_perp 和 χ 的统计信息
        """
        self.validate()

        # 计算垂直磁场大小
        B_mag = np.sqrt(self.B_perp_final**2)

        stats = {
            'B_los': {
                'min': float(self.B_los_final.min()),
                'max': float(self.B_los_final.max()),
                'mean': float(self.B_los_final.mean()),
                'std': float(self.B_los_final.std()),
                'rms': float(np.sqrt(np.mean(self.B_los_final**2))),
            },
            'B_perp': {
                'min': float(self.B_perp_final.min()),
                'max': float(self.B_perp_final.max()),
                'mean': float(self.B_perp_final.mean()),
                'std': float(self.B_perp_final.std()),
                'rms': float(np.sqrt(np.mean(self.B_perp_final**2))),
            },
            'B_mag': {
                'min': float(B_mag.min()),
                'max': float(B_mag.max()),
                'mean': float(B_mag.mean()),
                'rms': float(np.sqrt(np.mean(B_mag**2))),
            },
            'chi': {
                'min': float(self.chi_final.min()),
                'max': float(self.chi_final.max()),
                'mean': float(self.chi_final.mean()),
                'std': float(self.chi_final.std()),
            },
            'npix': len(self.B_los_final),
        }

        return stats

    def get_optimization_metrics(self) -> Dict[str, float]:
        """获取优化过程的关键指标
        
        Returns
        -------
        dict
            优化指标 (iterations, final_chi2, convergence_rate 等)
        """
        self.validate()

        metrics = {
            'iterations': self.iterations_completed,
            'final_chi2': self.final_chi2,
            'final_entropy': self.final_entropy,
            'converged': float(self.converged),
            'convergence_rate': self.get_convergence_rate() or 0.0,
        }

        # 添加拟合质量指标
        metrics.update(self.fit_quality)

        return metrics

    def create_summary(self) -> str:
        """生成反演结果摘要
        
        Returns
        -------
        str
            格式化的摘要字符串
        """
        self.validate()

        mag_stats = self.get_magnetic_field_stats()
        opt_metrics = self.get_optimization_metrics()

        lines = [
            "=" * 70,
            f"MEM 反演结果摘要 (相位 {self.phase_index})",
            "=" * 70,
            "",
            "收敛状态:",
            f"  已完成迭代: {self.iterations_completed}",
            f"  是否收敛: {self.converged}",
            f"  收敛原因: {self.convergence_reason}",
            f"  收敛速率: {opt_metrics['convergence_rate']:.3e}",
            "",
            "优化指标:",
            f"  最终 χ²: {self.final_chi2:.6e}",
            f"  最终熵: {self.final_entropy:.6e}",
            "",
            "磁场统计:",
            f"  B_los: [{mag_stats['B_los']['min']:.1f}, {mag_stats['B_los']['max']:.1f}] G",
            f"         均值 {mag_stats['B_los']['mean']:.1f}, RMS {mag_stats['B_los']['rms']:.1f}",
            f"  B_perp: [{mag_stats['B_perp']['min']:.1f}, {mag_stats['B_perp']['max']:.1f}] G",
            f"          均值 {mag_stats['B_perp']['mean']:.1f}, RMS {mag_stats['B_perp']['rms']:.1f}",
            f"  B_mag: RMS {mag_stats['B_mag']['rms']:.1f} G",
            "",
            "处理的偏振通道:",
            f"  {', '.join(self.pol_channels)}",
            "=" * 70,
        ]

        return "\n".join(lines)
