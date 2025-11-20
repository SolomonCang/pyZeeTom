# spot_simulator.py (精简版)
# Spot模拟库：多spot配置、几何模型构建、.tomog输出
# 功能：
#   1. 多个spot的管理（位置、磁场、时间演化）
#   2. 将spot映射到disk grid像素
#   3. 生成符合规范的.tomog模型文件
#   4. 注意：谱线合成由 pyzeetom/tomography.py 的0-iter正演处理
#           通过 initTomogFile 参数加载模型

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SpotConfig:
    """单个spot的配置参数"""
    r: float  # 径向位置（盘面半径单位）
    phi: float  # 初始方位角（弧度）
    amplitude: float  # 振幅（正值=发射，负值=吸收）
    spot_type: str = 'emission'  # 'emission' 或 'absorption'
    radius: float = 0.5  # spot的径向宽度（FWHM）
    width_type: str = 'gaussian'  # 宽度类型 'gaussian' 或 'tophat'
    B_los: float = 0.0  # 视向磁场
    B_perp: float = 0.0  # 横向磁场
    chi: float = 0.0  # 磁场方位角（弧度）
    velocity_shift: float = 0.0  # 附加速度偏移（km/s）

    def __post_init__(self):
        self.r = float(self.r)
        self.phi = float(self.phi)
        self.amplitude = float(self.amplitude)
        self.radius = float(self.radius)
        self.B_los = float(self.B_los)
        self.B_perp = float(self.B_perp)
        self.chi = float(self.chi)
        self.velocity_shift = float(self.velocity_shift)


class SpotSimulator:
    """
    Spot模拟器：
      - 管理多个spot配置
      - 将spot映射到disk grid像素
      - 计算响应函数和磁场分布
      - 输出.tomog模型文件供后续正演使用
    """

    def __init__(self,
                 grid,
                 inclination_rad: float = np.deg2rad(60.0),
                 phi0: float = 0.0,
                 pOmega: float = -0.5,
                 r0_rot: float = 1.0,
                 period_days: float = 1.0):
        """
        初始化spot模拟器

        参数:
        -------
        grid : diskGrid
            网格对象（来自 grid_tom.py）
        inclination_rad : float
            倾角（弧度），默认60度
        phi0 : float
            参考方位角（弧度）
        pOmega : float
            较差转动指数（Ω ∝ r^pOmega）
        r0_rot : float
            较差转动参考半径
        period_days : float
            自转周期（天）
        """
        self.grid = grid
        self.inclination_rad = float(inclination_rad)
        self.phi0 = float(phi0)
        self.pOmega = float(pOmega)
        self.r0_rot = float(r0_rot)
        self.period_days = float(period_days)

        # Spot列表
        self.spots: List[SpotConfig] = []

        # 观测参数数组（与 phase 同长度）
        # 这些参数用于生成多相位合成数据
        self.phases: np.ndarray = np.array([])  # 观测相位（0~1）
        self.vel_shifts: np.ndarray = np.array([])  # 速度偏移（km/s），每个相位一个值
        self.pol_channels: list = []  # 偏振通道，每个相位一个值（'I'/'V'/'Q'/'U'）

        # 初始化像素属性数组
        self._init_pixel_arrays()

    def _init_pixel_arrays(self):
        """初始化所有像素属性数组"""
        n = self.grid.numPoints
        self.amp = np.ones(n, dtype=float)  # 谱线振幅（>0，直接来自几何）
        self.B_los_map = np.zeros(n, dtype=float)
        self.B_perp_map = np.zeros(n, dtype=float)
        self.chi_map = np.zeros(n, dtype=float)

    def add_spot(self, spot_config: SpotConfig):
        """添加单个spot"""
        self.spots.append(spot_config)

    def add_spots(self, spot_configs: List[SpotConfig]):
        """添加多个spot"""
        self.spots.extend(spot_configs)

    def create_spot(self, r: float, phi: float, amplitude: float,
                    **kwargs) -> SpotConfig:
        """
        创建并添加单个spot（快捷方法）

        参数: 参见 SpotConfig
        返回: SpotConfig对象
        """
        spot = SpotConfig(r=r, phi=phi, amplitude=amplitude, **kwargs)
        self.add_spot(spot)
        return spot

    def evolve_spots_to_phase(self, phase: float) -> List[SpotConfig]:
        """
        将spot演化到指定相位（考虑较差转动）

        参数:
        -------
        phase : float
            旋转相位（0~1）

        返回:
        -------
        List[SpotConfig]
            演化后的spot列表
        """
        evolved_spots = []
        for spot in self.spots:
            # 复制当前spot配置
            s = SpotConfig(r=spot.r,
                           phi=spot.phi,
                           amplitude=spot.amplitude,
                           spot_type=spot.spot_type,
                           radius=spot.radius,
                           width_type=spot.width_type,
                           B_los=spot.B_los,
                           B_perp=spot.B_perp,
                           chi=spot.chi,
                           velocity_shift=spot.velocity_shift)

            # 考虑较差转动的相位演化
            # Δφ = 2π * phase * (r/r0_rot)^(pOmega+1)
            # 这是从 pOmega 定义出发：Ω(r) = Ω_ref * (r/r0_rot)^pOmega
            radius_ratio = s.r / self.r0_rot if self.r0_rot > 0 else 1.0
            delphi = 2.0 * np.pi * phase * (radius_ratio**(self.pOmega + 1.0))
            s.phi = s.phi + delphi

            # 归一化到 [0, 2π)
            s.phi = s.phi % (2.0 * np.pi)

            evolved_spots.append(s)

        return evolved_spots

    def _gaussian_weight(self, dr: np.ndarray, sigma: float) -> np.ndarray:
        """高斯权重函数"""
        return np.exp(-0.5 * (dr / sigma)**2)

    def _tophat_weight(self, dr: np.ndarray, radius: float) -> np.ndarray:
        """顶帽权重函数（rect函数）"""
        weight = np.zeros_like(dr)
        weight[np.abs(dr) <= radius] = 1.0
        return weight

    def _compute_azimuthal_weight(self, dphi: np.ndarray,
                                  sigma_phi: float) -> np.ndarray:
        """方位角权重（高斯）"""
        # 处理 2π 周期性
        dphi_wrapped = np.abs(dphi)
        dphi_wrapped = np.minimum(dphi_wrapped, 2.0 * np.pi - dphi_wrapped)
        return np.exp(-0.5 * (dphi_wrapped / sigma_phi)**2)

    def apply_spots_to_grid(self, phase: float = 0.0) -> None:
        """
        应用spot到网格，计算响应和磁场分布

        参数:
        -------
        phase : float
            旋转相位（0~1）
        """
        # 初始化
        self._init_pixel_arrays()

        # 获取演化的spot
        spots_at_phase = self.evolve_spots_to_phase(phase)

        # 对每个spot应用到网格
        for spot in spots_at_phase:
            self._apply_single_spot(spot)

    def _apply_single_spot(self, spot: SpotConfig) -> None:
        """将单个spot应用到网格"""
        # 计算每个像素到spot的距离
        dr = np.sqrt((self.grid.r - spot.r)**2)
        dphi = self.grid.phi - spot.phi

        # 径向权重
        if spot.width_type == 'gaussian':
            sigma_r = spot.radius / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            r_weight = self._gaussian_weight(dr, sigma_r)
        else:  # tophat
            r_weight = self._tophat_weight(dr, spot.radius)

        # 方位角权重（高斯，宽度为 spot.radius 在方位角方向）
        sigma_phi = spot.radius / (self.grid.r + 1e-10)  # 防止除零
        phi_weight = self._compute_azimuthal_weight(dphi, sigma_phi)

        # 总权重
        weight = r_weight * phi_weight

        # 应用谱线振幅
        if spot.spot_type == 'emission':
            # 发射：amp > 1
            self.amp += (spot.amplitude) * weight
        else:  # absorption
            # 吸收：amp < 1
            self.amp += (spot.amplitude) * weight

        # 应用磁场（加权平均）
        weight_sum = np.sum(weight)
        if weight_sum > 0:
            self.B_los_map += spot.B_los * weight
            self.B_perp_map += spot.B_perp * weight
            self.chi_map += spot.chi * weight

    def create_geometry_object(self, phase: float = 0.0):
        """
        创建几何对象供VelspaceDiskIntegrator使用

        参数:
        -------
        phase : float
            旋转相位

        返回:
        -------
        SimpleDiskGeometry
            几何对象，包含网格、磁场参数、谱线振幅等
        """
        from core.disk_geometry_integrator import SimpleDiskGeometry

        # 应用spot到网格
        self.apply_spots_to_grid(phase)

        # 构造SimpleDiskGeometry对象
        geom = SimpleDiskGeometry(grid=self.grid,
                                  inclination_deg=float(
                                      np.rad2deg(self.inclination_rad)),
                                  phi0=float(self.phi0),
                                  pOmega=float(self.pOmega),
                                  r0=float(self.r0_rot),
                                  period=float(self.period_days),
                                  enable_stellar_occultation=0,
                                  stellar_radius=1.0,
                                  amp=self.amp.copy(),
                                  B_los=self.B_los_map.copy(),
                                  B_perp=self.B_perp_map.copy(),
                                  chi=self.chi_map.copy())

        return geom

    def export_to_geomodel(self,
                           filepath: str,
                           phase: float = 0.0,
                           meta: Optional[Dict[str, Any]] = None) -> str:
        """
        将模型导出为.tomog文件供后续正演使用

        该方法使用 VelspaceDiskIntegrator.write_geomodel() 进行标准化输出。

        参数:
        -------
        filepath : str
            输出文件路径（.tomog格式）
        phase : float
            相位
        meta : dict, optional
            元信息（目标名、观测参数等）

        返回:
        -------
        str
            输出文件路径
        """
        from pathlib import Path
        from core.disk_geometry_integrator import VelspaceDiskIntegrator

        # 创建几何对象
        geom = self.create_geometry_object(phase=phase)

        # 设置元数据
        if meta is None:
            meta = {}
        meta['phase'] = float(phase)
        meta['source'] = 'SpotSimulator'

        # 创建一个虚拟的积分器以使用 write_geomodel 方法
        # 这里只需要积分器的 write_geomodel 方法和对几何的访问
        try:
            # 尝试创建完整的积分器（需要谱线模型）
            # 但仅用于调用 write_geomodel，不需要完整的spectral synthesis
            from core.local_linemodel_basic import GaussianZeemanWeakLineModel, LineData
            line_data = LineData('input/lines.txt')
            line_model = GaussianZeemanWeakLineModel(line_data)

            v_grid = np.array([0.0])  # 虚拟速度网格
            integrator = VelspaceDiskIntegrator(geom=geom,
                                                wl0_nm=656.3,
                                                v_grid=v_grid,
                                                line_model=line_model,
                                                inst_fwhm_kms=0.0,
                                                normalize_continuum=False)

            # 使用 integrator 的 write_geomodel 方法
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            integrator.write_geomodel(filepath, meta=meta)

        except Exception as e:
            # 如果无法创建完整的积分器，使用简化的导出方法
            import datetime as dt

            print(
                f"[SpotSimulator] Warning: Could not use VelspaceDiskIntegrator.write_geomodel(): {e}"
            )
            print(f"[SpotSimulator] Falling back to simplified export...")

            # 构造header
            header = {
                "format": "TOMOG_MODEL",
                "version": 1,
                "created_utc": dt.datetime.utcnow().isoformat() + "Z",
                "source": "SpotSimulator",
                "inclination_deg": float(np.rad2deg(self.inclination_rad)),
                "phi0": float(self.phi0),
                "pOmega": float(self.pOmega),
                "r0_rot": float(self.r0_rot),
                "period": float(self.period_days),
                "nr": int(self.grid.nr) if hasattr(self.grid, 'nr') else -1,
                "phase": float(phase),
            }

            if isinstance(meta, dict):
                for k, v in meta.items():
                    if k not in header:
                        header[str(k)] = v

            # 检查是否有 B_perp 和 chi
            has_B_perp = np.any(self.B_perp_map != 0.0)
            has_chi = np.any(self.chi_map != 0.0)

            # 列定义
            columns = [
                "idx", "ring_id", "phi_id", "r", "phi", "area", "Ic_weight",
                "amp", "Blos"
            ]
            if has_B_perp:
                columns.extend(["Bperp"])
            if has_chi:
                columns.extend(["chi"])

            # 写文件
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(
                    "# TOMOG Geometric Model File (generated by SpotSimulator)\n"
                )
                for k in sorted(header.keys()):
                    v = header[k]
                    if isinstance(v, (list, tuple, np.ndarray)):
                        arr = np.asarray(v).ravel()
                        vstr = ",".join(f"{x:.12g}" for x in arr)
                        f.write(f"# {k}: [{vstr}]\n")
                    else:
                        f.write(f"# {k}: {v}\n")

                # 尝试补充网格边界
                if hasattr(self.grid, "r_edges"):
                    vstr = ",".join(
                        f"{x:.12g}"
                        for x in np.asarray(self.grid.r_edges).ravel())
                    f.write(f"# r_edges: [{vstr}]\n")

                f.write("# COLUMNS: " + ", ".join(columns) + "\n")

                # 写数据
                n = self.grid.numPoints
                for i in range(n):
                    row = [
                        i,
                        int(self.grid.ring_id[i]) if hasattr(
                            self.grid, "ring_id") else -1,
                        int(getattr(self.grid, "phi_id", np.zeros(n, int))[i]),
                        float(self.grid.r[i]),
                        float(self.grid.phi[i]),
                        float(self.grid.area[i]),
                        float(1.0),  # Ic_weight (几何权重)
                        float(self.amp[i]),  # 谱线振幅
                        float(self.B_los_map[i]),
                    ]
                    if has_B_perp:
                        row.append(float(self.B_perp_map[i]))
                    if has_chi:
                        row.append(float(self.chi_map[i]))
                    f.write(" ".join(str(x) for x in row) + "\n")

        return filepath

    def configure_multi_phase_synthesis(
            self,
            phases: np.ndarray,
            vel_shifts: Optional[np.ndarray] = None,
            pol_channels: Optional[list] = None) -> None:
        """
        配置多相位合成参数

        参数:
        -------
        phases : np.ndarray
            观测相位数组，形状 (N_phase,)，值可为任意实数 (-∞, +∞)
            由于较差转动 (pOmega)，不同的相位值会导致 spot 位置不同。
            物理含义：
              - phase ∈ [0, 1)  : 自转周期内的分数位置
              - phase ≥ 1       : 多个自转周期后的位置
              - phase < 0       : 反向时间演化
            例如：phase=-0.3, 0.3, 1.3 三个相位会因 pOmega 产生不同的几何配置
        vel_shifts : np.ndarray, optional
            速度偏移数组，形状 (N_phase,)，单位 km/s
            默认为全零
        pol_channels : list, optional
            偏振通道列表，长度 N_phase
            每个元素为 'I', 'V', 'Q', 或 'U'
            默认为全 'V'

        返回:
        -------
        None
        """
        phases = np.asarray(phases, dtype=float)
        n_phase = len(phases)

        # 设置速度偏移
        if vel_shifts is None:
            vel_shifts = np.zeros(n_phase, dtype=float)
        else:
            vel_shifts = np.asarray(vel_shifts, dtype=float)
            if len(vel_shifts) != n_phase:
                raise ValueError(
                    f"vel_shifts 长度 ({len(vel_shifts)}) 必须与 phases 长度 ({n_phase}) 一致"
                )

        # 设置偏振通道
        if pol_channels is None:
            pol_channels = ['V'] * n_phase
        else:
            pol_channels = list(pol_channels)
            if len(pol_channels) != n_phase:
                raise ValueError(
                    f"pol_channels 长度 ({len(pol_channels)}) 必须与 phases 长度 ({n_phase}) 一致"
                )
            # 验证每个通道的合法性
            for ch in pol_channels:
                if ch not in ['I', 'V', 'Q', 'U']:
                    raise ValueError(f"偏振通道必须为 'I', 'V', 'Q', 或 'U'，得到 '{ch}'")

        self.phases = phases
        self.vel_shifts = vel_shifts
        self.pol_channels = pol_channels

    def generate_forward_model(self,
                               wl0_nm: float = 656.3,
                               verbose: int = 1) -> Dict[str, Any]:
        """
        利用 VelspaceDiskIntegrator 生成多相位合成光谱

        根据配置的 phases/vel_shifts/pol_channels，为每个相位生成 
        ForwardModelResult 对象。

        参数:
        -------
        wl0_nm : float
            谱线中心波长 (nm)
        verbose : int
            详细程度（0=静默，1=正常，2=详细）

        返回:
        -------
        Dict[str, Any]
            包含键：'results', 'phases', 'vel_shifts', 'pol_channels', 'metadata'

        异常:
        -------
        ValueError
            未配置 phases（调用 configure_multi_phase_synthesis）
        RuntimeError
            无法加载谱线模型
        """
        if len(self.phases) == 0:
            raise ValueError("必须先调用 configure_multi_phase_synthesis()")

        from core.tomography_result import ForwardModelResult
        from core.local_linemodel_basic import GaussianZeemanWeakLineModel, LineData
        from core.disk_geometry_integrator import VelspaceDiskIntegrator
        from scipy import interpolate

        if verbose:
            print(f"[SpotSimulator] 生成 {len(self.phases)} 个相位的合成模型...")

        results = []

        try:
            line_data = LineData('input/lines.txt')
            line_model = GaussianZeemanWeakLineModel(line_data)
        except Exception as e:
            raise RuntimeError(f"无法加载谱线模型: {e}")

        # 为每个相位生成结果
        for idx, (phase, vel_shift, pol_channel) in enumerate(
                zip(self.phases, self.vel_shifts, self.pol_channels)):

            if verbose > 1:
                print(f"  [{idx+1}/{len(self.phases)}] phase={phase:.3f}, "
                      f"vel_shift={vel_shift:.2f} km/s, pol={pol_channel}")

            try:
                # 创建该相位的几何对象
                geom = self.create_geometry_object(phase=phase)

                # 速度网格
                v_grid = np.linspace(-100, 100, 401)

                # 创建积分器
                integrator = VelspaceDiskIntegrator(geom=geom,
                                                    wl0_nm=wl0_nm,
                                                    v_grid=v_grid,
                                                    line_model=line_model,
                                                    inst_fwhm_kms=0.0,
                                                    normalize_continuum=True,
                                                    obs_phase=phase)

                # 获取 Stokes 参数
                stokes_i = getattr(integrator, 'I', np.ones_like(v_grid))
                stokes_v = getattr(integrator, 'V', np.zeros_like(v_grid))
                stokes_q = getattr(integrator, 'Q', np.zeros_like(v_grid))
                stokes_u = getattr(integrator, 'U', np.zeros_like(v_grid))

                # 应用速度偏移
                if vel_shift != 0.0:
                    v_shifted = v_grid - vel_shift

                    for arr in [stokes_i, stokes_v, stokes_q, stokes_u]:
                        f = interpolate.interp1d(v_grid,
                                                 arr,
                                                 kind='cubic',
                                                 bounds_error=False,
                                                 fill_value=np.nan)
                        arr[:] = np.nan_to_num(f(v_shifted), nan=0.0)

                    # 保持 I 的连续体为 1
                    stokes_i[np.isnan(stokes_i)] = 1.0

                # 创建 ForwardModelResult
                result = ForwardModelResult(stokes_i=stokes_i,
                                            stokes_v=stokes_v,
                                            stokes_q=stokes_q,
                                            stokes_u=stokes_u,
                                            wavelength=wl0_nm *
                                            (1.0 + v_grid / 299792.458),
                                            hjd=None,
                                            phase_index=idx,
                                            pol_channel=pol_channel,
                                            model_name="SpotSimulator")

                results.append(result)

            except Exception as e:
                if verbose:
                    print(f"  警告：相位 {phase} 失败: {e}")
                raise

        if verbose:
            print(f"[SpotSimulator] 成功生成 {len(results)} 个合成模型")

        return {
            'results': results,
            'phases': self.phases,
            'vel_shifts': self.vel_shifts,
            'pol_channels': self.pol_channels,
            'metadata': {
                'simulator': 'SpotSimulator',
                'num_phases': len(self.phases),
                'grid_npix': self.grid.numPoints,
                'num_spots': len(self.spots)
            }
        }


# ============================================================================
# 噪声添加函数
# ============================================================================


def add_noise_to_spectrum(spectrum,
                          noise_type='poisson',
                          snr=None,
                          sigma=None,
                          seed=None):
    """
    为光谱添加噪声

    参数:
    -------
    spectrum : np.ndarray
        输入光谱
    noise_type : str
        噪声类型：'poisson'（泊松噪声）、'gaussian'（高斯噪声）、'mixed'（混合）
    snr : float, optional
        信噪比（dB）。若提供，自动计算 sigma
    sigma : float, optional
        高斯噪声标准差
    seed : int, optional
        随机数种子

    返回:
    -------
    np.ndarray
        加噪后的光谱
    """
    if seed is not None:
        np.random.seed(seed)

    spectrum_noisy = spectrum.copy()

    if noise_type == 'poisson':
        # 泊松噪声：假设光子计数
        spectrum_noisy = np.random.poisson(spectrum)

    elif noise_type == 'gaussian':
        if sigma is None:
            if snr is not None:
                # SNR (dB) = 20 * log10(signal / noise)
                signal_level = np.mean(np.abs(spectrum))
                sigma = float(signal_level / (10**(snr / 20)))
            else:
                sigma = float(0.01 * np.mean(np.abs(spectrum)))

        noise = np.random.normal(0, sigma, len(spectrum))
        spectrum_noisy = spectrum + noise

    elif noise_type == 'mixed':
        # 混合：泊松 + 高斯
        spectrum_poisson = np.random.poisson(spectrum)

        if sigma is None:
            if snr is not None:
                signal_level = np.mean(np.abs(spectrum))
                sigma = float(signal_level / (10**(snr / 20)) * 0.5)
            else:
                sigma = float(0.005 * np.mean(np.abs(spectrum)))

        noise = np.random.normal(0, sigma, len(spectrum))
        spectrum_noisy = spectrum_poisson + noise

    else:
        raise ValueError(f"未知的噪声类型: {noise_type}")

    return spectrum_noisy


def add_noise_to_results(results,
                         noise_type='gaussian',
                         snr=None,
                         sigma=None,
                         seed=None):
    """
    为多个 ForwardModelResult 对象添加噪声

    参数:
    -------
    results : List
        结果列表（各为 ForwardModelResult）
    noise_type : str
        噪声类型
    snr : float, optional
        信噪比（dB）
    sigma : float, optional
        标准差
    seed : int, optional
        随机数种子

    返回:
    -------
    List
        加噪后的结果列表
    """
    from core.tomography_result import ForwardModelResult

    noisy_results = []

    for result in results:
        noisy_i = add_noise_to_spectrum(result.stokes_i, noise_type, snr,
                                        sigma, seed)
        noisy_v = add_noise_to_spectrum(result.stokes_v, noise_type, snr,
                                        sigma, seed)
        noisy_q = (add_noise_to_spectrum(result.stokes_q, noise_type, snr,
                                         sigma, seed)
                   if result.stokes_q is not None else None)
        noisy_u = (add_noise_to_spectrum(result.stokes_u, noise_type, snr,
                                         sigma, seed)
                   if result.stokes_u is not None else None)

        noisy_result = ForwardModelResult(
            stokes_i=noisy_i,
            stokes_v=noisy_v,
            stokes_q=noisy_q,
            stokes_u=noisy_u,
            wavelength=result.wavelength,
            error=sigma if sigma is not None else None,
            hjd=result.hjd,
            phase_index=result.phase_index,
            pol_channel=result.pol_channel,
            model_name=result.model_name + "_noisy")

        noisy_results.append(noisy_result)

    return noisy_results


# ============================================================================
# 便捷函数
# ============================================================================


def create_simple_spot_simulator(nr: int = 40,
                                 r_in: float = 0.5,
                                 r_out: float = 4.0,
                                 inclination_deg: float = 60.0,
                                 pOmega: float = -0.5,
                                 r0_rot: float = 1.0,
                                 period_days: float = 1.0) -> SpotSimulator:
    """
    快速创建SpotSimulator

    参数:
    -------
    nr : int
        环数
    r_in, r_out : float
        径向范围
    inclination_deg : float
        倾角（度）
    pOmega : float
        较差转动指数
    r0_rot : float
        参考半径
    period_days : float
        周期（天）

    返回:
    -------
    SpotSimulator
        simulator对象
    """
    from core.grid_tom import diskGrid

    grid = diskGrid(nr=nr, r_in=r_in, r_out=r_out)
    sim = SpotSimulator(grid,
                        inclination_rad=np.deg2rad(inclination_deg),
                        pOmega=pOmega,
                        r0_rot=r0_rot,
                        period_days=period_days)
    return sim


def create_test_spots() -> List[SpotConfig]:
    """创建测试用的spot配置"""
    spots = [
        SpotConfig(r=1.5, phi=0.0, amplitude=2.0, B_los=1000.0),
        SpotConfig(r=2.0, phi=np.pi, amplitude=1.5, B_los=-500.0),
        SpotConfig(r=3.0,
                   phi=np.pi / 2,
                   amplitude=-1.5,
                   spot_type='absorption'),
    ]
    return spots
