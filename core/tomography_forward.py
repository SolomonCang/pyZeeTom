"""
tomography_forward.py - 正演工作流执行引擎

实现从参数对象到正演结果的完整工作流。

主入口：run_forward_synthesis_with_physical_model()
"""

from typing import List, Any
import numpy as np
import logging

from core.tomography_config import ForwardModelConfig
from core.tomography_result import ForwardModelResult
from core.physical_model import create_physical_model
from core.disk_geometry_integrator import VelspaceDiskIntegrator
import core.mainFuncs as mf

logger = logging.getLogger(__name__)


def run_forward_synthesis(par,
                          obsSet: List[Any],
                          line_model: Any,
                          line_data: Any = None,
                          output_dir: str = './output',
                          verbose: bool = True) -> List[ForwardModelResult]:
    """正演工作流主入口：从参数对象执行完整的频谱合成
    
    该函数实现以下步骤：
    1. 创建物理模型（网格、几何、磁场）
    2. 创建正演配置
    3. 对每个观测相位进行频谱合成
    4. 返回合成结果
    
    Parameters
    ----------
    par : readParamsTomog
        参数对象，包含所有配置信息
    obsSet : List[ObservationProfile]
        观测数据集列表
    line_model : BaseLineModel
        谱线模型对象（必需）
    line_data : LineData, optional
        谱线参数数据。若为 None，则从 line_model 中提取
    output_dir : str, default='./output'
        输出目录
    verbose : bool, default=True
        详细输出标志
    
    Returns
    -------
    List[ForwardModelResult]
        每个相位的正演结果列表
    
    Examples
    --------
    >>> par = readParamsTomog('input/params_tomog.txt')
    >>> obsSet = obsProfSetInRange('input/inSpec/', 'obs_phase_*.spec')
    >>> line_model = GaussianZeemanWeakLineModel(LineData('input/lines.txt'))
    >>> results = run_forward_synthesis_with_physical_model(
    ...     par, obsSet, line_model, verbose=True)
    """
    if verbose:
        logger.info("[Forward] 创建物理模型...")

    # Extract v_grid from the first observation if available
    # This ensures the synthetic spectrum matches the observation grid (e.g. -300 to 300 km/s)
    v_grid = None
    if obsSet and len(obsSet) > 0:
        first_obs = obsSet[0]
        if hasattr(first_obs, 'wl'):
            # Check if observation is already in velocity domain (LSD)
            is_velocity_domain = False
            if hasattr(first_obs, 'profile_type') and (
                    'lsd' in first_obs.profile_type.lower()
                    or 'velocity' in first_obs.profile_type.lower()):
                is_velocity_domain = True

            # Also check value range: if values are small (e.g. -1000 to 1000) and contain negative values, likely velocity
            wl_vals = np.asarray(first_obs.wl, dtype=float)
            if np.max(np.abs(wl_vals)) < 3000 and np.min(wl_vals) < 0:
                is_velocity_domain = True

            if is_velocity_domain:
                v_grid = wl_vals
                if verbose:
                    logger.info(
                        f"[Forward] Using velocity grid from observation (LSD/Velocity): {v_grid.min():.1f} to {v_grid.max():.1f} km/s"
                    )
            else:
                # Convert wavelength to velocity
                # v = c * (wl - wl0) / wl0
                c_kms = 2.99792458e5
                wl0 = line_data.wl0 if line_data and hasattr(line_data,
                                                             'wl0') else 656.3
                v_grid = c_kms * (wl_vals - wl0) / wl0
                if verbose:
                    logger.info(
                        f"[Forward] Using velocity grid from observation (Wavelength converted): {v_grid.min():.1f} to {v_grid.max():.1f} km/s"
                    )

    # 创建物理模型
    # 注意：这里创建的 phys_model 已包含 VelspaceDiskIntegrator 实例 (phys_model.integrator)
    phys_model = create_physical_model(
        par,
        wl0_nm=(line_data.wl0
                if line_data and hasattr(line_data, 'wl0') else 656.3),
        v_grid=v_grid,
        line_model=line_model,
        verbose=2 if verbose else 0)
    phys_model.validate()

    # 验证 integrator 已正确创建
    if phys_model.integrator is None:
        raise RuntimeError("物理模型的 integrator 创建失败")

    # 提取谱线数据（如果未提供）
    if line_data is None and hasattr(line_model, 'ld'):
        line_data = line_model.ld

    # 创建正演配置
    if verbose:
        logger.info("[Forward] 创建正演配置...")

    config = ForwardModelConfig(par=par,
                                obsSet=obsSet,
                                lineData=line_data,
                                geom=phys_model.geometry,
                                line_model=line_model,
                                output_dir=output_dir,
                                save_intermediate=False,
                                verbose=1 if verbose else 0)
    config.validate()

    # 执行正演合成
    if verbose:
        logger.info(f"[Forward] 开始合成 {len(obsSet)} 个观测相位...")

    results = []
    for phase_idx, obs_data in enumerate(obsSet):
        try:
            if verbose:
                logger.info(f"[Forward] 处理相位 {phase_idx}/{len(obsSet)}")

            # 从观测数据提取波长/速度网格
            if hasattr(obs_data, 'wl'):
                # obs_v_grid = np.asarray(obs_data.wl, dtype=float)  # 观测波长
                pass  # 观测波长由 obs_data 保持，integrator 使用合理的 v_grid
            else:
                raise ValueError("观测数据缺少波长/速度信息")

            # 计算当前观测的相位
            current_phase = 0.0
            current_hjd = 0.0
            if hasattr(par, 'jDates') and len(par.jDates) > phase_idx:
                current_hjd = par.jDates[phase_idx]
                if hasattr(par, 'jDateRef') and hasattr(par, 'period'):
                    current_phase = mf.compute_phase_from_jd(
                        current_hjd, par.jDateRef, par.period)
            else:
                # Fallback: evenly distributed phases
                current_phase = phase_idx / len(obsSet) if len(
                    obsSet) > 0 else 0.0

            # 计算观测相位对应的旋转角
            # phi_obs = 2*pi * phase
            # 注意：这里 phase 是 0~1
            current_phase = par.phases[phase_idx] if hasattr(
                par, 'phases') and len(par.phases) > phase_idx else 0.0

            if verbose:
                print(
                    f"[Forward] Phase {phase_idx}: current_phase={current_phase:.4f}, phi_rot={2.0 * np.pi * current_phase:.4f}"
                )

            # 确保 integrator 使用正确的物理模型
            # Re-create integrator for this phase to ensure correct time evolution and computation
            # VelspaceDiskIntegrator computes spectrum in __init__
            integrator = VelspaceDiskIntegrator(
                geom=phys_model.geometry,
                wl0_nm=phys_model._wl0 if phys_model._wl0 else 656.3,
                line_model=phys_model._line_model,
                v_grid=phys_model.integrator.v,  # Reuse velocity grid
                inst_fwhm_kms=phys_model.integrator.inst_fwhm,
                time_phase=current_phase)

            # 获取 integrator 的实际 v_grid
            v_grid = integrator.v

            # 获取 Stokes 分量
            stokes_i = (integrator.I
                        if hasattr(integrator, 'I') else np.ones_like(v_grid))
            stokes_v = (integrator.V
                        if hasattr(integrator, 'V') else np.zeros_like(v_grid))
            stokes_q = (integrator.Q
                        if hasattr(integrator, 'Q') else np.zeros_like(v_grid))
            stokes_u = (integrator.U
                        if hasattr(integrator, 'U') else np.zeros_like(v_grid))

            # 创建结果对象
            # 注意：pol_channel 现在仅支持单一值 (I/V/Q/U)，默认为 V
            # 从 par.polChannels 获取当前观测的偏振通道
            pol_chan = 'V'
            if hasattr(par, 'polChannels') and len(
                    par.polChannels) > phase_idx:
                pol_chan = par.polChannels[phase_idx]

            result = ForwardModelResult(
                stokes_i=stokes_i,
                stokes_v=stokes_v,
                stokes_q=stokes_q,
                stokes_u=stokes_u,
                wavelength=v_grid,
                error=obs_data.sigma if hasattr(obs_data, 'sigma') else None,
                hjd=current_hjd,
                phase_index=phase_idx,
                pol_channel=pol_chan,
                model_name="forward_synthesis",
                integrator=integrator  # Attach integrator for save_geomodel
            )
            results.append(result)

            if verbose:
                logger.info(f"[Forward]   ✓ 相位 {phase_idx} 合成完成")
                # 注意：不在这里计算 chi2，因为观测波长网格可能与合成网格不同
                # chi2 计算应该在反演阶段进行，使用插值后的光谱

        except Exception as e:
            logger.error(f"[Forward] 相位 {phase_idx} 合成失败: {e}")
            if verbose:
                raise
            continue

    if len(results) == 0:
        raise RuntimeError("所有相位合成都失败了")

    if verbose:
        logger.info(f"[Forward] ✓ 成功合成 {len(results)}/{len(obsSet)} 个相位")

    return results
