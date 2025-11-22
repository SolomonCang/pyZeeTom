"""
tomography_inversion.py - MEM 反演工作流执行引擎

本模块实现 MEM（最大熵方法）反演工作流，包括：
  - 反演迭代控制
  - 收敛判定
  - 中间结果保存
  - 反演结果聚合
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

from core.tomography_config import InversionConfig
from core.tomography_result import InversionResult, ForwardModelResult
from core.mem_tomography import MEMTomographyAdapter
from core.mem_iteration_manager import create_iteration_manager_from_config
from core.disk_geometry_integrator import VelspaceDiskIntegrator

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 参数编码/解码工具函数（用于 MEM 优化器）
# ════════════════════════════════════════════════════════════════════════════


def _pack_magnetic_parameters(B_los: np.ndarray, B_perp: np.ndarray,
                              chi: np.ndarray) -> np.ndarray:
    """
    将磁场参数打包为一维向量。
    
    顺序：[Blos[0..Npix-1], Bperp[0..Npix-1], chi[0..Npix-1]]
    
    Parameters
    ----------
    B_los : np.ndarray
        视向磁场 (Npix,)
    B_perp : np.ndarray
        垂直平面内磁场强度 (Npix,)
    chi : np.ndarray
        磁场方位角 (Npix,)，单位为弧度
    
    Returns
    -------
    np.ndarray
        打包后的参数向量 (3*Npix,)
    """
    return np.concatenate([B_los, B_perp, chi])


def _unpack_magnetic_parameters(
        params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从一维向量解包磁场参数。
    
    Parameters
    ----------
    params : np.ndarray
        打包的参数向量 (3*Npix,)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (B_los, B_perp, chi)，各为 (Npix,) 形状
    """
    npix = len(params) // 3
    if len(params) != 3 * npix:
        raise ValueError(f"参数向量长度 {len(params)} 不能整除 3")
    return params[:npix], params[npix:2 * npix], params[2 * npix:]


def _adaptive_delta(param_value: float, scale: float = 1e-3) -> float:
    """
    计算自适应微分步长。
    
    Parameters
    ----------
    param_value : float
        参数值
    scale : float
        相对步长因子（默认 1e-3）
    
    Returns
    -------
    float
        自适应步长
    """
    abs_val = abs(float(param_value))
    delta = scale * max(abs_val, 0.1)
    return float(delta)


def _compute_response_matrix_analytical(integrator: Any,
                                        B_los: np.ndarray,
                                        B_perp: np.ndarray,
                                        chi: np.ndarray,
                                        config: InversionConfig,
                                        base_spectrum_parts: List[np.ndarray],
                                        delta_scale: float = 1e-3,
                                        method: str = 'numerical',
                                        verbose: bool = False) -> np.ndarray:
    """
    计算响应矩阵（支持数值微分或解析导数）。
    
    Phase A: 使用数值微分（中心差分）计算模型对磁场参数的偏导数。
    Phase B: 将扩展为支持解析导数计算。
    
    Parameters
    ----------
    integrator : VelspaceDiskIntegrator
        速度空间积分器实例
    B_los, B_perp, chi : np.ndarray
        当前磁场参数，形状 (Npix,)
    config : InversionConfig
        反演配置
    base_spectrum_parts : List[np.ndarray]
        基准频谱分量列表 [I, (V), (Q, U)]
    delta_scale : float
        相对微分步长因子（默认 1e-3）
    method : str
        计算方法 ('numerical' 或 'analytical')
    verbose : bool
        详细输出
    
    Returns
    -------
    np.ndarray
        响应矩阵，形状 (Ndata, 3*Npix)
    """
    npix = len(B_los)

    # 构建基准全向量以确定 Ndata
    base_full = np.concatenate(base_spectrum_parts)
    ndata = len(base_full)

    if verbose:
        logger.info(f"计算响应矩阵 ({method}): Npix={npix}, Ndata={ndata}")

    # 初始化响应矩阵
    Resp = np.zeros((ndata, 3 * npix), dtype=float)

    if method == 'analytical':
        # TODO: Phase B - 实现解析导数
        logger.warning("解析导数尚未实现，回退到数值微分")
        pass

    # 辅助函数：构建全向量
    def _build_full_vector(intg):
        parts = [intg.I]
        if config.enable_v:
            parts.append(intg.V)
        if config.enable_qu:
            parts.append(intg.Q)
            parts.append(intg.U)
        return np.concatenate(parts)

    # ════════════════════════════════════════════════════════════════════════════
    # Step 1: 数值微分 - B_los 导数
    # ════════════════════════════════════════════════════════════════════════════

    for ipix in range(npix):
        delta = _adaptive_delta(B_los[ipix], scale=delta_scale)

        # +Δ 方向
        B_los_plus = B_los.copy()
        B_los_plus[ipix] += delta
        try:
            integrator.compute_spectrum(B_los=B_los_plus)
            spec_plus = _build_full_vector(integrator)
        except Exception as e:
            if verbose:
                logger.warning(f"B_los +Δ 计算失败 (像素 {ipix}): {e}")
            spec_plus = base_full

        # -Δ 方向
        B_los_minus = B_los.copy()
        B_los_minus[ipix] -= delta
        try:
            integrator.compute_spectrum(B_los=B_los_minus)
            spec_minus = _build_full_vector(integrator)
        except Exception as e:
            if verbose:
                logger.warning(f"B_los -Δ 计算失败 (像素 {ipix}): {e}")
            spec_minus = base_full

        # 中心差分
        deriv = (spec_plus - spec_minus) / (2.0 * delta + 1e-30)
        Resp[:, ipix] = deriv

        # 恢复 B_los (虽然 compute_spectrum 会更新 geom，但为了安全起见，循环开始时应重置)
        # 但由于我们每次都传入完整的 B_los 数组给 compute_spectrum，
        # 下一次循环会覆盖，所以这里只需要确保最后恢复即可。
        # 为了保持循环内的状态一致性，我们在每次迭代后恢复 integrator 的状态是比较安全的，
        # 或者在每次调用 compute_spectrum 时都传入基于原始 B_los 修改后的数组（如上所示）。
        # 这里采用的是传入修改后的完整数组，所以 integrator 内部状态会变，
        # 但下一次循环使用的是原始 B_los 的拷贝进行修改，所以逻辑是正确的。

        if verbose and (ipix + 1) % max(1, npix // 5) == 0:
            logger.debug(f"  B_los 导数: {ipix + 1}/{npix} 完成")

    # 恢复 integrator 到基准状态
    integrator.compute_spectrum(B_los=B_los)

    # ════════════════════════════════════════════════════════════════════════════
    # Step 2: 数值微分 - B_perp 导数
    # ════════════════════════════════════════════════════════════════════════════

    for ipix in range(npix):
        delta = _adaptive_delta(B_perp[ipix], scale=delta_scale)

        # +Δ 方向
        B_perp_plus = B_perp.copy()
        B_perp_plus[ipix] += delta
        try:
            integrator.compute_spectrum(B_perp=B_perp_plus)
            spec_plus = _build_full_vector(integrator)
        except Exception as e:
            if verbose:
                logger.warning(f"B_perp +Δ 计算失败 (像素 {ipix}): {e}")
            spec_plus = base_full

        # -Δ 方向
        B_perp_minus = B_perp.copy()
        B_perp_minus[ipix] -= delta
        try:
            integrator.compute_spectrum(B_perp=B_perp_minus)
            spec_minus = _build_full_vector(integrator)
        except Exception as e:
            if verbose:
                logger.warning(f"B_perp -Δ 计算失败 (像素 {ipix}): {e}")
            spec_minus = base_full

        deriv = (spec_plus - spec_minus) / (2.0 * delta + 1e-30)
        Resp[:, npix + ipix] = deriv

        if verbose and (ipix + 1) % max(1, npix // 5) == 0:
            logger.debug(f"  B_perp 导数: {ipix + 1}/{npix} 完成")

    # 恢复 integrator 到基准状态
    integrator.compute_spectrum(B_perp=B_perp)

    # ════════════════════════════════════════════════════════════════════════════
    # Step 3: 数值微分 - chi 导数
    # ════════════════════════════════════════════════════════════════════════════

    for ipix in range(npix):
        delta = _adaptive_delta(chi[ipix], scale=delta_scale)

        # +Δ 方向
        chi_plus = chi.copy()
        chi_plus[ipix] += delta
        try:
            integrator.compute_spectrum(chi=chi_plus)
            spec_plus = _build_full_vector(integrator)
        except Exception as e:
            if verbose:
                logger.warning(f"chi +Δ 计算失败 (像素 {ipix}): {e}")
            spec_plus = base_full

        # -Δ 方向
        chi_minus = chi.copy()
        chi_minus[ipix] -= delta
        try:
            integrator.compute_spectrum(chi=chi_minus)
            spec_minus = _build_full_vector(integrator)
        except Exception as e:
            if verbose:
                logger.warning(f"chi -Δ 计算失败 (像素 {ipix}): {e}")
            spec_minus = base_full

        deriv = (spec_plus - spec_minus) / (2.0 * delta + 1e-30)
        Resp[:, 2 * npix + ipix] = deriv

        if verbose and (ipix + 1) % max(1, npix // 5) == 0:
            logger.debug(f"  chi 导数: {ipix + 1}/{npix} 完成")

    # 恢复 integrator 到基准状态
    integrator.compute_spectrum(chi=chi)

    if verbose:
        logger.info(f"响应矩阵计算完成: {Resp.shape}")
        logger.debug(f"  导数范围: [{np.min(Resp):.3e}, {np.max(Resp):.3e}]")
        logger.debug(f"  导数 RMS: {np.sqrt(np.mean(Resp**2)):.3e}")

    return Resp


def run_mem_inversion(config: InversionConfig,
                      forward_results: Optional[
                          List[ForwardModelResult]] = None,
                      verbose: bool = False) -> List[InversionResult]:
    """运行完整的 MEM 反演工作流
    
    这是反演工作流的主入口，执行以下步骤：
    1. 验证配置的完整性和一致性
    2. 初始化反演状态和历史记录
    3. 执行迭代反演
    4. 检查收敛条件
    5. 返回最终反演结果
    
    Parameters
    ----------
    config : InversionConfig
        反演配置对象，包含所有必需的参数
    forward_results : Optional[List[ForwardModelResult]]
        用于初始化或参考的正演结果
    verbose : bool, optional
        详细输出标志，默认为 False
    
    Returns
    -------
    List[InversionResult]
        每个相位的反演结果列表
    
    Raises
    ------
    ValueError
        配置对象验证失败
    RuntimeError
        反演过程中发生错误
    """

    # 验证配置
    try:
        config.validate()
    except (ValueError, AssertionError) as e:
        logger.error(f"配置验证失败: {e}")
        raise

    if verbose:
        logger.info(config.create_summary())

    # 初始化反演结果列表
    results = []

    # 对每个观测相位进行反演
    for phase_idx, obs_data in enumerate(config.obsSet):
        try:
            if verbose:
                logger.info(f"处理反演相位 {phase_idx}/{len(config.obsSet)}")

            result = _invert_single_phase(
                phase_idx=phase_idx,
                obs_data=obs_data,
                config=config,
                verbose=verbose,
            )

            results.append(result)

            if verbose:
                logger.info(f"  ✓ 相位 {phase_idx} 反演完成: "
                            f"χ²={result.final_chi2:.6e}, "
                            f"收敛={result.converged}")

        except Exception as e:
            logger.error(f"相位 {phase_idx} 反演失败: {e}")
            if verbose:
                raise
            continue

    if len(results) == 0:
        raise RuntimeError("所有相位反演都失败了")

    logger.info(f"✓ 成功反演 {len(results)}/{len(config.obsSet)} 个相位")

    return results


def _invert_single_phase(phase_idx: int,
                         obs_data: Any,
                         config: InversionConfig,
                         verbose: bool = False) -> InversionResult:
    """对单个观测相位进行 MEM 反演
    
    Parameters
    ----------
    phase_idx : int
        相位索引
    obs_data : ObservationProfile
        观测数据对象
    config : InversionConfig
        反演配置对象
    verbose : bool, optional
        详细输出
    
    Returns
    -------
    InversionResult
        该相位的反演结果
    """

    # 验证观测数据
    if not hasattr(obs_data, 'specI') or obs_data.specI is None:
        raise ValueError("观测数据缺少 specI")

    if not hasattr(obs_data, 'sigma') or obs_data.sigma is None:
        raise ValueError("观测数据缺少误差信息")

    # 初始化磁场模型
    # 使用自定义初始模型或配置中的初始值
    if config.initial_B_los is not None:
        B_los = config.initial_B_los.copy()
    else:
        B_los = config.B_los_init.copy()

    if config.initial_B_perp is not None:
        B_perp = config.initial_B_perp.copy()
    else:
        B_perp = config.B_perp_init.copy()

    if config.initial_chi is not None:
        chi = config.initial_chi.copy()
    else:
        chi = config.chi_init.copy()

    # ════════════════════════════════════════════════════════════════════════════
    # 初始化积分器 (新增)
    # ════════════════════════════════════════════════════════════════════════════

    # 计算速度网格
    c_kms = 2.99792458e5
    v_grid = (obs_data.wl - config.lineData.wl0) / config.lineData.wl0 * c_kms

    # 更新几何对象的磁场参数
    config.geom.B_los = B_los
    config.geom.B_perp = B_perp
    config.geom.chi = chi

    # 创建积分器实例
    integrator = VelspaceDiskIntegrator(
        geom=config.geom,
        wl0_nm=config.lineData.wl0,
        v_grid=v_grid,
        line_model=config.line_model,
        line_area=config.line_area,
        inst_fwhm_kms=config.inst_fwhm_kms,
        normalize_continuum=config.normalize_continuum,
        # 动力学参数从 config 传递
        disk_v0_kms=config.disk_v0_kms,
        disk_power_index=config.disk_power_index,
        disk_r0=config.disk_r0,
        inner_slowdown_mode=config.inner_slowdown_mode,
        inner_profile=config.inner_profile,
        inner_edge_blend=config.inner_edge_blend,
        inner_mode=config.inner_mode,
        inner_alpha=config.inner_alpha,
        inner_beta=config.inner_beta,
        obs_phase=getattr(obs_data, 'phase', 0.0))

    # ════════════════════════════════════════════════════════════════════════════
    # MEM 适配器初始化（新增）
    # ════════════════════════════════════════════════════════════════════════════

    npix = len(B_los)

    # 创建 MEM 适配器实例
    adapter = MEMTomographyAdapter(
        fit_brightness=False,
        fit_magnetic=True,
        entropy_weights_blos=np.ones(npix),  # 均匀权重（可通过网格面积优化）
        entropy_weights_bperp=np.ones(npix),
        entropy_weights_chi=np.ones(npix),
        default_blos=config.B_los_default
        if hasattr(config, 'B_los_default') else 0.1,
        default_bperp=config.B_perp_default
        if hasattr(config, 'B_perp_default') else 0.1,
        default_chi=0.0)

    # 熵计算的参数字典
    entropy_params = {
        'npix': npix,
        'n_blos': npix,
        'n_bperp': npix,
        'n_chi': npix
    }

    # ════════════════════════════════════════════════════════════════════════════
    # 数据准备：拼接 I, V, Q, U
    # ════════════════════════════════════════════════════════════════════════════

    # 确定启用的 Stokes 分量
    enable_v = config.enable_v
    enable_qu = config.enable_qu

    # 构建观测向量列表
    obs_parts = [obs_data.specI]
    sig_parts = [obs_data.sigma]

    if enable_v:
        specV = getattr(obs_data, 'specV', np.zeros_like(obs_data.specI))
        obs_parts.append(specV)
        sig_parts.append(obs_data.sigma)  # 假设 V 的误差与 I 相同

    if enable_qu:
        specQ = getattr(obs_data, 'specQ', np.zeros_like(obs_data.specI))
        specU = getattr(obs_data, 'specU', np.zeros_like(obs_data.specI))
        obs_parts.append(specQ)
        obs_parts.append(specU)
        sig_parts.append(obs_data.sigma)
        sig_parts.append(obs_data.sigma)

    # 拼接为一维向量
    Data = np.concatenate(obs_parts)
    Sigma = np.concatenate(sig_parts)

    if verbose:
        logger.info(f"数据准备: Ndata={len(Data)} (I={len(obs_data.specI)}, "
                    f"V={'on' if enable_v else 'off'}, "
                    f"QU={'on' if enable_qu else 'off'})")

    # 初始化迭代历史
    max_iters = config.max_iterations or config.num_iterations

    # 创建 IterationManager (集中式迭代管理)
    manager = create_iteration_manager_from_config({
        'max_iterations':
        max_iters,
        'convergence_rel_tol':
        config.convergence_threshold,
        'stall_threshold':
        3,
        'with_iteration_history':
        True,
        'with_progress_monitor':
        verbose,
    })

    # 执行迭代反演
    while True:
        should_stop, reason = manager.should_stop()
        if should_stop:
            break

        manager.start_iteration()
        iteration = manager.iteration

        # 执行一步反演
        try:
            chi2, entropy, regularization = _perform_inversion_step(
                B_los=B_los,
                B_perp=B_perp,
                chi=chi,
                obs_data_vector=Data,
                obs_error_vector=Sigma,
                config=config,
                iteration=iteration,
                adapter=adapter,
                entropy_params=entropy_params,
                integrator=integrator,
                verbose=verbose,
            )
        except Exception as e:
            logger.warning(f"迭代 {iteration} 失败: {e}")
            break

        # 记录历史 (统一使用 IterationManager)
        manager.record_iteration(
            chi2=chi2,
            entropy=entropy,
            gradient_norm=None,
            grad_S_norm=0.0,  # 熵梯度范数（暂未实现）
            grad_C_norm=0.0,  # 约束梯度范数（暂未实现）
        )

        if verbose and iteration % max(1, max_iters // 10) == 0:
            logger.debug(f"  迭代 {iteration}: χ²={chi2:.6e}, "
                         f"S={entropy:.6e}")

        # 检查收敛条件 (统一使用 IterationManager)
        should_stop, reason = manager.should_stop(chi2)
        if should_stop:
            if verbose:
                logger.info(f"  收敛于迭代 {iteration}: {reason}")
            break

    # 从 IterationManager 中提取历史
    if manager.iteration_history is not None:
        hist_data = manager.iteration_history.get_history()
        chi2_history = hist_data.get('chi2', [])
        entropy_history = hist_data.get('entropy', [])
        regularization_history = hist_data.get('regularization', [])
    else:
        # 降级方案：返回空历史
        chi2_history = []
        entropy_history = []
        regularization_history = []

    # 获取收敛信息
    summary = manager.get_summary()
    stop_reason = summary.get('stop_reason', 'unknown')
    converged = (stop_reason == 'convergence'
                 or stop_reason == 'convergence_stall')

    result = InversionResult(
        B_los_final=B_los,
        B_perp_final=B_perp,
        chi_final=chi,
        iterations_completed=manager.iteration,
        chi2_history=chi2_history,
        entropy_history=entropy_history,
        regularization_history=regularization_history,
        converged=converged,
        convergence_reason=stop_reason,
        final_chi2=chi2_history[-1] if chi2_history else 0.0,
        final_entropy=entropy_history[-1] if entropy_history else 0.0,
        phase_index=phase_idx,
        pol_channels=["I+V"],  # 当前支持 I+V
    )

    # 计算拟合质量
    result.fit_quality = _compute_fit_quality(
        B_los=B_los,
        B_perp=B_perp,
        chi=chi,
        obs_spectrum=obs_data.specI,
        obs_error=obs_data.sigma,
        integrator=integrator,
    )

    return result


def _perform_inversion_step(
        B_los: np.ndarray,
        B_perp: np.ndarray,
        chi: np.ndarray,
        obs_data_vector: np.ndarray,
        obs_error_vector: np.ndarray,
        config: InversionConfig,
        iteration: int,
        adapter: MEMTomographyAdapter,
        entropy_params: Dict[str, Any],
        integrator: Any,
        verbose: bool = False) -> Tuple[float, float, float]:
    """执行一步 MEM 反演（使用 Skilling & Bryan 算法）
    
    Parameters
    ----------
    B_los : np.ndarray
        当前视向磁场
    B_perp : np.ndarray
        当前垂直磁场
    chi : np.ndarray
        当前磁场方向角
    obs_data_vector : np.ndarray
        观测数据向量 (I, [V, Q, U])
    obs_error_vector : np.ndarray
        观测误差向量
    config : InversionConfig
        反演配置
    iteration : int
        当前迭代数
    adapter : MEMTomographyAdapter
        MEM 适配器实例
    entropy_params : Dict[str, Any]
        熵计算参数
    integrator : VelspaceDiskIntegrator
        积分器实例
    verbose : bool
        详细输出
    
    Returns
    -------
    Tuple[float, float, float]
        (chi2, entropy, regularization)
    """

    # ════════════════════════════════════════════════════════════════════════════
    # Step 0: 准备合成数据和响应矩阵
    # ════════════════════════════════════════════════════════════════════════════

    # 更新积分器状态并计算光谱
    integrator.compute_spectrum(B_los=B_los, B_perp=B_perp, chi=chi)

    # 构建合成数据向量
    syn_parts = [integrator.I]
    if config.enable_v:
        syn_parts.append(integrator.V)
    if config.enable_qu:
        syn_parts.append(integrator.Q)
        syn_parts.append(integrator.U)
    synthetic_spectrum = np.concatenate(syn_parts)

    # 计算响应矩阵 (使用数值微分)
    response_matrix = _compute_response_matrix_analytical(
        integrator=integrator,
        B_los=B_los,
        B_perp=B_perp,
        chi=chi,
        config=config,
        base_spectrum_parts=syn_parts,
        method='numerical',  # Phase A: 数值微分
        verbose=verbose)

    # ════════════════════════════════════════════════════════════════════════════
    # Step 1: 打包参数向量
    # ════════════════════════════════════════════════════════════════════════════

    Image = _pack_magnetic_parameters(B_los, B_perp, chi)

    # ════════════════════════════════════════════════════════════════════════════
    # Step 2: 调用 MEM 迭代器（Skilling & Bryan 算法）
    # ════════════════════════════════════════════════════════════════════════════

    try:
        entropy, chi2, test, Image_new = adapter.optimizer.iterate(
            Image=Image,
            Fmodel=synthetic_spectrum,  # 合成数据向量
            Data=obs_data_vector,  # 观测数据向量
            sig2=obs_error_vector**2,  # 噪声方差
            Resp=response_matrix,  # 响应矩阵
            weights=np.ones_like(Image),  # 均匀权重
            entropy_params=entropy_params,
            fixEntropy=0,  # 拟合到 χ² 目标
            targetAim=None)

    except Exception as e:
        if verbose:
            logger.warning(f"MEM 迭代在步骤 {iteration} 失败: {e}")
        # 降级到简单计算（不更新参数）
        entropy = _compute_entropy(B_los, B_perp, config)
        chi2 = _compute_chi2(obs_data_vector, obs_error_vector,
                             synthetic_spectrum)
        regularization = _compute_regularization(B_los, B_perp, chi, config)
        return chi2, entropy, regularization

    # ════════════════════════════════════════════════════════════════════════════
    # Step 3: 更新参数（直接从 Image_new 写回）
    # ════════════════════════════════════════════════════════════════════════════

    B_los_new, B_perp_new, chi_new = _unpack_magnetic_parameters(Image_new)
    B_los[:] = B_los_new
    B_perp[:] = B_perp_new
    chi[:] = chi_new

    # ════════════════════════════════════════════════════════════════════════════
    # Step 4: 计算正则化项（如果需要）
    # ════════════════════════════════════════════════════════════════════════════

    regularization = _compute_regularization(B_los, B_perp, chi, config)

    if verbose and iteration % max(1, 5) == 0:
        logger.debug(f"  MEM 迭代 {iteration}: χ²={chi2:.3e}, S={entropy:.3e}, "
                     f"test={test:.3e}, reg={regularization:.3e}")

    return chi2, entropy, regularization


def _check_convergence(chi2_history: List[float], entropy_history: List[float],
                       config: InversionConfig, iteration: int) -> bool:
    """检查是否满足收敛条件
    
    Parameters
    ----------
    chi2_history : List[float]
        χ² 历史
    entropy_history : List[float]
        熵值历史
    config : InversionConfig
        反演配置
    iteration : int
        当前迭代数
    
    Returns
    -------
    bool
        是否收敛
    """

    if len(chi2_history) < 2:
        return False

    # 检查 χ² 变化
    chi2_change = abs(chi2_history[-1] -
                      chi2_history[-2]) / (abs(chi2_history[-1]) + 1e-10)

    if chi2_change < config.convergence_threshold:
        return True

    # 检查是否达到最大迭代数
    max_iters = config.max_iterations or config.num_iterations
    if iteration >= max_iters - 1:
        return True

    return False


def _is_converged(chi2_history: List[float], config: InversionConfig) -> bool:
    """判断是否已收敛
    
    Parameters
    ----------
    chi2_history : List[float]
        χ² 历史
    config : InversionConfig
        反演配置
    
    Returns
    -------
    bool
        是否收敛
    """

    if len(chi2_history) < 2:
        return False

    chi2_change = abs(chi2_history[-1] -
                      chi2_history[-2]) / (abs(chi2_history[-1]) + 1e-10)

    return chi2_change < config.convergence_threshold


def _get_convergence_reason(chi2_history: List[float],
                            config: InversionConfig) -> str:
    """获取收敛原因的描述
    
    Parameters
    ----------
    chi2_history : List[float]
        χ² 历史
    config : InversionConfig
        反演配置
    
    Returns
    -------
    str
        收敛原因描述
    """

    if not chi2_history:
        return "未开始"

    max_iters = config.max_iterations or config.num_iterations

    if len(chi2_history) >= max_iters:
        return "达到最大迭代数"

    if _is_converged(chi2_history, config):
        return "χ² 变化低于阈值"

    return "未知"


def _compute_chi2(obs_spectrum: np.ndarray, obs_error: np.ndarray,
                  predicted_spectrum: np.ndarray) -> float:
    """计算 χ² 值
    
    Parameters
    ----------
    obs_spectrum : np.ndarray
        观测频谱
    obs_error : np.ndarray
        观测误差
    predicted_spectrum : np.ndarray
        预测频谱
    
    Returns
    -------
    float
        χ² 值
    """

    residuals = (obs_spectrum - predicted_spectrum) / obs_error
    chi2 = float(np.sum(residuals**2))

    return chi2


def _compute_entropy(B_los: np.ndarray, B_perp: np.ndarray,
                     config: InversionConfig) -> float:
    """计算熵值
    
    Parameters
    ----------
    B_los : np.ndarray
        视向磁场
    B_perp : np.ndarray
        垂直磁场
    config : InversionConfig
        反演配置
    
    Returns
    -------
    float
        熵值
    """

    # 简单的熵计算：磁场强度的平方和
    entropy = float(np.sum(B_los**2 + B_perp**2))

    return entropy


def _compute_regularization(B_los: np.ndarray, B_perp: np.ndarray,
                            chi: np.ndarray, config: InversionConfig) -> float:
    """计算正则化项
    
    Parameters
    ----------
    B_los : np.ndarray
        视向磁场
    B_perp : np.ndarray
        垂直磁场
    chi : np.ndarray
        磁场方向角
    config : InversionConfig
        反演配置
    
    Returns
    -------
    float
        正则化项值
    """

    # 使用配置中的权重计算正则化项
    term_entropy = config.entropy_weight * np.sum(B_los**2 + B_perp**2)
    term_smoothness = config.smoothness_weight * _compute_smoothness(
        B_los, B_perp)
    reg = term_entropy + term_smoothness

    return float(reg)


def _compute_smoothness(B_los: np.ndarray, B_perp: np.ndarray) -> float:
    """计算磁场的平滑性度量
    
    Parameters
    ----------
    B_los : np.ndarray
        视向磁场
    B_perp : np.ndarray
        垂直磁场
    
    Returns
    -------
    float
        平滑性度量值
    """

    # 计算相邻像素的差异
    smoothness = float(np.sum(np.diff(B_los)**2 + np.diff(B_perp)**2))

    return smoothness


def _compute_fit_quality(
    B_los: np.ndarray,
    B_perp: np.ndarray,
    chi: np.ndarray,
    obs_spectrum: np.ndarray,
    obs_error: np.ndarray,
    integrator: Any,
) -> Dict[str, float]:
    """计算拟合质量指标
    
    Parameters
    ----------
    B_los : np.ndarray
        最终视向磁场
    B_perp : np.ndarray
        最终垂直磁场
    chi : np.ndarray
        最终磁场方向角
    obs_spectrum : np.ndarray
        观测频谱
    obs_error : np.ndarray
        观测误差
    integrator : Any
        积分器实例
    
    Returns
    -------
    dict
        拟合质量指标字典
    """

    # 计算合成频谱
    integrator.compute_spectrum(B_los=B_los, B_perp=B_perp, chi=chi)
    predicted = integrator.I

    # 计算 RMS 残差
    residuals = obs_spectrum - predicted
    rms_residual = float(np.sqrt(np.mean(residuals**2)))
    max_residual = float(np.max(np.abs(residuals)))

    # 计算单点 χ²
    nchi2_per_point = float(
        np.mean(((obs_spectrum - predicted) / obs_error)**2))

    return {
        'rms_residual': rms_residual,
        'max_residual': max_residual,
        'nchi2_per_point': nchi2_per_point,
    }


def get_inversion_summary(results: List[InversionResult]) -> Dict[str, Any]:
    """生成反演结果的统计摘要
    
    Parameters
    ----------
    results : List[InversionResult]
        反演结果列表
    
    Returns
    -------
    dict
        包含统计信息的摘要字典
    """

    if not results:
        return {}

    summary = {
        'num_phases': len(results),
        'converged_count': sum(1 for r in results if r.converged),
        'phase_details': [],
    }

    # 收集每个相位的统计
    for result in results:
        result.validate()
        mag_stats = result.get_magnetic_field_stats()
        opt_metrics = result.get_optimization_metrics()

        phase_info = {
            'phase_index': result.phase_index,
            'converged': result.converged,
            'iterations': opt_metrics['iterations'],
            'final_chi2': opt_metrics['final_chi2'],
            'B_los_rms': mag_stats['B_los']['rms'],
            'B_perp_rms': mag_stats['B_perp']['rms'],
        }

        summary['phase_details'].append(phase_info)

    # 计算全局统计
    all_iterations = [r.iterations_completed for r in results]
    all_chi2 = [r.final_chi2 for r in results]

    summary['global'] = {
        'avg_iterations': float(np.mean(all_iterations)),
        'avg_final_chi2': float(np.mean(all_chi2)),
        'convergence_rate': float(summary['converged_count'] / len(results)),
    }

    return summary
