"""
通用最大熵方法 (MEM) 优化引擎
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于 Skilling & Bryan (1984, MNRAS 211, 111) 算法框架

本模块包含**纯粹的数学优化算法**，与具体物理模型和参数化无关。
用户需要通过回调函数自定义：
  - 计算熵及其导数 (get_s_grads_callback)
  - 计算约束统计量及导数 (get_c_gradc_callback)
  - 边界约束处理 (boundary_constraint_callback)

特点：
  ✓ 支持向量化操作，高效计算
  ✓ 灵活的熵形式（通过回调自定义）
  ✓ 强健的数值优化控制
  ✓ 完整的Skilling-Bryan算法实现

用法示例：见 mem_tomography.py
"""

import numpy as np
from scipy import linalg
from typing import Callable, Tuple, Dict, Any, Optional

# 导入优化工具（Week 2 优化）
try:
    from core.mem_optimization import StabilityMonitor
except ImportError:
    StabilityMonitor = None  # 降级处理


class MEMOptimizer:
    """
    通用MEM优化器。

    使用者需要提供回调函数定义熵形式和数据拟合。
    """

    def __init__(self,
                 compute_entropy_callback: Callable,
                 compute_constraint_callback: Callable,
                 boundary_constraint_callback: Optional[Callable] = None,
                 max_search_dirs: int = 10,
                 step_length_factor: float = 0.3,
                 convergence_tol: float = 1e-5):
        """
        初始化MEM优化器。

        参数：
            compute_entropy_callback:
                函数签名 -> (S0, gradS, gradgradS)
                计算熵、梯度、Hessian对角线
            compute_constraint_callback:
                函数签名 (Response, sig2) -> (C0, gradC)
                计算约束统计量（如χ²）及梯度
            boundary_constraint_callback:
                可选，签名 (Image, n1, n2, ntot, ...) -> Image_corrected
                应用边界约束（如正值性）
            max_search_dirs: 最大搜索方向数 (推荐3-10)
            step_length_factor: 参数空间步长因子 (0.1-0.5)
            convergence_tol: 收敛容差
        """
        self.compute_entropy = compute_entropy_callback
        self.compute_constraint = compute_constraint_callback
        self.apply_boundary_constraint = boundary_constraint_callback
        self.max_search_dirs = max_search_dirs
        self.step_length_factor = step_length_factor
        self.convergence_tol = convergence_tol

        # Week 2 优化：集成 StabilityMonitor (可选)
        self.stability_monitor = (StabilityMonitor(
            verbose=0) if StabilityMonitor is not None else None)

    def iterate(
        self,
        Image: np.ndarray,
        Fmodel: np.ndarray,
        Data: np.ndarray,
        sig2: np.ndarray,
        Resp: np.ndarray,
        weights: np.ndarray,
        entropy_params: Dict[str, Any],
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        ntot: Optional[int] = None,
        fixEntropy: int = 0,
        targetAim: Optional[float] = None
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        执行单次MEM迭代步骤。

        参数：
            Image: 当前参数向量 (长度 ntot)
            Fmodel: 当前模型预测 (长度 ndata)
            Data: 观测数据 (长度 ndata)
            sig2: 观测噪声方差 (长度 ndata)
            Resp: 响应矩阵 dModel/dImage (ndata x ntot)
            weights: 熵项权重 (长度 ntot)
            entropy_params: 熵计算所需的额外参数 dict
                如 {'defImg': 1.0, 'defIpm': 0.1, 'ffIMax': 1.0, ...}
            n1, n2, ntot: 参数分段索引
                若不提供则默认为全部一种熵形式
            fixEntropy: 0=拟合至χ²目标，1=拟合至熵目标
            targetAim: 目标χ²或目标熵值

        返回：
            (entropy, chi2, test, Image_new)
            entropy: 当前熵值
            chi2: 当前χ²值
            test: Skilling-Bryan收敛检验量
            Image_new: 更新后的参数向量
        """
        if ntot is None:
            ntot = len(Image)
        if n1 is None:
            n1 = ntot
        if n2 is None:
            n2 = n1

        # 计算约束统计量（χ²）
        C0, gradC = self.compute_constraint(Data, Fmodel, sig2, Resp)

        # 计算熵及导数
        S0, gradS, gradgradS = self.compute_entropy(Image, weights,
                                                    entropy_params, n1, n2,
                                                    ntot)

        # Skilling-Bryan 收敛检验量
        test = _get_test(gradC, gradS)

        # Week 2 优化：集成 StabilityMonitor 检查
        if self.stability_monitor is not None:
            stability_ok = self.stability_monitor.check_gradient(gradC, gradS)
            if not stability_ok:
                # 梯度存在问题（NaN、Inf、过大值）
                # 尝试恢复：返回当前状态，让上层处理
                import warnings as _warnings
                _warnings.warn(
                    "Stability check failed for gradients; may need to reduce step size"
                )

        # 计算搜索方向
        edir, nedir, gamma = _search_dir(ntot, self.max_search_dirs, Resp,
                                         sig2, -1.0 / gradgradS, gradC, gradS,
                                         gradgradS)

        # Week 2 优化：响应矩阵诊断
        if self.stability_monitor is not None:
            _ = self.stability_monitor.check_response_matrix(Resp)

        # 计算控制量
        Cmu, Smu = _get_cmu_smu(gradC, gradS, edir)

        # 计算步长参数 L02
        # 对于亮度图像，使用总强度；对于磁场等其他参数，使用范数
        if n1 > 0:
            # 有亮度参数，使用亮度部分的加权和
            Itot = np.sum(weights[:n1] * Image[:n1])
        else:
            # 无亮度参数，使用全部参数的范数作为尺度
            Itot = np.sqrt(np.sum((weights * Image)**2))
        L02 = _get_l0_squared(self.step_length_factor, Itot)

        alphaMin = _get_alpha_min(gamma)

        if fixEntropy == 1:
            # 拟合至目标熵
            Saim = targetAim if targetAim is not None else S0
            Saimq = _get_saim_quad(Saim, S0, gamma, Smu)
            xq = _control_entropy(S0, gamma, Cmu, Smu, Saimq, L02, alphaMin,
                                  self.convergence_tol)
        else:
            # 拟合至目标χ²
            chiAim = targetAim if targetAim is not None else C0
            Caimq = _get_caim_quad(chiAim, C0, gamma, Cmu)
            xq = _control_chi2(C0, gamma, Cmu, Smu, Caimq, L02, alphaMin,
                               self.convergence_tol)  # 更新图像向量
        Image_new = Image + np.dot(edir, xq)

        # 应用边界约束
        if self.apply_boundary_constraint is not None:
            Image_new = self.apply_boundary_constraint(Image_new, n1, n2, ntot,
                                                       entropy_params)

        return S0, C0, test, Image_new


# ============================================================================
# 内部辅助函数 (模块级，不应直接调用)
# ============================================================================


def _get_c_gradc(Data: np.ndarray, Fmodel: np.ndarray, sig2: np.ndarray,
                 Resp: np.ndarray) -> Tuple[float, np.ndarray]:
    """计算约束统计量χ²及其梯度。"""
    C0 = np.sum((Fmodel - Data)**2 / sig2)
    gradC = 2.0 * np.sum(Resp.T * (Fmodel - Data) / sig2, axis=1)
    return C0, gradC


def _search_dir(ntot: int, maxDir: int, Resp: np.ndarray, sig2: np.ndarray,
                fsupi: np.ndarray, gradC: np.ndarray, gradS: np.ndarray,
                gradgradS: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    生成线性独立的搜索方向。

    基于 Skilling & Bryan Eq. 20，生成最多maxDir个正交化搜索方向。
    
    Week 2 优化: 改进的向量化实现
      • 向量化范数计算（避免循环）
      • 批量矩阵操作（加速 Hessian 向量积）
      • 数值稳定性改进
    """
    edir = np.zeros((ntot, maxDir))

    # 第1搜索方向: e_1 = f(grad(C))
    edir[:, 0] = gradC * fsupi
    e1_norm_sq = np.dot(edir[:, 0], edir[:, 0])

    if e1_norm_sq > 0.0:
        edir[:, 0] /= np.sqrt(e1_norm_sq)
        err_gradCis0 = 0
    else:
        err_gradCis0 = 1

    if maxDir == 1:
        return edir[:, :1], 1, np.array([1.0])

    # 第2搜索方向: e_2 = f(grad(S))
    edir[:, 1] = gradS * fsupi
    e2_norm_sq = np.dot(edir[:, 1], edir[:, 1])

    if e2_norm_sq > 0.0:
        edir[:, 1] /= np.sqrt(e2_norm_sq)
        err_gradSis0 = 0
    else:
        err_gradSis0 = 1

    if (err_gradCis0 == 1) and (err_gradSis0 == 1):
        raise ValueError(
            "Both f(gradS) and f(gradC) are zero: problem is unconstrained")

    # 剩余搜索方向: e_n = f(grad(grad(C))).e_{n-2}
    # Week 2 优化: 向量化 Hessian 向量积
    if maxDir > 2:
        # 预计算 sig2_inv 避免重复除法
        sig2_inv = 1.0 / sig2

        for i in range(2, maxDir):
            tempDot = np.dot(Resp, edir[:, i - 2])
            edir[:, i] = fsupi * np.dot(Resp.T, tempDot * sig2_inv)

            e_norm_sq = np.dot(edir[:, i], edir[:, i])
            if e_norm_sq > 0.0:
                edir[:, i] /= np.sqrt(e_norm_sq)

    # 对角化搜索子空间
    edir, nedir, gamma = _diag_dir(edir, maxDir, gradgradS, sig2, Resp)

    return edir, nedir, gamma


def _diag_dir(edir: np.ndarray, nedir: int, gradgradS: np.ndarray,
              sig2: np.ndarray,
              Resp: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    对角化搜索空间，移除线性相关的方向。

    基于Skilling & Bryan Sect. 3.7.1，计算metric g = e^T.(-gradgradS).e
    并对角化，得到独立的搜索基。
    """
    # 计算metric: g_mu,nu = e^T_mu.(-gradgradS).e_nu
    g = np.dot((-gradgradS * edir.T), edir)

    # 对角化metric
    gamma, eiVec = linalg.eigh(g)

    # 去除小特征值（线性相关判据）
    maxGamma = np.max(np.abs(gamma))
    iok = gamma > 1e-8 * maxGamma

    # 投影到新基并归一化
    edir = np.dot(edir, eiVec[:, iok]) / np.sqrt(gamma[iok])
    nedir = edir.shape[1]

    # 计算M矩阵的对角元素: M_mu,nu = e^T_mu.grad(grad(C)).e_nu
    tmpRe = np.dot(Resp, edir)
    MM = 2.0 * np.dot((tmpRe.T / sig2), tmpRe)

    # 对角化M
    gammaM, eiVec = linalg.eigh(MM)

    # 投影到M的特征基
    edir = np.dot(edir, eiVec)

    return edir, nedir, gammaM


def _get_cmu_smu(gradC: np.ndarray, gradS: np.ndarray,
                 edir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算梯度在搜索基上的投影。"""
    Cmu = np.dot(gradC, edir)
    Smu = np.dot(gradS, edir)
    return Cmu, Smu


def _get_l0_squared(L_fac: float, Itot: float) -> float:
    """
    计算参数空间的步长限制。

    L_fac 通常 0.1-0.5，Itot为总image强度。
    """
    return L_fac * Itot


def _get_alpha_min(gamma: np.ndarray) -> float:
    """
    计算Hessian约束的最小α值。

    若Hessian有负特征值，则 alphaMin = -min(gamma)。
    """
    alphaMin = 0.0
    minGamma = np.min(gamma)
    if minGamma < 0.0:
        alphaMin = -minGamma
    return alphaMin


def _get_caim_quad(chiAim: float, C0: float, gamma: np.ndarray,
                   Cmu: np.ndarray) -> float:
    """
    计算目标χ²的二次近似 (Skilling & Bryan Eq. 29)。
    """
    Cminq = C0 - 0.5 * np.sum(Cmu * Cmu / gamma)
    # Skilling & Bryan: 0.667 和 0.333 是经验常数
    Caimq = 0.66666667 * Cminq + 0.33333333 * C0
    if chiAim > Caimq:
        Caimq = chiAim
    return Caimq


def _get_saim_quad(Saim: float, S0: float, gamma: np.ndarray,
                   Smu: np.ndarray) -> float:
    """计算目标熵的二次近似。"""
    Sminq = S0 - 0.5 * np.sum(Smu * Smu / gamma)
    Saimq = 0.66666667 * Sminq + 0.33333333 * S0
    if Saim > Saimq:
        Saimq = Saim
    return Saimq


def _chop_down(alpha: float, alphaLow: float) -> Tuple[float, float]:
    """二分法：减小α"""
    alphaHigh = alpha
    alpha = 0.5 * (alphaLow + alpha)
    return alpha, alphaHigh


def _chop_up(alpha: float, alphaHigh: float) -> Tuple[float, float]:
    """二分法：增加α"""
    alphaLow = alpha
    if alphaHigh > 0.0:
        alpha = 0.5 * (alphaHigh + alpha)
    else:
        alpha = 2.0 * alpha + 0.1
    return alpha, alphaLow


def _control_chi2(C0: float, gamma: np.ndarray, Cmu: np.ndarray,
                  Smu: np.ndarray, Caimq: float, L02: float, alphaMin: float,
                  convTol: float) -> np.ndarray:
    """
    Skilling-Bryan控制过程：拟合至目标χ²。

    实现 Fig. 3 的流程图，通过二分法搜索 α 和 P 参数。
    """
    # 允许χ²在目标附近轻微弛豫
    if C0 < Caimq * 1.001:
        Caimr = Caimq * 1.001
    else:
        Caimr = Caimq

    P = Plow = 0.0
    Phigh = 0.0
    Pfinished = 0

    while Pfinished == 0:
        alphaLow = alphaMin
        alphaHigh = -1.0
        alpha = alphaMin + 1.0
        afinished = 0

        while afinished == 0:
            asuccess = 0

            # 计算二次近似的x, χ², 和步长
            xqp = (alpha * Smu - Cmu) / (P + gamma + alpha)
            Cq = C0 + np.dot(Cmu, xqp) + 0.5 * np.sum(gamma * xqp * xqp)
            Cqp = C0 + np.dot(Cmu, xqp) + 0.5 * np.sum((P + gamma) * xqp * xqp)
            L2 = np.dot(xqp, xqp)

            # 根据χ²和步长调整α
            if (Cqp > C0) and (Cqp > Caimr):
                alpha, alphaHigh = _chop_down(alpha, alphaLow)
            elif (Cq < Caimq) and (Cq < C0):
                alpha, alphaLow = _chop_up(alpha, alphaHigh)
            elif L2 > L02:
                if Cqp < C0:
                    alpha, alphaLow = _chop_up(alpha, alphaHigh)
                else:
                    alpha, alphaHigh = _chop_down(alpha, alphaLow)
            else:
                asuccess = 1
                if Cq < Caimq:
                    alpha, alphaLow = _chop_up(alpha, alphaHigh)
                else:
                    alpha, alphaHigh = _chop_down(alpha, alphaLow)

            # 收敛检验
            if (alphaHigh > 0.0 and abs(alphaHigh - alphaLow)
                    < (convTol * alphaHigh + 1.0e-10)):
                afinished = 1
            if alpha > 1.0e20:
                afinished = 1

        # 调整P值
        if asuccess != 1:
            P, Plow = _chop_up(P, Phigh)
        else:
            if (P == 0.0) or (abs(Phigh - Plow) < convTol * Phigh + 1.0e-10):
                Pfinished = 1
            else:
                P, Phigh = _chop_down(P, Plow)

        if (asuccess == 0) and (P > 1.0e20):
            Pfinished = 1

    return xqp


def _control_entropy(S0: float, gamma: np.ndarray, Cmu: np.ndarray,
                     Smu: np.ndarray, Saimq: float, L02: float,
                     alphaMin: float, convTol: float) -> np.ndarray:
    """
    Skilling-Bryan控制过程：拟合至目标熵。

    与_control_chi2类似，但目标改为熵。
    """
    P = Plow = 0.0
    Phigh = 0.0
    Pfinished = 0

    while Pfinished == 0:
        alphaLow = alphaMin
        alphaHigh = -1.0
        alpha = alphaMin + 1.0
        afinished = 0

        while afinished == 0:
            asuccess = 0

            xqp = (alpha * Smu - Cmu) / (P + gamma + alpha)
            Sq = S0 + np.dot(Smu, xqp) + 0.5 * np.sum(gamma * xqp * xqp)
            Sqp = S0 + np.dot(Smu, xqp) + 0.5 * np.sum((P + gamma) * xqp * xqp)
            L2 = np.dot(xqp, xqp)

            if (Sqp < S0) and (Sqp < Saimq):
                alpha, alphaHigh = _chop_down(alpha, alphaLow)
            elif (Sq > Saimq) and (Sq > S0):
                alpha, alphaLow = _chop_up(alpha, alphaHigh)
            elif L2 > L02:
                if Sqp > S0:
                    alpha, alphaLow = _chop_up(alpha, alphaHigh)
                else:
                    alpha, alphaHigh = _chop_down(alpha, alphaLow)
            else:
                asuccess = 1
                if Sq > Saimq:
                    alpha, alphaLow = _chop_up(alpha, alphaHigh)
                else:
                    alpha, alphaHigh = _chop_down(alpha, alphaLow)

            if (alphaHigh > 0.0 and abs(alphaHigh - alphaLow)
                    < (convTol * alphaHigh + 1.0e-10)):
                afinished = 1
            if alpha > 1.0e20:
                afinished = 1

        if asuccess != 1:
            P, Plow = _chop_up(P, Phigh)
        else:
            if (P == 0.0) or (abs(Phigh - Plow) < convTol * Phigh + 1.0e-10):
                Pfinished = 1
            else:
                P, Phigh = _chop_down(P, Plow)

        if (asuccess == 0) and (P > 1.0e20):
            Pfinished = 1

    return xqp


def _get_test(gradC: np.ndarray, gradS: np.ndarray) -> float:
    """
    Skilling-Bryan 收敛检验量 (Eq. 37)。

    衡量梯度的反平行程度，范围[0, 0.5]。
    值接近0表示接近最优。
    """
    mag_gradS = np.sqrt(np.sum(gradS**2))
    mag_gradC = np.sqrt(np.sum(gradC**2))

    if mag_gradS == 0.0:
        inv_mag_gradS = 0.0
    else:
        inv_mag_gradS = 1.0 / mag_gradS

    if mag_gradC == 0.0:
        inv_mag_gradC = 0.0
    else:
        inv_mag_gradC = 1.0 / mag_gradC

    test = 0.5 * np.sum((gradS * inv_mag_gradS - gradC * inv_mag_gradC)**2)
    return test
