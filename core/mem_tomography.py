"""
MEM反演接口 - pyZeeTom项目特定参数化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

为通用MEM算法 (mem_generic.py) 提供项目特定的适配层。

包含：
  ✓ Stokes I, Q, U, V 谱线的拟合
  ✓ 局部磁场参数 (Blos, Bperp, chi) 的熵定义
  ✓ 数据打包和响应矩阵构建
  ✓ 参数向量的打包/解包
  ✓ 盘积分计算（与VelspaceDiskIntegrator集成）

典型工作流：
  1. 构建 MEMTomographyAdapter 实例
  2. 使用 pack_* 函数准备数据
  3. 调用 mem_optimizer.iterate() 进行单次迭代
  4. 使用 unpack_* 函数恢复物理参数

参考文献：
  - Skilling & Bryan 1984: Maximum Entropy Image Reconstruction
  - Hobson & Lasenby 1998: Magnetic field inversion (entropy for coefficients)
  - 本项目：core/velspace_DiskIntegrator.py 盘积分模型
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from core.mem_generic import MEMOptimizer, _get_c_gradc

# Week 2 优化：导入缓存和数据流管理工具
try:
    from core.mem_optimization import ResponseMatrixCache, DataPipeline
except ImportError:
    ResponseMatrixCache = None
    DataPipeline = None


class BrightnessDisk:
    """
    星周亮度分布容器。

    可以代表：
      - 尘埃盘的发射/吸收
      - 斑点或特征的亮度分布
    """

    def __init__(self, bright: np.ndarray):
        """
        初始化亮度映射。

        参数：
            bright: 每个像素的亮度值 (Npix,)
        """
        self.bright = np.asarray(bright)

    @property
    def nparam(self) -> int:
        return len(self.bright)

    def set_bright(self, new_bright: np.ndarray):
        """更新亮度值。"""
        self.bright[:] = new_bright


class MagneticFieldParams:
    """
    局部磁场参数容器。

    参数化：
      - Blos: 视向磁场分量 (Npix,)
      - Bperp: 垂直平面内磁场强度 (Npix,)
      - chi: 磁场方位角 (Npix,) 单位为弧度
    """

    def __init__(self, Blos: np.ndarray, Bperp: np.ndarray, chi: np.ndarray):
        """初始化磁场参数。"""
        self.Blos = np.asarray(Blos)
        self.Bperp = np.asarray(Bperp)
        self.chi = np.asarray(chi)

    @property
    def nparam(self) -> int:
        """返回总参数数 (3*Npix)。"""
        return 3 * len(self.Blos)

    @property
    def npix(self) -> int:
        """返回像素数。"""
        return len(self.Blos)

    def set_from_vector(self, vec: np.ndarray):
        """从向量解包参数。"""
        npix = self.npix
        self.Blos[:] = vec[0:npix]
        self.Bperp[:] = vec[npix:2 * npix]
        self.chi[:] = vec[2 * npix:3 * npix]

    def to_vector(self) -> np.ndarray:
        """打包为向量。"""
        return np.concatenate([self.Blos, self.Bperp, self.chi])


class MEMTomographyAdapter:
    """
    MEM反演的项目特定适配器。

    将通用MEM优化器与tomography项目的参数化绑定。
    """

    def __init__(self,
                 fit_brightness: bool = False,
                 fit_magnetic: bool = True,
                 fit_B_los: bool = True,
                 fit_B_perp: bool = True,
                 fit_chi: bool = True,
                 entropy_weights_bright: Optional[np.ndarray] = None,
                 entropy_weights_blos: Optional[np.ndarray] = None,
                 entropy_weights_bperp: Optional[np.ndarray] = None,
                 entropy_weights_chi: Optional[np.ndarray] = None,
                 default_bright: float = 1.0,
                 default_blos: float = 0.1,
                 default_bperp: float = 0.1,
                 default_chi: float = 0.0):
        """
        初始化适配器。

        参数：
            fit_brightness: 是否拟合亮度
            fit_magnetic: 是否拟合磁场 (已弃用，保留兼容性，若为False则覆盖fit_B_los等)
            fit_B_los: 是否拟合视向磁场
            fit_B_perp: 是否拟合垂直磁场
            fit_chi: 是否拟合磁场方位角
            entropy_weights_*: 各参数的熵权重 (Npix,)
                若为None则默认为全1
            default_*: 默认值（熵计算所需）
        """
        self.fit_brightness = fit_brightness

        # 兼容旧的 fit_magnetic 参数
        if not fit_magnetic:
            self.fit_B_los = False
            self.fit_B_perp = False
            self.fit_chi = False
        else:
            self.fit_B_los = fit_B_los
            self.fit_B_perp = fit_B_perp
            self.fit_chi = fit_chi

        # 存储权重和默认值
        self.entropy_weights_bright = entropy_weights_bright
        self.entropy_weights_blos = entropy_weights_blos
        self.entropy_weights_bperp = entropy_weights_bperp
        self.entropy_weights_chi = entropy_weights_chi
        self.default_bright = default_bright
        self.default_blos = default_blos
        self.default_bperp = default_bperp
        self.default_chi = default_chi

        # 初始化缓存和数据流支持（延迟初始化）
        self.resp_cache = ResponseMatrixCache(
            max_size=10) if ResponseMatrixCache else None
        self.data_pipeline = None  # 延迟初始化，需要在运行时传入观测数据
        self._constraint_cache = {}  # 简单约束计算缓存

        self.optimizer = MEMOptimizer(
            compute_entropy_callback=self.compute_entropy_callback,
            compute_constraint_callback=self.compute_constraint_callback,
            boundary_constraint_callback=self.apply_boundary_constraints,
            max_search_dirs=10,
            step_length_factor=0.3,
            convergence_tol=1e-5)

    def compute_entropy_callback(
            self, Image: np.ndarray, weights: np.ndarray,
            entropy_params: Dict[str, Any], n1: int, n2: int,
            ntot: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        计算总熵及其梯度。

        基于标准MaxEnt熵定义:
          S = -sum_i w_i * (I_i * (log(I_i/I0) - 1) + I0)

        对于Blos, Bperp, chi 可以是对称熵（允许正负值）。

        返回：
            (S0, gradS, gradgradS)
        """
        npix = entropy_params.get('npix', 0)
        gradS = np.zeros(len(Image))
        gradgradS = np.zeros(len(Image))

        S0 = 0.0
        idx = 0

        # 亮度部分（标准熵，正值）
        if self.fit_brightness:
            n_bright = entropy_params.get('n_bright', npix)
            w = self.entropy_weights_bright
            if w is None:
                w = np.ones(n_bright)
            defI = self.default_bright

            I_bright = Image[idx:idx + n_bright]

            S0 += np.sum(-w * (I_bright *
                               (np.log(I_bright / defI) - 1.0) + defI))
            gradS[idx:idx + n_bright] = w * (np.log(defI) - np.log(I_bright))
            gradgradS[idx:idx + n_bright] = -w / I_bright
            idx += n_bright

        # Blos 部分（允许正负，对称熵）
        if self.fit_B_los:
            n_blos = entropy_params.get('n_blos', npix)
            w = self.entropy_weights_blos
            if w is None:
                w = np.ones(n_blos)
            defB = self.default_blos

            Blos = Image[idx:idx + n_blos]
            # 对称形式：可以写为 S = -sum w_i * sign(Blos) * Blos * log(|Blos| / defB)
            # 简化为容许正负的标准形式
            abs_Blos = np.abs(Blos) + 1e-10  # 防止log(0)
            S0 += np.sum(-w * abs_Blos * (np.log(abs_Blos / defB) - 1.0))
            # 梯度（处理符号）
            sign_Blos = np.sign(Blos)
            gradS[idx:idx +
                  n_blos] = -w * sign_Blos * (np.log(abs_Blos / defB))
            gradgradS[idx:idx + n_blos] = -w / (abs_Blos + 1e-15)
            idx += n_blos

        # Bperp 部分（正值）
        if self.fit_B_perp:
            n_bperp = entropy_params.get('n_bperp', npix)
            w = self.entropy_weights_bperp
            if w is None:
                w = np.ones(n_bperp)
            defB = self.default_bperp

            Bperp = Image[idx:idx + n_bperp]
            S0 += np.sum(-w * (Bperp * (np.log(Bperp / defB) - 1.0) + defB))
            gradS[idx:idx + n_bperp] = w * (np.log(defB) - np.log(Bperp))
            gradgradS[idx:idx + n_bperp] = -w / Bperp
            idx += n_bperp

        # chi 部分（方位角，周期性）
        if self.fit_chi:
            n_chi = entropy_params.get('n_chi', npix)
            w = self.entropy_weights_chi
            if w is None:
                w = np.ones(n_chi)

            chi = Image[idx:idx + n_chi]
            # chi的熵：鼓励均匀分布，使用cos形式
            # S_chi = -sum w_i * cos(chi_i - chi_0)  (简化)
            # 或标准形式：S_chi = sum w_i * log(I0(kappa)) (von Mises)
            # 这里使用简化：S = - sum w * chi^2 (高斯先验，偏向于0)
            # 注意：MEM最大化熵，因此必须是负的二次项
            S0 -= 0.1 * np.sum(w * chi**2)
            gradS[idx:idx + n_chi] = -0.2 * w * chi
            gradgradS[idx:idx + n_chi] = -0.2 * w
            idx += n_chi

        return S0, gradS, gradgradS

    def compute_constraint_callback(
            self, Data: np.ndarray, Fmodel: np.ndarray, sig2: np.ndarray,
            Resp: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算约束统计量χ²及梯度。

        χ² = sum_k [(F_k - D_k)^2 / sigma_k^2]

        使用简单缓存加速重复计算。

        返回：
            (C0, gradC)
        """
        # 生成缓存键（基于 Resp 和 Data 的哈希）
        try:
            resp_key = hash(Resp.tobytes())
            data_key = hash(Data.tobytes())
            cache_key = (resp_key, data_key)

            # 检查缓存
            if cache_key in self._constraint_cache:
                return self._constraint_cache[cache_key]
        except (TypeError, ValueError):
            # 如果无法哈希（如大数组），跳过缓存
            cache_key = None

        # 计算约束
        C0, gradC = _get_c_gradc(Data, Fmodel, sig2, Resp)

        # 存储到缓存（如果键有效）
        if cache_key is not None:
            # 限制缓存大小
            if len(self._constraint_cache) > 20:
                self._constraint_cache.pop(next(iter(self._constraint_cache)))
            self._constraint_cache[cache_key] = (C0, gradC)

        return C0, gradC

    def apply_boundary_constraints(
            self, Image: np.ndarray, n1: int, n2: int, ntot: int,
            entropy_params: Dict[str, Any]) -> np.ndarray:
        """
        应用物理边界约束。

        亮度和Bperp必须为正；Blos和chi无限制。
        """
        npix = entropy_params.get('npix', 0)
        idx = 0

        # 亮度约束（正值）
        if self.fit_brightness:
            n_bright = entropy_params.get('n_bright', npix)
            Image[idx:idx + n_bright] = np.clip(Image[idx:idx + n_bright],
                                                1e-6, np.inf)
            idx += n_bright

        # Blos 无约束
        if self.fit_B_los:
            n_blos = entropy_params.get('n_blos', npix)
            # Blos 可以是任意值
            idx += n_blos

        # Bperp 约束（正值）
        if self.fit_B_perp:
            n_bperp = entropy_params.get('n_bperp', npix)
            Image[idx:idx + n_bperp] = np.clip(Image[idx:idx + n_bperp], 1e-6,
                                               np.inf)
            idx += n_bperp

        # chi 约束（wrap to [-π, π]）
        if self.fit_chi:
            n_chi = entropy_params.get('n_chi', npix)
            Image[idx:idx + n_chi] = np.angle(
                np.exp(1j * Image[idx:idx + n_chi]))

        return Image

    def pack_image_vector(
        self,
        bright_disk: Optional[BrightnessDisk] = None,
        mag_field: Optional[MagneticFieldParams] = None
    ) -> Tuple[np.ndarray, int, int, int, int]:
        """
        打包参数为优化向量。

        返回：
            (Image_vec, n_bright, n_blos, n_bperp, n_chi)
        """
        Image_parts = []

        n_bright = 0
        if self.fit_brightness and bright_disk is not None:
            Image_parts.append(bright_disk.bright)
            n_bright = len(bright_disk.bright)

        n_blos = 0
        if self.fit_B_los and mag_field is not None:
            Image_parts.append(mag_field.Blos)
            n_blos = len(mag_field.Blos)

        n_bperp = 0
        if self.fit_B_perp and mag_field is not None:
            Image_parts.append(mag_field.Bperp)
            n_bperp = len(mag_field.Bperp)

        n_chi = 0
        if self.fit_chi and mag_field is not None:
            Image_parts.append(mag_field.chi)
            n_chi = len(mag_field.chi)

        if not Image_parts:
            return np.array([]), 0, 0, 0, 0

        return np.concatenate(Image_parts), n_bright, n_blos, n_bperp, n_chi

    def unpack_image_vector(self,
                            Image: np.ndarray,
                            n_bright: int,
                            n_blos: int,
                            n_bperp: int,
                            n_chi: int,
                            bright_disk: Optional[BrightnessDisk] = None,
                            mag_field: Optional[MagneticFieldParams] = None):
        """
        从优化向量解包参数。
        """
        idx = 0

        if self.fit_brightness and bright_disk is not None:
            bright_disk.set_bright(Image[idx:idx + n_bright])
            idx += n_bright

        if mag_field is not None:
            if self.fit_B_los:
                mag_field.Blos[:] = Image[idx:idx + n_blos]
                idx += n_blos

            if self.fit_B_perp:
                mag_field.Bperp[:] = Image[idx:idx + n_bperp]
                idx += n_bperp

            if self.fit_chi:
                mag_field.chi[:] = Image[idx:idx + n_chi]
                idx += n_chi

    def init_data_pipeline(self,
                           observations: List[Any],
                           fit_I: bool = True,
                           fit_V: bool = True,
                           fit_Q: bool = False,
                           fit_U: bool = False) -> None:
        """
        初始化数据流水线（可选）。

        如果提供了观测数据，DataPipeline 可预处理和验证数据一致性。

        参数：
            observations: 观测数据对象列表
            fit_I/V/Q/U: 是否拟合各分量
        """
        if DataPipeline is None:
            return  # 如果 DataPipeline 不可用，优雅降级

        try:
            self.data_pipeline = DataPipeline(observations=observations,
                                              fit_I=fit_I,
                                              fit_V=fit_V,
                                              fit_Q=fit_Q,
                                              fit_U=fit_U,
                                              verbose=0)
        except Exception as e:
            # 如果初始化失败，记录警告但继续
            print(f"Warning: Failed to initialize DataPipeline: {e}")
            self.data_pipeline = None

    def set_entropy_weights(self,
                            npix: int,
                            grid_area: Optional[np.ndarray] = None):
        """
        设置熵权重。

        参数：
            npix: 像素数
            grid_area: 网格面积 (Npix,)，若提供则权重与面积成正比
        """
        if grid_area is not None:
            self.entropy_weights_bright = grid_area
            self.entropy_weights_blos = grid_area
            self.entropy_weights_bperp = grid_area
            self.entropy_weights_chi = grid_area
        else:
            self.entropy_weights_bright = np.ones(npix)
            self.entropy_weights_blos = np.ones(npix)
            self.entropy_weights_bperp = np.ones(npix)
            self.entropy_weights_chi = np.ones(npix)
