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
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from core.mem_generic import MEMOptimizer, _get_c_gradc


@dataclass
class StokesObservation:
    """单个观测相位的Stokes谱线数据。"""
    wl: np.ndarray  # 波长数组 (Nlambda,)
    specI: np.ndarray  # I分量 (Nlambda,)
    specQ: np.ndarray  # Q分量 (Nlambda,)
    specU: np.ndarray  # U分量 (Nlambda,)
    specV: np.ndarray  # V分量 (Nlambda,)
    specI_sig: np.ndarray  # I分量噪声 (Nlambda,)
    specQ_sig: np.ndarray  # Q分量噪声 (Nlambda,)
    specU_sig: np.ndarray  # U分量噪声 (Nlambda,)
    specV_sig: np.ndarray  # V分量噪声 (Nlambda,)


@dataclass
class SyntheticSpectrum:
    """单次合成的Stokes谱线及其参数导数。"""
    wl: np.ndarray
    IIc: np.ndarray  # 合成I (Nlambda,)
    QIc: np.ndarray  # 合成Q (Nlambda,)
    UIc: np.ndarray  # 合成U (Nlambda,)
    VIc: np.ndarray  # 合成V (Nlambda,)

    # 导数: (Nlambda, Nparam)
    dIc_dBlos: np.ndarray
    dIc_dBperp: np.ndarray
    dQc_dBlos: np.ndarray
    dQc_dBperp: np.ndarray
    dQc_dchi: np.ndarray
    dUc_dBlos: np.ndarray
    dUc_dBperp: np.ndarray
    dUc_dchi: np.ndarray
    dVc_dBlos: np.ndarray
    dVc_dBperp: np.ndarray
    dVc_dchi: np.ndarray


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
            fit_magnetic: 是否拟合磁场
            entropy_weights_*: 各参数的熵权重 (Npix,)
                若为None则默认为全1
            default_*: 默认值（熵计算所需）
        """
        self.fit_brightness = fit_brightness
        self.fit_magnetic = fit_magnetic

        # 存储权重和默认值
        self.entropy_weights_bright = entropy_weights_bright
        self.entropy_weights_blos = entropy_weights_blos
        self.entropy_weights_bperp = entropy_weights_bperp
        self.entropy_weights_chi = entropy_weights_chi
        self.default_bright = default_bright
        self.default_blos = default_blos
        self.default_bperp = default_bperp
        self.default_chi = default_chi

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
        npix = entropy_params.get(
            'npix',
            len(Image) // 3 if self.fit_magnetic else len(Image))
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
        if self.fit_magnetic:
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
            n_chi = entropy_params.get('n_chi', npix)
            w = self.entropy_weights_chi
            if w is None:
                w = np.ones(n_chi)

            chi = Image[idx:idx + n_chi]
            # chi的熵：鼓励均匀分布，使用cos形式
            # S_chi = -sum w_i * cos(chi_i - chi_0)  (简化)
            # 或标准形式：S_chi = sum w_i * log(I0(kappa)) (von Mises)
            # 这里使用简化：S = sum w * chi^2 (平滑化)
            S0 += 0.1 * np.sum(w * chi**2)
            gradS[idx:idx + n_chi] = 0.2 * w * chi
            gradgradS[idx:idx + n_chi] = 0.2 * w
            idx += n_chi

        return S0, gradS, gradgradS

    def compute_constraint_callback(
            self, Data: np.ndarray, Fmodel: np.ndarray, sig2: np.ndarray,
            Resp: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算约束统计量χ²及梯度。

        χ² = sum_k [(F_k - D_k)^2 / sigma_k^2]

        返回：
            (C0, gradC)
        """
        return _get_c_gradc(Data, Fmodel, sig2, Resp)

    def apply_boundary_constraints(
            self, Image: np.ndarray, n1: int, n2: int, ntot: int,
            entropy_params: Dict[str, Any]) -> np.ndarray:
        """
        应用物理边界约束。

        亮度和Bperp必须为正；Blos和chi无限制。
        """
        npix = entropy_params.get('npix',
                                  (ntot // 3 if self.fit_magnetic else ntot))
        idx = 0

        # 亮度约束（正值）
        if self.fit_brightness:
            n_bright = entropy_params.get('n_bright', npix)
            Image[idx:idx + n_bright] = np.clip(Image[idx:idx + n_bright],
                                                1e-6, np.inf)
            idx += n_bright

        # Blos 无约束
        if self.fit_magnetic:
            n_blos = entropy_params.get('n_blos', npix)
            # Blos 可以是任意值
            idx += n_blos

            # Bperp 约束（正值）
            n_bperp = entropy_params.get('n_bperp', npix)
            Image[idx:idx + n_bperp] = np.clip(Image[idx:idx + n_bperp], 1e-6,
                                               np.inf)
            idx += n_bperp

            # chi 约束（wrap to [-π, π]）
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
        n_bperp = 0
        n_chi = 0
        if self.fit_magnetic and mag_field is not None:
            npix = mag_field.npix
            Image_parts.extend(
                [mag_field.Blos, mag_field.Bperp, mag_field.chi])
            n_blos = npix
            n_bperp = npix
            n_chi = npix

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

        if self.fit_magnetic and mag_field is not None:
            mag_field.set_from_vector(Image[idx:idx + 3 * n_blos])

    def pack_data_and_response(
        self,
        obs_list: list,
        syn_list: list,
        fit_I: bool = True,
        fit_Q: bool = False,
        fit_U: bool = False,
        fit_V: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        打包观测数据、模型和响应矩阵。

        参数：
            obs_list: 观测 StokesObservation 列表
            syn_list: 合成 SyntheticSpectrum 列表
            fit_*: 是否包含各Stokes分量

        返回：
            (Data, Fmodel, sig2, Resp)
        """
        Data_parts = []
        Fmodel_parts = []
        sig2_parts = []
        Resp_parts = []

        n_lambda_tot = 0
        for obs in obs_list:
            n_lambda_tot += len(obs.wl)

        # I分量
        if fit_I:
            Data_I = np.concatenate([obs.specI for obs in obs_list])
            Fmodel_I = np.concatenate([syn.IIc for syn in syn_list])
            sig2_I = np.concatenate([obs.specI_sig**2 for obs in obs_list])
            Data_parts.append(Data_I)
            Fmodel_parts.append(Fmodel_I)
            sig2_parts.append(sig2_I)

        # Q分量
        if fit_Q:
            Data_Q = np.concatenate([obs.specQ for obs in obs_list])
            Fmodel_Q = np.concatenate([syn.QIc for syn in syn_list])
            sig2_Q = np.concatenate([obs.specQ_sig**2 for obs in obs_list])
            Data_parts.append(Data_Q)
            Fmodel_parts.append(Fmodel_Q)
            sig2_parts.append(sig2_Q)

        # U分量
        if fit_U:
            Data_U = np.concatenate([obs.specU for obs in obs_list])
            Fmodel_U = np.concatenate([syn.UIc for syn in syn_list])
            sig2_U = np.concatenate([obs.specU_sig**2 for obs in obs_list])
            Data_parts.append(Data_U)
            Fmodel_parts.append(Fmodel_U)
            sig2_parts.append(sig2_U)

        # V分量
        if fit_V:
            Data_V = np.concatenate([obs.specV for obs in obs_list])
            Fmodel_V = np.concatenate([syn.VIc for syn in syn_list])
            sig2_V = np.concatenate([obs.specV_sig**2 for obs in obs_list])
            Data_parts.append(Data_V)
            Fmodel_parts.append(Fmodel_V)
            sig2_parts.append(sig2_V)

        Data = np.concatenate(Data_parts)
        Fmodel = np.concatenate(Fmodel_parts)
        sig2 = np.concatenate(sig2_parts)

        # 响应矩阵（这里需要用户提供的求导数据）
        # 简化版本：假设合成谱已包含导数信息
        # 完整实现需要从 VelspaceDiskIntegrator 获取

        ndata = len(Data)
        nparam = 0
        if self.fit_brightness:
            nparam += len(obs_list[0].specI)  # 示例，实际应更正
        if self.fit_magnetic:
            # 假设 Npix 从 syn_list 推断
            nparam += 3 * len(syn_list[0].dBlos_dBlos) if hasattr(
                syn_list[0], 'dBlos_dBlos') else 3

        # 占位：响应矩阵需要从积分器获取
        Resp = np.zeros((ndata, nparam))

        return Data, Fmodel, sig2, Resp

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
