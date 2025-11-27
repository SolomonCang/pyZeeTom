"""
MEM Inversion Interface - pyZeeTom Project Specific Parameterization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provides a project-specific adapter layer for the generic MEM algorithm (mem_generic.py).

Includes:
  ✓ Fitting of Stokes I, Q, U, V spectra
  ✓ Entropy definition for local magnetic field parameters (Blos, Bperp, chi)
  ✓ Data packing and response matrix construction
  ✓ Parameter vector packing/unpacking
  ✓ Disk integration calculation (integrated with VelspaceDiskIntegrator)

Typical Workflow:
  1. Construct MEMTomographyAdapter instance
  2. Use pack_* functions to prepare data
  3. Call mem_optimizer.iterate() for a single iteration
  4. Use unpack_* functions to restore physical parameters

References:
  - Skilling & Bryan 1984: Maximum Entropy Image Reconstruction
  - Hobson & Lasenby 1998: Magnetic field inversion (entropy for coefficients)
  - This project: core/velspace_DiskIntegrator.py Disk integration model
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from core.mem_generic import MEMOptimizer, _get_c_gradc

# Week 2 Optimization: Import cache and data flow management tools
try:
    from core.mem_optimization import ResponseMatrixCache, DataPipeline
except ImportError:
    ResponseMatrixCache = None
    DataPipeline = None


class BrightnessDisk:
    """
    Circumstellar brightness distribution container.

    Can represent:
      - Emission/absorption of dust disk
      - Brightness distribution of spots or features
    """

    def __init__(self, bright: np.ndarray):
        """
        Initialize brightness map.

        Parameters:
            bright: Brightness value per pixel (Npix,)
        """
        self.bright = np.asarray(bright)

    @property
    def nparam(self) -> int:
        return len(self.bright)

    def set_bright(self, new_bright: np.ndarray):
        """Update brightness values."""
        self.bright[:] = new_bright


class MagneticFieldParams:
    """
    Local magnetic field parameter container.

    Parameterization:
      - Blos: Line-of-sight magnetic field component (Npix,)
      - Bperp: Magnetic field strength in perpendicular plane (Npix,)
      - chi: Magnetic field azimuth angle (Npix,) in radians
    """

    def __init__(self, Blos: np.ndarray, Bperp: np.ndarray, chi: np.ndarray):
        """Initialize magnetic field parameters."""
        self.Blos = np.asarray(Blos)
        self.Bperp = np.asarray(Bperp)
        self.chi = np.asarray(chi)

    @property
    def nparam(self) -> int:
        """Return total number of parameters (3*Npix)."""
        return 3 * len(self.Blos)

    @property
    def npix(self) -> int:
        """Return number of pixels."""
        return len(self.Blos)

    def set_from_vector(self, vec: np.ndarray):
        """Unpack parameters from vector."""
        npix = self.npix
        self.Blos[:] = vec[0:npix]
        self.Bperp[:] = vec[npix:2 * npix]
        self.chi[:] = vec[2 * npix:3 * npix]

    def to_vector(self) -> np.ndarray:
        """Pack into vector."""
        return np.concatenate([self.Blos, self.Bperp, self.chi])


class MEMTomographyAdapter:
    """
    Project-specific adapter for MEM inversion.

    Binds the generic MEM optimizer with the tomography project parameterization.
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
        Initialize adapter.

        Parameters:
            fit_brightness: Whether to fit brightness
            fit_magnetic: Whether to fit magnetic field (deprecated, kept for compatibility, if False overrides fit_B_los etc.)
            fit_B_los: Whether to fit line-of-sight magnetic field
            fit_B_perp: Whether to fit perpendicular magnetic field
            fit_chi: Whether to fit magnetic field azimuth
            entropy_weights_*: Entropy weights for each parameter (Npix,)
                If None, defaults to all ones
            default_*: Default values (required for entropy calculation)
        """
        self.fit_brightness = fit_brightness

        # Compatible with old fit_magnetic parameter
        if not fit_magnetic:
            self.fit_B_los = False
            self.fit_B_perp = False
            self.fit_chi = False
        else:
            self.fit_B_los = fit_B_los
            self.fit_B_perp = fit_B_perp
            self.fit_chi = fit_chi

        # Store weights and default values
        self.entropy_weights_bright = entropy_weights_bright
        self.entropy_weights_blos = entropy_weights_blos
        self.entropy_weights_bperp = entropy_weights_bperp
        self.entropy_weights_chi = entropy_weights_chi
        self.default_bright = default_bright
        self.default_blos = default_blos
        self.default_bperp = default_bperp
        self.default_chi = default_chi

        # Initialize cache and data flow support (lazy initialization)
        self.resp_cache = ResponseMatrixCache(
            max_size=10) if ResponseMatrixCache else None
        self.data_pipeline = None  # Lazy initialization, needs observation data passed at runtime
        self._constraint_cache = {}  # Simple constraint calculation cache

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
        Calculate total entropy and its gradient.

        Based on standard MaxEnt entropy definition:
          S = -sum_i w_i * (I_i * (log(I_i/I0) - 1) + I0)

        For Blos, Bperp, chi can be symmetric entropy (allowing positive and negative values).

        Returns:
            (S0, gradS, gradgradS)
        """
        npix = entropy_params.get('npix', 0)
        gradS = np.zeros(len(Image))
        gradgradS = np.zeros(len(Image))

        S0 = 0.0
        idx = 0

        # Brightness part (standard entropy, positive values)
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

        # Blos part (allows positive/negative, symmetric entropy)
        if self.fit_B_los:
            n_blos = entropy_params.get('n_blos', npix)
            w = self.entropy_weights_blos
            if w is None:
                w = np.ones(n_blos)
            defB = self.default_blos

            Blos = Image[idx:idx + n_blos]
            # Symmetric form: can be written as S = -sum w_i * sign(Blos) * Blos * log(|Blos| / defB)
            # Simplified to standard form allowing positive/negative
            abs_Blos = np.abs(Blos) + 1e-10  # Prevent log(0)
            S0 += np.sum(-w * abs_Blos * (np.log(abs_Blos / defB) - 1.0))
            # Gradient (handle sign)
            sign_Blos = np.sign(Blos)
            gradS[idx:idx +
                  n_blos] = -w * sign_Blos * (np.log(abs_Blos / defB))
            gradgradS[idx:idx + n_blos] = -w / (abs_Blos + 1e-15)
            idx += n_blos

        # Bperp part (positive values)
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

        # chi part (azimuth, periodic)
        if self.fit_chi:
            n_chi = entropy_params.get('n_chi', npix)
            w = self.entropy_weights_chi
            if w is None:
                w = np.ones(n_chi)

            chi = Image[idx:idx + n_chi]
            # Entropy of chi: encourage uniform distribution, use cos form
            # S_chi = -sum w_i * cos(chi_i - chi_0)  (simplified)
            # Or standard form: S_chi = sum w_i * log(I0(kappa)) (von Mises)
            # Here use simplified: S = - sum w * chi^2 (Gaussian prior, biased towards 0)
            # Note: MEM maximizes entropy, so must be negative quadratic term
            S0 -= 0.1 * np.sum(w * chi**2)
            gradS[idx:idx + n_chi] = -0.2 * w * chi
            gradgradS[idx:idx + n_chi] = -0.2 * w
            idx += n_chi

        return S0, gradS, gradgradS

    def compute_constraint_callback(
            self, Data: np.ndarray, Fmodel: np.ndarray, sig2: np.ndarray,
            Resp: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate constraint statistic χ² and gradient.

        χ² = sum_k [(F_k - D_k)^2 / sigma_k^2]

        Returns:
            (C0, gradC)
        """
        # Calculate constraint
        C0, gradC = _get_c_gradc(Data, Fmodel, sig2, Resp)

        return C0, gradC

    def apply_boundary_constraints(
            self, Image: np.ndarray, n1: int, n2: int, ntot: int,
            entropy_params: Dict[str, Any]) -> np.ndarray:
        """
        Apply physical boundary constraints.

        Brightness and Bperp must be positive; Blos and chi are unconstrained.
        """
        npix = entropy_params.get('npix', 0)
        idx = 0

        # Brightness constraint (positive values)
        if self.fit_brightness:
            n_bright = entropy_params.get('n_bright', npix)
            Image[idx:idx + n_bright] = np.clip(Image[idx:idx + n_bright],
                                                1e-6, np.inf)
            idx += n_bright

        # Blos unconstrained
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
