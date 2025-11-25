"""
Generic Maximum Entropy Method (MEM) Optimization Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on the algorithm framework of Skilling & Bryan (1984, MNRAS 211, 111)

This module contains **pure mathematical optimization algorithms**, independent of specific physical models and parameterizations.
Users need to customize via callback functions:
  - Calculate entropy and its derivatives (get_s_grads_callback)
  - Calculate constraint statistics and derivatives (get_c_gradc_callback)
  - Boundary constraint handling (boundary_constraint_callback)

Features:
  ✓ Supports vectorized operations for efficient computation
  ✓ Flexible entropy forms (customizable via callbacks)
  ✓ Robust numerical optimization control
  ✓ Complete implementation of the Skilling-Bryan algorithm

Usage example: see mem_tomography.py
"""

import numpy as np
from scipy import linalg
from typing import Callable, Tuple, Dict, Any, Optional

# Import optimization tools (Week 2 optimization)
try:
    from core.mem_optimization import StabilityMonitor
except ImportError:
    StabilityMonitor = None  # Fallback


class MEMOptimizer:
    """
    Generic MEM Optimizer.

    Users need to provide callback functions to define entropy forms and data fitting.
    """

    def __init__(self,
                 compute_entropy_callback: Callable,
                 compute_constraint_callback: Callable,
                 boundary_constraint_callback: Optional[Callable] = None,
                 max_search_dirs: int = 10,
                 step_length_factor: float = 0.3,
                 convergence_tol: float = 1e-5):
        """
        Initialize MEM Optimizer.

        Parameters:
            compute_entropy_callback:
                Function signature -> (S0, gradS, gradgradS)
                Calculate entropy, gradient, and Hessian diagonal
            compute_constraint_callback:
                Function signature (Response, sig2) -> (C0, gradC)
                Calculate constraint statistics (e.g., chi-squared) and gradient
            boundary_constraint_callback:
                Optional, signature (Image, n1, n2, ntot, ...) -> Image_corrected
                Apply boundary constraints (e.g., positivity)
            max_search_dirs: Maximum number of search directions (recommended 3-10)
            step_length_factor: Parameter space step length factor (0.1-0.5)
            convergence_tol: Convergence tolerance
        """
        self.compute_entropy = compute_entropy_callback
        self.compute_constraint = compute_constraint_callback
        self.apply_boundary_constraint = boundary_constraint_callback
        self.max_search_dirs = max_search_dirs
        self.step_length_factor = step_length_factor
        self.convergence_tol = convergence_tol

        # Week 2 Optimization: Integrate StabilityMonitor (optional)
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
        Execute a single MEM iteration step.

        Parameters:
            Image: Current parameter vector (length ntot)
            Fmodel: Current model prediction (length ndata)
            Data: Observational data (length ndata)
            sig2: Observational noise variance (length ndata)
            Resp: Response matrix dModel/dImage (ndata x ntot)
            weights: Entropy term weights (length ntot)
            entropy_params: Extra parameters required for entropy calculation
                e.g., {'defImg': 1.0, 'defIpm': 0.1, 'ffIMax': 1.0, ...}
            n1, n2, ntot: Parameter segment indices
                If not provided, defaults to all one entropy form
            fixEntropy: 0=fit to chi-squared target, 1=fit to entropy target
            targetAim: Target chi-squared or target entropy value

        Returns:
            (entropy, chi2, test, Image_new)
            entropy: Current entropy value
            chi2: Current chi-squared value
            test: Skilling-Bryan convergence test statistic
            Image_new: Updated parameter vector
        """
        if ntot is None:
            ntot = len(Image)
        if n1 is None:
            n1 = ntot
        if n2 is None:
            n2 = n1

        # Calculate constraint statistics (chi-squared)
        C0, gradC = self.compute_constraint(Data, Fmodel, sig2, Resp)

        # Calculate entropy and derivatives
        S0, gradS, gradgradS = self.compute_entropy(Image, weights,
                                                    entropy_params, n1, n2,
                                                    ntot)

        # Skilling-Bryan convergence test statistic
        test = _get_test(gradC, gradS)

        # Week 2 Optimization: Integrate StabilityMonitor check
        if self.stability_monitor is not None:
            stability_ok = self.stability_monitor.check_gradient(gradC, gradS)
            if not stability_ok:
                # Gradient issues (NaN, Inf, excessive values)
                # Attempt recovery: return current state, let upper layer handle
                import warnings as _warnings
                _warnings.warn(
                    "Stability check failed for gradients; may need to reduce step size"
                )

        # Calculate search directions
        edir, nedir, gamma = _search_dir(ntot, self.max_search_dirs, Resp,
                                         sig2, -1.0 / gradgradS, gradC, gradS,
                                         gradgradS)

        # Week 2 Optimization: Response matrix diagnosis
        # if self.stability_monitor is not None:
        #     _ = self.stability_monitor.check_response_matrix(Resp)

        # Calculate control quantities
        Cmu, Smu = _get_cmu_smu(gradC, gradS, edir)

        # Calculate step length parameter L02
        # For brightness image, use total intensity; for other parameters like magnetic field, use norm
        if n1 > 0:
            # Has brightness parameters, use weighted sum of brightness part
            Itot = np.sum(weights[:n1] * Image[:n1])
        else:
            # No brightness parameters, use norm of all parameters as scale
            Itot = np.sqrt(np.sum((weights * Image)**2))
        L02 = _get_l0_squared(self.step_length_factor, Itot)

        alphaMin = _get_alpha_min(gamma)

        if fixEntropy == 1:
            # Fit to target entropy
            Saim = targetAim if targetAim is not None else S0
            Saimq = _get_saim_quad(Saim, S0, gamma, Smu)
            xq = _control_entropy(S0, gamma, Cmu, Smu, Saimq, L02, alphaMin,
                                  self.convergence_tol)
        else:
            # Fit to target chi-squared
            chiAim = targetAim if targetAim is not None else C0
            Caimq = _get_caim_quad(chiAim, C0, gamma, Cmu)
            xq = _control_chi2(C0, gamma, Cmu, Smu, Caimq, L02, alphaMin,
                               self.convergence_tol)  # Update image vector
        Image_new = Image + np.dot(edir, xq)

        # Apply boundary constraints
        if self.apply_boundary_constraint is not None:
            Image_new = self.apply_boundary_constraint(Image_new, n1, n2, ntot,
                                                       entropy_params)

        return S0, C0, test, Image_new


# ============================================================================
# Internal Helper Functions (Module level, should not be called directly)
# ============================================================================


def _get_c_gradc(Data: np.ndarray, Fmodel: np.ndarray, sig2: np.ndarray,
                 Resp: np.ndarray) -> Tuple[float, np.ndarray]:
    """Calculate constraint statistic chi-squared and its gradient."""
    C0 = np.sum((Fmodel - Data)**2 / sig2)
    gradC = 2.0 * np.sum(Resp.T * (Fmodel - Data) / sig2, axis=1)
    return C0, gradC


def _search_dir(ntot: int, maxDir: int, Resp: np.ndarray, sig2: np.ndarray,
                fsupi: np.ndarray, gradC: np.ndarray, gradS: np.ndarray,
                gradgradS: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Generate linearly independent search directions.

    Based on Skilling & Bryan Eq. 20, generate up to maxDir orthogonalized search directions.
    
    Week 2 Optimization: Improved vectorized implementation
      • Vectorized norm calculation (avoid loops)
      • Batch matrix operations (accelerate Hessian vector product)
      • Numerical stability improvements
    """
    edir = np.zeros((ntot, maxDir))

    # 1st search direction: e_1 = f(grad(C))
    edir[:, 0] = gradC * fsupi
    e1_norm_sq = np.dot(edir[:, 0], edir[:, 0])

    if e1_norm_sq > 0.0:
        edir[:, 0] /= np.sqrt(e1_norm_sq)
        err_gradCis0 = 0
    else:
        err_gradCis0 = 1

    if maxDir == 1:
        return edir[:, :1], 1, np.array([1.0])

    # 2nd search direction: e_2 = f(grad(S))
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

    # Remaining search directions: e_n = f(grad(grad(C))).e_{n-2}
    # Week 2 Optimization: Vectorized Hessian vector product
    if maxDir > 2:
        # Precompute sig2_inv to avoid repeated division
        sig2_inv = 1.0 / sig2

        for i in range(2, maxDir):
            tempDot = np.dot(Resp, edir[:, i - 2])
            edir[:, i] = fsupi * np.dot(Resp.T, tempDot * sig2_inv)

            e_norm_sq = np.dot(edir[:, i], edir[:, i])
            if e_norm_sq > 0.0:
                edir[:, i] /= np.sqrt(e_norm_sq)

    # Diagonalize search subspace
    edir, nedir, gamma = _diag_dir(edir, maxDir, gradgradS, sig2, Resp)

    return edir, nedir, gamma


def _diag_dir(edir: np.ndarray, nedir: int, gradgradS: np.ndarray,
              sig2: np.ndarray,
              Resp: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Diagonalize search space, remove linearly dependent directions.

    Based on Skilling & Bryan Sect. 3.7.1, calculate metric g = e^T.(-gradgradS).e
    and diagonalize to get independent search basis.
    """
    # Calculate metric: g_mu,nu = e^T_mu.(-gradgradS).e_nu
    g = np.dot((-gradgradS * edir.T), edir)

    # Diagonalize metric
    gamma, eiVec = linalg.eigh(g)

    # Remove small eigenvalues (linear dependence criterion)
    maxGamma = np.max(np.abs(gamma))
    iok = gamma > 1e-8 * maxGamma

    # Project to new basis and normalize
    edir = np.dot(edir, eiVec[:, iok]) / np.sqrt(gamma[iok])
    nedir = edir.shape[1]

    # Calculate diagonal elements of M matrix: M_mu,nu = e^T_mu.grad(grad(C)).e_nu
    tmpRe = np.dot(Resp, edir)
    MM = 2.0 * np.dot((tmpRe.T / sig2), tmpRe)

    # Diagonalize M
    gammaM, eiVec = linalg.eigh(MM)

    # Project to M's eigenbasis
    edir = np.dot(edir, eiVec)

    return edir, nedir, gammaM


def _get_cmu_smu(gradC: np.ndarray, gradS: np.ndarray,
                 edir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate projection of gradients onto search basis."""
    Cmu = np.dot(gradC, edir)
    Smu = np.dot(gradS, edir)
    return Cmu, Smu


def _get_l0_squared(L_fac: float, Itot: float) -> float:
    """
    Calculate step length limit in parameter space.

    L_fac is typically 0.1-0.5, Itot is total image intensity.
    """
    return L_fac * Itot


def _get_alpha_min(gamma: np.ndarray) -> float:
    """
    Calculate minimum alpha value for Hessian constraint.

    If Hessian has negative eigenvalues, alphaMin = -min(gamma).
    """
    alphaMin = 0.0
    minGamma = np.min(gamma)
    if minGamma < 0.0:
        alphaMin = -minGamma
    return alphaMin


def _get_caim_quad(chiAim: float, C0: float, gamma: np.ndarray,
                   Cmu: np.ndarray) -> float:
    """
    Calculate quadratic approximation of target chi-squared (Skilling & Bryan Eq. 29).
    """
    Cminq = C0 - 0.5 * np.sum(Cmu * Cmu / gamma)
    # Skilling & Bryan: 0.667 and 0.333 are empirical constants
    Caimq = 0.66666667 * Cminq + 0.33333333 * C0
    if chiAim > Caimq:
        Caimq = chiAim
    return Caimq


def _get_saim_quad(Saim: float, S0: float, gamma: np.ndarray,
                   Smu: np.ndarray) -> float:
    """Calculate quadratic approximation of target entropy."""
    Sminq = S0 - 0.5 * np.sum(Smu * Smu / gamma)
    Saimq = 0.66666667 * Sminq + 0.33333333 * S0
    if Saim > Saimq:
        Saimq = Saim
    return Saimq


def _chop_down(alpha: float, alphaLow: float) -> Tuple[float, float]:
    """Bisection method: decrease alpha"""
    alphaHigh = alpha
    alpha = 0.5 * (alphaLow + alpha)
    return alpha, alphaHigh


def _chop_up(alpha: float, alphaHigh: float) -> Tuple[float, float]:
    """Bisection method: increase alpha"""
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
    Skilling-Bryan control process: fit to target chi-squared.

    Implements flowchart in Fig. 3, searching for alpha and P parameters via bisection.
    """
    # Allow chi-squared to relax slightly around target
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

            # Calculate x, chi-squared, and step length for quadratic approximation
            xqp = (alpha * Smu - Cmu) / (P + gamma + alpha)
            Cq = C0 + np.dot(Cmu, xqp) + 0.5 * np.sum(gamma * xqp * xqp)
            Cqp = C0 + np.dot(Cmu, xqp) + 0.5 * np.sum((P + gamma) * xqp * xqp)
            L2 = np.dot(xqp, xqp)

            # Adjust alpha based on chi-squared and step length
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

            # Convergence check
            if (alphaHigh > 0.0 and abs(alphaHigh - alphaLow)
                    < (convTol * alphaHigh + 1.0e-10)):
                afinished = 1
            if alpha > 1.0e20:
                afinished = 1

        # Adjust P value
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
    Skilling-Bryan control process: fit to target entropy.

    Similar to _control_chi2, but target is entropy.
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
    Skilling-Bryan convergence test statistic (Eq. 37).

    Measures the anti-parallelism of gradients, range [0, 0.5].
    Value close to 0 indicates close to optimal.
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
