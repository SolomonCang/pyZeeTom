"""
core/mem_optimization.py - MEM Optimization Toolkit

Contains core tools for high-performance inversion optimization:
  - ResponseMatrixCache: LRU cache for storing response matrices
  - DataPipeline: Data preprocessing pipeline
  - StabilityMonitor: Numerical stability monitoring

Features:
  ✓ Zero-copy cache mechanism to avoid redundant computations
  ✓ Standardized data pipeline for unified data processing
  ✓ Complete stability diagnostics to detect issues early

Usage Example:
  >>> from core.mem_optimization import ResponseMatrixCache, DataPipeline
  >>> cache = ResponseMatrixCache(max_size=5)
  >>> Resp = cache.get_or_compute(mag_field, obs_data, compute_fn)
"""

import hashlib
import numpy as np
import warnings
from typing import Callable, Tuple, Dict, Any, List
from dataclasses import dataclass

# ════════════════════════════════════════════════════════════════════════════
# ResponseMatrixCache: LRU Cache
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    size: int = 0
    max_size: int = 0
    memory_usage: int = 0  # bytes


class ResponseMatrixCache:
    """
    LRU cache for storing response matrices.
    
    Features:
    - Automatic LRU eviction
    - Cache key based on parameter hash
    - Cache hit rate statistics
    - Memory usage monitoring
    
    Usage Example:
    >>> cache = ResponseMatrixCache(max_size=5)
    >>> Resp = cache.get_or_compute(
    ...     mag_field, obs_data,
    ...     compute_fn=compute_response_matrix
    ... )
    >>> stats = cache.get_stats()
    >>> print(f"Hit rate: {stats.hit_rate:.1%}")
    """

    def __init__(self, max_size: int = 5, verbose: int = 0):
        """
        Initialize cache.
        
        Parameters:
            max_size: Maximum number of cache entries (recommended 3-10)
            verbose: Debug info verbosity level (0=none, 1=basic, 2=detailed)
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_order: List[str] = []  # LRU access order
        self.hits = 0
        self.misses = 0
        self.verbose = verbose

    def get_or_compute(self, mag_field: Any, obs_data: Any,
                       compute_fn: Callable[[Any], np.ndarray]) -> np.ndarray:
        """
        Get cached response matrix, compute if not present.
        
        Parameters:
            mag_field: Magnetic field parameter object (needs Blos, Bperp, chi attributes)
            obs_data: Observation data object (used for generating cache key)
            compute_fn: Computation function, signature compute_fn(mag_field) -> Resp
        
        Returns:
            Response matrix (Ndata, 3*Npix)
        
        raises:
            ValueError: If compute_fn return type is incorrect
        """
        key = self._make_key(mag_field, obs_data)

        if key in self.cache:
            self.hits += 1
            # Update access order (move to end)
            self.access_order.remove(key)
            self.access_order.append(key)

            if self.verbose >= 2:
                print(f"[Cache HIT] Key: {key[:16]}... | "
                      f"Total hits: {self.hits}")

            return self.cache[key]

        # Cache miss, compute
        self.misses += 1
        Resp = compute_fn(mag_field)

        # Validate output
        if not isinstance(Resp, np.ndarray):
            raise ValueError(
                f"compute_fn must return np.ndarray, got {type(Resp)}")

        # Ensure float array
        Resp = np.asarray(Resp, dtype=float)

        # Store in cache
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = Resp
        self.access_order.append(key)

        if self.verbose >= 1:
            print(f"[Cache MISS] Key: {key[:16]}... | "
                  f"Resp shape: {Resp.shape} | "
                  f"Cache size: {len(self.cache)}/{self.max_size}")

        return Resp

    def _make_key(self, mag_field: Any, obs_data: Any) -> str:
        """
        Generate cache key.
        
        Key based on:
        - Content hash of magnetic field parameters (Blos, Bperp, chi)
        - Object ID of observation data
        """
        try:
            # Parameter hash
            Blos_bytes = np.asarray(mag_field.Blos, dtype=float).tobytes()
            Bperp_bytes = np.asarray(mag_field.Bperp, dtype=float).tobytes()
            chi_bytes = np.asarray(mag_field.chi, dtype=float).tobytes()

            Blos_hash = hashlib.md5(Blos_bytes).hexdigest()
            Bperp_hash = hashlib.md5(Bperp_bytes).hexdigest()
            chi_hash = hashlib.md5(chi_bytes).hexdigest()
        except (AttributeError, TypeError) as e:
            raise ValueError(
                f"mag_field must have Blos, Bperp, chi attributes: {e}")

        # Observation ID (use object ID to avoid repeated hashing)
        obs_id = str(id(obs_data))

        # Combine key (simplified to first 16 chars of three hashes + obs_id)
        key = f"{Blos_hash[:8]}_{Bperp_hash[:8]}_{chi_hash[:8]}_{obs_id}"
        return key

    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if self.access_order:
            old_key = self.access_order.pop(0)
            del self.cache[old_key]
            if self.verbose >= 1:
                print(f"[Cache LRU] Evicted: {old_key[:16]}... | "
                      f"Cache size now: {len(self.cache)}")

    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0
        if self.verbose >= 1:
            print("[Cache CLEAR] All entries cleared")

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        # Estimate memory usage
        memory_usage = sum(arr.nbytes for arr in self.cache.values())

        return CacheStats(
            hits=self.hits,
            misses=self.misses,
            hit_rate=hit_rate,
            size=len(self.cache),
            max_size=self.max_size,
            memory_usage=memory_usage,
        )

    def __repr__(self) -> str:
        """Cache info string representation"""
        stats = self.get_stats()
        return (f"ResponseMatrixCache("
                f"size={stats.size}/{stats.max_size}, "
                f"hit_rate={stats.hit_rate:.1%}, "
                f"memory={stats.memory_usage/1e6:.1f}MB)")


# ════════════════════════════════════════════════════════════════════════════
# DataPipeline: Data Preprocessing Pipeline
# ════════════════════════════════════════════════════════════════════════════


class DataPipeline:
    """
    Standardized observation data preprocessing.
    
    Features:
    1. Data consistency check (wavelength match, valid noise)
    2. Data packing and index management
    3. Memory efficient pre-allocation
    4. Support selective Stokes component fitting
    
    Usage Example:
    >>> pipeline = DataPipeline(observations, fit_I=True, fit_V=True)
    >>> pipeline.pack_data()
    >>> Data = pipeline.Data
    >>> sig2 = pipeline.sig2
    """

    def __init__(
        self,
        observations: List[Any],
        fit_I: bool = True,
        fit_V: bool = True,
        fit_Q: bool = False,
        fit_U: bool = False,
        verbose: int = 0,
    ):
        """
        Initialize data pipeline.
        
        Parameters:
            observations: List of observation data (each should have wl, specI, specV etc. attributes)
            fit_I/V/Q/U: Whether to fit each component
            verbose: Info verbosity level
        
        raises:
            ValueError: If data is inconsistent or invalid
        """
        self.obs = observations
        self.fit_I = fit_I
        self.fit_V = fit_V
        self.fit_Q = fit_Q
        self.fit_U = fit_U
        self.verbose = verbose

        # Validate and preprocess
        self._validate()
        self._preprocess()

    def _validate(self) -> None:
        """Data consistency check"""
        if not self.obs:
            raise ValueError("Observations list is empty")

        # Check all observation wavelengths match
        wl_ref = np.asarray(self.obs[0].wl, dtype=float)
        for i, obs in enumerate(self.obs[1:], 1):
            wl_curr = np.asarray(obs.wl, dtype=float)
            if not np.allclose(wl_curr, wl_ref, rtol=1e-10):
                raise ValueError(
                    f"Observation {i}: wavelength grid differs from reference")

        # Check noise validity
        for i, obs in enumerate(self.obs):
            if self.fit_I:
                sig_I = np.asarray(obs.specI_sig, dtype=float)
                if np.any(sig_I <= 0):
                    raise ValueError(f"Obs {i}: Invalid I noise (must be > 0)")
            if self.fit_V:
                sig_V = np.asarray(obs.specV_sig, dtype=float)
                if np.any(sig_V <= 0):
                    raise ValueError(f"Obs {i}: Invalid V noise (must be > 0)")

        if self.verbose >= 1:
            print(f"[DataPipeline] ✓ Validated {len(self.obs)} observations")

    def _preprocess(self) -> None:
        """Preprocess and pre-allocate"""
        self.nwl = len(np.asarray(self.obs[0].wl, dtype=float))
        self.nobs = len(self.obs)

        # Calculate total data points
        ncomp = sum([self.fit_I, self.fit_V, self.fit_Q, self.fit_U])
        self.ndata_total = self.nwl * self.nobs * ncomp

        # Pre-allocate arrays
        self._Data = np.zeros(self.ndata_total, dtype=float)
        self._Fmodel = np.zeros(self.ndata_total, dtype=float)
        self._sig2 = np.zeros(self.ndata_total, dtype=float)

        # Build index map
        self._build_index_map()

        if self.verbose >= 1:
            print(f"[DataPipeline] Total data points: {self.ndata_total} "
                  f"({self.nobs} obs × {self.nwl} wl × {ncomp} comp)")

    def _build_index_map(self) -> None:
        """Build (obs_idx, wl_idx, component) -> data_idx map"""
        self.index_map: Dict[Tuple[int, int, str], int] = {}
        self.reverse_map: Dict[int, Tuple[int, int, str]] = {}

        idx = 0
        components = []
        if self.fit_I:
            components.append('I')
        if self.fit_V:
            components.append('V')
        if self.fit_Q:
            components.append('Q')
        if self.fit_U:
            components.append('U')

        for obs_idx in range(self.nobs):
            for wl_idx in range(self.nwl):
                for comp in components:
                    self.index_map[(obs_idx, wl_idx, comp)] = idx
                    self.reverse_map[idx] = (obs_idx, wl_idx, comp)
                    idx += 1

    def pack_data(self) -> None:
        """Pack all observation data into pre-allocated arrays"""
        for obs_idx, obs in enumerate(self.obs):
            # Convert to numpy arrays
            specI = np.asarray(obs.specI, dtype=float)
            specV = np.asarray(obs.specV, dtype=float) if self.fit_V else None
            specQ = np.asarray(obs.specQ, dtype=float) if self.fit_Q else None
            specU = np.asarray(obs.specU, dtype=float) if self.fit_U else None

            specI_sig = np.asarray(obs.specI_sig, dtype=float)
            specV_sig = np.asarray(obs.specV_sig,
                                   dtype=float) if self.fit_V else None

            for wl_idx in range(self.nwl):
                if self.fit_I:
                    idx = self.index_map[(obs_idx, wl_idx, 'I')]
                    self._Data[idx] = specI[wl_idx]
                    self._sig2[idx] = specI_sig[wl_idx]**2

                if self.fit_V:
                    idx = self.index_map[(obs_idx, wl_idx, 'V')]
                    self._Data[idx] = specV[wl_idx]
                    self._sig2[idx] = specV_sig[wl_idx]**2

                if self.fit_Q:
                    idx = self.index_map[(obs_idx, wl_idx, 'Q')]
                    self._Data[idx] = specQ[wl_idx]
                    self._sig2[idx] = specI_sig[wl_idx]**2  # Q/U use I noise

                if self.fit_U:
                    idx = self.index_map[(obs_idx, wl_idx, 'U')]
                    self._Data[idx] = specU[wl_idx]
                    self._sig2[idx] = specI_sig[wl_idx]**2

        if self.verbose >= 1:
            print(f"[DataPipeline] ✓ Data packed successfully")

    @property
    def Data(self) -> np.ndarray:
        """Observation data vector"""
        return self._Data

    @property
    def sig2(self) -> np.ndarray:
        """Noise variance vector"""
        return self._sig2

    @property
    def Fmodel(self) -> np.ndarray:
        """Model prediction vector (updated by inversion loop)"""
        return self._Fmodel

    def set_Fmodel(self, fmodel: np.ndarray) -> None:
        """Set model prediction"""
        if fmodel.shape != self._Fmodel.shape:
            raise ValueError(f"Fmodel shape mismatch: got {fmodel.shape}, "
                             f"expected {self._Fmodel.shape}")
        self._Fmodel[:] = fmodel


# ════════════════════════════════════════════════════════════════════════════
# StabilityMonitor: Numerical Stability Monitoring
# ════════════════════════════════════════════════════════════════════════════


class StabilityMonitor:
    """
    Monitor numerical stability during MEM optimization.
    
    Detection items:
    - Gradient saturation (gradient all zero or extremely large)
    - Singular search directions
    - Response matrix condition number
    - Step size too small or too large
    - NaN/Inf detection
    
    Usage Example:
    >>> monitor = StabilityMonitor(verbose=1)
    >>> if not monitor.check_gradient(gradC, gradS):
    ...     print("Warning: gradient issue detected")
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize stability monitor.
        
        Parameters:
            verbose: Output level (0=none, 1=warning, 2=detailed)
        """
        self.verbose = verbose
        self.warnings: List[str] = []

    def check_gradient(self,
                       gradC: np.ndarray,
                       gradS: np.ndarray,
                       tol: float = 1e-10) -> bool:
        """
        Check gradient health.
        
        Check items:
        - Whether gradient is all zero (convergence plateau)
        - Whether NaN/Inf exists (numerical overflow)
        - Whether gradient norm is too large (potential overflow)
        
        Returns:
            True means gradient is normal, False means issue detected
        """
        gradC = np.asarray(gradC, dtype=float)
        gradS = np.asarray(gradS, dtype=float)

        # Check all zero
        if np.allclose(gradC, 0, atol=tol) and np.allclose(gradS, 0, atol=tol):
            msg = "Gradient near-zero (possible convergence plateau)"
            if self.verbose >= 1:
                warnings.warn(msg)
            self.warnings.append(msg)
            return False

        # Check NaN/Inf
        if np.any(np.isnan(gradC)) or np.any(np.isnan(gradS)):
            msg = "NaN detected in gradients"
            if self.verbose >= 1:
                warnings.warn(msg)
            self.warnings.append(msg)
            return False

        if np.any(np.isinf(gradC)) or np.any(np.isinf(gradS)):
            msg = "Inf detected in gradients"
            if self.verbose >= 1:
                warnings.warn(msg)
            self.warnings.append(msg)
            return False

        # Check extreme values
        max_gradC = np.max(np.abs(gradC))
        max_gradS = np.max(np.abs(gradS))
        max_grad = max(max_gradC, max_gradS)

        if max_grad > 1e10:
            msg = f"Gradient norm very large: max={max_grad:.3e}"
            if self.verbose >= 1:
                warnings.warn(msg)
            self.warnings.append(msg)
            return False

        return True

    def check_response_matrix(self,
                              Resp: np.ndarray,
                              tol_cond: float = 1e10) -> Dict[str, Any]:
        """
        Check response matrix quality.
        
        Check items:
        - Condition number (numerical stability)
        - Effective rank (linear independence)
        - Eigenvalue distribution
        
        Returns:
            Diagnostic info dictionary (contains condition_number, effective_rank etc.)
        """
        Resp = np.asarray(Resp, dtype=float)
        diagnostics: Dict[str, Any] = {}

        # Calculate condition number
        try:
            U, s, Vt = np.linalg.svd(Resp, full_matrices=False)
            cond_num = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
            diagnostics['condition_number'] = float(cond_num)
            diagnostics['singular_values'] = s

            if cond_num > tol_cond:
                msg = f"Response matrix ill-conditioned: kappa={cond_num:.3e}"
                if self.verbose >= 1:
                    warnings.warn(msg)
                self.warnings.append(msg)
        except np.linalg.LinAlgError as e:
            diagnostics['condition_number'] = np.inf
            msg = f"SVD failed to converge: {e}"
            if self.verbose >= 1:
                warnings.warn(msg)
            self.warnings.append(msg)

        # Check rank
        rank = np.linalg.matrix_rank(Resp)
        diagnostics['effective_rank'] = int(rank)

        if rank < min(Resp.shape) // 2:
            msg = f"Low effective rank: {rank}/{min(Resp.shape)}"
            if self.verbose >= 1:
                warnings.warn(msg)
            self.warnings.append(msg)

        return diagnostics

    def check_step_length(self,
                          step_length: float,
                          min_step: float = 1e-15,
                          max_step: float = 1.0) -> bool:
        """
        Check step length rationality.
        
        Parameters:
            step_length: Current step length
            min_step: Minimum step threshold
            max_step: Maximum step threshold
        
        Returns:
            True means step length is reasonable, False means out of range
        """
        if step_length < min_step:
            msg = f"Step size too small: {step_length:.3e}"
            if self.verbose >= 1:
                warnings.warn(msg)
            self.warnings.append(msg)
            return False

        if step_length > max_step:
            msg = f"Step size too large: {step_length:.3e}"
            if self.verbose >= 1:
                warnings.warn(msg)
            self.warnings.append(msg)
            return False

        return True

    def get_summary(self) -> str:
        """Get diagnostic summary"""
        if not self.warnings:
            return "[StabilityMonitor] ✓ All checks passed"

        summary = f"[StabilityMonitor] {len(self.warnings)} warning(s) detected:\n"
        for i, w in enumerate(self.warnings, 1):
            summary += f"  {i}. {w}\n"

        return summary

    def clear(self) -> None:
        """Clear warning list"""
        self.warnings.clear()
