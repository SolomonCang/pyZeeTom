"""
core/mem_optimization.py - MEM 优化工具集

包含高性能反演优化的核心工具：
  - ResponseMatrixCache: LRU 缓存存储响应矩阵
  - DataPipeline: 数据预处理流水线
  - StabilityMonitor: 数值稳定性监控

特点：
  ✓ 零-复制缓存机制，避免重复计算
  ✓ 标准化数据流水线，统一数据处理
  ✓ 完整的稳定性诊断，提前发现问题

用法示例：
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
# ResponseMatrixCache: LRU 缓存
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    size: int = 0
    max_size: int = 0
    memory_usage: int = 0  # bytes


class ResponseMatrixCache:
    """
    LRU 缓存存储响应矩阵。
    
    特点：
    - 自动 LRU 淘汰
    - 缓存键基于参数哈希
    - 统计缓存命中率
    - 内存使用监控
    
    使用示例：
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
        初始化缓存。
        
        参数：
            max_size: 最大缓存条目数（推荐 3-10）
            verbose: 调试信息输出级别（0=无，1=基本，2=详细）
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_order: List[str] = []  # LRU 访问顺序
        self.hits = 0
        self.misses = 0
        self.verbose = verbose

    def get_or_compute(self, mag_field: Any, obs_data: Any,
                       compute_fn: Callable[[Any], np.ndarray]) -> np.ndarray:
        """
        获取缓存的响应矩阵，如无则计算。
        
        参数：
            mag_field: 磁场参数对象（需要 Blos, Bperp, chi 属性）
            obs_data: 观测数据对象（用于生成缓存键）
            compute_fn: 计算函数，签名为 compute_fn(mag_field) -> Resp
        
        返回：
            响应矩阵 (Ndata, 3*Npix)
        
        raises：
            ValueError: 如果 compute_fn 返回值类型不对
        """
        key = self._make_key(mag_field, obs_data)

        if key in self.cache:
            self.hits += 1
            # 更新访问顺序（移到末尾）
            self.access_order.remove(key)
            self.access_order.append(key)

            if self.verbose >= 2:
                print(f"[Cache HIT] Key: {key[:16]}... | "
                      f"Total hits: {self.hits}")

            return self.cache[key]

        # 缓存未命中，计算
        self.misses += 1
        Resp = compute_fn(mag_field)

        # 验证输出
        if not isinstance(Resp, np.ndarray):
            raise ValueError(
                f"compute_fn must return np.ndarray, got {type(Resp)}")

        # 确保是浮点数组
        Resp = np.asarray(Resp, dtype=float)

        # 存储到缓存
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
        生成缓存键。
        
        键基于：
        - 磁场参数的内容哈希（Blos, Bperp, chi）
        - 观测数据的对象 ID
        """
        try:
            # 参数哈希
            Blos_bytes = np.asarray(mag_field.Blos, dtype=float).tobytes()
            Bperp_bytes = np.asarray(mag_field.Bperp, dtype=float).tobytes()
            chi_bytes = np.asarray(mag_field.chi, dtype=float).tobytes()

            Blos_hash = hashlib.md5(Blos_bytes).hexdigest()
            Bperp_hash = hashlib.md5(Bperp_bytes).hexdigest()
            chi_hash = hashlib.md5(chi_bytes).hexdigest()
        except (AttributeError, TypeError) as e:
            raise ValueError(
                f"mag_field must have Blos, Bperp, chi attributes: {e}")

        # 观测 ID（使用对象 ID，避免重复哈希）
        obs_id = str(id(obs_data))

        # 组合键（简化为前 16 字符的三个哈希 + obs_id）
        key = f"{Blos_hash[:8]}_{Bperp_hash[:8]}_{chi_hash[:8]}_{obs_id}"
        return key

    def _evict_lru(self) -> None:
        """淘汰最近最少使用的条目"""
        if self.access_order:
            old_key = self.access_order.pop(0)
            del self.cache[old_key]
            if self.verbose >= 1:
                print(f"[Cache LRU] Evicted: {old_key[:16]}... | "
                      f"Cache size now: {len(self.cache)}")

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0
        if self.verbose >= 1:
            print("[Cache CLEAR] All entries cleared")

    def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        # 估算内存使用
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
        """缓存信息字符串表示"""
        stats = self.get_stats()
        return (f"ResponseMatrixCache("
                f"size={stats.size}/{stats.max_size}, "
                f"hit_rate={stats.hit_rate:.1%}, "
                f"memory={stats.memory_usage/1e6:.1f}MB)")


# ════════════════════════════════════════════════════════════════════════════
# DataPipeline: 数据预处理流水线
# ════════════════════════════════════════════════════════════════════════════


class DataPipeline:
    """
    标准化观测数据预处理。
    
    功能：
    1. 数据一致性检查（波长相同、噪声有效）
    2. 数据打包与索引管理
    3. 内存高效预分配
    4. 支持选择性 Stokes 分量拟合
    
    使用示例：
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
        初始化数据流水线。
        
        参数：
            observations: 观测数据列表（每个应有 wl, specI, specV 等属性）
            fit_I/V/Q/U: 是否拟合各分量
            verbose: 信息输出级别
        
        raises：
            ValueError: 如果数据不一致或无效
        """
        self.obs = observations
        self.fit_I = fit_I
        self.fit_V = fit_V
        self.fit_Q = fit_Q
        self.fit_U = fit_U
        self.verbose = verbose

        # 验证并预处理
        self._validate()
        self._preprocess()

    def _validate(self) -> None:
        """数据一致性检查"""
        if not self.obs:
            raise ValueError("Observations list is empty")

        # 检查所有观测波长相同
        wl_ref = np.asarray(self.obs[0].wl, dtype=float)
        for i, obs in enumerate(self.obs[1:], 1):
            wl_curr = np.asarray(obs.wl, dtype=float)
            if not np.allclose(wl_curr, wl_ref, rtol=1e-10):
                raise ValueError(
                    f"Observation {i}: wavelength grid differs from reference")

        # 检查噪声有效性
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
        """预处理与预分配"""
        self.nwl = len(np.asarray(self.obs[0].wl, dtype=float))
        self.nobs = len(self.obs)

        # 计算总数据点数
        ncomp = sum([self.fit_I, self.fit_V, self.fit_Q, self.fit_U])
        self.ndata_total = self.nwl * self.nobs * ncomp

        # 预分配数组
        self._Data = np.zeros(self.ndata_total, dtype=float)
        self._Fmodel = np.zeros(self.ndata_total, dtype=float)
        self._sig2 = np.zeros(self.ndata_total, dtype=float)

        # 构建索引映射
        self._build_index_map()

        if self.verbose >= 1:
            print(f"[DataPipeline] Total data points: {self.ndata_total} "
                  f"({self.nobs} obs × {self.nwl} wl × {ncomp} comp)")

    def _build_index_map(self) -> None:
        """构建 (obs_idx, wl_idx, component) -> data_idx 映射"""
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
        """打包所有观测数据到预分配数组"""
        for obs_idx, obs in enumerate(self.obs):
            # 转换为 numpy 数组
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
                    self._sig2[idx] = specI_sig[wl_idx]**2  # Q/U 用 I 噪声

                if self.fit_U:
                    idx = self.index_map[(obs_idx, wl_idx, 'U')]
                    self._Data[idx] = specU[wl_idx]
                    self._sig2[idx] = specI_sig[wl_idx]**2

        if self.verbose >= 1:
            print(f"[DataPipeline] ✓ Data packed successfully")

    @property
    def Data(self) -> np.ndarray:
        """观测数据向量"""
        return self._Data

    @property
    def sig2(self) -> np.ndarray:
        """噪声方差向量"""
        return self._sig2

    @property
    def Fmodel(self) -> np.ndarray:
        """模型预测向量（由反演循环更新）"""
        return self._Fmodel

    def set_Fmodel(self, fmodel: np.ndarray) -> None:
        """设置模型预测"""
        if fmodel.shape != self._Fmodel.shape:
            raise ValueError(f"Fmodel shape mismatch: got {fmodel.shape}, "
                             f"expected {self._Fmodel.shape}")
        self._Fmodel[:] = fmodel


# ════════════════════════════════════════════════════════════════════════════
# StabilityMonitor: 数值稳定性监控
# ════════════════════════════════════════════════════════════════════════════


class StabilityMonitor:
    """
    监控 MEM 优化过程中的数值稳定性。
    
    检测项目：
    - 梯度饱和（梯度全为零或极大）
    - 奇异搜索方向
    - 响应矩阵条件数
    - 步长过小或过大
    - NaN/Inf 检测
    
    使用示例：
    >>> monitor = StabilityMonitor(verbose=1)
    >>> if not monitor.check_gradient(gradC, gradS):
    ...     print("Warning: gradient issue detected")
    """

    def __init__(self, verbose: int = 0):
        """
        初始化稳定性监控器。
        
        参数：
            verbose: 输出级别（0=无，1=警告，2=详细）
        """
        self.verbose = verbose
        self.warnings: List[str] = []

    def check_gradient(self,
                       gradC: np.ndarray,
                       gradS: np.ndarray,
                       tol: float = 1e-10) -> bool:
        """
        检查梯度健康性。
        
        检查项：
        - 梯度是否全零（收敛平台）
        - 是否存在 NaN/Inf（数值溢出）
        - 梯度范数是否过大（潜在溢出）
        
        返回：
            True 表示梯度正常，False 表示检测到问题
        """
        gradC = np.asarray(gradC, dtype=float)
        gradS = np.asarray(gradS, dtype=float)

        # 检查全零
        if np.allclose(gradC, 0, atol=tol) and np.allclose(gradS, 0, atol=tol):
            msg = "Gradient near-zero (possible convergence plateau)"
            if self.verbose >= 1:
                warnings.warn(msg)
            self.warnings.append(msg)
            return False

        # 检查 NaN/Inf
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

        # 检查极值
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
        检查响应矩阵质量。
        
        检查项：
        - 条件数（数值稳定性）
        - 有效秩（线性独立性）
        - 特征值分布
        
        返回：
            诊断信息字典（包含 condition_number, effective_rank 等）
        """
        Resp = np.asarray(Resp, dtype=float)
        diagnostics: Dict[str, Any] = {}

        # 计算条件数
        try:
            U, s, Vt = np.linalg.svd(Resp, full_matrices=False)
            cond_num = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
            diagnostics['condition_number'] = float(cond_num)
            diagnostics['singular_values'] = s

            if cond_num > tol_cond:
                msg = f"Response matrix ill-conditioned: κ={cond_num:.3e}"
                if self.verbose >= 1:
                    warnings.warn(msg)
                self.warnings.append(msg)
        except np.linalg.LinAlgError as e:
            diagnostics['condition_number'] = np.inf
            msg = f"SVD failed to converge: {e}"
            if self.verbose >= 1:
                warnings.warn(msg)
            self.warnings.append(msg)

        # 检查秩
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
        检查步长合理性。
        
        参数：
            step_length: 当前步长
            min_step: 最小步长阈值
            max_step: 最大步长阈值
        
        返回：
            True 表示步长合理，False 表示超出范围
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
        """获取诊断摘要"""
        if not self.warnings:
            return "[StabilityMonitor] ✓ All checks passed"

        summary = f"[StabilityMonitor] {len(self.warnings)} warning(s) detected:\n"
        for i, w in enumerate(self.warnings, 1):
            summary += f"  {i}. {w}\n"

        return summary

    def clear(self) -> None:
        """清空警告列表"""
        self.warnings.clear()
