"""
core/mem_monitoring.py - MEM 反演迭代监控模块

功能：
  - IterationHistory: 记录各迭代步的状态、参数变化、诊断信息
  - ProgressMonitor: 实时监控迭代进度、估计完成时间、生成摘要报告
"""

import time
import warnings
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# IterationHistory
# ════════════════════════════════════════════════════════════════════════════


class IterationHistory:
    """
    记录 MEM 反演各迭代步的详细历史
    
    记录内容：
      - 熵值 (S), 卡方 (chisq), 目标函数 (Q)
      - 梯度范数 (||grad_S||, ||grad_C||)
      - 步长 (alpha), 参数更新幅度
      - 诊断信息 (条件数、秩、收敛指标)
      - 计时信息 (单步时间、累积时间)
    
    属性:
      iterations: List[Dict] - 每迭代的记录
      start_time: float - 记录开始时间 (Unix timestamp)
      niter: int - 已记录的迭代数
    """

    def __init__(self, verbose: int = 0):
        """
        初始化迭代历史记录器
        
        参数:
          verbose: 0=无输出, 1=关键点输出, 2=详细输出
        """
        self.iterations: List[Dict[str, Any]] = []
        self.start_time: float = time.time()
        self.verbose = verbose
        self.niter: int = 0

    def record_iteration(
        self,
        iteration: int,
        entropy: float,
        chisq: float,
        Q: float,
        grad_S_norm: float,
        grad_C_norm: float,
        alpha: float,
        param_delta: float,
        diagnostics: Optional[Dict[str, Any]] = None,
        elapsed_iter: float = 0.0,
    ) -> None:
        """
        记录单次迭代的所有信息
        
        参数:
          iteration: 迭代编号
          entropy: 熵值 S
          chisq: 卡方 chi^2
          Q: 目标函数值 Q = chi^2 - lambda * S
          grad_S_norm: ||grad_S||
          grad_C_norm: ||grad_C||
          alpha: 步长
          param_delta: 参数变化幅度 ||p_{i+1} - p_i||
          diagnostics: 诊断信息字典（条件数、秩等）
          elapsed_iter: 单步耗时（秒）
        
        异常:
          ValueError: 任何数值为 NaN/Inf
        """
        # 验证输入
        values = {
            'entropy': entropy,
            'chisq': chisq,
            'Q': Q,
            'grad_S_norm': grad_S_norm,
            'grad_C_norm': grad_C_norm,
            'alpha': alpha,
            'param_delta': param_delta,
        }

        for name, val in values.items():
            if not np.isfinite(val):
                raise ValueError(
                    f"Invalid iteration record: {name}={val} (must be finite)")

        record = {
            'iteration': iteration,
            'entropy': entropy,
            'chisq': chisq,
            'Q': Q,
            'grad_S_norm': grad_S_norm,
            'grad_C_norm': grad_C_norm,
            'alpha': alpha,
            'param_delta': param_delta,
            'elapsed_iter': elapsed_iter,
            'cum_elapsed': time.time() - self.start_time,
            'timestamp': datetime.now(),
            'diagnostics': diagnostics or {},
        }

        self.iterations.append(record)
        self.niter = len(self.iterations)

        if self.verbose >= 1:
            self._print_iteration(record)

    def _print_iteration(self, record: Dict[str, Any]) -> None:
        """输出单次迭代的摘要"""
        msg = (f"iter {record['iteration']:4d}: "
               f"S={record['entropy']:12.6e} "
               f"χ²={record['chisq']:12.6e} "
               f"α={record['alpha']:8.2e} "
               f"Δp={record['param_delta']:8.2e}")
        warnings.warn(msg)

    def get_summary(self) -> str:
        """
        生成历史摘要
        
        返回:
          包含全局统计、收敛趋势、诊断摘要的字符串
        """
        if self.niter == 0:
            return "IterationHistory: no iterations recorded"

        S_vals = np.array([it['entropy'] for it in self.iterations])
        chisq_vals = np.array([it['chisq'] for it in self.iterations])
        alpha_vals = np.array([it['alpha'] for it in self.iterations])
        delta_vals = np.array([it['param_delta'] for it in self.iterations])

        # 计算收敛指标
        dS = np.diff(S_vals)

        summary = [
            f"IterationHistory Summary (n_iter={self.niter})",
            f"  Entropy:    {S_vals[0]:.6e} → {S_vals[-1]:.6e} (Δ={S_vals[-1] - S_vals[0]:.6e})",
            f"  Chi2:       {chisq_vals[0]:.6e} → {chisq_vals[-1]:.6e} (Δ={chisq_vals[-1] - chisq_vals[0]:.6e})",
            f"  Step size:  {np.min(alpha_vals):.2e} ~ {np.max(alpha_vals):.2e}",
            f"  Param Δ:    {np.min(delta_vals):.2e} ~ {np.max(delta_vals):.2e}",
            f"  dS trend:   {np.mean(dS[dS > 0]):.2e} (pos) / {np.mean(dS[dS < 0]):.2e} (neg)",
            f"  Total time: {self.iterations[-1]['cum_elapsed']:.2f}s",
        ]

        return "\n".join(summary)

    def get_last_iteration(self) -> Optional[Dict[str, Any]]:
        """返回最后一次迭代的记录"""
        return self.iterations[-1] if self.niter > 0 else None

    def save_checkpoint(self, filepath: str) -> None:
        """
        保存检查点到 NPZ 文件
        
        参数:
          filepath: 输出文件路径 (.npz)
        """
        data = {
            'iterations': np.array(self.iterations, dtype=object),
            'niter': np.array(self.niter),
            'start_time': np.array(self.start_time),
        }
        np.savez(filepath, **data)
        if self.verbose >= 1:
            warnings.warn(f"IterationHistory checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        从 NPZ 文件加载检查点
        
        参数:
          filepath: 输入文件路径 (.npz)
        """
        loaded = np.load(filepath, allow_pickle=True)
        self.iterations = loaded['iterations'].tolist()
        self.niter = int(loaded['niter'])
        self.start_time = float(loaded['start_time'])
        if self.verbose >= 1:
            warnings.warn(
                f"IterationHistory checkpoint loaded from {filepath} "
                f"({self.niter} iterations)")

    def get_history(self) -> Dict[str, List[float]]:
        """
        获取历史数据的字典格式
        
        Returns
        -------
        Dict[str, List[float]]
            包含 'chi2', 'entropy', 'regularization' 等键的列表字典
        """
        if not self.iterations:
            return {}

        history = {}
        keys = [
            'entropy', 'chisq', 'Q', 'grad_S_norm', 'grad_C_norm', 'alpha',
            'param_delta'
        ]

        for k in keys:
            history[k] = [it.get(k, 0.0) for it in self.iterations]

        # Compatibility aliases
        history['chi2'] = history['chisq']
        history['regularization'] = history['Q']

        return history


# ════════════════════════════════════════════════════════════════════════════
# ProgressMonitor
# ════════════════════════════════════════════════════════════════════════════


class ProgressMonitor:
    """
    实时监控 MEM 反演迭代进度
    
    功能：
      - 跟踪迭代计时
      - 估计完成时间 (ETA)
      - 监控收敛行为
      - 检测异常（停滞、发散）
      - 生成进度报告
    
    属性:
      iter_times: List[float] - 各迭代的耗时
      start_iter_time: float - 当前迭代开始时间
    """

    def __init__(
        self,
        total_iterations: int,
        verbose: int = 0,
        check_convergence: bool = True,
    ):
        """
        初始化进度监控器
        
        参数:
          total_iterations: 预期总迭代数
          verbose: 输出级别
          check_convergence: 是否监控收敛行为
        """
        self.total_iterations = total_iterations
        self.verbose = verbose
        self.check_convergence = check_convergence

        self.iter_times: List[float] = []
        self.iter_count: int = 0
        self.start_iter_time: float = 0.0
        self.convergence_stalled: int = 0  # 连续无进展的迭代数
        self.last_entropy: Optional[float] = None

    def on_iteration_start(self) -> None:
        """记录迭代开始"""
        self.start_iter_time = time.time()

    def on_iteration_complete(self, entropy: float) -> None:
        """
        记录迭代完成
        
        参数:
          entropy: 当前熵值（用于收敛检查）
        """
        elapsed = time.time() - self.start_iter_time
        self.iter_times.append(elapsed)
        self.iter_count += 1

        # 检查收敛
        if self.check_convergence:
            if self.last_entropy is not None:
                # 相对变化小于 1e-6 视为停滞
                rel_change = abs(entropy - self.last_entropy) / (
                    abs(self.last_entropy) + 1e-20)
                if rel_change < 1e-6:
                    self.convergence_stalled += 1
                else:
                    self.convergence_stalled = 0

        self.last_entropy = entropy

        if self.verbose >= 2:
            eta = self._estimate_eta()
            msg = f"  Iter {self.iter_count}/{self.total_iterations} completed in {elapsed:.3f}s (ETA: {eta})"
            warnings.warn(msg)

    def _estimate_eta(self) -> str:
        """
        估计完成时间
        
        返回:
          ETA 字符串，格式 "HH:MM:SS"
        """
        if len(self.iter_times) < 2:
            return "estimating..."

        # 使用最近 5 次迭代的平均时间
        window_size = min(5, len(self.iter_times))
        avg_time = np.mean(self.iter_times[-window_size:])

        remaining_iters = self.total_iterations - self.iter_count
        remaining_time = avg_time * remaining_iters

        eta_timedelta = timedelta(seconds=remaining_time)
        return str(eta_timedelta).split('.')[0]  # 移除毫秒

    def get_eta_seconds(self) -> float:
        """
        获取预计完成时间（秒）
        
        返回:
          剩余时间（秒），如无法估计返回 -1
        """
        if len(self.iter_times) < 2:
            return -1.0

        window_size = min(5, len(self.iter_times))
        avg_time = np.mean(self.iter_times[-window_size:])
        remaining_iters = self.total_iterations - self.iter_count

        return avg_time * remaining_iters

    def is_converged(self,
                     tol: float = 1e-6,
                     stall_iterations: int = 5) -> bool:
        """
        检查是否已收敛
        
        参数:
          tol: 相对变化容限
          stall_iterations: 认为已收敛的连续停滞迭代数
        
        返回:
          是否已收敛
        """
        return self.convergence_stalled >= stall_iterations

    def get_summary(self) -> str:
        """
        生成进度摘要
        
        返回:
          包含速度、进度、ETA 的摘要字符串
        """
        if self.iter_count == 0:
            return "ProgressMonitor: no iterations completed"

        total_time = np.sum(self.iter_times)
        avg_time = np.mean(self.iter_times)

        progress_pct = 100 * self.iter_count / self.total_iterations
        eta = self._estimate_eta()

        summary = [
            f"ProgressMonitor Summary",
            f"  Progress:    {self.iter_count}/{self.total_iterations} ({progress_pct:.1f}%)",
            f"  Total time:  {total_time:.2f}s",
            f"  Avg time/iter: {avg_time:.3f}s",
            f"  ETA:         {eta}",
            f"  Convergence: {self.convergence_stalled} consecutive stalled iters",
        ]

        return "\n".join(summary)


if __name__ == "__main__":
    # 简单演示
    history = IterationHistory(verbose=1)
    monitor = ProgressMonitor(total_iterations=10, verbose=1)

    for i in range(5):
        monitor.on_iteration_start()

        # 模拟迭代
        time.sleep(0.01)

        entropy = -float(i) * 10.0
        chisq = 100.0 - i * 5

        history.record_iteration(
            iteration=i,
            entropy=entropy,
            chisq=chisq,
            Q=chisq - entropy,
            grad_S_norm=np.exp(-i),
            grad_C_norm=np.exp(-i * 0.5),
            alpha=0.1,
            param_delta=np.exp(-i),
        )

        monitor.on_iteration_complete(entropy)

    print("\n" + history.get_summary())
    print("\n" + monitor.get_summary())
