"""
core/mem_monitoring.py - MEM Inversion Iteration Monitoring Module

Features:
  - IterationHistory: Records state, parameter changes, diagnostic info for each iteration step
  - ProgressMonitor: Real-time monitoring of iteration progress, ETA estimation, summary report generation
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
    Records detailed history of each MEM inversion iteration step
    
    Recorded content:
      - Entropy (S), Chi-squared (chisq), Objective function (Q)
      - Gradient norms (||grad_S||, ||grad_C||)
      - Step size (alpha), Parameter update magnitude
      - Diagnostic info (condition number, rank, convergence metrics)
      - Timing info (step time, cumulative time)
    
    Attributes:
      iterations: List[Dict] - Records per iteration
      start_time: float - Recording start time (Unix timestamp)
      niter: int - Number of recorded iterations
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize iteration history recorder
        
        Parameters:
          verbose: 0=No output, 1=Key points output, 2=Detailed output
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
        Record all information for a single iteration
        
        Parameters:
          iteration: Iteration number
          entropy: Entropy value S
          chisq: Chi-squared chi^2
          Q: Objective function value Q = chi^2 - lambda * S
          grad_S_norm: ||grad_S||
          grad_C_norm: ||grad_C||
          alpha: Step size
          param_delta: Parameter change magnitude ||p_{i+1} - p_i||
          diagnostics: Diagnostic info dictionary (condition number, rank, etc.)
          elapsed_iter: Single step elapsed time (seconds)
        
        Exceptions:
          ValueError: Any value is NaN/Inf
        """
        # Validate input
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
        """Output summary of a single iteration"""
        msg = (f"iter {record['iteration']:4d}: "
               f"S={record['entropy']:12.6e} "
               f"χ²={record['chisq']:12.6e} "
               f"α={record['alpha']:8.2e} "
               f"Δp={record['param_delta']:8.2e}")
        warnings.warn(msg)

    def get_summary(self) -> str:
        """
        Generate history summary
        
        Returns:
          String containing global statistics, convergence trends, diagnostic summary
        """
        if self.niter == 0:
            return "IterationHistory: no iterations recorded"

        S_vals = np.array([it['entropy'] for it in self.iterations])
        chisq_vals = np.array([it['chisq'] for it in self.iterations])
        alpha_vals = np.array([it['alpha'] for it in self.iterations])
        delta_vals = np.array([it['param_delta'] for it in self.iterations])

        # Calculate convergence metrics
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
        """Return the record of the last iteration"""
        return self.iterations[-1] if self.niter > 0 else None

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save checkpoint to NPZ file
        
        Parameters:
          filepath: Output file path (.npz)
        """
        np.savez(filepath,
                 iterations=np.array(self.iterations, dtype=object),
                 niter=np.array(self.niter),
                 start_time=np.array(self.start_time),
                 allow_pickle=True)
        if self.verbose >= 1:
            warnings.warn(f"IterationHistory checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load checkpoint from NPZ file
        
        Parameters:
          filepath: Input file path (.npz)
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
        Get history data in dictionary format
        
        Returns
        -------
        Dict[str, List[float]]
            List dictionary containing keys like 'chi2', 'entropy', 'regularization'
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
    Real-time monitoring of MEM inversion iteration progress
    
    Features:
      - Track iteration timing
      - Estimate completion time (ETA)
      - Monitor convergence behavior
      - Detect anomalies (stagnation, divergence)
      - Generate progress report
    
    Attributes:
      iter_times: List[float] - Elapsed time for each iteration
      start_iter_time: float - Current iteration start time
    """

    def __init__(
        self,
        total_iterations: int,
        verbose: int = 0,
        check_convergence: bool = True,
    ):
        """
        Initialize progress monitor
        
        Parameters:
          total_iterations: Expected total iterations
          verbose: Output level
          check_convergence: Whether to monitor convergence behavior
        """
        self.total_iterations = total_iterations
        self.verbose = verbose
        self.check_convergence = check_convergence

        self.iter_times: List[float] = []
        self.iter_count: int = 0
        self.start_iter_time: float = 0.0
        self.convergence_stalled: int = 0  # Number of consecutive iterations with no progress
        self.last_entropy: Optional[float] = None

    def on_iteration_start(self) -> None:
        """Record iteration start"""
        self.start_iter_time = time.time()

    def on_iteration_complete(self, entropy: float) -> None:
        """
        Record iteration completion
        
        Parameters:
          entropy: Current entropy value (for convergence check)
        """
        elapsed = time.time() - self.start_iter_time
        self.iter_times.append(elapsed)
        self.iter_count += 1

        # Check convergence
        if self.check_convergence:
            if self.last_entropy is not None:
                # Relative change less than 1e-6 considered stagnation
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
        Estimate completion time
        
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

        eta_timedelta = timedelta(seconds=float(remaining_time))
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

        return float(avg_time * remaining_iters)

    def is_converged(self,
                     tol: float = 1e-3,
                     stall_iterations: int = 3) -> bool:
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
            "ProgressMonitor Summary",
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
