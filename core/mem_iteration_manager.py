"""
core/iteration_manager.py - MEM Inversion Iteration Process Manager

Features:
  - IterationManager: Unified management of inversion iteration process
  - ConvergenceChecker: Efficient convergence checking
  - Integration of ProgressMonitor and IterationHistory
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from core.mem_monitoring import ProgressMonitor, IterationHistory


class ConvergenceChecker:
    """
    Efficient Convergence Checker
    
    Features:
    - Incremental convergence check (avoid repeated calculations)
    - Support multiple convergence criteria
    - Stagnation detection (continuous no progress)
    
    Attributes:
      threshold: float - Convergence threshold
      stall_count: int - Consecutive stagnation count
      prev_chi2: Optional[float] - Previous iteration chi-squared value
    """

    def __init__(self,
                 convergence_threshold: float = 1e-6,
                 stall_threshold: int = 5,
                 verbose: int = 0):
        """
        Initialize convergence checker
        
        Parameters:
          convergence_threshold: Relative change threshold (default 1e-6)
          stall_threshold: Stagnation iteration count threshold (default 5)
          verbose: Output level
        """
        self.threshold = convergence_threshold
        self.stall_threshold = stall_threshold
        self.verbose = verbose

        self.prev_chi2: Optional[float] = None
        self.stall_count: int = 0
        self.consecutive_small_changes: int = 0

    def check(self, chi2: float) -> Tuple[bool, str]:
        """
        Check if convergence conditions are met
        
        Parameters:
          chi2: Current chi-squared value
        
        Returns:
          (should_stop, reason)
          - should_stop: Whether to stop iteration
          - reason: Stop reason (or empty string)
        """
        # First iteration, cannot judge
        if self.prev_chi2 is None:
            self.prev_chi2 = chi2
            return False, ""

        # Calculate relative change
        chi2_change = abs(chi2 - self.prev_chi2) / (abs(self.prev_chi2) +
                                                    1e-20)

        # Check for convergence
        if chi2_change < self.threshold:
            self.consecutive_small_changes += 1
        else:
            self.consecutive_small_changes = 0

        # Consecutive small changes -> Convergence
        if self.consecutive_small_changes >= self.stall_threshold:
            return True, f"convergence (Δχ²={chi2_change:.3e})"

        self.prev_chi2 = chi2
        return False, ""

    def reset(self) -> None:
        """Reset checker state"""
        self.prev_chi2 = None
        self.stall_count = 0
        self.consecutive_small_changes = 0


class IterationManager:
    """
    Unified management of MEM inversion iteration process
    
    Features:
    - Centralized iteration counting and control
    - Integrated progress monitoring (ProgressMonitor)
    - Integrated iteration history (IterationHistory)
    - Unified convergence judgment (ConvergenceChecker)
    - Checkpoint management (optional)
    
    Attributes:
      iteration: int - Current iteration number (starting from 0)
      max_iterations: int - Maximum number of iterations
      convergence_reason: str - Stop reason
      total_elapsed: float - Total elapsed time (seconds)
    
    Integrated components:
      progress_monitor: ProgressMonitor (optional)
      iteration_history: IterationHistory (optional)
      convergence_checker: ConvergenceChecker (required)
    """

    def __init__(
        self,
        max_iterations: int,
        config: Optional[Dict[str, Any]] = None,
        use_progress_monitor: bool = True,
        use_iteration_history: bool = True,
        convergence_threshold: float = 1e-6,
        verbose: int = 0,
    ):
        """
        Initialize iteration manager
        
        Parameters:
          max_iterations: Maximum number of iterations
          config: Configuration dictionary (optional)
            - 'convergence_threshold': Convergence threshold
            - 'stall_threshold': Stagnation threshold
          use_progress_monitor: Whether to enable progress monitoring
          use_iteration_history: Whether to enable iteration history
          convergence_threshold: Convergence threshold (overrides config)
          verbose: Output level
        """
        self.iteration: int = 0
        self.max_iterations: int = max_iterations
        self.verbose: int = verbose
        self.convergence_reason: str = ""
        self.start_time: float = 0.0
        self.total_elapsed: float = 0.0

        # Extract configuration parameters
        config = config or {}
        actual_threshold = config.get('convergence_threshold',
                                      convergence_threshold)
        stall_threshold = config.get('stall_threshold', 5)

        # Initialize convergence checker (required)
        self.convergence_checker = ConvergenceChecker(
            convergence_threshold=actual_threshold,
            stall_threshold=stall_threshold,
            verbose=verbose)

        # Optional: Progress monitoring
        self.progress_monitor: Optional[ProgressMonitor] = None
        if use_progress_monitor and ProgressMonitor is not None:
            self.progress_monitor = ProgressMonitor(
                total_iterations=max_iterations,
                verbose=verbose,
                check_convergence=True)

        # Optional: Iteration history
        self.iteration_history: Optional[IterationHistory] = None
        if use_iteration_history and IterationHistory is not None:
            self.iteration_history = IterationHistory(verbose=verbose)

    def start_iteration(self) -> None:
        """
        Mark iteration start
        
        Should be called at the start of each iteration
        """
        if self.iteration == 0:
            import time
            self.start_time = time.time()

        if self.progress_monitor is not None:
            self.progress_monitor.on_iteration_start()

    def record_iteration(
        self,
        chi2: float,
        entropy: float,
        grad_S_norm: float = 0.0,
        grad_C_norm: float = 0.0,
        alpha: float = 0.0,
        param_delta: float = 0.0,
        diagnostics: Optional[Dict[str, Any]] = None,
        gradient_norm: Optional[float] = None,
    ) -> None:
        """
        Record results of a single iteration
        
        Parameters:
          chi2: Chi-squared statistic
          entropy: Entropy value
          grad_S_norm: Entropy gradient norm
          grad_C_norm: Constraint gradient norm
          alpha: Step size (optional)
          param_delta: Parameter change magnitude (optional)
          diagnostics: Diagnostic information (optional)
          gradient_norm: Compatibility parameter (optional, used for grad_C_norm if provided)
        
        Exceptions:
          ValueError: Any value is NaN/Inf
        """
        # Compatibility handling
        if gradient_norm is not None and grad_C_norm == 0.0:
            grad_C_norm = gradient_norm

        # Validate input
        values = {
            'chi2': chi2,
            'entropy': entropy,
            'grad_S_norm': grad_S_norm,
            'grad_C_norm': grad_C_norm,
        }

        for name, val in values.items():
            if not np.isfinite(val):
                raise ValueError(
                    f"Invalid iteration record: {name}={val} (must be finite)")

        # Update progress monitoring
        if self.progress_monitor is not None:
            self.progress_monitor.on_iteration_complete(entropy)

        # Update iteration history
        if self.iteration_history is not None:
            self.iteration_history.record_iteration(
                iteration=self.iteration,
                entropy=entropy,
                chisq=chi2,
                Q=chi2 - entropy,
                grad_S_norm=grad_S_norm,
                grad_C_norm=grad_C_norm,
                alpha=alpha,
                param_delta=param_delta,
                diagnostics=diagnostics or {},
            )

        self.iteration += 1

    def should_stop(self, chi2: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if iteration should stop
        
        Parameters:
          chi2: Current chi-squared value (for convergence check)
        
        Returns:
          (should_stop, reason)
          - should_stop: Whether to stop
          - reason: Stop reason
        """
        # Check maximum iterations
        if self.iteration >= self.max_iterations:
            reason = f"max_iterations reached ({self.iteration}/{self.max_iterations})"
            self.convergence_reason = reason
            return True, reason

        # Check convergence conditions
        if chi2 is not None:
            converged, reason = self.convergence_checker.check(chi2)
            if converged:
                self.convergence_reason = f"converged: {reason}"
                return True, reason

        return False, ""

    def get_summary(self) -> Dict[str, Any]:
        """
        获取迭代过程的总结
        
        返回：
          包含统计信息的字典
        """
        import time
        self.total_elapsed = time.time(
        ) - self.start_time if self.start_time > 0 else 0.0

        summary = {
            'iterations_completed':
            self.iteration,
            'max_iterations':
            self.max_iterations,
            'converged':
            len(self.convergence_reason) > 0
            and 'max_iterations' not in self.convergence_reason,
            'convergence_reason':
            self.convergence_reason,
            'total_elapsed_seconds':
            self.total_elapsed,
            'avg_time_per_iteration':
            self.total_elapsed / max(1, self.iteration),
        }

        # 附加来自子组件的摘要
        if self.progress_monitor is not None:
            summary['progress_summary'] = self.progress_monitor.get_summary()

        if self.iteration_history is not None:
            summary['history_summary'] = self.iteration_history.get_summary()

        return summary

    def get_iteration_history(self) -> Optional[IterationHistory]:
        """获取迭代历史对象（用于详细分析）"""
        return self.iteration_history

    def reset(self) -> None:
        """重置管理器到初始状态"""
        self.iteration = 0
        self.convergence_reason = ""
        self.start_time = 0.0
        self.total_elapsed = 0.0

        if self.progress_monitor is not None:
            # ProgressMonitor 没有 reset，需要创建新实例
            self.progress_monitor = ProgressMonitor(
                total_iterations=self.max_iterations,
                verbose=self.verbose,
                check_convergence=True)

        if self.iteration_history is not None:
            self.iteration_history = IterationHistory(verbose=self.verbose)

        self.convergence_checker.reset()


# ════════════════════════════════════════════════════════════════════════════
# 工作流辅助函数
# ════════════════════════════════════════════════════════════════════════════


def create_iteration_manager_from_config(
    config: Any,
    use_progress_monitor: bool = True,
    use_iteration_history: bool = True,
    verbose: int = 0,
) -> IterationManager:
    """
    从反演配置创建迭代管理器
    
    参数：
      config: InversionConfig 对象
      use_progress_monitor: 是否启用进度监控
      use_iteration_history: 是否启用迭代历史
      verbose: 输出级别
    
    返回：
      配置好的 IterationManager 实例
    """
    if isinstance(config, dict):
        max_iters = config.get('max_iterations') or config.get(
            'num_iterations', 100)
        convergence_threshold = config.get('convergence_threshold', 1e-6)
        stall_threshold = config.get('stall_threshold', 5)
    else:
        max_iters = getattr(config, 'max_iterations', None) or getattr(
            config, 'num_iterations', 100)
        convergence_threshold = getattr(config, 'convergence_threshold', 1e-6)
        stall_threshold = getattr(config, 'stall_threshold', 5)

    config_dict = {
        'convergence_threshold': convergence_threshold,
        'stall_threshold': stall_threshold,
    }

    return IterationManager(
        max_iterations=max_iters,
        config=config_dict,
        use_progress_monitor=use_progress_monitor,
        use_iteration_history=use_iteration_history,
        verbose=verbose,
    )


if __name__ == "__main__":
    # 简单演示
    import time

    print("=== IterationManager 演示 ===\n")

    # 创建管理器
    manager = IterationManager(max_iterations=20,
                               use_progress_monitor=True,
                               use_iteration_history=True,
                               verbose=1)

    # 模拟迭代过程
    for i in range(15):
        manager.start_iteration()

        # 模拟计算
        time.sleep(0.01)

        chi2 = 100.0 * np.exp(-i * 0.15)  # 指数衰减
        entropy = -float(i) * 5.0

        manager.record_iteration(
            chi2=chi2,
            entropy=entropy,
            grad_S_norm=np.exp(-i * 0.1),
            grad_C_norm=np.exp(-i * 0.15),
            alpha=0.1,
            param_delta=np.exp(-i * 0.2),
        )

        # 检查收敛条件
        should_stop, reason = manager.should_stop(chi2)
        if should_stop:
            print(f"  停止原因: {reason}")
            break

    # 获取总结
    summary = manager.get_summary()
    print("\n=== 管理器总结 ===")
    print(f"迭代完成数: {summary['iterations_completed']}")
    print(f"是否收敛: {summary['converged']}")
    print(f"收敛原因: {summary['convergence_reason']}")
    print(f"总耗时: {summary['total_elapsed_seconds']:.3f}s")
    print(f"平均耗时/迭代: {summary['avg_time_per_iteration']:.3f}s")
