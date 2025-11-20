"""
core/iteration_manager.py - MEM 反演迭代过程管理器

功能：
  - IterationManager: 统一管理反演迭代过程
  - ConvergenceChecker: 高效的收敛检查
  - 集成 ProgressMonitor 和 IterationHistory
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

# 可选导入优化组件
try:
    from core.mem_monitoring import ProgressMonitor, IterationHistory
except ImportError:
    ProgressMonitor = None
    IterationHistory = None


class ConvergenceChecker:
    """
    高效的收敛判定器
    
    功能：
    - 增量式收敛检查（避免重复计算）
    - 支持多种收敛准则
    - 停滞检测（连续无进展）
    
    属性：
      threshold: float - 收敛阈值
      stall_count: int - 连续停滞次数
      prev_chi2: Optional[float] - 前一迭代的χ²值
    """

    def __init__(self,
                 convergence_threshold: float = 1e-6,
                 stall_threshold: int = 5,
                 verbose: int = 0):
        """
        初始化收敛检查器
        
        参数：
          convergence_threshold: 相对变化阈值（默认 1e-6）
          stall_threshold: 停滞迭代数阈值（默认 5）
          verbose: 输出级别
        """
        self.threshold = convergence_threshold
        self.stall_threshold = stall_threshold
        self.verbose = verbose

        self.prev_chi2: Optional[float] = None
        self.stall_count: int = 0
        self.consecutive_small_changes: int = 0

    def check(self, chi2: float) -> Tuple[bool, str]:
        """
        检查是否满足收敛条件
        
        参数：
          chi2: 当前χ²值
        
        返回：
          (should_stop, reason)
          - should_stop: 是否应停止迭代
          - reason: 停止原因（或空字符串）
        """
        # 首次迭代，无法判断
        if self.prev_chi2 is None:
            self.prev_chi2 = chi2
            return False, ""

        # 计算相对变化
        chi2_change = abs(chi2 - self.prev_chi2) / (abs(self.prev_chi2) +
                                                    1e-20)

        # 检查是否收敛
        if chi2_change < self.threshold:
            self.consecutive_small_changes += 1
        else:
            self.consecutive_small_changes = 0

        # 连续多次小变化 → 收敛
        if self.consecutive_small_changes >= self.stall_threshold:
            return True, f"convergence (Δχ²={chi2_change:.3e})"

        self.prev_chi2 = chi2
        return False, ""

    def reset(self) -> None:
        """重置检查器状态"""
        self.prev_chi2 = None
        self.stall_count = 0
        self.consecutive_small_changes = 0


class IterationManager:
    """
    统一管理 MEM 反演迭代过程
    
    功能：
    - 集中化迭代计数和控制
    - 集成进度监控（ProgressMonitor）
    - 集成迭代历史（IterationHistory）
    - 统一收敛判定（ConvergenceChecker）
    - 检查点管理（可选）
    
    属性：
      iteration: int - 当前迭代编号（从 0 开始）
      max_iterations: int - 最大迭代数
      convergence_reason: str - 停止原因
      total_elapsed: float - 总耗时（秒）
    
    集成组件：
      progress_monitor: ProgressMonitor (可选)
      iteration_history: IterationHistory (可选)
      convergence_checker: ConvergenceChecker (必需)
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
        初始化迭代管理器
        
        参数：
          max_iterations: 最大迭代数
          config: 配置字典（可选）
            - 'convergence_threshold': 收敛阈值
            - 'stall_threshold': 停滞阈值
          use_progress_monitor: 是否启用进度监控
          use_iteration_history: 是否启用迭代历史
          convergence_threshold: 收敛阈值（覆盖 config）
          verbose: 输出级别
        """
        self.iteration: int = 0
        self.max_iterations: int = max_iterations
        self.verbose: int = verbose
        self.convergence_reason: str = ""
        self.start_time: float = 0.0
        self.total_elapsed: float = 0.0

        # 配置参数提取
        config = config or {}
        actual_threshold = config.get('convergence_threshold',
                                      convergence_threshold)
        stall_threshold = config.get('stall_threshold', 5)

        # 初始化收敛检查器（必需）
        self.convergence_checker = ConvergenceChecker(
            convergence_threshold=actual_threshold,
            stall_threshold=stall_threshold,
            verbose=verbose)

        # 可选：进度监控
        self.progress_monitor: Optional[ProgressMonitor] = None
        if use_progress_monitor and ProgressMonitor is not None:
            self.progress_monitor = ProgressMonitor(
                total_iterations=max_iterations,
                verbose=verbose,
                check_convergence=True)

        # 可选：迭代历史
        self.iteration_history: Optional[IterationHistory] = None
        if use_iteration_history and IterationHistory is not None:
            self.iteration_history = IterationHistory(verbose=verbose)

    def start_iteration(self) -> None:
        """
        标记迭代开始
        
        应在每次迭代的开始调用
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
        grad_S_norm: float,
        grad_C_norm: float,
        alpha: float = 0.0,
        param_delta: float = 0.0,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录单次迭代的结果
        
        参数：
          chi2: 卡方统计量
          entropy: 熵值
          grad_S_norm: 熵梯度范数
          grad_C_norm: 约束梯度范数
          alpha: 步长（可选）
          param_delta: 参数变化幅度（可选）
          diagnostics: 诊断信息（可选）
        
        异常：
          ValueError: 任何数值为 NaN/Inf
        """
        # 验证输入
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

        # 更新进度监控
        if self.progress_monitor is not None:
            self.progress_monitor.on_iteration_complete(entropy)

        # 更新迭代历史
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
        检查是否应停止迭代
        
        参数：
          chi2: 当前χ²值（用于收敛检查）
        
        返回：
          (should_stop, reason)
          - should_stop: 是否应停止
          - reason: 停止原因
        """
        # 检查最大迭代数
        if self.iteration >= self.max_iterations:
            reason = f"max_iterations reached ({self.iteration}/{self.max_iterations})"
            self.convergence_reason = reason
            return True, reason

        # 检查收敛条件
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
    max_iters = config.max_iterations or config.num_iterations

    config_dict = {
        'convergence_threshold':
        config.convergence_threshold
        if hasattr(config, 'convergence_threshold') else 1e-6,
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
    print(f"\n=== 管理器总结 ===")
    print(f"迭代完成数: {summary['iterations_completed']}")
    print(f"是否收敛: {summary['converged']}")
    print(f"收敛原因: {summary['convergence_reason']}")
    print(f"总耗时: {summary['total_elapsed_seconds']:.3f}s")
    print(f"平均耗时/迭代: {summary['avg_time_per_iteration']:.3f}s")
