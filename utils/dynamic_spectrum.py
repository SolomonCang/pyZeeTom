"""
动态光谱工具类
--------------
提供不规则时间采样的动态光谱可视化功能
基于 tinyTools/Dynamic_spec.py 的 IrregularDynamicSpectrum 类
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d


class IrregularDynamicSpectrum:
    """
    不规则时间采样的动态光谱类
    
    用于处理和可视化时间不均匀分布的光谱序列
    """

    def __init__(self, times: np.ndarray, xs: list, intensities: list):
        """
        Parameters
        ----------
        times : np.ndarray, shape (N,)
            不均匀的观测时间点，必须升序
        xs : list of np.ndarray
            长度 = N，每项是该时刻的横坐标 array (Mi,)
        intensities : list of np.ndarray
            长度 = N，每项是对应强度 array (Mi,)
        """
        assert len(times) == len(xs) == len(intensities)
        self.times = np.array(times, dtype=float)
        self.xs = xs
        self.Is = intensities
        self.N = len(times)

    def plot(self,
             xlim=None,
             ylim=None,
             cmap='viridis',
             vmin=None,
             vmax=None,
             log_scale=False,
             gap_thresh=None,
             gap_color='white',
             title='Dynamic Spectrum',
             time_widths=None,
             xlabel='Velocity (km/s)',
             ylabel='Phase'):
        """
        绘制不规则动态谱。

        Parameters
        ----------
        xlim, ylim : tuple or None
            坐标轴范围
        cmap : str
            色标名称
        vmin, vmax : float or None
            色标范围
        log_scale : bool
            是否使用对数色标
        gap_thresh : float or None
            相邻观测若 dt > gap_thresh，则留白
        gap_color : str
            留白区域颜色
        title : str
            图题
        time_widths : None | float | array-like
            每条谱带的时间宽度；None 时自动按相邻时间差计算
        xlabel, ylabel : str
            坐标轴标签
            
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor(gap_color)

        times = self.times
        N = self.N

        # 1) 计算每条谱带的高度 dt
        if time_widths is None:
            # 自动按相邻时间差
            dt = np.zeros(N)
            if N == 1:
                dt[0] = 1.0
            else:
                dt[0] = times[1] - times[0]
                dt[-1] = times[-1] - times[-2]
                if N > 2:
                    dt[1:-1] = (times[2:] - times[:-2]) / 2
        else:
            # 用户指定
            if np.isscalar(time_widths):
                dt = np.full(N, float(time_widths))
            else:
                dt = np.array(time_widths, dtype=float)
                if dt.shape[0] != N:
                    raise ValueError(f"time_widths 长度应为 {N}，但得到 {dt.shape[0]}")

        # 2) colormap / norm
        norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
        last_mesh = None

        # 3) 逐条绘制
        for i, t in enumerate(times):
            # 跳过大 gap
            if i > 0 and gap_thresh is not None and (
                    times[i] - times[i - 1]) > gap_thresh:
                continue

            x = self.xs[i]
            I = self.Is[i]
            M = x.size

            # 3.1) x 边界
            if M > 1:
                dx = np.diff(x)
                x_edges = np.empty(M + 1, dtype=float)
                x_edges[1:-1] = (x[:-1] + x[1:]) / 2
                x_edges[0] = x[0] - dx[0] / 2
                x_edges[-1] = x[-1] + dx[-1] / 2
            else:
                w = 0.5
                x_edges = np.array([x[0] - w / 2, x[0] + w / 2])

            # 3.2) y 边界
            half = dt[i] / 2
            y0 = t - half
            y1 = t + half

            X = np.vstack([x_edges, x_edges])
            Y = np.array([[y0] * (M + 1), [y1] * (M + 1)])
            Z = I[np.newaxis, :]

            mesh = ax.pcolormesh(X,
                                 Y,
                                 Z,
                                 cmap=cmap,
                                 norm=norm,
                                 vmin=vmin,
                                 vmax=vmax,
                                 shading='flat')
            last_mesh = mesh

        # 4) 坐标与色标
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if last_mesh is not None:
            cb = fig.colorbar(last_mesh, ax=ax)
            cb.set_label('Intensity (I/Ic)')
        plt.tight_layout()
        return fig, ax

    def remove_baseline(self,
                        method='median',
                        x_common=None,
                        n_common=500,
                        smooth_sigma=None):
        """
        全谱统一基线去除：
        1) （可选）对每条谱做 1D 高斯平滑
        2) 插值到共同横坐标 x_common
        3) 计算全体谱在 x_common 上的基线
        4) 插值回各自原始 x 并减去

        Parameters
        ----------
        method : str
            'median' 或 'mean'
        x_common : array-like or None
            统一网格；None 时自动从所有谱的 x 范围按 n_common 等分生成
        n_common : int
            若 x_common=None，则生成 n_common 点
        smooth_sigma : float or None
            若不为 None，则对每条谱先做 gaussian_filter1d（沿 x 方向）
        """
        xs_source = getattr(self, "xs_proc", self.xs)
        Is_source = getattr(self, "Is_proc", self.Is)

        if x_common is None:
            x_min = min(np.nanmin(x) for x in xs_source)
            x_max = max(np.nanmax(x) for x in xs_source)
            x_common = np.linspace(x_min, x_max, n_common)

        I_common_list = []
        for x, I in zip(xs_source, Is_source):
            I_arr = np.asarray(I, dtype=float)
            if smooth_sigma is not None:
                med = np.nanmedian(I_arr)
                I_filled = np.where(np.isnan(I_arr), med, I_arr)
                I_arr = gaussian_filter1d(I_filled, smooth_sigma)

            Ic = np.interp(x_common, x, I_arr, left=np.nan, right=np.nan)
            I_common_list.append(Ic)

        M = np.vstack(I_common_list)  # shape (N_spectra, n_common)
        if method == 'median':
            baseline_common = np.nanmedian(M, axis=0)
        elif method == 'mean':
            baseline_common = np.nanmean(M, axis=0)
        else:
            raise ValueError("method must be 'median' or 'mean'")

        # 更新强度数据
        for idx, (x, I) in enumerate(zip(xs_source, Is_source)):
            b_i = np.interp(x, x_common, baseline_common)
            self.Is[idx] = np.asarray(self.Is[idx], dtype=float) - b_i

        self.x_baseline = x_common
        self.baseline_common = baseline_common
