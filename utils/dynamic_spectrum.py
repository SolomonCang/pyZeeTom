"""
Dynamic Spectrum Tool Class
---------------------------
Provides visualization functionality for dynamic spectra with irregular time sampling
Based on IrregularDynamicSpectrum class from tinyTools/Dynamic_spec.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d


class IrregularDynamicSpectrum:
    """
    Irregular Time Sampling Dynamic Spectrum Class
    
    Used for processing and visualizing spectrum sequences with uneven time distribution
    """

    def __init__(self, times: np.ndarray, xs: list, intensities: list):
        """
        Parameters
        ----------
        times : np.ndarray, shape (N,)
            Uneven observation time points, must be ascending
        xs : list of np.ndarray
            Length = N, each item is the x-coordinate array (Mi,) for that moment
        intensities : list of np.ndarray
            Length = N, each item is the corresponding intensity array (Mi,)
        """
        assert len(times) == len(xs) == len(intensities)
        self.times = np.array(times, dtype=float)
        self.xs = xs
        self.Is = intensities
        self.N = len(times)

    def plot(self,
             ax=None,
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
             ylabel='Phase',
             colorbar=True,
             cbar_label='Intensity (I/Ic)'):
        """
        Plot irregular dynamic spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, create new figure and axes.
        xlim, ylim : tuple or None
            Axis limits
        cmap : str
            Colormap name
        vmin, vmax : float or None
            Colormap range
        log_scale : bool
            Whether to use logarithmic colormap
        gap_thresh : float or None
            If adjacent observations have dt > gap_thresh, leave blank
        gap_color : str
            Color for blank areas
        title : str
            Plot title
        time_widths : None | float | array-like
            Time width for each spectrum strip; if None, calculated automatically from adjacent time differences
        xlabel, ylabel : str
            Axis labels
        colorbar : bool
            Whether to show colorbar
        cbar_label : str
            Label for colorbar
            
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        ax.set_facecolor(gap_color)

        times = self.times
        N = self.N

        # 1) Calculate height dt for each spectrum strip
        if time_widths is None:
            # Automatically based on adjacent time differences
            dt = np.zeros(N)
            if N == 1:
                dt[0] = 1.0
            else:
                dt[0] = times[1] - times[0]
                dt[-1] = times[-1] - times[-2]
                if N > 2:
                    dt[1:-1] = (times[2:] - times[:-2]) / 2
        else:
            # User specified
            if np.isscalar(time_widths):
                dt = np.full(N, float(time_widths))
            else:
                dt = np.array(time_widths, dtype=float)
                if dt.shape[0] != N:
                    raise ValueError(
                        f"time_widths length should be {N}, but got {dt.shape[0]}"
                    )

        # 2) colormap / norm
        norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
        last_mesh = None

        # 3) Plot strip by strip
        for i, t in enumerate(times):
            # Skip large gaps
            if i > 0 and gap_thresh is not None and (
                    times[i] - times[i - 1]) > gap_thresh:
                continue

            x = self.xs[i]
            I = self.Is[i]
            M = x.size

            # 3.1) x boundaries
            if M > 1:
                dx = np.diff(x)
                x_edges = np.empty(M + 1, dtype=float)
                x_edges[1:-1] = (x[:-1] + x[1:]) / 2
                x_edges[0] = x[0] - dx[0] / 2
                x_edges[-1] = x[-1] + dx[-1] / 2
            else:
                w = 0.5
                x_edges = np.array([x[0] - w / 2, x[0] + w / 2])

            # 3.2) y boundaries
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

        # 4) Axes and Colorbar
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if last_mesh is not None and colorbar:
            cb = fig.colorbar(last_mesh, ax=ax)
            cb.set_label(cbar_label)
        plt.tight_layout()
        return fig, ax

    def remove_baseline(self,
                        method='median',
                        x_common=None,
                        n_common=500,
                        smooth_sigma=None):
        """
        Global baseline removal:
        1) (Optional) 1D Gaussian smoothing for each spectrum
        2) Interpolate to common x-coordinate x_common
        3) Calculate baseline of all spectra on x_common
        4) Interpolate back to original x and subtract

        Parameters
        ----------
        method : str
            'median' or 'mean'
        x_common : array-like or None
            Common grid; if None, generated by dividing x range of all spectra into n_common parts
        n_common : int
            If x_common=None, generate n_common points
        smooth_sigma : float or None
            If not None, apply gaussian_filter1d (along x direction) to each spectrum first
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

        # Update intensity data
        for idx, (x, I) in enumerate(zip(xs_source, Is_source)):
            b_i = np.interp(x, x_common, baseline_common)
            self.Is[idx] = np.asarray(self.Is[idx], dtype=float) - b_i

        self.x_baseline = x_common
        self.baseline_common = baseline_common
