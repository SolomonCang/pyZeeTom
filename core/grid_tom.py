# grid_tom.py
# 等Δr分层的平面盘网格（每环宽度一致），允许每环像素数不一致

import numpy as np
from math import pi


class diskGrid:

    def __init__(self,
                 nr=60,
                 r_in=0.0,
                 r_out=5.0,
                 phi_min=0.0,
                 phi_max=2 * pi,
                 target_pixels_per_ring=None,
                 verbose=1):
        self.nr = int(nr)
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)

        if not (self.r_out > self.r_in):
            raise ValueError("r_out 必须大于 r_in")

        # 等 r 分箱（每环宽度一致）
        r_edges = np.linspace(self.r_in, self.r_out, self.nr + 1)
        self.r_edges = r_edges
        self.r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        self.dr = (r_edges[1:] - r_edges[:-1])  # 每环的Δr（常数，但保留数组形式）
        dr_const = self.dr[0]

        # 每环真实面积（极坐标环带面积）可由像素面积之和严格重构，无需单独存储

        # 决定每环的 nphi（允许不一致）
        if target_pixels_per_ring is None:
            # 近似等面积像素：nφ_i 使 r_i*Δr*Δφ ~ 常数 => nφ_i ∝ r_i
            nphi_ref = max(8, 2 * self.nr)
            r_mid = np.median(self.r_centers) if self.nr > 0 else 1.0
            nphi_list = np.clip(
                np.round(nphi_ref *
                         (self.r_centers / max(r_mid, 1e-12))).astype(int),
                4,
                None,
            )
            nphi_list = np.where(self.r_centers <= 0.0, 4, nphi_list)
        else:
            if isinstance(target_pixels_per_ring, int):
                nphi_list = np.full(self.nr,
                                    int(target_pixels_per_ring),
                                    dtype=int)
            else:
                nphi_arr = np.asarray(target_pixels_per_ring, dtype=int)
                if nphi_arr.shape[0] != self.nr:
                    raise ValueError("target_pixels_per_ring 长度必须等于 nr")
                nphi_list = np.maximum(1, nphi_arr)

        # 展开像素属性（全部用一维数组，避免零维）
        rs, phis, drs, dphis, areas, ring_ids, phi_ids = [], [], [], [], [], [], []
        phi_span = (self.phi_max - self.phi_min)
        for i in range(self.nr):
            nphi_i = int(nphi_list[i])
            nphi_i = max(1, nphi_i)
            dphi = phi_span / nphi_i
            phi_cent = self.phi_min + (np.arange(nphi_i) + 0.5) * dphi

            r_vec = np.full(nphi_i, self.r_centers[i], dtype=float)
            dr_vec = np.full(nphi_i, self.dr[i], dtype=float)
            dphi_vec = np.full(nphi_i, dphi, dtype=float)
            area_vec = r_vec * dr_vec * dphi_vec
            ring_vec = np.full(nphi_i, i, dtype=int)
            phi_idx_vec = np.arange(nphi_i, dtype=int)

            rs.append(r_vec)
            phis.append(phi_cent)
            drs.append(dr_vec)
            dphis.append(dphi_vec)
            areas.append(area_vec)
            ring_ids.append(ring_vec)
            phi_ids.append(phi_idx_vec)

        def _safe_concat(lst, dtype=float):
            return np.concatenate(lst) if len(lst) > 0 else np.array(
                [], dtype=dtype)

        self.r = _safe_concat(rs, float)
        self.phi = _safe_concat(phis, float)
        self.dr_cell = _safe_concat(drs, float)
        self.dphi_cell = _safe_concat(dphis, float)
        self.area = _safe_concat(areas, float)
        self.ring_id = _safe_concat(ring_ids, int)
        self.phi_id = _safe_concat(phi_ids, int)

        self.numPoints = self.r.shape[0]
        if verbose:
            print(
                f"[diskGrid] nr={self.nr}, r=[{self.r_in},{self.r_out}] (Δr={dr_const:.6g}), "
                f"N_pix={self.numPoints}, total_area≈{np.sum(self.area):.6f}")

    def cell_edges(self, idx):
        r_c = self.r[idx]
        dr = self.dr_cell[idx]
        dphi = self.dphi_cell[idx]
        phi_c = self.phi[idx]
        r_in = r_c - 0.5 * dr
        r_out = r_c + 0.5 * dr
        phi_in = phi_c - 0.5 * dphi
        phi_out = phi_c + 0.5 * dphi
        return r_in, r_out, phi_in, phi_out


def sector_polygon(r_in, r_out, phi_in, phi_out, samples_per_edge=12):
    n = max(2, int(samples_per_edge))
    phi_outer = np.linspace(phi_in, phi_out, n)
    x_outer = r_out * np.cos(phi_outer)
    y_outer = r_out * np.sin(phi_outer)
    phi_inner = np.linspace(phi_out, phi_in, n)
    x_inner = r_in * np.cos(phi_inner)
    y_inner = r_in * np.sin(phi_inner)
    x = np.concatenate([x_outer, x_inner])
    y = np.concatenate([y_outer, y_inner])
    return np.column_stack([x, y])


def visualize_grid(nr=60,
                   r_in=0.0,
                   r_out=5.0,
                   target_pixels_per_ring=None,
                   samples_per_edge=12,
                   show_indices=False,
                   cmap='viridis',
                   fig_dpi=140,
                   color_mode='area',
                   save_path=None,
                   show=True,
                   return_fig=False):
    """
    color_mode: 'ring'（按环号着色）或 'area'（按像素面积着色）
    """
    import importlib
    try:
        plt = importlib.import_module('matplotlib.pyplot')
        mpl = importlib.import_module('matplotlib')
        colors_mod = importlib.import_module('matplotlib.colors')
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required for visualize_grid(); install it via 'pip install matplotlib'"
        ) from e
    grid = diskGrid(nr=nr,
                    r_in=r_in,
                    r_out=r_out,
                    target_pixels_per_ring=target_pixels_per_ring,
                    verbose=1)

    colors = grid.ring_id if color_mode == 'ring' else grid.area

    fig, ax = plt.subplots(figsize=(7, 7), dpi=fig_dpi)
    ax.set_aspect('equal', 'box')

    if grid.numPoints > 0:
        cmin, cmax = np.min(colors), np.max(colors)
        if cmax == cmin:
            cmax = cmin + 1.0
    else:
        cmin, cmax = 0.0, 1.0

    # Get colormap robustly across Matplotlib versions
    if hasattr(mpl, 'colormaps') and hasattr(mpl.colormaps, 'get_cmap'):
        cmap_obj = mpl.colormaps.get_cmap(cmap)
    else:
        cmap_obj = mpl.cm.get_cmap(cmap)

    for i in range(grid.numPoints):
        r_in_i, r_out_i, phi_in_i, phi_out_i = grid.cell_edges(i)
        poly = sector_polygon(r_in_i,
                              r_out_i,
                              phi_in_i,
                              phi_out_i,
                              samples_per_edge=samples_per_edge)
        color_val = (colors[i] - cmin) / (cmax - cmin)
        ax.fill(poly[:, 0],
                poly[:, 1],
                color=cmap_obj(color_val),
                edgecolor='k',
                linewidth=0.2)

    lim = r_out * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Disk grid visualization: nr={nr}, pixels={grid.numPoints}')

    Normalize = getattr(colors_mod, 'Normalize')
    norm = Normalize(vmin=cmin, vmax=cmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Ring index' if color_mode == 'ring' else 'Cell area')

    if show_indices:
        for i in range(grid.numPoints):
            ax.text(grid.r[i] * np.cos(grid.phi[i]),
                    grid.r[i] * np.sin(grid.phi[i]),
                    f"{i}\n({grid.ring_id[i]},{grid.phi_id[i]})",
                    fontsize=5,
                    ha='center',
                    va='center',
                    color='white')

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=fig_dpi, bbox_inches='tight')
    if show:
        plt.show()
    if return_fig:
        return fig, ax, grid
