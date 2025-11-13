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

    def rotate_to_phase(self, phase, pOmega=0.0, r0=1.0, period=1.0):
        """根据相位和差速转动参数，更新像素的方位角
        
        适用于时间演化场景：每个观测时刻，盘面结构随差速转动而变化。
        
        Parameters
        ----------
        phase : float
            观测相位，phase = (JD - JD0) / period，表示从参考时刻经过的转动周期数
        pOmega : float, optional
            差速转动幂律指数，Ω(r) = Ω_ref × (r/r0)^pOmega
            - pOmega = 0.0  : 刚体转动（默认）
            - pOmega = -0.5 : 开普勒型（类太阳盘）
            - pOmega = -1.0 : 恒定角动量
        r0 : float, optional
            参考半径，默认 1.0（通常取星球半径）
        period : float, optional
            参考半径处的转动周期（天），默认 1.0
            
        Returns
        -------
        phi_new : ndarray
            更新后的方位角（弧度），shape = (numPoints,)
            
        Notes
        -----
        - 刚体转动（pOmega=0）：所有环以相同角速度 Ω_ref 转动，
          Δφ = 2π × phase（与半径无关）
        - 差速转动（pOmega≠0）：每环角速度 Ω(r) = Ω_ref × (r/r0)^pOmega，
          Δφ(r) = 2π × phase × (r/r0)^pOmega
        - 本方法不修改 self.phi，而是返回新的方位角数组，便于每个观测独立计算
        """
        phase = float(phase)
        pOmega = float(pOmega)
        r0 = float(max(r0, 1e-30))
        period = float(period)

        # 计算每个像素的角位移
        if abs(pOmega) < 1e-12:
            # 刚体转动：所有环转动相同角度
            delta_phi = 2.0 * np.pi * phase
        else:
            # 差速转动：每环转动角度与半径相关
            # Ω(r) = Ω_ref × (r/r0)^pOmega
            # Δφ(r) = Ω(r) × Δt = Ω_ref × (r/r0)^pOmega × (phase × period)
            # 而 Ω_ref × period = 2π（参考半径处一个周期转动 2π）
            # 因此 Δφ(r) = 2π × phase × (r/r0)^pOmega
            r_ratio = self.r / r0
            delta_phi = 2.0 * np.pi * phase * np.power(r_ratio, pOmega)

        # 更新方位角（周期性边界条件）
        phi_new = (self.phi + delta_phi) % (2.0 * np.pi)
        return phi_new

    def compute_stellar_occultation_mask(self,
                                         phi_obs,
                                         inclination_deg,
                                         stellar_radius=1.0,
                                         verbose=0):
        """计算恒星遮挡mask（无限薄赤道盘，光学薄）
        
        Parameters
        ----------
        phi_obs : float
            观察者方向（弧度），观察者从这个方位角看向恒星中心
            phi_obs=0 表示从"上方"（+x方向）观测
        inclination_deg : float
            倾角（度），i=0为face-on（从极点看），i=90为edge-on（从赤道面看）
        stellar_radius : float, optional
            恒星半径（与盘面半径相同单位），默认1.0
        verbose : int, optional
            是否输出调试信息
            
        Returns
        -------
        mask : ndarray (bool)
            遮挡mask，shape = (numPoints,)
            True = 被恒星遮挡，False = 可见
            
        Notes
        -----
        物理模型：
        - 盘为无限薄，位于赤道面（z=0）
        - 恒星为半径 R* 的球体，中心在原点
        - 观测者方向固定，从 phi_obs 方向看过来
        - 遮挡mask只依赖于几何参数（phi_obs, inclination, R*），不随时间变化
        
        坐标系约定：
        - 盘面坐标系：(r, φ) 柱坐标，z=0 平面
        - 观察者视线：从 (x, y) = (cos(phi_obs), sin(phi_obs)) 方向看向原点
        - 倾角：视线与z轴的夹角
        
        遮挡判据：
        - 将像素位置投影到垂直于视线的平面上
        - 若投影距离 < R*，且像素在恒星后方（沿视线方向），则被遮挡
        """
        if self.numPoints == 0:
            return np.array([], dtype=bool)

        inclination_rad = np.deg2rad(float(inclination_deg))
        phi_obs = float(phi_obs)
        R_star = float(stellar_radius)

        # 像素笛卡尔坐标（盘面坐标系，z=0）
        x_disk = self.r * np.cos(self.phi)
        y_disk = self.r * np.sin(self.phi)
        z_disk = np.zeros_like(self.r)  # 无限薄盘

        # 观察者视线方向单位矢量（从远处指向恒星中心）
        # 视线在 x-y 平面的投影沿 phi_obs 方向，与 z 轴夹角为 inclination
        sin_i = np.sin(inclination_rad)
        cos_i = np.cos(inclination_rad)

        # 视线方向: n_obs = (sin_i * cos(phi_obs), sin_i * sin(phi_obs), cos_i)
        # 观察者坐标系：视线沿 -z' 轴，需要坐标变换

        # 方法：计算像素沿视线方向的距离（正=远离观察者，负=靠近观察者）
        # 距离 = r_pixel · n_obs
        dist_along_view = (x_disk * sin_i * np.cos(phi_obs) +
                           y_disk * sin_i * np.sin(phi_obs) + z_disk * cos_i)

        # 像素到视线的垂直距离（投影距离）
        # r_perp^2 = |r_pixel|^2 - (r_pixel · n_obs)^2
        r_pixel_sq = x_disk**2 + y_disk**2 + z_disk**2
        r_perp = np.sqrt(np.maximum(0.0, r_pixel_sq - dist_along_view**2))

        # 遮挡判据：
        # 1. 垂直距离 < R*（在恒星投影圆盘内）
        # 2. dist_along_view < 0（在恒星后方，相对观察者）
        mask = (r_perp < R_star) & (dist_along_view < 0)

        if verbose:
            n_occult = np.sum(mask)
            print(
                f"[Occultation] phi_obs={np.degrees(phi_obs):.1f}°, i={inclination_deg:.1f}°, "
                f"{n_occult}/{self.numPoints} pixels occulted ({100.0*n_occult/max(self.numPoints,1):.1f}%)"
            )

        return mask


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
