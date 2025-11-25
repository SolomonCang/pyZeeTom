# grid_tom.py
# Equal Δr stratified planar disk grid (constant width per ring), allowing variable number of pixels per ring

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
            raise ValueError("r_out must be greater than r_in")

        # Equal r binning (constant width per ring)
        r_edges = np.linspace(self.r_in, self.r_out, self.nr + 1)
        self.r_edges = r_edges
        self.r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        self.dr = (r_edges[1:] - r_edges[:-1]
                   )  # Δr for each ring (constant, but kept as array)
        dr_const = self.dr[0]

        # The true area of each ring (polar annulus area) can be strictly reconstructed from the sum of pixel areas, no need to store separately

        # Determine nphi for each ring (can be variable)
        if target_pixels_per_ring is None:
            # Approximate equal area pixels: nφ_i makes r_i*Δr*Δφ ~ constant => nφ_i ∝ r_i
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
                    raise ValueError(
                        "target_pixels_per_ring length must equal nr")
                nphi_list = np.maximum(1, nphi_arr)

        # Expand pixel attributes (all use 1D arrays, avoid zero-dimensional)
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
        """Update pixel azimuth based on phase and differential rotation parameters
        
        Suitable for time evolution scenarios: disk structure changes due to differential rotation at each observation time.
        
        Parameters
        ----------
        phase : float
            Observation phase, phase = (JD - JD0) / period, representing the number of rotation periods since the reference time
        pOmega : float, optional
            Differential rotation power law index, Ω(r) = Ω_ref × (r/r0)^pOmega
            - pOmega = 0.0  : Rigid body rotation (default)
            - pOmega = -0.5 : Keplerian (solar-like disk)
            - pOmega = -1.0 : Constant angular momentum
        r0 : float, optional
            Reference radius, default 1.0 (usually planet radius)
        period : float, optional
            Rotation period at reference radius (days), default 1.0
            
        Returns
        -------
        phi_new : ndarray
            Updated azimuth (radians), shape = (numPoints,)
            
        Notes
        -----
        - Rigid body rotation (pOmega=0): All rings rotate with same angular velocity Ω_ref,
          Δφ = 2π × phase (independent of radius)
        - Differential rotation (pOmega≠0): Angular velocity per ring Ω(r) = Ω_ref × (r/r0)^pOmega,
          Δφ(r) = 2π × phase × (r/r0)^pOmega
        - This method does not modify self.phi, but returns a new azimuth array for independent calculation per observation
        """
        phase = float(phase)
        pOmega = float(pOmega)
        r0 = float(max(r0, 1e-30))
        period = float(period)

        # Calculate angular displacement for each pixel
        if abs(pOmega) < 1e-12:
            # Rigid body rotation: all rings rotate by same angle
            delta_phi = 2.0 * np.pi * phase
        else:
            # Differential rotation: rotation angle depends on radius
            # Ω(r) = Ω_ref × (r/r0)^pOmega
            # Δφ(r) = Ω(r) × Δt = Ω_ref × (r/r0)^pOmega × (phase × period)
            # And Ω_ref × period = 2π (one period rotation at reference radius is 2π)
            # Therefore Δφ(r) = 2π × phase × (r/r0)^pOmega
            r_ratio = self.r / r0
            delta_phi = 2.0 * np.pi * phase * np.power(r_ratio, pOmega)

        # Update azimuth (periodic boundary condition)
        phi_new = (self.phi + delta_phi) % (2.0 * np.pi)
        return phi_new

    def compute_stellar_occultation_mask(self,
                                         phi_obs,
                                         inclination_deg,
                                         stellar_radius=1.0,
                                         verbose=0):
        """Calculate stellar occultation mask (infinitely thin equatorial disk, optically thin)
        
        Parameters
        ----------
        phi_obs : float
            Observer direction (radians), observer looks towards star center from this azimuth
            phi_obs=0 means observing from "above" (+x direction)
        inclination_deg : float
            Inclination (degrees), i=0 is face-on (pole-on), i=90 is edge-on (equator-on)
        stellar_radius : float, optional
            Stellar radius (same unit as disk radius), default 1.0
        verbose : int, optional
            Whether to print debug info
            
        Returns
        -------
        mask : ndarray (bool)
            Occultation mask, shape = (numPoints,)
            True = Occulted by star, False = Visible
            
        Notes
        -----
        Physical model:
        - Disk is infinitely thin, located at equatorial plane (z=0)
        - Star is a sphere of radius R* centered at origin
        - Observer direction is fixed, looking from phi_obs direction
        - Occultation mask depends only on geometric parameters (phi_obs, inclination, R*), not time-varying
        
        Coordinate system convention:
        - Disk coordinate system: (r, φ) cylindrical coordinates, z=0 plane
        - Observer line of sight: Looking towards origin from (x, y) = (cos(phi_obs), sin(phi_obs)) direction
        - Inclination: Angle between line of sight and z-axis
        
        Occultation criterion:
        - Project pixel position onto plane perpendicular to line of sight
        - If projected distance < R*, and pixel is behind star (along line of sight), then it is occulted
        """
        if self.numPoints == 0:
            return np.array([], dtype=bool)

        inclination_rad = np.deg2rad(float(inclination_deg))
        phi_obs = float(phi_obs)
        R_star = float(stellar_radius)

        # Pixel Cartesian coordinates (disk coordinate system, z=0)
        x_disk = self.r * np.cos(self.phi)
        y_disk = self.r * np.sin(self.phi)
        z_disk = np.zeros_like(self.r)  # Infinitely thin disk

        # Observer line of sight unit vector (pointing from far away to star center)
        # Line of sight projection on x-y plane is along phi_obs direction, angle with z axis is inclination
        sin_i = np.sin(inclination_rad)
        cos_i = np.cos(inclination_rad)

        # Line of sight: n_obs = (sin_i * cos(phi_obs), sin_i * sin(phi_obs), cos_i)
        # Observer coordinate system: Line of sight along -z' axis, coordinate transformation needed

        # Method: Calculate pixel distance along line of sight (positive = away from observer, negative = towards observer)
        # Distance = r_pixel · n_obs
        dist_along_view = (x_disk * sin_i * np.cos(phi_obs) +
                           y_disk * sin_i * np.sin(phi_obs) + z_disk * cos_i)

        # Perpendicular distance from pixel to line of sight (projected distance)
        # r_perp^2 = |r_pixel|^2 - (r_pixel · n_obs)^2
        r_pixel_sq = x_disk**2 + y_disk**2 + z_disk**2
        r_perp = np.sqrt(np.maximum(0.0, r_pixel_sq - dist_along_view**2))

        # Occultation criterion:
        # 1. Perpendicular distance < R* (within star projected disk)
        # 2. dist_along_view < 0 (behind star, relative to observer)
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
    color_mode: 'ring' (color by ring index) or 'area' (color by pixel area)
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
