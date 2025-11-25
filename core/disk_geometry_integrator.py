"""Integrated disk geometry and velocity space integrator for tomography.

This module combines SimpleDiskGeometry and VelspaceDiskIntegrator functionality,
providing unified disk model, magnetic field configuration, and spectral synthesis.

The 'amp' parameter represents the spectral line amplitude (response weight),
which is used in local profile calculations to determine emission/absorption strength.

Classes
-------
SimpleDiskGeometry : Disk geometry container with magnetic field support
VelspaceDiskIntegrator : Velocity space integrator for disk Stokes spectra synthesis

Functions
---------
create_disk_geometry_from_params : Create geometry from parameter objects
convolve_gaussian_1d : Gaussian convolution utility
disk_velocity_rigid_inner : Rigid body inner rotation velocity profile
"""

import numpy as np
from typing import Optional
from scipy.ndimage import convolve1d
from core.grid_tom import diskGrid

__all__ = [
    'SimpleDiskGeometry',
    'VelspaceDiskIntegrator',
    'create_disk_geometry_from_params',
    'convolve_gaussian_1d',
    'disk_velocity_rigid_inner',
]

C_KMS = 2.99792458e5  # km/s

# ====================================================================
# Utility Functions
# ====================================================================


def convolve_gaussian_1d(y, dv, fwhm):
    """Convolve spectrum with Gaussian kernel.
    
    Parameters
    ----------
    y : np.ndarray
        Input spectrum (1D or 2D). If 2D, convolve along axis 0.
    dv : float
        Velocity grid spacing (km/s)
    fwhm : float
        Gaussian FWHM (km/s)
    
    Returns
    -------
    np.ndarray
        Convolved spectrum
    """
    if fwhm <= 0.0:
        return y

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_pix = sigma / np.maximum(dv, 1e-30)

    # Kernel radius (4 sigma is sufficient)
    radius = int(np.ceil(4.0 * sigma_pix))
    if radius < 1:
        return y

    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_pix)**2)
    kernel /= np.sum(kernel)

    # Use scipy.ndimage.convolve1d with nearest padding (replicating edge values)
    # This matches the previous behavior of padding with y[0] and y[-1]
    return convolve1d(y, kernel, axis=0, mode='nearest')


def disk_velocity_rigid_inner(r, v0_r0, p, r0):
    """Compute disk velocity with rigid body rotation inside r0.
    
    Parameters
    ----------
    r : np.ndarray
        Radii array
    v0_r0 : float
        Velocity at reference radius (km/s)
    p : float
        Power index for outer region
    r0 : float
        Reference radius
        
    Returns
    -------
    np.ndarray
        Velocity array
    """
    r = np.asarray(r, dtype=float)
    rr0 = float(max(r0, 1e-30))
    Omega0 = float(v0_r0) / rr0
    x = r / rr0

    # Omega profile:
    # r < r0: Omega = Omega0 (Rigid body)
    # r >= r0: Omega = Omega0 * (r/r0)^p

    Omega = np.zeros_like(r)
    mask_inner = x < 1.0
    mask_outer = ~mask_inner

    Omega[mask_inner] = Omega0
    Omega[mask_outer] = Omega0 * np.power(x[mask_outer], p)

    v = r * Omega
    return v


# ====================================================================
# SimpleDiskGeometry Class
# ====================================================================


class SimpleDiskGeometry:
    """Minimal disk geometry container for VelspaceDiskIntegrator.
    
    Provides: grid, area_proj, inclination_rad, phi0, pOmega, r0, period,
         enable_stellar_occultation, stellar_radius, B_los, B_perp, chi, amp.
    
    The 'amp' parameter represents the spectral line amplitude (response weight),
    which modulates emission/absorption in the line profile calculation.
    
    Parameters
    ----------
    grid : diskGrid
        Disk surface grid object with pixel positions and areas
    inclination_deg : float, default=60.0
        Disk inclination angle (degrees)
    phi0 : float, default=0.0
        Reference azimuthal position (radians)
    pOmega : float, default=0.0
        Differential rotation index (d(ln Omega)/d(ln r))
        - pOmega = 0: Solid body rotation
        - pOmega = -0.5: v ∝ r^0.5 (Keplerian-like)
        - pOmega = -1.0: Angular momentum conserved
    r0 : float, default=1.0
        Reference radius (R_sun)
    period : float, default=1.0
        Rotation period (days)
    enable_stellar_occultation : int, default=0
        Enable stellar occultation effect (0=disabled, 1=enabled)
    stellar_radius : float, default=1.0
        Stellar radius (R_sun)
    B_los : np.ndarray, optional
        Line-of-sight magnetic field (Gauss), shape (grid.numPoints,)
        Defaults to zeros if None
    B_perp : np.ndarray, optional
        Perpendicular magnetic field strength (Gauss)
        Defaults to zeros if None
    chi : np.ndarray, optional
        Magnetic field azimuthal angle (rad)
        Defaults to zeros if None
    amp : np.ndarray, optional
        Spectral line amplitude distribution (response weight, >0).
        Represents the emission/absorption strength directly from geometry.
        Used as multiplicative factor in local profile calculation.
        Defaults to ones if None. Values represent absolute emission/absorption strength.
    
    Attributes
    ----------
    grid : diskGrid
        Disk grid
    area_proj : np.ndarray
        Projected area array
    inclination_rad : float
        Inclination angle (radians)
    phi0 : float
        Reference azimuthal position
    pOmega : float
        Differential rotation index
    r0 : float
        Reference radius
    period : float
        Rotation period
    enable_stellar_occultation : bool
        Stellar occultation flag
    stellar_radius : float
        Stellar radius
    B_los : np.ndarray
        Line-of-sight magnetic field
    B_perp : np.ndarray
        Perpendicular magnetic field strength
    chi : np.ndarray
        Magnetic field azimuthal angle
    amp : np.ndarray
        Spectral line amplitude (response weight)
    """

    def __init__(self,
                 grid: diskGrid,
                 inclination_deg: float = 60.0,
                 phi0: float = 0.0,
                 pOmega: float = 0.0,
                 r0: float = 1.0,
                 period: float = 1.0,
                 enable_stellar_occultation: int = 0,
                 stellar_radius: float = 1.0,
                 amp: Optional[np.ndarray] = None,
                 B_los: Optional[np.ndarray] = None,
                 B_perp: Optional[np.ndarray] = None,
                 chi: Optional[np.ndarray] = None):
        """Initialize SimpleDiskGeometry."""
        self.grid = grid
        self.area_proj = np.asarray(grid.area)
        self.inclination_rad = np.deg2rad(float(inclination_deg))
        self.phi0 = float(phi0)
        self.pOmega = float(pOmega)
        self.r0 = float(r0)
        self.period = float(period)
        self.enable_stellar_occultation = bool(enable_stellar_occultation)
        self.stellar_radius = float(stellar_radius)

        # Validate magnetic field and amplitude parameter dimensions
        npix = grid.numPoints
        if B_los is not None:
            B_los = np.asarray(B_los, dtype=float)
            if len(B_los) != npix:
                raise ValueError(
                    f"B_los length ({len(B_los)}) must match grid.numPoints ({npix})"
                )
        if B_perp is not None:
            B_perp = np.asarray(B_perp, dtype=float)
            if len(B_perp) != npix:
                raise ValueError(
                    f"B_perp length ({len(B_perp)}) must match grid.numPoints ({npix})"
                )
        if chi is not None:
            chi = np.asarray(chi, dtype=float)
            if len(chi) != npix:
                raise ValueError(
                    f"chi length ({len(chi)}) must match grid.numPoints ({npix})"
                )
        if amp is not None:
            amp = np.asarray(amp, dtype=float)
            if len(amp) != npix:
                raise ValueError(
                    f"amp length ({len(amp)}) must match grid.numPoints ({npix})"
                )

        # Initialize magnetic field and amplitude parameters
        self.B_los = B_los if B_los is not None else np.zeros(npix)
        self.B_perp = B_perp if B_perp is not None else np.zeros(npix)
        self.chi = chi if chi is not None else np.zeros(npix)
        self.amp = amp if amp is not None else np.ones(npix)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SimpleDiskGeometry(inclination={np.rad2deg(self.inclination_rad):.1f}°, "
            f"pOmega={self.pOmega:.2f}, period={self.period:.3f} d, "
            f"npix={self.grid.numPoints})")

    def get_magnetic_field_summary(self) -> dict:
        """Get summary of magnetic field and amplitude parameters.
        
        Returns
        -------
        dict
            Dictionary with magnetic field and amplitude statistics
        """
        return {
            'B_los_min': float(np.min(self.B_los)),
            'B_los_max': float(np.max(self.B_los)),
            'B_los_mean': float(np.mean(self.B_los)),
            'B_perp_min': float(np.min(self.B_perp)),
            'B_perp_max': float(np.max(self.B_perp)),
            'B_perp_mean': float(np.mean(self.B_perp)),
            'chi_min': float(np.min(self.chi)),
            'chi_max': float(np.max(self.chi)),
            'amp_min': float(np.min(self.amp)),
            'amp_max': float(np.max(self.amp)),
            'amp_mean': float(np.mean(self.amp)),
        }


# ====================================================================
# VelspaceDiskIntegrator Class
# ====================================================================


class VelspaceDiskIntegrator:
    """Velocity space integrator for disk Stokes spectra.
    
    Integrates disk model in velocity space, combining:
    - Pixel velocity field (outer: power-law Omega, inner: adaptive slowdown)
    - External line model (computes local I/V/Q/U profiles)
    - Instrumental convolution and continuum normalization
    
    The 'amp' parameter from geometry represents the spectral amplitude weight,
    which is passed directly to the line_model for local profile computation.
    
    Parameters
    ----------
    geom : SimpleDiskGeometry
        Disk geometry and magnetic field container. The 'amp' parameter from
        geometry represents the spectral amplitude weight.
    wl0_nm : float
        Rest wavelength (nm)
    v_grid : np.ndarray
        Velocity grid (km/s)
    line_model : BaseLineModel
        External line model with compute_local_profile() method
    line_area : float, default=1.0
        Line area factor
    inst_fwhm_kms : float, default=0.0
        Instrumental FWHM (km/s)
    normalize_continuum : bool, default=True
        Normalize to continuum level
    use_geom_vlos_if_available : bool, default=True
        Use geometry's v_los if available, else compute from disk velocity
    disk_v0_kms : float, default=200.0
        Velocity at reference radius (km/s)
    disk_power_index : float, default=-0.05
        Power index for outer region
    disk_r0 : float, default=1.0
        Reference radius
    los_proj_func : callable, optional
        Custom line-of-sight projection function
    obs_phase : float, optional
        Observation phase
    time_phase : float, optional
        Time evolution phase parameter
    """

    def __init__(
            self,
            geom,
            wl0_nm,
            v_grid,
            line_model,  # Required parameter
            line_area=1.0,
            inst_fwhm_kms=0.0,
            normalize_continuum=False,
            # Velocity field parameters (outer region power-law)
            use_geom_vlos_if_available=True,
            disk_v0_kms=50.0,
            disk_power_index=-0.05,
            disk_r0=0.8,
            # Line-of-sight projection
            los_proj_func=None,
            obs_phase=None,
            # Time evolution support
            time_phase=None):
        self.geom = geom
        self.grid = geom.grid
        self.wl0 = float(wl0_nm)
        self.v = np.asarray(v_grid)

        self.dv = np.mean(np.diff(self.v)) if self.v.size > 1 else 1.0
        self.inst_fwhm = float(inst_fwhm_kms)
        self.normalize_continuum = bool(normalize_continuum)
        self.line_area = float(line_area)

        # Store phase information for time evolution support
        self.time_phase = time_phase if time_phase is not None else obs_phase

        # Compute evolved azimuthal angles (for time evolution support)
        phi_evolved = self.grid.phi
        if self.time_phase is not None and hasattr(self.grid,
                                                   'rotate_to_phase'):
            pOmega = getattr(geom, 'pOmega', disk_power_index)
            r0_rot = getattr(geom, 'r0', disk_r0)
            period = getattr(geom, 'period', 1.0)
            phi_evolved = self.grid.rotate_to_phase(self.time_phase,
                                                    pOmega=pOmega,
                                                    r0=r0_rot,
                                                    period=period)

        # Projected geometric area weight
        W = np.asarray(self.geom.area_proj)
        assert W.shape[0] == self.grid.numPoints

        # Unprojected line velocity v_phi or use geometry's v_los
        if use_geom_vlos_if_available and hasattr(self.geom, "v_los"):
            v_los = np.asarray(self.geom.v_los)
            assert v_los.shape[0] == self.grid.numPoints
            v_phi = None
        else:
            v0_r0 = float(disk_v0_kms)
            p = float(disk_power_index)
            r0 = float(disk_r0)
            # Store for export
            self._disk_v0_kms = v0_r0
            self._disk_power_index = p
            self._disk_r0 = r0

            # Use rigid inner rotation model (r < r0: rigid, r >= r0: power law)
            v_phi = disk_velocity_rigid_inner(self.grid.r,
                                              v0_r0=v0_r0,
                                              p=p,
                                              r0=r0)

            # Line-of-sight projection (using evolved phi)
            if los_proj_func is not None:
                proj = los_proj_func(self.grid.r, phi_evolved, self.geom,
                                     obs_phase)
            elif hasattr(self.geom, "proj_factor"):
                proj = np.asarray(self.geom.proj_factor)
            else:
                inc = getattr(self.geom, "inclination_rad", np.deg2rad(90.0))
                phi0 = getattr(self.geom, "phi0", 0.0)
                proj = np.sin(inc) * np.sin(phi_evolved - phi0)
            proj = np.asarray(proj)
            if proj.shape == ():
                proj = np.full(self.grid.numPoints, float(proj))
            assert proj.shape[0] == self.grid.numPoints
            v_los = v_phi * proj

        # Check for stellar occultation
        occultation_mask = np.zeros(self.grid.numPoints, dtype=bool)
        if hasattr(self.geom, 'enable_stellar_occultation'
                   ) and self.geom.enable_stellar_occultation:
            phi_obs = getattr(self.geom, "phi_obs", 0.0)

            # Update phi_obs from obs_phase or time_phase if available
            if obs_phase is not None:
                phi_obs = 2.0 * np.pi * float(obs_phase)
            elif time_phase is not None:
                phi_obs = 2.0 * np.pi * float(time_phase)

            inclination_deg = np.rad2deg(
                getattr(self.geom, "inclination_rad", np.deg2rad(60.0)))
            stellar_radius = getattr(self.geom, "stellar_radius", 1.0)
            occultation_mask = self.grid.compute_stellar_occultation_mask(
                phi_obs=phi_obs,
                inclination_deg=inclination_deg,
                stellar_radius=stellar_radius,
                verbose=1)

        # Map observation velocity grid -> local wavelength grid per pixel
        c = C_KMS
        wl_obs = self.wl0 * (1.0 + self.v / c)  # (Nv,)
        denom = (1.0 + v_los / c)  # (Npix,)
        self.wl_cells = (wl_obs[:, None] / denom[None, :])  # (Nv, Npix)

        # Continuum weight Ic_weight: geometric weight
        # Apply occultation mask
        self.Ic_weight = W.copy()  # (Npix,)
        self.Ic_weight[occultation_mask] = 0.0

        # Store line model
        if line_model is None:
            raise ValueError(
                "line_model parameter is required. Pass a BaseLineModel instance "
                "(e.g., GaussianZeemanWeakLineModel)")
        self.line_model = line_model

        # Store geometric properties needed for spectrum computation
        self.W = W
        # self.cont should represent the total continuum flux of the VISIBLE region
        self.cont = np.sum(self.Ic_weight)
        self.v_los = v_los
        self.v_phi = v_phi if 'v_phi' in locals() else None

        # Compute initial spectrum
        self.compute_spectrum()

    def compute_spectrum(self, B_los=None, B_perp=None, chi=None, amp=None):
        """Compute Stokes spectra for current or updated magnetic field configuration.
        
        Parameters
        ----------
        B_los : np.ndarray, optional
            Line-of-sight magnetic field (Gauss). If provided, updates geometry.
        B_perp : np.ndarray, optional
            Perpendicular magnetic field (Gauss). If provided, updates geometry.
        chi : np.ndarray, optional
            Azimuthal angle (rad). If provided, updates geometry.
        amp : np.ndarray, optional
            Amplitude/weight. If provided, updates geometry.
            
        Returns
        -------
        np.ndarray
            Stokes I spectrum.
            Note: Stokes V, Q, U spectra are also computed and stored as
            self.V, self.Q, self.U attributes.
        """
        # Update geometry if parameters provided
        if B_los is not None:
            self.geom.B_los = np.asarray(B_los)
        if B_perp is not None:
            self.geom.B_perp = np.asarray(B_perp)
        if chi is not None:
            self.geom.chi = np.asarray(chi)
        if amp is not None:
            self.geom.amp = np.asarray(amp)

        # Get parameters from geometry
        if hasattr(self.geom, "amp") and self.geom.amp is not None:
            amp_arr = np.asarray(self.geom.amp, dtype=float)
        else:
            amp_arr = np.ones(self.grid.numPoints, dtype=float)

        # Update local amp reference for diagnostics
        self.amp = amp_arr

        # Convert intensity-like amp (1=continuum) to additive amp (0=continuum)
        # as expected by GaussianZeemanWeakLineModel
        # User requirement: amp=1 -> no signal, amp<1 -> absorption
        amp_model = amp_arr

        # Magnetic fields
        if hasattr(self.geom, "B_los") and self.geom.B_los is not None:
            Blos_arr = np.asarray(self.geom.B_los)
        else:
            Blos_arr = np.zeros(self.grid.numPoints, dtype=float)

        if hasattr(self.geom, "B_perp") and self.geom.B_perp is not None:
            Bperp_arr = np.asarray(self.geom.B_perp)
        else:
            Bperp_arr = None

        if hasattr(self.geom, "chi") and self.geom.chi is not None:
            chi_arr = np.asarray(self.geom.chi)
        else:
            chi_arr = None

        # Compute local profiles
        profiles = self.line_model.compute_local_profile(
            self.wl_cells,
            amp_model,
            Blos=Blos_arr,
            Bperp=Bperp_arr,
            chi=chi_arr,
            Ic_weight=self.Ic_weight)

        I_loc = profiles.get("I", None)
        V_loc = profiles.get("V", None)
        Q_loc = profiles.get("Q", None)
        U_loc = profiles.get("U", None)

        if I_loc is None:
            raise ValueError("line_model result missing key 'I'")
        if V_loc is None:
            V_loc = np.zeros_like(I_loc)
        if Q_loc is None:
            Q_loc = np.zeros_like(I_loc)
        if U_loc is None:
            U_loc = np.zeros_like(I_loc)

        # Subtract continuum contribution from I_loc before summing to get pure signal deviation
        # I_loc = (1 + signal) * weight = weight + signal * weight
        # I_dev = I_loc - weight = signal * weight
        I_dev = I_loc - self.Ic_weight

        # Sum over pixels
        I_sum_dev = np.sum(I_dev, axis=1)
        V_sum = np.sum(V_loc, axis=1)
        Q_sum = np.sum(Q_loc, axis=1)
        U_sum = np.sum(U_loc, axis=1)

        # Instrumental convolution
        if self.inst_fwhm > 0.0:
            I_conv_dev = convolve_gaussian_1d(I_sum_dev, self.dv,
                                              self.inst_fwhm)
            V_conv = convolve_gaussian_1d(V_sum, self.dv, self.inst_fwhm)
            Q_conv = convolve_gaussian_1d(Q_sum, self.dv, self.inst_fwhm)
            U_conv = convolve_gaussian_1d(U_sum, self.dv, self.inst_fwhm)
        else:
            I_conv_dev = I_sum_dev
            V_conv = V_sum
            Q_conv = Q_sum
            U_conv = U_sum

        # Normalization
        if self.normalize_continuum:
            baseline = 1.0
            if self.cont > 0:
                self.I = baseline + I_conv_dev / self.cont
                self.V = V_conv / self.cont
                self.Q = Q_conv / self.cont
                self.U = U_conv / self.cont
            else:
                self.I = np.full_like(I_conv_dev, baseline)
                self.V = V_conv
                self.Q = Q_conv
                self.U = U_conv
        else:
            self.I = self.cont + I_conv_dev
            self.V = V_conv
            self.Q = Q_conv
            self.U = U_conv

        return self.I

    def compute_derivatives(self, B_los=None, B_perp=None, chi=None, amp=None):
        """Compute analytical derivatives of Stokes spectra w.r.t parameters.
        
        Returns
        -------
        dict
            Dictionary of derivative matrices (Nlam, Npix).
            Keys: 'dI_damp', 'dV_dBlos', etc.
        """
        # Update geometry if parameters provided
        if B_los is not None:
            self.geom.B_los = np.asarray(B_los)
        if B_perp is not None:
            self.geom.B_perp = np.asarray(B_perp)
        if chi is not None:
            self.geom.chi = np.asarray(chi)
        if amp is not None:
            self.geom.amp = np.asarray(amp)

        # Get parameters from geometry
        if hasattr(self.geom, "amp") and self.geom.amp is not None:
            amp_arr = np.asarray(self.geom.amp, dtype=float)
        else:
            amp_arr = np.ones(self.grid.numPoints, dtype=float)

        amp_model = amp_arr

        # Magnetic fields
        if hasattr(self.geom, "B_los") and self.geom.B_los is not None:
            Blos_arr = np.asarray(self.geom.B_los)
        else:
            Blos_arr = np.zeros(self.grid.numPoints, dtype=float)

        if hasattr(self.geom, "B_perp") and self.geom.B_perp is not None:
            Bperp_arr = np.asarray(self.geom.B_perp)
        else:
            Bperp_arr = None

        if hasattr(self.geom, "chi") and self.geom.chi is not None:
            chi_arr = np.asarray(self.geom.chi)
        else:
            chi_arr = None

        # Compute local derivatives
        # Note: Ic_weight is passed to apply geometric weights to derivatives
        derivs = self.line_model.compute_local_derivatives(
            self.wl_cells,
            amp_model,
            Blos=Blos_arr,
            Bperp=Bperp_arr,
            chi=chi_arr,
            Ic_weight=self.Ic_weight)

        # Apply convolution and normalization
        processed_derivs = {}

        # Pre-calculate normalization factor
        norm_factor = 1.0
        if self.normalize_continuum and self.cont > 0:
            norm_factor = 1.0 / self.cont

        for key, val in derivs.items():
            # Convolution (along axis 0)
            if self.inst_fwhm > 0.0:
                val_conv = convolve_gaussian_1d(val, self.dv, self.inst_fwhm)
            else:
                val_conv = val

            # Normalization
            if norm_factor != 1.0:
                processed_derivs[key] = val_conv * norm_factor
            else:
                processed_derivs[key] = val_conv

        return processed_derivs

    # ========================================
    # Model I/O: geomodel.tomog
    # ========================================

    def write_geomodel(self, filepath, meta=None):
        """Export current geometry model and node physical quantities.
        
        Writes to text file (geomodel.tomog) for inspection and version control.
        
        Structure:
        - Header section with '#' prefix (key:value pairs)
        - Column names line (# COLUMNS: ...)
        - One line per pixel
        
        Pixel fields include: idx, ring_id, phi_id, r, phi, area, Ic_weight, amp,
        Blos, and optionally Bperp, chi
        
        Parameters
        ----------
        filepath : str
            Output file path
        meta : dict, optional
            Additional metadata to include in header
        """
        import datetime as _dt

        g = self.grid
        geom = self.geom

        # Collect metadata
        header = {
            "format":
            "TOMOG_MODEL",
            "version":
            1,
            "created_utc":
            _dt.datetime.utcnow().isoformat() + "Z",
            "wl0_nm":
            float(self.wl0),
            "inst_fwhm_kms":
            float(self.inst_fwhm),
            "normalize_continuum":
            bool(self.normalize_continuum),
            # Velocity field parameters
            "disk_v0_kms":
            getattr(self, "_disk_v0_kms", None),
            "disk_power_index":
            getattr(self, "_disk_power_index", None),
            "disk_r0":
            getattr(self, "_disk_r0", None),
            # Differential rotation and geometry
            "inclination_deg":
            float(
                np.rad2deg(getattr(geom, "inclination_rad",
                                   np.deg2rad(90.0)))),
            "phi0":
            float(getattr(geom, "phi0", 0.0)),
            "pOmega":
            float(getattr(geom, "pOmega", 0.0)),
            "r0_rot":
            float(getattr(geom, "r0", getattr(self, "_disk_r0", 1.0))),
            "period":
            float(getattr(geom, "period", 1.0)),
            # Grid definition
            "nr":
            int(getattr(g, "nr", len(getattr(g, "r_centers", []))))
        }

        # Optional: target/observation info from caller via meta
        if isinstance(meta, dict):
            for k, v in meta.items():
                header[str(k)] = v

        # Magnetic field components
        has_Blos = hasattr(geom, "B_los") and geom.B_los is not None
        has_Bperp = hasattr(geom, "B_perp") and geom.B_perp is not None
        has_chi = hasattr(geom, "chi") and geom.chi is not None

        # Spectral amplitude and geometry weight
        # Prefer geometry's intrinsic amp over integrator's effective amp (which might be masked)
        if hasattr(geom, "amp") and geom.amp is not None:
            amp = np.asarray(geom.amp)
        else:
            amp = np.asarray(getattr(self, "amp", np.ones(g.numPoints)))

        Ic_weight = np.asarray(getattr(self, "W", np.ones(g.numPoints)))

        # Column definitions
        columns = [
            "idx", "ring_id", "phi_id", "r", "phi", "area", "Ic_weight", "amp",
            "Blos", "Bperp", "chi"
        ]

        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# TOMOG Geometric Model File\n")
            for k in sorted(header.keys()):
                v = header[k]
                if isinstance(v, (list, tuple, np.ndarray)):
                    arr = np.asarray(v).ravel()
                    vstr = ",".join(f"{x:.12g}" for x in arr)
                    f.write(f"# {k}: [{vstr}]\n")
                else:
                    f.write(f"# {k}: {v}\n")

            # Append grid boundaries if available
            if hasattr(g, "r_edges"):
                vstr = ",".join(f"{x:.12g}"
                                for x in np.asarray(g.r_edges).ravel())
                f.write(f"# r_edges: [{vstr}]\n")

            f.write("# COLUMNS: " + ", ".join(columns) + "\n")

            N = g.numPoints
            for i in range(N):
                row = [
                    i,
                    int(g.ring_id[i]) if hasattr(g, "ring_id") else -1,
                    int(getattr(g, "phi_id", np.zeros_like(g.r, int))[i]),
                    float(g.r[i]),
                    float(g.phi[i]),
                    float(g.area[i]),
                    float(Ic_weight[i]),
                    float(amp[i]),
                    float(geom.B_los[i]) if has_Blos else 0.0,
                ]
                if has_Bperp:
                    row.append(float(geom.B_perp[i]))
                if has_chi:
                    row.append(float(geom.chi[i]))
                f.write(" ".join(str(x) for x in row) + "\n")

    @staticmethod
    def read_geomodel(filepath):
        """Read geomodel.tomog file.
        
        Parameters
        ----------
        filepath : str
            Path to geomodel.tomog file
        
        Returns
        -------
        tuple
            (geom_like, meta, table) where:
            - geom_like: geometry object with required attributes
            - meta: header key-value dictionary
            - table: data table (dict of np.ndarray)
        """
        import re as _re

        meta = {}
        rows = []
        columns = None
        r_edges = None

        with open(filepath, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("#"):
                    # Parse header key-value
                    m = _re.match(r"^#\s*([^:]+):\s*(.*)$", ln.strip())
                    if m:
                        k = m.group(1).strip()
                        v = m.group(2).strip()
                        if k == "COLUMNS":
                            columns = [s.strip() for s in v.split(",")]
                        elif k == "r_edges":
                            vs = v.strip()
                            vs = vs.strip("[]")
                            if vs:
                                r_edges = np.array(
                                    [float(x) for x in vs.split(",")])
                        else:
                            # Try to parse as number
                            vv = v
                            if vv.startswith("[") and vv.endswith("]"):
                                try:
                                    arr = [
                                        float(x)
                                        for x in vv.strip("[]").split(",")
                                        if x.strip()
                                    ]
                                    meta[k] = np.array(arr)
                                except Exception:
                                    meta[k] = vv
                            else:
                                try:
                                    if _re.match(r"^-?\d+$", vv):
                                        meta[k] = int(vv)
                                    else:
                                        meta[k] = float(vv)
                                except Exception:
                                    meta[k] = vv
                    continue
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                rows.append(parts)

        if columns is None:
            columns = [
                "idx", "ring_id", "phi_id", "r", "phi", "area", "Ic_weight",
                "amp", "Blos", "Bperp", "chi"
            ]

        # Organize into array table
        data = list(zip(*rows)) if rows else []
        table = {}
        for i, name in enumerate(columns):
            if i < len(data):
                try:
                    arr = np.array([float(x) for x in data[i]], dtype=float)
                except Exception:
                    arr = np.array(data[i])
                table[name] = arr

        # Construct grid-like object
        from types import SimpleNamespace as _NS

        grid = _NS()
        grid.r = table.get("r", np.array([]))
        grid.phi = table.get("phi", np.array([]))
        grid.area = table.get("area", np.array([]))
        base_r = grid.r if isinstance(grid.r, np.ndarray) else np.array([])
        grid.ring_id = table.get("ring_id", np.zeros_like(base_r,
                                                          int)).astype(int)
        grid.phi_id = table.get("phi_id", np.zeros_like(base_r,
                                                        int)).astype(int)
        grid.numPoints = base_r.shape[0]

        # Try to recover r_edges / r_centers / dr
        if r_edges is None and "nr" in meta and grid.numPoints > 0:
            nr = int(meta["nr"]) if isinstance(meta["nr"],
                                               (int, float)) else int(
                                                   meta["nr"])  # type: ignore
            r_med = []
            for rid in range(nr):
                mask = (grid.ring_id == rid)
                if np.any(mask):
                    r_med.append(np.median(grid.r[mask]))
            if r_med:
                r_med = np.array(r_med)
                dr = np.diff(r_med)
                dr0 = dr[0] if dr.size > 0 else (
                    r_med[0] if r_med.size > 0 else 1.0)
                r_edges = np.concatenate(
                    [[r_med[0] - 0.5 * dr0], 0.5 * (r_med[:-1] + r_med[1:]),
                     [r_med[-1] + 0.5 * (dr[-1] if dr.size > 0 else dr0)]])

        if r_edges is None:
            unique_r = np.unique(grid.r)
            if unique_r.size >= 2:
                mid = 0.5 * (unique_r[:-1] + unique_r[1:])
                dr0 = unique_r[1] - unique_r[0]
                r_edges = np.concatenate(
                    [[unique_r[0] - 0.5 * dr0], mid,
                     [unique_r[-1] + 0.5 * (unique_r[-1] - unique_r[-2])]])
            else:
                r_edges = np.array(
                    [0.0, unique_r[0] if unique_r.size else 1.0])

        grid.r_edges = r_edges
        grid.r_centers = 0.5 * (
            r_edges[:-1] + r_edges[1:]) if r_edges.size >= 2 else np.array([])
        grid.dr = (r_edges[1:] -
                   r_edges[:-1]) if r_edges.size >= 2 else np.array([])

        # Construct geom-like object
        geom = _NS()
        geom.grid = grid
        geom.area_proj = grid.area
        geom.inclination_rad = np.deg2rad(
            float(meta.get("inclination_deg", 90.0)))
        geom.phi0 = float(meta.get("phi0", 0.0))
        geom.pOmega = float(meta.get("pOmega", 0.0))
        geom.r0 = float(meta.get("r0_rot", meta.get("disk_r0", 1.0)))
        geom.period = float(meta.get("period", 1.0))

        # Magnetic field
        geom.B_los = table["Blos"].astype(float)
        geom.B_perp = table["Bperp"].astype(float)
        geom.chi = table["chi"].astype(float)

        # Spectral amplitude
        geom.amp = table["amp"].astype(float)

        return geom, meta, table


# ====================================================================
# Helper Functions
# ====================================================================


def create_disk_geometry_from_params(par,
                                     grid,
                                     B_los=None,
                                     B_perp=None,
                                     chi=None,
                                     amp=None,
                                     verbose=0):
    """Create SimpleDiskGeometry from parameter objects.
    
    Convenience function to create geometry container from readParamsTomog object.
    
    Parameters
    ----------
    par : readParamsTomog
        Parameter object with attributes:
        - inclination : float (degrees)
        - pOmega : float
        - radius : float (R_sun)
        - period : float (days)
        - enable_stellar_occultation : int, optional
    grid : diskGrid
        Disk grid
    B_los : np.ndarray, optional
        Line-of-sight magnetic field (Gauss)
    B_perp : np.ndarray, optional
        Perpendicular magnetic field strength (Gauss)
    chi : np.ndarray, optional
        Magnetic field azimuthal angle (rad)
    amp : np.ndarray, optional
        Spectral line amplitude (response weight), shape (grid.numPoints,)
    verbose : int, default=0
        Verbosity (0=silent, 1=normal, 2=detailed)
    
    Returns
    -------
    SimpleDiskGeometry
        Created geometry object
    """
    npix = grid.numPoints

    if verbose:
        print("[disk_geometry_integrator] Creating geometry from params...")
        print(
            f"  Grid: {grid.numPoints} pixels, r ∈ [{grid.r_in:.2f}, {grid.r_out:.2f}] R_sun"
        )

    geom = SimpleDiskGeometry(
        grid=grid,
        inclination_deg=float(getattr(par, 'inclination', 60.0)),
        phi0=0.0,
        pOmega=float(getattr(par, 'pOmega', 0.0)),
        r0=float(getattr(par, 'radius', 1.0)),
        period=float(getattr(par, 'period', 1.0)),
        enable_stellar_occultation=int(
            getattr(par, 'enable_stellar_occultation', 0)),
        stellar_radius=float(getattr(par, 'radius', 1.0)),
        B_los=B_los if B_los is not None else np.zeros(npix),
        B_perp=B_perp if B_perp is not None else np.zeros(npix),
        chi=chi if chi is not None else np.zeros(npix),
        amp=amp if amp is not None else np.ones(npix))

    if verbose:
        print(f"  {geom}")
        if verbose > 1:
            mag_summary = geom.get_magnetic_field_summary()
            print("  Magnetic field:")
            print(
                f"    B_los: [{mag_summary['B_los_min']:.1f}, {mag_summary['B_los_max']:.1f}] G, "
                f"mean={mag_summary['B_los_mean']:.1f} G")
            print(
                f"    B_perp: [{mag_summary['B_perp_min']:.1f}, {mag_summary['B_perp_max']:.1f}] G, "
                f"mean={mag_summary['B_perp_mean']:.1f} G")
            print("  Spectral amplitude:")
            print(
                f"    [{mag_summary['amp_min']:.2f}, {mag_summary['amp_max']:.2f}], "
                f"mean={mag_summary['amp_mean']:.2f}")

    return geom
