r"""Complete Physical Model Integration Module (physical_model.py)

This module serves as the central integration point for the physical modeling of the system.
It coordinates the initialization and interaction of the following core components:

1.  **Grid Generation (`grid_tom.py`)**: Creates the spatial discretization of the stellar/disk surface (annular rings).
2.  **Geometry & Physics (`disk_geometry.py`)**: Manages the physical properties on the grid, including:
    *   **Magnetic Field**: Vector magnetic field distribution ($B_{los}$, $B_{\perp}$, $\chi$).
    *   **Brightness/Response**: Local spectral line amplitude/weight ($amp$).
    *   **Kinematics**: Velocity field calculation (rotation, differential rotation).
3.  **Spectral Synthesis (`velspace_DiskIntegrator.py`)**: Performs the integration over the velocity space to synthesize Stokes profiles.

Functionality
-------------
*   **Unified Initialization**: Provides a single interface (`create_physical_model`) to initialize the entire physical system from a parameter file (`readParamsTomog`).
*   **Model Building**: The `PhysicalModelBuilder` class allows for step-by-step construction and customization of the model components.
*   **State Management**: The `PhysicalModel` class acts as a container for the current state of the system, essential for both forward modeling and the iterative inversion process.

Physics
-------
The model represents a rotating object (star or disk) where:
*   The surface is divided into a grid of pixels.
*   Each pixel has a local velocity vector determined by the rotation law (rigid or differential).
*   Each pixel has a local magnetic field vector and intrinsic line profile properties.
*   The observed spectrum is the integration of local profiles over the visible surface, accounting for Doppler shifts and projection effects.

Classes
-------
PhysicalModelBuilder : Builder class for coordinating the initialization of sub-modules.
PhysicalModel : Data container holding the complete state of the physical model.

Functions
---------
create_physical_model : Factory function to create a `PhysicalModel` instance directly from parameters.
"""

import numpy as np
from typing import Optional, Any, Dict
from dataclasses import dataclass, field

from core.grid_tom import diskGrid
from core.disk_geometry_integrator import SimpleDiskGeometry, VelspaceDiskIntegrator, create_disk_geometry_from_params
from core.mainFuncs import readParamsTomog

__all__ = [
    'PhysicalModel',
    'PhysicalModelBuilder',
    'create_physical_model',
]


@dataclass
class PhysicalModel:
    """Complete physical model container.
    
    Contains disk grid, geometry parameters, magnetic field configuration, brightness distribution, and velocity space integrator.
    This container provides all physical model components required for executing forward or inversion workflows.
    
    Attributes
    ----------
    par : readParamsTomog
        Original parameter object (kept for tracking)
    grid : diskGrid
        Disk grid (equal Δr stratification)
    geometry : SimpleDiskGeometry
        Disk geometry container (contains magnetic field parameters and brightness distribution)
    integrator : VelspaceDiskIntegrator or None
        Velocity space integrator (lazy initialization)
    
    Parameters from par
    -------------------
    inclination_deg : float
        Disk inclination (degrees)
    pOmega : float
        Differential rotation index
    period : float
        Rotation period (days)
    radius : float
        Stellar radius (R_sun)
    vsini : float
        Projected equatorial velocity (km/s)
    Vmax : float
        Grid maximum velocity (km/s)
    enable_stellar_occultation : int
        Stellar occultation flag
    nRingsStellarGrid : int
        Number of grid rings
        
    Magnetic Field
    ---------------
    B_los : np.ndarray
        Line-of-sight magnetic field component (Gauss)
    B_perp : np.ndarray
        Perpendicular magnetic field strength (Gauss)
    chi : np.ndarray
        Magnetic field direction angle (rad)
    
    Spectral Amplitude (Response)
    ---------------------------
    amp : np.ndarray
        Spectral line amplitude (response weight) (0-1), used to modulate emission/absorption
    """

    par: readParamsTomog
    grid: diskGrid
    geometry: SimpleDiskGeometry
    integrator: Optional[VelspaceDiskIntegrator] = None

    # Cache velocity-related parameters
    _v_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    _wl0: Optional[float] = None
    _line_model: Optional[Any] = None

    def __post_init__(self):
        """Validate physical model consistency."""
        self.validate()

    def validate(self) -> bool:
        """Validate consistency and completeness of physical model parameters.
        
        Check items:
        - grid and geometry pixel counts match
        - Magnetic field parameter dimensions match pixel count
        - Brightness distribution dimensions match pixel count
        - Parameter value ranges are reasonable
        
        Returns
        -------
        bool
            Returns True if validation passes, raises ValueError if fails
        
        Raises
        ------
        ValueError
            Raised when parameters are inconsistent or ranges are unreasonable
        """
        npix_grid = self.grid.numPoints
        npix_geom = self.geometry.grid.numPoints

        if npix_grid != npix_geom:
            raise ValueError(
                f"Grid pixel mismatch: grid.numPoints={npix_grid} vs "
                f"geometry.grid.numPoints={npix_geom}")

        # Magnetic field dimension check
        if len(self.geometry.B_los) != npix_grid:
            raise ValueError(f"B_los length ({len(self.geometry.B_los)}) != "
                             f"grid.numPoints ({npix_grid})")
        if len(self.geometry.B_perp) != npix_grid:
            raise ValueError(f"B_perp length ({len(self.geometry.B_perp)}) != "
                             f"grid.numPoints ({npix_grid})")
        if len(self.geometry.chi) != npix_grid:
            raise ValueError(f"chi length ({len(self.geometry.chi)}) != "
                             f"grid.numPoints ({npix_grid})")

        # Spectral line amplitude dimension check
        if len(self.geometry.amp) != npix_grid:
            raise ValueError(f"amp length ({len(self.geometry.amp)}) != "
                             f"grid.numPoints ({npix_grid})")

        # Parameter range check
        if self.par.inclination < 0 or self.par.inclination > 90:
            raise ValueError(
                f"inclination out of range: {self.par.inclination}° "
                f"(must be in [0, 90])")

        if self.par.period <= 0:
            raise ValueError(f"period must be positive, got {self.par.period}")

        if self.par.radius <= 0:
            raise ValueError(
                f"stellar radius must be positive, got {self.par.radius}")

        if self.grid.r_out <= self.grid.r_in:
            raise ValueError(
                f"grid radial range invalid: r_in={self.grid.r_in}, "
                f"r_out={self.grid.r_out}")

        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information of the physical model.
        
        Returns
        -------
        dict
            Dictionary containing key model parameters
        """
        return {
            'grid_npix': self.grid.numPoints,
            'grid_nr': self.grid.nr,
            'grid_r_range': (self.grid.r_in, self.grid.r_out),
            'inclination_deg': self.par.inclination,
            'period_day': self.par.period,
            'pOmega': self.par.pOmega,
            'vsini_kms': self.par.vsini,
            'Vmax_kms': self.par.Vmax,
            'stellar_radius_rsun': self.par.radius,
            'magnetic_field': self.geometry.get_magnetic_field_summary(),
        }

    def update_magnetic_field(self,
                              B_los: Optional[np.ndarray] = None,
                              B_perp: Optional[np.ndarray] = None,
                              chi: Optional[np.ndarray] = None) -> None:
        """Update magnetic field parameters.
        
        Parameters
        ----------
        B_los : np.ndarray, optional
            Line-of-sight magnetic field (Gauss)
        B_perp : np.ndarray, optional
            Perpendicular magnetic field strength (Gauss)
        chi : np.ndarray, optional
            Magnetic field direction angle (rad)
        
        Raises
        ------
        ValueError
            Raised when magnetic field dimensions do not match
        """
        npix = self.grid.numPoints

        if B_los is not None:
            B_los = np.asarray(B_los, dtype=float)
            if len(B_los) != npix:
                raise ValueError(
                    f"B_los length ({len(B_los)}) != grid.numPoints ({npix})")
            self.geometry.B_los = B_los

        if B_perp is not None:
            B_perp = np.asarray(B_perp, dtype=float)
            if len(B_perp) != npix:
                raise ValueError(
                    f"B_perp length ({len(B_perp)}) != grid.numPoints ({npix})"
                )
            self.geometry.B_perp = B_perp

        if chi is not None:
            chi = np.asarray(chi, dtype=float)
            if len(chi) != npix:
                raise ValueError(
                    f"chi length ({len(chi)}) != grid.numPoints ({npix})")
            self.geometry.chi = chi

    def update_amplitude(self, amp: np.ndarray) -> None:
        """Update spectral line amplitude (response weight).
        
        Spectral line amplitude is used to modulate emission/absorption, range [0, 1].
        amp=1.0 means maximum emission, amp=0.0 means no signal.
        
        Parameters
        ----------
        amp : np.ndarray
            Spectral line amplitude (normalized, 0-1), length should be grid.numPoints
        
        Raises
        ------
        ValueError
            Raised when dimensions do not match
        """
        npix = self.grid.numPoints
        amp = np.asarray(amp, dtype=float)

        if len(amp) != npix:
            raise ValueError(
                f"amp length ({len(amp)}) != grid.numPoints ({npix})")

        # 裁剪到 [0, 1] 范围
        amp = np.clip(amp, 0.0, 1.0)
        self.geometry.amp = amp


class PhysicalModelBuilder:
    """Physical model builder, coordinating initialization of sub-modules.
    
    Provides a flexible way to build physical models, supporting step-by-step initialization and custom parameters.
    
    Parameters
    ----------
    par : readParamsTomog
        Parameter object
    verbose : int, default=1
        Verbosity level (0=silent, 1=normal, 2=detailed)
    """

    def __init__(self, par: readParamsTomog, verbose: int = 1):
        """Initialize builder."""
        if not isinstance(par, readParamsTomog):
            raise TypeError(
                f"par must be readParamsTomog instance, got {type(par)}")
        self.par = par
        self.verbose = int(verbose)
        self._grid = None
        self._geometry = None

    def build_grid(self,
                   nr: Optional[int] = None,
                   r_in: Optional[float] = None,
                   r_out: Optional[float] = None,
                   target_pixels_per_ring: Optional[Any] = None) -> diskGrid:
        """Build disk grid.
        
        Parameters
        ----------
        nr : int, optional
            Number of rings. If not provided, automatically adjusted based on r_out to maintain radial resolution consistent with par configuration.
        r_in : float, optional
            Inner radius (R_sun), default 0.0
        r_out : float, optional
            Outer radius (R_sun), default from par.r_out
        target_pixels_per_ring : optional
            Pixels per ring configuration, supports int or array-like
        
        Returns
        -------
        diskGrid
            Created grid object
        """
        # 1. Determine geometric boundaries
        r_in_val = float(r_in) if r_in is not None else 0.0

        # Logic: Determine r_out from Vmax if provided (Projected velocity limit)
        par_r_out = float(getattr(self.par, 'r_out', 0.0))
        par_Vmax = float(getattr(self.par, 'Vmax', 0.0))

        if r_out is not None:
            r_out_val = float(r_out)
        elif par_Vmax > 1e-6:
            # Calculate from Vmax using vsini (projected) to ensure independence from inclination
            vsini = float(getattr(self.par, 'vsini', 10.0))
            pOmega = float(getattr(self.par, 'pOmega', 0.0))
            radius = float(getattr(self.par, 'radius', 1.0))

            # Avoid division by zero or invalid power
            if abs(vsini) > 1e-9 and abs(pOmega + 1.0) > 1e-6:
                r_out_stellar = (par_Vmax / vsini)**(1.0 / (pOmega + 1.0))
                r_out_val = radius * r_out_stellar
                if self.verbose:
                    print(
                        f"[PhysicalModelBuilder] Calculated r_out={r_out_val:.2f} from Vmax={par_Vmax:.1f}, vsini={vsini:.1f}"
                    )
            else:
                r_out_val = par_r_out if par_r_out > 0 else 5.0
        else:
            r_out_val = par_r_out if par_r_out > 0 else 5.0

        # 2. Determine number of rings nr
        if nr is not None:
            nr_val = int(nr)
        else:
            # Auto-adjustment logic: maintain radial resolution
            par_nr = int(getattr(self.par, 'nRingsStellarGrid', 60))
            par_r_out = float(getattr(self.par, 'r_out', 5.0))

            # If r_out changes significantly, adjust nr to keep dr constant
            # Assuming par config is based on r_in=0 (usual case)
            if abs(r_out_val - par_r_out) > 1e-6 and par_r_out > 0:
                # base_dr = par_r_out / par_nr
                # new_nr = (r_out - r_in) / base_dr
                base_dr = par_r_out / max(par_nr, 1)
                nr_val = int(np.ceil((r_out_val - r_in_val) / base_dr))
                # Ensure at least 1 ring
                nr_val = max(1, nr_val)

                if self.verbose:
                    print(
                        f"[PhysicalModelBuilder] Auto-adjusted nr={nr_val} "
                        f"(base: nr={par_nr} @ r_out={par_r_out}) to maintain resolution"
                    )
            else:
                nr_val = par_nr

        if self.verbose:
            print(f"[PhysicalModelBuilder] Creating grid: nr={nr_val}, "
                  f"r ∈ [{r_in_val:.2f}, {r_out_val:.2f}] R_sun")

        self._grid = diskGrid(nr=nr_val,
                              r_in=r_in_val,
                              r_out=r_out_val,
                              target_pixels_per_ring=target_pixels_per_ring,
                              verbose=self.verbose)

        return self._grid

    def build_geometry(self,
                       grid: Optional[diskGrid] = None,
                       B_los: Optional[np.ndarray] = None,
                       B_perp: Optional[np.ndarray] = None,
                       chi: Optional[np.ndarray] = None,
                       amp: Optional[np.ndarray] = None) -> SimpleDiskGeometry:
        """Build geometry object.
        
        Parameters
        ----------
        grid : diskGrid, optional
            Grid object (default uses grid created by build_grid).
        B_los : np.ndarray, optional
            Line-of-sight magnetic field (Gauss).
        B_perp : np.ndarray, optional
            Perpendicular magnetic field strength (Gauss).
        chi : np.ndarray, optional
            Magnetic field direction angle (rad).
        amp : np.ndarray, optional
            Spectral line amplitude distribution (response weight, >0), directly from geometry.
        
        Returns
        -------
        SimpleDiskGeometry
            Created geometry container.
        """
        if grid is None:
            if self._grid is None:
                self.build_grid()
            grid = self._grid

        if self.verbose:
            print(
                "[PhysicalModelBuilder] Creating geometry from parameters...")

        self._geometry = create_disk_geometry_from_params(self.par,
                                                          grid,
                                                          B_los=B_los,
                                                          B_perp=B_perp,
                                                          chi=chi,
                                                          amp=amp,
                                                          verbose=self.verbose)

        return self._geometry

    def build_integrator(self,
                         geometry: Optional[SimpleDiskGeometry] = None,
                         wl0_nm: float = 656.3,
                         v_grid: Optional[np.ndarray] = None,
                         line_model: Optional[Any] = None,
                         **integrator_kwargs) -> VelspaceDiskIntegrator:
        """Build velocity space integrator.
        
        Parameters
        ----------
        geometry : SimpleDiskGeometry, optional
            Geometry object (default uses geometry created by build_geometry).
        wl0_nm : float, default=656.3
            Central wavelength (nm).
        v_grid : np.ndarray, optional
            Velocity grid (km/s).
        line_model : optional
            Line model object (required).
        **integrator_kwargs
            Other arguments passed to VelspaceDiskIntegrator.
        
        Returns
        -------
        VelspaceDiskIntegrator
            Created integrator.
        
        Raises
        ------
        ValueError
            Raised when line_model is None.
        """
        if geometry is None:
            if self._geometry is None:
                self.build_geometry()
            geometry = self._geometry

        if line_model is None:
            raise ValueError(
                "line_model is required for VelspaceDiskIntegrator")

        # Auto-extract velocity parameters from par (if not provided in kwargs)
        if 'disk_v0_kms' not in integrator_kwargs:
            # Calculate equatorial velocity v_eq = vsini / sin(i)
            vsini = float(getattr(self.par, 'vsini', 10.0))
            inc_deg = float(getattr(self.par, 'inclination', 60.0))
            inc_rad = np.deg2rad(inc_deg)
            # Avoid division by zero
            if abs(np.sin(inc_rad)) > 1e-6:
                v_eq = vsini / np.sin(inc_rad)
            else:
                v_eq = vsini  # Fallback for face-on (though vsini is 0 then)

            integrator_kwargs['disk_v0_kms'] = v_eq
            if self.verbose:
                print(
                    f"[PhysicalModelBuilder] Auto-set disk_v0_kms={v_eq:.2f} km/s "
                    f"(from vsini={vsini}, i={inc_deg}°)")

        if 'disk_power_index' not in integrator_kwargs:
            integrator_kwargs['disk_power_index'] = float(
                getattr(self.par, 'pOmega', 0))

        if 'disk_r0' not in integrator_kwargs:
            # Prefer r0_rot (if exists in par), otherwise use radius
            # Note: readParamsTomog usually stores radius as par.radius
            integrator_kwargs['disk_r0'] = float(
                getattr(self.par, 'radius', 1.0))

        if 'normalize_continuum' not in integrator_kwargs:
            integrator_kwargs['normalize_continuum'] = bool(
                getattr(self.par, 'normalize_continuum', True))

        # Generate default velocity grid (if not provided)
        if v_grid is None:
            Vmax = float(self.par.Vmax)
            dv = 1.0  # km/s per pixel (default)
            n_vel = int(2 * Vmax / dv) + 1
            v_grid = np.linspace(-Vmax, Vmax, n_vel)
            if self.verbose:
                print(f"[PhysicalModelBuilder] Generated velocity grid: "
                      f"v ∈ [{-Vmax:.1f}, {Vmax:.1f}] km/s, "
                      f"N_v={len(v_grid)}, dv={dv:.1f}")

        if self.verbose:
            print(f"[PhysicalModelBuilder] Creating integrator: "
                  f"wl0={wl0_nm:.1f} nm")

        integrator = VelspaceDiskIntegrator(geom=geometry,
                                            wl0_nm=wl0_nm,
                                            v_grid=v_grid,
                                            line_model=line_model,
                                            **integrator_kwargs)

        return integrator

    def build(self,
              wl0_nm: float = 656.3,
              v_grid: Optional[np.ndarray] = None,
              line_model: Optional[Any] = None,
              B_los: Optional[np.ndarray] = None,
              B_perp: Optional[np.ndarray] = None,
              chi: Optional[np.ndarray] = None,
              amp: Optional[np.ndarray] = None,
              **grid_kwargs) -> PhysicalModel:
        """Build complete physical model (one-stop interface).
        
        Parameters
        ----------
        wl0_nm : float, default=656.3
            Central wavelength (nm).
        v_grid : np.ndarray, optional
            Velocity grid (km/s).
        line_model : optional
            Line model object (required).
        B_los : np.ndarray, optional
            Line-of-sight magnetic field (Gauss).
        B_perp : np.ndarray, optional
            Perpendicular magnetic field strength (Gauss).
        chi : np.ndarray, optional
            Magnetic field direction angle (rad).
        amp : np.ndarray, optional
            Spectral line amplitude distribution (response weight, >0), directly from geometry.
        **grid_kwargs
            Keyword arguments passed to build_grid.
        
        Returns
        -------
        PhysicalModel
            Complete physical model object.
        """
        # Step-by-step build
        self.build_grid(**grid_kwargs)
        self.build_geometry(B_los=B_los, B_perp=B_perp, chi=chi, amp=amp)
        integrator = self.build_integrator(wl0_nm=wl0_nm,
                                           v_grid=v_grid,
                                           line_model=line_model)

        # Ensure geometry is not None
        if self._geometry is None:
            raise RuntimeError("Geometry object was not created successfully")

        # Ensure grid is not None
        if self._grid is None:
            raise RuntimeError("Grid object was not created successfully")

        # Assemble PhysicalModel
        model = PhysicalModel(
            par=self.par,
            grid=self._grid,
            geometry=self._geometry,
            integrator=integrator,
            _v_grid=v_grid if v_grid is not None else np.linspace(
                -self.par.Vmax, self.par.Vmax,
                int(2 * self.par.Vmax) + 1),
            _wl0=float(wl0_nm),
            _line_model=line_model,
        )

        if self.verbose:
            print("[PhysicalModelBuilder] Physical model built successfully")
            print(f"  {model.geometry}")
            summary = model.get_summary()
            print(f"  Summary: {summary['grid_npix']} pixels, "
                  f"r ∈ [{summary['grid_r_range'][0]:.1f}, "
                  f"{summary['grid_r_range'][1]:.1f}] R_sun, "
                  f"inclination={summary['inclination_deg']:.0f}°")

        return model


def create_physical_model(par: readParamsTomog,
                          wl0_nm: float = 656.3,
                          v_grid: Optional[np.ndarray] = None,
                          line_model: Optional[Any] = None,
                          B_los: Optional[np.ndarray] = None,
                          B_perp: Optional[np.ndarray] = None,
                          chi: Optional[np.ndarray] = None,
                          amp: Optional[np.ndarray] = None,
                          verbose: int = 1,
                          **grid_kwargs) -> PhysicalModel:
    """Convenience function: create complete physical model directly from parameter object.

    Parameters
    ----------
    par : readParamsTomog
        Parameter object
    wl0_nm : float, default=656.3
        Central wavelength (nm)
    v_grid : np.ndarray, optional
        Velocity grid (km/s)
    line_model : optional
        Line model object (required)
    B_los : np.ndarray, optional
        Line-of-sight magnetic field (Gauss)
    B_perp : np.ndarray, optional
        Perpendicular magnetic field strength (Gauss)
    chi : np.ndarray, optional
        Magnetic field direction angle (rad)
    amp : np.ndarray, optional
        Spectral line amplitude distribution (response weight, >0), directly from geometry
    verbose : int, default=1
        Verbosity level
    **grid_kwargs
        Keyword arguments passed to grid building
    
    Returns
    -------
    PhysicalModel
        Complete physical model object
    
    Examples
    --------
    >>> from core.mainFuncs import readParamsTomog
    >>> from core.local_linemodel_basic import GaussianZeemanWeakLineModel
    >>> from core.physical_model import create_physical_model
    >>> 
    >>> # Read parameters
    >>> par = readParamsTomog('input/params_tomog.txt')
    >>> 
    >>> # Create line model
    >>> line_model = GaussianZeemanWeakLineModel()
    >>> 
    >>> # Create physical model
    >>> phys_model = create_physical_model(
    ...     par,
    ...     wl0_nm=656.3,
    ...     line_model=line_model,
    ...     verbose=1
    ... )
    >>> 
    >>> # Check model
    >>> print(phys_model.get_summary())
    >>> phys_model.validate()
    """
    # -------------------------------------------------------------------------
    # Handle initialization from .tomog file if enabled
    # -------------------------------------------------------------------------
    if getattr(par, 'initTomogFile', 0) == 1 and getattr(
            par, 'initModelPath', None):
        model_path = par.initModelPath
        if verbose:
            print(
                f"[create_physical_model] Loading initial model from: {model_path}"
            )

        try:
            # Load the model
            geom_loaded, meta, table = VelspaceDiskIntegrator.read_geomodel(
                model_path)

            # 1. Override parameters in par
            if 'inclination_deg' in meta:
                par.inclination = float(meta['inclination_deg'])
            if 'pOmega' in meta:
                par.pOmega = float(meta['pOmega'])
            if 'period' in meta:
                par.period = float(meta['period'])
            if 'r0_rot' in meta:
                par.radius = float(
                    meta['r0_rot'])  # Assuming r0_rot corresponds to radius
            if 'nr' in meta:
                par.nRingsStellarGrid = int(meta['nr'])

            # Update grid extent from loaded model
            if hasattr(geom_loaded.grid, 'r'):
                # Try to get r_in and r_out from r_edges if available
                r_min_val = np.min(geom_loaded.grid.r)
                r_max_val = np.max(geom_loaded.grid.r)

                if hasattr(geom_loaded.grid, 'r_edges') and len(
                        geom_loaded.grid.r_edges) > 0:
                    r_min_val = np.min(geom_loaded.grid.r_edges)
                    r_max_val = np.max(geom_loaded.grid.r_edges)

                par.r_out = float(r_max_val)

                # Update r_in in grid_kwargs
                grid_kwargs['r_in'] = float(r_min_val)

                if verbose:
                    print(
                        f"  ✓ Updated grid extent: r_in={r_min_val:.2f}, r_out={par.r_out:.2f}"
                    )

            if verbose:
                print(
                    f"  ✓ Parameters updated from model file (inc={par.inclination}°, nr={par.nRingsStellarGrid})"
                )

            # 2. Extract physical quantities (if not explicitly provided)
            if B_los is None and hasattr(geom_loaded, 'B_los'):
                B_los = geom_loaded.B_los
                if verbose and B_los is not None:
                    print(f"  ✓ Loaded B_los ({len(B_los)} pixels)")

            if B_perp is None and hasattr(geom_loaded, 'B_perp'):
                B_perp = geom_loaded.B_perp
                if verbose and B_perp is not None:
                    print(f"  ✓ Loaded B_perp ({len(B_perp)} pixels)")

            if chi is None and hasattr(geom_loaded, 'chi'):
                chi = geom_loaded.chi
                if verbose and chi is not None:
                    print(f"  ✓ Loaded chi ({len(chi)} pixels)")

            if amp is None and hasattr(geom_loaded, 'amp'):
                amp = geom_loaded.amp
                if verbose and amp is not None:
                    print(f"  ✓ Loaded amp ({len(amp)} pixels)")

        except Exception as e:
            print(
                f"[create_physical_model] ⚠️  Error loading model from {model_path}: {e}"
            )
            print("  Continuing with default initialization...")

    builder = PhysicalModelBuilder(par, verbose=verbose)
    return builder.build(wl0_nm=wl0_nm,
                         v_grid=v_grid,
                         line_model=line_model,
                         B_los=B_los,
                         B_perp=B_perp,
                         chi=chi,
                         amp=amp,
                         **grid_kwargs)
