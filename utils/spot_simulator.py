# spot_simulator.py (Simplified)
# Spot Simulation Library: Multi-spot configuration, geometry model construction, .tomog output
# Features:
#   1. Management of multiple spots (position, magnetic field, time evolution)
#   2. Mapping spots to disk grid pixels
#   3. Generating compliant .tomog model files
#   4. Note: Spectral synthesis is handled by 0-iter forward modeling in pyzeetom/tomography.py
#           Load model via initTomogFile parameter

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import interpolate

# Internal module imports
# Note: If circular dependency issues arise, some imports may need to be moved inside functions
try:
    from core.disk_geometry_integrator import (SimpleDiskGeometry,
                                               VelspaceDiskIntegrator)
    from core.grid_tom import diskGrid
    from core.local_linemodel_basic import (GaussianZeemanWeakLineModel,
                                            LineData)
    from core.tomography_result import ForwardModelResult
except ImportError:
    # Only to prevent syntax check errors in environments missing the core module;
    # ensure core is in the path during actual runtime
    pass


@dataclass
class SpotConfig:
    """Configuration parameters for a single spot"""
    r: float  # Radial position (in disk radius units)
    phi: float  # Initial azimuth (radians)
    amplitude: float  # Amplitude (positive=emission, negative=absorption)
    spot_type: str = 'emission'  # 'emission' or 'absorption'
    radius: float = 0.5  # Radial width of the spot (FWHM)
    width_type: str = 'gaussian'  # Width type 'gaussian' or 'tophat'
    B_los: float = 0.0  # Line-of-sight magnetic field
    B_perp: float = 0.0  # Transverse magnetic field
    chi: float = 0.0  # Magnetic field azimuth (radians)
    velocity_shift: float = 0.0  # Additional velocity shift (km/s)

    def __post_init__(self):
        self.r = float(self.r)
        self.phi = float(self.phi)
        self.amplitude = float(self.amplitude)
        self.radius = float(self.radius)
        self.B_los = float(self.B_los)
        self.B_perp = float(self.B_perp)
        self.chi = float(self.chi)
        self.velocity_shift = float(self.velocity_shift)


class SpotSimulator:
    """
    Spot Simulator:
      - Manages multiple spot configurations
      - Maps spots to disk grid pixels
      - Computes response functions and magnetic field distributions
      - Outputs .tomog model files for subsequent forward modeling
    """

    def __init__(self,
                 grid,
                 inclination_rad: float = np.deg2rad(60.0),
                 phi0: float = 0.0,
                 pOmega: float = -0.5,
                 r0_rot: float = 1.0,
                 period_days: float = 1.0):
        """
        Initialize the spot simulator

        Parameters:
        -------
        grid : diskGrid
            Grid object (from grid_tom.py)
        inclination_rad : float
            Inclination (radians), default 60 degrees
        phi0 : float
            Reference azimuth (radians)
        pOmega : float
            Differential rotation index (Omega propto r^pOmega)
        r0_rot : float
            Differential rotation reference radius
        period_days : float
            Rotation period (days)
        """
        self.grid = grid
        self.inclination_rad = float(inclination_rad)
        self.phi0 = float(phi0)
        self.pOmega = float(pOmega)
        self.r0_rot = float(r0_rot)
        self.period_days = float(period_days)

        # List of spots
        self.spots: List[SpotConfig] = []

        # Observation parameter arrays (same length as phase)
        # These parameters are used to generate multi-phase synthetic data
        self.phases: np.ndarray = np.array([])  # Observation phases (0~1)
        self.vel_shifts: np.ndarray = np.array(
            [])  # Velocity shifts (km/s), one per phase
        self.pol_channels: list = [
        ]  # Polarization channels, one per phase ('I'/'V'/'Q'/'U')

        # Initialize pixel attribute arrays
        self._init_pixel_arrays()

    def _init_pixel_arrays(self):
        """Initialize all pixel attribute arrays"""
        n = self.grid.numPoints
        self.amp = np.ones(
            n, dtype=float)  # Line amplitude (>0, directly from geometry)
        self.B_los_map = np.zeros(n, dtype=float)
        self.B_perp_map = np.zeros(n, dtype=float)
        self.chi_map = np.zeros(n, dtype=float)

    def add_spot(self, spot_config: SpotConfig):
        """Add a single spot"""
        self.spots.append(spot_config)

    def add_spots(self, spot_configs: List[SpotConfig]):
        """Add multiple spots"""
        self.spots.extend(spot_configs)

    def create_spot(self, r: float, phi: float, amplitude: float,
                    **kwargs) -> SpotConfig:
        """
        Create and add a single spot (shortcut method)

        Parameters: See SpotConfig
        Returns: SpotConfig object
        """
        spot = SpotConfig(r=r, phi=phi, amplitude=amplitude, **kwargs)
        self.add_spot(spot)
        return spot

    def evolve_spots_to_phase(self, phase: float) -> List[SpotConfig]:
        """
        Evolve spots to a specified phase (considering differential rotation)

        Parameters:
        -------
        phase : float
            Rotation phase (0~1)

        Returns:
        -------
        List[SpotConfig]
            List of evolved spots
        """
        evolved_spots = []
        for spot in self.spots:
            # Copy current spot configuration
            s = SpotConfig(r=spot.r,
                           phi=spot.phi,
                           amplitude=spot.amplitude,
                           spot_type=spot.spot_type,
                           radius=spot.radius,
                           width_type=spot.width_type,
                           B_los=spot.B_los,
                           B_perp=spot.B_perp,
                           chi=spot.chi,
                           velocity_shift=spot.velocity_shift)

            # Phase evolution considering differential rotation
            # Delta phi = 2pi * phase * (r/r0_rot)^(pOmega+1)
            # This follows from pOmega definition: Omega(r) = Omega_ref * (r/r0_rot)^pOmega
            radius_ratio = s.r / self.r0_rot if self.r0_rot > 0 else 1.0
            delphi = 2.0 * np.pi * phase * (radius_ratio**(self.pOmega + 1.0))
            s.phi = s.phi + delphi

            # Normalize to [0, 2pi)
            s.phi = s.phi % (2.0 * np.pi)

            evolved_spots.append(s)

        return evolved_spots

    def _gaussian_weight(self, dr: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian weight function"""
        return np.exp(-0.5 * (dr / sigma)**2)

    def _tophat_weight(self, dr: np.ndarray, radius: float) -> np.ndarray:
        """Top-hat weight function (rect function)"""
        weight = np.zeros_like(dr)
        weight[np.abs(dr) <= radius] = 1.0
        return weight

    def _compute_azimuthal_weight(self, dphi: np.ndarray,
                                  sigma_phi: float) -> np.ndarray:
        """Azimuthal weight (Gaussian)"""
        # Handle 2pi periodicity
        dphi_wrapped = np.abs(dphi)
        dphi_wrapped = np.minimum(dphi_wrapped, 2.0 * np.pi - dphi_wrapped)
        return np.exp(-0.5 * (dphi_wrapped / sigma_phi)**2)

    def apply_spots_to_grid(self, phase: float = 0.0) -> None:
        """
        Apply spots to the grid, computing response and magnetic field distribution

        Parameters:
        -------
        phase : float
            Rotation phase (0~1)
        """
        # Initialize
        self._init_pixel_arrays()

        # Get evolved spots
        spots_at_phase = self.evolve_spots_to_phase(phase)

        # Apply each spot to the grid
        for spot in spots_at_phase:
            self._apply_single_spot(spot)

    def _apply_single_spot(self, spot: SpotConfig) -> None:
        """Apply a single spot to the grid"""
        # Calculate distance from each pixel to the spot
        dr = np.sqrt((self.grid.r - spot.r)**2)
        dphi = self.grid.phi - spot.phi

        # Radial weight
        if spot.width_type == 'gaussian':
            sigma_r = spot.radius / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            r_weight = self._gaussian_weight(dr, sigma_r)
        else:  # tophat
            r_weight = self._tophat_weight(dr, spot.radius)

        # Azimuthal weight (Gaussian, width is spot.radius in azimuthal direction)
        sigma_phi = spot.radius / (self.grid.r + 1e-10
                                   )  # Prevent division by zero
        phi_weight = self._compute_azimuthal_weight(dphi, sigma_phi)

        # Total weight
        weight = r_weight * phi_weight

        # Apply line amplitude
        if spot.spot_type == 'emission':
            # Emission: amp > 1
            self.amp += (spot.amplitude) * weight
        else:  # absorption
            # Absorption: amp < 1
            self.amp += (spot.amplitude) * weight

        # Apply magnetic field (weighted average)
        weight_sum = np.sum(weight)
        if weight_sum > 0:
            self.B_los_map += spot.B_los * weight
            self.B_perp_map += spot.B_perp * weight
            self.chi_map += spot.chi * weight

    def create_geometry_object(self, phase: float = 0.0):
        """
        Create a geometry object for use by VelspaceDiskIntegrator

        Parameters:
        -------
        phase : float
            Rotation phase

        Returns:
        -------
        SimpleDiskGeometry
            Geometry object containing grid, magnetic field parameters, line amplitudes, etc.
        """
        # Apply spots to grid
        self.apply_spots_to_grid(phase)

        # Construct SimpleDiskGeometry object
        geom = SimpleDiskGeometry(grid=self.grid,
                                  inclination_deg=float(
                                      np.rad2deg(self.inclination_rad)),
                                  phi0=float(self.phi0),
                                  pOmega=float(self.pOmega),
                                  r0=float(self.r0_rot),
                                  period=float(self.period_days),
                                  enable_stellar_occultation=0,
                                  stellar_radius=1.0,
                                  amp=self.amp.copy(),
                                  B_los=self.B_los_map.copy(),
                                  B_perp=self.B_perp_map.copy(),
                                  chi=self.chi_map.copy())

        return geom

    def export_to_geomodel(self,
                           filepath: str,
                           phase: float = 0.0,
                           meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Export the model to a .tomog file for subsequent forward modeling

        This method uses VelspaceDiskIntegrator.write_geomodel() for standardized output.

        Parameters:
        -------
        filepath : str
            Output file path (.tomog format)
        phase : float
            Phase
        meta : dict, optional
            Metadata (target name, observation parameters, etc.)

        Returns:
        -------
        str
            Output file path
        """
        # Create geometry object
        geom = self.create_geometry_object(phase=phase)

        # Set metadata
        if meta is None:
            meta = {}
        meta['phase'] = float(phase)
        meta['source'] = 'SpotSimulator'

        # Create a dummy integrator to use the write_geomodel method
        # Here we only need the integrator's write_geomodel method and access to the geometry
        try:
            # Try to create a full integrator (requires line model)
            # But only for calling write_geomodel, not for full spectral synthesis
            line_data = LineData('input/lines.txt')
            line_model = GaussianZeemanWeakLineModel(line_data)

            v_grid = np.array([0.0])  # Dummy velocity grid
            integrator = VelspaceDiskIntegrator(geom=geom,
                                                wl0_nm=656.3,
                                                v_grid=v_grid,
                                                line_model=line_model,
                                                inst_fwhm_kms=0.0,
                                                normalize_continuum=False)

            # Use integrator's write_geomodel method
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            integrator.write_geomodel(filepath, meta=meta)

        except Exception as e:
            # If full integrator cannot be created, use simplified export method
            print(
                f"[SpotSimulator] Warning: Could not use VelspaceDiskIntegrator.write_geomodel(): {e}"
            )
            print(f"[SpotSimulator] Falling back to simplified export...")

            # Construct header
            header = {
                "format": "TOMOG_MODEL",
                "version": 1,
                "created_utc": dt.datetime.utcnow().isoformat() + "Z",
                "source": "SpotSimulator",
                "inclination_deg": float(np.rad2deg(self.inclination_rad)),
                "phi0": float(self.phi0),
                "pOmega": float(self.pOmega),
                "r0_rot": float(self.r0_rot),
                "period": float(self.period_days),
                "nr": int(self.grid.nr) if hasattr(self.grid, 'nr') else -1,
                "phase": float(phase),
            }

            if isinstance(meta, dict):
                for k, v in meta.items():
                    if k not in header:
                        header[str(k)] = v

            # Check for B_perp and chi
            has_B_perp = np.any(self.B_perp_map != 0.0)
            has_chi = np.any(self.chi_map != 0.0)

            # Column definitions
            columns = [
                "idx", "ring_id", "phi_id", "r", "phi", "area", "Ic_weight",
                "amp", "Blos"
            ]
            if has_B_perp:
                columns.extend(["Bperp"])
            if has_chi:
                columns.extend(["chi"])

            # Write file
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(
                    "# TOMOG Geometric Model File (generated by SpotSimulator)\n"
                )
                for k in sorted(header.keys()):
                    v = header[k]
                    if isinstance(v, (list, tuple, np.ndarray)):
                        arr = np.asarray(v).ravel()
                        vstr = ",".join(f"{x:.12g}" for x in arr)
                        f.write(f"# {k}: [{vstr}]\n")
                    else:
                        f.write(f"# {k}: {v}\n")

                # Try to supplement grid edges
                if hasattr(self.grid, "r_edges"):
                    vstr = ",".join(
                        f"{x:.12g}"
                        for x in np.asarray(self.grid.r_edges).ravel())
                    f.write(f"# r_edges: [{vstr}]\n")

                f.write("# COLUMNS: " + ", ".join(columns) + "\n")

                # Write data
                n = self.grid.numPoints
                for i in range(n):
                    row = [
                        i,
                        int(self.grid.ring_id[i]) if hasattr(
                            self.grid, "ring_id") else -1,
                        int(getattr(self.grid, "phi_id", np.zeros(n, int))[i]),
                        float(self.grid.r[i]),
                        float(self.grid.phi[i]),
                        float(self.grid.area[i]),
                        float(1.0),  # Ic_weight (geometric weight)
                        float(self.amp[i]),  # Line amplitude
                        float(self.B_los_map[i]),
                    ]
                    if has_B_perp:
                        row.append(float(self.B_perp_map[i]))
                    if has_chi:
                        row.append(float(self.chi_map[i]))
                    f.write(" ".join(str(x) for x in row) + "\n")

        return filepath

    def configure_multi_phase_synthesis(
            self,
            phases: np.ndarray,
            vel_shifts: Optional[np.ndarray] = None,
            pol_channels: Optional[list] = None) -> None:
        """
        Configure multi-phase synthesis parameters

        Parameters:
        -------
        phases : np.ndarray
            Observation phase array, shape (N_phase,), values can be any real number (-inf, +inf)
            Due to differential rotation (pOmega), different phase values result in different spot positions.
            Physical meaning:
              - phase in [0, 1)  : Fractional position within rotation period
              - phase >= 1       : Position after multiple rotation periods
              - phase < 0        : Reverse time evolution
            Example: phase=-0.3, 0.3, 1.3 result in different geometric configurations due to pOmega
        vel_shifts : np.ndarray, optional
            Velocity shift array, shape (N_phase,), unit km/s
            Default is all zeros
        pol_channels : list, optional
            Polarization channel list, length N_phase
            Each element is 'I', 'V', 'Q', or 'U'
            Default is all 'V'

        Returns:
        -------
        None
        """
        phases = np.asarray(phases, dtype=float)
        n_phase = len(phases)

        # Set velocity shifts
        if vel_shifts is None:
            vel_shifts = np.zeros(n_phase, dtype=float)
        else:
            vel_shifts = np.asarray(vel_shifts, dtype=float)
            if len(vel_shifts) != n_phase:
                raise ValueError(
                    f"vel_shifts length ({len(vel_shifts)}) must match phases length ({n_phase})"
                )

        # Set polarization channels
        if pol_channels is None:
            pol_channels = ['V'] * n_phase
        else:
            pol_channels = list(pol_channels)
            if len(pol_channels) != n_phase:
                raise ValueError(
                    f"pol_channels length ({len(pol_channels)}) must match phases length ({n_phase})"
                )
            # Validate each channel
            for ch in pol_channels:
                if ch not in ['I', 'V', 'Q', 'U']:
                    raise ValueError(
                        f"Polarization channel must be 'I', 'V', 'Q', or 'U', got '{ch}'"
                    )

        self.phases = phases
        self.vel_shifts = vel_shifts
        self.pol_channels = pol_channels

    def generate_forward_model(self,
                               wl0_nm: float = 656.3,
                               verbose: int = 1) -> Dict[str, Any]:
        """
        Generate multi-phase synthetic spectra using VelspaceDiskIntegrator

        Generates ForwardModelResult objects for each phase based on configured 
        phases/vel_shifts/pol_channels.

        Parameters:
        -------
        wl0_nm : float
            Line center wavelength (nm)
        verbose : int
            Verbosity level (0=silent, 1=normal, 2=detailed)

        Returns:
        -------
        Dict[str, Any]
            Contains keys: 'results', 'phases', 'vel_shifts', 'pol_channels', 'metadata'

        Raises:
        -------
        ValueError
            If phases are not configured (call configure_multi_phase_synthesis)
        RuntimeError
            If line model cannot be loaded
        """
        if len(self.phases) == 0:
            raise ValueError(
                "Must call configure_multi_phase_synthesis() first")

        if verbose:
            print(
                f"[SpotSimulator] Generating synthetic models for {len(self.phases)} phases..."
            )

        results = []

        try:
            line_data = LineData('input/lines.txt')
            line_model = GaussianZeemanWeakLineModel(line_data)
        except Exception as e:
            raise RuntimeError(f"Could not load line model: {e}")

        # Generate results for each phase
        for idx, (phase, vel_shift, pol_channel) in enumerate(
                zip(self.phases, self.vel_shifts, self.pol_channels)):

            if verbose > 1:
                print(f"  [{idx+1}/{len(self.phases)}] phase={phase:.3f}, "
                      f"vel_shift={vel_shift:.2f} km/s, pol={pol_channel}")

            try:
                # Create geometry object for this phase
                geom = self.create_geometry_object(phase=phase)

                # Velocity grid
                v_grid = np.linspace(-100, 100, 401)

                # Create integrator
                integrator = VelspaceDiskIntegrator(geom=geom,
                                                    wl0_nm=wl0_nm,
                                                    v_grid=v_grid,
                                                    line_model=line_model,
                                                    inst_fwhm_kms=0.0,
                                                    normalize_continuum=True,
                                                    obs_phase=phase)

                # Get Stokes parameters
                stokes_i = getattr(integrator, 'I', np.ones_like(v_grid))
                stokes_v = getattr(integrator, 'V', np.zeros_like(v_grid))
                stokes_q = getattr(integrator, 'Q', np.zeros_like(v_grid))
                stokes_u = getattr(integrator, 'U', np.zeros_like(v_grid))

                # Apply velocity shift
                if vel_shift != 0.0:
                    v_shifted = v_grid - vel_shift

                    for arr in [stokes_i, stokes_v, stokes_q, stokes_u]:
                        f = interpolate.interp1d(v_grid,
                                                 arr,
                                                 kind='cubic',
                                                 bounds_error=False,
                                                 fill_value=np.nan)
                        arr[:] = np.nan_to_num(f(v_shifted), nan=0.0)

                    # Keep continuum of I at 1
                    stokes_i[np.isnan(stokes_i)] = 1.0

                # Create ForwardModelResult
                result = ForwardModelResult(stokes_i=stokes_i,
                                            stokes_v=stokes_v,
                                            stokes_q=stokes_q,
                                            stokes_u=stokes_u,
                                            wavelength=wl0_nm *
                                            (1.0 + v_grid / 299792.458),
                                            hjd=None,
                                            phase_index=idx,
                                            pol_channel=pol_channel,
                                            model_name="SpotSimulator")

                results.append(result)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Phase {phase} failed: {e}")
                raise

        if verbose:
            print(
                f"[SpotSimulator] Successfully generated {len(results)} synthetic models"
            )

        return {
            'results': results,
            'phases': self.phases,
            'vel_shifts': self.vel_shifts,
            'pol_channels': self.pol_channels,
            'metadata': {
                'simulator': 'SpotSimulator',
                'num_phases': len(self.phases),
                'grid_npix': self.grid.numPoints,
                'num_spots': len(self.spots)
            }
        }


# ============================================================================
# Noise Addition Functions
# ============================================================================


def add_noise_to_spectrum(spectrum,
                          noise_type='gaussian',
                          snr=None,
                          snr_linear=None,
                          sigma=None,
                          seed=None):
    """
    Add noise to spectrum

    Parameters:
    -------
    spectrum : np.ndarray
        Input spectrum
    noise_type : str
        Noise type: 'poisson' (Poisson noise), 'gaussian' (Gaussian noise), 'mixed' (Mixed)
    snr : float, optional
        Signal-to-noise ratio (dB). If provided, sigma is calculated automatically.
    snr_linear : float, optional
        Linear signal-to-noise ratio (Signal/Noise). E.g., 100 means S/N = 100.
        If provided, priority is higher than snr (dB).
    sigma : float, optional
        Gaussian noise standard deviation. If provided, highest priority.
    seed : int, optional
        Random number seed

    Returns:
    -------
    np.ndarray
        Noisy spectrum
    """
    if seed is not None:
        np.random.seed(seed)

    spectrum_noisy = spectrum.copy()

    if noise_type == 'poisson':
        # Poisson noise: assumes photon counts
        # Note: Poisson noise usually requires knowing the true electron/photon count,
        # here applying Poisson directly to normalized spectrum might not be physical
        # unless spectrum is already photon counts
        spectrum_noisy = np.random.poisson(spectrum)

    elif noise_type == 'gaussian':
        if sigma is None:
            signal_level = np.mean(np.abs(spectrum))
            # If spectrum is Stokes V/Q/U, mean might be close to 0, then max(abs) might be more appropriate
            # Or usually S/N is defined relative to continuum intensity (Ic=1)
            if signal_level < 0.1:  # Likely polarization spectrum
                signal_level = 1.0  # Assume noise defined relative to continuum

            if snr_linear is not None:
                sigma = float(signal_level / snr_linear)
            elif snr is not None:
                # SNR (dB) = 20 * log10(signal / noise)
                sigma = float(signal_level / (10**(snr / 20)))
            else:
                sigma = float(0.01 * signal_level)

        noise = np.random.normal(0, sigma, len(spectrum))
        spectrum_noisy = spectrum + noise

    elif noise_type == 'mixed':
        # Mixed: Poisson + Gaussian
        spectrum_poisson = np.random.poisson(spectrum)

        if sigma is None:
            signal_level = np.mean(np.abs(spectrum))
            if signal_level < 0.1:
                signal_level = 1.0

            if snr_linear is not None:
                sigma = float(signal_level / snr_linear * 0.5)
            elif snr is not None:
                sigma = float(signal_level / (10**(snr / 20)) * 0.5)
            else:
                sigma = float(0.005 * signal_level)

        noise = np.random.normal(0, sigma, len(spectrum))
        spectrum_noisy = spectrum_poisson + noise

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return spectrum_noisy


def add_noise_to_results(results,
                         noise_type='gaussian',
                         snr=None,
                         snr_linear=None,
                         sigma=None,
                         seed=None):
    """
    Add noise to multiple ForwardModelResult objects

    Parameters:
    -------
    results : List
        List of results (each is ForwardModelResult)
    noise_type : str
        Noise type
    snr : float, optional
        Signal-to-noise ratio (dB)
    snr_linear : float, optional
        Linear signal-to-noise ratio
    sigma : float, optional
        Standard deviation
    seed : int, optional
        Random number seed

    Returns:
    -------
    List
        List of noisy results
    """
    noisy_results = []

    for result in results:
        # Use snr_linear to calculate sigma for Stokes I
        # Use same sigma for Stokes V/Q/U (usually noise level is the same)

        # Calculate sigma (if not provided)
        current_sigma = sigma
        if current_sigma is None:
            # Assume continuum intensity is 1.0
            signal_ref = 1.0
            if snr_linear is not None:
                current_sigma = signal_ref / snr_linear
            elif snr is not None:
                current_sigma = signal_ref / (10**(snr / 20))
            else:
                current_sigma = 0.01  # Default S/N=100

        noisy_i = add_noise_to_spectrum(result.stokes_i,
                                        noise_type,
                                        sigma=current_sigma,
                                        seed=seed)

        # Polarization components use same sigma
        noisy_v = add_noise_to_spectrum(result.stokes_v,
                                        noise_type,
                                        sigma=current_sigma,
                                        seed=seed)

        noisy_q = None
        if result.stokes_q is not None:
            noisy_q = add_noise_to_spectrum(result.stokes_q,
                                            noise_type,
                                            sigma=current_sigma,
                                            seed=seed)

        noisy_u = None
        if result.stokes_u is not None:
            noisy_u = add_noise_to_spectrum(result.stokes_u,
                                            noise_type,
                                            sigma=current_sigma,
                                            seed=seed)

        noisy_result = ForwardModelResult(
            stokes_i=noisy_i,
            stokes_v=noisy_v,
            stokes_q=noisy_q,
            stokes_u=noisy_u,
            wavelength=result.wavelength,
            error=current_sigma,  # Record noise level
            hjd=result.hjd,
            phase_index=result.phase_index,
            pol_channel=result.pol_channel,
            model_name=result.model_name + "_noisy")

        noisy_results.append(noisy_result)

    return noisy_results


# ============================================================================
# Convenience Functions
# ============================================================================


def create_simple_spot_simulator(nr: int = 40,
                                 r_in: float = 0.5,
                                 r_out: float = 4.0,
                                 inclination_deg: float = 60.0,
                                 pOmega: float = -0.5,
                                 r0_rot: float = 1.0,
                                 period_days: float = 1.0) -> SpotSimulator:
    """
    Quickly create a SpotSimulator

    Parameters:
    -------
    nr : int
        Number of rings
    r_in, r_out : float
        Radial range
    inclination_deg : float
        Inclination (degrees)
    pOmega : float
        Differential rotation index
    r0_rot : float
        Reference radius
    period_days : float
        Period (days)

    Returns:
    -------
    SpotSimulator
        simulator object
    """
    grid = diskGrid(nr=nr, r_in=r_in, r_out=r_out)
    sim = SpotSimulator(grid,
                        inclination_rad=np.deg2rad(inclination_deg),
                        pOmega=pOmega,
                        r0_rot=r0_rot,
                        period_days=period_days)
    return sim


def create_test_spots() -> List[SpotConfig]:
    """Create test spot configurations"""
    spots = [
        SpotConfig(r=1.5, phi=0.0, amplitude=2.0, B_los=1000.0),
        SpotConfig(r=2.0, phi=np.pi, amplitude=1.5, B_los=-500.0),
        SpotConfig(r=3.0,
                   phi=np.pi / 2,
                   amplitude=-1.5,
                   spot_type='absorption'),
    ]
    return spots
