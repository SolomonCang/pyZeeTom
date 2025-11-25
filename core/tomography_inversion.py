"""
tomography_inversion.py - MEM Inversion Workflow Engine

This module implements the MEM (Maximum Entropy Method) inversion workflow, including:
  - Inversion iteration control
  - Convergence criteria
  - Intermediate result saving
  - Inversion result aggregation
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

from core.tomography_config import InversionConfig
from core.tomography_result import InversionResult, ForwardModelResult
from core.mem_tomography import MEMTomographyAdapter
from core.mem_iteration_manager import create_iteration_manager_from_config
from core.disk_geometry_integrator import VelspaceDiskIntegrator

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# Parameter Encoding/Decoding Utility Functions (for MEM Optimizer)
# ════════════════════════════════════════════════════════════════════════════


def _pack_parameters(B_los: np.ndarray,
                     B_perp: np.ndarray,
                     chi: np.ndarray,
                     brightness: Optional[np.ndarray] = None,
                     fit_brightness: bool = False,
                     fit_B_los: bool = True,
                     fit_B_perp: bool = True,
                     fit_chi: bool = True) -> np.ndarray:
    """
    Pack model parameters into a 1D vector.
    
    Order: [Brightness (optional), Blos (optional), Bperp (optional), chi (optional)]
    
    Parameters
    ----------
    B_los : np.ndarray
        Line-of-sight magnetic field (Npix,)
    B_perp : np.ndarray
        Perpendicular magnetic field strength (Npix,)
    chi : np.ndarray
        Magnetic field azimuth angle (Npix,)
    brightness : Optional[np.ndarray]
        Brightness distribution (Npix,), optional
    fit_brightness : bool
        Whether to pack brightness
    fit_B_los : bool
        Whether to pack B_los
    fit_B_perp : bool
        Whether to pack B_perp
    fit_chi : bool
        Whether to pack chi
    
    Returns
    -------
    np.ndarray
        Packed parameter vector
    """
    parts = []
    if fit_brightness and brightness is not None:
        parts.append(brightness)
    if fit_B_los:
        parts.append(B_los)
    if fit_B_perp:
        parts.append(B_perp)
    if fit_chi:
        parts.append(chi)
    return np.concatenate(parts) if parts else np.array([])


def _unpack_parameters(
    params: np.ndarray,
    npix: int,
    fit_brightness: bool = False,
    fit_B_los: bool = True,
    fit_B_perp: bool = True,
    fit_chi: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray]]:
    """
    Unpack model parameters from a 1D vector.
    
    Parameters
    ----------
    params : np.ndarray
        Packed parameter vector
    npix : int
        Number of pixels
    fit_brightness : bool
        Whether brightness parameter is included
    fit_B_los : bool
        Whether B_los is included
    fit_B_perp : bool
        Whether B_perp is included
    fit_chi : bool
        Whether chi is included
    
    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
        (brightness, B_los, B_perp, chi) - Returns None if not fitted
    """
    idx = 0
    brightness = None
    if fit_brightness:
        brightness = params[idx:idx + npix]
        idx += npix

    B_los = None
    if fit_B_los:
        B_los = params[idx:idx + npix]
        idx += npix

    B_perp = None
    if fit_B_perp:
        B_perp = params[idx:idx + npix]
        idx += npix

    chi = None
    if fit_chi:
        chi = params[idx:idx + npix]
        idx += npix

    return brightness, B_los, B_perp, chi


def _adaptive_delta(param_value: float, scale: float = 1e-3) -> float:
    """
    Calculate adaptive differentiation step size.
    
    Parameters
    ----------
    param_value : float
        Parameter value
    scale : float
        Relative step size factor (default 1e-3)
    
    Returns
    -------
    float
        Adaptive step size
    """
    abs_val = abs(float(param_value))
    delta = scale * max(abs_val, 0.1)
    return float(delta)


def _compute_response_matrix_analytical(integrator: Any,
                                        B_los: np.ndarray,
                                        B_perp: np.ndarray,
                                        chi: np.ndarray,
                                        brightness: Optional[np.ndarray],
                                        config: InversionConfig,
                                        base_spectrum_parts: List[np.ndarray],
                                        delta_scale: float = 1e-3,
                                        method: str = 'numerical',
                                        verbose: bool = False) -> np.ndarray:
    """
    Compute response matrix (supports numerical differentiation or analytical derivatives).
    
    Phase A: Use numerical differentiation (central difference) to compute partial derivatives of the model with respect to parameters.
    
    Parameters
    ----------
    integrator : VelspaceDiskIntegrator
        Velocity space integrator instance
    B_los, B_perp, chi : np.ndarray
        Current magnetic field parameters, shape (Npix,)
    brightness : Optional[np.ndarray]
        Current brightness parameters, shape (Npix,)
    config : InversionConfig
        Inversion configuration
    base_spectrum_parts : List[np.ndarray]
        Base spectrum components list [I, (V), (Q, U)]
    delta_scale : float
        Relative differentiation step size factor (default 1e-3)
    method : str
        Calculation method ('numerical' or 'analytical')
    verbose : bool
        Detailed output
    
    Returns
    -------
    np.ndarray
        Response matrix, shape (Ndata, Nparams)
    """
    npix = len(B_los)

    # Build baseline full vector to determine Ndata
    base_full = np.concatenate(base_spectrum_parts)
    ndata = len(base_full)

    # Determine total number of parameters
    n_params = 0
    fit_brightness = config.fit_brightness and brightness is not None
    if fit_brightness:
        n_params += npix
    if config.fit_B_los:
        n_params += npix
    if config.fit_B_perp:
        n_params += npix
    if config.fit_chi:
        n_params += npix

    if verbose:
        logger.info(
            f"Computing response matrix ({method}): Npix={npix}, Ndata={ndata}, Nparams={n_params}"
        )

    # Initialize response matrix
    Resp = np.zeros((ndata, n_params), dtype=float)

    if method == 'analytical':
        if verbose:
            logger.info("  Using analytical derivatives...")

        # Compute all derivatives at once
        derivs = integrator.compute_derivatives(B_los=B_los,
                                                B_perp=B_perp,
                                                chi=chi,
                                                amp=brightness)

        # Determine row offsets
        # Note: integrator.I should be available from previous compute_spectrum call
        # or we can use the shape of one of the derivatives
        n_lam = derivs['dI_damp'].shape[0]

        current_row = 0

        # I block
        row_I_start = current_row
        current_row += n_lam

        # V block
        row_V_start = -1
        if config.enable_v:
            row_V_start = current_row
            current_row += n_lam

        # Q/U blocks
        row_Q_start = -1
        row_U_start = -1
        if config.enable_qu:
            row_Q_start = current_row
            current_row += n_lam
            row_U_start = current_row
            current_row += n_lam

        # Fill Response Matrix
        col_offset = 0

        # 1. Brightness
        if fit_brightness:
            # I rows
            Resp[row_I_start:row_I_start + n_lam,
                 col_offset:col_offset + npix] = derivs['dI_damp']
            # V rows
            if config.enable_v:
                Resp[row_V_start:row_V_start + n_lam,
                     col_offset:col_offset + npix] = derivs['dV_damp']
            # Q/U rows
            if config.enable_qu:
                Resp[row_Q_start:row_Q_start + n_lam,
                     col_offset:col_offset + npix] = derivs['dQ_damp']
                Resp[row_U_start:row_U_start + n_lam,
                     col_offset:col_offset + npix] = derivs['dU_damp']
            col_offset += npix

        # 2. B_los
        if config.fit_B_los:
            # V rows
            if config.enable_v:
                Resp[row_V_start:row_V_start + n_lam,
                     col_offset:col_offset + npix] = derivs['dV_dBlos']
            col_offset += npix

        # 3. B_perp
        if config.fit_B_perp:
            # Q/U rows
            if config.enable_qu:
                Resp[row_Q_start:row_Q_start + n_lam,
                     col_offset:col_offset + npix] = derivs['dQ_dBperp']
                Resp[row_U_start:row_U_start + n_lam,
                     col_offset:col_offset + npix] = derivs['dU_dBperp']
            col_offset += npix

        # 4. chi
        if config.fit_chi:
            # Q/U rows
            if config.enable_qu:
                Resp[row_Q_start:row_Q_start + n_lam,
                     col_offset:col_offset + npix] = derivs['dQ_dchi']
                Resp[row_U_start:row_U_start + n_lam,
                     col_offset:col_offset + npix] = derivs['dU_dchi']
            col_offset += npix

        if verbose:
            logger.info(
                f"Response matrix computation completed (analytical): {Resp.shape}"
            )
            logger.info(
                f"  Derivative range: [{np.min(Resp):.3e}, {np.max(Resp):.3e}]"
            )
            logger.info(f"  Derivative RMS: {np.sqrt(np.mean(Resp**2)):.3e}")

        return Resp

    # Helper function: Build full vector
    def _build_full_vector(intg):
        parts = [intg.I]
        if config.enable_v:
            parts.append(intg.V)
        if config.enable_qu:
            parts.append(intg.Q)
            parts.append(intg.U)
        return np.concatenate(parts)

    col_offset = 0

    # ════════════════════════════════════════════════════════════════════════════
    # Step 0: Numerical differentiation - Brightness derivatives
    # ════════════════════════════════════════════════════════════════════════════

    if fit_brightness:
        if verbose:
            logger.info("  Starting Brightness derivatives...")
        for ipix in range(npix):
            delta = _adaptive_delta(brightness[ipix], scale=delta_scale)

            # +Δ direction
            bright_plus = brightness.copy()
            bright_plus[ipix] += delta
            try:
                integrator.compute_spectrum(amp=bright_plus)
                spec_plus = _build_full_vector(integrator)
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Brightness +Δ computation failed (pixel {ipix}): {e}"
                    )
                spec_plus = base_full

            # -Δ direction
            bright_minus = brightness.copy()
            bright_minus[ipix] -= delta
            try:
                integrator.compute_spectrum(amp=bright_minus)
                spec_minus = _build_full_vector(integrator)
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Brightness -Δ computation failed (pixel {ipix}): {e}"
                    )
                spec_minus = base_full

            deriv = (spec_plus - spec_minus) / (2.0 * delta + 1e-30)
            Resp[:, col_offset + ipix] = deriv

            if verbose and (ipix + 1) % max(1, npix // 10) == 0:
                logger.info(
                    f"  Brightness derivative: {ipix + 1}/{npix} ({(ipix + 1) / npix * 100:.0f}%) completed"
                )

        # Restore integrator to baseline state
        integrator.compute_spectrum(amp=brightness)
        col_offset += npix

    # ════════════════════════════════════════════════════════════════════════════
    # Step 1: Numerical differentiation - B_los derivatives
    # ════════════════════════════════════════════════════════════════════════════

    if config.fit_B_los:
        if verbose:
            logger.info("  Starting B_los derivatives...")
        for ipix in range(npix):
            delta = _adaptive_delta(B_los[ipix], scale=delta_scale)

            # +Δ direction
            B_los_plus = B_los.copy()
            B_los_plus[ipix] += delta
            try:
                integrator.compute_spectrum(B_los=B_los_plus)
                spec_plus = _build_full_vector(integrator)
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"B_los +Δ computation failed (pixel {ipix}): {e}")
                spec_plus = base_full

            # -Δ direction
            B_los_minus = B_los.copy()
            B_los_minus[ipix] -= delta
            try:
                integrator.compute_spectrum(B_los=B_los_minus)
                spec_minus = _build_full_vector(integrator)
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"B_los -Δ computation failed (pixel {ipix}): {e}")
                spec_minus = base_full

            # Central difference
            deriv = (spec_plus - spec_minus) / (2.0 * delta + 1e-30)
            Resp[:, col_offset + ipix] = deriv

            if verbose and (ipix + 1) % max(1, npix // 10) == 0:
                logger.info(
                    f"  B_los derivative: {ipix + 1}/{npix} ({(ipix + 1) / npix * 100:.0f}%) completed"
                )

        # Restore integrator to baseline state
        integrator.compute_spectrum(B_los=B_los)
        col_offset += npix

    # ════════════════════════════════════════════════════════════════════════════
    # Step 2: Numerical differentiation - B_perp derivatives
    # ════════════════════════════════════════════════════════════════════════════

    if config.fit_B_perp:
        if verbose:
            logger.info("  Starting B_perp derivatives...")
        for ipix in range(npix):
            delta = _adaptive_delta(B_perp[ipix], scale=delta_scale)

            # +Δ direction
            B_perp_plus = B_perp.copy()
            B_perp_plus[ipix] += delta
            try:
                integrator.compute_spectrum(B_perp=B_perp_plus)
                spec_plus = _build_full_vector(integrator)
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"B_perp +Δ computation failed (pixel {ipix}): {e}")
                spec_plus = base_full

            # -Δ direction
            B_perp_minus = B_perp.copy()
            B_perp_minus[ipix] -= delta
            try:
                integrator.compute_spectrum(B_perp=B_perp_minus)
                spec_minus = _build_full_vector(integrator)
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"B_perp -Δ computation failed (pixel {ipix}): {e}")
                spec_minus = base_full

            deriv = (spec_plus - spec_minus) / (2.0 * delta + 1e-30)
            Resp[:, col_offset + ipix] = deriv

            if verbose and (ipix + 1) % max(1, npix // 10) == 0:
                logger.info(
                    f"  B_perp derivative: {ipix + 1}/{npix} ({(ipix + 1) / npix * 100:.0f}%) completed"
                )

        # Restore integrator to baseline state
        integrator.compute_spectrum(B_perp=B_perp)
        col_offset += npix

    # ════════════════════════════════════════════════════════════════════════════
    # Step 3: Numerical differentiation - chi derivatives
    # ════════════════════════════════════════════════════════════════════════════

    if config.fit_chi:
        if verbose:
            logger.info("  Starting chi derivatives...")
        for ipix in range(npix):
            delta = _adaptive_delta(chi[ipix], scale=delta_scale)

            # +Δ direction
            chi_plus = chi.copy()
            chi_plus[ipix] += delta
            try:
                integrator.compute_spectrum(chi=chi_plus)
                spec_plus = _build_full_vector(integrator)
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"chi +Δ computation failed (pixel {ipix}): {e}")
                spec_plus = base_full

            # -Δ direction
            chi_minus = chi.copy()
            chi_minus[ipix] -= delta
            try:
                integrator.compute_spectrum(chi=chi_minus)
                spec_minus = _build_full_vector(integrator)
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"chi -Δ computation failed (pixel {ipix}): {e}")
                spec_minus = base_full

            deriv = (spec_plus - spec_minus) / (2.0 * delta + 1e-30)
            Resp[:, col_offset + ipix] = deriv

            if verbose and (ipix + 1) % max(1, npix // 10) == 0:
                logger.info(
                    f"  chi derivative: {ipix + 1}/{npix} ({(ipix + 1) / npix * 100:.0f}%) completed"
                )

        # Restore integrator to baseline state
        integrator.compute_spectrum(chi=chi)

    if verbose:
        logger.info(f"Response matrix computation completed: {Resp.shape}")
        logger.info(
            f"  Derivative range: [{np.min(Resp):.3e}, {np.max(Resp):.3e}]")
        logger.info(f"  Derivative RMS: {np.sqrt(np.mean(Resp**2)):.3e}")

    return Resp


def run_mem_inversion(config: InversionConfig,
                      forward_results: Optional[
                          List[ForwardModelResult]] = None,
                      verbose: int = 0) -> List[InversionResult]:
    """Run the complete MEM inversion workflow (simultaneous fitting of all phases)
    
    This is the main entry point of the inversion workflow, performing the following steps:
    1. Validate the completeness and consistency of the configuration
    2. Initialize inversion state and history
    3. Perform iterative inversion (simultaneous fitting of all phases)
    4. Check convergence criteria
    5. Return the final inversion result (including Master Map)
    
    Parameters
    ----------
    config : InversionConfig
        Inversion configuration object, containing all necessary parameters
    forward_results : Optional[List[ForwardModelResult]]
        Forward results for initialization or reference
    verbose : int, optional
        Verbosity level (0=minimal, 1=normal, 2=debug), default is 0
    
    Returns
    -------
    List[InversionResult]
        A list containing one InversionResult object, representing the global fit
    
    Raises
    ------
    ValueError
        Configuration object validation failed
    RuntimeError
        Error occurred during inversion
    """

    # Validate configuration
    try:
        config.validate()
    except (ValueError, AssertionError) as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    if verbose >= 2:
        logger.info(config.create_summary())

    # Perform simultaneous inversion
    try:
        if verbose >= 1:
            logger.info(
                f"Starting simultaneous inversion for {len(config.obsSet)} phases..."
            )

        result = _invert_simultaneous(
            config=config,
            verbose=verbose,
        )

        if verbose >= 0:
            logger.info(f"✓ Inversion completed: "
                        f"χ²={result.final_chi2:.6e}, "
                        f"Converged={result.converged}")

        return [result]

    except Exception as e:
        logger.error(f"Inversion failed: {e}")
        raise


def _invert_simultaneous(config: InversionConfig,
                         verbose: int = 0) -> InversionResult:
    """Simultaneous MEM inversion for all observed phases
    
    Parameters
    ----------
    config : InversionConfig
        Inversion configuration object
    verbose : int, optional
        Verbosity level
    
    Returns
    -------
    InversionResult
        Global inversion result
    """

    # Initialize magnetic field model (Master Map at Phase 0)
    if config.initial_B_los is not None:
        B_los = config.initial_B_los.copy()
    else:
        B_los = config.B_los_init.copy()

    if config.initial_B_perp is not None:
        B_perp = config.initial_B_perp.copy()
    else:
        B_perp = config.B_perp_init.copy()

    if config.initial_chi is not None:
        chi = config.initial_chi.copy()
    else:
        chi = config.chi_init.copy()

    # Initialize brightness model
    if config.initial_brightness is not None:
        brightness = config.initial_brightness.copy()
    elif config.amp_init is not None:
        brightness = config.amp_init.copy()
    else:
        brightness = np.ones(len(B_los))

    # ----------------------------------------------------------------
    # Zero-field initialization handling
    # ----------------------------------------------------------------
    # If the initial magnetic field is all zeros, and the field is to be fitted,
    # add a small random perturbation. This is because at B=0, the derivatives
    # of Stokes Q/U with respect to B_perp and chi are usually zero,
    # causing the MEM optimizer to stall (zero gradient).

    is_zero_field = (np.all(B_los == 0) and np.all(B_perp == 0))
    if is_zero_field and (config.fit_B_los or config.fit_B_perp):
        if verbose >= 1:
            logger.info(
                "  [Init] Detected zero magnetic field initialization, adding small random perturbations to break symmetry..."
            )

        # Use a fixed seed for reproducibility
        rng = np.random.RandomState(42)

        if config.fit_B_los:
            # Add ~10 Gauss noise
            noise_blos = rng.normal(0, 10.0, size=len(B_los))
            B_los += noise_blos

        if config.fit_B_perp:
            # Add ~10 Gauss noise (take absolute value to ensure non-negativity)
            noise_bperp = np.abs(rng.normal(0, 10.0, size=len(B_perp)))
            B_perp += noise_bperp

        if config.fit_chi:
            # Random azimuthal angles
            chi += rng.uniform(0, 2 * np.pi, size=len(chi))

    # ════════════════════════════════════════════════════════════════════════════
    # Initialize integrator list (one for each phase)
    # ════════════════════════════════════════════════════════════════════════════

    integrators = []
    obs_data_list = []
    obs_error_list = []

    c_kms = 2.99792458e5

    for i, obs_data in enumerate(config.obsSet):
        if verbose >= 2:
            logger.info(
                f"  Initializing integrator {i + 1}/{len(config.obsSet)} (Phase {getattr(obs_data, 'phase', 0.0):.3f})..."
            )

        # Validate observation data
        if not hasattr(obs_data, 'specI') or obs_data.specI is None:
            raise ValueError("Observation data is missing specI")

        # Get error information (preferably use specIsig)
        sigma_I = getattr(obs_data, 'specIsig', None)
        if sigma_I is None:
            sigma_I = getattr(obs_data, 'sigma', None)

        if sigma_I is None:
            raise ValueError(
                "Observation data is missing error information (specIsig or sigma)"
            )

        # Compute velocity grid
        # Check if observation is already in velocity domain (LSD)
        is_velocity_domain = False
        if hasattr(obs_data, 'profile_type') and (
                'lsd' in obs_data.profile_type.lower()
                or 'velocity' in obs_data.profile_type.lower()):
            is_velocity_domain = True

        # Also check value range: if values are small (e.g. -1000 to 1000) and contain negative values, likely velocity
        wl_vals = np.asarray(obs_data.wl, dtype=float)
        if np.max(np.abs(wl_vals)) < 3000 and np.min(wl_vals) < 0:
            is_velocity_domain = True

        if is_velocity_domain:
            v_grid = wl_vals
        else:
            v_grid = (obs_data.wl -
                      config.lineData.wl0) / config.lineData.wl0 * c_kms

        # Create integrator instance (note: share config.geom, but each integrator has its own time_phase)
        # We need to ensure config.geom is correctly used in compute_spectrum
        # Since VelspaceDiskIntegrator stores self.geom, we can update config.geom externally
        # All integrators will see the updated B_los etc. arrays (if passed by reference)
        # But for safety, we explicitly pass parameters in compute_spectrum

        integrator = VelspaceDiskIntegrator(
            geom=config.geom,
            wl0_nm=config.lineData.wl0,
            v_grid=v_grid,
            line_model=config.line_model,
            line_area=config.line_area,
            inst_fwhm_kms=config.inst_fwhm_kms,
            normalize_continuum=config.normalize_continuum,
            disk_v0_kms=config.velEq,
            disk_power_index=config.pOmega,
            disk_r0=config.radius,
            obs_phase=getattr(obs_data, 'phase', 0.0),
            time_phase=getattr(obs_data, 'phase',
                               0.0)  # Enable time evolution/rotation
        )
        integrators.append(integrator)

        # Prepare data vector
        obs_parts = [obs_data.specI]
        sig_parts = [sigma_I]

        if config.enable_v:
            specV = getattr(obs_data, 'specV', np.zeros_like(obs_data.specI))
            obs_parts.append(specV)
            # Use specVsig (if exists), otherwise use sigma_I
            sigma_V = getattr(obs_data, 'specVsig', sigma_I)
            sig_parts.append(sigma_V)

        if config.enable_qu:
            specQ = getattr(obs_data, 'specQ', np.zeros_like(obs_data.specI))
            specU = getattr(obs_data, 'specU', np.zeros_like(obs_data.specI))
            obs_parts.append(specQ)
            obs_parts.append(specU)
            # Q/U errors are usually similar to V, or use specIsig
            # Here we assume same as V, or use sigma_I
            sigma_pol = getattr(obs_data, 'specVsig', sigma_I)
            sig_parts.append(sigma_pol)
            sig_parts.append(sigma_pol)

        obs_data_list.append(np.concatenate(obs_parts))
        obs_error_list.append(np.concatenate(sig_parts))

    # Concatenate global data vector
    Data_total = np.concatenate(obs_data_list)
    Sigma_total = np.concatenate(obs_error_list)

    if verbose >= 2:
        logger.info(
            f"Global data preparation: Ndata_total={len(Data_total)} (Phases={len(integrators)})"
        )

    # ════════════════════════════════════════════════════════════════════════════
    # MEM Adapter Initialization
    # ════════════════════════════════════════════════════════════════════════════

    npix = len(B_los)
    grid_area = config.geom.grid.area if hasattr(config.geom.grid,
                                                 'area') else np.ones(npix)

    # ----------------------------------------------------------------
    # Compute total visibility and enhance entropy weights for invisible pixels
    # ----------------------------------------------------------------
    # For pixels that are never observed (total visibility very low),
    # the data constraint is zero.
    # To prevent these pixels from drifting randomly in the inversion or staying
    # at wrong initial values, we significantly increase their entropy weights,
    # forcing them to quickly relax to the default model (Default Model).

    total_visibility = np.zeros(npix)
    for integrator in integrators:
        if hasattr(integrator, 'Ic_weight'):
            total_visibility += integrator.Ic_weight

    # Threshold: 1% of mean visibility
    mean_vis = np.mean(total_visibility)

    if config.force_all_pixels_visible:
        if verbose >= 1:
            logger.info(
                "  [Regularization] Forcing all pixels visible: disabling entropy enhancement for invisible pixels."
            )
        invisible_mask = np.zeros(npix, dtype=bool)
    elif mean_vis > 1e-9:
        invisible_mask = total_visibility < (0.01 * mean_vis)
    else:
        invisible_mask = np.zeros(npix, dtype=bool)

    entropy_weights_bright = grid_area.copy()
    entropy_weights_blos = grid_area.copy()
    entropy_weights_bperp = grid_area.copy()
    entropy_weights_chi = grid_area.copy() * 0.1

    if np.any(invisible_mask):
        # Boost weights: use 100 times the max grid area as reference
        boost_val = np.max(grid_area) * 100.0

        entropy_weights_bright[invisible_mask] = boost_val
        entropy_weights_blos[invisible_mask] = boost_val
        entropy_weights_bperp[invisible_mask] = boost_val
        entropy_weights_chi[invisible_mask] = boost_val * 0.1

        if verbose >= 1:
            logger.info(
                f"  [Regularization] Detected {np.sum(invisible_mask)} invisible pixels, enhanced their entropy weights to force regression to default model."
            )

    # Create MEM adapter instance (for Master Map)
    # Note: fit_magnetic must be True to enable fine-grained control like fit_B_los
    # If False, MEMTomographyAdapter will forcibly disable all magnetic field fitting
    adapter = MEMTomographyAdapter(
        fit_brightness=config.fit_brightness,
        fit_magnetic=
        True,  # Enable magnetic fitting framework, controlled by fit_B_los etc.
        fit_B_los=config.fit_B_los,
        fit_B_perp=config.fit_B_perp,
        fit_chi=config.fit_chi,
        entropy_weights_bright=entropy_weights_bright,
        entropy_weights_blos=entropy_weights_blos,
        entropy_weights_bperp=entropy_weights_bperp,
        entropy_weights_chi=entropy_weights_chi,
        default_bright=1.0,
        default_blos=config.B_los_default
        if hasattr(config, 'B_los_default') else 0.1,
        default_bperp=config.B_perp_default
        if hasattr(config, 'B_perp_default') else 0.1,
        default_chi=0.0)

    entropy_params = {
        'npix': npix,
        'n_bright': npix if config.fit_brightness else 0,
        'n_blos': npix if config.fit_B_los else 0,
        'n_bperp': npix if config.fit_B_perp else 0,
        'n_chi': npix if config.fit_chi else 0
    }

    # ════════════════════════════════════════════════════════════════════════════
    # MEM Adapter Initialization

    # ----------------------------------------------------------------
    # Iterative Inversion
    # ----------------------------------------------------------------

    max_iters = config.max_iterations or config.num_iterations
    manager = create_iteration_manager_from_config(
        {
            'max_iterations': max_iters,
            'convergence_rel_tol': config.convergence_threshold,
            'stall_threshold': 3,
            'with_iteration_history': True,
            'with_progress_monitor': True,
        },
        verbose=0)  # Silence internal monitor to avoid duplicate output

    while True:
        should_stop, reason = manager.should_stop()
        if should_stop:
            break

        manager.start_iteration()
        iteration = manager.iteration

        try:
            chi2, entropy, regularization = _perform_simultaneous_step(
                B_los=B_los,
                B_perp=B_perp,
                chi=chi,
                brightness=brightness,
                obs_data_total=Data_total,
                obs_error_total=Sigma_total,
                config=config,
                iteration=iteration,
                adapter=adapter,
                entropy_params=entropy_params,
                integrators=integrators,
                verbose=verbose,
            )
        except Exception as e:
            logger.warning(f"Iteration {iteration} failed: {e}")
            import traceback
            traceback.print_exc()
            break

        # Normalize Chi2 (Reduced Chi2)
        ndata = len(Data_total)
        chi2_reduced = chi2 / ndata if ndata > 0 else chi2

        manager.record_iteration(
            chi2=chi2_reduced,
            entropy=entropy,
            gradient_norm=None,
            grad_S_norm=0.0,
            grad_C_norm=0.0,
        )

        if verbose >= 0:
            eta_msg = ""
            if manager.progress_monitor:
                eta_msg = f", ETA: {manager.progress_monitor._estimate_eta()}"
            logger.info(
                f"  Iter {iteration}: χ²_red={chi2_reduced:.6e}, S={entropy:.6e}{eta_msg}"
            )

        should_stop, reason = manager.should_stop(chi2_reduced)
        if should_stop:
            if verbose >= 0:
                logger.info(f"  Converged at iteration {iteration}: {reason}")
            break

    # ════════════════════════════════════════════════════════════════════════════
    # Result Packaging
    # ════════════════════════════════════════════════════════════════════════════

    if manager.iteration_history is not None:
        hist_data = manager.iteration_history.get_history()
        chi2_history = hist_data.get('chi2', [])
        entropy_history = hist_data.get('entropy', [])
        regularization_history = hist_data.get('regularization', [])
    else:
        chi2_history = []
        entropy_history = []
        regularization_history = []

    summary = manager.get_summary()
    stop_reason = summary.get('stop_reason', 'unknown')
    converged = (stop_reason == 'convergence'
                 or stop_reason == 'convergence_stall')

    result = InversionResult(
        B_los_final=B_los,
        B_perp_final=B_perp,
        chi_final=chi,
        brightness_final=brightness,
        iterations_completed=manager.iteration,
        chi2_history=chi2_history,
        entropy_history=entropy_history,
        regularization_history=regularization_history,
        converged=converged,
        convergence_reason=stop_reason,
        final_chi2=chi2_history[-1] if chi2_history else 0.0,
        final_entropy=entropy_history[-1] if entropy_history else 0.0,
        phase_index=-1,  # Indicates global result
        pol_channels=["I+V"],
    )

    # Compute fit quality (using the first phase as a representative, or can be extended to all phases' average)
    # Here we simply compute the fit quality for the first phase
    if integrators:
        obs0 = config.obsSet[0]
        sigma0 = getattr(obs0, 'specIsig', getattr(obs0, 'sigma', None))

        result.fit_quality = _compute_fit_quality(
            B_los=B_los,
            B_perp=B_perp,
            chi=chi,
            brightness=brightness,
            obs_spectrum=obs0.specI,
            obs_error=sigma0,
            integrator=integrators[0],
        )

    return result


def _perform_simultaneous_step(B_los: np.ndarray,
                               B_perp: np.ndarray,
                               chi: np.ndarray,
                               brightness: np.ndarray,
                               obs_data_total: np.ndarray,
                               obs_error_total: np.ndarray,
                               config: InversionConfig,
                               iteration: int,
                               adapter: MEMTomographyAdapter,
                               entropy_params: Dict[str, Any],
                               integrators: List[Any],
                               verbose: int = 0) -> Tuple[float, float, float]:
    """Perform one step of simultaneous MEM inversion
    
    Parameters
    ----------
    B_los, B_perp, chi : np.ndarray
        Current Master Map parameters
    brightness : np.ndarray
        Current brightness parameters
    obs_data_total : np.ndarray
        Global observation data vector
    obs_error_total : np.ndarray
        Global observation error vector
    config : InversionConfig
        Configuration
    iteration : int
        Iteration number
    adapter : MEMTomographyAdapter
        Adapter
    entropy_params : Dict
        Entropy parameters
    integrators : List[VelspaceDiskIntegrator]
        List of integrators
    verbose : int
        Verbosity level
    
    Returns
    -------
    Tuple[float, float, float]
        (chi2, entropy, regularization)
    """

    # ════════════════════════════════════════════════════════════════════════════
    # Step 0: Prepare synthetic data and response matrix (iterate over all phases)
    # ════════════════════════════════════════════════════════════════════════════

    syn_spec_list = []
    resp_matrix_list = []

    if verbose >= 2:
        logger.info(
            f"  [Iter {iteration}] Computing response matrices for {len(integrators)} phases..."
        )

    # Update geometry parameters for all integrators (Master Map)
    # Note: compute_spectrum will use the passed parameters for computation, and update integrator.geom
    # Since all integrator may share the same geom object (depending on initialization), or have their own copies
    # We explicitly pass parameters to compute_spectrum to ensure correctness

    for i, integrator in enumerate(integrators):
        if verbose >= 2:
            logger.info(
                f"    Phase {i + 1}/{len(integrators)}: Computing spectrum & response..."
            )

        # 1. Compute synthetic spectrum for this phase
        integrator.compute_spectrum(B_los=B_los,
                                    B_perp=B_perp,
                                    chi=chi,
                                    amp=brightness)

        syn_parts = [integrator.I]
        if config.enable_v:
            syn_parts.append(integrator.V)
        if config.enable_qu:
            syn_parts.append(integrator.Q)
            syn_parts.append(integrator.U)

        syn_spec_phase = np.concatenate(syn_parts)
        syn_spec_list.append(syn_spec_phase)

        # 2. Compute response matrix for this phase (with respect to Master Map)
        # Since integrator is already configured with time_phase, its internal grid is already rotated
        # Perturbations to Master Map parameters will produce correct spectrum variations through the rotated grid
        # Therefore _compute_response_matrix_analytical is directly applicable

        # Performance optimization: compute response matrix only when necessary (e.g. every few iterations, or using BFGS approximation)
        # Currently computed every time (Phase A)

        resp_phase = _compute_response_matrix_analytical(
            integrator=integrator,
            B_los=B_los,
            B_perp=B_perp,
            chi=chi,
            brightness=brightness,
            config=config,
            base_spectrum_parts=syn_parts,
            method='analytical',
            verbose=(verbose >= 2)  # Reduce logging
        )
        resp_matrix_list.append(resp_phase)

        if verbose >= 2 and iteration == 1 and i % 5 == 0:
            logger.debug(f"  Phase {i} response computed.")

    # Concatenate global synthetic spectrum and response matrix
    synthetic_spectrum_total = np.concatenate(syn_spec_list)
    response_matrix_total = np.vstack(resp_matrix_list)

    # ════════════════════════════════════════════════════════════════════════════
    # Step 1: Pack parameter vector (Master Map)
    # ════════════════════════════════════════════════════════════════════════════

    Image = _pack_parameters(B_los,
                             B_perp,
                             chi,
                             brightness,
                             fit_brightness=config.fit_brightness,
                             fit_B_los=config.fit_B_los,
                             fit_B_perp=config.fit_B_perp,
                             fit_chi=config.fit_chi)

    # ════════════════════════════════════════════════════════════════════════════
    # Step 2: Call MEM iterator
    # ════════════════════════════════════════════════════════════════════════════

    if verbose >= 2:
        logger.info(f"  [Iter {iteration}] Optimizing (MEM step)...")

    # Determine optimization target
    target_aim = None
    fix_entropy = 0

    if hasattr(config, 'target_form'):
        if config.target_form == 'C':
            # Target is Chi^2 (assuming target_value is reduced Chi^2)
            ndata = len(obs_data_total)
            target_aim = config.target_value * ndata
            fix_entropy = 0
        elif config.target_form == 'E':
            # Target is entropy (assuming target_value is normalized entropy per pixel)
            npix = len(B_los)
            target_aim = config.target_value * npix
            fix_entropy = 1

    try:
        entropy, chi2, test, Image_new = adapter.optimizer.iterate(
            Image=Image,
            Fmodel=synthetic_spectrum_total,
            Data=obs_data_total,
            sig2=obs_error_total**2,
            Resp=response_matrix_total,
            weights=np.ones_like(Image),
            entropy_params=entropy_params,
            fixEntropy=fix_entropy,
            targetAim=target_aim)

    except Exception as e:
        if verbose:
            logger.warning(f"MEM iteration step {iteration} failed: {e}")
        entropy = _compute_entropy(B_los, B_perp, brightness, config)
        chi2 = _compute_chi2(obs_data_total, obs_error_total,
                             synthetic_spectrum_total)
        regularization = _compute_regularization(B_los, B_perp, chi,
                                                 brightness, config)
        return chi2, entropy, regularization

    # ════════════════════════════════════════════════════════════════════════════
    # Step 3: Update parameters
    # ════════════════════════════════════════════════════════════════════════════

    npix = len(B_los)
    brightness_new, B_los_new, B_perp_new, chi_new = _unpack_parameters(
        Image_new,
        npix,
        fit_brightness=config.fit_brightness,
        fit_B_los=config.fit_B_los,
        fit_B_perp=config.fit_B_perp,
        fit_chi=config.fit_chi)

    if B_los_new is not None:
        B_los[:] = B_los_new
    if B_perp_new is not None:
        B_perp[:] = B_perp_new
    if chi_new is not None:
        chi[:] = chi_new
    if brightness_new is not None:
        brightness[:] = brightness_new

    # ════════════════════════════════════════════════════════════════════════════
    # Step 4: Compute regularization term
    # ════════════════════════════════════════════════════════════════════════════

    regularization = _compute_regularization(B_los, B_perp, chi, brightness,
                                             config)

    return chi2, entropy, regularization


def _check_convergence(chi2_history: List[float], entropy_history: List[float],
                       config: InversionConfig, iteration: int) -> bool:
    """Check if convergence criteria are met
    
    Parameters
    ----------
    chi2_history : List[float]
        χ² history
    entropy_history : List[float]
        Entropy history
    config : InversionConfig
        Inversion configuration
    iteration : int
        Current iteration number
    
    Returns
    -------
    bool
        Whether converged
    """

    if len(chi2_history) < 2:
        return False

    # Check χ² change
    chi2_change = abs(chi2_history[-1] -
                      chi2_history[-2]) / (abs(chi2_history[-1]) + 1e-10)

    if chi2_change < config.convergence_threshold:
        return True

    # Check if maximum iteration number is reached
    max_iters = config.max_iterations or config.num_iterations
    if iteration >= max_iters - 1:
        return True

    return False


def _is_converged(chi2_history: List[float], config: InversionConfig) -> bool:
    """Determine if converged
    
    Parameters
    ----------
    chi2_history : List[float]
        χ² history
    config : InversionConfig
        Inversion configuration
    
    Returns
    -------
    bool
        Whether converged
    """

    if len(chi2_history) < 2:
        return False

    chi2_change = abs(chi2_history[-1] -
                      chi2_history[-2]) / (abs(chi2_history[-1]) + 1e-10)

    return chi2_change < config.convergence_threshold


def _get_convergence_reason(chi2_history: List[float],
                            config: InversionConfig) -> str:
    """Get description of convergence reason
    
    Parameters
    ----------
    chi2_history : List[float]
        χ² history
    config : InversionConfig
        Inversion configuration
    
    Returns
    -------
    str
        Convergence reason description
    """

    if not chi2_history:
        return "Not started"

    max_iters = config.max_iterations or config.num_iterations

    if len(chi2_history) >= max_iters:
        return "Reached maximum iteration number"

    if _is_converged(chi2_history, config):
        return "χ² change below threshold"

    return "Unknown"


def _compute_chi2(obs_spectrum: np.ndarray, obs_error: np.ndarray,
                  predicted_spectrum: np.ndarray) -> float:
    """Compute χ² value
    
    Parameters
    ----------
    obs_spectrum : np.ndarray
        Observed spectrum
    obs_error : np.ndarray
        Observation error
    predicted_spectrum : np.ndarray
        Predicted spectrum
    
    Returns
    -------
    float
        χ² value
    """

    residuals = (obs_spectrum - predicted_spectrum) / obs_error
    chi2 = float(np.sum(residuals**2))

    return chi2


def _compute_entropy(B_los: np.ndarray, B_perp: np.ndarray,
                     brightness: Optional[np.ndarray],
                     config: InversionConfig) -> float:
    """Compute entropy value
    
    Parameters
    ----------
    B_los : np.ndarray
        Line-of-sight magnetic field
    B_perp : np.ndarray
        Perpendicular magnetic field
    brightness : Optional[np.ndarray]
        Brightness distribution
    config : InversionConfig
        Inversion configuration
    
    Returns
    -------
    float
        Entropy value
    """

    # Simple entropy calculation: sum of squares of magnetic field strengths
    entropy = float(np.sum(B_los**2 + B_perp**2))

    if config.fit_brightness and brightness is not None:
        entropy += float(np.sum(brightness**2))

    return entropy


def _compute_regularization(B_los: np.ndarray, B_perp: np.ndarray,
                            chi: np.ndarray, brightness: Optional[np.ndarray],
                            config: InversionConfig) -> float:
    """Compute regularization term
    
    Parameters
    ----------
    B_los : np.ndarray
        Line-of-sight magnetic field
    B_perp : np.ndarray
        Perpendicular magnetic field
    chi : np.ndarray
        Magnetic field azimuth angle
    brightness : Optional[np.ndarray]
        Brightness distribution
    config : InversionConfig
        Inversion configuration
    
    Returns
    -------
    float
        Regularization term value
    """

    # Compute regularization term using configuration weights
    term_entropy = config.entropy_weight * np.sum(B_los**2 + B_perp**2)
    if config.fit_brightness and brightness is not None:
        term_entropy += config.entropy_weight * np.sum(brightness**2)

    term_smoothness = config.smoothness_weight * _compute_smoothness(
        B_los, B_perp, brightness)
    reg = term_entropy + term_smoothness

    return float(reg)


def _compute_smoothness(B_los: np.ndarray,
                        B_perp: np.ndarray,
                        brightness: Optional[np.ndarray] = None) -> float:
    """Compute smoothness measure of the magnetic field
    
    Parameters
    ----------
    B_los : np.ndarray
        Line-of-sight magnetic field
    B_perp : np.ndarray
        Perpendicular magnetic field
    brightness : Optional[np.ndarray]
        Brightness distribution
    
    Returns
    -------
    float
        Smoothness measure value
    """

    # Compute differences between adjacent pixels
    smoothness = float(np.sum(np.diff(B_los)**2 + np.diff(B_perp)**2))

    if brightness is not None:
        smoothness += float(np.sum(np.diff(brightness)**2))

    return smoothness


def _compute_fit_quality(
    B_los: np.ndarray,
    B_perp: np.ndarray,
    chi: np.ndarray,
    brightness: Optional[np.ndarray],
    obs_spectrum: np.ndarray,
    obs_error: np.ndarray,
    integrator: Any,
) -> Dict[str, float]:
    """Compute fit quality metrics
    
    Parameters
    ----------
    B_los : np.ndarray
        Final line-of-sight magnetic field
    B_perp : np.ndarray
        Final perpendicular magnetic field
    chi : np.ndarray
        Final magnetic field azimuth angle
    brightness : Optional[np.ndarray]
        Final brightness distribution
    obs_spectrum : np.ndarray
        Observed spectrum
    obs_error : np.ndarray
        Observation error
    integrator : Any
        Integrator instance
    
    Returns
    -------
    dict
        Fit quality metrics dictionary
    """

    # Compute synthetic spectrum
    integrator.compute_spectrum(B_los=B_los,
                                B_perp=B_perp,
                                chi=chi,
                                amp=brightness)
    predicted = integrator.I

    # Compute RMS residuals
    residuals = obs_spectrum - predicted
    rms_residual = float(np.sqrt(np.mean(residuals**2)))
    max_residual = float(np.max(np.abs(residuals)))

    # Compute per-point χ²
    nchi2_per_point = float(
        np.mean(((obs_spectrum - predicted) / obs_error)**2))

    return {
        'rms_residual': rms_residual,
        'max_residual': max_residual,
        'nchi2_per_point': nchi2_per_point,
    }


def get_inversion_summary(results: List[InversionResult]) -> Dict[str, Any]:
    """Generate statistical summary of inversion results
    
    Parameters
    ----------
    results : List[InversionResult]
        List of inversion results
    
    Returns
    -------
    dict
        Summary dictionary containing statistical information
    """

    if not results:
        return {}

    summary = {
        'num_phases': len(results),
        'converged_count': sum(1 for r in results if r.converged),
        'phase_details': [],
    }

    # Collect statistics for each phase
    for result in results:
        result.validate()
        mag_stats = result.get_magnetic_field_stats()
        opt_metrics = result.get_optimization_metrics()

        phase_info = {
            'phase_index': result.phase_index,
            'converged': result.converged,
            'iterations': opt_metrics['iterations'],
            'final_chi2': opt_metrics['final_chi2'],
            'B_los_rms': mag_stats['B_los']['rms'],
            'B_perp_rms': mag_stats['B_perp']['rms'],
        }

        summary['phase_details'].append(phase_info)

    # Compute global statistics
    all_iterations = [r.iterations_completed for r in results]
    all_chi2 = [r.final_chi2 for r in results]

    summary['global'] = {
        'avg_iterations': float(np.mean(all_iterations)),
        'avg_final_chi2': float(np.mean(all_chi2)),
        'convergence_rate': float(summary['converged_count'] / len(results)),
    }

    return summary
