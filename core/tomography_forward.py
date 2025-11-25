"""
tomography_forward.py - Forward Synthesis Workflow Engine

Implements the complete workflow from parameter object to forward synthesis results.

Main Entry Point: run_forward_synthesis()
"""

from typing import List, Any
import numpy as np
import logging

from core.tomography_config import ForwardModelConfig
from core.tomography_result import ForwardModelResult
from core.physical_model import create_physical_model
from core.disk_geometry_integrator import VelspaceDiskIntegrator
import core.mainFuncs as mf

logger = logging.getLogger(__name__)


def run_forward_synthesis(par,
                          obsSet: List[Any],
                          line_model: Any,
                          line_data: Any = None,
                          output_dir: str = './output',
                          verbose: bool = True) -> List[ForwardModelResult]:
    """Forward Synthesis Main Entry Point: Execute complete spectrum synthesis from parameter object
    
    This function implements the following steps:
    1. Create physical model (grid, geometry, magnetic field)
    2. Create forward configuration
    3. Perform spectrum synthesis for each observed phase
    4. Return synthesis results
    
    Parameters
    ----------
    par : readParamsTomog
        Parameter object containing all configuration information
    obsSet : List[ObservationProfile]
        List of observation datasets
    line_model : BaseLineModel
        Line model object (required)
    line_data : LineData, optional
        Line parameter data. If None, extracted from line_model
    output_dir : str, default='./output'
        Output directory
    verbose : bool, default=True
        Detailed output flag
    
    Returns
    -------
    List[ForwardModelResult]
        List of forward results for each phase
    
    Examples
    --------
    >>> par = readParamsTomog('input/params_tomog.txt')
    >>> obsSet = obsProfSetInRange('input/inSpec/', 'obs_phase_*.spec')
    >>> line_model = GaussianZeemanWeakLineModel(LineData('input/lines.txt'))
    >>> results = run_forward_synthesis(
    ...     par, obsSet, line_model, verbose=True)
    """
    if verbose:
        logger.info("[Forward] Creating physical model...")

    # Extract v_grid from the first observation if available
    # This ensures the synthetic spectrum matches the observation grid (e.g. -300 to 300 km/s)
    v_grid = None
    if obsSet and len(obsSet) > 0:
        first_obs = obsSet[0]
        if hasattr(first_obs, 'wl'):
            # Check if observation is already in velocity domain (LSD)
            is_velocity_domain = False
            if hasattr(first_obs, 'profile_type') and (
                    'lsd' in first_obs.profile_type.lower()
                    or 'velocity' in first_obs.profile_type.lower()):
                is_velocity_domain = True

            # Also check value range: if values are small (e.g. -1000 to 1000) and contain negative values, likely velocity
            wl_vals = np.asarray(first_obs.wl, dtype=float)
            if np.max(np.abs(wl_vals)) < 3000 and np.min(wl_vals) < 0:
                is_velocity_domain = True

            if is_velocity_domain:
                v_grid = wl_vals
                if verbose:
                    logger.info(
                        f"[Forward] Using velocity grid from observation (LSD/Velocity): {v_grid.min():.1f} to {v_grid.max():.1f} km/s"
                    )
            else:
                # Convert wavelength to velocity
                # v = c * (wl - wl0) / wl0
                c_kms = 2.99792458e5
                wl0 = line_data.wl0 if line_data and hasattr(line_data,
                                                             'wl0') else 656.3
                v_grid = c_kms * (wl_vals - wl0) / wl0
                if verbose:
                    logger.info(
                        f"[Forward] Using velocity grid from observation (Wavelength converted): {v_grid.min():.1f} to {v_grid.max():.1f} km/s"
                    )

    # Create physical model
    # Note: The created phys_model already contains a VelspaceDiskIntegrator instance (phys_model.integrator)
    phys_model = create_physical_model(
        par,
        wl0_nm=(line_data.wl0
                if line_data and hasattr(line_data, 'wl0') else 656.3),
        v_grid=v_grid,
        line_model=line_model,
        verbose=2 if verbose else 0)
    phys_model.validate()

    # Verify integrator is correctly created
    if phys_model.integrator is None:
        raise RuntimeError("Failed to create integrator for physical model")

    # Extract line data (if not provided)
    if line_data is None and hasattr(line_model, 'ld'):
        line_data = line_model.ld

    # Create forward configuration
    if verbose:
        logger.info("[Forward] Creating forward configuration...")

    # Extract normalize_continuum from par if available, default to True
    normalize_continuum = bool(getattr(par, 'normalize_continuum', True))

    config = ForwardModelConfig(par=par,
                                obsSet=obsSet,
                                lineData=line_data,
                                geom=phys_model.geometry,
                                line_model=line_model,
                                output_dir=output_dir,
                                save_intermediate=False,
                                verbose=1 if verbose else 0,
                                normalize_continuum=normalize_continuum)
    config.validate()

    # Execute forward synthesis
    if verbose:
        logger.info(
            f"[Forward] Starting synthesis for {len(obsSet)} observed phases..."
        )

    results = []
    for phase_idx, obs_data in enumerate(obsSet):
        try:
            if verbose:
                logger.info(
                    f"[Forward] Processing phase {phase_idx}/{len(obsSet)}")

            # Extract wavelength/velocity grid from observation data
            if hasattr(obs_data, 'wl'):
                # obs_v_grid = np.asarray(obs_data.wl, dtype=float)  # Observation wavelength
                pass  # Observation wavelength is kept by obs_data, integrator uses reasonable v_grid
            else:
                raise ValueError(
                    "Observation data missing wavelength/velocity information")

            # Calculate phase for current observation
            current_phase = 0.0
            current_hjd = 0.0
            if hasattr(par, 'jDates') and len(par.jDates) > phase_idx:
                current_hjd = par.jDates[phase_idx]
                if hasattr(par, 'jDateRef') and hasattr(par, 'period'):
                    current_phase = mf.compute_phase_from_jd(
                        current_hjd, par.jDateRef, par.period)
            else:
                # Fallback: evenly distributed phases
                current_phase = phase_idx / len(obsSet) if len(
                    obsSet) > 0 else 0.0

            # Calculate rotation angle for observation phase
            # phi_obs = 2*pi * phase
            # Note: phase is 0~1 here
            current_phase = par.phases[phase_idx] if hasattr(
                par, 'phases') and len(par.phases) > phase_idx else 0.0

            if verbose:
                print(
                    f"[Forward] Phase {phase_idx}: current_phase={current_phase:.4f}, phi_rot={2.0 * np.pi * current_phase:.4f}"
                )

            # Ensure integrator uses correct physical model
            # Re-create integrator for this phase to ensure correct time evolution and computation
            # VelspaceDiskIntegrator computes spectrum in __init__
            integrator = VelspaceDiskIntegrator(
                geom=phys_model.geometry,
                wl0_nm=phys_model._wl0 if phys_model._wl0 else 656.3,
                line_model=phys_model._line_model,
                v_grid=phys_model.integrator.v,  # Reuse velocity grid
                inst_fwhm_kms=phys_model.integrator.inst_fwhm,
                time_phase=current_phase,
                normalize_continuum=config.normalize_continuum)

            # Get actual v_grid from integrator
            v_grid = integrator.v

            # Get Stokes components
            stokes_i = (integrator.I
                        if hasattr(integrator, 'I') else np.ones_like(v_grid))
            stokes_v = (integrator.V
                        if hasattr(integrator, 'V') else np.zeros_like(v_grid))
            stokes_q = (integrator.Q
                        if hasattr(integrator, 'Q') else np.zeros_like(v_grid))
            stokes_u = (integrator.U
                        if hasattr(integrator, 'U') else np.zeros_like(v_grid))

            # Create result object
            # Note: pol_channel currently supports single value (I/V/Q/U), default is V
            # Get polarization channel for current observation from par.polChannels
            pol_chan = 'V'
            if hasattr(par, 'polChannels') and len(
                    par.polChannels) > phase_idx:
                pol_chan = par.polChannels[phase_idx]

            result = ForwardModelResult(
                stokes_i=stokes_i,
                stokes_v=stokes_v,
                stokes_q=stokes_q,
                stokes_u=stokes_u,
                wavelength=v_grid,
                error=obs_data.sigma if hasattr(obs_data, 'sigma') else None,
                hjd=current_hjd,
                phase_index=phase_idx,
                pol_channel=pol_chan,
                model_name="forward_synthesis",
                integrator=integrator  # Attach integrator for save_geomodel
            )
            results.append(result)

            if verbose:
                logger.info(
                    f"[Forward]   ✓ Phase {phase_idx} synthesis complete")
                # Note: Chi2 is not calculated here because observation wavelength grid might differ from synthetic grid
                # Chi2 calculation should be done in inversion phase using interpolated spectrum

        except Exception as e:
            logger.error(f"[Forward] Phase {phase_idx} synthesis failed: {e}")
            if verbose:
                raise
            continue

    if len(results) == 0:
        raise RuntimeError("All phase synthesis failed")

    if verbose:
        logger.info(
            f"[Forward] ✓ Successfully synthesized {len(results)}/{len(obsSet)} phases"
        )

    return results
