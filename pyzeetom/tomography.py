"""pyzeetom.tomography - Phase 2.5.2.1 completely refactored

This version uses the new architecture without backward compatibility.

Core design:
- Two main entry points: forward_tomography(), inversion_tomography()
- Complete use of Phase 2.5 configuration system
- Clear architecture layers: UI -> Config -> Workflows -> Physics -> Core
"""
import sys
from pathlib import Path
from typing import List
import logging
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Add project root to path for direct script execution
if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

# Core library imports
import core.mainFuncs as mf
import core.SpecIO as SpecIO
from core.local_linemodel_basic import LineData as BasicLineData, GaussianZeemanWeakLineModel, ConstantAmpLineModel
from core.tomography_config import InversionConfig
from core.tomography_result import ForwardModelResult, InversionResult
from core.tomography_forward import run_forward_synthesis
from core.tomography_inversion import run_mem_inversion


def forward_tomography(
        param_file: str = 'input/params_tomog.txt',
        verbose: int = 1,
        output_dir: str = './output') -> List[ForwardModelResult]:
    """Forward synthesis main entry point - Phase 2.5.3.1 refactored
    
    Execute forward synthesis for all observed phases using new architecture.
    
    Parameters
    ----------
    param_file : str
        Parameter file path
    verbose : int
        Verbosity: 0=silent, 1=normal, 2=detailed
    output_dir : str
        Output directory
        
    Returns
    -------
    List[ForwardModelResult]
        One result per observed phase
        
    Examples
    --------
    >>> results = forward_tomography('input/params_tomog.txt', verbose=1)
    >>> for result in results:
    ...     print(f"Phase {result.phase_index}: I in [{result.stokes_i.min():.3f}, {result.stokes_i.max():.3f}]")
    """
    if verbose:
        print(
            "[forward_tomography] Phase 2.5.3.1 - High-level API entry point")

    # 1. Load parameters and observations
    if verbose:
        print(f"[forward_tomography] Reading parameters: {param_file}")
    par = mf.readParamsTomog(param_file)

    if verbose:
        print(
            f"[forward_tomography] Reading observations: {len(par.fnames)} files"
        )
    obsSet = SpecIO.obsProfSetInRange(
        list(par.fnames),
        par.velStart,
        par.velEnd,
        par.velRs,
        file_type=getattr(par, 'obsFileType', 'auto'),
        pol_channels=getattr(par, 'polChannels', None),
        phases=getattr(par, 'phases', None))

    # 1.5 Check for potential scale mismatch (Warning only)
    if len(obsSet) > 0:
        median_flux = np.median(obsSet[0].specI)
        if median_flux > 1.5:
            if verbose:
                print(
                    f"[forward_tomography] Warning: Observations appear unnormalized (median flux ~ {median_flux:.1f})."
                )
                print(
                    "  This may cause large Chi2 values if the model is normalized to 1.0."
                )

    # 2. Load line parameters
    line_file = getattr(par, 'lineParamFile', 'input/lines.txt')
    if verbose:
        print(f"[forward_tomography] Reading line parameters: {line_file}")
    lineData = BasicLineData(line_file)

    # 3. Create line model
    if verbose:
        print("[forward_tomography] Creating line model")
    k_qu = float(getattr(par, 'lineKQU', 1.0))
    enable_v = bool(getattr(par, 'lineEnableV', 1))
    enable_qu = bool(getattr(par, 'lineEnableQU', 1))
    amp_const = float(getattr(par, 'lineAmpConst', 1))

    base_model = GaussianZeemanWeakLineModel(lineData,
                                             k_QU=k_qu,
                                             enable_V=enable_v,
                                             enable_QU=enable_qu)

    # Determine if we should use ConstantAmpLineModel or the model's intrinsic amplitude
    # If initTomogFile is enabled (1), we assume the loaded model contains the
    # desired amplitude distribution (e.g. spots), so we skip the constant amplitude wrapper.
    if getattr(par, 'initTomogFile', 0) == 1:
        if verbose:
            print(
                "[forward_tomography] Using model amplitude (skipping ConstantAmpLineModel)"
            )
        line_model = base_model
    else:
        line_model = ConstantAmpLineModel(base_model, amp=amp_const)

    # 4. Execute high-level forward synthesis workflow
    if verbose:
        print("[forward_tomography] Starting forward synthesis workflow")

    results = run_forward_synthesis(par=par,
                                    obsSet=obsSet,
                                    line_model=line_model,
                                    line_data=lineData,
                                    output_dir=output_dir,
                                    verbose=(verbose > 0))

    if verbose:
        print(
            f"[forward_tomography] Forward synthesis complete: {len(results)} phases"
        )

    # 5. Save results (Spectra and Geomodels)
    if verbose:
        print(f"[forward_tomography] Saving results to {output_dir}")

    # 5.1 Save Spectra (Batch)
    # Convert results to tuples for save_model_spectra_to_outModelSpec
    result_tuples = []
    for res in results:
        result_tuples.append((res.wavelength, res.stokes_i, res.stokes_v,
                              res.stokes_q, res.stokes_u, res.pol_channel))

    mf.save_model_spectra_to_outModelSpec(par,
                                          result_tuples,
                                          obsSet,
                                          output_dir=output_dir,
                                          verbose=verbose)

    # 5.2 Save Geomodels (Individual)
    for i, result in enumerate(results):
        try:
            # Note: result.integrator is available thanks to core/tomography_forward.py update
            geom_file = result.save_geomodel(output_dir=output_dir,
                                             verbose=(verbose > 1))
            if verbose > 1:
                print(
                    f"  Saved geomodel for phase {i}: {Path(geom_file).name}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to save geomodel for phase {i}: {e}")

    return results


def inversion_tomography(
        param_file: str = 'input/params_tomog.txt',
        verbose: int = 1,
        output_dir: str = './output',
        config_overrides: dict = None) -> List[InversionResult]:
    """Inversion main entry point - Phase 2.5.2.1 complete refactor
    
    Execute MEM inversion optimization for iterative parameter refinement.
    
    Parameters
    ----------
    param_file : str
        Parameter file path
    verbose : int
        Verbosity: 0=silent, 1=normal, 2=detailed
    output_dir : str
        Output directory
    config_overrides : dict, optional
        Dictionary of parameters to override in the configuration object.
        Useful for programmatic control without modifying files.
        
    Returns
    -------
    List[InversionResult]
        One result per iteration
        
    Examples
    --------
    >>> results = inversion_tomography('input/params_tomog.txt', verbose=1)
    >>> final_result = results[-1]
    >>> print(f"Final chi2: {final_result.chi2:.3e}")
    """
    if verbose >= 1:
        print(
            "[inversion_tomography] Phase 2.5.2.1 - New architecture entry point"
        )

    # 1. Load parameters and observations
    if verbose >= 1:
        print(f"[inversion_tomography] Reading parameters: {param_file}")
    par = mf.readParamsTomog(param_file)

    # 1.1 Apply overrides
    if config_overrides:
        if verbose >= 1:
            print(
                f"[inversion_tomography] Applying {len(config_overrides)} config overrides"
            )
        for k, v in config_overrides.items():
            setattr(par, k, v)
            if verbose >= 2:
                print(f"  Override: {k} = {v}")

    if verbose >= 1:
        print(
            f"[inversion_tomography] Reading observations: {len(par.fnames)} files"
        )
    obsSet = SpecIO.obsProfSetInRange(
        list(par.fnames),
        par.velStart,
        par.velEnd,
        par.velRs,
        file_type=getattr(par, 'obsFileType', 'auto'),
        pol_channels=getattr(par, 'polChannels', None),
        phases=getattr(par, 'phases', None))

    # 1.5 Check for potential scale mismatch (Warning only)
    if len(obsSet) > 0:
        median_flux = np.median(obsSet[0].specI)
        if median_flux > 1.5:
            if verbose >= 1:
                print(
                    f"[inversion_tomography] Warning: Observations appear unnormalized (median flux ~ {median_flux:.1f})."
                )
                print(
                    "  This may cause large model parameters (brightness >> 1) if the model is normalized to 1.0."
                )
                print(
                    "  Consider normalizing your data or ensuring the model scale matches the observations."
                )

    # 2. Load line parameters
    line_file = getattr(par, 'lineParamFile', 'input/lines.txt')
    if verbose >= 1:
        print(f"[inversion_tomography] Reading line parameters: {line_file}")
    lineData = BasicLineData(line_file)

    # 3. Create inversion config (new architecture)
    if verbose >= 1:
        print("[inversion_tomography] Creating InversionConfig")
    config = InversionConfig.from_par(par,
                                      obsSet,
                                      lineData,
                                      verbose=(verbose > 1))

    # DEBUG: Check line model amplitude
    if verbose >= 2:
        if hasattr(config.line_model, 'amp'):
            print(f"[DEBUG] config.line_model.amp = {config.line_model.amp}")
        else:
            print("[DEBUG] config.line_model has no 'amp' attribute")

    # 4. Set output directory and inversion parameters
    config.output_dir = output_dir
    config.save_intermediate = (verbose > 1)
    config.verbose = verbose
    config.save_every_iter = max(1, config.num_iterations // 5)

    # 5. Validate config
    if verbose >= 1:
        print("[inversion_tomography] Validating configuration")
    config.validate()

    # 6. Show summary (Delegated to run_mem_inversion for verbose >= 2)
    # if verbose:
    #    print(config.create_summary())

    # 7. Execute inversion workflow - DELEGATE TO CORE (thin wrapper pattern)
    if verbose >= 1:
        print("[inversion_tomography] Executing MEM inversion")

    results = run_mem_inversion(config, verbose=verbose)

    if verbose >= 0:
        final = results[-1]
        print("[inversion_tomography] Inversion complete")
        print(f"  Final chi2: {final.final_chi2:.3e}")
        print(f"  Final entropy: {final.final_entropy:.3e}")
        print(f"  Converged: {'Yes' if final.converged else 'No'}")

    # Save final result
    if output_dir and len(results) > 0:
        import os
        from pathlib import Path
        os.makedirs(output_dir, exist_ok=True)
        final_res = results[-1]

        # 1. Save Spectra (Batch)
        # Convert results to tuples for save_model_spectra_to_outModelSpec
        # Note: InversionResult doesn't store full spectra history for all phases directly in a list like ForwardModelResult
        # We need to reconstruct the spectra from the final model for all phases

        if verbose >= 1:
            print(
                f"[inversion_tomography] Saving final model spectra to {output_dir}"
            )

        # Re-compute spectra for all phases using the final model
        # We need to access the integrators or re-create them.
        # Since we don't have easy access to the integrators here without refactoring run_mem_inversion to return them,
        # we will rely on the fact that InversionResult might be enhanced to store the final fit spectra,
        # OR we re-calculate them here using the config and final parameters.

        # Re-calculation approach:
        final_spectra_tuples = []

        # Use the config to recreate the physical model context
        # We need to iterate over config.obsSet and compute spectrum for each
        from core.disk_geometry_integrator import VelspaceDiskIntegrator

        c_kms = 2.99792458e5

        for i, obs_data in enumerate(config.obsSet):
            # Re-create integrator for this phase
            # Logic similar to _invert_simultaneous in tomography_inversion.py

            # Calculate v_grid
            # Check if observation is already in velocity domain (LSD)
            is_velocity_domain = False
            if hasattr(obs_data, 'profile_type') and (
                    'lsd' in obs_data.profile_type.lower()
                    or 'velocity' in obs_data.profile_type.lower()):
                is_velocity_domain = True

            wl_vals = np.asarray(obs_data.wl, dtype=float)
            if np.max(np.abs(wl_vals)) < 3000 and np.min(wl_vals) < 0:
                is_velocity_domain = True

            if is_velocity_domain:
                v_grid = wl_vals
            else:
                v_grid = (obs_data.wl -
                          config.lineData.wl0) / config.lineData.wl0 * c_kms

            integrator = VelspaceDiskIntegrator(
                geom=config.
                geom,  # Shared geometry, will be updated with final params
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
                time_phase=getattr(obs_data, 'phase', 0.0))

            # Compute spectrum with final parameters
            if verbose >= 2:
                print(
                    f"  [Debug] Phase {i}: B_los range [{np.min(final_res.B_los_final):.2f}, {np.max(final_res.B_los_final):.2f}]"
                )
                print(
                    f"  [Debug] Phase {i}: Brightness range [{np.min(final_res.brightness_final):.2f}, {np.max(final_res.brightness_final):.2f}]"
                )

            integrator.compute_spectrum(B_los=final_res.B_los_final,
                                        B_perp=final_res.B_perp_final,
                                        chi=final_res.chi_final,
                                        amp=final_res.brightness_final)

            if verbose >= 2:
                print(
                    f"  [Debug] Phase {i}: I range [{np.min(integrator.I):.6f}, {np.max(integrator.I):.6f}]"
                )
                print(
                    f"  [Debug] Phase {i}: V range [{np.min(integrator.V):.6f}, {np.max(integrator.V):.6f}]"
                )

            # Collect result
            # (wavelength, I, V, Q, U, pol_channel)
            # Note: integrator.wl_cells is 2D, we want the velocity grid or wavelength grid corresponding to obs
            # obs_data.wl is the x-axis

            final_spectra_tuples.append(
                (obs_data.wl, integrator.I, integrator.V, integrator.Q,
                 integrator.U, getattr(obs_data, 'pol_channel', 'V')))

        mf.save_model_spectra_to_outModelSpec(par,
                                              final_spectra_tuples,
                                              obsSet,
                                              output_dir=output_dir,
                                              verbose=verbose)

        # 2. Save Geomodel (Single Master Map)
        # Since MEM inversion produces a single master map (unlike forward which might have phase-specific maps),
        # we save just one geomodel.

        if verbose >= 1:
            print(
                f"[inversion_tomography] Saving final geomodel to {output_dir}"
            )

        geom_filename = Path(output_dir) / "mem_inversion_model.tomog"

        try:
            # Use the last integrator from the loop above to save the geomodel
            # The integrator's geometry has been updated with the final parameters
            if 'integrator' in locals() and integrator is not None:
                # Create metadata
                meta = {
                    'phase_index': 0,
                    'hjd': 0.0,
                    'pol_channel': 'I+V',
                    'model_name': 'mem_inversion',
                    'creation_time': datetime.now().isoformat(),
                    'final_chi2': final_res.final_chi2,
                    'final_entropy': final_res.final_entropy
                }

                integrator.write_geomodel(str(geom_filename), meta=meta)

                if verbose >= 1:
                    print(f"  Saved master geomodel: {geom_filename.name}")
            else:
                if verbose >= 1:
                    print(
                        "  Warning: No integrator available to save geomodel.")

        except Exception as e:
            if verbose >= 1:
                print(f"  Warning: Failed to save geomodel: {e}")

    return results


if __name__ == "__main__":
    input_params = 'output/spot_forward/params_tomog_spotsimu.txt'
    output_dir = 'output/spot_forward/simuspec'
    # Example usage
    print("=" * 70)
    print("pyZeeTom - Phase 2.5.2.1 Complete Refactor")
    print("=" * 70)
    print()

    # Forward
    print("Executing forward synthesis...")
    forward_results = forward_tomography(param_file=input_params,
                                         output_dir=output_dir,
                                         verbose=2)
    print(f"OK - Got {len(forward_results)} phase results\n")

    # Inversion (requires num_iterations in parameter file)
    # print("Executing MEM inversion...")
    # inversion_results = inversion_tomography(verbose=2)
    # print(f"OK - Completed {len(inversion_results)} iterations\n")
