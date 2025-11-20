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
        pol_channels=getattr(par, 'polChannels', None))

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
        output_dir: str = './output') -> List[InversionResult]:
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
    if verbose:
        print(
            "[inversion_tomography] Phase 2.5.2.1 - New architecture entry point"
        )

    # 1. Load parameters and observations
    if verbose:
        print(f"[inversion_tomography] Reading parameters: {param_file}")
    par = mf.readParamsTomog(param_file)

    if verbose:
        print(
            f"[inversion_tomography] Reading observations: {len(par.fnames)} files"
        )
    obsSet = SpecIO.obsProfSetInRange(
        list(par.fnames),
        par.velStart,
        par.velEnd,
        par.velRs,
        file_type=getattr(par, 'obsFileType', 'auto'),
        pol_channels=getattr(par, 'polChannels', None))

    # 2. Load line parameters
    line_file = getattr(par, 'lineParamFile', 'input/lines.txt')
    if verbose:
        print(f"[inversion_tomography] Reading line parameters: {line_file}")
    lineData = BasicLineData(line_file)

    # 3. Create inversion config (new architecture)
    if verbose:
        print("[inversion_tomography] Creating InversionConfig")
    config = InversionConfig.from_par(par,
                                      obsSet,
                                      lineData,
                                      verbose=(verbose > 1))

    # 4. Set output directory and inversion parameters
    config.output_dir = output_dir
    config.save_intermediate = (verbose > 1)
    config.verbose = verbose
    config.save_every_iter = max(1, config.num_iterations // 5)

    # 5. Validate config
    if verbose:
        print("[inversion_tomography] Validating configuration")
    config.validate()

    # 6. Show summary
    if verbose:
        print(config.create_summary())

    # 7. Execute inversion workflow - DELEGATE TO CORE (thin wrapper pattern)
    if verbose:
        print("[inversion_tomography] Executing MEM inversion")

    results = run_mem_inversion(config, verbose=(verbose > 0))

    if verbose:
        final = results[-1]
        print("[inversion_tomography] Inversion complete")
        print(f"  Final chi2: {final.final_chi2:.3e}")
        print(f"  Final entropy: {final.final_entropy:.3e}")
        print(f"  Converged: {'Yes' if final.converged else 'No'}")

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
