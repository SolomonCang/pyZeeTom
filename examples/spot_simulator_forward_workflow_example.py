#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete example: Spot model generation and forward synthesis

This example demonstrates the complete workflow:
1. Generate .tomog model files using SpotSimulator
2. Create adjusted parameter file
3. Run forward synthesis using forward_tomography()
4. Save synthetic spectra using SpecIO.write_model_spectrum()

Workflow steps:
  [Step 1] Generate .tomog model files
    - Create diskGrid and SpotSimulator
    - Configure spots for each phase
    - Use export_to_geomodel() to output .tomog files

  [Step 2] Create adjusted parameter file
    - Read reference parameter file (input/params_tomog.txt)
    - Adjust initTomogFile and initModelPath
    - Create observation file list from phases
    - Use write_params_file() to output new parameter file

  [Step 3] Forward synthesis
    - Call forward_tomography() for spectrum synthesis
    - Returns list of ForwardModelResult objects

  [Step 4] Save synthetic spectra
    - Use SpecIO.write_model_spectrum() to save each phase spectrum
    - Support multiple polarization channels (I/V/Q/U)

Usage:
    python examples/spot_simulator_forward_workflow_example.py [--mode full|manual]
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from spot_forward_workflow import (
    full_workflow,
    generate_spot_tomog_models,
    create_params_tomog_spotsimu,
    generate_synthetic_spectra,
    save_synthetic_spectra,
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def example_full_workflow(verbose: int = 1):
    """
    Method 1: Use full_workflow() to complete all steps automatically
    
    This is the simplest way to complete the entire workflow.
    """
    print("\n" + "=" * 80)
    print("Example 1: Full Automatic Workflow (full_workflow)")
    print("=" * 80)

    # Define phases
    phases = np.array([0.0, 0.25, 0.5, 0.75])
    output_dir = './output/example_spot_forward_auto'

    # Execute complete workflow
    result = full_workflow(phases=phases,
                           output_dir=output_dir,
                           verbose=verbose)

    # Display results
    print("\nReturned data structure:")
    print(f"  phases: {result['phases'].tolist()}")
    print(f"  tomog_files: {len(result['tomog_files'])} files")
    for f in result['tomog_files']:
        print(f"    - {Path(f).name}")
    print(f"  param_file: {result['param_file']}")
    print(f"  results: {len(result['results'])} ForwardModelResult objects")
    print(f"  saved_files: {len(result['saved_files'])} spectrum files")
    for f in result['saved_files']:
        print(f"    - {Path(f).name}")

    return result


def example_step_by_step(verbose: int = 1):
    """
    Method 2: Call functions manually for more control
    
    This approach gives users more control at each step.
    """
    print("\n" + "=" * 80)
    print("Example 2: Step-by-Step Workflow")
    print("=" * 80)

    phases = np.array([0.0, 0.3, 0.6])
    output_dir = './output/example_spot_forward_manual'

    # ====================================================================
    # Step 1: Generate .tomog models
    # ====================================================================
    print("\n[Step 1] Generate .tomog model files")
    print("-" * 80)

    step1_result = generate_spot_tomog_models(phases=phases,
                                              output_dir=output_dir,
                                              nr=40,
                                              r_in=0.5,
                                              r_out=4.0,
                                              inclination_deg=60.0,
                                              pOmega=-0.05,
                                              r0_rot=1.0,
                                              period_days=1.0,
                                              verbose=verbose)

    tomog_files = step1_result['tomog_files']
    print("\nGenerated .tomog files:")
    for idx, f in enumerate(tomog_files):
        file_size = Path(f).stat().st_size / 1024
        print(f"  [{idx}] {Path(f).name} ({file_size:.1f} KB)")

    # ====================================================================
    # Step 2: Create adjusted parameter file
    # ====================================================================
    print("\n[Step 2] Create adjusted parameter file")
    print("-" * 80)

    param_file = create_params_tomog_spotsimu(
        ref_param_file='input/params_tomog.txt',
        phases=phases,
        tomog_files=tomog_files,
        output_dir=output_dir,
        output_param_file=f"{output_dir}/params_tomog_spotsimu.txt",
        verbose=verbose)

    print("\nParameter file created:")
    print(f"  {param_file}")

    # Read and display parameter file content
    print("\nParameter file content summary:")
    with open(param_file, 'r') as f:
        lines = f.readlines()
        data_lines = [
            line.strip() for line in lines
            if line.strip() and not line.strip().startswith('#')
        ]
        for line_idx, line in enumerate(data_lines[:10]):
            print(f"  Line {line_idx}: {line}")

    # ====================================================================
    # Step 3: Forward synthesis
    # ====================================================================
    print("\n[Step 3] Forward synthesis")
    print("-" * 80)

    results = generate_synthetic_spectra(param_file=param_file,
                                         output_dir=output_dir,
                                         verbose=verbose)

    print("\nSynthesis result summary:")
    for idx, result in enumerate(results):
        print(f"  [Phase {idx}] {type(result).__name__}")
        i_min = result.stokes_i.min()
        i_max = result.stokes_i.max()
        v_min = result.stokes_v.min()
        v_max = result.stokes_v.max()
        print(f"    stokes_i: shape={result.stokes_i.shape}, "
              f"range=[{i_min:.3f}, {i_max:.3f}]")
        print(f"    stokes_v: shape={result.stokes_v.shape}, "
              f"range=[{v_min:.3f}, {v_max:.3f}]")

    # ====================================================================
    # Step 4: Save synthetic spectra
    # ====================================================================
    print("\n[Step 4] Save synthetic spectra")
    print("-" * 80)

    pol_channels = ['I', 'V', 'Q']
    saved_files = save_synthetic_spectra(results=results,
                                         phases=phases,
                                         pol_channels=pol_channels,
                                         output_dir=output_dir,
                                         verbose=verbose)

    print("\nSaved spectrum files:")
    for idx, f in enumerate(saved_files):
        file_size = Path(f).stat().st_size / 1024
        print(f"  [{idx}] {Path(f).name} ({file_size:.1f} KB)")

    return {
        'phases': phases,
        'tomog_files': tomog_files,
        'param_file': param_file,
        'results': results,
        'saved_files': saved_files,
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description=
        'Complete example: Spot model generation and forward synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full automatic workflow
  python examples/spot_simulator_forward_workflow_example.py --mode full
  
  # Run step-by-step workflow (more control)
  python examples/spot_simulator_forward_workflow_example.py --mode manual
  
  # Verbose output
  python examples/spot_simulator_forward_workflow_example.py --verbose 2
        """)

    parser.add_argument('--mode',
                        type=str,
                        choices=['full', 'manual', 'both'],
                        default='both',
                        help='Execution mode (default: both)')

    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        choices=[0, 1, 2],
                        help='Verbosity level (default: 1)')

    args = parser.parse_args()

    try:
        if args.mode in ['full', 'both']:
            result1 = example_full_workflow(verbose=args.verbose)

        if args.mode in ['manual', 'both']:
            result2 = example_step_by_step(verbose=args.verbose)

        print("\n" + "=" * 80)
        print("✅ Example execution completed")
        print("=" * 80)
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose > 1:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
