import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pyzeetom.tomography import inversion_tomography
from core.disk_geometry_integrator import SimpleDiskGeometry, VelspaceDiskIntegrator


def main():
    print("=" * 60)
    print("Running Inversion Test Project")
    print("=" * 60)

    # Paths
    param_file = project_root / 'input/intomog_ap149_update.txt'
    output_dir = project_root / 'output/ap149_test'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Inversion
    print(f"\n[Step 1] Running inversion using {param_file}...")
    results = inversion_tomography(param_file=str(param_file),
                                   verbose=0,
                                   output_dir=str(output_dir),
                                   config_overrides={
                                       'entropyWeight': 0.0001,
                                       'smoothnessWeight': 0.001
                                   })

    if not results:
        print("Error: Inversion returned no results.")
        return

    final_result = results[-1]
    print(f"\nInversion completed. Final Chi2: {final_result.final_chi2:.4f}")

    print("-" * 60)
    print("Test Complete.")


if __name__ == "__main__":
    main()
