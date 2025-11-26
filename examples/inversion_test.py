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


def compare_maps(name, map1, map2, mask=None):
    """Compare two maps and return metrics."""
    if map1.shape != map2.shape:
        print(
            f"{name:10s} | RMSE: N/A    | Corr: N/A    (Shape mismatch: {map1.shape} vs {map2.shape})"
        )
        return 0.0, 0.0

    if mask is None:
        mask = np.ones_like(map1, dtype=bool)

    m1 = map1[mask]
    m2 = map2[mask]

    # RMS Error
    rmse = np.sqrt(np.mean((m1 - m2)**2))

    # Correlation Coefficient
    if np.std(m1) > 1e-9 and np.std(m2) > 1e-9:
        corr = np.corrcoef(m1, m2)[0, 1]
    else:
        corr = 0.0

    print(f"{name:10s} | RMSE: {rmse:.4f} | Corr: {corr:.4f}")
    return rmse, corr


def main():
    print("=" * 60)
    print("Running Inversion Test Project")
    print("=" * 60)

    # Paths
    param_file = project_root / 'input/intomog_ap149_05Dec06_updated.txt'
    truth_file = project_root / 'output/spot_forward/truth_model.tomog'
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

    # 2. Load Truth Model
    print(f"\n[Step 2] Loading truth model from {truth_file}...")
    if not truth_file.exists():
        print(f"Error: Truth model file not found: {truth_file}")
        return

    # SimpleDiskGeometry.read_geomodel returns (geom_like, meta, table)
    # table is a dict with keys like 'r', 'phi', 'B_los', 'B_perp', 'chi', 'amp'
    # Note: read_geomodel is a static method of VelspaceDiskIntegrator
    _, _, truth_table = VelspaceDiskIntegrator.read_geomodel(str(truth_file))

    # 3. Compare Results
    print(f"\n[Step 3] Comparing Inversion Result vs Truth Model")
    print("-" * 60)

    # Extract maps
    # Inversion result
    inv_blos = final_result.B_los_final
    inv_bperp = final_result.B_perp_final
    inv_chi = final_result.chi_final
    inv_bri = final_result.brightness_final

    # Truth model
    # Note: keys in table might be lowercase or uppercase depending on implementation
    # Let's check keys
    keys = truth_table.keys()
    # Map keys
    key_blos = next((k for k in keys if k.lower() == 'b_los'), None)
    key_bperp = next((k for k in keys if k.lower() == 'b_perp'), None)
    key_chi = next((k for k in keys if k.lower() == 'chi'), None)
    key_bri = next((k for k in keys if k.lower() in ['amp', 'brightness']),
                   None)

    truth_blos = truth_table[key_blos] if key_blos else np.zeros_like(inv_blos)
    truth_bperp = truth_table[key_bperp] if key_bperp else np.zeros_like(
        inv_bperp)
    truth_chi = truth_table[key_chi] if key_chi else np.zeros_like(inv_chi)
    truth_bri = truth_table[key_bri] if key_bri else np.ones_like(inv_bri)

    # Check size match
    if len(inv_blos) != len(truth_blos):
        print(
            f"Warning: Grid size mismatch (Inversion: {len(inv_blos)}, Truth: {len(truth_blos)})"
        )
        print("Skipping detailed map comparison.")
        return

    # Compare
    compare_maps("Brightness", inv_bri, truth_bri)
    compare_maps("B_los", inv_blos, truth_blos)
    compare_maps("B_perp", inv_bperp, truth_bperp)
    compare_maps("Chi", inv_chi, truth_chi)

    print("-" * 60)
    print("Test Complete.")


if __name__ == "__main__":
    main()
