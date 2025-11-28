"""Example 05: Plot dynamic spectrum from forward results

This script shows how to:

1. Run (or re-run) the forward tomography using `input/params_tomog.txt`
2. Build an `IrregularDynamicSpectrum` from the forward results
3. Plot a phase–wavelength dynamic spectrum of Stokes I

How to run (from project root)::

	python examples/05_dynamic_spectrum.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root (parent of examples) is on sys.path so that
# `pyzeetom`, `core`, and `utils` can be imported when running this
# script directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

from pyzeetom import tomography
from utils.dynamic_spectrum import IrregularDynamicSpectrum


def get_project_root() -> Path:
    return project_root


def main() -> None:
    project_root = get_project_root()
    param_file = project_root / "input" / "params_tomog.txt"

    if not param_file.exists():
        raise FileNotFoundError(
            f"Parameter file not found: {param_file}. "
            "Please run examples/01_generate_spot_data.py first.")

    print("Running forward tomography (for dynamic spectrum)...")
    forward_results = tomography.forward_tomography(
        param_file=str(param_file),
        verbose=0,
        output_dir=str(project_root / "output" / "forward_test"),
    )

    # Build dynamic spectrum
    times = np.array([r.phase_index for r in forward_results])
    xs = [r.wavelength for r in forward_results]
    intensities = [r.stokes_i for r in forward_results]

    dynspec = IrregularDynamicSpectrum(times, xs, intensities)

    fig, ax = dynspec.plot(
        cmap="RdBu_r",
        vmin=float(np.min(intensities)),
        vmax=float(np.max(intensities)),
        xlabel="Wavelength (nm)",
        ylabel="Phase",
        title="Forward Model Dynamic Spectrum",
    )
    plt.show()


if __name__ == "__main__":
    main()
