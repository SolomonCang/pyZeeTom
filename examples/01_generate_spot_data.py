"""Example 01: Generate three-spot synthetic observation data

This script corresponds to steps 1–3 in the quickstart notebook:

- Set the project root so that `pyzeetom` and `utils` can be imported
- Use `SpotSimulator` to place 3 magnetic spots on the disk
- Synthesize multi-phase Stokes I/V/Q/U spectra for a set of phases
- Create a shared forward / inversion parameter file `input/params_tomog.txt`

How to run (from project root)::

	python examples/01_generate_spot_data.py

After running you will get:

- `input/spot_forward_obs/` with *.lsd observation files (with noise)
- `output/spot_truth.tomog` containing the true geometric model
- `input/params_tomog.txt` used by both forward and inversion examples
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Ensure project root (parent of examples) is on sys.path so that
# `pyzeetom`, `core`, and `utils` can be imported when running this
# script directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

from core.grid_tom import diskGrid
from utils.spot_simulator import SpotSimulator, SpotConfig


def get_project_root() -> Path:
    """Return the project root directory (parent of examples)."""

    return project_root


def main() -> None:
    project_root = get_project_root()
    print(f"Project root: {project_root}")

    # 1) Prepare output directory for simulated observations
    sim_output_dir = project_root / "input/spot_forward_obs"
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(project_root)
    print("Generating 3-spot simulation data...")

    # 2) Create disk grid (typical configuration, consistent with three_spot_simulation.py)
    grid = diskGrid(
        nr=10,
        r_in=0.0,
        r_out=3.0,
    )

    # 3) Initialize three-spot simulator
    sim = SpotSimulator(
        grid=grid,
        inclination_rad=np.deg2rad(60.0),
        phi0=0.0,
        pOmega=-0.05,
        r0_rot=0.8,
        period_days=1.0,
    )

    # 4) Place 3 spots on the disk
    spots = [
        SpotConfig(r=1.5 * 0.8,
                   phi=np.deg2rad(30),
                   amplitude=4.2,
                   B_los=500,
                   radius=0.2),
        SpotConfig(r=0.8 * 0.8,
                   phi=np.deg2rad(100),
                   amplitude=5.5,
                   B_los=-300,
                   radius=0.2),
        SpotConfig(r=2.0 * 0.8,
                   phi=np.deg2rad(270),
                   amplitude=4.5,
                   B_los=500,
                   radius=0.3),
    ]
    sim.add_spots(spots)
    print("Created 3 spots.")

    # 5) Specify observation phases (same as in the notebook / three_spot_simulation)
    phases = np.array([
        0.0,
        0.05,
        0.12,
        0.18,
        0.25,
        0.33,
        0.41,
        0.48,
        0.55,
        0.65,
        0.72,
        0.80,
        0.88,
        0.92,
        0.98,
    ])

    sim.configure_multi_phase_synthesis(phases=phases)

    # 6) Generate multi-phase forward model (Stokes I/V/Q/U)
    multi_results = sim.generate_forward_model()
    print(f"Generated {len(multi_results['results'])} synthetic spectra.")

    # 7) Write shared parameter file for forward / inversion
    param_file_path = project_root / "input/params_tomog.txt"
    print(f"Creating parameter file at: {param_file_path}")
    param_file_path.parent.mkdir(exist_ok=True)

    truth_model = project_root / "output/spot_truth.tomog"
    truth_model.parent.mkdir(exist_ok=True)

    # Export the current three-spot model as a geomodel file
    sim.export_to_geomodel(str(truth_model), phase=0.0)

    linepath = project_root / "input/lines.txt"

    print(f"Writing parameter file: {param_file_path}")
    with open(param_file_path, "w") as f:
        f.write("# Forward tomography parameter file\n")
        f.write("60.0 40.0 1.0 -0.05\n")  # inclination vsini period pOmega
        f.write("1.0 0.8 0.0 3.0 0\n")  # mass r0 vmax r_out occ
        f.write("10\n")  # nrings
        f.write("C 1.5 5\n")  # target form
        f.write("1e-3\n")
        f.write("1 1.0 0 0\n")
        # Use the just-exported three-spot geomodel as initial model
        f.write(f"1 {truth_model}\n")
        f.write("1 1 1 1 0\n")  # fit flags
        f.write(f"65000 {linepath}\n")
        f.write("-400 400 lsd_pol polOut=V\n")
        f.write("2450000.0\n")

        # Observation file list (using .lsd files written by SpotSimulator)
        for i, phase in enumerate(phases):
            fname = project_root / "input" / "spot_forward_obs" / f"obs_phase_{phase:.3f}.lsd"
            hjd = 2450000 + phase
            f.write(f"{fname} {hjd:.5f} 0.0 V\n")

    print("Parameter file created.")


if __name__ == "__main__":
    main()
