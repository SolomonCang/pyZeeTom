"""Example 04: Visualize geomodel (magnetic / brightness maps)

This script demonstrates how to load a `.tomog` geometric model and
visualize it using `utils.visualize_geomodel`.

By default it loads the MEM inversion result created by
`examples/03_inversion_tomography.py`:

- `output/inverse_test/mem_inversion_model.tomog`

How to run (from project root)::

	python examples/04_visualize_geomodel.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root (parent of examples) is on sys.path so that
# `pyzeetom`, `core`, and `utils` can be imported when running this
# script directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

from core.disk_geometry_integrator import VelspaceDiskIntegrator
from utils.visualize_geomodel import PARAM_CONFIG, plot_geomodel_contour


def get_project_root() -> Path:
    return project_root


def main() -> None:
    project_root = get_project_root()

    #  forward 
    # 1)  forward  tomog 
    #    model_file = project_root / "output" / "forward_test" / "geomodel_phase_00.tomog"
    # 2)  MEM 
    model_file = project_root / "output" / "inverse_test" / "mem_inversion_model.tomog"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Geomodel file not found: {model_file}. "
            "Have you run examples/03_inversion_tomography.py?")

    print(f"Loading geomodel: {model_file}")
    geom, meta, table = VelspaceDiskIntegrator.read_geomodel(str(model_file))

    # 
    plot_geomodel_contour(
        geom=geom,
        meta=meta,
        table=table,
        config=PARAM_CONFIG,
    )


if __name__ == "__main__":
    main()
