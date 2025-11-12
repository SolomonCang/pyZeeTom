"""tomography.py - root-level thin entrypoint (simplified)

This file delegates to the packaged entry `pyzeetom.tomography.main` so
running from the workspace root works as well.
"""

from typing import List, Tuple
import numpy as np

from pyzeetom.tomography import main as run


def main() -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return run()


if __name__ == "__main__":
    main()
