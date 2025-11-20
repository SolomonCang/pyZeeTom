import numpy as np
from core.grid_tom import diskGrid
from utils.spot_simulator import SpotSimulator, SpotConfig


def debug_spot():
    # Setup grid similar to the one in the file
    nr = 40
    r_in = 0.5
    r_out = 4.0
    grid = diskGrid(nr=nr, r_in=r_in, r_out=r_out, verbose=1)

    # Setup simulator
    sim = SpotSimulator(grid,
                        inclination_rad=np.deg2rad(60.0),
                        phi0=0.0,
                        pOmega=-0.05,
                        r0_rot=1.0,
                        period_days=1.0)

    # Add spot
    spot = SpotConfig(r=0.6,
                      phi=0.0,
                      amplitude=2.0,
                      spot_type='emission',
                      radius=0.1,
                      B_los=1500.0,
                      B_perp=500.0,
                      chi=0.5)
    sim.add_spot(spot)

    # Apply spots
    sim.apply_spots_to_grid(phase=0.0)

    # Check max B_los
    max_blos = np.max(sim.B_los_map)
    print(f"Max B_los: {max_blos}")

    # Check value at a specific pixel near the spot
    # Find pixel closest to r=0.6, phi=0
    dist = np.sqrt((grid.r - 0.6)**2 + (grid.phi - 0.0)**2)
    idx = np.argmin(dist)
    print(f"Pixel {idx}: r={grid.r[idx]:.4f}, phi={grid.phi[idx]:.4f}")
    print(f"B_los at pixel {idx}: {sim.B_los_map[idx]}")

    # Check weights manually
    dr = np.sqrt((grid.r[idx] - 0.6)**2)
    dphi = grid.phi[idx] - 0.0
    sigma_r = 0.1 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_phi = 0.1 / (grid.r[idx] + 1e-10)

    w_r = np.exp(-0.5 * (dr / sigma_r)**2)
    w_phi = np.exp(-0.5 * (dphi / sigma_phi)**2)
    print(
        f"Manual weight: w_r={w_r:.4f}, w_phi={w_phi:.4f}, total={w_r*w_phi:.4f}"
    )
    print(f"Expected B_los: {1500 * w_r * w_phi}")


if __name__ == "__main__":
    debug_spot()
