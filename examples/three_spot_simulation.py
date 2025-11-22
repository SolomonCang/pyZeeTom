import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.spot_simulator import SpotSimulator, SpotConfig
from core.grid_tom import diskGrid
from pyzeetom import tomography
import core.SpecIO as SpecIO


# =============================================================================
# Simulation Configuration
# =============================================================================
class SimConfig:
    """Configuration parameters for the 3-spot simulation."""

    # --- Directories (relative to project root) ---
    OUTPUT_DIR_NAME = 'spot_forward'
    INPUT_OBS_DIR_NAME = 'spot_forward_obs'

    # --- Star & Disk Parameters ---
    INCLINATION = 60.0  # degrees
    VSINI = 40.0  # km/s (Increased for better Doppler resolution)
    PERIOD = 1.0  # days
    P_OMEGA = -0.05  # differential rotation
    RADIUS = 0.8  # stellar radius (arbitrary units, used as r0)
    R_OUT = 4.0  # disk outer radius (Restored to 2.0)
    N_RINGS = 80  # grid resolution

    # --- Spot 1 (High latitude) ---
    SPOT1_R_FACTOR = 1.5  # Multiplier for RADIUS -> r=0.88
    SPOT1_PHI = 30.0  # degrees
    SPOT1_AMP = 1.2
    SPOT1_SIZE = 0.2  # Slightly smaller for sharper peaks
    SPOT1_BLOS = 500.0  # Increased field
    SPOT1_BPERP = 0.0
    SPOT1_CHI = 45.0  # degrees

    # --- Spot 2 (Equatorial) ---
    SPOT2_R_FACTOR = 3.3  # Multiplier for RADIUS -> r=1.04
    SPOT2_PHI = 100.0  # degrees (Changed from 150 to avoid overlap with Spot 1)
    SPOT2_AMP = 2.5
    SPOT2_SIZE = 0.2
    SPOT2_BLOS = -300.0  # Increased field
    SPOT2_BPERP = 0.0
    SPOT2_CHI = 90.0  # degrees

    # --- Spot 3 (Far disk) ---
    SPOT3_R_FACTOR = 4.4  # Multiplier for RADIUS -> r=1.44
    SPOT3_PHI = 270.0  # degrees
    SPOT3_AMP = 1.5
    SPOT3_SIZE = 0.3
    SPOT3_BLOS = 500.0
    SPOT3_BPERP = 0.0
    SPOT3_CHI = 0.0  # degrees

    # --- Observation Parameters ---
    PHASES = [
        0.0, 0.05, 0.12, 0.18, 0.25, 0.33, 0.41, 0.48, 0.55, 0.65, 0.72, 0.80,
        0.88, 0.92, 0.98
    ]
    VEL_MIN = -400.0
    VEL_MAX = 400.0
    VEL_POINTS = 401
    JDATE_REF = 2450000.0

    # --- Noise ---
    SNR = 1000.0
    RANDOM_SEED = 42


def run_simulation():
    # 1. Setup directories
    output_dir = project_root / 'output' / SimConfig.OUTPUT_DIR_NAME
    input_obs_dir = project_root / 'input' / SimConfig.INPUT_OBS_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    input_obs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Input obs directory: {input_obs_dir}")

    # 2. Initialize Grid and SpotSimulator
    # Parameters from SimConfig
    inclination = SimConfig.INCLINATION
    vsini = SimConfig.VSINI
    period = SimConfig.PERIOD
    pOmega = SimConfig.P_OMEGA
    radius = SimConfig.RADIUS
    r_out = SimConfig.R_OUT
    nRings = SimConfig.N_RINGS

    # Create grid
    # Note: diskGrid needs Vmax. If we use r_out mode, we calculate Vmax.
    inc_rad = np.deg2rad(inclination)
    # vsini is the projected rotational velocity at the stellar equator (radius)
    # The velocity at any radius r is: v(r) = vel_eq_star * (r/radius)^(pOmega+1)
    # So at r_out: v_max = vel_eq_star * (r_out/radius)^(pOmega+1)

    v_max_disk = vsini * (r_out / radius)

    # Ensure VEL_MAX covers the disk velocity range, but don't force them to be equal
    if SimConfig.VEL_MAX < v_max_disk:
        print(
            f"Warning: VEL_MAX ({SimConfig.VEL_MAX}) is smaller than projected disk max velocity ({v_max_disk * np.sin(inc_rad):.2f})"
        )

    print(f"Initializing grid with Vmax_disk={v_max_disk:.2f} km/s")
    # We pass r_out explicitly to define the physical size of the disk
    grid = diskGrid(nr=nRings, r_in=0.0, r_out=r_out)

    simulator = SpotSimulator(
        grid=grid,
        inclination_rad=inc_rad,
        phi0=0.0,
        pOmega=pOmega,
        r0_rot=radius,  # radius is used as r0 for differential rotation
        period_days=period)

    # 3. Configure 3 Emission Spots
    # Spot 1
    spot1 = SpotConfig(r=SimConfig.SPOT1_R_FACTOR * radius,
                       phi=np.deg2rad(SimConfig.SPOT1_PHI),
                       amplitude=SimConfig.SPOT1_AMP,
                       spot_type='emission',
                       radius=SimConfig.SPOT1_SIZE,
                       B_los=SimConfig.SPOT1_BLOS,
                       B_perp=SimConfig.SPOT1_BPERP,
                       chi=np.deg2rad(SimConfig.SPOT1_CHI))

    # Spot 2
    spot2 = SpotConfig(r=SimConfig.SPOT2_R_FACTOR * radius,
                       phi=np.deg2rad(SimConfig.SPOT2_PHI),
                       amplitude=SimConfig.SPOT2_AMP,
                       spot_type='emission',
                       radius=SimConfig.SPOT2_SIZE,
                       B_los=SimConfig.SPOT2_BLOS,
                       B_perp=SimConfig.SPOT2_BPERP,
                       chi=np.deg2rad(SimConfig.SPOT2_CHI))

    # Spot 3
    spot3 = SpotConfig(r=SimConfig.SPOT3_R_FACTOR * radius,
                       phi=np.deg2rad(SimConfig.SPOT3_PHI),
                       amplitude=SimConfig.SPOT3_AMP,
                       spot_type='emission',
                       radius=SimConfig.SPOT3_SIZE,
                       B_los=SimConfig.SPOT3_BLOS,
                       B_perp=SimConfig.SPOT3_BPERP,
                       chi=np.deg2rad(SimConfig.SPOT3_CHI))

    simulator.add_spots([spot1, spot2, spot3])
    print("Added 3 emission spots.")

    # 4. Export "Truth" Model at Phase 0
    model_path = output_dir / 'truth_model.tomog'
    simulator.export_to_geomodel(str(model_path), phase=0.0)
    print(f"Exported truth model to {model_path}")

    # 5. Generate Dummy Observations
    phases = np.array(SimConfig.PHASES)

    # Create a dummy LSD profile for the velocity grid
    vel_grid = np.linspace(SimConfig.VEL_MIN, SimConfig.VEL_MAX,
                           SimConfig.VEL_POINTS)
    dummy_spec = np.ones_like(vel_grid)
    dummy_sig = np.ones_like(vel_grid) * 0.001
    dummy_pol = np.zeros_like(vel_grid)

    obs_filenames = []
    jdates = []

    jdate_ref = SimConfig.JDATE_REF

    for i, ph in enumerate(phases):
        fname = f"obs_phase_{ph:.3f}.lsd"
        fpath = input_obs_dir / fname

        # Write dummy LSD file
        # Format: RV Int sigma_int V sigma_pol Null1 sigma_null1
        with open(fpath, 'w') as f:
            f.write("RV Int sigma_int V sigma_pol Null1 sigma_null1\n")
            for v, s, sig, p in zip(vel_grid, dummy_spec, dummy_sig,
                                    dummy_pol):
                f.write(
                    f"{v:.4f} {s:.4f} {sig:.4f} {p:.4f} {sig:.4f} 0.0 {sig:.4f}\n"
                )

        obs_filenames.append(str(fpath))
        # Calculate JDate: phase = (jd - jd_ref) / period => jd = phase * period + jd_ref
        jd = ph * period + jdate_ref
        jdates.append(jd)

    print(f"Generated {len(phases)} dummy observation files.")

    # 6. Create Parameter File
    param_file_path = project_root / 'input' / 'params_spot_forward.txt'

    with open(param_file_path, 'w') as f:
        f.write("# Spot Simulation Forward Model Parameters\n")
        f.write(
            f"{inclination} {vsini} {period} {pOmega}  # inclination vsini period pOmega\n"
        )
        f.write(
            f"1.0 {radius} {v_max_disk:.2f} {r_out} 0  # mass radius Vmax r_out enable_occultation\n"
        )
        f.write(f"{nRings}  # nRingsStellarGrid\n")
        f.write("C 1.0 0  # targetForm targetValue numIterations\n")
        f.write("1e-3  # test_aim\n")
        f.write("1 1.0 1 1  # lineAmpConst k_QU enableV enableQU\n")
        f.write(f"1 {model_path}  # initTomogFile initModelPath\n")
        f.write("0 1.0 1.0  # fitBri chiScaleI brightEntScale\n")
        f.write("1 1.0 2.0  # fEntropyBright defaultBright maximumBright\n")
        f.write("0 0 0 # 9 deprecated\n")
        f.write("\n")
        f.write("0  # estimateStrenght\n")
        f.write("65000 input/lines.txt  # spectralResolution lineParamFile\n")
        f.write(
            f"{SimConfig.VEL_MIN} {SimConfig.VEL_MAX} lsd_pol polOut=V  # velStart velEnd obsFileType\n"
        )
        f.write(f"{jdate_ref}  # jDateRef\n")

        # Observation files
        for fname, jd in zip(obs_filenames, jdates):
            # Path relative to project root or absolute
            # params_tomog.txt usually uses relative paths
            rel_path = os.path.relpath(fname, project_root)
            f.write(f"{rel_path} {jd:.5f} 0.0 V\n")

    print(f"Created parameter file: {param_file_path}")

    # 7. Run Forward Tomography
    print("Running forward tomography...")
    results = tomography.forward_tomography(str(param_file_path),
                                            verbose=1,
                                            output_dir=str(output_dir))

    print(f"Forward tomography completed. Generated {len(results)} spectra.")

    # 8. Add Noise
    snr = SimConfig.SNR
    print(f"Adding noise (S/N = {snr}) to generated spectra...")

    # Find generated files
    import glob
    spec_files = glob.glob(str(output_dir / "phase_*.spec"))

    np.random.seed(SimConfig.RANDOM_SEED)

    for fpath in spec_files:
        # Use SpecIO.loadObsProfile to read the spectrum correctly
        # The generated file might be in a format that loadObsProfile can auto-detect (e.g. spec_pol with 6 cols)
        obs = SpecIO.loadObsProfile(fpath)

        if obs is None:
            print(f"Failed to load {fpath}")
            continue

        # Extract data
        # Note: If loaded as spec_pol, obs.wl holds the x-axis (RV in this case)
        rv = obs.wl
        spec_i = obs.specI
        spec_v = obs.specV
        spec_q = obs.specQ
        spec_u = obs.specU

        # Debug: Check signal strength before noise
        v_amp = np.max(np.abs(spec_v))
        if v_amp == 0:
            print(
                f"Warning: Phase {fpath} has ZERO Stokes V signal before noise!"
            )
        else:
            print(f"Phase {fpath} V amplitude: {v_amp:.2e}")

        # Add noise
        # Signal amplitude is roughly 1.0 (continuum).
        # Noise sigma = 1.0 / SNR
        sigma = 1.0 / snr

        # Generate noise
        noise_I = np.random.normal(0, sigma, spec_i.shape)
        noise_V = np.random.normal(0, sigma, spec_v.shape)
        noise_Q = np.random.normal(0, sigma, spec_q.shape)
        noise_U = np.random.normal(0, sigma, spec_u.shape)

        # Apply noise
        new_I = spec_i + noise_I
        new_V = spec_v + noise_V
        new_Q = spec_q + noise_Q
        new_U = spec_u + noise_U

        # Update sigmas
        new_sig = np.ones_like(spec_i) * sigma

        # Write back (overwrite)
        # We use 'lsd_pol' hint to write 7 columns, which SpecIO CAN read later as lsd_pol
        SpecIO.write_model_spectrum(fpath,
                                    rv,
                                    new_I,
                                    V=new_V,
                                    Q=new_Q,
                                    U=new_U,
                                    sigmaI=new_sig,
                                    fmt='lsd',
                                    pol_channel='V',
                                    file_type_hint='lsd_pol')

    print("Noisy spectra saved (overwritten).")


if __name__ == "__main__":
    run_simulation()
