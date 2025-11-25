#!/usr/bin/env python
"""Dynamic spectrum visualization tool.

Features:
1. Adaptive search for spectrum files in .lsd, .s, .spec formats.
2. Force use of SpecIO to read spectra, supporting full Stokes parameters.
3. Supports two plotting modes:
   - image: Dynamic spectrum (2D colormap)
   - stacked: Stacked line plot (Waterfall plot), arranged by phase
4. Stokes display logic:
   - stokes='I': Single plot showing Stokes I
   - stokes='V'/'Q'/'U': Dual plots, left showing Stokes I, right showing Stokes V/Q/U
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path (assuming script is in tools/ or similar subdirectory)
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

# Import custom modules
from utils.dynamic_spectrum import IrregularDynamicSpectrum
from core.mainFuncs import readParamsTomog, compute_phase_from_jd
from core.SpecIO import loadObsProfile, ObservationProfile

# ==============================================================================
#                                 Configuration Area
# ==============================================================================
CONFIG = {
    # --- Data Input Configuration ---
    'params_file': 'input/params_inverse_test.txt',  # Parameter file path
    # 'model_dir': 'output/inverse_test',  # Directory containing spectrum files
    'model_dir': 'input/inSpec',
    # Specify file list (optional)
    # None: Auto search (phase001.lsd, .s, .spec etc.)
    'file_list': None,

    # Select Stokes parameter to display
    # 'I': Plot I only
    # 'V', 'Q', 'U': Plot two panels, left I, right V/Q/U
    'stokes': 'V',
    'file_type': 'auto',  # File type hint passed to SpecIO

    # --- Plotting Mode Configuration ---
    # 'image':   Plot dynamic spectrum (colormap/heatmap)
    # 'stacked': Plot stacked line plot (all spectra on one plot, offset by phase)
    'plot_mode': 'stacked',

    # --- Output Configuration ---
    'out_file': None,  # Output path (e.g. 'plot.png'), None for popup window

    # --- 'image' Mode Specific Configuration ---
    'cmap': 'RdBu_r',  # Colormap
    'vmin': 0.98,  # Color lower limit for Stokes I
    'vmax': 1.02,  # Color upper limit for Stokes I
    # Polarization component color range usually needs to be smaller
    'vmin_pol': -0.001,
    'vmax_pol': 0.001,

    # --- 'stacked' Mode Specific Configuration ---
    # Scaling factor: Controls the height of spectral line fluctuations.
    # Stokes I formula: Y = Phase + (Intensity - 1.0) * stack_scale
    # Stokes P formula: Y = Phase + Intensity * stack_scale * pol_scale_mult
    'stack_scale': 1.0,
    # Extra magnification for polarization components (V usually 100, Q/U may need larger, e.g. 10000)
    'pol_scale_mult': 1,
    'line_color': 'black',  # Line color
    'line_width': 0.6,  # Line width

    # --- Data Processing Configuration ---
    'remove_baseline':
    False,  # Whether to remove baseline (subtract mean profile)
    'align_continuum':
    False,  # Whether to force align continuum (align edges to 1.0)
}
# ==============================================================================


def find_file_for_index(base_dir, index):
    """Adaptive search for spectrum file corresponding to index."""
    base_dir = Path(base_dir)

    # Define possible search patterns (glob patterns)
    # Supports 3 or 4 digit numbers, supports extra suffixes in filename
    patterns = [
        f"phase{index:03d}*",  # phase001...
        f"phase_{index:03d}*",  # phase_001...
        f"phase{index:04d}*",  # phase0001...
        f"phase_{index:04d}*",  # phase_0001...
        f"spec{index:03d}*",  # spec001...
        f"spec_{index:03d}*",  # spec_001...
        f"{index:03d}*",  # 001...
    ]

    # Possible extensions
    extensions = {'.lsd', '.s', '.spec', '.dat', '.txt'}

    for pattern in patterns:
        # Use glob to search for files matching prefix
        candidates = list(base_dir.glob(pattern))

        # Filter valid extensions
        valid_candidates = [
            c for c in candidates
            if c.suffix.lower() in extensions and c.is_file()
        ]

        if valid_candidates:
            # If found, return the first one (sort to ensure determinism)
            valid_candidates.sort()
            return valid_candidates[0]

    return None


def load_model_spectra(model_dir,
                       params_file,
                       file_list=None,
                       file_type='auto'):
    """
    Load spectrum data.
    
    Returns:
        times (np.array): Phase array
        obs_list (list[ObservationProfile]): List of spectrum objects read by SpecIO
    """
    model_dir = Path(model_dir)

    # 1. Read parameter file to get phase information
    if not Path(params_file).exists():
        raise FileNotFoundError(f"Params file not found: {params_file}")

    params = readParamsTomog(params_file, verbose=0)
    phases = compute_phase_from_jd(params.jDates, params.jDateRef,
                                   params.period)
    num_obs = params.numObs

    print(f"  Params info: {num_obs} observations")
    print(f"  Phase range: {phases[0]:.3f} - {phases[-1]:.3f}")

    # 2. Prepare file list
    target_files = []
    if file_list is not None:
        limit = min(len(file_list), num_obs)
        for i in range(limit):
            target_files.append(Path(file_list[i]))
    else:
        print(f"  Auto-searching files in {model_dir} ...")
        for i in range(num_obs):
            f_path = find_file_for_index(model_dir, i)
            if f_path is None:
                print(
                    f"  Warning: Could not find file for index {i} (Phase {phases[i]:.3f})"
                )
            target_files.append(f_path)

    # 3. Use SpecIO to read data
    times = []
    obs_list = []

    for i, f_path in enumerate(target_files):
        if f_path is None: continue
        if not f_path.exists():
            print(f"  Warning: File does not exist: {f_path}")
            continue

        # Force use of SpecIO to read
        try:
            # Note: loadObsProfile automatically handles I, V, Q, U columns
            obs = loadObsProfile(str(f_path), file_type=file_type)
        except Exception as e:
            print(f"  Error loading {f_path.name}: {e}")
            obs = None

        if obs is not None:
            times.append(phases[i])
            obs_list.append(obs)
        else:
            print(f"  Warning: Failed to parse {f_path.name} with SpecIO.")

    if len(times) == 0:
        raise FileNotFoundError("No valid spectra loaded.")

    # Sort by phase
    order = np.argsort(times)
    sorted_times = np.array([times[i] for i in order])
    sorted_obs = [obs_list[i] for i in order]

    return sorted_times, sorted_obs


def get_stokes_data(obs_list, stokes_char):
    """Extract specific Stokes component array list from obs_list."""
    data_list = []
    stokes_char = stokes_char.upper()

    for obs in obs_list:
        if stokes_char == 'I':
            data_list.append(obs.specI)
        elif stokes_char == 'V':
            data_list.append(
                obs.specV if obs.hasV else np.zeros_like(obs.specI))
        elif stokes_char == 'Q':
            data_list.append(
                obs.specQ if obs.hasQ else np.zeros_like(obs.specI))
        elif stokes_char == 'U':
            data_list.append(
                obs.specU if obs.hasU else np.zeros_like(obs.specI))
        else:
            # Default fallback to I
            data_list.append(obs.specI)
    return data_list


def main():
    # Read from CONFIG
    params_file = CONFIG['params_file']
    model_dir = CONFIG['model_dir']
    file_list = CONFIG['file_list']
    stokes_cfg = CONFIG['stokes'].upper(
    )  # User configured Stokes ('I', 'V', 'Q', 'U')
    plot_mode = CONFIG['plot_mode']
    out_file = CONFIG['out_file']
    remove_baseline = CONFIG['remove_baseline']

    print("Initializing Spectrum Plotter...")

    try:
        times, obs_list = load_model_spectra(model_dir=model_dir,
                                             params_file=params_file,
                                             file_list=file_list,
                                             file_type=CONFIG['file_type'])
        print(f"✓ Successfully loaded {len(times)} spectra")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    # Extract wavelength/velocity axis (assume all files have consistent grid, take first one)
    xs_list = [obs.wl for obs in obs_list]
    x_sample = xs_list[0]

    # Auto-determine X axis label
    if np.mean(x_sample) < 2000:
        xlabel = 'Velocity (km/s)' if np.max(
            np.abs(x_sample)) < 1000 else 'Wavelength (nm)'
    else:
        xlabel = 'Wavelength (Å)'

    # Prepare data: Always need I, if stokes_cfg is not I, also need Pol
    intensities_I = get_stokes_data(obs_list, 'I')
    intensities_Pol = None

    if stokes_cfg in ['V', 'Q', 'U']:
        intensities_Pol = get_stokes_data(obs_list, stokes_cfg)
        # Check signal strength and suggest scaling
        max_pol = np.max(np.abs(intensities_Pol)) if intensities_Pol else 0.0
        print(f"  Max {stokes_cfg} amplitude: {max_pol:.3e}")

        current_mult = CONFIG.get('pol_scale_mult', 100.0)
        if max_pol > 0 and max_pol * current_mult * CONFIG[
                'stack_scale'] < 0.005:
            suggested_mult = 0.1 / max_pol
            print(
                f"  Warning: {stokes_cfg} signal is very weak. Auto-adjusting pol_scale_mult to {suggested_mult:.1e} for visibility."
            )
            CONFIG['pol_scale_mult'] = suggested_mult

    # Remove baseline (subtract mean profile)
    if remove_baseline:
        print("Removing baseline (subtracting mean profile)...")
        # Process I
        mean_I = np.mean(intensities_I, axis=0)
        intensities_I = [spec - mean_I + 1.0
                         for spec in intensities_I]  # Keep I around 1.0

        # Process Pol (if exists)
        if intensities_Pol is not None:
            mean_Pol = np.mean(intensities_Pol, axis=0)
            intensities_Pol = [spec - mean_Pol for spec in intensities_Pol
                               ]  # Keep Pol around 0.0

    # Continuum alignment (force edges to 1.0)
    if CONFIG.get('align_continuum', False) and not remove_baseline:
        print("Aligning continuum (shifting edges to 1.0)...")
        # Calculate edge mean for each spectrum (take first 5 and last 5 points)
        # Assume edges are continuum
        new_I = []
        for spec in intensities_I:
            edge_val = (np.mean(spec[:5]) + np.mean(spec[-5:])) / 2.0
            # Shift so edge is at 1.0
            new_I.append(spec - edge_val + 1.0)
        intensities_I = new_I

    # ==========================================================================
    # Plot Initialization
    # ==========================================================================
    # If I, one plot; if V/Q/U, two plots (1 row 2 columns)
    # Restore sharey=True to ensure phase alignment
    if stokes_cfg == 'I':
        fig, ax_main = plt.subplots(figsize=(6, 10))
        axes = [ax_main]
        data_pairs = [('I', intensities_I)]
    else:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 10), sharey=True)
        axes = [ax_l, ax_r]
        data_pairs = [('I', intensities_I), (stokes_cfg, intensities_Pol)]

    scale = CONFIG['stack_scale']

    # ==========================================================================
    # Loop to plot each subplot (Left I, Right Pol)
    # ==========================================================================
    for ax_idx, ax in enumerate(axes):
        label, current_intensities = data_pairs[ax_idx]

        # --- Mode 1: Dynamic Spectrum (Image) ---
        if plot_mode == 'image':
            # Dynamic spectrum requires regular grid handling, calling utility class here
            dynspec = IrregularDynamicSpectrum(times, xs_list,
                                               current_intensities)

            # Color range distinction
            if label == 'I':
                vmin, vmax = CONFIG['vmin'], CONFIG['vmax']
                cmap = CONFIG['cmap']
            else:
                vmin, vmax = CONFIG['vmin_pol'], CONFIG['vmax_pol']
                cmap = 'RdBu_r'  # Polarization usually uses Red-Blue

            # Note: IrregularDynamicSpectrum.plot creates a new figure, here we need to manually plot on ax
            # We use pcolormesh directly
            # For simplicity, assume consistent grid, build 2D array
            img_data = np.array(current_intensities)
            # img_data shape: (n_phases, n_pixels)

            # Construct grid
            X, Y = np.meshgrid(x_sample, times)

            im = ax.pcolormesh(X,
                               Y,
                               img_data,
                               cmap=cmap,
                               vmin=vmin,
                               vmax=vmax,
                               shading='auto')
            if ax_idx == 1 or len(axes) == 1:
                plt.colorbar(im, ax=ax, label='Intensity')

            ax.set_title(f'Dynamic Spectrum (Stokes {label})')

        # --- Mode 2: Stacked Lines (Waterfall) ---
        elif plot_mode == 'stacked':
            line_color = CONFIG['line_color']
            lw = CONFIG['line_width']

            y_data_min = np.inf
            y_data_max = -np.inf

            for i in range(len(times)):
                phase = times[i]
                x = xs_list[i]
                y = current_intensities[i]

                # Core formula
                if label == 'I':
                    # Stokes I is usually normalized to 1, subtract 1 to fluctuate around 0, then add phase
                    y_plot = phase + (y - 1.0) * scale
                else:
                    # Stokes V/Q/U usually fluctuate around 0, add directly
                    pol_mult = CONFIG.get('pol_scale_mult', 100.0)
                    y_plot = phase + (y) * pol_mult * scale

                # Update data range
                if len(y_plot) > 0:
                    y_data_min = min(y_data_min, np.min(y_plot))
                    y_data_max = max(y_data_max, np.max(y_plot))

                ax.plot(x, y_plot, color=line_color, linewidth=lw, alpha=0.8)

            ax.set_title(f'Stacked (Stokes {label}, Scale={scale})')

            # Dynamically set Y axis range
            # Base range is phase range
            base_min = times[0] - 0.05
            base_max = times[-1] + 0.05

            # Combine data range (prevent data from going out of view)
            if np.isfinite(y_data_min) and np.isfinite(y_data_max):
                final_min = min(base_min, y_data_min - 0.05)
                final_max = max(base_max, y_data_max + 0.05)
            else:
                final_min, final_max = base_min, base_max

            ax.set_ylim(final_min, final_max)
            ax.set_xlim(x_sample[0], x_sample[-1])

        # Common axis labels
        ax.set_xlabel(xlabel)
        if ax_idx == 0:
            ax.set_ylabel('Rotation Phase')

    plt.tight_layout()

    # Output or show
    if out_file:
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved image to: {out_file}")
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
