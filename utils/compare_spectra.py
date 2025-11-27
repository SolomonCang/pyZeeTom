#!/usr/bin/env python
"""Spectrum comparison tool.

Features:
1. Compare input observed spectra and output model spectra.
2. Support two plotting modes:
   - image: Plot residual dynamic spectrum (Obs - Model)
   - stacked: Stacked line plot, showing both observation and model
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

# Import custom modules
from utils.dynamic_spectrum import IrregularDynamicSpectrum
from core.mainFuncs import readParamsTomog, compute_phase_from_jd  # noqa: E402
from core.SpecIO import loadObsProfile  # noqa: E402

# ==============================================================================
#                                 Configuration Area
# ==============================================================================
CONFIG = {
    'params_file': 'input/intomog_ap149_update.txt',
    'input_dir':
    '/Users/tianqi/Documents/Project/mag2acc/ap149/Ha_spec',  # Observation data directory
    'output_dir': 'output/ap149_test',  # Model data directory

    # Select Stokes parameter to display: 'I', 'V', 'Q', 'U'
    'stokes': 'V',
    'file_type': 'auto',

    # 'image':   Plot residual dynamic spectrum (Obs - Model)
    # 'stacked': Plot stacked line plot (Obs and Model in same plot)
    'plot_mode': 'image',
    'out_file': None,

    # --- Image Mode Config (Residuals) ---
    'cmap': 'RdBu_r',
    'vmin_res': -0.1,  # Residual color range (adjust based on signal strength)
    'vmax_res': 0.1,
    'vmin_I': 0.85,
    'vmax_I': 1.15,
    'vmin_pol': -0.005,
    'vmax_pol': 0.005,

    # --- Stacked Mode Config ---
    'stack_scale': 0.5,
    'pol_scale_mult': 1,
    'obs_color': 'black',
    'model_color': 'red',
    'line_width': 0.8,
    'remove_baseline': False,
    'align_continuum': False,

    # --- Phase Configuration ---
    'fold_phase': False,  # Whether to fold phases to 0-1 range and re-sort
}
# ==============================================================================


def find_file_for_index(base_dir, index):
    """Adaptively search for spectrum file corresponding to index."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None

    # Define possible search patterns (glob patterns)
    patterns = [
        f"phase{index:03d}*",  # phase001...
        f"phase_{index:03d}*",  # phase_001...
        f"phase{index:04d}*",  # phase0001...
        f"phase_{index:04d}*",  # phase_0001...
        f"model_phase_{index}*",  # model_phase_0...
        f"model_phase_{index:03d}*",  # model_phase_000...
        f"spec{index:03d}*",  # spec001...
        f"spec_{index:03d}*",  # spec_001...
        f"{index:03d}*",  # 001...
        f"*_{index:03d}*",  # ap149_001.s
    ]

    # Possible extensions
    extensions = {'.lsd', '.s', '.spec', '.dat', '.txt'}

    for pattern in patterns:
        candidates = list(base_dir.glob(pattern))
        valid_candidates = [
            c for c in candidates
            if c.suffix.lower() in extensions and c.is_file()
        ]

        if valid_candidates:
            valid_candidates.sort()
            return valid_candidates[0]

    return None


def load_spectra_set(data_dir, params_file, label="Data"):
    """
    Load a set of spectrum data.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Warning: Directory not found: {data_dir}")
        return [], []

    # 1. Read params file to get phase info
    if not Path(params_file).exists():
        raise FileNotFoundError(f"Params file not found: {params_file}")

    params = readParamsTomog(params_file, verbose=0)
    phases = compute_phase_from_jd(params.jDates, params.jDateRef,
                                   params.period)
    num_obs = params.numObs

    print(f"[{label}] Searching files in {data_dir} ...")

    # Sort phases first to match dynamic_spec_plot logic
    sorted_indices = np.argsort(phases)
    sorted_phases = phases[sorted_indices]

    times = []
    obs_list = []

    # We iterate through the sorted phases, but we need to find the file corresponding to the ORIGINAL index
    # Wait, dynamic_spec_plot logic was:
    # 1. Sort phases.
    # 2. Assign file index 0 to smallest phase, 1 to next, etc.
    # This assumes files are named phase000, phase001... corresponding to time order.

    for i in range(num_obs):
        # i is the index in the sorted list (0 = earliest phase)
        # We assume file naming follows this order
        f_path = find_file_for_index(data_dir, i)

        if f_path is None:
            # Fallback: maybe files are indexed by original input order?
            # If so, we should use sorted_indices[i] to find the file?
            # But usually output files are generated in phase order or input order.
            # If input order is random, output usually follows input.
            # If input order is random, phase000 might be phase=0.8.
            # But dynamic_spec_plot assumes phase000 is the earliest phase.
            # Let's stick to dynamic_spec_plot logic: index i corresponds to i-th sorted phase.
            pass

        if f_path is None or not f_path.exists():
            print(
                f"  Warning: Missing file for index {i} (Phase {sorted_phases[i]:.3f})"
            )
            continue

        try:
            obs = loadObsProfile(str(f_path), file_type=CONFIG['file_type'])
            times.append(sorted_phases[i])
            obs_list.append(obs)
        except Exception as e:
            print(f"  Error loading {f_path.name}: {e}")

    return np.array(times), obs_list


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
            data_list.append(obs.specI)
    return data_list


def interpolate_data_list(x_target, x_list, data_list):
    """Interpolate all spectra in data_list to x_target grid."""
    new_data = []
    for x, y in zip(x_list, data_list):
        if len(x) != len(x_target) or not np.allclose(x, x_target):
            new_data.append(np.interp(x_target, x, y))
        else:
            new_data.append(y)
    return new_data


def main():
    params_file = CONFIG['params_file']
    input_dir = CONFIG['input_dir']
    output_dir = CONFIG['output_dir']
    stokes_cfg = CONFIG['stokes'].upper()
    plot_mode = CONFIG['plot_mode']
    out_file = CONFIG['out_file']

    print("Initializing Spectrum Comparison...")

    # 1. Load Input (Obs)
    times_in, obs_in = load_spectra_set(input_dir, params_file, label="Input")
    if len(times_in) == 0:
        print("Error: No input spectra loaded.")
        return 1

    # 2. Load Output (Model)
    times_out, obs_out = load_spectra_set(output_dir,
                                          params_file,
                                          label="Output")
    if len(times_out) == 0:
        print("Error: No output spectra loaded.")
        return 1

    # 3. Validate alignment
    if len(times_in) != len(times_out):
        print(
            f"Warning: Mismatch in number of spectra (In: {len(times_in)}, Out: {len(times_out)})"
        )
        # Find common phases? For now, just truncate to min
        n_common = min(len(times_in), len(times_out))
        times_in = times_in[:n_common]
        obs_in = obs_in[:n_common]
        times_out = times_out[:n_common]
        obs_out = obs_out[:n_common]

    # Check phase alignment
    if not np.allclose(times_in, times_out, atol=1e-4):
        print("Warning: Phases do not match exactly between input and output!")

    # Apply phase folding if requested
    if CONFIG.get('fold_phase', False):
        print("Folding phases to 0-1 range...")

        # Fold and sort Input
        times_in = np.array(times_in) % 1.0
        sort_idx_in = np.argsort(times_in)
        times_in = times_in[sort_idx_in]
        obs_in = [obs_in[i] for i in sort_idx_in]

        # Fold and sort Output
        times_out = np.array(times_out) % 1.0
        sort_idx_out = np.argsort(times_out)
        times_out = times_out[sort_idx_out]
        obs_out = [obs_out[i] for i in sort_idx_out]

        print("✓ Re-sorted spectra by folded phase")

    # 4. Prepare Data
    xs_list_in = [obs.wl for obs in obs_in]
    xs_list_out = [obs.wl for obs in obs_out]
    x_sample = xs_list_in[0]

    # Auto X-label
    if np.mean(x_sample) < 2000:
        xlabel = 'Velocity (km/s)' if np.max(
            np.abs(x_sample)) < 1000 else 'Wavelength (nm)'
    else:
        xlabel = 'Wavelength (Å)'

    # Get Stokes Data
    data_in_I = get_stokes_data(obs_in, 'I')
    data_out_I = get_stokes_data(obs_out, 'I')

    # Interpolate to common grid
    data_in_I = interpolate_data_list(x_sample, xs_list_in, data_in_I)
    data_out_I = interpolate_data_list(x_sample, xs_list_out, data_out_I)

    data_in_Pol = None
    data_out_Pol = None

    if stokes_cfg in ['V', 'Q', 'U']:
        data_in_Pol = get_stokes_data(obs_in, stokes_cfg)
        data_out_Pol = get_stokes_data(obs_out, stokes_cfg)

        # Interpolate Pol
        data_in_Pol = interpolate_data_list(x_sample, xs_list_in, data_in_Pol)
        data_out_Pol = interpolate_data_list(x_sample, xs_list_out,
                                             data_out_Pol)

        # Auto-scale check
        max_pol = np.max(np.abs(data_in_Pol)) if data_in_Pol else 0.0
        print(f"  Max {stokes_cfg} amplitude (Input): {max_pol:.3e}")

        current_mult = CONFIG.get('pol_scale_mult', 100.0)
        if max_pol > 0 and max_pol * current_mult * CONFIG[
                'stack_scale'] < 0.005:
            suggested_mult = 0.1 / max_pol
            print(f"  Auto-adjusting pol_scale_mult to {suggested_mult:.1e}")
            CONFIG['pol_scale_mult'] = suggested_mult

    # Baseline removal (Optional)
    if CONFIG['remove_baseline']:
        print("Removing baseline...")
        mean_I = np.mean(data_in_I, axis=0)
        data_in_I = [d - mean_I + 1.0 for d in data_in_I]
        data_out_I = [d - mean_I + 1.0
                      for d in data_out_I]  # Subtract SAME mean

        if data_in_Pol is not None and data_out_Pol is not None:
            mean_P = np.mean(data_in_Pol, axis=0)
            data_in_Pol = [d - mean_P for d in data_in_Pol]
            data_out_Pol = [d - mean_P for d in data_out_Pol]

    # ==========================================================================
    # Plotting
    # ==========================================================================

    # Define rows and columns
    # If Image mode: 3 columns (Obs, Model, Residual)
    # If Stacked mode: 1 column (Comparison)

    axes_image = None
    axes_stacked = None

    if plot_mode == 'image':
        nrows = 1 if stokes_cfg == 'I' else 2
        ncols = 3
        figsize = (16, 5 * nrows)
        fig, axes_raw = plt.subplots(nrows,
                                     ncols,
                                     figsize=figsize,
                                     sharex=True,
                                     sharey=True)
        if nrows == 1:
            axes_image = axes_raw.reshape(1, -1)
        else:
            axes_image = axes_raw
    else:
        # Stacked mode
        nrows = 1 if stokes_cfg == 'I' else 2
        ncols = 1
        figsize = (8, 10)
        fig, axes_raw = plt.subplots(nrows,
                                     ncols,
                                     figsize=figsize,
                                     sharex=True)
        if nrows == 1:
            axes_stacked = [axes_raw]  # Make it iterable
        else:
            axes_stacked = list(axes_raw)

    # Prepare data pairs for iteration
    # Each item: (Label, DataIn, DataOut, RowIndex)
    plot_items = [('I', data_in_I, data_out_I, 0)]
    if stokes_cfg != 'I' and data_in_Pol is not None and data_out_Pol is not None:
        plot_items.append((stokes_cfg, data_in_Pol, data_out_Pol, 1))

    scale = CONFIG['stack_scale']

    for label, d_in, d_out, row_idx in plot_items:

        # --- Mode: Image (Obs, Model, Residual) ---
        if plot_mode == 'image' and axes_image is not None:
            # Calculate Residuals
            residuals = [obs - mod for obs, mod in zip(d_in, d_out)]

            # Determine color limits
            if label == 'I':
                vmin, vmax = CONFIG.get('vmin_I',
                                        0.85), CONFIG.get('vmax_I', 1.15)
                vmin_res, vmax_res = CONFIG['vmin_res'], CONFIG['vmax_res']
                cmap = 'viridis'  # or gray
                cmap_res = CONFIG['cmap']
            else:
                vmin, vmax = CONFIG.get('vmin_pol',
                                        -0.005), CONFIG.get('vmax_pol', 0.005)
                vmin_res, vmax_res = CONFIG['vmin_res'], CONFIG['vmax_res']
                cmap = 'RdBu_r'
                cmap_res = CONFIG['cmap']

            # Data to plot
            datasets = [d_in, d_out, residuals]
            titles = [
                f'Observed {label}', f'Model {label}', f'Residual {label}'
            ]
            cmaps = [cmap, cmap, cmap_res]
            vmins = [vmin, vmin, vmin_res]
            vmaxs = [vmax, vmax, vmax_res]

            # Create uniform xs list since we interpolated data
            xs_uniform = [x_sample] * len(times_in)

            for col_idx in range(3):
                ax = axes_image[row_idx, col_idx]
                data_list = datasets[col_idx]

                # Create IrregularDynamicSpectrum object
                dynspec = IrregularDynamicSpectrum(np.array(times_in),
                                                   xs_uniform, data_list)

                # Plot
                ylabel_arg = 'Rotation Phase' if col_idx == 0 else ''
                xlabel_arg = xlabel if row_idx == nrows - 1 else ''
                cbar_label_arg = 'Intensity' if col_idx < 2 else 'Residual'

                dynspec.plot(ax=ax,
                             cmap=cmaps[col_idx],
                             vmin=vmins[col_idx],
                             vmax=vmaxs[col_idx],
                             title=titles[col_idx],
                             xlabel=xlabel_arg,
                             ylabel=ylabel_arg,
                             colorbar=True,
                             cbar_label=cbar_label_arg)

        # --- Mode: Stacked (Comparison) ---
        elif plot_mode == 'stacked' and axes_stacked is not None:
            ax = axes_stacked[row_idx]
            pol_mult = CONFIG['pol_scale_mult'] if label != 'I' else 1.0

            for i in range(len(times_in)):
                phase = times_in[i]
                x = x_sample
                y_in = d_in[i]
                y_out = d_out[i]

                # Offset logic
                if label == 'I':
                    offset = phase + (y_in - 1.0) * scale
                    offset_out = phase + (y_out - 1.0) * scale
                else:
                    offset = phase + y_in * pol_mult * scale
                    offset_out = phase + y_out * pol_mult * scale

                # Plot Obs
                ax.plot(x,
                        offset,
                        color=CONFIG['obs_color'],
                        lw=CONFIG['line_width'],
                        alpha=0.6,
                        label='Obs' if i == 0 else "")
                # Plot Model
                ax.plot(x,
                        offset_out,
                        color=CONFIG['model_color'],
                        lw=CONFIG['line_width'],
                        alpha=0.8,
                        linestyle='--',
                        label='Model' if i == 0 else "")

            ax.set_title(f'Comparison Stokes {label}')
            if row_idx == 0:
                ax.legend(loc='upper right')

            # Set Y limits
            ax.set_ylim(times_in[0] - 0.1, times_in[-1] + 0.1)
            ax.set_xlim(x_sample[0], x_sample[-1])

            if row_idx == nrows - 1:
                ax.set_xlabel(xlabel)
            ax.set_ylabel('Rotation Phase')

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {out_file}")
    else:
        plt.show()


if __name__ == '__main__':
    sys.exit(main())
