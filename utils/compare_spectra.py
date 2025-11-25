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
# from utils.dynamic_spectrum import IrregularDynamicSpectrum
from core.mainFuncs import readParamsTomog, compute_phase_from_jd  # noqa: E402
from core.SpecIO import loadObsProfile  # noqa: E402

# ==============================================================================
#                                 Configuration Area
# ==============================================================================
CONFIG = {
    'params_file': 'input/params_inverse_test.txt',
    'input_dir': 'input/inSpec',  # Observation data directory
    'output_dir': 'output/inverse_test',  # Model data directory

    # Select Stokes parameter to display: 'I', 'V', 'Q', 'U'
    'stokes': 'V',
    'file_type': 'auto',

    # 'image':   Plot residual dynamic spectrum (Obs - Model)
    # 'stacked': Plot stacked line plot (Obs and Model in same plot)
    'plot_mode': 'stacked',
    'out_file': None,

    # --- Image Mode Config (Residuals) ---
    'cmap': 'RdBu_r',
    'vmin_res':
    -0.001,  # Residual color range (adjust based on signal strength)
    'vmax_res': 0.001,

    # --- Stacked Mode Config ---
    'stack_scale': 1.0,
    'pol_scale_mult': 1,
    'obs_color': 'black',
    'model_color': 'red',
    'line_width': 0.8,
    'remove_baseline': False,
    'align_continuum': False,
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

    times = []
    obs_list = []

    for i in range(num_obs):
        f_path = find_file_for_index(data_dir, i)

        if f_path is None:
            # Try matching without leading zeros
            f_path = find_file_for_index(data_dir, i)

        if f_path is None or not f_path.exists():
            print(
                f"  Warning: Missing file for index {i} (Phase {phases[i]:.3f})"
            )
            continue

        try:
            obs = loadObsProfile(str(f_path), file_type=CONFIG['file_type'])
            times.append(phases[i])
            obs_list.append(obs)
        except Exception as e:
            print(f"  Error loading {f_path.name}: {e}")

    # Sort by phase
    if times:
        order = np.argsort(times)
        sorted_times = np.array([times[i] for i in order])
        sorted_obs = [obs_list[i] for i in order]
        return sorted_times, sorted_obs
    else:
        return np.array([]), []


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

    # 4. Prepare Data
    xs_list = [obs.wl for obs in obs_in]  # Assume same grid
    x_sample = xs_list[0]

    # Auto X-label
    if np.mean(x_sample) < 2000:
        xlabel = 'Velocity (km/s)' if np.max(
            np.abs(x_sample)) < 1000 else 'Wavelength (nm)'
    else:
        xlabel = 'Wavelength (Å)'

    # Get Stokes Data
    data_in_I = get_stokes_data(obs_in, 'I')
    data_out_I = get_stokes_data(obs_out, 'I')

    data_in_Pol = None
    data_out_Pol = None

    if stokes_cfg in ['V', 'Q', 'U']:
        data_in_Pol = get_stokes_data(obs_in, stokes_cfg)
        data_out_Pol = get_stokes_data(obs_out, stokes_cfg)

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

        if data_in_Pol:
            mean_P = np.mean(data_in_Pol, axis=0)
            data_in_Pol = [d - mean_P for d in data_in_Pol]
            data_out_Pol = [d - mean_P for d in data_out_Pol]

    # ==========================================================================
    # Plotting
    # ==========================================================================
    if stokes_cfg == 'I':
        fig, ax_main = plt.subplots(figsize=(6, 10))
        axes = [ax_main]
        pairs = [('I', data_in_I, data_out_I)]
    else:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 10), sharey=True)
        axes = [ax_l, ax_r]
        pairs = [('I', data_in_I, data_out_I),
                 (stokes_cfg, data_in_Pol, data_out_Pol)]

    scale = CONFIG['stack_scale']

    for ax_idx, ax in enumerate(axes):
        label, d_in, d_out = pairs[ax_idx]

        # --- Mode: Image (Residuals) ---
        if plot_mode == 'image':
            # Calculate Residuals
            residuals = [obs - mod for obs, mod in zip(d_in, d_out)]

            # Setup limits
            if label == 'I':
                vmin, vmax = CONFIG['vmin_res'], CONFIG['vmax_res']
                # I residuals might be larger or smaller, usually small if fit is good
                # Let's use same as Pol for now or config
            else:
                vmin, vmax = CONFIG['vmin_res'], CONFIG['vmax_res']

            # Plot
            img_data = np.array(residuals)
            X, Y = np.meshgrid(x_sample, times_in)

            im = ax.pcolormesh(X,
                               Y,
                               img_data,
                               cmap=CONFIG['cmap'],
                               vmin=vmin,
                               vmax=vmax,
                               shading='auto')
            plt.colorbar(im, ax=ax, label='Residual (Obs - Model)')
            ax.set_title(f'Residuals Stokes {label}')

        # --- Mode: Stacked (Comparison) ---
        elif plot_mode == 'stacked':
            pol_mult = CONFIG['pol_scale_mult'] if label != 'I' else 1.0

            for i in range(len(times_in)):
                phase = times_in[i]
                x = xs_list[i]
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
            ax.legend(loc='upper right')

            # Set Y limits
            ax.set_ylim(times_in[0] - 0.1, times_in[-1] + 0.1)
            ax.set_xlim(x_sample[0], x_sample[-1])

        ax.set_xlabel(xlabel)
        if ax_idx == 0:
            ax.set_ylabel('Rotation Phase')

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {out_file}")
    else:
        plt.show()


if __name__ == '__main__':
    sys.exit(main())
