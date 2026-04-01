#!/usr/bin/env python3
"""Comprehensive demonstration of veiling effects in pyZeeTom.

This example demonstrates all veiling capabilities:
1. **Constant veiling**: Basic veiling effect with different factors
2. **Phase-dependent veiling**: Veiling that varies with orbital phase
3. **Regional veiling**: Veiling applied to specific spatial regions

Physical Background
-------------------
**Veiling** is the dilution of stellar absorption lines by additional continuum
emission from an accretion disk or other hot sources. Common in:
- T Tauri stars: Strong accretion disk continuum
- Herbig Ae/Be stars: Circumstellar disk emission
- Cataclysmic variables: Accretion disk radiation

Physical Model
--------------
I_obs(λ) = [I_star(λ) + r·I_cont] / (1 + r)

Effects:
- Line depth: d_obs = d₀/(1+r)
- Equivalent width: EW_obs = EW₀/(1+r)
- All Stokes parameters diluted by factor 1/(1+r)

Usage
-----
python examples/06_veiling_comprehensive_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mainFuncs import readParamsTomog
from core.local_linemodel_basic import LineData, GaussianZeemanWeakLineModel
from core.physical_model import create_physical_model

# ============================================================================
# Phase-dependent veiling models
# ============================================================================


def veiling_hotspot(phase, r_max=2.0, phase_max=0.25, width=0.15):
    """Localized accretion hotspot model.
    
    Maximum veiling when hotspot faces observer.
    """
    delta = phase - phase_max
    if delta > 0.5:
        delta -= 1.0
    elif delta < -0.5:
        delta += 1.0
    return r_max * np.exp(-0.5 * (delta / width)**2) + 0.1


def veiling_disk_occultation(phase, r_max=1.5, inclination_deg=60):
    """Disk occultation geometry model.
    
    Veiling modulated by viewing angle.
    """
    phi = 2 * np.pi * phase
    inc_rad = np.deg2rad(inclination_deg)
    proj = np.abs(np.cos(phi) * np.sin(inc_rad))
    return r_max * proj + 0.2


def veiling_sinusoidal(phase, r_mean=1.0, amplitude=0.8):
    """Simple sinusoidal variation."""
    phi = 2 * np.pi * phase
    return max(0.0, r_mean + amplitude * np.sin(phi))


# ============================================================================
# Main demonstration function
# ============================================================================


def demonstrate_veiling_comprehensive():
    """Comprehensive veiling demonstration."""

    print("=" * 70)
    print("Comprehensive Veiling Effect Demonstration")
    print("=" * 70)

    # Load parameters
    param_file = 'input/params_tomog.txt'
    print(f"\n1. Loading parameters from {param_file}")
    par = readParamsTomog(param_file)

    # Load line data
    line_file = 'input/lines.txt'
    print(f"2. Loading line data from {line_file}")
    line_data = LineData(line_file)

    # Create line model
    line_model = GaussianZeemanWeakLineModel(line_data)

    # Create figure with subplots
    print("\n3. Computing demonstrations...")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.35)

    # ========================================================================
    # PART 1: Constant Veiling Effect
    # ========================================================================
    print("\n   Part 1: Constant veiling effect")

    veiling_factors = [0.0, 0.5, 1.0, 2.0]
    colors_const = ['black', 'blue', 'green', 'red']
    results_const = {}

    for i, r in enumerate(veiling_factors):
        model = create_physical_model(par,
                                      wl0_nm=line_data.wl0,
                                      line_model=line_model,
                                      veiling_factor=r,
                                      verbose=0)
        integrator = model.integrator
        integrator.compute_spectrum()

        results_const[r] = {
            'v': integrator.v,
            'I': integrator.I.copy(),
            'V': integrator.V.copy(),
        }

        line_depth = 1.0 - np.min(results_const[r]['I'])
        print(f"      r={r:.1f}: depth={line_depth:.4f}")

    # Plot 1.1: Stokes I comparison
    ax1 = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(veiling_factors):
        res = results_const[r]
        label = f'r={r:.1f}' if r > 0 else 'No veiling'
        ax1.plot(res['v'],
                 res['I'],
                 color=colors_const[i],
                 label=label,
                 linewidth=2)
    ax1.set_xlabel('Velocity (km/s)', fontsize=10)
    ax1.set_ylabel('Stokes I', fontsize=10)
    ax1.set_title('Constant Veiling: Stokes I', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 1.2: Line depth vs veiling
    ax2 = fig.add_subplot(gs[0, 1])
    depths = [1.0 - np.min(results_const[r]['I']) for r in veiling_factors]
    r_theory = np.linspace(0, max(veiling_factors), 100)
    depth_theory = depths[0] / (1 + r_theory)

    ax2.plot(r_theory,
             depth_theory,
             'k--',
             label='Theory',
             linewidth=2,
             alpha=0.7)
    ax2.plot(veiling_factors,
             depths,
             'o-',
             color='crimson',
             markersize=10,
             label='Computed',
             linewidth=2)
    ax2.set_xlabel('Veiling Factor (r)', fontsize=10)
    ax2.set_ylabel('Line Depth', fontsize=10)
    ax2.set_title('Dilution: d = d₀/(1+r)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ========================================================================
    # PART 2: Phase-Dependent Veiling
    # ========================================================================
    print("\n   Part 2: Phase-dependent veiling")

    n_phases = 20
    phases = np.linspace(0, 1, n_phases)

    models_phase = {
        'Hotspot': veiling_hotspot,
        'Disk Occultation': veiling_disk_occultation,
        'Sinusoidal': veiling_sinusoidal,
    }
    colors_phase = ['red', 'blue', 'green']
    results_phase = {}

    for model_name, veiling_func in models_phase.items():
        print(f"      Computing {model_name} model...")

        model = create_physical_model(par,
                                      wl0_nm=line_data.wl0,
                                      line_model=line_model,
                                      veiling_factor=veiling_func,
                                      verbose=0)
        integrator = model.integrator

        phase_results = []
        veiling_values = []
        line_depths = []

        for phase in phases:
            result = integrator.compute_spectrum_single_phase(phase)
            phase_results.append(result)
            veiling_values.append(result['veiling_factor'])
            line_depth = 1.0 - np.min(result['I'])
            line_depths.append(line_depth)

        results_phase[model_name] = {
            'phases': phases,
            'phase_results': phase_results,
            'veiling_values': np.array(veiling_values),
            'line_depths': np.array(line_depths),
        }

        print(f"         Veiling range: [{min(veiling_values):.2f}, "
              f"{max(veiling_values):.2f}]")

    # Plot 2.1: Veiling vs Phase
    ax3 = fig.add_subplot(gs[0, 2:])
    for i, (model_name, data) in enumerate(results_phase.items()):
        ax3.plot(data['phases'],
                 data['veiling_values'],
                 'o-',
                 color=colors_phase[i],
                 label=model_name,
                 linewidth=2,
                 markersize=5)
    ax3.set_xlabel('Phase', fontsize=10)
    ax3.set_ylabel('Veiling Factor (r)', fontsize=10)
    ax3.set_title('Phase-Dependent Veiling Models',
                  fontsize=11,
                  fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    # Plot 2.2: Line depth vs Phase
    ax4 = fig.add_subplot(gs[1, 0])
    for i, (model_name, data) in enumerate(results_phase.items()):
        ax4.plot(data['phases'],
                 data['line_depths'],
                 'o-',
                 color=colors_phase[i],
                 label=model_name,
                 linewidth=2,
                 markersize=4)
    ax4.set_xlabel('Phase', fontsize=10)
    ax4.set_ylabel('Line Depth', fontsize=10)
    ax4.set_title('Line Depth Variation', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)

    # Plot 2.3: Dynamic spectrum (Hotspot model)
    ax5 = fig.add_subplot(gs[1, 1])
    hotspot_data = results_phase['Hotspot']
    v_grid = hotspot_data['phase_results'][0]['v']
    I_matrix = np.array([res['I'] for res in hotspot_data['phase_results']])

    im = ax5.imshow(I_matrix.T,
                    aspect='auto',
                    origin='lower',
                    extent=[0, 1, v_grid[0], v_grid[-1]],
                    cmap='RdBu_r',
                    vmin=0.98,
                    vmax=1.02)
    ax5.set_xlabel('Phase', fontsize=10)
    ax5.set_ylabel('Velocity (km/s)', fontsize=10)
    ax5.set_title('Dynamic Spectrum: Hotspot', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    cbar.set_label('Stokes I', fontsize=8)

    # ========================================================================
    # PART 3: Regional Veiling
    # ========================================================================
    print("\n   Part 3: Regional veiling (stellar vs disk)")

    veiling_regional = 1.5
    regions = ['all', 'stellar', 'disk']
    region_labels = {
        'all': 'Global',
        'stellar': 'Stellar Only',
        'disk': 'Disk Only'
    }
    colors_region = ['black', 'red', 'blue']
    results_region = {}

    for region in regions:
        model = create_physical_model(par,
                                      wl0_nm=line_data.wl0,
                                      line_model=line_model,
                                      veiling_factor=veiling_regional,
                                      veiling_region=region,
                                      verbose=0)
        integrator = model.integrator

        # Get grid info
        grid = integrator.grid
        stellar_radius = getattr(integrator.geom, 'stellar_radius', 1.0)
        n_stellar = np.sum(grid.r <= stellar_radius)
        n_disk = np.sum(grid.r > stellar_radius)

        integrator.compute_spectrum()

        results_region[region] = {
            'v': integrator.v.copy(),
            'I': integrator.I.copy(),
            'stellar_radius': stellar_radius,
            'n_stellar': n_stellar,
            'n_disk': n_disk,
        }

        line_depth = 1.0 - np.min(integrator.I)
        print(f"      {region_labels[region]:15s}: depth={line_depth:.4f}")

    # Also compute no-veiling reference
    model_ref = create_physical_model(par,
                                      wl0_nm=line_data.wl0,
                                      line_model=line_model,
                                      veiling_factor=0.0,
                                      verbose=0)
    integrator_ref = model_ref.integrator
    integrator_ref.compute_spectrum()
    ref_depth = 1.0 - np.min(integrator_ref.I)

    # Plot 3.1: Stokes I comparison
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.plot(integrator_ref.v,
             integrator_ref.I,
             'k:',
             label='No veiling',
             linewidth=2,
             alpha=0.7)
    for i, region in enumerate(regions):
        data = results_region[region]
        ax6.plot(data['v'],
                 data['I'],
                 color=colors_region[i],
                 label=f"{region_labels[region]} (r={veiling_regional})",
                 linewidth=2)
    ax6.set_xlabel('Velocity (km/s)', fontsize=10)
    ax6.set_ylabel('Stokes I', fontsize=10)
    ax6.set_title(f'Regional Veiling Comparison (r={veiling_regional})',
                  fontsize=11,
                  fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 3.2: Dilution factors
    ax7 = fig.add_subplot(gs[2, 0])
    dilution_factors = []
    theoretical_dilution = 1.0 / (1.0 + veiling_regional)

    for region in regions:
        depth = 1.0 - np.min(results_region[region]['I'])
        dilution = depth / ref_depth
        dilution_factors.append(dilution)

    bars = ax7.bar(range(len(regions)),
                   dilution_factors,
                   color=colors_region,
                   alpha=0.7,
                   edgecolor='black')
    ax7.axhline(y=theoretical_dilution,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Theory: 1/(1+r)={theoretical_dilution:.3f}')
    ax7.set_xticks(range(len(regions)))
    ax7.set_xticklabels([region_labels[r] for r in regions], fontsize=9)
    ax7.set_ylabel('Dilution (d/d₀)', fontsize=10)
    ax7.set_title('Regional Dilution Factors', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3, axis='y')

    for bar, dil in zip(bars, dilution_factors):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2.,
                 height,
                 f'{dil:.3f}',
                 ha='center',
                 va='bottom',
                 fontsize=9)

    # Plot 3.3: System geometry schematic
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.set_aspect('equal')

    stellar_radius = results_region['stellar']['stellar_radius']
    circle_star = plt.Circle((0, 0),
                             stellar_radius,
                             color='orange',
                             alpha=0.5,
                             label='Stellar Photosphere')
    ax8.add_patch(circle_star)

    theta = np.linspace(0, 2 * np.pi, 100)
    r_disk_inner = stellar_radius
    r_disk_outer = stellar_radius * 3
    ax8.fill_between(r_disk_outer * np.cos(theta),
                     r_disk_outer * np.sin(theta),
                     r_disk_inner * np.cos(theta),
                     r_disk_inner * np.sin(theta),
                     color='lightblue',
                     alpha=0.3,
                     label='Disk')

    ax8.set_xlim(-r_disk_outer * 1.2, r_disk_outer * 1.2)
    ax8.set_ylim(-r_disk_outer * 1.2, r_disk_outer * 1.2)
    ax8.set_xlabel('x (R*)', fontsize=9)
    ax8.set_ylabel('y (R*)', fontsize=9)
    ax8.set_title('System Geometry', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    ax8.text(0,
             0,
             'Star\n(Absorption)',
             ha='center',
             va='center',
             fontsize=8,
             fontweight='bold')
    ax8.text(r_disk_outer * 0.7,
             r_disk_outer * 0.7,
             'Disk\n(Emission)',
             ha='center',
             va='center',
             fontsize=8)

    # Plot 3.4: Flux budget
    ax9 = fig.add_subplot(gs[2, 2:])

    n_stellar = results_region['stellar']['n_stellar']
    n_total = n_stellar + results_region['stellar']['n_disk']
    f_stellar = n_stellar / n_total if n_total > 0 else 0.5
    f_disk = 1.0 - f_stellar

    categories = ['No veiling', 'Stellar veiling', 'Global veiling']
    stellar_contrib = [
        f_stellar, f_stellar / (1 + veiling_regional),
        f_stellar / (1 + veiling_regional)
    ]
    disk_contrib = [f_disk, f_disk, f_disk / (1 + veiling_regional)]
    veiling_contrib = [
        0, veiling_regional / (1 + veiling_regional),
        veiling_regional / (1 + veiling_regional)
    ]

    x = np.arange(len(categories))
    width = 0.6

    p1 = ax9.bar(x,
                 stellar_contrib,
                 width,
                 label='Stellar flux',
                 color='orange',
                 alpha=0.7)
    p2 = ax9.bar(x,
                 disk_contrib,
                 width,
                 bottom=stellar_contrib,
                 label='Disk flux',
                 color='lightblue',
                 alpha=0.7)
    p3 = ax9.bar(x,
                 veiling_contrib,
                 width,
                 bottom=[s + d for s, d in zip(stellar_contrib, disk_contrib)],
                 label='Veiling continuum',
                 color='red',
                 alpha=0.5)

    ax9.set_ylabel('Relative Flux', fontsize=10)
    ax9.set_title('Flux Budget Breakdown', fontsize=11, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels(categories, fontsize=9)
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3, axis='y')

    # Save figure
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'veiling_comprehensive_demo.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n4. Figure saved to: {output_file}")

    plt.show()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✓ Part 1: Constant Veiling")
    print(f"  Verified d = d₀/(1+r) for r = {veiling_factors}")
    print("\n✓ Part 2: Phase-Dependent Veiling")
    print(f"  Three models demonstrated over {n_phases} phases:")
    print("    - Hotspot: Localized accretion region")
    print("    - Disk Occultation: Viewing angle modulation")
    print("    - Sinusoidal: Periodic variation")
    print("\n✓ Part 3: Regional Veiling")
    print(f"  Veiling factor r={veiling_regional} applied to:")
    print(f"    - Global: All flux (dilution={dilution_factors[0]:.3f})")
    print(f"    - Stellar: Only r≤R* (dilution={dilution_factors[1]:.3f})")
    print(f"    - Disk: Only r>R* (dilution={dilution_factors[2]:.3f})")
    print(f"  Grid: {f_stellar*100:.0f}% stellar, {f_disk*100:.0f}% disk")
    print("\n✓ Physical Recommendation:")
    print("  For T Tauri stars → USE veiling_region='stellar'")
    print("  (Accretion continuum only dilutes stellar absorption lines)")
    print("=" * 70)


if __name__ == '__main__':
    demonstrate_veiling_comprehensive()
