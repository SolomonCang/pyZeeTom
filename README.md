# pyZeeTom

**pyZeeTom** is a tomography tool designed for the inversion and forward modeling of 4 Stokes parameters (I, Q, U, V) polarization spectra.

The basic idea of this code comes from the tomogrpaphy method described in Donati et al. (2001), using the Skilling & Bryan (1984) Maximum Entropy Method (MEM) algorithm as the optimization engine. The Python version of MEE method is developed based on Folsom et al. (2018) implementation in the ZDI code `ZDIpy` [https://github.com/folsomcp/ZDIpy].

In this project, an optimized MEM python module is developed to handle the inversion process efficiently, it could be easily extended to other inversion problems in astrophysics.

## Project Overview

This project addresses the following physical scenarios:
- **Central Object & Circumstellar Matter**: A central object surrounded by circumstellar matter (dust clumps, disks, planets, small bodies, etc.) orbiting in rigid body or differential rotation.
- **Phase Observation**: The observer and the central object are in the same inertial frame. Different viewing angles are observed solely through the "phase" changes caused by the object's rotation.
- **Multi-channel Observation**: Polarization spectra of Stokes I and VQU components can be obtained for each observation phase.
- **Working Modes**: Supports both Forward Modeling and MEM (Maximum Entropy Method) Inversion.

## Key Features

- **Multi-format Support**: Compatible with various observation data formats (LSD/spec/pol/I/V/Q/U).
- **Flexible Line Models**: Built-in weak-field Gaussian Zeeman line profile; supports custom spectral line models.
- **Advanced Grid System**: Annular/Disk grid supporting both rigid body and differential rotation.
- **Velocity Space Integration**: Synthesizes global Stokes spectra through velocity space integration.
- **Time Evolution**: Supports disk structure evolution (e.g., shearing of clumps) due to differential rotation.
- **Automatic Phase Calculation**: Calculates observation phase automatically based on JD, JD0, and period.
- **Extensible Architecture**: Clear structure designed for easy extension of inversion and optimization modules.

## Quick Start

### Installation

```bash
pip install -e .
# Or install with development dependencies
pip install -e .[dev]
```

### Running Forward Synthesis

```python
from pyzeetom import tomography
# Returns List[ForwardModelResult]
results = tomography.forward_tomography('input/params_tomog.txt', verbose=1)
```

### Running MEM Inversion

```python
from pyzeetom import tomography
# Returns InversionResult containing reconstructed magnetic field (B_los, B_perp, chi)
result = tomography.inversion_tomography('input/params_tomog.txt', verbose=1)
```

## Input Parameters

The main configuration is handled via a parameter file (e.g., `input/params_tomog.txt`).

### Core Stellar Parameters (Lines 0-1)

**Line 0: Radial Velocity & Rotation**
- `inclination` (deg): Inclination angle. Affects projected radial velocity `vlos = vφ·sin(i)·sin(φ)` and projected disk area.
- `vsini` (km/s): Projected equatorial rotation velocity. Used to calculate equatorial velocity `veq = vsini/sin(i)`.
- `period` (day): Rotation period at reference radius r₀. Defines reference angular velocity `Ω₀ = 2π/P`.
- `pOmega`: Differential rotation power-law index. `Ω(r) = Ω₀·(r/r₀)^pOmega`.
  - `0.0`: Rigid body rotation.
  - `-0.5`: Keplerian (disk).
  - `-1.0`: Constant angular momentum.

**Line 1: Physical Scale & Grid Definition**
- `mass` (M☉): Stellar mass. Used to calculate and output the synchronous orbit radius.
- `radius` (R☉): Stellar radius. Acts as the reference radius r₀.
- `Vmax` (km/s, optional): **Mode 1**: Directly specifies maximum disk velocity. If >0, `r_out` is ignored for velocity scaling.
- `r_out` (R*, optional): **Mode 2**: Outer disk radius (in units of stellar radius). Used if `Vmax=0`.
- `enable_occultation` (0/1, optional): Stellar occultation switch. 1 = Enable (considers occultation of the thin equatorial disk by the star).

**Grid Construction Modes:**
1. **Direct Velocity Mode**: Set `Vmax > 0` (e.g., 300 km/s). Grid is defined by `Vmax` and `nr`.
   *Note*: `r_out` must still be provided to define the spatial extent.
2. **Physical Derived Mode**: Set `Vmax = 0` and specify `r_out`. `Vmax` is calculated via differential rotation formula.

### Grid & Inversion Control (Lines 2-4)

- `nRingsStellarGrid`: Number of radial rings. Ring width `Δr = r_out_grid/nr`.
- `targetForm/targetValue/numIterations`: MEM inversion targets (e.g., Chi-squared target).
- `test_aim`: Convergence threshold.

### Line Model Parameters (Line 5)

- `lineAmpConst`: Line amplitude (<0 for absorption, >0 for emission). `I = 1 + amp·G(λ)`.
- `k_QU`: Scaling factor for Q/U second-order terms. Adjusts sensitivity to transverse fields.
- `enableV`: Enable Stokes V calculation (1=Yes, 0=No).
- `enableQU`: Enable Stokes Q/U calculation (1=Yes, 0=No).

### Instrument & File Settings (Lines 11+)

- `spectralResolution`: Spectral resolution (e.g., 65000). Converted to FWHM: `FWHM (km/s) = c/R`.
- `lineParamFile`: Path to line parameter file (Format: `wl0 sigWl g`).
- `velStart/velEnd` (km/s): Velocity grid range for synthetic spectra.
- `obsFileType`: Hint for observation file format (e.g., `auto`, `lsd_i`, `lsd_pol`, `spec_i`).
- `jDateRef` (HJD): Reference epoch HJD₀ for phase calculation: `phase = (JD - HJD₀)/period`.

### Observation Sequence (Lines 14+)
Each line follows: `filename  JD  velR`
- `JD`: Heliocentric Julian Date of observation.
- `velR` (km/s): Radial velocity correction (added to observation velocity axis).

## File Formats

### Line Parameter File (`lines.txt`)
Columns:
- `wl0` (Å): Central wavelength.
- `sigWl` (Å): Gaussian width σ (intrinsic line width).
- `g`: Landé factor (determines Zeeman splitting strength).

### Observation Data
Supported formats include columns for `wl`, `specI`, `specV`, `specQ`, `specU`, `sigma`, etc.

## Physical Model

### Velocity Field
- **Outer (r ≥ r₀)**: Power-law rotation `Ω(r) = Ω₀(r/r₀)^p`.
- **Inner (r < r₀)**: Adaptive deceleration sequence ensuring v=0 at r=0.

### Line Model (Weak Field Approximation)
- **Stokes I**: `1 + a·G(d)`
- **Stokes V**: `Cg·Blos·a·G(d)·d/σ`
- **Stokes Q/U**: `-C2·Bperp²·a·G(d)/σ²·(1-2d²)·cos/sin(2χ)`

Where `G(d)` is the Gaussian basis and `d` is the dimensionless deviation.

## Physics Notes

### Effect of Inclination on Inversion

Changing the `inclination` parameter can significantly alter the radial distribution of matter in the inversion results. This is a physical effect due to projection:

- **Velocity Projection**: The line-of-sight velocity is $v_{los} \propto v_{eq} \cdot \sin(i)$.
- **Result**: If `inclination` decreases (more face-on), $\sin(i)$ decreases. To match the same observed spectral line width (maximum $v_{los}$), the matter must be placed at a **larger radius** (where $v_{\phi}$ is higher) or the star must rotate faster.
- **Observation**: You may observe the reconstructed matter distribution "expanding" outward as you decrease the inclination in your model.

## Documentation

For detailed architecture, data flow, and module descriptions, please refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
