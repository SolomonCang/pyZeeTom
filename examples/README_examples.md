# pyZeeTom Examples Guide

This guide explains the small example scripts in the `examples/` folder.
They are designed for beginners to run the full workflow:

1. Generate synthetic data with magnetic spots
2. Run forward tomography
3. Run MEM inversion tomography
4. Visualize the geometric (disk + magnetic field) model
5. Visualize dynamic spectra

All commands below assume you are in the project root directory
(`pyZeeTom/`).

```bash
cd /path/to/pyZeeTom
```

---

## Shared Concepts and Files

### 1. Parameter file: `input/params_tomog.txt`

This text file controls both the forward and inversion tomography runs. It
contains:

- Stellar / disk geometry parameters (inclination, projected rotation, etc.)
- Disk structure (inner/outer radius, number of rings)
- Target type and regularization settings
- Reference geomodel file (initial disk + magnetic field distribution)
- Line parameters file (`input/lines.txt`)
- Wavelength grid and output format
- Observation list: each line points to one observed spectrum file

In the examples, this file is automatically created by
`examples/01_generate_spot_data.py` and then reused by all other scripts.

### 2. Observation files: `input/spot_forward_obs/obs_phase_*.lsd`

Each `.lsd` file represents a synthetic polarized spectrum at one rotational
phase. They store Stokes I, V, Q, U as a function of wavelength.

These files are used as **input data** for both:

- Forward tomography (to compare model vs. data)
- MEM inversion (to reconstruct the disk magnetic field)

### 3. Geometric model (`*.tomog`)

`*.tomog` files store the **disk geometry and physical parameters** on a
polar grid:

- Radius and azimuth of each cell (pixel) in the disk
- Brightness / emissivity of each cell
- Line-of-sight magnetic field $B_\text{los}$
- Perpendicular magnetic field $B_\perp$ and azimuth $\chi$

These parameters feed the line formation model, which then predicts the
polarized spectra.

### 4. Physics background (very brief)

The code models a circumstellar disk with magnetic regions ("spots"). As the
star-disk system rotates, different parts of the disk move toward/away from
the observer and contribute to different Doppler-shifted wavelengths.

Under the **weak-field Zeeman approximation**, the emergent Stokes profiles
from each cell are:

- **Intensity $I$**: Gaussian line profile describing absorption or emission
- **Circular polarization $V$**: proportional to the line-of-sight magnetic
	field $B_\text{los}$ and the derivative of the line profile
- **Linear polarization $Q, U$**: depend on the transverse field
	$B_\perp$ and its azimuth $\chi$

By integrating the contribution from all disk cells in velocity space, we get
the synthetic Stokes spectra for a given phase.

---

## Example 01: Generate spot data

**Script**: `examples/01_generate_spot_data.py`

### Purpose

Create a simple three-spot magnetic model on a disk, compute synthetic
polarized spectra at multiple phases, and prepare all files needed for
forward and inversion runs.

### Inputs

- **Internal defaults** inside the script:
	- Disk grid: 10 radial rings, radius from 0.0 to 3.0
	- Inclination: 60°
	- Differential rotation parameters (`pOmega`, `r0_rot`)
	- Three magnetic spots: radius, position (r, $\phi$), amplitude, and
		$B_\text{los}$
- `input/lines.txt`: line parameters file (central wavelength, width,
	Landé factor, etc.). You should already have this in the repository.

### Outputs

- `input/spot_forward_obs/obs_phase_*.lsd`
	- Synthetic polarized spectra (with noise) for 15 phases between 0.0 and
		0.98.
- `output/spot_truth.tomog`
	- The "truth" geometric model: the actual three-spot configuration used
		to generate the data.
- `input/params_tomog.txt`
	- Parameter file describing geometry, rotation, line data, and the list of
		observation files.

### Physics meaning

This example creates a **controlled toy model**:

- You know exactly where the magnetic spots are on the disk.
- You know the line / geometry parameters.
- You generate synthetic data using the same physics that will later be used
	in the inversion.

This is ideal for testing whether the inversion can recover the known input
structure.

---

## Example 02: Forward tomography

**Script**: `examples/02_forward_tomography.py`

### Purpose

Use the parameter file from Example 01 to run the **forward tomography**
pipeline. Forward tomography means: given a disk + magnetic field model,
predict the observable Stokes spectra for each phase.

### Inputs

- `input/params_tomog.txt` (created by Example 01)
- `input/spot_forward_obs/obs_phase_*.lsd` (listed inside the parameter file);
	these are used mostly for comparison, not for fitting here.

### Outputs

- `output/forward_test/`
	- Per-phase `.tomog` files that store the model geometry used for each
		phase.
	- Corresponding synthetic spectra written by the forward tomography engine.

### Physics meaning

This step answers the question: **"If my guess of the disk and magnetic
field is correct, what spectra should I observe at each phase?"**

In the examples, the model is consistent with how the data were generated, so
the synthetic spectra should match the input spectra fairly well (up to
noise).

---

## Example 03: MEM inversion tomography

**Script**: `examples/03_inversion_tomography.py`

### Purpose

Run the **Maximum Entropy Method (MEM) inversion** using the same parameter
file as Examples 01/02, but with **forward spectra + added noise** as the
"observations":

1. Read noiseless model spectra from Example 02
   (`output/forward_test/phase_*.spec`).
2. Add Gaussian noise in the script to each phase.
3. Write noisy spectra to `input/spot_forward_obs/obs_phase_*.lsd` (file
   names exactly match those listed in `input/params_tomog.txt`).
4. Run MEM inversion using these noisy spectra as observational data.

The goal remains to reconstruct:

- Line-of-sight magnetic field $B_\text{los}$ on the disk
- Transverse field $B_\perp$ and azimuth $\chi$
- Brightness distribution

such that the synthetic spectra match the noisy "observations" while keeping
the solution as simple (high entropy) as possible.

### Inputs

- `input/params_tomog.txt`
- `output/forward_test/phase_*.spec` (generated by Example 02)
  - Inside the script, these are converted into noisy
    `input/spot_forward_obs/obs_phase_*.lsd`.

### Outputs

- `input/spot_forward_obs/obs_phase_*.lsd`
	- Noisy LSD spectra generated from model spectra; S/N is controlled by the
	  `snr` parameter in `examples/03_inversion_tomography.py` (default 1000).

- `output/inverse_test/`
	- `mem_inversion_model.tomog`: final reconstructed disk + magnetic field
	  model.
	- `phase_*.spec` / `phase_*.lsd`: synthetic spectra from the inversion
	  model.
	- Additional intermediate files with iteration history and convergence
	  information.

### Physics meaning

As before, this example solves a **regularized inverse problem**:

> Find the model that fits the (noisy) data (low $\chi^2$) but does not
> introduce unnecessary structure (maximal entropy).

- $\chi^2$ measures the misfit between observed and synthetic spectra.
- Entropy terms favor smooth, simple, or prior-like distributions of
	brightness and magnetic field.

The important difference compared to the original version is that the
"observations" are now **true model spectra plus explicit Gaussian noise**.
This lets you study how the inversion behaves as a function of S/N: by
changing `snr` in `examples/03_inversion_tomography.py`, you can directly
compare MEM reconstructions under different noise levels.

---

## Example 04: Visualize geometric model

**Script**: `examples/04_visualize_geomodel.py`

### Purpose

Load a `.tomog` geometric model and visualize the disk structure and magnetic
field maps.

By default, the script uses the **inversion result**:

- `output/inverse_test/mem_inversion_model.tomog`

You can edit the script to instead visualize, for example:

- `output/forward_test/geomodel_phase_00.tomog` (forward model at a given
	phase)

### Inputs

- A `.tomog` geomodel file (by default the MEM inversion result).

### Outputs

- A matplotlib figure created by
	`utils.visualize_geomodel.plot_geomodel_contour(...)` showing:
	- Disk in polar coordinates (radius and azimuth)
	- Color maps for brightness and/or magnetic components

### Physics meaning

This visualization helps you **see** the reconstructed structure:

- Where are the bright regions and dark regions on the disk?
- Where is the magnetic field strong or weak?
- How do the line-of-sight and transverse components vary with radius and
	azimuth?

By comparing the **truth model** (`output/spot_truth.tomog`) with the
**inversion result** (`mem_inversion_model.tomog`), you can judge how well
the tomography has recovered the underlying physics.

---

## Example 05: Dynamic spectrum from forward results

**Script**: `examples/05_dynamic_spectrum.py`

### Purpose

Generate and plot a **dynamic spectrum** from the forward tomography results.
The dynamic spectrum is a 2D image with:

- Horizontal axis: wavelength
- Vertical axis: rotational phase
- Color scale: intensity (Stokes I)

### Inputs

- `input/params_tomog.txt`
- Forward results (recomputed on the fly by the script).

### Outputs

- A matplotlib figure showing the dynamic spectrum created using
	`utils.dynamic_spectrum.IrregularDynamicSpectrum`.

### Physics meaning

The dynamic spectrum makes it easy to see **how spectral features move with
phase**:

- Emission/absorption bumps from spots move in wavelength due to rotation.
- Their trajectories encode the velocity field and location of spots in the
	disk.

This is a common diagnostic tool in stellar and disk tomography: patterns in
the dynamic spectrum directly reflect the geometry and kinematics of the
emitting regions.

---

## Recommended run order for beginners

From the project root:

```bash
python examples/01_generate_spot_data.py
python examples/02_forward_tomography.py
python examples/03_inversion_tomography.py
python examples/04_visualize_geomodel.py
python examples/05_dynamic_spectrum.py
```

This sequence will:

1. Build a controlled three-spot magnetic disk model and synthetic data
2. Run forward tomography to predict spectra from a model
3. Run MEM inversion to recover the model from the spectra
4. Visualize the reconstructed geometry and magnetic field
5. Visualize how spectral features evolve with rotational phase

Once you are comfortable with these scripts, you can:

- Change the spot configuration (positions, strengths, number of spots)
- Adjust disk and rotation parameters
- Use different line parameters in `input/lines.txt`
- Apply the same workflow to your own observed spectra

