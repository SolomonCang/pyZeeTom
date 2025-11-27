
# pyZeeTom Architecture Overview

**Last Updated**: 2025-11-15  
**Version**: Phase 2.5.4.1 (Refactoring Complete)

---

## Table of Contents
1. Project Overview
2. Layered Architecture
3. Module Breakdown
4. Data Flow & Workflow
5. Physical Model
6. Extensibility & Development
7. References & Principles

---

## 1. Project Overview

**pyZeeTom** is a tomography toolkit for forward modeling and inversion of 4 Stokes parameters (I, Q, U, V) polarization spectra, designed for stellar and circumstellar systems.

**Physical Scenario:**
- Central object surrounded by circumstellar matter (disk, clumps, planets, etc.) in rigid or differential rotation
- Observer and object share the inertial frame; phase changes provide different viewing angles
- Multi-channel Stokes spectra for each phase
- Forward modeling and MEM (Maximum Entropy Method) inversion supported

**Key Features:**
- Multi-format observation data (LSD/spec/pol/I/V/Q/U)
- Weak-field Gaussian Zeeman and custom line models
- Annular/disk grid, rigid/differential rotation
- Velocity space integration for global Stokes synthesis
- Time evolution: disk structure changes with rotation
- Automatic phase calculation (JD, JD0, period)
- MEM inversion for magnetic field reconstruction

---

## 2. Layered Architecture

```
┌───────────── User Interface ─────────────┐
│ pyzeetom/tomography.py                   │
│   - forward_tomography()                 │
│   - inversion_tomography()               │
└───────────────┬─────────────────────────┘
                     │
┌───────────────▼─────────────────────────┐
│ Workflow Engine                        │
│   - tomography_forward.py              │
│   - tomography_inversion.py            │
└───────────────┬─────────────────────────┘
                     │
┌───────────────▼─────────────────────────┐
│ Config & Result Layer                   │
│   - tomography_config.py                │
│   - tomography_result.py                │
└───────────────┬─────────────────────────┘
                     │
┌───────────────▼─────────────────────────┐
│ Physics Core                            │
│   - velspace_DiskIntegrator.py          │
│   - local_linemodel_basic.py            │
│   - mem_tomography.py                   │
└───────────────┬─────────────────────────┘
                     │
┌───────────────▼─────────────────────────┐
│ Utility Layer                           │
│   - grid_tom.py                         │
│   - disk_geometry.py                    │
│   - SpecIO.py                           │
│   - mainFuncs.py                        │
│   - mem_generic.py                      │
│   - iteration_manager.py                │
│   - mem_optimization.py                 │
│   - mem_monitoring.py                   │
└─────────────────────────────────────────┘
```

---

## 3. Module Breakdown

### User Interface
- **pyzeetom/tomography.py**: Main entry, provides `forward_tomography()` and `inversion_tomography()` APIs. Thin wrapper, unified error handling.

### Workflow Engine
- **core/tomography_forward.py**: Forward synthesis engine. Validates config, runs velocity space integration, aggregates results.
- **core/tomography_inversion.py**: MEM inversion engine. Manages iterations, optimization, convergence, and result aggregation.

### Config & Result Layer
- **core/tomography_config.py**: Dataclass containers for forward/inversion config. Type safety, validation, serialization.
- **core/tomography_result.py**: Unified result containers for forward and inversion outputs.

### Physics Core
- **core/velspace_DiskIntegrator.py**: Disk velocity field modeling, Stokes synthesis, velocity space integration, derivatives for inversion.
- **core/local_linemodel_basic.py**: Weak-field Zeeman line model. Extensible for custom line models.
- **core/mem_tomography.py**: Adapter between generic MEM optimizer and project-specific parameterization.

### Utility Layer
- **core/grid_tom.py**: Disk grid generation and management (equal Δr rings, vectorized arrays).
- **core/disk_geometry.py**: Disk geometry and dynamics parameter container.
- **core/SpecIO.py**: Spectrum I/O, multi-format support.
- **core/mainFuncs.py**: Parameter parsing, backward compatibility.
- **core/mem_generic.py**: Generic MEM algorithm, project-agnostic.
- **core/iteration_manager.py**: Iteration control, convergence, intermediate saving.
- **core/mem_optimization.py**: Optimization acceleration, caching, data flow management.
- **core/mem_monitoring.py**: Monitoring, logging, performance metrics.

---

## 4. Data Flow & Workflow

### Forward Synthesis
1. Read parameters (`mainFuncs.readParamsTomog`)
2. Read observation data (`SpecIO.obsProfSetInRange`)
3. Read line parameters (`LineData`)
4. Build config (`ForwardModelConfig`)
5. Validate config
6. Run synthesis (`run_forward_synthesis`)
7. For each phase:
    - Compute velocity/B-field projection
    - Local Stokes profile calculation
    - Velocity space integration
    - Aggregate results (`ForwardModelResult`)
8. Output files: model spectra, summary

### MEM Inversion
1. Prepare forward results, obs data, initial B-field guess
2. Build inversion config (`InversionConfig`)
3. Run inversion (`run_mem_inversion`)
4. Iteration manager controls optimization
5. For each iteration:
    - Compute synthetic spectra
    - Calculate gradients, response matrix
    - MEM optimization step
    - Update B-field parameters
    - Check convergence, save intermediate results
6. Output files: inversion results, summary, intermediates

---

## 5. Physical Model

### Disk Velocity Field
- **Outer (r ≥ r₀):** Power-law rotation: Ω(r) = Ω₀ (r/r₀)^p
- **Inner (r < r₀):** Smooth deceleration sequence
- **Linear velocity:** v_φ(r) = r · Ω(r)

### Line Model (Weak Field Approximation)
- **Stokes I:** I(λ) = 1 + a · G(d)
- **Stokes V:** V(λ) = C_g · B_los · a · G(d) · d/σ
- **Stokes Q/U:** Q(λ) = -C_2 · B_perp² · a · G(d)/σ² · (1-2d²) · cos(2χ)
                      U(λ) = -C_2 · B_perp² · a · G(d)/σ² · (1-2d²) · sin(2χ)
  - G(d) = exp(-d²), d = (λ - λ₀)/σ

### Doppler Shift
- v = c · (λ - λ_ref)/λ_ref
- ν = ν₀ · (1 - v/c)

### Disk Integration
- S_obs(λ) = Σ w_i · S_local(i, λ)

---

## 6. Extensibility & Development

### Custom Line Model
1. Inherit `BaseLineModel`, implement `compute_local_profile()`
2. Return dict with I, V, Q, U arrays

### New Observation Format
1. Add parser in `SpecIO.py`, return `ObservationProfile`
2. Integrate with `obsProfSetInRange()`

### New Inversion Method
1. Create new module (e.g., `tomography_mcmc.py`)
2. Implement interface similar to `run_mem_inversion()`
3. Use existing config/result containers
4. Expose in main entry

### Development Workflow
1. Validate config (`config.validate()`)
2. Run forward synthesis (`forward_tomography`)
3. Run inversion (`inversion_tomography`)
4. Extend models as needed
5. Optimize performance (caching, monitoring)

---

## 7. References & Principles

### Core Algorithms
- Skilling & Bryan (1984): Maximum Entropy Image Reconstruction
- Hobson & Lasenby (1998): Magnetic field inversion using entropy methods

### Design Patterns
- Layered architecture: UI → Config → Workflow → Physics → Utility
- Type-safe config objects
- Unified result containers
- Adapter pattern for decoupling
- Callback design for generic optimizers

### Extensibility
- New line model: Inherit `BaseLineModel`
- New geometry: Edit `disk_geometry.py`
- New obs format: Extend `SpecIO.py`
- New inversion: Add workflow module

---

**Documentation is updated after major refactoring.**
**Last Modified:** 2025-11-15
**Contributors:** Tianqi Cang

### 1. User Interface Layer

#### `pyzeetom/tomography.py` (235 lines)
Main entry module, providing two core APIs:

```python
def forward_tomography(
    param_file: str = 'input/params_tomog.txt',
    verbose: int = 1,
    output_dir: str = './output'
) -> List[ForwardModelResult]
```
Executes forward spectrum synthesis, returning forward results for each phase.

```python
def inversion_tomography(
    param_file: str = 'input/params_tomog.txt',
    obs_file: str = None,
    verbose: int = 1,
    output_dir: str = './output'
) -> InversionResult
```
Executes MEM inversion workflow, returning the reconstructed magnetic field distribution.

**Key Points**:
- Thin wrapper design, delegating to the workflow engine
- Unified parameter handling and error control
- Automatic path deduction

---

### 2. Workflow Execution Engine

#### `core/tomography_forward.py` (246 lines)
Forward workflow main engine.

**Core Function**:
```python
def run_forward_synthesis(
    config: ForwardModelConfig,
    verbose: bool = False
) -> List[ForwardModelResult]
```

**Workflow Steps**:
1. Validate configuration integrity
2. Create disk integrator (VelspaceDiskIntegrator)
3. Perform spectrum synthesis for each phase
4. Collect and return results

**Key Operations**:
- Phase iteration
- Velocity space integration
- Stokes spectrum synthesis
- Result aggregation

---

#### `core/tomography_inversion.py` (1026 lines)
MEM inversion workflow main engine.

**Core Function**:
```python
def run_mem_inversion(
    config: InversionConfig,
    verbose: bool = False
) -> InversionResult
```

**Workflow Steps**:
1. Initialize inversion iteration manager
2. Perform MEM optimization for each phase and each pixel
3. Monitor convergence
4. Save intermediate results
5. Return final inversion result

**Key Components**:
- Parameter encoding/decoding (B-field parameter packing)
- MEM optimizer adapter layer
- Iteration convergence control
- Result aggregation

---

### 3. Config & Result Container

#### `core/tomography_config.py` (621 lines)
Unified configuration objects.

**Core Classes**:

```python
@dataclass
class ForwardModelConfig:
    """Forward configuration container"""
    par: Any                    # Parameter object
    obsSet: List[Any]          # Observation dataset
    lineData: BasicLineData     # Line parameters
    geom: SimpleDiskGeometry    # Disk geometry
    line_model: Any            # Line model
    velEq: float = 100.0       # Equatorial velocity (km/s)
    pOmega: float = 0.0        # Differential rotation index
    radius: float = 1.0        # Central radius
    # ... more parameters
    
    def validate(self) -> bool
    def create_summary(self) -> str
    @classmethod
    def from_par(cls, par, obsSet, lineData, **kwargs)
```

```python
@dataclass
class InversionConfig:
    """Inversion configuration container"""
    forward_config: ForwardModelConfig
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    entropy_regularization: float = 0.1
    # ... more parameters
    
    def validate(self) -> bool
```

**Advantages**:
- Type safety and IDE auto-completion
- Built-in validation logic
- Clear parameter documentation
- Convenient serialization/deserialization

---

#### `core/tomography_result.py` (16 KB)
Unified result container.

```python
@dataclass
class ForwardModelResult:
    """Forward result"""
    phase_index: int
    stokes_i: np.ndarray
    stokes_v: np.ndarray
    stokes_q: np.ndarray
    stokes_u: np.ndarray
    wavelength: np.ndarray
    model_parameters: dict
    
    def create_summary(self) -> str
    def save_to_file(self, filename: str)
```

```python
@dataclass
class InversionResult:
    """Inversion result"""
    forward_results: List[ForwardModelResult]
    B_los: np.ndarray
    B_perp: np.ndarray
    chi: np.ndarray
    final_entropy: float
    convergence_flag: bool
    
    def create_summary(self) -> str
    def save_to_file(self, output_dir: str)
```

---

### 4. Physics Calculation Core

#### `core/velspace_DiskIntegrator.py` (702 lines)
Velocity space disk integrator - Core physics module.

**Core Class**:
```python
class VelspaceDiskIntegrator:
    """Disk model velocity space integrator
    
    Functions:
    - Disk grid velocity mapping
    - Local Stokes spectrum generation
    - Velocity space integration synthesis
    - Multi-phase processing
    """
    
    def __init__(self, grid, line_model, verbose=False)
    
    def compute_spectrum_single_phase(
        self, 
        phase: float,
        Blos: np.ndarray,
        Bperp: np.ndarray,
        chi: np.ndarray
    ) -> dict
        # Returns: {'I': I_spec, 'V': V_spec, 'Q': Q_spec, 'U': U_spec}
    
    def derivative_Blos(self, ...) -> np.ndarray
        # Derivative of Stokes parameters w.r.t. Blos
    
    def derivative_Bperp_chi(self, ...) -> Tuple[np.ndarray, np.ndarray]
        # Derivative of Stokes parameters w.r.t. Bperp and chi
```

**Key Algorithms**:
1. **Disk Velocity Field Modeling**:
   - Outer: Power-law Ω(r) = Ω₀(r/r₀)^p
   - Inner: Adaptive deceleration sequence

2. **Local Line Calculation**:
   - Uses injected line model (line_model)
   - Supports I/V/Q/U multi-channels

3. **Velocity Space Integration**:
   - Disk grid pixel summation
   - Convolution smoothing (FWHM broadening)
   - Velocity to observation frequency mapping

4. **Derivative Calculation**:
   - Automatic calculation of Stokes parameter derivatives w.r.t. magnetic field parameters
   - Supports inversion optimization

---

#### `core/local_linemodel_basic.py` (230 lines)
Line model - Weak field approximation.

**Core Classes**:
```python
class LineData:
    """Line parameter container (read from file)"""
    wl0: float      # Line center wavelength
    sigWl: float    # Line width
    g: float        # Landé g factor
    
    def __init__(self, filename: str)
```

```python
class GaussianZeemanWeakLineModel(BaseLineModel):
    """Weak field approximation + Gaussian profile
    
    Let: d = (λ - λ0)/σ, G = exp(-d²)
    
    Output:
      I = 1 + amp × G
      V = Cg × Blos × (amp × G × d / σ)
      Q = -C2 × Bperp² × (amp × (G/σ²) × (1 - 2d²)) × cos(2χ)
      U = -C2 × Bperp² × (amp × (G/σ²) × (1 - 2d²)) × sin(2χ)
    
    Parameters:
      - amp: Line amplitude (positive=emission, negative=absorption)
      - Blos: Line-of-Sight magnetic field (km/s)
      - Bperp: Perpendicular magnetic field
      - chi: Magnetic field azimuth (radians)
    """
    
    def compute_local_profile(
        self,
        wl_grid: np.ndarray,
        amp: np.ndarray,
        Blos: np.ndarray = None,
        **kwargs
    ) -> dict
```

**Extension Point**: Inherit `BaseLineModel` to implement custom line models.

---

#### `core/mem_tomography.py` (554 lines)
MEM inversion adapter layer.

**Core Class**:
```python
class MEMTomographyAdapter:
    """Adapter layer between generic MEM algorithm and project-specific parameterization
    
    Functions:
    - Fitting of Stokes I, Q, U, V lines
    - Entropy definition for magnetic field parameters (Blos, Bperp, chi)
    - Data packing/unpacking
    - Response matrix construction
    """
    
    def __init__(self, config, grid, line_model, obs_data)
    
    def compute_synthetic(self, B_los, B_perp, chi) -> SyntheticSpectrum
        # Synthesize Stokes spectra
    
    def compute_gradients(self, B_los, B_perp, chi) -> dict
        # Compute residual gradients (for MEM optimization)
    
    def pack_parameters(self, B_los, B_perp, chi) -> np.ndarray
        # Pack into optimization vector
    
    def unpack_parameters(self, x: np.ndarray) -> Tuple[np.ndarray, ...]
        # Unpack back to physical parameters
```

**Integration Point**: Integrates with the generic MEM optimizer in `mem_generic.py`.

---

---

### 5. Basic Utility Library

#### `core/grid_tom.py` (358 lines)
Grid generation and management.

**Core Class**:
```python
class diskGrid:
    """Equal Δr layered disk grid (consistent width per ring)
    
    Attributes (stored as 1D arrays):
    - r: Cylindrical radius
    - phi: Azimuth angle
    - dr: Radial pixel width
    - dphi: Angular pixel width
    - area: Pixel area
    - ring_id: Ring ID
    - phi_id: Angular ID
    """
    
    def __init__(
        self,
        nr: int = 60,
        r_in: float = 0.0,
        r_out: float = 5.0,
        target_pixels_per_ring: Optional[Union[int, List]] = None
    )
    
    @property
    def numPoints(self) -> int
        # Total number of pixels
    
    def get_ring(self, ring_idx: int) -> dict
        # Get all pixels in a specific ring
```

**Design Highlights**:
- 1D array storage, supporting vectorized operations
- Flexible control of pixels per ring
- Automatic calculation of equal area or equal number configurations

---

#### `core/disk_geometry.py` (7.8 KB)
Disk geometry and dynamics parameter container.

```python
class SimpleDiskGeometry:
    """Disk geometry and dynamics parameters
    
    Contains:
    - diskGrid instance
    - Dynamics parameters (velEq, pOmega, r0)
    - Physical parameters (inclination, posang, etc.)
    """
```

---

#### `core/SpecIO.py` (728 lines)
Spectrum data I/O (supports multiple formats).

**Core Functions**:
```python
def obsProfSetInRange(
    fnames: List[str],
    vel_start: float,
    vel_end: float,
    vel_rs: float,
    file_type: str = 'auto',
    pol_channels: Optional[Dict] = None
) -> List[ObservationProfile]
    # Read observation dataset
```

```python
def write_model_spectrum(
    filename: str,
    wavelength: np.ndarray,
    spec_i: np.ndarray,
    spec_v: np.ndarray = None,
    spec_q: np.ndarray = None,
    spec_u: np.ndarray = None,
    file_type_hint: str = 'spec_i'
)
    # Write model spectrum
```

**Supported Formats**:
- `lsd_i`: LSD intensity-only format
- `lsd_pol`: LSD full polarimetry (I,V,Q,U,σ)
- `spec_i`: Simple spectrum (λ, I)
- `spec_pol`: Full polarimetry spectrum

---

#### `core/mainFuncs.py` (37 KB)
Parameter parsing and compatibility layer.

```python
def readParamsTomog(filename: str) -> ParamObject
    """Read parameter file (backward compatible with old format)
    
    Returns object containing all configuration parameters
    """

def parseParamLine(s: str) -> Tuple[str, str]
    """Parse parameter line"""

# ... other parameter handling functions
```

---

#### `core/mem_generic.py` (17 KB)
Generic Maximum Entropy Method (MEM) algorithm.

**Core Class**:
```python
class MEMOptimizer:
    """Generic MEM optimization algorithm
    
    Supports:
    - Maximum Likelihood under Maximum Entropy constraint
    - Adaptive convergence control
    - Lagrange multiplier management
    """
    
    def iterate(
        self,
        model_fn,
        residual_fn,
        data,
        x0: np.ndarray,
        lambda_coeff: float = 1.0
    ) -> Tuple[np.ndarray, float, dict]
```

**Design Principles**:
- Completely project-agnostic generic implementation
- Accepts project-specific physical models via callback functions
- Easy to integrate with other projects

---

#### `core/iteration_manager.py` (13 KB)
Inversion iteration control and management.

```python
class IterationManager:
    """Manage MEM iteration process
    
    Functions:
    - Iteration counting and convergence check
    - Intermediate result saving
    - Parameter convergence curve tracking
    - Adaptive step size control
    """
    
    def update(self, residual: float, entropy: float, params: np.ndarray)
    def should_continue(self) -> bool
    def get_summary(self) -> str
```

---

#### `core/mem_optimization.py` (19 KB)
MEM optimization acceleration and caching.

**Core Classes**:
```python
class ResponseMatrixCache:
    """Response matrix cache, avoiding repeated calculations"""
    
class DataPipeline:
    """Data flow management, optimizing memory usage"""
```

**Week 2 Optimization**:
- Cache response matrix (avoid repeated calculations)
- Stream processing of observation data
- Automatic memory management

---

#### `core/mem_monitoring.py` (12 KB)
Inversion monitoring and logging.

```python
class MEMMonitor:
    """Monitor MEM inversion process
    
    Records:
    - Residual, entropy, magnetic field for each iteration
    - Convergence history
    - Performance metrics
    """
```

---



## Data Flow and Workflow

### Forward Workflow (Forward Synthesis)

```
Input Files
├── params_tomog.txt (Params)
├── lines.txt (Line Params)
└── inSpec/*.lsd (Obs Data)
       │
       ▼
[pyzeetom/tomography.py::forward_tomography]
       │
       ├─ mainFuncs.readParamsTomog(params_tomog.txt)
       │  └─ ParamObject {velEq, pOmega, radius, ...}
       │
       ├─ SpecIO.obsProfSetInRange(inSpec)
       │  └─ [ObservationProfile, ...]
       │
       ├─ LineData(lines.txt)
       │  └─ LineData {wl0, sigWl, g}
       │
       ▼
[ForwardModelConfig]
       │
       ├─ SimpleDiskGeometry (Disk Geometry)
       │  └─ diskGrid + Dynamics Params
       │
       ├─ GaussianZeemanWeakLineModel (Line Model)
       │
       ├─ validate() (Config Validation)
       │
       ▼
[tomography_forward.run_forward_synthesis]
       │
       ├─ FOR each phase in [phase_0, phase_1, ...]
       │  ├─ VelspaceDiskIntegrator.compute_spectrum_single_phase
       │  │  ├─ Compute velocity and B-field projection for each grid pixel
       │  │  ├─ Call line_model.compute_local_profile
       │  │  │  └─ Return {I, V, Q, U} (Nλ,)
       │  │  ├─ Velocity space integration synthesis
       │  │  └─ Return synthetic spectra {I, V, Q, U}
       │  │
       │  └─ ForwardModelResult
       │     └─ {phase_index, stokes_i/v/q/u, wavelength, ...}
       │
       ▼
Output Files
├── output/model_phase_0.lsd
├── output/model_phase_1.lsd
└── output/outFitSummary.txt
```

### Inversion Workflow (MEM Inversion)

```
Forward Result (ForwardModelResult)
       │
       ├─ Stokes Spectra {I, V, Q, U}
       ├─ Obs Data {Iobs, Vobs, Qobs, Uobs}
       ├─ Initial B-field Guess {Blos_0, Bperp_0, chi_0}
       │
       ▼
[InversionConfig]
       │
       ├─ forward_config (ForwardModelConfig)
       ├─ max_iterations, convergence_threshold
       ├─ entropy_regularization
       │
       ▼
[tomography_inversion.run_mem_inversion]
       │
       ├─ IterationManager (Iteration Control)
       │
       ├─ FOR iteration = 0, 1, 2, ...
       │  ├─ FOR each pixel in diskGrid
       │  │  ├─ MEMTomographyAdapter.compute_synthetic
       │  │  │  ├─ Call VelspaceDiskIntegrator.compute_spectrum
       │  │  │  └─ Return synthetic spectra
       │  │  │
       │  │  ├─ MEMOptimizer.iterate (Single Step MEM Optimization)
       │  │  │  ├─ Calc Residual: χ² = Σ((S_syn - S_obs)²/σ²)
       │  │  │  ├─ Calc Entropy: H = -Σ p_i log(p_i)
       │  │  │  ├─ Maximize: Q = H - λ·χ²
       │  │  │  ├─ Update Params: Blos, Bperp, chi
       │  │  │  └─ Return x_new, χ²_new, ...
       │  │  │
       │  │  └─ Update B-field {Blos, Bperp, chi}
       │  │
       │  ├─ IterationManager.update (Convergence Check)
       │  │  ├─ Check |Δχ²| < threshold
       │  │  ├─ Check max iterations
       │  │  └─ Save intermediate results
       │  │
       │  └─ Intermediate Result Saving (Optional)
       │
       ▼
[InversionResult]
       │
       ├─ forward_results (Forward Results)
       ├─ B_los (Final LOS B-field)
       ├─ B_perp (Final Perp B-field)
       ├─ chi (Final Azimuth)
       ├─ final_entropy (Final Entropy)
       ├─ convergence_flag (Converged?)
       │
       ▼
Output Files
├── output/mem_inversion_result.npz
├── output/inversion_summary.txt
└── output/inversion_intermediate_*.npz
```

---

## Physical Model

### 1. Disk Velocity Field

**Outer** (r ≥ r₀): Power-law rotation
$$\Omega(r) = \Omega_0 \left(\frac{r}{r_0}\right)^p$$

**Inner** (r < r₀): Adaptive deceleration sequence
- Use cosine or other smooth functions for transition
- Avoid physical singularities

**Linear Velocity**:
$$v_\phi(r) = r \cdot \Omega(r)$$

### 2. Line Model (Weak Field Approximation)

Let Gaussian line center be λ₀, width be σ, dimensionless deviation:
$$d = \frac{\lambda - \lambda_0}{\sigma}$$

Then Gaussian basis:
$$G(d) = \exp(-d^2)$$

#### Intensity (Stokes I)
$$I(\lambda) = I_c + a \cdot G(d)$$
Where $I_c = 1$ (continuum), $a$ is amplitude (positive=emission, negative=absorption)

#### Circular Polarization (Stokes V)
$$V(\lambda) = C_g \cdot B_\text{los} \cdot a \cdot G(d) \cdot \frac{d}{\sigma}$$
Where $C_g$ is Zeeman coefficient, $B_\text{los}$ is Line-of-Sight magnetic field

#### Linear Polarization (Stokes Q, U)
$$Q(\lambda) = -C_2 \cdot B_\perp^2 \cdot a \cdot \frac{G(d)}{\sigma^2} \cdot (1 - 2d^2) \cdot \cos(2\chi)$$
$$U(\lambda) = -C_2 \cdot B_\perp^2 \cdot a \cdot \frac{G(d)}{\sigma^2} \cdot (1 - 2d^2) \cdot \sin(2\chi)$$

Where:
- $B_\perp$ is perpendicular plane magnetic field strength
- $\chi$ is magnetic field azimuth (radians)
- $C_2$ is second-order Zeeman coefficient

### 3. Velocity to Observation Frequency Mapping

Doppler Shift:
$$v = c \frac{\lambda - \lambda_\text{ref}}{\lambda_\text{ref}}$$

Or Frequency Space:
$$\nu = \nu_0 \left(1 - \frac{v}{c}\right)$$

### 4. Disk Integration and Summation

Contribution of each pixel to the observed spectrum:
$$S_\text{obs}(\lambda) = \sum_i w_i \cdot S_\text{local}(i, \lambda)$$

Where $w_i$ is pixel weight (area/visibility factor, etc.).

---

## Extension and Integration

### Custom Line Model

1. Inherit `BaseLineModel`
2. Implement `compute_local_profile()` method
3. Return `{'I': ..., 'V': ..., 'Q': ..., 'U': ...}`

```python
class MyCustomLineModel(BaseLineModel):
    def compute_local_profile(self, wl_grid, amp, **kwargs):
        # Custom calculation logic
        return {'I': I, 'V': V, 'Q': Q, 'U': U}
```

### New Observation Format Support

1. Add parsing function in `SpecIO.py`
2. Return `ObservationProfile` object
3. Integrate into `obsProfSetInRange()`

### Inversion Method Extension

1. Create new module under `core/` (e.g., `tomography_mcmc.py`)
2. Implement interface similar to `run_mem_inversion()`
3. Use existing config and result containers
4. Expose new interface in main entry (`pyzeetom/tomography.py`)

---

## Typical Development Workflow

### Step 1: Problem Diagnosis
- Use `tomography_config.validate()` to check parameters
- View `tomography_result.create_summary()` to understand output

### Step 2: Forward Verification
```python
from pyzeetom import tomography
results = tomography.forward_tomography('input/params.txt', verbose=2)
```

### Step 3: Inversion Debugging
```python
results = tomography.inversion_tomography('input/params.txt', verbose=2)
```

### Step 4: Model Extension
- Modify `disk_geometry.py` to add new geometry models
- Inherit `BaseLineModel` to implement custom lines
- Configure parameters in `tomography_config.py`

### Step 5: Performance Optimization
- Use cache and stream management in `mem_optimization.py`
- Track performance metrics with `mem_monitoring.py`
- Adjust convergence parameters according to `iteration_manager.py`

---

## File Size and Complexity Overview

| Module | Size | Main Responsibility |
|-----|------|--------|
| mainFuncs.py | 37 KB | Parameter parsing, compatibility |
| velspace_DiskIntegrator.py | 27 KB | Core physics integration |
| SpecIO.py | 27 KB | Spectrum I/O |
| tomography_inversion.py | 34 KB | MEM inversion workflow |
| tomography_config.py | 21 KB | Config container |
| mem_optimization.py | 19 KB | MEM optimization acceleration |
| mem_tomography.py | 19 KB | MEM adapter layer |
| mem_generic.py | 17 KB | Generic MEM algorithm |
| tomography_result.py | 16 KB | Result container |
| grid_tom.py | 14 KB | Grid generation |
| iteration_manager.py | 13 KB | Iteration control |
| mem_monitoring.py | 12 KB | Monitoring & Logging |
| local_linemodel_basic.py | 8 KB | Line model |
| tomography_forward.py | 7.1 KB | Forward workflow |
| disk_geometry.py | 7.8 KB | Disk geometry |

**Total**: Approx. 327 KB of core codebase.

---

## References and Design Principles

### Core Algorithms
- Skilling & Bryan (1984): Maximum Entropy Image Reconstruction
- Hobson & Lasenby (1998): Magnetic field inversion using entropy methods

### Design Patterns
- **Layered Architecture**: UI → Config → Workflow → Physics → Tools
- **Config Object**: Type-safe parameter encapsulation
- **Result Container**: Unified output structure
- **Adapter Pattern**: Decoupling of generic algorithms and project-specific physics
- **Callback Design**: MEM optimizer is project-agnostic

### Extensibility Principles
- New Line Model: Inherit `BaseLineModel`
- New Geometry Model: Modify `disk_geometry.py`
- New Obs Format: Extend `SpecIO.py`
- New Inversion Method: Create new workflow module

---

**Documentation Maintenance**: Update after every major refactoring
**Last Modified**: 2025-11-15
**Contributors**: Tianqi Cang
