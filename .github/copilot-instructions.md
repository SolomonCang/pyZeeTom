
# pyZeeTom Copilot Quick Guide & Project Architecture

**Last Updated**: 2025-11-15
**Version**: v 0.3.0

## Quick Navigation

- 📐 **Full Architecture Documentation**: See `docs/ARCHITECTURE.md`
- 🎯 **Quick Start**: [Quick Start](#quick-start)
- 🔧 **Core Modules**: [Core Architecture](#core-architecture)
- 📊 **Data Flow**: [Data Flow & Workflow](#data-flow--workflow)
- 🧪 **Dev Guide**: [Development & Style Conventions](#development--style-conventions)

---

## Project Overview

**pyZeeTom** is a tomography tool for the inversion and forward modeling of 4 Stokes parameters (I, Q, U, V) polarization spectra.

### Physical Scenario
- **Central Object + Circumstellar Matter**: A central object surrounded by circumstellar matter (dust clumps, disks, planets, etc.) orbiting in rigid body or differential rotation.
- **Phase Observation**: The observer and the central object are in the same inertial frame, observing different viewing angles only through the "phase" brought by the object's rotation.
- **Multi-channel Observation**: Polarization spectra of Stokes I and VQU components can be obtained for each observation phase.
- **Working Mode**: Forward modeling + MEM inversion method.

---

## Quick Start

### Forward Synthesis
```python
from pyzeetom import tomography
results = tomography.forward_tomography('input/params_tomog.txt', verbose=1)
# Returns List[ForwardModelResult], each element corresponds to an observation phase
```

### MEM Inversion
```python
result = tomography.inversion_tomography('input/params_tomog.txt', verbose=1)
# Returns InversionResult, containing reconstructed magnetic field distribution (B_los, B_perp, chi)
```

---

## Core Architecture

### Layered Design

```
┌─ pyzeetom/tomography.py ─────────────────┐  User Interface Layer
│  forward_tomography() / inversion_tomography()
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐  Workflow Execution Layer
│  tomography_forward.py                   │
│  tomography_inversion.py                 │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐  Config & Result Layer
│  tomography_config.py (Config objects)   │
│  tomography_result.py (Result objects)   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐  Physics Calculation Layer
│  velspace_DiskIntegrator.py (Core Integ) │
│  local_linemodel_basic.py (Line Model)   │
│  mem_tomography.py (MEM Adapter)         │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐  Basic Utility Layer
│  grid_tom.py (Grid)                      │
│  disk_geometry.py (Disk Geometry)        │
│  SpecIO.py (Spectrum IO)                 │
│  mainFuncs.py (Param Parsing)            │
│  mem_generic.py (MEM Algorithm)          │
│  mem_iteration_manager.py (Iteration Ctrl)   │
│  mem_optimization.py (Cache/Opt)         │
│  mem_monitoring.py (Monitoring/Log)      │
└─────────────────────────────────────────┘
```

### 1. core/ Physics & Numerical Core

| File | Size | Function |
|------|------|----------|
| **velspace_DiskIntegrator.py** | 27 KB | Velocity space integration, disk model, Stokes synthesis |
| **tomography_inversion.py** | 34 KB | MEM inversion workflow execution engine |
| **tomography_config.py** | 21 KB | Forward/Inversion configuration containers (dataclass) |
| **SpecIO.py** | 27 KB | Spectrum data I/O (multi-format support) |
| **mainFuncs.py** | 37 KB | Parameter parsing, backward compatibility |
| **mem_tomography.py** | 19 KB | MEM inversion adapter layer (project-specific parameterization) |
| **mem_optimization.py** | 19 KB | MEM optimization acceleration, caching, data flow management |
| **mem_generic.py** | 17 KB | Generic MEM algorithm (project-agnostic) |
| **tomography_result.py** | 16 KB | Forward/Inversion result containers |
| **grid_tom.py** | 14 KB | Annular disk grid generation (equal Δr layering) |
| **mem_iteration_manager.py** | 13 KB | MEM iteration control, convergence check, intermediate saving |
| **mem_monitoring.py** | 12 KB | Inversion monitoring, performance metrics, logging |
| **local_linemodel_basic.py** | 8 KB | Weak-field Gaussian Zeeman line model |
| **tomography_forward.py** | 7.1 KB | Forward workflow execution |
| **disk_geometry.py** | 7.8 KB | Disk geometry and dynamics parameters |

### 2. pyzeetom/ Main Entry & Scheduling

| File | Function |
|------|----------|
| **tomography.py** | Main entry point, providing `forward_tomography()` and `inversion_tomography()` APIs |
| **__init__.py** | Package initialization |

---

## Data Flow & Workflow

### Forward Workflow (Forward Synthesis)

```
Input Data
├── params_tomog.txt (Master Params)
├── lines.txt (Line Params: wl0, sigWl, g)
└── inSpec/*.lsd (Obs Data)
       │
       ▼
readParamsTomog() / SpecIO.obsProfSetInRange() / LineData()
       │
       ├─ ParamObject (Dynamics params, formats, etc.)
       ├─ [ObservationProfile] (Obs profile set)
       └─ LineData (Line params)
       │
       ▼
ForwardModelConfig (Config Container)
       │
       ├─ SimpleDiskGeometry (Disk Grid + Dynamics)
       ├─ GaussianZeemanWeakLineModel (Line Model)
       └─ validate()
       │
       ▼
run_forward_synthesis() [tomography_forward.py]
       │
       ├─ FOR each phase:
       │  ├─ VelspaceDiskIntegrator.compute_spectrum_single_phase()
       │  │  ├─ Compute velocity and B-field projection per pixel
       │  │  ├─ Call line_model.compute_local_profile()
       │  │  │  └─ Return {I, V, Q, U}
       │  │  └─ Velocity space integration synthesis
       │  │
       │  └─ ForwardModelResult(Phase Result)
       │
       ▼
Output Files
├── output/model_phase_0.lsd
├── output/model_phase_1.lsd
└── output/outFitSummary.txt
```

### Inversion Workflow (MEM Inversion)

```
Forward Result + Obs Data
       │
       ├─ Synthetic Stokes Spectra {I, V, Q, U}
       ├─ Observed Stokes Spectra {Iobs, Vobs, Qobs, Uobs}
       └─ Initial B-field Guess {Blos_0, Bperp_0, chi_0}
       │
       ▼
InversionConfig (Config Container)
       │
       ├─ forward_config
       ├─ max_iterations, convergence_threshold
       └─ entropy_regularization
       │
       ▼
run_mem_inversion() [tomography_inversion.py]
       │
       ├─ MEMTomographyAdapter (Init Adapter)
       ├─ VelspaceDiskIntegrator (Init Integrator)
       ├─ IterationManager (Iteration Control)
       │
       ├─ FOR iteration:
       │  ├─ VelspaceDiskIntegrator.compute_spectrum() -> S_syn (Synthetic Spec)
       │  ├─ _compute_response_matrix() -> Resp (Response Matrix)
       │  ├─ MEMTomographyAdapter.pack_image_vector() -> Image (Param Vector)
       │  │
       │  ├─ MEMOptimizer.iterate(Image, S_syn, Data, Resp)
       │  │  ├─ MEMTomographyAdapter.compute_entropy_callback() (Calc Entropy S, ∇S)
       │  │  └─ MEMTomographyAdapter.compute_constraint_callback() (Calc χ², ∇χ²)
       │  │
       │  ├─ MEMTomographyAdapter.unpack_image_vector() -> (Blos, Bperp, chi)
       │  └─ Convergence Check
       │
       ▼
InversionResult
       │
       ├─ B_los (Final LOS B-field)
       ├─ B_perp (Final Perp B-field)
       ├─ chi (Final Azimuth)
       ├─ final_entropy
       └─ convergence_flag
       │
       ▼
Output Files
├── output/mem_inversion_result.npz
├── output/inversion_summary.txt
└── output/inversion_intermediate_*.npz
```

### MEM Adapter Layer (mem_tomography.py)

The `MEMTomographyAdapter` class acts as a bridge between the generic MEM optimizer (`mem_generic.py`) and the specific physical problem:

1.  **Parameter Mapping**: Packs/unpacks physical parameters (`MagneticFieldParams`, `BrightnessDisk`) into a 1D `Image` vector for the optimizer.
2.  **Entropy Definition**: Implements entropy functions for different physical quantities:
    *   **Brightness/Bperp**: Standard positive entropy $S = - \sum w_i (x \ln(x/def) - x + def)$
    *   **Blos**: Symmetric entropy (allows positive/negative values)
    *   **chi**: Smoothness/Periodicity entropy
3.  **Constraint Calculation**: Computes $\chi^2$ and its gradient, providing a simple caching mechanism (`_constraint_cache`) to accelerate repeated calculations.
4.  **Boundary Constraints**: Enforces physical constraints (e.g., Brightness > 0).

---

## Physical Model

### Disk Velocity Field

**Outer** (r ≥ r₀): Power-law rotation
$$\Omega(r) = \Omega_0 \left(\frac{r}{r_0}\right)^p, \quad v_\phi = r \cdot \Omega(r)$$

**Inner** (r < r₀): Adaptive deceleration sequence (smooth transition)

### Line Model (Weak Field Approximation)

Let dimensionless deviation $d = (\lambda - \lambda_0) / \sigma$, Gaussian basis $G(d) = \exp(-d^2)$

#### Stokes I (Intensity)
$$I(\lambda) = 1 + a \cdot G(d)$$

#### Stokes V (Circular Polarization)
$$V(\lambda) = C_g \cdot B_{\text{los}} \cdot a \cdot G(d) \cdot \frac{d}{\sigma}$$

#### Stokes Q, U (Linear Polarization)
$$Q(\lambda) = -C_2 \cdot B_\perp^2 \cdot a \cdot \frac{G(d)}{\sigma^2} \cdot (1-2d^2) \cdot \cos(2\chi)$$
$$U(\lambda) = -C_2 \cdot B_\perp^2 \cdot a \cdot \frac{G(d)}{\sigma^2} \cdot (1-2d^2) \cdot \sin(2\chi)$$

Where:
- $a$ is amplitude (positive=emission, negative=absorption)
- $B_{\text{los}}$ is Line-of-Sight magnetic field
- $B_\perp, \chi$ are perpendicular magnetic field and azimuth angle

---

## Development & Style Conventions

### Naming & Unit Conventions
- All pixel attributes (r, phi, Blos, etc.) are 1D arrays, consistent with pixel count.
- Velocity unit: km/s (primary)
- Magnetic field: Gauss
- Azimuth: Radians

### Array Shape Conventions
- Grid pixels: (Npix,)
- Wavelength/Frequency: (Nlambda,)
- Stokes spectra: (Nlambda,) or (Nlambda, Nphase)
- B-field parameter derivatives: (Nlambda, Npix)

### Config Object Design
```python
# Use dataclass instead of dictionary
@dataclass
class ForwardModelConfig:
    par: Any
    obsSet: List[Any]
    lineData: BasicLineData
    # ... params and type annotations
    
    def validate(self) -> bool:
        # Validate parameter consistency
        pass
```

### Spectrum Output Consistency

When using `SpecIO.write_model_spectrum()`, explicitly specify the output format:
```python
SpecIO.write_model_spectrum(
    filename='output/model.lsd',
    wavelength=wl,
    spec_i=I_spec,
    spec_v=V_spec,
    file_type_hint='lsd_pol'  # Explicitly specify format
)
```

Supported formats:
- `lsd_i`: LSD intensity only (3 columns)
- `lsd_pol`: LSD full polarimetry (I,V,Q,U,σ)
- `spec_i`: Simple spectrum (λ, I)
- `spec_pol`: Spectrum + Polarimetry (Wav, Int, Pol, σ)

### Main Entry Convention
- User Entry: `pyzeetom/tomography.py`
- Ensure `PYTHONPATH` includes the project root directory before running.

---

## Typical Extension Points

### Custom Line Model
Inherit `BaseLineModel` and implement `compute_local_profile()`:
```python
from core.local_linemodel_basic import BaseLineModel

class MyLineModel(BaseLineModel):
    def compute_local_profile(self, wl_grid, amp, Blos=None, **kwargs):
        # Custom calculation logic
        return {'I': I, 'V': V, 'Q': Q, 'U': U}

# Use in config
config.line_model = MyLineModel()
```

### New Observation Format Support
Extend in `SpecIO.py`:
```python
def load_custom_format(filename):
    # Parse custom format
    return ObservationProfile(...)

# Integrate into obsProfSetInRange()
```

### New Inversion Method
Create a new workflow module (e.g., `tomography_mcmc.py`):
```python
def run_mcmc_inversion(config: InversionConfig) -> InversionResult:
    # Use existing ForwardModelConfig / InversionResult containers
    pass

# Expose interface in main entry
```

---

## Core File Quick Reference

| Requirement | File | Key Function/Class |
|-------------|------|--------------------|
| Forward Synthesis | tomography_forward.py | `run_forward_synthesis()` |
| MEM Inversion | tomography_inversion.py | `run_mem_inversion()` |
| Param Parsing | mainFuncs.py | `readParamsTomog()` |
| Spectrum I/O | SpecIO.py | `obsProfSetInRange()`, `write_model_spectrum()` |
| Grid Gen | grid_tom.py | `diskGrid` |
| Velocity Integ | velspace_DiskIntegrator.py | `VelspaceDiskIntegrator` |
| Line Model | local_linemodel_basic.py | `GaussianZeemanWeakLineModel` |
| MEM Algo | mem_generic.py | `MEMOptimizer` |
| Iteration Ctrl | iteration_manager.py | `IterationManager` |

---

## Notes

⚠️ **Common Errors**
- ❌ B-field array length inconsistent with pixel count → ValueError
- ❌ Velocity unit confusion (km/s vs m/s)
- ❌ Line parameter file format irregular → Parsing failure
- ❌ Observation data format specified incorrectly → Data read failure

✅ **Best Practices**
- Always use `config.validate()` to check parameters
- Use `result.create_summary()` to understand output
- Use `verbose=2` for debugging
- Save intermediate results for issue tracking

---

## Full Documentation

For more details, please refer to **`docs/ARCHITECTURE.md`**, including:
- Detailed physical model derivation
- Data flow diagrams
- Module interface descriptions
- References and design principles
- Performance optimization guide
