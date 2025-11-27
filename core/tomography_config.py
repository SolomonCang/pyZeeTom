"""
tomography_config.py - Configuration Objects for Forward and Inversion Workflows

This module provides unified configuration containers encapsulating all parameters required for forward and inversion workflows.
Using configuration objects instead of dictionaries provides:
  - Type safety and IDE autocompletion
  - Built-in validation logic
  - Clear documentation and type annotations
  - Convenient serialization/deserialization
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
import numpy as np
from datetime import datetime

from core.local_linemodel_basic import LineData as BasicLineData
from core.disk_geometry_integrator import SimpleDiskGeometry
from core.grid_tom import diskGrid
from core.local_linemodel_basic import GaussianZeemanWeakLineModel, ConstantAmpLineModel


@dataclass
class ForwardModelConfig:
    """Forward Model Configuration Container
    
    Encapsulates all parameters required for the forward workflow (run_forward_synthesis), including:
    - Observation datasets and parameter objects
    - Geometry and physical models
    - Dynamics parameters
    - Magnetic field initial conditions
    - Output control options
    """

    # ============ Core Inputs ============

    par: Any  # Parameter object returned by readParamsTomog
    """Parameter object containing all configuration information"""

    obsSet: List[Any]
    """List of observation datasets, each element is an ObservationProfile object"""

    lineData: Optional[BasicLineData] = None
    """Line parameter data"""

    # ============ Physical Model ============

    geom: Optional[SimpleDiskGeometry] = None
    """Disk geometry object containing grid and dynamics parameters"""

    line_model: Optional[Any] = None
    """Line model object providing compute_local_profile() method"""

    # ============ Dynamics Parameters ============

    velEq: float = 100.0
    """Equatorial velocity, unit km/s"""

    pOmega: float = 0.0
    """Differential rotation index"""

    radius: float = 1.0
    """Reference radius, unit R_sun"""

    # ============ Instrument Parameters ============

    inst_fwhm_kms: float = 0.1
    """Instrument profile FWHM (km/s)"""

    # ============ Line Model Parameters ============

    enable_v: bool = True
    """Whether to enable Stokes V calculation"""

    enable_qu: bool = True
    """Whether to enable Stokes Q/U calculation"""

    line_area: float = 1.0
    """Line area factor"""

    normalize_continuum: bool = True
    """Whether to normalize continuum"""

    amp_init: Optional[np.ndarray] = None
    """Initial amplitude distribution"""

    B_los_init: Optional[np.ndarray] = None
    """Initial line-of-sight magnetic field"""

    B_perp_init: Optional[np.ndarray] = None
    """Initial perpendicular magnetic field"""

    chi_init: Optional[np.ndarray] = None
    """Initial azimuth angle"""

    # ============ Output Control ============

    output_dir: str = "./output"
    """Output directory"""

    save_intermediate: bool = False
    """Whether to save intermediate results"""

    verbose: int = 0
    """Verbosity level: 0=silent, 1=normal, 2=detailed"""

    # ============ Internal State ============

    _validated: bool = field(default=False, init=False, repr=False)
    """Whether configuration is validated"""

    _creation_time: str = field(
        default_factory=lambda: datetime.now().isoformat(),
        init=False,
        repr=False)
    """Creation time"""

    def validate(self) -> None:
        """Validate configuration completeness and consistency
        
        Raises
        ------
        ValueError
            When any required parameter is None or invalid
        AssertionError
            When parameter dimensions do not match
        """
        if self.par is None:
            raise ValueError("Parameter object (par) cannot be None")

        if not self.obsSet or len(self.obsSet) == 0:
            raise ValueError("Observation dataset (obsSet) cannot be empty")

        if self.lineData is None:
            raise ValueError("Line parameters (lineData) cannot be None")

        if self.geom is None:
            raise ValueError("Geometry object (geom) cannot be None")

        if self.line_model is None:
            raise ValueError("Line model (line_model) cannot be None")

        # Check dynamics parameters
        if self.velEq <= 0:
            raise ValueError(
                f"Equatorial velocity (velEq) must be positive, got {self.velEq}"
            )

        if self.radius <= 0:
            raise ValueError(
                f"Reference radius (radius) must be positive, got {self.radius}"
            )

        # Check amplitude and magnetic field array dimensions
        npix = self.geom.grid.numPoints

        if self.amp_init is not None:
            if len(self.amp_init) != npix:
                raise AssertionError(
                    f"amp_init dimension ({len(self.amp_init)}) does not match grid pixel count ({npix})"
                )
        else:
            # Create default all-ones array (unit amplitude)
            self.amp_init = np.ones(npix)

        if self.B_los_init is not None:
            if len(self.B_los_init) != npix:
                raise AssertionError(
                    f"B_los_init dimension ({len(self.B_los_init)}) does not match grid pixel count ({npix})"
                )
        else:
            # Create default all-zeros array
            self.B_los_init = np.zeros(npix)

        if self.B_perp_init is not None:
            if len(self.B_perp_init) != npix:
                raise AssertionError(
                    f"B_perp_init dimension ({len(self.B_perp_init)}) does not match grid pixel count ({npix})"
                )
        else:
            self.B_perp_init = np.zeros(npix)

        if self.chi_init is not None:
            if len(self.chi_init) != npix:
                raise AssertionError(
                    f"chi_init dimension ({len(self.chi_init)}) does not match grid pixel count ({npix})"
                )
        else:
            self.chi_init = np.zeros(npix)

        # Check output directory
        from pathlib import Path
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._validated = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary
        
        Returns
        -------
        dict
            Configuration dictionary, numpy arrays converted to lists, objects converted to string representation
        """
        self.validate()

        result = {
            'par': str(type(self.par).__name__),
            'obsSet_count': len(self.obsSet),
            'lineData_wl0':
            float(self.lineData.wl0) if self.lineData else None,
            'geom_npix': self.geom.grid.numPoints,
            'line_model': str(type(self.line_model).__name__),
            'velEq': float(self.velEq),
            'pOmega': float(self.pOmega),
            'radius': float(self.radius),
            'B_los_init_stats': {
                'min': float(self.B_los_init.min()),
                'max': float(self.B_los_init.max()),
                'mean': float(self.B_los_init.mean()),
            } if self.B_los_init is not None else None,
            'B_perp_init_stats': {
                'min': float(self.B_perp_init.min()),
                'max': float(self.B_perp_init.max()),
                'mean': float(self.B_perp_init.mean()),
            } if self.B_perp_init is not None else None,
            'output_dir': self.output_dir,
            'save_intermediate': self.save_intermediate,
            'verbose': self.verbose,
            'creation_time': self._creation_time,
        }

        return result

    @classmethod
    def from_par(cls,
                 par,
                 obsSet,
                 lineData,
                 verbose=False) -> 'ForwardModelConfig':
        """Create forward configuration from legacy parameter object
        
        Convert parameter object returned by readParamsTomog, observation set, and line data
        into ForwardModelConfig object for integration with new workflow.
        
        Parameters
        ----------
        par : readParamsTomog
            Parameter object from mf.readParamsTomog()
        obsSet : list
            Observation set from SpecIO.obsProfSetInRange()
        lineData : LineData
            Line parameters from BasicLineData()
        verbose : bool
            Detailed output
            
        Returns
        -------
        ForwardModelConfig
            New forward configuration object
            
        Raises
        ------
        ValueError
            When required parameters are missing or invalid
        """

        if verbose:
            print(
                "[ForwardModelConfig.from_par] Starting parameter conversion..."
            )

        # Extract dynamics parameters
        velEq = float(getattr(par, 'velEq', getattr(par, 'vsini', 100.0)))
        pOmega = float(getattr(par, 'pOmega', 0.0))
        radius = float(getattr(par, 'radius', 1.0))
        period = float(getattr(par, 'period', 1.0))
        inclination_deg = float(getattr(par, 'inclination', 90.0))

        # Calculate grid outer radius
        nr = int(getattr(par, 'nRingsStellarGrid', 60))
        Vmax = float(getattr(par, 'Vmax', 0.0))

        if abs(pOmega + 1.0) > 1e-6:
            # General case: infer r_out from Vmax
            # FIX: Use vsini (projected) instead of velEq (intrinsic) to avoid inclination dependency
            vsini_val = float(getattr(par, 'vsini', 10.0))
            r_out_stellar = (Vmax / vsini_val)**(
                1.0 / (pOmega + 1.0)) if Vmax > 0 else float(
                    getattr(par, 'r_out', 5.0))
            r_out_grid = radius * r_out_stellar
        else:
            # Special case: pOmega = -1 (constant angular momentum)
            r_out_stellar = float(getattr(par, 'r_out', 5.0))
            r_out_grid = radius * r_out_stellar

        if verbose:
            print(f"[from_par] Grid: nr={nr}, r_out={r_out_grid:.3f} R_sun")

        # Build grid
        grid = diskGrid(nr=nr, r_in=0.0, r_out=r_out_grid, verbose=verbose)

        # Build geometry
        enable_occultation = bool(getattr(par, 'enable_stellar_occultation',
                                          0))
        geom = SimpleDiskGeometry(
            grid=grid,
            inclination_deg=inclination_deg,
            pOmega=pOmega,
            r0=radius,
            period=period,
            enable_stellar_occultation=enable_occultation,
            stellar_radius=radius,
            B_los=np.zeros(grid.numPoints),
            B_perp=np.zeros(grid.numPoints),
            chi=np.zeros(grid.numPoints),
            amp=np.ones(grid.numPoints))

        if verbose:
            print(
                f"[from_par] Geometry: inc={inclination_deg}°, period={period}d"
            )

        # Build line model
        k_qu = float(getattr(par, 'lineKQU', 1.0))
        enable_v = bool(getattr(par, 'lineEnableV', 1))
        enable_qu = bool(getattr(par, 'lineEnableQU', 1))
        amp_const = float(getattr(par, 'lineAmpConst', -0.5))

        base_model = GaussianZeemanWeakLineModel(lineData,
                                                 k_QU=k_qu,
                                                 enable_V=enable_v,
                                                 enable_QU=enable_qu)
        line_model = ConstantAmpLineModel(base_model, amp=amp_const)

        if verbose:
            print(
                f"[from_par] Line Model: amp={amp_const}, k_QU={k_qu}, V={enable_v}, QU={enable_qu}"
            )

        # Calculate instrument FWHM
        if hasattr(par, 'compute_instrument_fwhm'):
            par.compute_instrument_fwhm(lineData.wl0, verbose=False)
        instrument_res = float(getattr(par, 'instrumentFWHM', 0.1))

        # Create configuration object
        config = cls(par=par,
                     obsSet=obsSet,
                     lineData=lineData,
                     geom=geom,
                     line_model=line_model,
                     velEq=velEq,
                     pOmega=pOmega,
                     radius=radius,
                     inst_fwhm_kms=instrument_res,
                     enable_v=enable_v,
                     enable_qu=enable_qu,
                     amp_init=np.full(grid.numPoints, amp_const),
                     B_los_init=np.zeros(grid.numPoints),
                     B_perp_init=np.zeros(grid.numPoints),
                     chi_init=np.zeros(grid.numPoints),
                     output_dir=str(getattr(par, 'outputDir', './output')),
                     save_intermediate=bool(
                         getattr(par, 'saveIntermediate', False)),
                     verbose=1 if verbose else 0)

        # Store extra attributes for later use
        config._instrument_res = instrument_res
        config._nr = nr
        config._r_out_grid = r_out_grid
        config._phases = np.array(
            [float(getattr(obs, 'phase', 0.0)) for obs in obsSet])

        if verbose:
            print(
                f"[from_par] Conversion complete: {len(obsSet)} observed phases"
            )

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ForwardModelConfig':
        """Deserialize configuration from dictionary
        
        Note: This method is mainly for debugging and logging.
        Actual configuration reconstruction requires creating from original data sources (parameter file, observation data, etc.).
        """
        raise NotImplementedError(
            "from_dict() requires original data sources. Please reconstruct configuration using parameter file and observation data."
        )

    def create_summary(self) -> str:
        """Generate configuration summary string
        
        Returns
        -------
        str
            Formatted configuration summary, suitable for log output
        """
        self.validate()

        lines = [
            "=" * 70,
            "Forward Configuration Summary (ForwardModelConfig)",
            "=" * 70,
            f"Observation Dataset: {len(self.obsSet)} phases",
            f"Line Parameters: λ₀ = {self.lineData.wl0:.4f} nm",
            f"Grid Pixels: {self.geom.grid.numPoints}",
            f"Dynamics Parameters:",
            f"  - Equatorial Velocity: {self.velEq:.1f} km/s",
            f"  - Differential Rotation Index: {self.pOmega:.3f}",
            f"  - Reference Radius: {self.radius:.3f} R_sun",
            f"Geometry Parameters:",
            f"  - Inclination: {self.geom.inclination_rad * 180 / np.pi:.1f}°",
            f"  - Period: {self.geom.period:.4f} d",
            f"Magnetic Field Init:",
            f"  - B_los: [{self.B_los_init.min():.1f}, {self.B_los_init.max():.1f}] G",
            f"  - B_perp: [{self.B_perp_init.min():.1f}, {self.B_perp_init.max():.1f}] G",
            f"  - χ: [{self.chi_init.min():.3f}, {self.chi_init.max():.3f}] rad",
            f"Output Settings:",
            f"  - Directory: {self.output_dir}",
            f"  - Save Intermediate: {self.save_intermediate}",
            f"  - Verbosity: {self.verbose}",
            "=" * 70,
        ]

        return "\n".join(lines)


@dataclass
class InversionConfig(ForwardModelConfig):
    """MEM Inversion Configuration Container
    
    Inherits from ForwardModelConfig, adding inversion-specific parameters and control options.
    """

    # ============ Inversion Iteration Parameters ============

    num_iterations: int = 10
    """Maximum number of iterations"""

    entropy_weight: float = 1.0
    """Entropy regularization weight (λ_S)"""

    data_weight: float = 1.0
    """Data fitting weight (λ_D)"""

    smoothness_weight: float = 0.1
    """Smoothness regularization weight (λ_R)"""

    # ============ Target Control ============

    target_form: str = 'C'
    """Optimization target form ('C'=Chi^2, 'E'=Entropy)"""

    target_value: float = 1.0
    """Optimization target value"""

    # ============ Convergence Criteria ============

    convergence_threshold: float = 1e-3
    """Convergence threshold (test_aim)"""

    max_iterations: Optional[int] = None
    """Maximum number of iterations (if None, use num_iterations)"""

    # ============ Visibility Control ============

    force_all_pixels_visible: bool = True
    """Whether to force all pixels visible (disable entropy enhancement for invisible pixels)"""

    # ============ Fitting Switches ============

    initial_B_los: Optional[np.ndarray] = None
    """Initial B_los model (if None, use B_los_init)"""

    initial_B_perp: Optional[np.ndarray] = None
    """Initial B_perp model (if None, use B_perp_init)"""

    initial_chi: Optional[np.ndarray] = None
    """Initial χ model (if None, use chi_init)"""

    initial_brightness: Optional[np.ndarray] = None
    """Initial brightness model (if None, use amp_init)"""

    fit_brightness: bool = True
    """Whether to fit brightness distribution"""

    fit_B_los: bool = True
    """Whether to fit line-of-sight magnetic field"""

    fit_B_perp: bool = True
    """Whether to fit perpendicular magnetic field"""

    fit_chi: bool = True
    """Whether to fit magnetic field azimuth"""

    # ============ Saving Strategy ============

    save_every_iter: int = 1
    """Save checkpoint every N iterations"""

    save_final_only: bool = False
    """Whether to save only final result"""

    def validate(self) -> None:
        """Validate inversion configuration, including parent validation
        
        Raises
        ------
        ValueError
            When inversion-specific parameters are invalid
        """
        # Call parent validation first
        super().validate()

        # Validate inversion-specific parameters
        if self.num_iterations < 0:
            raise ValueError(
                f"Number of iterations must be non-negative, got {self.num_iterations}"
            )

        if not 0 < self.entropy_weight <= 10.0:
            raise ValueError(
                f"Entropy weight should be in (0, 10], got {self.entropy_weight}"
            )

        if not 0 < self.data_weight <= 10.0:
            raise ValueError(
                f"Data weight should be in (0, 10], got {self.data_weight}")

        if not 0 <= self.smoothness_weight <= 10.0:
            raise ValueError(
                f"Smoothness weight should be in [0, 10], got {self.smoothness_weight}"
            )

        if self.convergence_threshold <= 0:
            raise ValueError(
                f"Convergence threshold must be positive, got {self.convergence_threshold}"
            )

        if self.save_every_iter < 1:
            raise ValueError(
                f"Save interval must be >= 1, got {self.save_every_iter}")

        # If initial model is specified, check dimensions
        npix = self.geom.grid.numPoints
        if self.initial_B_los is not None and len(self.initial_B_los) != npix:
            raise AssertionError(
                f"initial_B_los dimension ({len(self.initial_B_los)}) does not match grid ({npix})"
            )

        if self.initial_B_perp is not None and len(
                self.initial_B_perp) != npix:
            raise AssertionError(
                f"initial_B_perp dimension ({len(self.initial_B_perp)}) does not match grid ({npix})"
            )

        if self.initial_chi is not None and len(self.initial_chi) != npix:
            raise AssertionError(
                f"initial_chi dimension ({len(self.initial_chi)}) does not match grid ({npix})"
            )

    def get_mem_adapter_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for MEMTomographyAdapter
        
        Returns
        -------
        dict
            Configuration parameters required by MEMTomographyAdapter
        """
        self.validate()

        npix = self.geom.grid.numPoints

        return {
            'fit_brightness':
            self.fit_brightness,
            'fit_magnetic':
            False,  # Use fine-grained control
            'fit_B_los':
            self.fit_B_los,
            'fit_B_perp':
            self.fit_B_perp,
            'fit_chi':
            self.fit_chi,
            'entropy_weights_blos':
            self.geom.grid.area
            if hasattr(self.geom.grid, 'area') else np.ones(npix),
            'entropy_weights_bperp':
            self.geom.grid.area
            if hasattr(self.geom.grid, 'area') else np.ones(npix),
            'entropy_weights_chi':
            (self.geom.grid.area *
             0.1) if hasattr(self.geom.grid, 'area') else np.ones(npix) * 0.1,
            'default_blos':
            1.0,
            'default_bperp':
            1.0,
            'default_chi':
            0.0,
        }

    @classmethod
    def from_par(cls,
                 par,
                 obsSet,
                 lineData,
                 verbose=False) -> 'InversionConfig':
        """Create inversion configuration from legacy parameter object
        
        Similar to ForwardModelConfig.from_par(), but adds inversion-specific parameters.
        
        Parameters
        ----------
        par : readParamsTomog
            Parameter object from mf.readParamsTomog()
        obsSet : list
            Observation set from SpecIO.obsProfSetInRange()
        lineData : LineData
            Line parameters from BasicLineData()
        verbose : bool
            Detailed output
            
        Returns
        -------
        InversionConfig
            New inversion configuration object
            
        Raises
        ------
        ValueError
            When required parameters are missing or invalid
        """
        if verbose:
            print(
                "[InversionConfig.from_par] Starting inversion parameter conversion..."
            )

        # First create base forward configuration using parent method
        forward_config = ForwardModelConfig.from_par(par,
                                                     obsSet,
                                                     lineData,
                                                     verbose=verbose)

        # Extract inversion-specific parameters
        num_iterations = int(getattr(par, 'numIterations', 10))
        target_form = str(getattr(par, 'targetForm', 'C'))
        target_value = float(getattr(par, 'targetValue', 1.0))

        entropy_weight = float(getattr(par, 'entropyWeight', 0.01))
        data_weight = float(getattr(par, 'dataWeight', 1.0))
        smoothness_weight = float(getattr(par, 'smoothnessWeight', 0.1))

        # Use test_aim as convergence threshold
        convergence_threshold = float(getattr(par, 'test_aim', 1e-3))

        # Extract fitting switches
        fit_brightness = bool(getattr(par, 'fitBri', 1))
        fit_magnetic = bool(getattr(par, 'fitMag', 1))
        fit_B_los = bool(getattr(par, 'fitBlos', 1))
        fit_B_perp = bool(getattr(par, 'fitBperp', 1))
        fit_chi = bool(getattr(par, 'fitChi', 1))

        # If total magnetic switch is off, force all magnetic components off
        if not fit_magnetic:
            fit_B_los = False
            fit_B_perp = False
            fit_chi = False

        # Extract visibility control
        force_all_pixels_visible = bool(
            getattr(par, 'forceAllPixelsVisible', False))

        # If we are fitting brightness, or to ensure amp parameter is effective during inversion,
        # we need to unpack ConstantAmpLineModel
        if isinstance(forward_config.line_model, ConstantAmpLineModel):
            if verbose:
                print(
                    "[from_par] Unpacking ConstantAmpLineModel to enable brightness fitting"
                )
            forward_config.line_model = forward_config.line_model.base_model

        if verbose:
            print(
                f"[from_par] Inversion Params: iterations={num_iterations}, "
                f"entropy={entropy_weight}, data={data_weight}, smooth={smoothness_weight}"
            )
            print(
                f"[from_par] Fitting Switches: Bri={fit_brightness}, Mag={fit_magnetic} "
                f"(Blos={fit_B_los}, Bperp={fit_B_perp}, Chi={fit_chi})")
            print(
                f"[from_par] Visibility Control: forceAllPixelsVisible={force_all_pixels_visible}"
            )

        # Create inversion configuration (copy all parent fields)
        config = cls(
            par=forward_config.par,
            obsSet=forward_config.obsSet,
            lineData=forward_config.lineData,
            geom=forward_config.geom,
            line_model=forward_config.line_model,
            velEq=forward_config.velEq,
            pOmega=forward_config.pOmega,
            radius=forward_config.radius,
            # Instrument parameters
            inst_fwhm_kms=forward_config.inst_fwhm_kms,
            line_area=forward_config.line_area,
            normalize_continuum=forward_config.normalize_continuum,
            enable_v=forward_config.enable_v,
            enable_qu=forward_config.enable_qu,
            # Initial state
            amp_init=forward_config.amp_init.copy()
            if forward_config.amp_init is not None else None,
            B_los_init=forward_config.B_los_init.copy()
            if forward_config.B_los_init is not None else None,
            B_perp_init=forward_config.B_perp_init.copy()
            if forward_config.B_perp_init is not None else None,
            chi_init=forward_config.chi_init.copy()
            if forward_config.chi_init is not None else None,
            output_dir=forward_config.output_dir,
            save_intermediate=forward_config.save_intermediate,
            verbose=forward_config.verbose,
            # Inversion-specific parameters
            num_iterations=num_iterations,
            target_form=target_form,
            target_value=target_value,
            entropy_weight=entropy_weight,
            data_weight=data_weight,
            smoothness_weight=smoothness_weight,
            convergence_threshold=convergence_threshold,
            # Fitting switches
            fit_brightness=fit_brightness,
            fit_B_los=fit_B_los,
            fit_B_perp=fit_B_perp,
            fit_chi=fit_chi,
            force_all_pixels_visible=force_all_pixels_visible)

        if verbose:
            print("[from_par] Inversion configuration conversion complete")

        return config

    def create_summary(self) -> str:
        """Generate inversion configuration summary
        
        Returns
        -------
        str
            Formatted inversion configuration summary
        """
        parent_summary = super().create_summary()

        lines = [
            "",
            "Inversion Parameter Extension:",
            f"  - Max Iterations: {self.num_iterations}",
            f"  - Convergence Threshold: {self.convergence_threshold:.3e}",
            f"  - Entropy Weight: {self.entropy_weight}",
            f"  - Data Weight: {self.data_weight}",
            f"  - Smoothness Weight: {self.smoothness_weight}",
            f"  - Save Interval: {self.save_every_iter}",
            f"  - Save Final Only: {self.save_final_only}",
            f"  - Force All Pixels Visible: {self.force_all_pixels_visible}",
        ]

        # If initial model is specified, add info
        if self.initial_B_los is not None or self.initial_B_perp is not None or self.initial_chi is not None:
            lines.append("  - Using custom initial model")

        return parent_summary + "\n" + "\n".join(lines)
