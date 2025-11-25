"""
tomography_result.py - Result Objects for Forward and Inversion Workflows

This module provides structured result containers, replacing tuples returned by functions.
Provides:
  - Type-safe data access
  - Built-in statistical and diagnostic methods
  - Convenient serialization/export functions
  - Clear documentation and IDE autocompletion
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime


@dataclass
class ForwardModelResult:
    """Forward Model Result Container
    
    Stores output results of single-phase forward workflow.
    
    Attributes
    ----------
    stokes_i : np.ndarray
        Stokes I component spectrum (shape: Nλ)
    stokes_v : np.ndarray
        Stokes V component spectrum (shape: Nλ)
    stokes_q : Optional[np.ndarray]
        Stokes Q component spectrum (shape: Nλ), optional
    stokes_u : Optional[np.ndarray]
        Stokes U component spectrum (shape: Nλ), optional
    wavelength : np.ndarray
        Wavelength grid (shape: Nλ)
    error : Optional[np.ndarray]
        Observation error (shape: Nλ)
    """

    # ============ Core Results ============

    stokes_i: np.ndarray
    """Stokes I spectrum array (Nλ,)"""

    stokes_v: np.ndarray
    """Stokes V spectrum array (Nλ,)"""

    stokes_q: Optional[np.ndarray] = None
    """Stokes Q spectrum array (Nλ,), optional"""

    stokes_u: Optional[np.ndarray] = None
    """Stokes U spectrum array (Nλ,), optional"""

    wavelength: np.ndarray = field(default_factory=lambda: np.array([]))
    """Wavelength grid (Nλ,)"""

    error: Optional[np.ndarray] = None
    """Observation error (Nλ,)"""

    # ============ Metadata ============

    hjd: Optional[float] = None
    """Heliocentric Julian Date (HJD)"""

    phase_index: int = 0
    """Observation phase index"""

    pol_channel: str = "V"
    """Polarization channel label (only supports: 'I', 'V', 'Q', 'U')"""

    model_name: str = "forward_synthesis"
    """Model name"""

    # ============ Internal State ============

    integrator: Any = field(default=None, repr=False, compare=False)
    """Associated integrator object (VelspaceDiskIntegrator)"""

    _creation_time: str = field(
        default_factory=lambda: datetime.now().isoformat(),
        init=False,
        repr=False)
    """Creation time"""

    def validate(self) -> None:
        """Validate result data integrity and consistency
        
        Raises
        ------
        ValueError
            Data inconsistent or incomplete
        """
        if self.stokes_i is None or len(self.stokes_i) == 0:
            raise ValueError("Stokes I cannot be empty")

        if self.stokes_v is None or len(self.stokes_v) == 0:
            raise ValueError("Stokes V cannot be empty")

        nl = len(self.stokes_i)

        if len(self.stokes_v) != nl:
            raise ValueError(
                f"Stokes V length ({len(self.stokes_v)}) does not match I ({nl})"
            )

        if self.stokes_q is not None and len(self.stokes_q) != nl:
            raise ValueError(
                f"Stokes Q length ({len(self.stokes_q)}) does not match I ({nl})"
            )

        if self.stokes_u is not None and len(self.stokes_u) != nl:
            raise ValueError(
                f"Stokes U length ({len(self.stokes_u)}) does not match I ({nl})"
            )

        if len(self.wavelength) > 0 and len(self.wavelength) != nl:
            raise ValueError(
                f"Wavelength length ({len(self.wavelength)}) does not match spectrum ({nl})"
            )

        if self.error is not None and len(self.error) != nl:
            raise ValueError(
                f"Error length ({len(self.error)}) does not match spectrum ({nl})"
            )

    def get_chi2(self,
                 obs_spectrum: np.ndarray,
                 obs_spectrum_other: Optional[np.ndarray] = None,
                 obs_error: Optional[np.ndarray] = None,
                 obs_error_other: Optional[np.ndarray] = None) -> float:
        """Calculate Chi^2 value with observation data
        
        Calculate Chi^2 for corresponding Stokes component based on pol_channel.
        
        Parameters
        ----------
        obs_spectrum : np.ndarray
            Observed Stokes I spectrum (Nλ,)
        obs_spectrum_other : Optional[np.ndarray]
            Observed Stokes V/Q/U spectrum (Nλ,), used only when pol_channel != 'I'
        obs_error : Optional[np.ndarray]
            Stokes I observation error (Nλ,), defaults to error in result
        obs_error_other : Optional[np.ndarray]
            Stokes V/Q/U observation error (Nλ,)
        
        Returns
        -------
        float
            Chi^2 value = Σ((obs - model) / σ)²
            
        Notes
        -----
        - pol_channel='I': Calculate Chi^2 for Stokes I only
        - pol_channel='V'/'Q'/'U': Calculate Chi^2 for corresponding component (excluding I)
        """
        self.validate()

        if obs_spectrum is None or len(obs_spectrum) != len(self.stokes_i):
            raise ValueError("Observation spectrum dimension mismatch")

        pol_ch = self.pol_channel.upper()

        # Validate pol_channel value
        if pol_ch not in ('I', 'V', 'Q', 'U'):
            raise ValueError(
                f"pol_channel must be 'I', 'V', 'Q' or 'U', got '{pol_ch}'")

        # Stokes I channel
        if pol_ch == 'I':
            sigma_i = obs_error if obs_error is not None else self.error
            if sigma_i is None or np.all(sigma_i == 0):
                residuals = (obs_spectrum - self.stokes_i)**2
            else:
                residuals = ((obs_spectrum - self.stokes_i) / sigma_i)**2
            return float(np.sum(residuals))

        # Stokes V channel
        if pol_ch == 'V':
            if obs_spectrum_other is None:
                raise ValueError(
                    "obs_spectrum_other must be provided when pol_channel='V'")
            if len(obs_spectrum_other) != len(self.stokes_v):
                raise ValueError(
                    "Stokes V observation spectrum dimension mismatch")
            sigma_v = obs_error_other if obs_error_other is not None else (
                self.error if self.error is not None else
                np.ones_like(self.stokes_v) * 1e-5)
            if np.all(sigma_v == 0):
                residuals = (obs_spectrum_other - self.stokes_v)**2
            else:
                residuals = ((obs_spectrum_other - self.stokes_v) / sigma_v)**2
            return float(np.sum(residuals))

        # Stokes Q channel
        if pol_ch == 'Q':
            if obs_spectrum_other is None:
                raise ValueError(
                    "obs_spectrum_other must be provided when pol_channel='Q'")
            if self.stokes_q is None:
                raise ValueError("Stokes Q does not exist in model")
            if len(obs_spectrum_other) != len(self.stokes_q):
                raise ValueError(
                    "Stokes Q observation spectrum dimension mismatch")
            sigma_q = obs_error_other if obs_error_other is not None else (
                self.error if self.error is not None else
                np.ones_like(self.stokes_q) * 1e-5)
            if np.all(sigma_q == 0):
                residuals = (obs_spectrum_other - self.stokes_q)**2
            else:
                residuals = ((obs_spectrum_other - self.stokes_q) / sigma_q)**2
            return float(np.sum(residuals))

        # Stokes U channel
        if pol_ch == 'U':
            if obs_spectrum_other is None:
                raise ValueError(
                    "obs_spectrum_other must be provided when pol_channel='U'")
            if self.stokes_u is None:
                raise ValueError("Stokes U does not exist in the model")
            if len(obs_spectrum_other) != len(self.stokes_u):
                raise ValueError(
                    "Stokes U observation spectrum dimension mismatch")
            sigma_u = obs_error_other if obs_error_other is not None else (
                self.error if self.error is not None else
                np.ones_like(self.stokes_u) * 1e-5)
            if np.all(sigma_u == 0):
                residuals = (obs_spectrum_other - self.stokes_u)**2
            else:
                residuals = ((obs_spectrum_other - self.stokes_u) / sigma_u)**2
            return float(np.sum(residuals))

        # This line should not be reached (pol_ch value already validated)
        raise ValueError(f"Invalid pol_channel value: {pol_ch}")

    def get_relative_residuals(
            self,
            obs_spectrum: np.ndarray,
            obs_spectrum_other: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate relative residuals
        
        Calculate relative residuals for the corresponding Stokes component based on pol_channel.
        
        Parameters
        ----------
        obs_spectrum : np.ndarray
            Observed Stokes I spectrum
        obs_spectrum_other : Optional[np.ndarray]
            Observed Stokes V/Q/U spectrum, used only when pol_channel != 'I'
        
        Returns
        -------
        np.ndarray
            Relative residuals array = (obs - model) / obs
            
        Notes
        -----
        - pol_channel='I': Calculate Stokes I residuals
        - pol_channel='V'/'Q'/'U': Calculate residuals for corresponding component (excluding I)
        """
        self.validate()

        pol_ch = self.pol_channel.upper()

        # Validate pol_channel value
        if pol_ch not in ('I', 'V', 'Q', 'U'):
            raise ValueError(
                f"pol_channel must be 'I', 'V', 'Q' or 'U', received '{pol_ch}'"
            )

        # Stokes I channel
        if pol_ch == 'I':
            safe_obs = np.where(
                np.abs(obs_spectrum) > 1e-10, obs_spectrum, 1e-10)
            return (obs_spectrum - self.stokes_i) / safe_obs

        # Stokes V channel
        if pol_ch == 'V':
            if obs_spectrum_other is None:
                raise ValueError(
                    "obs_spectrum_other must be provided when pol_channel='V'")
            if len(obs_spectrum_other) != len(self.stokes_v):
                raise ValueError(
                    "Stokes V observation spectrum dimension mismatch")
            safe_obs = np.where(
                np.abs(obs_spectrum_other) > 1e-10, obs_spectrum_other, 1e-10)
            return (obs_spectrum_other - self.stokes_v) / safe_obs

        # Stokes Q channel
        if pol_ch == 'Q':
            if obs_spectrum_other is None:
                raise ValueError(
                    "obs_spectrum_other must be provided when pol_channel='Q'")
            if self.stokes_q is None:
                raise ValueError("Stokes Q does not exist in the model")
            if len(obs_spectrum_other) != len(self.stokes_q):
                raise ValueError(
                    "Stokes Q observation spectrum dimension mismatch")
            safe_obs = np.where(
                np.abs(obs_spectrum_other) > 1e-10, obs_spectrum_other, 1e-10)
            return (obs_spectrum_other - self.stokes_q) / safe_obs

        # Stokes U channel
        if pol_ch == 'U':
            if obs_spectrum_other is None:
                raise ValueError(
                    "obs_spectrum_other must be provided when pol_channel='U'")
            if self.stokes_u is None:
                raise ValueError("Stokes U does not exist in the model")
            if len(obs_spectrum_other) != len(self.stokes_u):
                raise ValueError(
                    "Stokes U observation spectrum dimension mismatch")
            safe_obs = np.where(
                np.abs(obs_spectrum_other) > 1e-10, obs_spectrum_other, 1e-10)
            return (obs_spectrum_other - self.stokes_u) / safe_obs

        # This line should not be reached (pol_ch value already validated)
        raise ValueError(f"Invalid pol_channel value: {pol_ch}")

    def get_spectrum_stats(self) -> Dict[str, Any]:
        """Get spectrum statistics
        
        Returns
        -------
        dict
            Dictionary containing statistics for each component
        """
        self.validate()

        stats = {
            'wavelength_range':
            (float(self.wavelength.min()), float(self.wavelength.max()))
            if len(self.wavelength) > 0 else None,
            'stokes_i': {
                'min': float(self.stokes_i.min()),
                'max': float(self.stokes_i.max()),
                'mean': float(self.stokes_i.mean()),
                'std': float(self.stokes_i.std()),
            },
            'stokes_v': {
                'min': float(self.stokes_v.min()),
                'max': float(self.stokes_v.max()),
                'mean': float(self.stokes_v.mean()),
                'std': float(self.stokes_v.std()),
                'amplitude': float(self.stokes_v.max() - self.stokes_v.min()),
            },
        }

        if self.stokes_q is not None:
            stats['stokes_q'] = {
                'min': float(self.stokes_q.min()),
                'max': float(self.stokes_q.max()),
                'mean': float(self.stokes_q.mean()),
                'std': float(self.stokes_q.std()),
            }

        if self.stokes_u is not None:
            stats['stokes_u'] = {
                'min': float(self.stokes_u.min()),
                'max': float(self.stokes_u.max()),
                'mean': float(self.stokes_u.mean()),
                'std': float(self.stokes_u.std()),
            }

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary (for saving)
        
        Returns
        -------
        dict
            Result dictionary
        """
        self.validate()

        result = {
            'hjd': self.hjd,
            'phase_index': self.phase_index,
            'pol_channel': self.pol_channel,
            'model_name': self.model_name,
            'creation_time': self._creation_time,
        }

        return result

    def save_spectrum(self,
                      output_path: str,
                      par: Optional[Any] = None,
                      obsSet: Optional[List[Any]] = None,
                      verbose: int = 1) -> str:
        """Save synthetic spectrum to file
        
        Calls mainFuncs.save_model_spectra_to_outModelSpec to ensure correct output format
        and pol_channel parameter passing.
        
        Parameters
        ----------
        output_path : str
            Output file path or directory
        par : Optional[readParamsTomog]
            Parameter object, uses metadata in result if None
        obsSet : Optional[List[ObservationProfile]]
            Observation data object list, uses default format if None
        verbose : int
            Verbosity level (0=silent, 1=normal, 2=detailed)
        
        Returns
        -------
        str
            Generated output file path
        """
        from pathlib import Path
        import core.mainFuncs as mf

        self.validate()

        # Construct temporary par object (if not provided)
        if par is None:
            from types import SimpleNamespace
            par = SimpleNamespace()
            par.jDates = np.array(
                [self.hjd] if self.hjd is not None else [0.0])
            par.velRs = np.array([0.0])
            par.polChannels = np.array([self.pol_channel])
            par.phases = np.array([self.phase_index])

        # Construct temporary obsSet (if not provided)
        if obsSet is None:
            from core.SpecIO import ObservationProfile
            # Create temporary observation object to infer format
            obsSet = [
                ObservationProfile(wl=self.wavelength,
                                   specI=self.stokes_i,
                                   specV=self.stokes_v,
                                   specQ=self.stokes_q,
                                   specU=self.stokes_u,
                                   profile_type="spec",
                                   pol_channel=self.pol_channel)
            ]

        # Organize results as list (format: (v_grid, specI, specV, specQ, specU, pol_channel))
        # Note: wavelength as x-axis (can be wavelength or velocity)
        result_tuple = (self.wavelength, self.stokes_i, self.stokes_v,
                        self.stokes_q, self.stokes_u, self.pol_channel)
        results = [result_tuple]

        # Call mainFuncs.save_model_spectra_to_outModelSpec
        output_files = mf.save_model_spectra_to_outModelSpec(
            par,
            results,
            obsSet,
            output_dir=str(Path(output_path).parent),
            verbose=verbose)

        if output_files:
            return output_files[0]
        else:
            raise ValueError("Save failed: Failed to generate output file")

    def save_geomodel(self,
                      output_dir: str = './output',
                      integrator=None,
                      meta: Optional[Dict[str, Any]] = None,
                      verbose: int = 1) -> str:
        """Save geometric model to file (geomodel.tomog format)
        
        Calls VelspaceDiskIntegrator.write_geomodel() to save current geometry and physical model.
        
        Parameters
        ----------
        output_dir : str
            Output directory path
        integrator : VelspaceDiskIntegrator, optional
            Integrator object containing geometric model info. If None, tries to get from result
        meta : Optional[Dict[str, Any]]
            Additional metadata (e.g. observation info) to include in file header
        verbose : int
            Verbosity level (0=silent, 1=normal, 2=detailed)
            
        Returns
        -------
        str
            Generated output file path
            
        Raises
        ------
        ValueError
            If integrator is not provided and cannot be retrieved from result
            
        Notes
        -----
        geomodel.tomog file contains:
        - Velocity field parameters (disk_v0_kms, disk_power_index etc.)
        - Geometric parameters (inclination, differential rotation index etc.)
        - Grid definition (radial layers, grid boundaries etc.)
        - Physical quantities per pixel (magnetic field, area weight, amplitude etc.)
        """
        from pathlib import Path

        if integrator is None:
            integrator = getattr(self, 'integrator', None)

        if integrator is None:
            raise ValueError(
                "integrator parameter must be provided to save geometric model."
                "integrator should be the VelspaceDiskIntegrator instance used when generating ForwardModelResult."
            )

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Construct output filename
        filename = f"geomodel_phase_{self.phase_index:02d}.tomog"
        filepath = output_path / filename

        # Construct metadata
        if meta is None:
            meta = {}

        # Add info from ForwardModelResult
        meta.update({
            'phase_index': self.phase_index,
            'hjd': self.hjd,
            'pol_channel': self.pol_channel,
            'model_name': self.model_name,
            'creation_time': self._creation_time,
        })

        if verbose:
            print(f"[save_geomodel] Saving geometric model to: {filepath}")
            print(f"[save_geomodel] Phase index: {self.phase_index}")
            print(f"[save_geomodel] HJD: {self.hjd}")

        try:
            # Call integrator's write_geomodel method
            integrator.write_geomodel(str(filepath), meta=meta)

            if verbose:
                print(
                    f"[save_geomodel] ✓ Geometric model saved successfully: {filepath}"
                )

            return str(filepath)

        except Exception as e:
            raise ValueError(
                f"Failed to save geometric model: {type(e).__name__}: {e}")

    def save_model_data(self,
                        output_dir: str = './output',
                        integrator=None,
                        par=None,
                        obsSet=None,
                        verbose: int = 1) -> Dict[str, str]:
        """Save all model data at once (spectrum + geometric model)
        
        Convenience wrapper method to save both synthetic spectrum and geometric model file.
        
        Parameters
        ----------
        output_dir : str
            Output directory path
        integrator : VelspaceDiskIntegrator, optional
            Integrator object (for saving geometric model)
        par : Optional[readParamsTomog]
            Parameter object (for saving spectrum)
        obsSet : Optional[List[ObservationProfile]]
            Observation dataset (for saving spectrum)
        verbose : int
            Verbosity level
            
        Returns
        -------
        dict
            Dictionary containing generated file paths:
            {
                'spectrum': '...',  # Spectrum file
                'geomodel': '...',  # Geometric model file (if integrator provided)
            }
            
        Examples
        --------
        >>> result = forward_tomography(...)
        >>> files = result[0].save_model_data(
        ...     output_dir='./output',
        ...     integrator=phys_model.integrator,
        ...     verbose=1
        ... )
        >>> print(f"Spectrum saved to: {files['spectrum']}")
        >>> print(f"Geometric model saved to: {files['geomodel']}")
        """
        output_files = {}

        if verbose:
            print(f"[save_model_data] Saving model data to: {output_dir}")

        # Save spectrum
        try:
            spectrum_file = self.save_spectrum(output_dir,
                                               par=par,
                                               obsSet=obsSet,
                                               verbose=verbose)
            output_files['spectrum'] = spectrum_file
        except Exception as e:
            if verbose:
                print(f"[save_model_data] ⚠️  Spectrum save failed: {e}")
            output_files['spectrum'] = None

        # Save geometric model (if integrator provided)
        if integrator is not None:
            try:
                geomodel_file = self.save_geomodel(output_dir=output_dir,
                                                   integrator=integrator,
                                                   verbose=verbose)
                output_files['geomodel'] = geomodel_file
            except Exception as e:
                if verbose:
                    print(
                        f"[save_model_data] ⚠️  Geometric model save failed: {e}"
                    )
                output_files['geomodel'] = None

        if verbose:
            print("[save_model_data] ✓ Model data save completed")
            for key, path in output_files.items():
                if path:
                    print(f"  - {key}: {path}")

        return output_files

    def create_summary(self) -> str:
        """Generate result summary string
        
        Returns
        -------
        str
            Formatted result summary
        """
        self.validate()
        stats = self.get_spectrum_stats()

        lines = [
            "=" * 70,
            f"Forward Result Summary (Phase {self.phase_index}, HJD={self.hjd})",
            "=" * 70,
            f"Model: {self.model_name}",
            f"Pol Channel: {self.pol_channel}",
            f"Spectrum Points: {len(self.stokes_i)}",
        ]

        if stats['wavelength_range']:
            lines.append(
                f"Wavelength Range: {stats['wavelength_range'][0]:.4f} - {stats['wavelength_range'][1]:.4f} nm"
            )

        lines.extend([
            "",
            "Stokes I Stats:",
            f"  Range: [{stats['stokes_i']['min']:.6f}, {stats['stokes_i']['max']:.6f}]",
            f"  Mean: {stats['stokes_i']['mean']:.6f}",
            f"  Std: {stats['stokes_i']['std']:.6f}",
            "",
            "Stokes V Stats:",
            f"  Range: [{stats['stokes_v']['min']:.6f}, {stats['stokes_v']['max']:.6f}]",
            f"  Amplitude: {stats['stokes_v']['amplitude']:.6f}",
            f"  Mean: {stats['stokes_v']['mean']:.6f}",
        ])

        if 'stokes_q' in stats:
            lines.extend([
                "",
                "Stokes Q Stats:",
                f"  Range: [{stats['stokes_q']['min']:.6f}, {stats['stokes_q']['max']:.6f}]",
                f"  Mean: {stats['stokes_q']['mean']:.6f}",
            ])

        if 'stokes_u' in stats:
            lines.extend([
                "",
                "Stokes U Stats:",
                f"  Range: [{stats['stokes_u']['min']:.6f}, {stats['stokes_u']['max']:.6f}]",
                f"  Mean: {stats['stokes_u']['mean']:.6f}",
            ])

        lines.append("=" * 70)

        return "\n".join(lines)


@dataclass
class InversionResult:
    """MEM Inversion Result Container
    
    Stores the output of the MEM inversion workflow, including evolution information over iterations.
    """

    # ============ Final Magnetic Field Solution ============

    B_los_final: np.ndarray
    """Final line-of-sight magnetic field (Npix,)"""

    B_perp_final: np.ndarray
    """Final perpendicular magnetic field (Npix,)"""

    chi_final: np.ndarray
    """Final magnetic field azimuth angle (Npix,)"""

    brightness_final: Optional[np.ndarray] = None
    """Final brightness distribution (Npix,), optional"""

    # ============ Iteration Tracking ============

    iterations_completed: int = 0
    """Number of iterations completed"""

    chi2_history: List[float] = field(default_factory=list)
    """Chi-squared value history"""

    entropy_history: List[float] = field(default_factory=list)
    """Entropy value history"""

    regularization_history: List[float] = field(default_factory=list)
    """Regularization term history"""

    # ============ Convergence Info ============

    converged: bool = False
    """Whether converged"""

    convergence_reason: str = ""
    """Reason for convergence"""

    final_chi2: float = 0.0
    """Final chi-squared value"""

    final_entropy: float = 0.0
    """Final entropy value"""

    # ============ Statistics ============

    magnetic_field_stats: Optional[Dict[str, Any]] = None
    """Magnetic field statistics"""

    fit_quality: Dict[str, float] = field(default_factory=dict)
    """Fit quality metrics (e.g. 'rms_residual', 'max_residual' etc.)"""

    # ============ Metadata ============

    phase_index: int = 0
    """Phase index"""

    pol_channels: List[str] = field(default_factory=lambda: ["I+V"])
    """List of processed polarization channels"""

    # ============ Internal State ============

    _creation_time: str = field(
        default_factory=lambda: datetime.now().isoformat(),
        init=False,
        repr=False)
    """Creation time"""

    def validate(self) -> None:
        """Validate result data integrity
        
        Raises
        ------
        ValueError
            Data inconsistent or incomplete
        """
        if self.B_los_final is None or len(self.B_los_final) == 0:
            raise ValueError("B_los_final cannot be empty")

        npix = len(self.B_los_final)

        if len(self.B_perp_final) != npix:
            raise ValueError(
                f"B_perp_final length ({len(self.B_perp_final)}) mismatch with B_los ({npix})"
            )

        if len(self.chi_final) != npix:
            raise ValueError(
                f"chi_final length ({len(self.chi_final)}) mismatch with B_los ({npix})"
            )

        if self.iterations_completed < 0:
            raise ValueError(
                f"Iterations cannot be negative: {self.iterations_completed}")

        # Check consistency only if iterations > 0 and history exists
        if self.iterations_completed > 0:
            if len(self.chi2_history) > 0 and len(
                    self.chi2_history) != self.iterations_completed:
                raise ValueError(
                    f"Chi2 history length ({len(self.chi2_history)}) mismatch with iterations ({self.iterations_completed})"
                )

    def get_convergence_rate(self) -> Optional[float]:
        """Calculate convergence rate
        
        Returns
        -------
        Optional[float]
            Mean relative change of chi-squared between adjacent iterations, None if history insufficient
        """
        if len(self.chi2_history) < 2:
            return None

        chi2_array = np.array(self.chi2_history)
        # Avoid division by zero
        denom = np.where(
            np.abs(chi2_array[:-1]) > 1e-10, chi2_array[:-1], 1e-10)
        changes = np.abs(np.diff(chi2_array) / denom)

        return float(np.mean(changes))

    def get_magnetic_field_stats(self) -> Dict[str, Any]:
        """Calculate magnetic field statistics
        
        Returns
        -------
        dict
            Statistics for B_los, B_perp and chi
        """
        self.validate()

        # Calculate perpendicular magnetic field magnitude
        B_mag = np.sqrt(self.B_perp_final**2)

        stats = {
            'B_los': {
                'min': float(self.B_los_final.min()),
                'max': float(self.B_los_final.max()),
                'mean': float(self.B_los_final.mean()),
                'std': float(self.B_los_final.std()),
                'rms': float(np.sqrt(np.mean(self.B_los_final**2))),
            },
            'B_perp': {
                'min': float(self.B_perp_final.min()),
                'max': float(self.B_perp_final.max()),
                'mean': float(self.B_perp_final.mean()),
                'std': float(self.B_perp_final.std()),
                'rms': float(np.sqrt(np.mean(self.B_perp_final**2))),
            },
            'B_mag': {
                'min': float(B_mag.min()),
                'max': float(B_mag.max()),
                'mean': float(B_mag.mean()),
                'rms': float(np.sqrt(np.mean(B_mag**2))),
            },
            'chi': {
                'min': float(self.chi_final.min()),
                'max': float(self.chi_final.max()),
                'mean': float(self.chi_final.mean()),
                'std': float(self.chi_final.std()),
            },
            'npix': len(self.B_los_final),
        }

        return stats

    def get_optimization_metrics(self) -> Dict[str, float]:
        """Get key optimization metrics
        
        Returns
        -------
        dict
            Optimization metrics (iterations, final_chi2, convergence_rate etc.)
        """
        self.validate()

        metrics = {
            'iterations': self.iterations_completed,
            'final_chi2': self.final_chi2,
            'final_entropy': self.final_entropy,
            'converged': float(self.converged),
            'convergence_rate': self.get_convergence_rate() or 0.0,
        }

        # Add fit quality metrics
        metrics.update(self.fit_quality)

        return metrics

    def create_summary(self) -> str:
        """Generate inversion result summary
        
        Returns
        -------
        str
            Formatted summary string
        """
        self.validate()

        mag_stats = self.get_magnetic_field_stats()
        opt_metrics = self.get_optimization_metrics()

        lines = [
            "=" * 70,
            f"MEM Inversion Result Summary (Phase {self.phase_index})",
            "=" * 70,
            "",
            "Convergence Status:",
            f"  Iterations Completed: {self.iterations_completed}",
            f"  Converged: {self.converged}",
            f"  Reason: {self.convergence_reason}",
            f"  Convergence Rate: {opt_metrics['convergence_rate']:.3e}",
            "",
            "Optimization Metrics:",
            f"  Final Chi2: {self.final_chi2:.6e}",
            f"  Final Entropy: {self.final_entropy:.6e}",
            "",
            "Magnetic Field Stats:",
            f"  B_los: [{mag_stats['B_los']['min']:.1f}, {mag_stats['B_los']['max']:.1f}] G",
            f"         Mean {mag_stats['B_los']['mean']:.1f}, RMS {mag_stats['B_los']['rms']:.1f}",
            f"  B_perp: [{mag_stats['B_perp']['min']:.1f}, {mag_stats['B_perp']['max']:.1f}] G",
            f"          Mean {mag_stats['B_perp']['mean']:.1f}, RMS {mag_stats['B_perp']['rms']:.1f}",
            f"  B_mag: RMS {mag_stats['B_mag']['rms']:.1f} G",
            "",
            "Processed Pol Channels:",
            f"  {', '.join(self.pol_channels)}",
            "=" * 70,
        ]

        return "\n".join(lines)
