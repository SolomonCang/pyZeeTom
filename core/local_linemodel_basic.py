# linemodel.py
import numpy as np


class LineData:
    """
    Read single line parameter file:
    Common line format: wl0  sigWl  g
    Only uses wl0, sigWl, g.
    """

    def __init__(self, filename):
        self.wl0 = None
        self.sigWl = None
        self.g = None
        self.numLines = 0
        with open(filename, 'r') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                parts = s.split()
                # Extract parsable float columns
                vals = []
                for p in parts:
                    try:
                        vals.append(float(p))
                    except Exception:
                        pass
                if len(vals) < 3:
                    continue
                # Take first three columns: [wl0, sigWl, g]
                self.wl0 = vals[0]
                self.sigWl = vals[1]
                self.g = vals[2]
                self.numLines += 1
                break
        if self.numLines == 0:
            raise ValueError("No valid line parameters read from file")


class BaseLineModel:
    """
    Local line model interface:
      compute_local_profile(wl_grid, amp, Blos=None, **kwargs) -> dict with keys 'I','V','Q','U'
    Description:
      - wl_grid: (Nlambda,) or (Nlambda, Npix)
      - amp: Line amplitude (scalar or (Npix,) or (1,Npix)), amp<0 absorption, amp=0 no line, amp>0 emission
      - Blos: Line-of-sight magnetic field per pixel ((Npix,))
      - Optional Q/U input:
          Bperp: |B_⊥| ((Npix,))
          chi:   Transverse field azimuth angle χ (radians, (Npix,)), relative to Q reference axis
      - Calculation switches (kwargs):
          enable_V:  Default True
          enable_QU: Default True
      - Ic_weight (optional): If provided, multiplied as pixel weighting coefficient in result (e.g. for disk integration).
        If not provided, default all 1. Line continuum baseline is always 1.
    """

    def compute_local_profile(self, wl_grid, amp, Blos=None, **kwargs):
        raise NotImplementedError


class GaussianZeemanWeakLineModel(BaseLineModel):
    """
    Weak line approximation + Gaussian profile (continuum=1; profile controlled by amp alone)
    Notation:
      d = (λ - λ0)/σ,  G = exp(-d^2)

    Output:
      I = 1 + (amp - 1) * G
      (amp > 1 is emission line, amp < 1 is absorption line)

      V = C_V * Blos * (dI/dv)
      Where C_V = -4.6686e-12 * g * wl0 * c
      dI/dv is derivative of Stokes I with respect to velocity

      Q = -C2 * Bperp^2 * ((amp-1) * (G/σ^2) * (1 - 2 d^2)) * cos(2χ)
      U = -C2 * Bperp^2 * ((amp-1) * (G/σ^2) * (1 - 2 d^2)) * sin(2χ)

    Note:
      - amp can be scalar or per-pixel value; if (Npix,) passed, it will be broadcast to (1,Npix) and aligned with (Nλ,Npix) wl_grid.
      - If Ic_weight is provided, it will be multiplied at the end (weighting effect), without changing the definition of continuum baseline=1.
    """

    def __init__(self,
                 line_data: LineData,
                 k_QU: float = 1.0,
                 enable_V: bool = True,
                 enable_QU: bool = True):
        self.ld = line_data

        # Ensure line data is valid to satisfy type checkers
        if self.ld.wl0 is None or self.ld.g is None or self.ld.sigWl is None:
            raise ValueError(
                "LineData contains None values. Please check the input file.")

        self.c_kms = 2.99792458e5  # speed of light in km/s

        # Proportional constant for V
        # Coefficient for Stokes V: -4.6686e-12 * g * wl0 * c
        # This coefficient is used with dI/dv
        self.C_V_coeff = -4.6686e-12 * self.ld.g * self.ld.wl0 * self.c_kms

        # Proportional constant for Q/U (weak field second order)
        base = 4.6686e-12 * (self.ld.wl0**2) * self.ld.g
        self.C2 = (base**2) * float(k_QU)
        self.enable_V_default = bool(enable_V)
        self.enable_QU_default = bool(enable_QU)

    def compute_local_profile(self, wl_grid, amp, Blos=None, **kwargs):
        # Switches
        enable_V = bool(kwargs.get("enable_V", self.enable_V_default))
        enable_QU = bool(kwargs.get("enable_QU", self.enable_QU_default))

        Bperp = kwargs.get("Bperp", None)
        chi = kwargs.get("chi", None)

        # Pixel weight (optional): for final output weighting, does not change line baseline=1
        Ic_weight = kwargs.get("Ic_weight", None)

        # Shape handling
        wl_grid = np.asarray(wl_grid, dtype=float)
        if wl_grid.ndim == 1:
            wl_grid = wl_grid[:, None]  # (Nλ,1)
        Nlam, Npix = wl_grid.shape

        # amp handling and broadcast to (1,Npix)
        amp = np.asarray(amp, dtype=float)
        if amp.ndim == 0:
            amp = amp.reshape(1, 1)
        elif amp.ndim == 1:
            amp = amp.reshape(1, -1)
        # Broadcast check
        try:
            amp = np.broadcast_to(amp, (1, Npix))
        except ValueError:
            raise ValueError(
                "amp must be scalar or length Npix, broadcastable to (1,Npix)."
            )

        # Common kernel
        sig = float(self.ld.sigWl)
        d = (wl_grid - self.ld.wl0) / sig
        G = np.exp(-(d * d))

        # Calculate Gaussian amplitude A
        # amp is peak intensity: I_peak = amp
        # I = 1 + A * G => I_peak = 1 + A => A = amp - 1
        A = amp - 1.0

        # I (continuum=1)
        I = 1.0 + A * G

        # V
        # Stokes V = C_V * Blos * (dI/dv)
        if enable_V and (Blos is not None):
            Blos_arr = np.asarray(Blos, dtype=float).reshape(1, Npix)

            # dI/dlambda
            dI_dlam = A * G * (-2.0 * d / sig)
            # dI/dv = dI/dlambda * (lambda0 / c)
            dI_dv = dI_dlam * (self.ld.wl0 / self.c_kms)

            V = self.C_V_coeff * Blos_arr * dI_dv
        else:
            V = np.zeros((Nlam, Npix), dtype=float)

        # Q/U
        if enable_QU:
            Bperp = kwargs.get("Bperp", None)
            chi = kwargs.get("chi", None)
            if (Bperp is not None) and (chi is not None):
                Bperp = np.asarray(Bperp, dtype=float).reshape(1, Npix)
                chi = np.asarray(chi, dtype=float).reshape(1, Npix)
                d2_core = (G * (1.0 - 2.0 * d * d)) / (sig * sig)
                cos2c = np.cos(2.0 * chi)
                sin2c = np.sin(2.0 * chi)
                Bperp2 = Bperp * Bperp
                # Use A (amp-1) instead of amp
                Q = -self.C2 * Bperp2 * (A * d2_core) * cos2c
                U = -self.C2 * Bperp2 * (A * d2_core) * sin2c
            else:
                Q = np.zeros((Nlam, Npix), dtype=float)
                U = np.zeros((Nlam, Npix), dtype=float)
        else:
            Q = np.zeros((Nlam, Npix), dtype=float)
            U = np.zeros((Nlam, Npix), dtype=float)

        # Optional pixel weight: multiply at the end (does not change I baseline definition)
        if Ic_weight is not None:
            w = np.asarray(Ic_weight, dtype=float).reshape(1, Npix)
            I = I * w
            V = V * w
            Q = Q * w
            U = U * w

        return {"I": I, "V": V, "Q": Q, "U": U}

    def compute_local_derivatives(self, wl_grid, amp, Blos=None, **kwargs):
        """
        Compute partial derivatives of Stokes parameters with respect to model parameters (analytical derivatives).
        
        Returns
        -------
        dict
            Dictionary containing derivative matrices. Key format 'd{Stokes}_d{Param}'.
            e.g.: 'dI_damp', 'dV_dBlos', 'dQ_dchi' etc.
            Each value is (Nlam, Npix) array.
        """
        # Switches
        enable_V = bool(kwargs.get("enable_V", self.enable_V_default))
        enable_QU = bool(kwargs.get("enable_QU", self.enable_QU_default))

        Bperp = kwargs.get("Bperp", None)
        chi = kwargs.get("chi", None)
        Ic_weight = kwargs.get("Ic_weight", None)

        # Shape handling
        wl_grid = np.asarray(wl_grid, dtype=float)
        if wl_grid.ndim == 1:
            wl_grid = wl_grid[:, None]
        Nlam, Npix = wl_grid.shape

        # amp handling
        amp = np.asarray(amp, dtype=float)
        if amp.ndim == 0:
            amp = amp.reshape(1, 1)
        elif amp.ndim == 1:
            amp = amp.reshape(1, -1)
        try:
            amp = np.broadcast_to(amp, (1, Npix))
        except ValueError:
            raise ValueError("amp must be scalar or length Npix")

        # Basic variables
        sig = float(self.ld.sigWl)
        d = (wl_grid - self.ld.wl0) / sig
        G = np.exp(-(d * d))

        # A = amp - 1
        A = amp - 1.0

        # Precompute dG/dv (for V derivatives)
        # dI/dlam = A * dG/dlam = A * G * (-2d/sig)
        # dG/dlam = G * (-2d/sig)
        # dG/dv = dG/dlam * (wl0/c)
        dG_dlam = G * (-2.0 * d / sig)
        dG_dv = dG_dlam * (self.ld.wl0 / self.c_kms)

        # Precompute Q/U core terms (part divided by A and Bperp^2)
        # Q_core = -C2 * (G/sig^2) * (1-2d^2) * cos(2chi)
        # Q = Bperp^2 * A * Q_core_no_B_A
        d2_core = (G * (1.0 - 2.0 * d * d)) / (sig * sig)

        # Initialize result dictionary
        derivs = {}

        # 1. dI/damp = G
        derivs['dI_damp'] = G.copy()

        # 2. V related derivatives
        if enable_V and (Blos is not None):
            Blos_arr = np.asarray(Blos, dtype=float).reshape(1, Npix)

            # V = C_V * Blos * A * dG/dv

            # dV/damp = C_V * Blos * dG/dv
            derivs['dV_damp'] = self.C_V_coeff * Blos_arr * dG_dv

            # dV/dBlos = C_V * A * dG/dv = V / Blos
            # Note: if Blos=0, V=0, but derivative is not 0
            derivs['dV_dBlos'] = self.C_V_coeff * A * dG_dv
        else:
            derivs['dV_damp'] = np.zeros((Nlam, Npix))
            derivs['dV_dBlos'] = np.zeros((Nlam, Npix))

        # 3. Q/U related derivatives
        if enable_QU and (Bperp is not None) and (chi is not None):
            Bperp = np.asarray(Bperp, dtype=float).reshape(1, Npix)
            chi = np.asarray(chi, dtype=float).reshape(1, Npix)

            cos2c = np.cos(2.0 * chi)
            sin2c = np.sin(2.0 * chi)
            Bperp2 = Bperp * Bperp

            # Q = -C2 * Bperp^2 * A * d2_core * cos2c
            # U = -C2 * Bperp^2 * A * d2_core * sin2c

            # dQ/damp = Q / A
            derivs['dQ_damp'] = -self.C2 * Bperp2 * d2_core * cos2c
            derivs['dU_damp'] = -self.C2 * Bperp2 * d2_core * sin2c

            # dQ/dBperp = 2 * Q / Bperp
            derivs['dQ_dBperp'] = -self.C2 * (2.0 *
                                              Bperp) * A * d2_core * cos2c
            derivs['dU_dBperp'] = -self.C2 * (2.0 *
                                              Bperp) * A * d2_core * sin2c

            # dQ/dchi = -2 * U
            # dU/dchi = 2 * Q
            # Q ~ cos(2chi) -> d/dchi ~ -2sin(2chi) ~ U term
            # U ~ sin(2chi) -> d/dchi ~ 2cos(2chi) ~ Q term
            # Check signs:
            # Q = K * cos(2chi) -> dQ = -2 K sin(2chi) = -2 * (K sin(2chi)) = -2 * (-U) = 2U ?
            # Wait. U = K * sin(2chi). So K = U / sin(2chi).
            # dQ/dchi = -2 * K * sin(2chi) = -2 * U. Correct.

            # U = K * sin(2chi) -> dU = 2 K cos(2chi) = 2 * Q. Correct.

            # Re-calculate explicitly to avoid division by zero or confusion
            common_factor = -self.C2 * Bperp2 * A * d2_core
            derivs['dQ_dchi'] = common_factor * (-2.0 * sin2c)
            derivs['dU_dchi'] = common_factor * (2.0 * cos2c)

        else:
            zeros = np.zeros((Nlam, Npix))
            derivs['dQ_damp'] = zeros
            derivs['dU_damp'] = zeros
            derivs['dQ_dBperp'] = zeros
            derivs['dU_dBperp'] = zeros
            derivs['dQ_dchi'] = zeros
            derivs['dU_dchi'] = zeros

        # Apply weight Ic_weight
        if Ic_weight is not None:
            w = np.asarray(Ic_weight, dtype=float).reshape(1, Npix)
            for key in derivs:
                derivs[key] *= w

        return derivs


class ConstantAmpLineModel(BaseLineModel):
    """
    Adapter wrapping base line model with constant intensity.

    Provides a convenient interface to run any base line model with constant amplitude
    without explicitly passing amp parameter. Useful in forward modeling
    where brightness distribution is assumed known (fixed).

    Parameters
    ----------
    base_model : BaseLineModel
        Underlying line model object, should have
        compute_local_profile(wl_grid, amp, **kwargs) method
    amp : float, default=1.0
        Constant amplitude value (applied to all pixels)

    Attributes
    ----------
    base_model : BaseLineModel
        Underlying model
    amp : float
        Constant amplitude

    Examples
    --------
    >>> from core.local_linemodel_basic import LineData, GaussianZeemanWeakLineModel, ConstantAmpLineModel
    >>> ld = LineData('input/lines.txt')
    >>> base = GaussianZeemanWeakLineModel(ld)
    >>> adapter = ConstantAmpLineModel(base, amp=0.5)
    >>>
    >>> # No need to pass amp, use constant value
    >>> wl_grid = np.linspace(6562.5, 6563.5, 200)  # Wavelength grid only
    >>> result = adapter.compute_local_profile(wl_grid, Blos=np.zeros(100))
    >>> print(result['I'].shape)  # (200, 100)
    """

    def __init__(self, base_model: BaseLineModel, amp: float = 1.0):
        """Initialize ConstantAmpLineModel."""
        self.base_model = base_model
        self.amp = float(amp)

    def compute_local_profile(self, wl_grid, amp_unused=None, **kwargs):
        """Compute local line profile using stored constant amplitude.

        Parameters
        ----------
        wl_grid : np.ndarray
            Wavelength grid
        amp_unused : ignored
            This parameter is ignored, self.amp is used instead
        **kwargs
            Other keyword arguments passed to base_model.compute_local_profile

        Returns
        -------
        dict
            Dictionary containing 'I', 'V', 'Q', 'U' keys, corresponding to computed Stokes parameters
        """
        # Remove amp from kwargs (if present) as we want to use self.amp
        kwargs.pop('amp', None)

        return self.base_model.compute_local_profile(wl_grid, self.amp,
                                                     **kwargs)

    def compute_local_derivatives(self, wl_grid, amp_unused=None, **kwargs):
        """Delegate derivative computation to base model using constant amp.
        
        Parameters
        ----------
        wl_grid : np.ndarray
            Wavelength grid
        amp_unused : ignored
            Ignored, uses self.amp
        **kwargs
            Passed to base_model.compute_local_derivatives
            
        Returns
        -------
        dict
            Derivatives dictionary
        """
        # Remove amp from kwargs if present
        kwargs.pop('amp', None)

        return self.base_model.compute_local_derivatives(
            wl_grid, self.amp, **kwargs)
