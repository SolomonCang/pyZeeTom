# linemodel.py  (ME + Unno–Rachkovsky; Input is suggested format, supports additional Voigt peak for emission component)
import numpy as np

_C_KMS = 2.99792458e5  # km/s
_ZEEMAN_CONST = 4.66864e-12  # ΔλB(nm) = const * g * λ0^2 * B(G)


class LineData:
    """
    Single line parameters (only uses the first non-comment line):
      wl0  g  eta0  beta  widthGauss(km/s)  widthLorentz_ratio  [fI]  [fV]  [emitAmp]  [emitWidthScale]
    Description:
      - eta0: Line center absorption/continuum ratio (>0)
      - beta: Source function slope, S(τ)=S0(1+βτ)
      - widthGauss: sqrt(2)*σ_v, unit km/s
      - widthLorentz_ratio: Γ/(sqrt(2)*σ_v) = gamma/widthGauss
      - fI, fV: Filling factors, default 1, and fI ∈ [0,1] (preserve physical meaning)
      - emitAmp: Additional emission peak amplitude A_e (≥0, default 0, suggested only for "emission component")
      - emitWidthScale: Emission peak width scale s_e (>0, default 1.0)
    """

    def __init__(self, filename):
        self.wl0 = None
        self.g = None
        self.eta0 = None
        self.beta = None
        self.widthGauss = None
        self.widthLorentz = None
        self.fI = 1.0
        self.fV = 1.0
        self.emitAmp = 0.0
        self.emitWidthScale = 1.0
        self.numLines = 0

        with open(filename, 'r') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                parts = s.split()
                vals = []
                for p in parts:
                    vals.append(float(p))

                if len(vals) < 6:
                    raise ValueError(
                        "ME suggested format requires at least 6 columns: wl0 g eta0 beta widthGauss widthLorentz_ratio [fI] [fV] [emitAmp] [emitWidthScale]"
                    )

                self.wl0 = float(vals[0])
                self.g = float(vals[1])
                self.eta0 = float(vals[2])
                self.beta = float(vals[3])
                self.widthGauss = float(vals[4])
                self.widthLorentz = float(vals[5])
                if len(vals) >= 7:
                    self.fI = float(vals[6])
                if len(vals) >= 8:
                    self.fV = float(vals[7])
                if len(vals) >= 9:
                    self.emitAmp = float(vals[8])
                if len(vals) >= 10:
                    self.emitWidthScale = float(vals[9])

                # Simple boundary checks
                if self.widthGauss <= 0:
                    raise ValueError("widthGauss must be positive (km/s).")
                if self.widthLorentz < 0:
                    raise ValueError("widthLorentz_ratio cannot be negative.")
                if not (0.0 <= self.fI <= 1.0):
                    raise ValueError(
                        "fI should be in [0,1] (preserve filling factor physical meaning)."
                    )
                if not (0.0 <= self.fV <= 1.0):
                    raise ValueError("fV should be in [0,1].")
                if self.emitAmp < 0.0:
                    raise ValueError(
                        "emitAmp should be ≥ 0 (use positive value only in emission component)."
                    )
                if self.emitWidthScale <= 0.0:
                    raise ValueError("emitWidthScale should be > 0.")

                self.numLines = 1
                break

        if self.numLines == 0:
            raise ValueError("No valid line parameters read.")


def _as_col(arr, Npix, dtype=float):
    arr = np.asarray(arr, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.broadcast_to(arr, (1, Npix))


def _voigt_faraday_humlicek(u, a):
    a = np.asarray(a, dtype=float)
    z = a - 1j * u
    zz = z * z
    s = np.abs(u) + a
    w4 = np.zeros_like(z, dtype=complex)

    con1 = (s >= 15.0)
    if np.any(con1):
        zt = z[con1]
        w4[con1] = 0.56418958355 * zt / (0.5 + zt * zt)

    con2 = (s >= 5.5) & (s < 15.0)
    if np.any(con2):
        zt = z[con2]
        zzt = zz[con2]
        w4[con2] = zt * (1.4104739589 + 0.56418958355 * zzt) / ((
            (3.0 + zzt) * zzt) + 0.7499999999)

    con3 = (a >= 0.195 * np.abs(u) - 0.176) & (s < 5.5)
    if np.any(con3):
        zt = z[con3]
        w4[con3] = (16.4954955 + zt *
                    (20.2093334 + zt *
                     (11.9648172 + zt * (3.77898687 + zt * 0.564223565)))) / (
                         16.4954955 + zt * (38.8236274 + zt *
                                            (39.2712051 + zt *
                                             (21.6927370 + zt *
                                              (6.69939801 + zt)))))

    con4 = (w4 == 0.0 + 0.0j)
    if np.any(con4):
        zt = z[con4]
        zzt = zz[con4]
        w4[con4] = np.exp(zzt) - zt * (
            36183.30536 - zzt *
            (3321.990492 - zzt *
             (1540.786893 - zzt *
              (219.0312964 - zzt *
               (35.76682780 - zzt * (1.320521697 - zzt * 0.5641900381)))))) / (
                   32066.59372 - zzt * (24322.84021 - zzt *
                                        (9022.227659 - zzt *
                                         (2186.181081 - zzt *
                                          (364.2190727 - zzt *
                                           (61.57036588 - zzt *
                                            (1.841438936 - zzt)))))))
    return w4


class MilneEddingtonURDual:
    """
    ME + UR Full Stokes solution, dual parameter set (absorption/emission) linear weighting:
      S_total = w_abs * S_abs + w_emit * S_emit

    New feature for emission component:
      Superimpose an additional source E(λ) = emitAmp * H_e(λ) on the I of this component,
      where H_e is the Voigt profile calculated with emitWidthScale times width (only real part H).

    Usage:
      - Absorption component file does not need to provide emitAmp/emitWidthScale (default 0, 1).
      - Emission component file can provide emitAmp > 0 (suggested 0.1~1.5), emitWidthScale adjustable width.

    """

    def __init__(self, line_absorb: LineData, line_emit: LineData):
        # Both lines must be the same line (wl0, g consistent)
        if (abs(line_absorb.wl0 - line_emit.wl0)
                > 1e-9) or (abs(line_absorb.g - line_emit.g) > 1e-9):
            raise ValueError(
                "wl0 and g of absorption and emission parameters must be consistent"
            )
        self.ld_abs = line_absorb
        self.ld_emit = line_emit

    def compute(self,
                wl_grid,
                Ic_weight,
                weight_absorb=1.0,
                weight_emit=0.0,
                Blos=None,
                Bmod=None,
                Btheta=None,
                Bchi=None,
                mu=None,
                enable_IQVU=(True, True, True, True)):
        """
        - wl_grid: (Nλ,) or (Nλ,Npix)
        - Ic_weight: (Npix,) Surface element weight (limb darkening, area, etc.)
        - weight_absorb, weight_emit: Can be scalar or (Npix,); default 1/0
        - Magnetic field: Prefer Bmod, Btheta, Bchi; otherwise try Blos only for longitudinal
        - mu: Line-of-sight cosine (Npix,); default 1 if not given
        - enable_IQVU: Tuple switch
        Returns dict: 'I','Q','U','V' -> (Nλ,Npix)
        """
        enable_I, enable_Q, enable_U, enable_V = enable_IQVU

        wl_grid = np.asarray(wl_grid, dtype=float)
        if wl_grid.ndim == 1:
            wl_grid = wl_grid[:, None]
        Nlam, Npix = wl_grid.shape

        w_area = np.asarray(Ic_weight, dtype=float).reshape(1, Npix)

        # Geometry and Magnetic Field
        if (Bmod is not None) and (Btheta is not None):
            Bmod = _as_col(Bmod, Npix)
            Btheta = _as_col(Btheta, Npix)
            if Bchi is None:
                Bchi = np.zeros((1, Npix), dtype=float)
            else:
                Bchi = _as_col(Bchi, Npix)
        else:
            # Fallback to Blos only case: assume purely longitudinal field (θ=0 or π), Q/U≈0
            if Blos is None:
                Bmod = np.zeros((1, Npix), dtype=float)
                Btheta = np.zeros((1, Npix), dtype=float)
                Bchi = np.zeros((1, Npix), dtype=float)
            else:
                Blos_arr = _as_col(Blos, Npix)
                Bmod = np.abs(Blos_arr)
                Btheta = np.where(Blos_arr >= 0, 0.0,
                                  np.pi) * np.ones_like(Blos_arr)
                Bchi = np.zeros((1, Npix), dtype=float)

        if mu is None:
            mu = np.ones((1, Npix), dtype=float)
        else:
            mu = _as_col(mu, Npix)

        # Component weights
        w_abs = _as_col(weight_absorb, Npix)
        w_em = _as_col(weight_emit, Npix)

        # Component solution
        I_abs, Q_abs, U_abs, V_abs = self._solve_me_ur_component(
            wl_grid, self.ld_abs, Bmod, Btheta, Bchi, mu, enable_IQVU)

        I_em, Q_em, U_em, V_em = self._solve_me_ur_component(
            wl_grid, self.ld_emit, Bmod, Btheta, Bchi, mu, enable_IQVU)

        # Superimpose additional source E(λ) = emitAmp * H_e(λ) on I of emission component
        if enable_I and (self.ld_emit.emitAmp > 0.0):
            He = self._voigt_profile_for_emission(wl_grid, self.ld_emit)
            I_em = I_em + self.ld_emit.emitAmp * He

        # Linear superposition of two components
        I = (w_abs * I_abs + w_em * I_em) * w_area if enable_I else np.zeros(
            (Nlam, Npix))
        Q = (w_abs * Q_abs + w_em * Q_em) * w_area if enable_Q else np.zeros(
            (Nlam, Npix))
        U = (w_abs * U_abs + w_em * U_em) * w_area if enable_U else np.zeros(
            (Nlam, Npix))
        V = (w_abs * V_abs + w_em * V_em) * w_area if enable_V else np.zeros(
            (Nlam, Npix))

        return {"I": I, "Q": Q, "U": U, "V": V}

    def _voigt_profile_for_emission(self, wl_grid, ld: LineData):
        """
        Construct Voigt real part H_e(λ) for additional source based on emission component parameters:
          - Center wl0 consistent with line
          - Width: Δλ_D_e = emitWidthScale * Δλ_D(line)
          - Damping: a_e = emitWidthScale * a(line)    (Optional proportional scaling, simple and stable)
          - Only take real part H_e of Voigt (unpolarized additional source, does not change Q, U, V)
          - Normalization: Consistent with main solution: H = Re(W)/sqrt(pi)
        """
        wl0 = float(ld.wl0)
        a = ld.widthLorentz
        vG = ld.widthGauss
        s = ld.emitWidthScale

        # Δλ_D(line) = (vG/c)*λ0
        dlamD_line = (vG / _C_KMS) * wl0
        dlamD_e = s * dlamD_line
        a_e = s * a

        u = (wl_grid - wl0) / dlamD_e
        W = _voigt_faraday_humlicek(u, a_e)
        He = W.real / np.sqrt(np.pi)
        return He

    def _solve_me_ur_component(self, wl_grid, ld: LineData, Bmod, Btheta, Bchi,
                               mu, enable_IQVU):
        """
        Single component ME+UR full analytical solution. Returns I, Q, U, V (all (Nλ, Npix)).
        Ic normalized to 1; externally weighted by w_area.
        """
        enable_I, enable_Q, enable_U, enable_V = enable_IQVU
        Nlam, Npix = wl_grid.shape

        wl0 = float(ld.wl0)
        g_eff = float(ld.g)
        eta0 = _as_col(ld.eta0, Npix)
        beta = _as_col(ld.beta, Npix)
        a = _as_col(ld.widthLorentz, Npix)
        vG = _as_col(ld.widthGauss, Npix)  # km/s
        fI = _as_col(ld.fI, Npix)
        fV = _as_col(ld.fV, Npix)

        # Magnetic splitting displacement Δλ_B (nm)
        dlamB = _ZEEMAN_CONST * g_eff * (wl0**2) * Bmod / np.maximum(fV, 1e-12)

        # Gaussian width Δλ_D (nm) = (vG/c)*λ0
        dlamD = (vG / _C_KMS) * wl0

        # Angle factors
        theta = Btheta
        chi = Bchi
        cos1 = np.cos(theta)
        sin1 = np.sin(theta)
        sin2 = sin1**2
        cos2chi = np.cos(2.0 * chi)
        sin2chi = np.sin(2.0 * chi)

        # Normalized frequency shift for Π, σ±
        u_pi = (wl_grid - wl0) / dlamD
        u_r = (wl_grid - wl0 - dlamB) / dlamD
        u_l = (wl_grid - wl0 + dlamB) / dlamD

        # Compute Faddeeva
        Wpi = _voigt_faraday_humlicek(u_pi, a)
        Wr = _voigt_faraday_humlicek(u_r, a)
        Wl = _voigt_faraday_humlicek(u_l, a)

        # Normalization: Voigt H = Re(W)/sqrt(pi), Faraday F = Im(W)/sqrt(pi)
        sqpi = np.sqrt(np.pi)
        H_pi = Wpi.real / sqpi
        F_pi = Wpi.imag / sqpi
        H_r = Wr.real / sqpi
        F_r = Wr.imag / sqpi
        H_l = Wl.real / sqpi
        F_l = Wl.imag / sqpi

        # Propagation and dispersion coefficients (according to Unno-Rachkovsky)
        kL = eta0
        eta_I = 1.0 + 0.5 * kL * (H_r + H_l) * (
            1.0 + cos1**2) * 0.5 + kL * H_pi * 0.5 * sin2
        # Rewritten for clarity: traditional notation
        eta_I = 1.0 + kL * (0.5 * (H_r + H_l) * (1.0 + cos1**2) / 2.0 + H_pi *
                            (sin2 / 2.0))
        eta_Q = kL * (H_pi - 0.5 * (H_r + H_l)) * (sin2 * cos2chi) / 2.0
        eta_U = kL * (H_pi - 0.5 * (H_r + H_l)) * (sin2 * sin2chi) / 2.0
        eta_V = kL * (H_r - H_l) * cos1 / 2.0

        rho_Q = kL * (F_pi - 0.5 * (F_r + F_l)) * (sin2 * cos2chi) / 2.0
        rho_U = kL * (F_pi - 0.5 * (F_r + F_l)) * (sin2 * sin2chi) / 2.0
        rho_V = kL * (F_r - F_l) * cos1 / 2.0

        # Compact notation
        etaI = eta_I
        etaQ = eta_Q
        etaU = eta_U
        etaV = eta_V
        rhoQ = rho_Q
        rhoU = rho_U
        rhoV = rho_V

        # Robust Δ expression
        Delta = ((etaI**2 - etaQ**2 - etaU**2 - etaV**2 + rhoQ**2 + rhoU**2 +
                  rhoV**2)**2 + 4.0 *
                 ((etaQ * rhoQ + etaU * rhoU + etaV * rhoV)**2 +
                  (etaU * rhoV - etaV * rhoU)**2 +
                  (etaV * rhoQ - etaQ * rhoV)**2 +
                  (etaQ * rhoU - etaU * rhoQ)**2))
        Delta = np.maximum(Delta, 1e-300)

        # Source function term
        beta_mu = beta * mu
        denom = 1.0 + beta_mu

        # A_* terms
        etaI2 = etaI**2
        K = etaI2 + rhoQ**2 + rhoU**2 + rhoV**2
        A_I = etaI * K
        A_Q = etaI2 * etaQ + etaI * (rhoV * etaU - rhoU * etaV) + etaQ * (
            rhoQ**2 + rhoU**2 + rhoV**2 - (etaQ**2 + etaU**2 + etaV**2))
        A_U = etaI2 * etaU + etaI * (rhoQ * etaV - rhoV * etaQ) + etaU * (
            rhoQ**2 + rhoU**2 + rhoV**2 - (etaQ**2 + etaU**2 + etaV**2))
        A_V = etaI2 * etaV + etaI * (rhoU * etaQ - rhoQ * etaU) + etaV * (
            rhoQ**2 + rhoU**2 + rhoV**2 - (etaQ**2 + etaU**2 + etaV**2))

        # Stokes (emergent intensity normalized to Ic=1)
        I = (1.0 + (beta_mu * A_I) / Delta) / denom if enable_I else np.zeros(
            (Nlam, Npix))
        Q = ((-beta_mu) * A_Q / (Delta * denom)) if enable_Q else np.zeros(
            (Nlam, Npix))
        U = ((-beta_mu) * A_U / (Delta * denom)) if enable_U else np.zeros(
            (Nlam, Npix))
        V = ((-beta_mu) * A_V / (Delta * denom)) if enable_V else np.zeros(
            (Nlam, Npix))

        # Non-magnetic/unresolved mixture: fI, fV (keep fI ∈ [0,1])
        I0 = np.ones_like(I)
        I = ld.fI * I + (1.0 - ld.fI) * I0
        Q *= ld.fV
        U *= ld.fV
        V *= ld.fV

        return I, Q, U, V
