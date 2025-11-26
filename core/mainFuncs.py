import numpy as np
from pathlib import Path
import core.SpecIO as SpecIO
import datetime as dt

c = 2.99792458e5


def compute_phase_from_jd(jd, jd_ref, period):
    """Compute observation phase (rotation phase)
    
    Parameters
    ----------
    jd : float or array-like
        Observation Julian Date (Heliocentric Julian Date)
    jd_ref : float
        Reference Julian Date (HJD0)
    period : float
        Rotation period (days)
        
    Returns
    -------
    phase : float or ndarray
        Phase, phase = (jd - jd_ref) / period
        For rigid body rotation, phase directly corresponds to observation angle offset: Δφ = 2π × phase
        For differential rotation, phase evolution of each ring is determined by its own angular velocity
    """
    return (np.asarray(jd) - float(jd_ref)) / float(period)


class readParamsTomog:
    """Read tomography parameter file
    
    Supports two grid definition methods:
    1. Directly specify Vmax (row 1 col 3 non-zero): independent of radius/vsini/inclination
    2. Use radius + r_out + vsini + inclination to calculate Vmax
    """

    def __init__(self, inParamsName, verbose=1):
        # Read in the model and control parameters
        fInTomog = open(inParamsName, 'r')
        self.fnames = np.array([])
        self.jDates = np.array([])
        self.velRs = np.array([])
        self.polChannels = np.array(
            [])  # New: polarization channel for each observation (I/V/Q/U)
        self.numObs = 0
        # New defaults for tomography workflow
        self.lineParamFile = 'lines.txt'  # path to line model parameters
        self.obsFileType = 'auto'  # observation format hint for readObs
        self.enable_stellar_occultation = 0  # Default disable stellar occultation
        i = 0
        for inLine in fInTomog:
            if (inLine.strip() == ''):  # skip blank lines
                continue
            # check for comments (ignoring white-space)
            if (inLine.strip()[0] != '#'):
                if (i == 0):
                    # Line 0: inclination vsini period pOmega
                    parts = inLine.split()
                    self.inclination = float(parts[0])
                    self.vsini = float(parts[1])
                    self.period = float(parts[2])
                    self.pOmega = float(parts[3])
                    # Legacy alias for backward compatibility
                    self.dOmega = self.pOmega
                elif (i == 1):
                    # Line 1: mass radius [Vmax] [r_out] [enable_occultation]
                    # If Vmax is non-zero, use Vmax to define grid, otherwise calculate from radius+r_out+vsini+inclination
                    parts = inLine.split()
                    self.mass = float(parts[0])
                    self.radius = float(parts[1])
                    self.Vmax = float(parts[2]) if len(parts) > 2 else 0.0
                    self.r_out = float(parts[3]) if len(parts) > 3 else 0.0
                    self.enable_stellar_occultation = int(
                        parts[4]) if len(parts) > 4 else 0
                elif (i == 2):
                    self.nRingsStellarGrid = int(inLine.split()[0])
                elif (i == 3):
                    self.targetForm = inLine.split()[0]
                    self.targetValue = float(inLine.split()[1])
                    self.numIterations = int(inLine.split()[2])
                elif (i == 4):
                    self.test_aim = float(inLine.split()[0])
                elif (i == 5):
                    # Line model configuration
                    # Format: ampConst  k_QU  enableV  enableQU
                    parts = inLine.split()
                    self.lineAmpConst = float(parts[0])
                    self.lineKQU = float(parts[1]) if len(parts) > 1 else 1.0
                    self.lineEnableV = int(parts[2]) if len(parts) > 2 else 1
                    self.lineEnableQU = int(parts[3]) if len(parts) > 3 else 1
                elif (i == 6):
                    # New unified model initialization: initTomogFile modelPath
                    # initTomogFile: 0=disabled, 1=enabled (read model from .tomog file)
                    # modelPath: path to .tomog file (e.g., output/geomodel_phase0.tomog)
                    # When enabled, model parameters override input params and generate params_tomog_int.txt
                    parts = inLine.split()
                    self.initTomogFile = int(parts[0])
                    self.initModelPath = parts[1] if len(parts) > 1 else None
                elif (i == 7):
                    # Line 7: fitBri fitMag fitBlos fitBperp fitChi
                    # Control fitting switches (0=off, 1=on)
                    parts = inLine.split()
                    self.fitBri = int(parts[0])
                    self.fitMag = int(parts[1])
                    self.fitBlos = int(parts[2])
                    self.fitBperp = int(parts[3])
                    self.fitChi = int(parts[4])
                elif (i == 8):
                    # Line 8 (was 11): spectralResolution lineParamFile
                    # spectralResolution is now resolution (e.g. 65000), will be converted to FWHM (km/s)
                    parts = inLine.split()
                    self.spectralResolution = float(parts[0])
                    # optional: path to line parameter file (e.g., lines.txt)
                    if len(parts) >= 2:
                        self.lineParamFile = parts[1]
                elif (i == 9):
                    # Line 9 (was 12): velStart velEnd [obsFileType] [key=val...]
                    parts = inLine.split()
                    self.velStart = float(parts[0])
                    self.velEnd = float(parts[1])
                    # optional: global observation file type hint (auto|lsd_i|lsd_pol|spec_i|...)
                    if len(parts) >= 3 and '=' not in parts[2]:
                        self.obsFileType = parts[2]
                        extra = parts[3:]
                    else:
                        extra = parts[2:]
                    # optional key=val tokens e.g., polOut=V|Q|U, specType=auto|spec|lsd
                    self.polOut = 'V'  # default output polarization
                    self.specType = 'auto'  # default data domain type (auto|spec|lsd)
                    for tok in extra:
                        if '=' in tok:
                            k, v = tok.split('=', 1)
                            k = k.strip()
                            v = v.strip()
                            if k.lower() in ('polout', 'stokesout'):
                                vv = v.upper()
                                if vv in ('V', 'Q', 'U'):
                                    self.polOut = vv
                            elif k.lower() == 'spectype':
                                vv = v.lower()
                                if vv in ('auto', 'spec', 'lsd'):
                                    self.specType = vv
                    if verbose:
                        print(
                            f"[Params] polOut = {self.polOut}, specType = {self.specType}"
                        )
                elif (i == 10):
                    # Line 10 (was 13): jDateRef
                    self.jDateRef = float(inLine.split()[0])
                elif (i >= 11):
                    # Line 11+ (was 14+): Observations
                    parts = inLine.split()
                    self.fnames = np.append(self.fnames, [parts[0]])
                    self.jDates = np.append(self.jDates, [float(parts[1])])
                    self.velRs = np.append(self.velRs, [float(parts[2])])
                    # New: read polchannel (column 4), default to 'V'
                    polchan = parts[3].upper() if len(parts) > 3 else 'V'
                    if polchan not in ('I', 'V', 'Q', 'U'):
                        if verbose:
                            print(
                                f'Warning: invalid polchannel "{parts[3]}" in line {i+1}, using default "V"'
                            )
                        polchan = 'V'
                    self.polChannels = np.append(self.polChannels, [polchan])
                    self.numObs += 1
                    if (np.abs(self.jDateRef - self.jDates[self.numObs - 1])
                            > 500.):
                        print(
                            'Warning: possible miss-match between date and reference date {:} {:}'
                            .format(self.jDateRef,
                                    self.jDates[self.numObs - 1]))
                    if (np.abs(self.velRs[self.numObs - 1]) > 500.):
                        print('Warning: extreem Vr read:{:}'.format(
                            self.velRs[self.numObs - 1]))

                i += 1

        # Calculate derived parameters
        self.incRad = self.inclination / 180. * np.pi
        self.velEq = self.vsini / np.sin(self.incRad) if np.sin(
            self.incRad) > 1e-6 else 0.0

        # Determine grid velocity range (one of two methods)
        if abs(self.Vmax) > 1e-6:
            # Method 1: Use Vmax directly
            if verbose:
                print(f"[Grid] Using direct Vmax = {self.Vmax:.2f} km/s")
        else:
            # Method 2: Calculate from radius + r_out + vsini + inclination
            if self.r_out > 0 and self.radius > 0:
                # Differential rotation velocity field: v(r) = veq * (r/R*)^(pOmega+1)
                # Vmax (projected) = vsini * (r_out/radius)^(pOmega+1)
                # Note: Use vsini instead of velEq here to match logic in examples/three_spot_simulation.py,
                # and ensure r_out is normalized by radius.
                self.Vmax = self.vsini * (
                    (self.r_out / self.radius)**(self.pOmega + 1.0))
                if verbose:
                    print(
                        f"[Grid] Computed Vmax = {self.Vmax:.2f} km/s from r_out={self.r_out:.2f}, radius={self.radius:.2f}, vsini={self.vsini:.2f}"
                    )
            else:
                # Fallback: Use vsini
                self.Vmax = self.vsini
                if verbose:
                    print(
                        f"[Grid] Warning: r_out not specified, using Vmax = vsini = {self.Vmax:.2f} km/s"
                    )

        # Calculate synchronous orbit radius and output
        if self.mass > 0 and self.period > 0:
            # Kepler's third law: a^3 = M * P^2 (in AU, M_sun, Year units)
            # Units: solar mass, days, solar radius
            P_year = self.period / 365.25
            a_AU = (self.mass * P_year**2)**(1. / 3.)
            a_Rsun = a_AU * 215.032  # 1 AU = 215.032 R_sun
            self.corotation_radius = a_Rsun / self.radius  # In units of stellar radius
            if verbose:
                print(
                    f"[Corotation] Synchronous orbit at r_sync = {self.corotation_radius:.3f} R* ({a_Rsun:.3f} R_sun)"
                )
        else:
            self.corotation_radius = None

        # Convert spectral resolution to FWHM (km/s)
        # Need wl0 from line file, record here first, actual conversion after reading lines
        self.instrumentRes = None  # Will be set after reading lines

        # Calculate phase for each observation (based on jDateRef and period)
        if hasattr(self, 'jDateRef') and hasattr(self, 'period'):
            self.phases = compute_phase_from_jd(self.jDates, self.jDateRef,
                                                self.period)
        else:
            # If no jDateRef, set phases to None
            self.phases = None

        # Preserve legacy attribute normalization; tolerated but not required in new flow
        if hasattr(self, 'magGeomType'):
            magGeomType = self.magGeomType.lower()
            if not (magGeomType == 'full' or magGeomType == 'poloidal'
                    or magGeomType == 'pottor' or magGeomType == 'potential'):
                print(('WARNING: unrecognized magnetic geometry type ({:}).  '
                       'This project no longer relies on magnetic geometry.'
                       ).format(self.magGeomType))
            self.magGeomType = magGeomType

    def compute_instrument_fwhm(self, wl0, verbose=1):
        """Calculate instrument FWHM (km/s) from spectral resolution and central wavelength
        
        Parameters
        ----------
        wl0 : float
            Line central wavelength (Angstrom)
        verbose : int
            Whether to print info
        
        Returns
        -------
        fwhm_kms : float
            Instrument FWHM (km/s)
        """
        if self.spectralResolution > 0:
            # FWHM (km/s) = c / R
            fwhm_kms = c / self.spectralResolution
            self.instrumentRes = fwhm_kms
            if verbose:
                print(
                    f"[Instrument] R = {self.spectralResolution:.0f}, FWHM = {fwhm_kms:.3f} km/s at wl0={wl0:.2f}Å"
                )
            return fwhm_kms
        else:
            if verbose:
                print(
                    "[Instrument] Warning: spectralResolution not set, no convolution will be applied"
                )
            self.instrumentRes = 0.0
            return 0.0

    def calcCycles(self, verbose=1):
        """Calculate observation phase/cycle number (already calculated in __init__, this method keeps backward compatibility)
        
        Unified use of self.phases as main attribute, self.cycleList as alias.
        """
        # cycleList is backward compatible alias for phases
        if hasattr(self, 'phases') and self.phases is not None:
            self.cycleList = self.phases
        else:
            # If phases not calculated in __init__ (e.g. missing jDateRef), calculate here
            self.cycleList = compute_phase_from_jd(self.jDates, self.jDateRef,
                                                   self.period)
            self.phases = self.cycleList

        if ((self.pOmega != 0.) & (verbose == 1)):
            # Note: Original dOmega interpretation as angular shear rate is deprecated
            # Now pOmega is the power-law index for differential rotation: Ω(r) ∝ r^pOmega
            print('Differential rotation: Omega(r) ~ r^{:.2f}'.format(
                self.pOmega))
            print(
                '    observations span: {:8.4f} days, or {:8.4f} rotation cycles'
                .format(
                    np.max(self.jDates) - np.min(self.jDates),
                    np.max(self.cycleList) - np.min(self.cycleList)))

    def load_initial_model_from_tomog(self, verbose=1):
        """
        Load initial model parameters (magnetic field, brightness, etc.) from .tomog file
        
        If initTomogFile=1 and initModelPath is valid, then:
        1. Read .tomog file to get model data and metadata
        2. Extract geometric parameters (inclination, pOmega, r0, period, etc.)
        3. If parameters in .tomog differ from current parameters, generate params_tomog_int.txt (internal parameter file)
        4. Return (geom_loaded, meta_loaded, model_data); otherwise return (None, None, None)
        
        Returns
        -------
        geom_loaded : SimpleNamespace or None
            Loaded geometry object (containing grid, B_los, B_perp, chi, etc.)
        meta_loaded : dict or None
            Metadata from .tomog file
        model_data : dict or None
            Raw model data table (r, phi, Blos, Bperp, chi, brightness, etc.)
        """
        if not hasattr(self, 'initTomogFile') or self.initTomogFile != 1:
            if verbose:
                print("[Model] initTomogFile disabled (or not set)")
            return None, None, None

        if not hasattr(self, 'initModelPath') or self.initModelPath is None:
            if verbose:
                print(
                    "[Model] Warning: initTomogFile=1 but initModelPath not specified"
                )
            return None, None, None

        model_path = Path(self.initModelPath)
        if not model_path.exists():
            if verbose:
                print(
                    f"[Model] Error: initModelPath '{self.initModelPath}' does not exist"
                )
            return None, None, None

        # Call VelspaceDiskIntegrator.read_geomodel to read model
        try:
            from core.disk_geometry_integrator import VelspaceDiskIntegrator
            geom_loaded, meta_loaded, model_table = VelspaceDiskIntegrator.read_geomodel(
                str(model_path))

            if verbose:
                print(
                    f"[Model] Successfully loaded initial model from {self.initModelPath}"
                )
                print(
                    f"[Model]   inclination_deg: {meta_loaded.get('inclination_deg', 'N/A')}"
                )
                print(f"[Model]   pOmega: {meta_loaded.get('pOmega', 'N/A')}")
                print(f"[Model]   r0_rot: {meta_loaded.get('r0_rot', 'N/A')}")
                print(f"[Model]   period: {meta_loaded.get('period', 'N/A')}")

            # Check if params_tomog_int.txt needs to be generated
            self._check_and_generate_internal_params(geom_loaded, meta_loaded,
                                                     verbose)

            return geom_loaded, meta_loaded, model_table

        except Exception as e:
            if verbose:
                print(
                    f"[Model] Error loading model from {self.initModelPath}: {e}"
                )
            import traceback
            traceback.print_exc()
            return None, None, None

    def _check_and_generate_internal_params(self, geom, meta, verbose=1):
        """
        Check if parameters in .tomog model conflict with parameters in current parameter file.
        If conflict exists, generate params_tomog_int.txt file, recording authoritative parameters from .tomog.
        
        Parameters
        ----------
        geom : SimpleNamespace
            Geometry object read from .tomog
        meta : dict
            Metadata read from .tomog
        verbose : int
            Output verbosity
        """

        # Parameter dictionary to compare (key in .tomog -> attribute in current readParamsTomog)
        param_mapping = {
            "inclination_deg": ("inclination", lambda x: float(x)),
            "pOmega": ("pOmega", lambda x: float(x)),
            "r0_rot": ("radius", lambda x: float(x)),
            "period": ("period", lambda x: float(x)),
        }

        conflicts = {}
        for tomog_key, (attr_name, converter) in param_mapping.items():
            if tomog_key not in meta:
                continue

            tomog_val = converter(meta[tomog_key])
            current_val = getattr(self, attr_name, None)

            if current_val is not None:
                # Compare (allow small numerical error)
                if abs(tomog_val - current_val) > 1e-6:
                    conflicts[attr_name] = {
                        "input_file": current_val,
                        "tomog_model": tomog_val
                    }

        if not conflicts:
            if verbose:
                print("[Model] No parameter conflicts detected.")
            return

        # Conflict exists, generate params_tomog_int.txt
        if verbose:
            print("[Model] Parameter conflicts detected:")
            for attr_name, vals in conflicts.items():
                print(
                    f"  {attr_name}: input={vals['input_file']}, tomog={vals['tomog_model']}"
                )
            print(
                "[Model] Generating params_tomog_int.txt with .tomog parameters as authority..."
            )

        # Generate internal parameter file
        int_params_path = Path("output") / "params_tomog_int.txt"
        int_params_path.parent.mkdir(parents=True, exist_ok=True)

        with open(int_params_path, 'w') as f:
            f.write(
                "# params_tomog_int.txt - Internal parameters from loaded .tomog model\n"
            )
            f.write(f"# Generated from: {self.initModelPath}\n")
            f.write(f"# Generated at: {dt.datetime.now().isoformat()}\n")
            f.write(
                "# This file records parameters from the .tomog model that differ from input/params_tomog.txt\n"
            )
            f.write(
                "# These parameters should be used as the authority for subsequent analyses.\n"
            )
            f.write("#\n")

            for attr_name, vals in conflicts.items():
                f.write(f"# {attr_name}:\n")
                f.write(f"#   input_file value: {vals['input_file']}\n")
                f.write(f"#   tomog_model value: {vals['tomog_model']}\n")

            f.write("#\n")
            f.write("# Full .tomog metadata:\n")
            for k in sorted(meta.keys()):
                f.write(f"#   {k}: {meta[k]}\n")

        if verbose:
            print(f"[Model] Generated {int_params_path}")

    def setTarget(self):
        # Check whether to fit to a target chi^2 or to a target entropy
        if (self.targetForm == 'C'):
            self.fixedEntropy = 0
            self.chiTarget = self.targetValue
            self.ent_aim = -1e6
        elif (self.targetForm == 'E'):
            self.fixedEntropy = 1
            self.ent_aim = self.targetValue
            self.chiTarget = 1.0
        else:
            print(
                'ERROR unknown format for goodness of fit target: {:}'.format(
                    self.targetForm))
            import sys
            sys.exit()

    def write_params_file(self, outfile, verbose=1):
        """
        Write new parameter file based on existing parameter information.
        
        Parameters
        ----------
        outfile : str
            Output file path (e.g. output/params_tomog_new.txt)
        verbose : int
            Output verbosity (0=no output, 1=basic info, 2=detailed info)
            
        Returns
        -------
        bool
            True if write successful, False otherwise
            
        Notes
        -----
        The written format is identical to the original parameter file format, including all 14+ lines of parameters and observation data.
        This method is used for:
        1. Saving parameters after modification (e.g. modified inclination, vsini etc.)
        2. Generating new standard parameter file after loading parameters from other sources (e.g. .tomog file)
        3. Parameter validation and documentation (output parameter file for review)
        """

        outpath = Path(outfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(outfile, 'w') as f:
                # File header comments
                f.write("# pyZeeTom parameter file (auto-generated)\n")
                f.write(f"# Generated: {dt.datetime.now().isoformat()}\n")
                f.write(
                    "# Lines are positional. Blank lines and lines starting with # are ignored.\n"
                )
                f.write(
                    "# Units: angles in degrees, velocities km/s, dates in HJD (Heliocentric Julian Date).\n"
                )
                f.write("\n")

                # Line 0: inclination vsini period pOmega
                f.write("#0 inclination  vsini  period  pOmega\n")
                f.write(
                    "# period: rotation period at reference radius r0 (in days)\n"
                )
                f.write(
                    "# pOmega: power-law index for differential rotation, Ω(r) = Ω_ref × (r/r0)^pOmega\n"
                )
                f.write(
                    f"{self.inclination:.1f}  {self.vsini:.1f}  {self.period:.4f}  {self.pOmega:.3f}\n"
                )
                f.write("\n")

                # Line 1: mass radius [Vmax] [r_out] [enable_occultation]
                f.write(
                    "#1 mass  radius  [Vmax]  [r_out]  [enable_occultation]\n")
                f.write(
                    "# radius: stellar radius (R_sun), also used as reference radius r0\n"
                )
                f.write(
                    "# r_out: outer disk radius (in units of stellar radius)\n"
                )
                f.write("# enable_occultation: 0=off, 1=on\n")
                enable_occ = getattr(self, 'enable_stellar_occultation', 0)
                f.write(
                    f"{self.mass:.2f}  {self.radius:.2f}  {self.Vmax:.4f}  {self.r_out:.2f}  {enable_occ}\n"
                )
                f.write("\n")

                # Line 2: nRingsStellarGrid
                f.write("#2 nRingsStellarGrid\n")
                f.write(f"{self.nRingsStellarGrid}\n")
                f.write("\n")

                # Line 3: targetForm targetValue numIterations
                f.write(
                    "#3 targetForm  targetValue  numIterations   (C=chi^2 target, E=entropy target)\n"
                )
                f.write(
                    f"{self.targetForm}  {self.targetValue:.4f}  {self.numIterations}\n"
                )
                f.write("\n")

                # Line 4: test_aim
                f.write(
                    "#4 test_aim (convergence threshold for Test statistic)\n")
                f.write(f"{self.test_aim:.2e}\n")
                f.write("\n")

                # Line 5: lineAmpConst k_QU enableV enableQU
                f.write("#5 lineAmpConst  k_QU  enableV  enableQU\n")
                f.write(
                    f"{self.lineAmpConst:.1f}  {self.lineKQU:.1f}  {self.lineEnableV}  {self.lineEnableQU}\n"
                )
                f.write("\n")

                # Line 6: initTomogFile initModelPath
                f.write("#6 initTomogFile  initModelPath\n")
                init_tomog = getattr(self, 'initTomogFile', 0)
                init_path = getattr(self, 'initModelPath', '')
                f.write(f"{init_tomog}  {init_path}\n")
                f.write("\n")

                # Line 7: fitBri fitMag fitBlos fitBperp fitChi
                f.write("#7 fitBri fitMag fitBlos fitBperp fitChi\n")
                f.write(
                    f"{self.fitBri} {self.fitMag} {self.fitBlos} {self.fitBperp} {self.fitChi}\n"
                )
                f.write("\n")

                # Line 8: spectralResolution lineParamFile
                f.write("#8 spectralResolution  lineParamFile\n")
                f.write(
                    f"{self.spectralResolution:.0f}  {self.lineParamFile}\n")
                f.write("\n")

                # Line 9: velStart velEnd obsFileType specType
                f.write("#9 velStart  velEnd  obsFileType  specType\n")
                spec_type_str = getattr(self, 'specType', 'auto')
                f.write(
                    f"{self.velStart:.1f}  {self.velEnd:.1f}  {self.obsFileType}  specType={spec_type_str}\n"
                )
                f.write("\n")

                # Line 10: jDateRef
                f.write(
                    "#10 jDateRef  (HJD0, reference epoch for phase calculation)\n"
                )
                f.write(f"{self.jDateRef:.4f}\n")
                f.write("\n")

                # Line 11+: Observation data
                f.write(
                    "#11+ observation entries: filename  HJD  velR  [polchannel]\n"
                )
                f.write("# polchannel: I/V/Q/U (optional, default=V)\n")
                for i in range(self.numObs):
                    fname = self.fnames[i]
                    hjd = self.jDates[i]
                    velr = self.velRs[i]
                    polch = self.polChannels[i] if i < len(
                        self.polChannels) else 'V'
                    f.write(f"{fname}  {hjd:.2f}  {velr:.2f}  {polch}\n")

                if verbose >= 1:
                    print(
                        f"[writeParamsFile] Successfully wrote parameters to {outfile}"
                    )
                    if verbose >= 2:
                        print(f"  - {self.numObs} observation entries")
                        print(
                            f"  - Grid: inclination={self.inclination:.1f}°, vsini={self.vsini:.1f} km/s"
                        )
                        print(
                            f"  - Grid velocity range: Vmax={self.Vmax:.1f} km/s"
                        )
                        print(
                            f"  - Line parameters from: {self.lineParamFile}")
                        print(
                            f"  - Spectral resolution: {self.spectralResolution:.0f}"
                        )

                return True

        except Exception as e:
            if verbose >= 1:
                print(f"[writeParamsFile] Error writing to {outfile}: {e}")
            import traceback
            traceback.print_exc()
            return False


#############################################
# MEM Iteration Loop Helper Functions
#############################################


def compute_forward_single_phase(integrator,
                                 mag_field=None,
                                 brightness=None,
                                 compute_derivatives=False,
                                 eps_blos=10.0,
                                 eps_bperp=10.0,
                                 eps_chi=0.01):
    """
    Compute forward model (and optional parameter derivatives) for a single observation phase
    
    Parameters
    ----------
    integrator : VelspaceDiskIntegrator
        Configured disk integrator instance
    mag_field : MagneticFieldParams, optional
        Magnetic field parameter object (containing Blos, Bperp, chi)
    brightness : np.ndarray, optional
        Brightness distribution (Npix,)
    compute_derivatives : bool
        Whether to compute parameter derivatives (response matrix)
    eps_blos, eps_bperp, eps_chi : float
        Numerical differentiation step size
        
    Returns
    -------
    result : dict
        Contains:
        - specI, specQ, specU, specV : Synthetic Stokes spectra (Nlambda,)
        - dI_dBlos, dV_dBlos, ... : Derivative matrices (Nlambda, Npix), if compute_derivatives=True
    """
    result = {}

    # Baseline forward
    result['specI'] = integrator.I.copy()
    result['specV'] = integrator.V.copy()
    # Q and U may not be available, use zero array
    result['specQ'] = (integrator.Q.copy() if hasattr(integrator, 'Q') else
                       np.zeros_like(integrator.I))
    result['specU'] = (integrator.U.copy() if hasattr(integrator, 'U') else
                       np.zeros_like(integrator.I))

    if not compute_derivatives:
        return result

    # Compute derivatives (placeholder implementation - full implementation requires perturbing parameters and re-integrating)
    npix = len(integrator.geom.grid.r)
    nlambda = len(integrator.v)  # v_grid is stored as integrator.v

    result['dI_dBlos'] = np.zeros((nlambda, npix))
    result['dV_dBlos'] = np.zeros((nlambda, npix))
    result['dI_dBperp'] = np.zeros((nlambda, npix))
    result['dV_dBperp'] = np.zeros((nlambda, npix))
    result['dQ_dBperp'] = np.zeros((nlambda, npix))
    result['dU_dBperp'] = np.zeros((nlambda, npix))
    result['dQ_dchi'] = np.zeros((nlambda, npix))
    result['dU_dchi'] = np.zeros((nlambda, npix))

    # TODO: Full derivative calculation requires perturbing each parameter and re-integrating
    # Placeholder values used here

    return result


def save_iteration_summary(outfile,
                           iteration,
                           chi2,
                           entropy,
                           test_value,
                           mag_field=None,
                           brightness=None,
                           extra_info=None,
                           mode='append'):
    """
    Save single MEM iteration summary to file
    
    Parameters
    ----------
    outfile : str
        Output file path (e.g. output/outSummary.txt)
    iteration : int
        Iteration number
    chi2 : float
        Current χ² value
    entropy : float
        Current entropy value
    test_value : float
        Convergence test value
    mag_field : MagneticFieldParams, optional
        Magnetic field parameters (calculate average Blos etc.)
    brightness : np.ndarray, optional
        Brightness distribution (calculate average brightness)
    extra_info : dict, optional
        Extra info (e.g. number of phases, number of parameters)
    mode : str
        'append' or 'overwrite'
    """

    outpath = Path(outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # If first iteration or overwrite mode, write header
    if mode == 'overwrite' or (mode == 'append' and iteration == 0):
        with open(outfile, 'w') as f:
            f.write("# MEM Inversion Summary\n")
            f.write("# Generated by pyZeeTom\n")
            f.write("#\n")
            f.write(
                "# Columns: Iteration  Chi2  Entropy  TestValue  AvgBlos  AvgBperp  AvgBrightness\n"
            )
            f.write("#" + "-" * 78 + "\n")

    # Calculate statistics
    avg_blos = np.mean(mag_field.Blos) if mag_field is not None else 0.0
    avg_bperp = np.mean(mag_field.Bperp) if mag_field is not None else 0.0
    avg_bright = np.mean(brightness) if brightness is not None else 1.0

    # Append current iteration info
    with open(outfile, 'a') as f:
        f.write(
            f"{iteration:4d}  {chi2:12.4f}  {entropy:12.6f}  {test_value:12.6e}  "
            f"{avg_blos:10.3f}  {avg_bperp:10.3f}  {avg_bright:10.6f}\n")

    # If extra_info provided, append detailed info
    if extra_info is not None and iteration == 0:
        with open(outfile, 'a') as f:
            f.write("\n# Additional Information:\n")
            for key, val in extra_info.items():
                f.write(f"#   {key}: {val}\n")


def save_model_spectra(results,
                       phase_indices,
                       output_dir="output/outModel",
                       fmt="lsd",
                       prefix="phase"):
    """
    Save model spectra for each observation phase to separate files
    
    Parameters
    ----------
    results : list of tuples
        Each tuple is (v_grid, specI, specV, specQ, specU, pol_channel) or
        (v_grid, specI, specV, specQ, specU) or (v_grid, specI, specV)
    phase_indices : list of int
        Phase indices
    output_dir : str
        Output directory
    fmt : str
        Output format ('lsd' or 'spec')
    prefix : str
        Filename prefix
    """

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        phase_idx = phase_indices[i] if i < len(phase_indices) else i

        # Unpack results, support new format (including pol_channel)
        if len(result) >= 6:
            v_grid, specI, specV, specQ, specU, pol_channel = result[:6]
        elif len(result) >= 5:
            v_grid, specI, specV, specQ, specU = result[:5]
            pol_channel = 'V'  # Default
        elif len(result) >= 3:
            v_grid, specI, specV = result[:3]
            specQ = np.zeros_like(specI)
            specU = np.zeros_like(specI)
            pol_channel = 'V'
        else:
            continue

        # Build output filename
        if fmt.lower() == 'lsd':
            outfile = outdir / f"{prefix}{phase_idx:03d}.lsd"
        else:
            outfile = outdir / f"{prefix}{phase_idx:03d}.spec"

        # Use SpecIO.write_model_spectrum to write file, support pol_channel
        header = {"phase_index": str(phase_idx), "pol_channel": pol_channel}
        SpecIO.write_model_spectrum(str(outfile),
                                    v_grid,
                                    specI,
                                    V=specV,
                                    Q=specQ,
                                    U=specU,
                                    fmt=fmt,
                                    header=header,
                                    pol_channel=pol_channel)


def save_model_spectra_to_outModelSpec(par,
                                       results,
                                       obsSet,
                                       output_dir="output/outModelSpec",
                                       verbose=1):
    """
    Save model spectra to output/outModelSpec directory, organized by observation file format.

    Generate model spectrum file for each observation in corresponding format based on observation file format (spec/lsd), wavelength range, velocity range and polarization channel info.

    Parameters
    ----------
    par : readParamsTomog
        Parameter object, containing fnames, jDates, velRs, polChannels, phases etc.
    results : list of tuples
        Each tuple is (v_grid, specI, specV, specQ, specU, pol_channel)
        or (v_grid, specI, specV, specQ, specU)
    obsSet : list of ObservationProfile
        Observation data object list, used to get format, wavelength range etc.
    output_dir : str
        Output directory (default: output/outModelSpec)
    verbose : int
        Verbosity level

    Returns
    -------
    list of str
        List of generated file paths
    """

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_files = []

    if verbose:
        print(
            f"\n[save_model_spectra_to_outModelSpec] Saving {len(results)} model spectra..."
        )

    for i, result in enumerate(results):
        if i >= len(obsSet):
            if verbose:
                print(
                    f"  Warning: result count ({len(results)}) > obsSet count ({len(obsSet)})"
                )
            break

        # Get observation info
        obs = obsSet[i]

        # Extract observation parameters
        hjd = par.jDates[i] if i < len(par.jDates) else 0.0
        vel_r = par.velRs[i] if i < len(par.velRs) else 0.0
        pol_channel = str(par.polChannels[i]).upper() if i < len(
            par.polChannels) else 'V'
        phase = par.phases[i] if hasattr(par, 'phases') and i < len(
            par.phases) else (i / len(results) if len(results) > 0 else 0.0)

        # Unpack results
        if len(result) >= 6:
            v_grid, specI, specV, specQ, specU, pol_ch_from_result = result[:6]
            # Use pol_channel from result (if available)
            if pol_ch_from_result is not None:
                pol_channel = str(pol_ch_from_result).upper()
        elif len(result) >= 5:
            v_grid, specI, specV, specQ, specU = result[:5]
        elif len(result) >= 3:
            v_grid, specI, specV = result[:3]
            specQ = np.zeros_like(specI)
            specU = np.zeros_like(specI)
        else:
            if verbose:
                print(f"  Warning: result {i} format incorrect, skipping")
            continue

        # Determine output format (inferred from observation object)
        obs_format = obs.profile_type.lower() if hasattr(
            obs, 'profile_type') else 'spec'
        if obs_format == 'velocity' or obs_format == 'lsd':
            fmt = 'lsd'
            ext = '.lsd'
        else:
            fmt = 'spec'
            ext = '.spec'

        # Determine file_type_hint based on pol_channel
        if pol_channel == 'I':
            file_type_hint = 'lsd_i' if fmt == 'lsd' else 'spec_i'
        else:
            file_type_hint = 'lsd_pol' if fmt == 'lsd' else 'spec_pol'

        # Build output filename
        # Format: phase_XXXX_HJDpYYY_VRsZZZ_CH.ext
        # e.g.: phase_0000_HJDp0p200_VRs0p00_V.lsd
        hjd_str = f"{hjd:.3f}".replace('.', 'p')
        vel_r_sign = 'p' if vel_r >= 0 else 'm'
        vel_r_abs = abs(vel_r)
        vel_r_str = f"{vel_r_sign}{vel_r_abs:.2f}".replace('.', 'p')

        outfile_name = f"phase_{i:04d}_HJD{hjd_str}_VR{vel_r_str}_{pol_channel}{ext}"
        outfile = outdir / outfile_name

        # Use SpecIO to save model spectrum
        try:
            header = {
                "phase_index": str(i),
                "HJD": f"{hjd:.6f}",
                "velR": f"{vel_r:.2f}",
                "pol_channel": str(pol_channel),
                "phase": f"{phase:.4f}",
            }

            # Call SpecIO.write_model_spectrum, ensure consistent format
            SpecIO.write_model_spectrum(str(outfile),
                                        v_grid,
                                        specI,
                                        V=specV,
                                        Q=specQ,
                                        U=specU,
                                        fmt=fmt,
                                        header=header,
                                        pol_channel=pol_channel,
                                        file_type_hint=file_type_hint)

            output_files.append(str(outfile))

            if verbose > 1:
                print(
                    f"  [{i:2d}] HJD={hjd:.3f}, phase={phase:.4f}, VR={vel_r:+.2f}, "
                    f"CH={pol_channel}: {outfile.name}")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to save file {outfile.name}: {e}")

    if verbose:
        print(
            f"[save_model_spectra_to_outModelSpec] Done! Generated {len(output_files)} files to {outdir}"
        )
        print("  File format: phase_XXXX_HJDYYY_VRZZ_CH.ext")
        print("    XXXX = Observation index (0000-9999)")
        print("    YYY = Heliocentric Julian Date (p=decimal point)")
        print("    ZZ = Radial velocity correction (km/s)")
        print("    CH = Polarization channel (I/V/Q/U)")
        print("    ext = Format suffix (.spec or .lsd)")

    return output_files


#############################################
