import numpy as np

c = 2.99792458e5


def compute_phase_from_jd(jd, jd_ref, period):
    """计算观测相位（rotation phase）
    
    Parameters
    ----------
    jd : float or array-like
        观测的 Julian Date（Heliocentric Julian Date）
    jd_ref : float
        参考时刻的 Julian Date（HJD0）
    period : float
        自转周期（天）
        
    Returns
    -------
    phase : float or ndarray
        相位，phase = (jd - jd_ref) / period
        对于刚体转动，phase 直接对应观测角度偏移：Δφ = 2π × phase
        对于差速转动，每个环的相位演化由各自的角速度决定
    """
    return (np.asarray(jd) - float(jd_ref)) / float(period)


class readParamsTomog:
    """读取tomography参数文件
    
    支持两种网格定义方式：
    1. 直接指定 Vmax（行1第3列非零）：不依赖 radius/vsini/inclination
    2. 使用 radius + r_out + vsini + inclination 计算 Vmax
    """

    def __init__(self, inParamsName, verbose=1):
        # Read in the model and control parameters
        fInZDI = open(inParamsName, 'r')
        self.fnames = np.array([])
        self.jDates = np.array([])
        self.velRs = np.array([])
        self.numObs = 0
        # New defaults for tomography workflow
        self.lineParamFile = 'lines.txt'  # path to line model parameters
        self.obsFileType = 'auto'  # observation format hint for readObs
        self.enable_stellar_occultation = 0  # 默认关闭恒星遮挡
        i = 0
        for inLine in fInZDI:
            if (inLine.strip() == ''):  # skip blank lines
                continue
            # check for comments (ignoring white-space)
            if (inLine.strip()[0] != '#'):
                if (i == 0):
                    # 行0: inclination vsini period pOmega
                    parts = inLine.split()
                    self.inclination = float(parts[0])
                    self.vsini = float(parts[1])
                    self.period = float(parts[2])
                    self.pOmega = float(parts[3])
                    # Legacy alias for backward compatibility
                    self.dOmega = self.pOmega
                elif (i == 1):
                    # 行1: mass radius [Vmax] [r_out] [enable_occultation]
                    # Vmax非零时使用Vmax定义网格，否则从radius+r_out+vsini+inclination计算
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
                    self.initMagFromFile = int(inLine.split()[0])
                    self.initMagGeomFile = inLine.split()[1]
                elif (i == 7):
                    self.fitBri = int(inLine.split()[0])
                    self.chiScaleI = float(inLine.split()[1])
                    self.brightEntScale = float(inLine.split()[2])
                elif (i == 8):
                    self.fEntropyBright = int(inLine.split()[0])
                    self.defaultBright = float(inLine.split()[1])
                    self.maximumBright = float(inLine.split()[2])
                elif (i == 9):
                    self.initBrightFromFile = int(inLine.split()[0])
                    self.initBrightFile = inLine.split()[1]
                elif (i == 10):
                    self.estimateStrenght = int(inLine.split()[0])
                elif (i == 11):
                    # 行11: spectralResolution lineParamFile
                    # spectralResolution 现为分辨率（如65000），将转换为FWHM (km/s)
                    parts = inLine.split()
                    self.spectralResolution = float(parts[0])
                    # optional: path to line parameter file (e.g., lines.txt)
                    if len(parts) >= 2:
                        self.lineParamFile = parts[1]
                elif (i == 12):
                    parts = inLine.split()
                    self.velStart = float(parts[0])
                    self.velEnd = float(parts[1])
                    # optional: global observation file type hint (auto|lsd_i|lsd_pol|spec_i|...)
                    if len(parts) >= 3:
                        self.obsFileType = parts[2]
                elif (i == 13):
                    self.jDateRef = float(inLine.split()[0])
                elif (i >= 14):
                    self.fnames = np.append(self.fnames, [inLine.split()[0]])
                    self.jDates = np.append(self.jDates,
                                            [float(inLine.split()[1])])
                    self.velRs = np.append(self.velRs,
                                           [float(inLine.split()[2])])
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

        # 计算派生参数
        self.incRad = self.inclination / 180. * np.pi
        self.velEq = self.vsini / np.sin(self.incRad) if np.sin(
            self.incRad) > 1e-6 else 0.0

        # 确定网格速度范围（两种方式二选一）
        if abs(self.Vmax) > 1e-6:
            # 方式1: 直接使用 Vmax
            if verbose:
                print(f"[Grid] Using direct Vmax = {self.Vmax:.2f} km/s")
        else:
            # 方式2: 从 radius + r_out + vsini + inclination 计算
            if self.r_out > 0 and self.radius > 0:
                # 盘外边缘速度（刚体转动）：v(r_out) = veq * r_out / radius
                # 考虑差速：v(r) = veq * (r/radius)^(pOmega+1)
                r_ratio = self.r_out / self.radius
                self.Vmax = self.velEq * (r_ratio**(self.pOmega + 1.0))
                if verbose:
                    print(
                        f"[Grid] Computed Vmax = {self.Vmax:.2f} km/s from r_out={self.r_out:.2f}R*, vsini={self.vsini:.2f} km/s"
                    )
            else:
                # 回退：使用 vsini
                self.Vmax = self.vsini
                if verbose:
                    print(
                        f"[Grid] Warning: r_out not specified, using Vmax = vsini = {self.Vmax:.2f} km/s"
                    )

        # 计算同步轨道半径并输出
        if self.mass > 0 and self.period > 0:
            # Kepler第三定律: a^3 = G*M*P^2/(4π^2)
            # 以太阳质量、天、太阳半径为单位
            G_solar = 4 * np.pi**2  # AU^3 / (M_sun * year^2)
            P_year = self.period / 365.25
            a_AU = (G_solar * self.mass * P_year**2)**(1. / 3.)
            a_Rsun = a_AU * 215.032  # 1 AU = 215.032 R_sun
            self.corotation_radius = a_Rsun / self.radius  # 以恒星半径为单位
            if verbose:
                print(
                    f"[Corotation] Synchronous orbit at r_sync = {self.corotation_radius:.3f} R* ({a_Rsun:.3f} R_sun)"
                )
        else:
            self.corotation_radius = None

        # 将光谱分辨率转换为 FWHM (km/s)
        # 需要结合谱线文件中的 wl0，此处先记录，实际转换在读取谱线后进行
        self.instrumentRes = None  # 将在读取谱线后设置

        # 计算每个观测的相位（基于 jDateRef 和 period）
        if hasattr(self, 'jDateRef') and hasattr(self, 'period'):
            self.phases = compute_phase_from_jd(self.jDates, self.jDateRef,
                                                self.period)
        else:
            # 如果没有 jDateRef，则相位设为 None
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
        """根据光谱分辨率和中心波长计算仪器FWHM (km/s)
        
        Parameters
        ----------
        wl0 : float
            谱线中心波长 (Angstrom)
        verbose : int
            是否输出信息
        
        Returns
        -------
        fwhm_kms : float
            仪器FWHM (km/s)
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
        """计算观测相位/周期数（已在 __init__ 中计算，此方法保持向后兼容）
        
        统一使用 self.phases 作为主属性，self.cycleList 作为别名。
        """
        # cycleList 是 phases 的向后兼容别名
        if hasattr(self, 'phases') and self.phases is not None:
            self.cycleList = self.phases
        else:
            # 如果 __init__ 未计算 phases（例如缺少 jDateRef），则在此计算
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

    def setCalcdIdV(self, verbose=1):
        # Check which set of line profile to fit and which derivatives needs to be calculated
        self.calcDI = 0
        self.calcDV = 0
        if (self.fitBri == 1):
            self.calcDI = 1
        elif ((self.fitBri > 1) | (self.fitBri < 0)):
            print("ERROR: invalid value of fitBri: {:}".format(self.fitBri))
        if (self.fitMag == 1):
            self.calcDV = 1
        elif ((self.fitMag > 1) | (self.fitMag < 0)):
            print("ERROR: invalid value of fitMag: {:}".format(self.fitMag))
        if ((self.calcDI == 0) & (self.calcDV == 0)):
            if (verbose == 1):
                print("Warning: no parameters to fit!")
            self.numIterations = 0
            # import sys  # legacy early exit suppressed for tomography workflow

        if ((self.fEntropyBright != 1) & (self.fEntropyBright != 2)):
            print('error unrecognized brightness entropy flag: {:}'.format(
                self.fEntropyBright))
            import sys
            sys.exit()


def mainFittingLoop(par,
                    lineData,
                    wlSynSet,
                    sGrid,
                    briMap,
                    magGeom,
                    listGridView,
                    dMagCart0,
                    setSynSpec,
                    coMem,
                    nDataTot,
                    Data,
                    sig2,
                    allModeldIdV,
                    weightEntropy,
                    verbose=1):

    import core.memSimple3 as memSimple
    # import core.lineprofileVoigt as lineprofile

    chi_aim = par.chiTarget * float(coMem.nDataTotIV)
    if (par.fixedEntropy == 1):
        target_aim = par.ent_aim
    else:
        target_aim = chi_aim

    fOutFitSummary = open('outFitSummary.txt', 'w')

    # Initialize goodness of fit and convergence parameters
    Chi2 = chi2nu = 0.0
    entropy = 0.0
    test = 1.0
    meanBright = meanBrightDiff = meanMag = 0.0
    iIter = 0
    bConverged = False

    # Loop over fitting iterations; allows fitting to target entropy or chi^2
    while (iIter < par.numIterations) and (not bConverged):

        # Compute set of new spectra and derivatives
        iIter += 1
        if (iIter > 1):
            memSimple.unpackImageVector(Image, briMap, magGeom,
                                        par.magGeomType, par.fitBri,
                                        par.fitMag)

    # First get magnetic vectors (their derivatives can be pre-calculated)
        vecMagCart = magGeom.getAllMagVectorsCart()

        nObs = 0
        for phase in par.cycleList:
            # get stellar geometry calculations for this phase
            sGridView = listGridView[nObs]
            # calculate spectrum and derivatives
            spec = setSynSpec[nObs]
            spec.updateIntProfDeriv(sGridView, vecMagCart, dMagCart0, briMap,
                                    lineData, par.calcDI, par.calcDV)
            spec.convolveIGnumpy(par.instrumentRes)

            nObs += 1
        #finished computing spectra and derivatives

        #Pack the input arrays for mem_iter
        allModelIV = memSimple.packModelVector(setSynSpec, par.fitBri,
                                               par.fitMag)
        Image = memSimple.packImageVector(briMap, magGeom, par.magGeomType,
                                          par.fitBri, par.fitMag)
        if ((par.calcDI == 1) | (par.calcDV == 1)):
            allModeldIdV = memSimple.packResponseMatrix(
                setSynSpec, nDataTot, coMem.npBriMap, magGeom, par.magGeomType,
                par.calcDI, par.calcDV)

        #Call the mem_iter routine controlling the fit.  This returns the entropy, Chi2, test
        # for the current model, and then proposes a new best fit model in Image
        entropy, Chi2, test, Image, entStand, entFF, entMag = \
            memSimple.mem_iter(coMem.n1Model, coMem.n2Model, coMem.nModelTot, \
                               Image, Data, allModelIV, sig2, allModeldIdV, \
                               weightEntropy, par.defaultBright, par.defaultBent, \
                               par.maximumBright, target_aim, par.fixedEntropy)

        meanBright = np.sum(briMap.bright * sGrid.area) / np.sum(sGrid.area)
        meanBrightDiff = np.sum(
            np.abs(briMap.bright - par.defaultBright) * sGrid.area) / np.sum(
                sGrid.area)
        absMagCart = np.sqrt(vecMagCart[0, :]**2 + vecMagCart[1, :]**2 +
                             vecMagCart[2, :]**2)
        meanMag = np.sum(absMagCart * sGrid.area) / np.sum(sGrid.area)

        #evaluate some convergence criteria:
        if (par.fixedEntropy == 1):
            bConverged = ((entropy >= par.ent_aim * 1.001)
                          and (test < par.test_aim) and (iIter > 2))
        else:
            bConverged = ((Chi2 <= chi_aim * 1.001) and (test < par.test_aim))
        if (coMem.nDataTotIV > 0):  #protect against fitting nothing
            chi2nu = Chi2 / float(coMem.nDataTotIV)
        else:
            chi2nu = Chi2

        if (verbose == 1):
            print(
                'it {:3n}  entropy {:13.5f}  chi2 {:10.6f}  Test {:10.6f} meanBright {:10.7f} meanSpot {:10.7f} meanMag {:10.4f}'
                .format(iIter, entropy, chi2nu, test, meanBright,
                        meanBrightDiff, meanMag))
        fOutFitSummary.write(
            'it {:3n}  entropy {:13.5f}  chi2 {:10.6f}  Test {:10.6f} meanBright {:10.7f} meanSpot {:10.7f} meanMag {:10.4f}\n'
            .format(iIter, entropy, chi2nu, test, meanBright, meanBrightDiff,
                    meanMag))
        if ((verbose == 1) and (bConverged == True)):
            print('Success: sufficiently small value of Test achieved')

    #In case this was run with no iterations, just calculate the model diagonstics
    if (iIter == 0):
        allModelIV = memSimple.packModelVector(setSynSpec, par.fitBri,
                                               par.fitMag)
        chi2nu = np.sum(
            (allModelIV - Data)**2 / sig2) / float(coMem.nDataTotIV)

        Image = memSimple.packImageVector(briMap, magGeom, par.magGeomType,
                                          par.fitBri, par.fitMag)
        if (par.fitBri == 1 or par.fitMag == 1):
            entropy, tmpgS, tmpggS, tmp3, tmp4, SI1, SI2, SB \
                = memSimple.get_s_grads(coMem.n1Model, coMem.n2Model, coMem.nModelTot,
                                        Image, weightEntropy, par.defaultBright,
                                        par.defaultBent, par.maximumBright)

        meanBright = np.sum(briMap.bright * sGrid.area) / np.sum(sGrid.area)
        meanBrightDiff = np.sum(np.abs(briMap.bright-par.defaultBright)*sGrid.area) \
                         /np.sum(sGrid.area)
        vecMagCart = magGeom.getAllMagVectorsCart()
        absMagCart = np.sqrt(vecMagCart[0, :]**2 + vecMagCart[1, :]**2 +
                             vecMagCart[2, :]**2)
        meanMag = np.sum(absMagCart * sGrid.area) / np.sum(sGrid.area)

    if (verbose != 1 and par.numIterations > 0) or (verbose == 1
                                                    and iIter == 0):
        print(
            'it {:3n}  entropy {:13.5f}  chi2 {:10.6f}  Test {:10.6f} meanBright {:10.7f} meanSpot {:10.7f} meanMag {:10.4f}'
            .format(iIter, entropy, chi2nu, test, meanBright, meanBrightDiff,
                    meanMag))
        fOutFitSummary.write(
            'it {:3n}  entropy {:13.5f}  chi2 {:10.6f}  Test {:10.6f} meanBright {:10.7f} meanSpot {:10.7f} meanMag {:10.4f}\n'
            .format(iIter, entropy, chi2nu, test, meanBright, meanBrightDiff,
                    meanMag))
    fOutFitSummary.close()

    return iIter, entropy, chi2nu, test, meanBright, meanBrightDiff, meanMag


#############################################
