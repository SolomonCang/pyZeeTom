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
        self.polChannels = np.array([])  # 新增：每个观测的偏振通道（I/V/Q/U）
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
                elif (i == 13):
                    self.jDateRef = float(inLine.split()[0])
                elif (i >= 14):
                    parts = inLine.split()
                    self.fnames = np.append(self.fnames, [parts[0]])
                    self.jDates = np.append(self.jDates, [float(parts[1])])
                    self.velRs = np.append(self.velRs, [float(parts[2])])
                    # 新增：读取polchannel（第4列），默认为'V'
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
                # 差速转动速度场：v(r) = veq * (r/R*)^(pOmega+1)
                # r_out 已经是恒星半径为单位（r/R*），所以：
                # Vmax = v(r_out) = veq * r_out^(pOmega+1)
                self.Vmax = self.velEq * (self.r_out**(self.pOmega + 1.0))
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


#############################################
# MEM迭代循环辅助函数
#############################################


def compute_forward_single_phase(integrator,
                                 mag_field=None,
                                 brightness=None,
                                 compute_derivatives=False,
                                 eps_blos=10.0,
                                 eps_bperp=10.0,
                                 eps_chi=0.01):
    """
    为单个观测相位计算正演模型（及可选的参数导数）
    
    Parameters
    ----------
    integrator : VelspaceDiskIntegrator
        已配置的盘积分器实例
    mag_field : MagneticFieldParams, optional
        磁场参数对象（包含Blos, Bperp, chi）
    brightness : np.ndarray, optional
        亮度分布 (Npix,)
    compute_derivatives : bool
        是否计算参数导数（响应矩阵）
    eps_blos, eps_bperp, eps_chi : float
        数值微分步长
        
    Returns
    -------
    result : dict
        包含：
        - specI, specQ, specU, specV : 合成Stokes谱 (Nlambda,)
        - dI_dBlos, dV_dBlos, ... : 导数矩阵 (Nlambda, Npix)，若compute_derivatives=True
    """
    result = {}

    # 基准正演
    result['specI'] = integrator.I.copy()
    result['specV'] = integrator.V.copy()
    # Q和U可能不可用，使用零数组
    result['specQ'] = (integrator.Q.copy() if hasattr(integrator, 'Q') else
                       np.zeros_like(integrator.I))
    result['specU'] = (integrator.U.copy() if hasattr(integrator, 'U') else
                       np.zeros_like(integrator.I))

    if not compute_derivatives:
        return result

    # 计算导数（占位实现 - 完整实现需要扰动参数并重新积分）
    npix = len(integrator.geom.grid.r)
    nlambda = len(integrator.v)  # v_grid被存储为integrator.v

    result['dI_dBlos'] = np.zeros((nlambda, npix))
    result['dV_dBlos'] = np.zeros((nlambda, npix))
    result['dI_dBperp'] = np.zeros((nlambda, npix))
    result['dV_dBperp'] = np.zeros((nlambda, npix))
    result['dQ_dBperp'] = np.zeros((nlambda, npix))
    result['dU_dBperp'] = np.zeros((nlambda, npix))
    result['dQ_dchi'] = np.zeros((nlambda, npix))
    result['dU_dchi'] = np.zeros((nlambda, npix))

    # TODO: 完整导数计算需扰动各参数并重新积分
    # 此处暂用占位值

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
    保存单次MEM迭代的统计摘要到文件
    
    Parameters
    ----------
    outfile : str
        输出文件路径（如 output/outSummary.txt）
    iteration : int
        迭代次数
    chi2 : float
        当前χ²值
    entropy : float
        当前熵值
    test_value : float
        收敛检验值
    mag_field : MagneticFieldParams, optional
        磁场参数（计算平均Blos等）
    brightness : np.ndarray, optional
        亮度分布（计算平均亮度）
    extra_info : dict, optional
        额外信息（如相位数、参数数等）
    mode : str
        'append' 或 'overwrite'
    """
    from pathlib import Path

    outpath = Path(outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # 如果是第一次迭代或overwrite模式，写入表头
    if mode == 'overwrite' or (mode == 'append' and iteration == 0):
        with open(outfile, 'w') as f:
            f.write("# MEM Inversion Summary\n")
            f.write("# Generated by pyZeeTom\n")
            f.write("#\n")
            f.write(
                "# Columns: Iteration  Chi2  Entropy  TestValue  AvgBlos  AvgBperp  AvgBrightness\n"
            )
            f.write("#" + "-" * 78 + "\n")

    # 计算统计量
    avg_blos = np.mean(mag_field.Blos) if mag_field is not None else 0.0
    avg_bperp = np.mean(mag_field.Bperp) if mag_field is not None else 0.0
    avg_bright = np.mean(brightness) if brightness is not None else 1.0

    # 追加当前迭代信息
    with open(outfile, 'a') as f:
        f.write(
            f"{iteration:4d}  {chi2:12.4f}  {entropy:12.6f}  {test_value:12.6e}  "
            f"{avg_blos:10.3f}  {avg_bperp:10.3f}  {avg_bright:10.6f}\n")

    # 如果提供了extra_info，追加详细信息
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
    保存每个观测相位的模型光谱到独立文件
    
    Parameters
    ----------
    results : list of tuples
        每个元组为 (v_grid, specI, specV, specQ, specU, pol_channel) 或
        (v_grid, specI, specV, specQ, specU) 或 (v_grid, specI, specV)
    phase_indices : list of int
        相位索引
    output_dir : str
        输出目录
    fmt : str
        输出格式 ('lsd' 或 'spec')
    prefix : str
        文件名前缀
    """
    from pathlib import Path
    import core.SpecIO as SpecIO

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        phase_idx = phase_indices[i] if i < len(phase_indices) else i

        # 解包结果，支持新格式（包含pol_channel）
        if len(result) >= 6:
            v_grid, specI, specV, specQ, specU, pol_channel = result[:6]
        elif len(result) >= 5:
            v_grid, specI, specV, specQ, specU = result[:5]
            pol_channel = 'V'  # 默认
        elif len(result) >= 3:
            v_grid, specI, specV = result[:3]
            specQ = np.zeros_like(specI)
            specU = np.zeros_like(specI)
            pol_channel = 'V'
        else:
            continue

        # 构建输出文件名
        if fmt.lower() == 'lsd':
            outfile = outdir / f"{prefix}{phase_idx:03d}.lsd"
        else:
            outfile = outdir / f"{prefix}{phase_idx:03d}.spec"

        # 使用SpecIO.write_model_spectrum写入文件，支持pol_channel
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


def save_geomodel_tomog(grid,
                        mag_field=None,
                        brightness=None,
                        output_file="output/outGeoModel.tomog",
                        meta=None,
                        geom=None,
                        integrator=None):
    """
    保存tomography几何模型到.tomog文件
    
    优先使用 VelspaceDiskIntegrator.write_geomodel() 方法以保存完整信息。
    如果未提供 integrator，则使用简化格式。
    
    Parameters
    ----------
    grid : diskGrid
        盘网格对象
    mag_field : MagneticFieldParams, optional
        磁场参数
    brightness : np.ndarray, optional
        亮度分布
    output_file : str
        输出文件路径
    meta : dict, optional
        元信息（目标名、周期等）
    geom : SimpleDiskGeometry, optional
        几何对象，用于构造完整的 integrator
    integrator : VelspaceDiskIntegrator, optional
        积分器对象，包含完整的物理信息
    """
    from pathlib import Path

    outpath = Path(output_file)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # 如果提供了 integrator，使用其 write_geomodel 方法
    if integrator is not None:
        # 更新 integrator.geom 中的磁场参数
        if mag_field is not None:
            integrator.geom.B_los = mag_field.Blos
            integrator.geom.B_perp = mag_field.Bperp
            integrator.geom.chi = mag_field.chi
        integrator.write_geomodel(output_file, meta=meta)
        return

    # 否则使用简化格式（向后兼容）
    with open(output_file, 'w') as f:
        # 写入表头
        f.write("# pyZeeTom Tomography Model (simplified format)\n")
        f.write(
            "# Format: r(R*)  phi(rad)  Blos(G)  Bperp(G)  chi(rad)  brightness\n"
        )

        # 写入元信息
        if meta is not None:
            f.write("#\n# Metadata:\n")
            for key, val in meta.items():
                f.write(f"#   {key}: {val}\n")

        # 如果有 geom，写入几何参数
        if geom is not None:
            f.write("#\n# Geometry:\n")
            if hasattr(geom, 'inclination_rad'):
                f.write(
                    f"#   inclination_deg: {np.rad2deg(geom.inclination_rad):.2f}\n"
                )
            if hasattr(geom, 'phi0'):
                f.write(f"#   phi0: {geom.phi0:.6f}\n")
            if hasattr(geom, 'pOmega'):
                f.write(f"#   pOmega: {geom.pOmega:.6f}\n")
            if hasattr(geom, 'r0'):
                f.write(f"#   r0: {geom.r0:.6f}\n")
            if hasattr(geom, 'period'):
                f.write(f"#   period: {geom.period:.6f}\n")

        # 写入网格信息
        f.write("#\n# Grid:\n")
        f.write(f"#   nr: {grid.nr}\n")
        f.write(f"#   npix: {len(grid.r)}\n")
        if hasattr(grid, 'r_in'):
            f.write(f"#   r_in: {grid.r_in:.6f}\n")
        if hasattr(grid, 'r_out'):
            f.write(f"#   r_out: {grid.r_out:.6f}\n")

        f.write("#" + "-" * 78 + "\n")

        # 写入数据
        npix = len(grid.r)
        for i in range(npix):
            r = grid.r[i]
            phi = grid.phi[i]
            blos = mag_field.Blos[i] if mag_field is not None else 0.0
            bperp = mag_field.Bperp[i] if mag_field is not None else 0.0
            chi = mag_field.chi[i] if mag_field is not None else 0.0
            bright = brightness[i] if brightness is not None else 1.0

            f.write(
                f"{r:10.6f}  {phi:10.6f}  {blos:10.3f}  {bperp:10.3f}  {chi:10.6f}  {bright:10.6f}\n"
            )


#############################################
