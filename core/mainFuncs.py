import numpy as np
from pathlib import Path
import core.SpecIO as SpecIO
import datetime as dt

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
        fInTomog = open(inParamsName, 'r')
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
        for inLine in fInTomog:
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
                    # New unified model initialization: initTomogFile modelPath
                    # initTomogFile: 0=disabled, 1=enabled (read model from .tomog file)
                    # modelPath: path to .tomog file (e.g., output/geomodel_phase0.tomog)
                    # When enabled, model parameters override input params and generate params_tomog_int.txt
                    parts = inLine.split()
                    self.initTomogFile = int(parts[0])
                    self.initModelPath = parts[1] if len(parts) > 1 else None
                elif (i == 7):
                    self.fitBri = int(inLine.split()[0])
                    self.chiScaleI = float(inLine.split()[1])
                    self.brightEntScale = float(inLine.split()[2])
                elif (i == 8):
                    self.fEntropyBright = int(inLine.split()[0])
                    self.defaultBright = float(inLine.split()[1])
                    self.maximumBright = float(inLine.split()[2])
                elif (i == 9):
                    # Line 9 deprecated: initBrightFromFile was merged into initTomogFile
                    # This line is now ignored; keep for backward compatibility
                    pass
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

    def load_initial_model_from_tomog(self, verbose=1):
        """
        从 .tomog 文件加载初始模型参数（磁场、亮度等）
        
        如果 initTomogFile=1 且 initModelPath 有效，则：
        1. 读取 .tomog 文件获取模型数据和元信息
        2. 提取几何参数（inclination, pOmega, r0, period 等）
        3. 如果 .tomog 中的参数与当前参数不同，生成 params_tomog_int.txt（内部参数文件）
        4. 返回 (geom_loaded, meta_loaded, model_data)；否则返回 (None, None, None)
        
        Returns
        -------
        geom_loaded : SimpleNamespace or None
            加载的几何对象（包含 grid、B_los、B_perp、chi 等）
        meta_loaded : dict or None
            .tomog 文件中的元信息
        model_data : dict or None
            原始模型数据表（r, phi, Blos, Bperp, chi, brightness 等）
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

        # 调用 VelspaceDiskIntegrator.read_geomodel 读取模型
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

            # 检查是否需要生成 params_tomog_int.txt
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
        检查 .tomog 模型中的参数是否与当前参数文件中的参数冲突。
        如果存在冲突，生成 params_tomog_int.txt 文件，记录 .tomog 中的权威参数。
        
        Parameters
        ----------
        geom : SimpleNamespace
            从 .tomog 读取的几何对象
        meta : dict
            从 .tomog 读取的元信息
        verbose : int
            输出详细度
        """

        # 需要对比的参数字典（.tomog 中的键 → 当前 readParamsTomog 的属性）
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
                # 对比（允许小的数值误差）
                if abs(tomog_val - current_val) > 1e-6:
                    conflicts[attr_name] = {
                        "input_file": current_val,
                        "tomog_model": tomog_val
                    }

        if not conflicts:
            if verbose:
                print("[Model] No parameter conflicts detected.")
            return

        # 存在冲突，生成 params_tomog_int.txt
        if verbose:
            print("[Model] Parameter conflicts detected:")
            for attr_name, vals in conflicts.items():
                print(
                    f"  {attr_name}: input={vals['input_file']}, tomog={vals['tomog_model']}"
                )
            print(
                "[Model] Generating params_tomog_int.txt with .tomog parameters as authority..."
            )

        # 生成内部参数文件
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
        根据已有的参数信息写入新的参数文件。
        
        Parameters
        ----------
        outfile : str
            输出文件路径（如 output/params_tomog_new.txt）
        verbose : int
            输出详细度（0=无输出，1=基本信息，2=详细信息）
            
        Returns
        -------
        bool
            写入成功返回 True，否则返回 False
            
        Notes
        -----
        写入的格式与原参数文件格式完全相同，包含所有14+行的参数和观测数据。
        此方法用于：
        1. 修改后保存参数（如修改了 inclination, vsini 等）
        2. 从其他来源（如 .tomog 文件）加载参数后，生成新的标准参数文件
        3. 参数验证和文档化（输出参数文件便于审核）
        """

        outpath = Path(outfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(outfile, 'w') as f:
                # 文件头注释
                f.write("# pyZeeTom parameter file (auto-generated)\n")
                f.write(f"# Generated: {dt.datetime.now().isoformat()}\n")
                f.write(
                    "# Lines are positional. Blank lines and lines starting with # are ignored.\n"
                )
                f.write(
                    "# Units: angles in degrees, velocities km/s, dates in HJD (Heliocentric Julian Date).\n"
                )
                f.write("\n")

                # 行0: inclination vsini period pOmega
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

                # 行1: mass radius [Vmax] [r_out] [enable_occultation]
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

                # 行2: nRingsStellarGrid
                f.write("#2 nRingsStellarGrid\n")
                f.write(f"{self.nRingsStellarGrid}\n")
                f.write("\n")

                # 行3: targetForm targetValue numIterations
                f.write(
                    "#3 targetForm  targetValue  numIterations   (C=chi^2 target, E=entropy target)\n"
                )
                f.write(
                    f"{self.targetForm}  {self.targetValue:.4f}  {self.numIterations}\n"
                )
                f.write("\n")

                # 行4: test_aim
                f.write(
                    "#4 test_aim (convergence threshold for Test statistic)\n")
                f.write(f"{self.test_aim:.2e}\n")
                f.write("\n")

                # 行5: lineAmpConst k_QU enableV enableQU
                f.write("#5 lineAmpConst  k_QU  enableV  enableQU\n")
                f.write(
                    f"{self.lineAmpConst:.1f}  {self.lineKQU:.1f}  {self.lineEnableV}  {self.lineEnableQU}\n"
                )
                f.write("\n")

                # 行6: initTomogFile initModelPath
                f.write("#6 initTomogFile  initModelPath\n")
                init_tomog = getattr(self, 'initTomogFile', 0)
                init_path = getattr(self, 'initModelPath', '')
                f.write(f"{init_tomog}  {init_path}\n")
                f.write("\n")

                # 行7: fitBri chiScaleI brightEntScale
                f.write("#7 fitBri  chiScaleI  brightEntScale\n")
                f.write(
                    f"{self.fitBri}  {self.chiScaleI:.1f}  {self.brightEntScale:.1f}\n"
                )
                f.write("\n")

                # 行8: fEntropyBright defaultBright maximumBright
                f.write("#8 fEntropyBright  defaultBright  maximumBright\n")
                f.write(
                    f"{self.fEntropyBright}  {self.defaultBright:.1f}  {self.maximumBright:.1f}\n"
                )
                f.write("\n")

                # 行9: 已弃用（占位符）
                f.write(
                    "#9 (deprecated - line ignored for backward compatibility)\n"
                )
                f.write("0\n")
                f.write("\n")

                # 行10: estimateStrenght
                f.write(
                    "#10 estimateStrenght (legacy line strength fitting flag)\n"
                )
                f.write(f"{self.estimateStrenght}\n")
                f.write("\n")

                # 行11: spectralResolution lineParamFile
                f.write("#11 spectralResolution  lineParamFile\n")
                f.write(
                    f"{self.spectralResolution:.0f}  {self.lineParamFile}\n")
                f.write("\n")

                # 行12: velStart velEnd obsFileType specType
                f.write("#12 velStart  velEnd  obsFileType  specType\n")
                spec_type_str = getattr(self, 'specType', 'auto')
                f.write(
                    f"{self.velStart:.1f}  {self.velEnd:.1f}  {self.obsFileType}  specType={spec_type_str}\n"
                )
                f.write("\n")

                # 行13: jDateRef
                f.write(
                    "#13 jDateRef  (HJD0, reference epoch for phase calculation)\n"
                )
                f.write(f"{self.jDateRef:.4f}\n")
                f.write("\n")

                # 行14+: 观测数据
                f.write(
                    "#14+ observation entries: filename  HJD  velR  [polchannel]\n"
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


def save_model_spectra_to_outModelSpec(par,
                                       results,
                                       obsSet,
                                       output_dir="output/outModelSpec",
                                       verbose=1):
    """
    将模型光谱保存到 output/outModelSpec 目录，按观测文件格式组织。

    根据观测文件的格式（spec/lsd）、波长范围、速度范围和偏振通道信息，
    为每个观测生成对应格式的模型光谱文件。

    Parameters
    ----------
    par : readParamsTomog
        参数对象，包含 fnames, jDates, velRs, polChannels, phases 等
    results : list of tuples
        每个元组为 (v_grid, specI, specV, specQ, specU, pol_channel)
        或 (v_grid, specI, specV, specQ, specU)
    obsSet : list of ObservationProfile
        观测数据对象列表，用于获取格式、波长范围等信息
    output_dir : str
        输出目录（默认：output/outModelSpec）
    verbose : int
        详细程度

    Returns
    -------
    list of str
        生成的文件路径列表
    """

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_files = []

    if verbose:
        print(
            f"\n[save_model_spectra_to_outModelSpec] 保存 {len(results)} 个模型光谱..."
        )

    for i, result in enumerate(results):
        if i >= len(obsSet):
            if verbose:
                print(
                    f"  警告：result 数量({len(results)}) > obsSet 数量({len(obsSet)})"
                )
            break

        # 获取观测信息
        obs = obsSet[i]

        # 提取观测参数
        hjd = par.jDates[i] if i < len(par.jDates) else 0.0
        vel_r = par.velRs[i] if i < len(par.velRs) else 0.0
        pol_channel = str(par.polChannels[i]).upper() if i < len(
            par.polChannels) else 'V'
        phase = par.phases[i] if hasattr(par, 'phases') and i < len(
            par.phases) else (i / len(results) if len(results) > 0 else 0.0)

        # 解包结果
        if len(result) >= 6:
            v_grid, specI, specV, specQ, specU, pol_ch_from_result = result[:6]
            # 使用结果中的 pol_channel（如果有的话）
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
                print(f"  警告：结果 {i} 格式不正确，跳过")
            continue

        # 确定输出格式（从观测对象推断）
        obs_format = obs.profile_type.lower() if hasattr(
            obs, 'profile_type') else 'spec'
        if obs_format == 'velocity' or obs_format == 'lsd':
            fmt = 'lsd'
            ext = '.lsd'
        else:
            fmt = 'spec'
            ext = '.spec'

        # 根据 pol_channel 确定 file_type_hint
        if pol_channel == 'I':
            file_type_hint = 'lsd_i' if fmt == 'lsd' else 'spec_i'
        else:
            file_type_hint = 'lsd_pol' if fmt == 'lsd' else 'spec_pol'

        # 构建输出文件名
        # 格式：phase_XXXX_HJDpYYY_VRsZZZ_CH.ext
        # 例如：phase_0000_HJDp0p200_VRs0p00_V.lsd
        hjd_str = f"{hjd:.3f}".replace('.', 'p')
        vel_r_sign = 'p' if vel_r >= 0 else 'm'
        vel_r_abs = abs(vel_r)
        vel_r_str = f"{vel_r_sign}{vel_r_abs:.2f}".replace('.', 'p')

        outfile_name = f"phase_{i:04d}_HJD{hjd_str}_VR{vel_r_str}_{pol_channel}{ext}"
        outfile = outdir / outfile_name

        # 使用 SpecIO 保存模型光谱
        try:
            header = {
                "phase_index": str(i),
                "HJD": f"{hjd:.6f}",
                "velR": f"{vel_r:.2f}",
                "pol_channel": str(pol_channel),
                "phase": f"{phase:.4f}",
            }

            # 调用 SpecIO.write_model_spectrum，确保格式一致
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
                print(f"  警告：保存文件 {outfile.name} 失败：{e}")

    if verbose:
        print(
            f"[save_model_spectra_to_outModelSpec] 完成！生成 {len(output_files)} 个文件到 {outdir}"
        )
        print("  文件格式：phase_XXXX_HJDYYY_VRZZ_CH.ext")
        print("    XXXX = 观测索引（0000-9999）")
        print("    YYY = Heliocentric Julian Date (p=小数点)")
        print("    ZZ = 径向速度修正（km/s）")
        print("    CH = 偏振通道（I/V/Q/U）")
        print("    ext = 格式后缀（.spec 或 .lsd）")

    return output_files


#############################################
