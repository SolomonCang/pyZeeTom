# velspace_DiskIntegrator.py
# 盘模型的速度空间积分器（支持外部谱线模型 linemodel）
# - 外侧：角速度幂律 Ω(r)=Ω0*(r/r0)^p，线速度 vφ=r*Ω(r)
# - 内侧：自适应环数的“Ω 序列”同步减速（Ω 从 Ω0 到 0，再用 vφ=r*Ω）
# - 谱线：通过注入的 line_model(BaseLineModel) 计算局部 I/V/Q/U，再在观测网格上求和

import numpy as np

C_KMS = 2.99792458e5  # km/s


def convolve_gaussian_1d(y, dv, fwhm):
    if fwhm <= 0.0:
        return y
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = int(np.ceil(3.0 * sigma / np.maximum(dv, 1e-30)))
    if half < 1:
        return y
    x = np.arange(-half, half + 1) * dv
    ker = np.exp(-0.5 * (x / sigma)**2)
    ker /= np.sum(ker)
    padL = np.full(half, y[0])
    padR = np.full(half, y[-1])
    tmp = np.concatenate([padL, y, padR])
    conv = np.convolve(tmp, ker, mode='valid')
    return conv


def slowdown_profile_sequence(n, kind="cosine"):
    n = int(n)
    if n <= 1:
        return np.array([0.0], dtype=float)
    k = np.arange(n, dtype=float)
    t = k / (n - 1.0)
    if kind == "linear":
        s = 1.0 - t
    elif kind == "quadratic":
        s = (1.0 - t)**2
    elif kind == "cubic":
        s = (1.0 - t)**3
    else:  # cosine
        s = 0.5 * (1.0 + np.cos(np.pi * t))
    return s


def disk_velocity_adaptive_inner_omega_sequence(grid,
                                                v0_r0,
                                                p,
                                                r0,
                                                profile="cosine",
                                                blend_edge=True):
    r_centers = grid.r_centers
    nr = len(r_centers)
    if nr == 0:
        return np.zeros(0, dtype=float)

    rr0 = float(max(r0, 1e-30))
    Omega0 = float(v0_r0) / rr0

    diff = rr0 - r_centers
    mask_in = diff > 0
    if np.any(mask_in):
        start_ring = np.where(mask_in)[0][-1]
    else:
        start_ring = int(np.argmin(np.abs(r_centers - rr0)))

    N_inner = start_ring + 1
    seq = slowdown_profile_sequence(N_inner, kind=profile)  # s[0]=1, s[-1]=0

    v_ring = np.zeros(nr, dtype=float)
    for ir in range(nr):
        r_c = float(r_centers[ir])
        if r_c >= rr0:
            x = r_c / rr0
            Omega = Omega0 * (x**p)
            v_ring[ir] = r_c * Omega
        else:
            offset = start_ring - ir
            if 0 <= offset < N_inner:
                Omega = Omega0 * seq[offset]
                v_ring[ir] = r_c * Omega
            else:
                v_ring[ir] = 0.0

    if blend_edge and 0 <= start_ring < nr:
        if hasattr(grid, "dr"):
            dr_ring = float(grid.dr[start_ring])
        else:
            dr_ring = float(grid.r_edges[start_ring + 1] -
                            grid.r_edges[start_ring])
        r_center = float(r_centers[start_ring])
        rel = (rr0 - r_center) / max(dr_ring, 1e-30)
        rel = float(np.clip(rel, -0.5, 0.5))
        w_outer = float(np.clip(0.5 + rel, 0.0, 1.0))

        x = (r_center / rr0) if rr0 > 0 else 0.0
        Omega_outer = Omega0 * (x**p) if r_center >= rr0 else Omega0
        v_start_outer = r_center * Omega_outer

        Omega_seq = Omega0 * seq[0]
        v_start_seq = r_center * Omega_seq

        v_ring[start_ring] = w_outer * v_start_outer + (1.0 -
                                                        w_outer) * v_start_seq

        next_inner = start_ring - 1
        if next_inner >= 0 and N_inner >= 2:
            w = 0.2
            v_ring[next_inner] = (
                1 - w) * v_ring[next_inner] + w * v_ring[start_ring]

    v_phi = v_ring[grid.ring_id]
    return v_phi


def disk_velocity_continuous_omega(r,
                                   v0_r0,
                                   p,
                                   r0,
                                   enable_inner_slowdown=True,
                                   inner_mode="poly",
                                   inner_alpha=0.6,
                                   inner_beta=2.0):
    r = np.asarray(r, dtype=float)
    rr0 = float(max(r0, 1e-30))
    Omega0 = float(v0_r0) / rr0
    x = r / rr0

    if not enable_inner_slowdown:
        return r * Omega0 * np.power(x, p)

    xc = np.clip(x, 0.0, 1.0)
    if inner_mode == "poly":
        f = 1.0 - inner_alpha * (1.0 - np.power(xc, inner_beta))
    elif inner_mode == "lat":
        f = xc * (1.0 - inner_alpha * (1.0 - xc * xc))
    else:
        raise ValueError("inner_mode must be 'poly' or 'lat'")

    v_out = r * Omega0 * np.power(np.maximum(x, 1.0), p)
    v_in = r * Omega0 * f
    v = np.where(x >= 1.0, v_out, v_in)
    return v


class VelspaceDiskIntegrator:
    """
    在观测速度栅格上积分的盘模型，谱线来自外部 line_model:
      - 外部 line_model: 必须提供，需实现 BaseLineModel 接口的 compute_local_profile(wl_grid, amp, Blos, **kwargs)
      - 本类负责:
          1) 生成像素未投影环向速度 v_phi；
          2) 乘以投影得到 v_los；
          3) 将观测速度网格映射为每像素的"局部波长网格"；
          4) 调用 line_model 取得局部 I/V/Q/U，再对像素求和；
          5) 做仪器卷积和连续谱归一。
    注意：
      - line_model 参数为必需参数，需传入 core/local_linemodel_basic.py 中的模型实例（如 GaussianZeemanWeakLineModel）
      - linemodel 内部用 amp（相对于 1 的偏离）决定是吸收还是发射
      - 本积分器的 Ic_weight 仅体现连续强度权重（几何×响应），不再携带"发射/吸收"信息
    """

    def __init__(
            self,
            geom,
            wl0_nm,
            v_grid,
            line_model,  # 必需参数，必须提供谱线模型
            line_area=1.0,
            response_func=None,
            response_map=None,
            inst_fwhm_kms=0.0,
            normalize_continuum=True,
            # 速度场参数（外侧角速度幂律）
            use_geom_vlos_if_available=True,
            disk_v0_kms=200.0,
            disk_power_index=-0.5,
            disk_r0=1.0,
            # 内侧减速模式
            inner_slowdown_mode="adaptive",
            inner_profile="cosine",
            inner_edge_blend=True,
            # 连续模式参数（若选择 continuous）
            inner_mode="poly",
            inner_alpha=0.6,
            inner_beta=2.0,
            # 视向投影
            los_proj_func=None,
            obs_phase=None,
            # 时间演化支持（新增）
            time_phase=None):
        self.geom = geom
        self.grid = geom.grid
        self.wl0 = float(wl0_nm)
        self.v = np.asarray(v_grid)
        self.dv = np.mean(np.diff(self.v)) if self.v.size > 1 else 1.0
        self.inst_fwhm = float(inst_fwhm_kms)
        self.normalize_continuum = bool(normalize_continuum)
        self.line_area = float(line_area)

        # 存储相位信息（用于时间演化支持）
        self.time_phase = time_phase if time_phase is not None else obs_phase

        # 计算时间演化后的方位角（用于响应函数和投影）
        phi_evolved = self.grid.phi
        if self.time_phase is not None and hasattr(self.grid,
                                                   'rotate_to_phase'):
            # 从几何对象获取差速转动参数
            pOmega = getattr(geom, 'pOmega', disk_power_index)
            r0_rot = getattr(geom, 'r0', disk_r0)
            period = getattr(geom, 'period', 1.0)
            # 计算演化后的方位角
            phi_evolved = self.grid.rotate_to_phase(self.time_phase,
                                                    pOmega=pOmega,
                                                    r0=r0_rot,
                                                    period=period)

        # 响应 A_i（>1 发射更强，<1 较弱），此处不将 A 转为"振幅"，直接乘到连续权重中
        # 支持时间相关的响应函数：若 response_func 可接受 phase 参数，则传入
        if response_func is not None:
            # 尝试调用带 phase 参数的响应函数
            import inspect
            sig = inspect.signature(response_func)
            if 'phase' in sig.parameters and self.time_phase is not None:
                # 时间相关响应函数：f(r, phi, phase)，使用演化后的 phi
                A = response_func(self.grid.r,
                                  phi_evolved,
                                  phase=self.time_phase)
            else:
                # 静态响应函数：f(r, phi)，使用演化后的 phi
                A = response_func(self.grid.r, phi_evolved)
        elif response_map is not None:
            A = np.asarray(response_map)
            assert A.shape[0] == self.grid.numPoints
        else:
            A = np.ones(self.grid.numPoints, dtype=float)

        # 投影/几何面积权重
        W = np.asarray(self.geom.area_proj)
        assert W.shape[0] == self.grid.numPoints

        # 未投影线速度 v_phi 或使用几何自带 v_los
        if use_geom_vlos_if_available and hasattr(self.geom, "v_los"):
            v_los = np.asarray(self.geom.v_los)
            assert v_los.shape[0] == self.grid.numPoints
            v_phi = None
        else:
            v0_r0 = float(disk_v0_kms)
            p = float(disk_power_index)
            r0 = float(disk_r0)
            # 保存到对象以便导出
            self._disk_v0_kms = v0_r0
            self._disk_power_index = p
            self._disk_r0 = r0
            mode = str(inner_slowdown_mode).lower()
            if mode == "adaptive":
                v_phi = disk_velocity_adaptive_inner_omega_sequence(
                    self.grid,
                    v0_r0=v0_r0,
                    p=p,
                    r0=r0,
                    profile=str(inner_profile).lower(),
                    blend_edge=bool(inner_edge_blend))
            elif mode == "continuous":
                v_phi = disk_velocity_continuous_omega(
                    self.grid.r,
                    v0_r0=v0_r0,
                    p=p,
                    r0=r0,
                    enable_inner_slowdown=True,
                    inner_mode=str(inner_mode),
                    inner_alpha=float(inner_alpha),
                    inner_beta=float(inner_beta))
            else:
                raise ValueError(
                    "inner_slowdown_mode must be 'adaptive' or 'continuous'")

            # 视向投影（使用时间演化后的方位角 phi_evolved）
            if los_proj_func is not None:
                proj = los_proj_func(self.grid.r, phi_evolved, self.geom,
                                     obs_phase)
            elif hasattr(self.geom, "proj_factor"):
                proj = np.asarray(self.geom.proj_factor)
            else:
                inc = getattr(self.geom, "inclination_rad", np.deg2rad(90.0))
                phi0 = getattr(self.geom, "phi0", 0.0)
                proj = np.sin(inc) * np.sin(phi_evolved - phi0)
            proj = np.asarray(proj)
            if proj.shape == ():
                proj = np.full(self.grid.numPoints, float(proj))
            assert proj.shape[0] == self.grid.numPoints
            v_los = v_phi * proj

        # 检查是否需要计算恒星遮挡
        occultation_mask = np.zeros(self.grid.numPoints, dtype=bool)
        if hasattr(self.geom, 'enable_stellar_occultation'
                   ) and self.geom.enable_stellar_occultation:
            # 从几何对象获取必要参数
            # 观察者方向固定（通常为phi_obs=0）
            phi_obs = getattr(self.geom, "phi_obs", 0.0)
            inclination_deg = np.rad2deg(
                getattr(self.geom, "inclination_rad", np.deg2rad(60.0)))
            stellar_radius = getattr(self.geom, "stellar_radius", 1.0)
            occultation_mask = self.grid.compute_stellar_occultation_mask(
                phi_obs=phi_obs,
                inclination_deg=inclination_deg,
                stellar_radius=stellar_radius,
                verbose=1)

        # 映射观测速度网格 -> 每像素局部波长网格
        c = C_KMS
        wl_obs = self.wl0 * (1.0 + self.v / c)  # (Nv,)
        denom = (1.0 + v_los / c)  # (Npix,)
        wl_cells = (wl_obs[:, None] / denom[None, :])  # (Nv, Npix)

        # 连续谱权重 Ic_weight：几何权重（不包含响应）
        # 应用遮挡mask：被遮挡的像素权重设为0
        Ic_weight = W.copy()  # (Npix,)
        Ic_weight[occultation_mask] = 0.0

        # 振幅：响应权重 A 作为谱线振幅（可为发射/吸收）
        # 同样应用遮挡mask
        amp = A.copy()  # (Npix,)
        amp[occultation_mask] = 0.0

        # 视向磁场（可选）：若 geom 提供 Blos 则使用，否则为 0
        if hasattr(self.geom, "B_los"):
            Blos = np.asarray(self.geom.B_los)
            assert Blos.shape[0] == self.grid.numPoints
        else:
            Blos = np.zeros(self.grid.numPoints, dtype=float)

        # 横向磁场和方位角（用于Q/U计算）
        if hasattr(self.geom, "B_perp"):
            Bperp = np.asarray(self.geom.B_perp)
            assert Bperp.shape[0] == self.grid.numPoints
        else:
            Bperp = None

        if hasattr(self.geom, "chi"):
            chi = np.asarray(self.geom.chi)
            assert chi.shape[0] == self.grid.numPoints
        else:
            chi = None

        # 调用外部谱线模型（必须提供）
        if line_model is None:
            raise ValueError(
                "line_model 参数必须提供，请传入 BaseLineModel 实例（如 GaussianZeemanWeakLineModel）"
            )

        profiles = line_model.compute_local_profile(wl_cells,
                                                    amp,
                                                    Blos=Blos,
                                                    Bperp=Bperp,
                                                    chi=chi,
                                                    Ic_weight=Ic_weight)
        I_loc = profiles.get("I", None)
        V_loc = profiles.get("V", None)
        Q_loc = profiles.get("Q", None)
        U_loc = profiles.get("U", None)
        if I_loc is None:
            raise ValueError("line_model 返回结果缺少键 'I'")
        if V_loc is None:
            V_loc = np.zeros_like(I_loc)
        if Q_loc is None:
            Q_loc = np.zeros_like(I_loc)
        if U_loc is None:
            U_loc = np.zeros_like(I_loc)

        # 对像素求和
        I_sum = np.sum(I_loc, axis=1)  # (Nv,)
        V_sum = np.sum(V_loc, axis=1)
        Q_sum = np.sum(Q_loc, axis=1)
        U_sum = np.sum(U_loc, axis=1)

        # 仪器卷积（按速度网格）
        if self.inst_fwhm > 0.0:
            I_conv = convolve_gaussian_1d(I_sum, self.dv, self.inst_fwhm)
            V_conv = convolve_gaussian_1d(V_sum, self.dv, self.inst_fwhm)
            Q_conv = convolve_gaussian_1d(Q_sum, self.dv, self.inst_fwhm)
            U_conv = convolve_gaussian_1d(U_sum, self.dv, self.inst_fwhm)
        else:
            I_conv = I_sum
            V_conv = V_sum
            Q_conv = Q_sum
            U_conv = U_sum

        # 连续谱基线：几何权重总和（与旧版保持一致）
        self.cont = np.sum(W)

        if self.normalize_continuum:
            baseline = 1.0
            if self.cont > 0:
                # I_conv 已经包含基线（每个像素的 I=1+amp*G 被 Ic_weight 加权求和）
                # 归一化后基线应为 1.0，所以 I = (I_conv - cont) / cont + 1.0
                self.I = (I_conv - self.cont) / self.cont + baseline
                self.V = V_conv / self.cont
                self.Q = Q_conv / self.cont
                self.U = U_conv / self.cont
            else:
                self.I = I_conv
                self.V = V_conv
                self.Q = Q_conv
                self.U = U_conv
        else:
            self.I = I_conv
            self.V = V_conv
            self.Q = Q_conv
            self.U = U_conv

        # 便于诊断
        self.v_los = v_los
        self.response = A
        self.W = W
        self.v_phi = v_phi if 'v_phi' in locals() else None

    def to_wavelength(self):
        c = C_KMS
        lam = self.wl0 * (1.0 + self.v / c)
        return lam, self.I

    # ------------------------------
    # 模型读/写：geomodel.tomog
    # ------------------------------
    def write_geomodel(self, filepath, meta=None):
        """
        将当前几何模型与节点物理量导出为文本文件 geomodel.tomog。

        文件结构（纯文本，便于审阅与版本控制）：
        - 以 '#' 开头的头部区，键值对形式，包含可重建模型所需的参数
        - 一行列名（# COLUMNS: ...）
        - 每个像素一行数据。

        节点字段至少包含：
        - idx, ring_id, phi_id, r, phi, area, Ic_weight, A(response), Blos
        - 若可用：Bperp, chi
        """
        import datetime as _dt

        g = self.grid
        geom = self.geom

        # 元信息汇总，尽量完整但不过度依赖外部对象
        header = {
            "format":
            "TOMOG_MODEL",
            "version":
            1,
            "created_utc":
            _dt.datetime.utcnow().isoformat() + "Z",
            "wl0_nm":
            float(self.wl0),
            "inst_fwhm_kms":
            float(self.inst_fwhm),
            "normalize_continuum":
            bool(self.normalize_continuum),
            # 速度场参数（用于可再现）
            "disk_v0_kms":
            getattr(self, "_disk_v0_kms", None),
            "disk_power_index":
            getattr(self, "_disk_power_index", None),
            "disk_r0":
            getattr(self, "_disk_r0", None),
            # 差速与几何
            "inclination_deg":
            float(
                np.rad2deg(getattr(geom, "inclination_rad",
                                   np.deg2rad(90.0)))),
            "phi0":
            float(getattr(geom, "phi0", 0.0)),
            "pOmega":
            float(getattr(geom, "pOmega", 0.0)),
            "r0_rot":
            float(getattr(geom, "r0", getattr(self, "_disk_r0", 1.0))),
            "period":
            float(getattr(geom, "period", 1.0)),
            # 网格定义（用于重建）
            "nr":
            int(getattr(g, "nr", len(getattr(g, "r_centers", []))))
        }

        # 可选：目标/观测信息由调用方通过 meta 传入
        if isinstance(meta, dict):
            for k, v in meta.items():
                header[str(k)] = v

        # 磁场分量
        has_Blos = hasattr(geom, "B_los") and geom.B_los is not None
        has_Bperp = hasattr(geom, "B_perp") and geom.B_perp is not None
        has_chi = hasattr(geom, "chi") and geom.chi is not None

        # 响应与几何权重
        A = np.asarray(getattr(self, "response", np.ones(g.numPoints)))
        Ic_weight = np.asarray(getattr(self, "W", np.ones(g.numPoints)))

        # 列定义
        columns = [
            "idx", "ring_id", "phi_id", "r", "phi", "area", "Ic_weight", "A",
            "Blos"
        ]
        if has_Bperp:
            columns += ["Bperp"]
        if has_chi:
            columns += ["chi"]

        # 写文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# TOMOG Geometric Model File\n")
            for k in sorted(header.keys()):
                v = header[k]
                # 处理数组型头字段（例如 r_edges）
                if isinstance(v, (list, tuple, np.ndarray)):
                    arr = np.asarray(v).ravel()
                    vstr = ",".join(f"{x:.12g}" for x in arr)
                    f.write(f"# {k}: [{vstr}]\n")
                else:
                    f.write(f"# {k}: {v}\n")

            # 尝试补充网格边界（若可用）
            if hasattr(g, "r_edges"):
                vstr = ",".join(f"{x:.12g}"
                                for x in np.asarray(g.r_edges).ravel())
                f.write(f"# r_edges: [{vstr}]\n")

            f.write("# COLUMNS: " + ", ".join(columns) + "\n")

            N = g.numPoints
            for i in range(N):
                row = [
                    i,
                    int(g.ring_id[i]) if hasattr(g, "ring_id") else -1,
                    int(getattr(g, "phi_id", np.zeros_like(g.r, int))[i]),
                    float(g.r[i]),
                    float(g.phi[i]),
                    float(g.area[i]),
                    float(Ic_weight[i]),
                    float(A[i]),
                    float(geom.B_los[i]) if has_Blos else 0.0,
                ]
                if has_Bperp:
                    row.append(float(geom.B_perp[i]))
                if has_chi:
                    row.append(float(geom.chi[i]))
                f.write(" ".join(str(x) for x in row) + "\n")

    @staticmethod
    def read_geomodel(filepath):
        """
        读取 geomodel.tomog，返回 (geom_like, meta, table)

        - geom_like: 一个提供 integrator 所需属性的几何对象：
          .grid（包含 r, phi, area, ring_id, phi_id, r_edges/centers/dr 若可用）
          .area_proj, .inclination_rad, .phi0, .pOmega, .r0, .period
          .B_los（可选）, .B_perp（可选）, .chi（可选）
        - meta: 头部键值对（字典）
        - table: 原始数据表（dict of np.ndarray）
        """
        import re as _re

        meta = {}
        rows = []
        columns = None
        r_edges = None

        with open(filepath, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("#"):
                    # 解析头部键值
                    m = _re.match(r"^#\s*([^:]+):\s*(.*)$", ln.strip())
                    if m:
                        k = m.group(1).strip()
                        v = m.group(2).strip()
                        if k == "COLUMNS":
                            columns = [s.strip() for s in v.split(",")]
                        elif k == "r_edges":
                            # 解析数组
                            vs = v.strip()
                            vs = vs.strip("[]")
                            if vs:
                                r_edges = np.array(
                                    [float(x) for x in vs.split(",")])
                        else:
                            # 尝试将数字转为 float/int
                            vv = v
                            if vv.startswith("[") and vv.endswith("]"):
                                try:
                                    arr = [
                                        float(x)
                                        for x in vv.strip("[]").split(",")
                                        if x.strip()
                                    ]
                                    meta[k] = np.array(arr)
                                except Exception:
                                    meta[k] = vv
                            else:
                                try:
                                    if _re.match(r"^-?\d+$", vv):
                                        meta[k] = int(vv)
                                    else:
                                        meta[k] = float(vv)
                                except Exception:
                                    meta[k] = vv
                    continue
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                rows.append(parts)

        if columns is None:
            # 使用默认列顺序回退
            columns = [
                "idx", "ring_id", "phi_id", "r", "phi", "area", "Ic_weight",
                "A", "Blos", "Bperp", "chi"
            ]

        # 整理为数组表
        data = list(zip(*rows)) if rows else []
        table = {}
        for i, name in enumerate(columns):
            if i < len(data):
                try:
                    arr = np.array([float(x) for x in data[i]], dtype=float)
                except Exception:
                    # 混合类型，保持字符串
                    arr = np.array(data[i])
                table[name] = arr

        # 构造 grid-like 对象
        from types import SimpleNamespace as _NS

        grid = _NS()
        grid.r = table.get("r", np.array([]))
        grid.phi = table.get("phi", np.array([]))
        grid.area = table.get("area", np.array([]))
        base_r = grid.r if isinstance(grid.r, np.ndarray) else np.array([])
        grid.ring_id = table.get("ring_id", np.zeros_like(base_r,
                                                          int)).astype(int)
        grid.phi_id = table.get("phi_id", np.zeros_like(base_r,
                                                        int)).astype(int)
        grid.numPoints = base_r.shape[0]

        # 尝试恢复 r_edges / r_centers / dr
        if r_edges is None and "nr" in meta and grid.numPoints > 0:
            # 简单回退：用每 ring_id 的 r 中值 + 边界补点
            nr = int(meta["nr"]) if isinstance(meta["nr"],
                                               (int, float)) else int(
                                                   meta["nr"])  # type: ignore
            r_med = []
            for rid in range(nr):
                mask = (grid.ring_id == rid)
                if np.any(mask):
                    r_med.append(np.median(grid.r[mask]))
            if r_med:
                r_med = np.array(r_med)
                dr = np.diff(r_med)
                dr0 = dr[0] if dr.size > 0 else (
                    r_med[0] if r_med.size > 0 else 1.0)
                r_edges = np.concatenate(
                    [[r_med[0] - 0.5 * dr0], 0.5 * (r_med[:-1] + r_med[1:]),
                     [r_med[-1] + 0.5 * (dr[-1] if dr.size > 0 else dr0)]])

        if r_edges is None:
            # 最后回退：用唯一半径排序构造
            unique_r = np.unique(grid.r)
            if unique_r.size >= 2:
                mid = 0.5 * (unique_r[:-1] + unique_r[1:])
                dr0 = unique_r[1] - unique_r[0]
                r_edges = np.concatenate(
                    [[unique_r[0] - 0.5 * dr0], mid,
                     [unique_r[-1] + 0.5 * (unique_r[-1] - unique_r[-2])]])
            else:
                r_edges = np.array(
                    [0.0, unique_r[0] if unique_r.size else 1.0])

        grid.r_edges = r_edges
        grid.r_centers = 0.5 * (
            r_edges[:-1] + r_edges[1:]) if r_edges.size >= 2 else np.array([])
        grid.dr = (r_edges[1:] -
                   r_edges[:-1]) if r_edges.size >= 2 else np.array([])

        # 构造 geom-like 对象
        geom = _NS()
        geom.grid = grid
        geom.area_proj = grid.area
        geom.inclination_rad = np.deg2rad(
            float(meta.get("inclination_deg", 90.0)))
        geom.phi0 = float(meta.get("phi0", 0.0))
        geom.pOmega = float(meta.get("pOmega", 0.0))
        geom.r0 = float(meta.get("r0_rot", meta.get("disk_r0", 1.0)))
        geom.period = float(meta.get("period", 1.0))

        # 磁场
        if "Blos" in table:
            geom.B_los = table["Blos"].astype(float)
        if "Bperp" in table:
            geom.B_perp = table["Bperp"].astype(float)
        if "chi" in table:
            geom.chi = table["chi"].astype(float)

        return geom, meta, table
