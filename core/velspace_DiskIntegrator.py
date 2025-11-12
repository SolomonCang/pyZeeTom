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
      - 外部 line_model: 需实现 BaseLineModel 接口的 compute_local_profile(wl_grid, Ic_weight, Blos, **kwargs)
      - 本类负责:
          1) 生成像素未投影环向速度 v_phi；
          2) 乘以投影得到 v_los；
          3) 将观测速度网格映射为每像素的“局部波长网格”；
          4) 调用 line_model 取得局部 I/V/Q/U，再对像素求和；
          5) 做仪器卷积和连续谱归一。
    注意：
      - 适配新的 linemodel：linemodel 内部用 strength 相对于 1 的偏离决定是吸收还是发射，
        本积分器的 Ic_weight 仅体现连续强度权重（几何×响应），不再携带“发射/吸收”信息。
    """

    def __init__(
            self,
            geom,
            wl0_nm,
            v_grid,
            line_model=None,
            local_sigma_kms=5.0,  # 若 line_model=None 时的默认高斯宽度
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

        # 响应 A_i（>1 发射更强，<1 较弱），此处不将 A 转为"振幅"，直接乘到连续权重中
        # 支持时间相关的响应函数：若 response_func 可接受 phase 参数，则传入
        if response_func is not None:
            # 尝试调用带 phase 参数的响应函数
            import inspect
            sig = inspect.signature(response_func)
            if 'phase' in sig.parameters and self.time_phase is not None:
                # 时间相关响应函数：f(r, phi, phase)
                A = response_func(self.grid.r,
                                  self.grid.phi,
                                  phase=self.time_phase)
            else:
                # 静态响应函数：f(r, phi)
                A = response_func(self.grid.r, self.grid.phi)
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

            # 视向投影
            if los_proj_func is not None:
                proj = los_proj_func(self.grid.r, self.grid.phi, self.geom,
                                     obs_phase)
            elif hasattr(self.geom, "proj_factor"):
                proj = np.asarray(self.geom.proj_factor)
            else:
                inc = getattr(self.geom, "inclination_rad", np.deg2rad(90.0))
                phi0 = getattr(self.geom, "phi0", 0.0)
                proj = np.sin(inc) * np.sin(self.grid.phi - phi0)
            proj = np.asarray(proj)
            if proj.shape == ():
                proj = np.full(self.grid.numPoints, float(proj))
            assert proj.shape[0] == self.grid.numPoints
            v_los = v_phi * proj

        # 映射观测速度网格 -> 每像素局部波长网格
        c = C_KMS
        wl_obs = self.wl0 * (1.0 + self.v / c)  # (Nv,)
        denom = (1.0 + v_los / c)  # (Npix,)
        wl_cells = (wl_obs[:, None] / denom[None, :])  # (Nv, Npix)

        # 连续谱权重 Ic_weight：几何权重（不包含响应）
        Ic_weight = W  # (Npix,)

        # 振幅：响应权重 A 作为谱线振幅（可为发射/吸收）
        amp = A  # (Npix,)

        # 视向磁场（可选）：若 geom 提供 Blos 则使用，否则为 0
        if hasattr(self.geom, "B_los"):
            Blos = np.asarray(self.geom.B_los)
            assert Blos.shape[0] == self.grid.numPoints
        else:
            Blos = np.zeros(self.grid.numPoints, dtype=float)

        # 调用外部谱线模型
        if line_model is None:
            # 默认退化模型：速度空间高斯核（仅 I），用于快速测试通道
            local_sigma_kms = float(local_sigma_kms)
            dv = self.v[:, None] - v_los[None, :]
            # 单位面积的归一高斯（仅为形状演示），与 Ic_weight 相乘
            sigma = max(local_sigma_kms, 1e-30)
            G = np.exp(-0.5 * (dv / sigma)**2) / (np.sqrt(2 * np.pi) * sigma
                                                  )  # (Nv,Npix)
            I_loc = Ic_weight[None, :] * G
            V_loc = np.zeros_like(I_loc)
        else:
            profiles = line_model.compute_local_profile(wl_cells,
                                                        amp,
                                                        Blos=Blos,
                                                        Ic_weight=Ic_weight)
            I_loc = profiles.get("I", None)
            V_loc = profiles.get("V", None)
            if I_loc is None:
                raise ValueError("line_model 返回结果缺少键 'I'")
            if V_loc is None:
                V_loc = np.zeros_like(I_loc)

        # 对像素求和
        I_sum = np.sum(I_loc, axis=1)  # (Nv,)
        V_sum = np.sum(V_loc, axis=1)

        # 仪器卷积（按速度网格）
        if self.inst_fwhm > 0.0:
            I_conv = convolve_gaussian_1d(I_sum, self.dv, self.inst_fwhm)
            V_conv = convolve_gaussian_1d(V_sum, self.dv, self.inst_fwhm)
        else:
            I_conv = I_sum
            V_conv = V_sum

        # 连续谱基线：几何权重总和（与旧版保持一致）
        self.cont = np.sum(W)

        if self.normalize_continuum:
            baseline = 1.0
            if self.cont > 0:
                # I_conv 已经包含基线（每个像素的 I=1+amp*G 被 Ic_weight 加权求和）
                # 归一化后基线应为 1.0，所以 I = (I_conv - cont) / cont + 1.0
                self.I = (I_conv - self.cont) / self.cont + baseline
                self.V = V_conv / self.cont
            else:
                self.I = I_conv
                self.V = V_conv
        else:
            self.I = I_conv
            self.V = V_conv

        # 便于诊断
        self.v_los = v_los
        self.response = A
        self.W = W
        self.v_phi = v_phi if 'v_phi' in locals() else None

    def to_wavelength(self):
        c = C_KMS
        lam = self.wl0 * (1.0 + self.v / c)
        return lam, self.I
