# linemodel.py  (ME + Unno–Rachkovsky; 输入为建议格式，支持发射分量附加 Voigt 峰)
import numpy as np

_C_KMS = 2.99792458e5  # km/s
_ZEEMAN_CONST = 4.66864e-12  # ΔλB(nm) = const * g * λ0^2 * B(G)


class LineData:
    """
    单行谱线参数（仅使用首行非注释）：
      wl0  g  eta0  beta  widthGauss(km/s)  widthLorentz_ratio  [fI]  [fV]  [emitAmp]  [emitWidthScale]
    说明：
      - eta0: 线心吸收/连续比（>0）
      - beta: 源函数斜率，S(τ)=S0(1+βτ)
      - widthGauss: sqrt(2)*σ_v，单位 km/s
      - widthLorentz_ratio: Γ/(sqrt(2)*σ_v) = gamma/widthGauss
      - fI, fV: 填充因子，默认 1，且 fI ∈ [0,1]（保留物理含义）
      - emitAmp: 附加发射峰幅度 A_e（≥0，默认 0，仅建议在“发射分量”里给）
      - emitWidthScale: 发射峰宽度缩放 s_e（>0，默认 1.0）
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
                        "ME 建议格式至少需 6 列: wl0 g eta0 beta widthGauss widthLorentz_ratio [fI] [fV] [emitAmp] [emitWidthScale]"
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

                # 简单边界检查
                if self.widthGauss <= 0:
                    raise ValueError("widthGauss 必须为正 (km/s)。")
                if self.widthLorentz < 0:
                    raise ValueError("widthLorentz_ratio 不可为负。")
                if not (0.0 <= self.fI <= 1.0):
                    raise ValueError("fI 应在 [0,1] 内（保持填充因子物理意义）。")
                if not (0.0 <= self.fV <= 1.0):
                    raise ValueError("fV 应在 [0,1] 内。")
                if self.emitAmp < 0.0:
                    raise ValueError("emitAmp 应 ≥ 0（仅在发射分量中使用正值）。")
                if self.emitWidthScale <= 0.0:
                    raise ValueError("emitWidthScale 应 > 0。")

                self.numLines = 1
                break

        if self.numLines == 0:
            raise ValueError("未读取到有效谱线参数。")


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
    ME + UR 全 Stokes 解，双参数集（吸收/发射）线性加权：
      S_total = w_abs * S_abs + w_emit * S_emit

    发射分量新特性：
      在该分量的 I 上叠加附加源 E(λ) = emitAmp * H_e(λ)，其中 H_e 是以
      emitWidthScale 倍宽度计算的 Voigt 轮廓（仅取实部 H）。

    使用方式：
      - 吸收分量文件不需提供 emitAmp/emitWidthScale（默认 0,1）。
      - 发射分量文件可提供 emitAmp>0（建议 0.1~1.5），emitWidthScale 可调宽度。

    """
    def __init__(self, line_absorb: LineData, line_emit: LineData):
        # 两组谱线必须同一条线（wl0, g 一致）
        if (abs(line_absorb.wl0 - line_emit.wl0) >
                1e-9) or (abs(line_absorb.g - line_emit.g) > 1e-9):
            raise ValueError("吸收与发射参数的 wl0 与 g 必须一致")
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
        - wl_grid: (Nλ,) 或 (Nλ,Npix)
        - Ic_weight: (Npix,) 面元权重（肢暗、面积等）
        - weight_absorb, weight_emit: 可为标量或 (Npix,)；默认 1/0
        - 磁场：优先 Bmod, Btheta, Bchi；否则尝试 Blos 仅给纵向
        - mu: 视向余弦 (Npix,)；未给则默认为 1
        - enable_IQVU: 元组开关
        返回 dict: 'I','Q','U','V' -> (Nλ,Npix)
        """
        enable_I, enable_Q, enable_U, enable_V = enable_IQVU

        wl_grid = np.asarray(wl_grid, dtype=float)
        if wl_grid.ndim == 1:
            wl_grid = wl_grid[:, None]
        Nlam, Npix = wl_grid.shape

        w_area = np.asarray(Ic_weight, dtype=float).reshape(1, Npix)

        # 几何与磁场
        if (Bmod is not None) and (Btheta is not None):
            Bmod = _as_col(Bmod, Npix)
            Btheta = _as_col(Btheta, Npix)
            if Bchi is None:
                Bchi = np.zeros((1, Npix), dtype=float)
            else:
                Bchi = _as_col(Bchi, Npix)
        else:
            # 回退到仅给 Blos 的情况：认为纯纵向场（θ=0 或 π），Q/U≈0
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

        # 组件权重
        w_abs = _as_col(weight_absorb, Npix)
        w_em = _as_col(weight_emit, Npix)

        # 分量求解
        I_abs, Q_abs, U_abs, V_abs = self._solve_me_ur_component(
            wl_grid, self.ld_abs, Bmod, Btheta, Bchi, mu, enable_IQVU)

        I_em, Q_em, U_em, V_em = self._solve_me_ur_component(
            wl_grid, self.ld_emit, Bmod, Btheta, Bchi, mu, enable_IQVU)

        # 在发射分量的 I 上叠加附加源 E(λ) = emitAmp * H_e(λ)
        if enable_I and (self.ld_emit.emitAmp > 0.0):
            He = self._voigt_profile_for_emission(wl_grid, self.ld_emit)
            I_em = I_em + self.ld_emit.emitAmp * He

        # 线性叠加两分量
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
        基于发射分量参数构造附加源的 Voigt 实部 H_e(λ)：
          - 中心 wl0 与谱线一致
          - 宽度: Δλ_D_e = emitWidthScale * Δλ_D(line)
          - 阻尼: a_e = emitWidthScale * a(line)    （可选的等比缩放，简单且稳定）
          - 仅取 Voigt 的实部 H_e（非极化附加源，不改动 Q,U,V）
          - 归一：与 main 解一致：H = Re(W)/sqrt(pi)
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
        单组件 ME+UR 全解析解。返回 I,Q,U,V（均为 (Nλ,Npix)）。
        Ic 归一为 1；外部用 w_area 再加权。
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

        # 磁分裂位移 Δλ_B (nm)
        dlamB = _ZEEMAN_CONST * g_eff * (wl0**2) * Bmod / np.maximum(fV, 1e-12)

        # 高斯宽度 Δλ_D (nm) = (vG/c)*λ0
        dlamD = (vG / _C_KMS) * wl0

        # 角度因子
        theta = Btheta
        chi = Bchi
        cos1 = np.cos(theta)
        sin1 = np.sin(theta)
        sin2 = sin1**2
        cos2chi = np.cos(2.0 * chi)
        sin2chi = np.sin(2.0 * chi)

        # Π, σ± 的归一频移
        u_pi = (wl_grid - wl0) / dlamD
        u_r = (wl_grid - wl0 - dlamB) / dlamD
        u_l = (wl_grid - wl0 + dlamB) / dlamD

        # 计算 Faddeeva
        Wpi = _voigt_faraday_humlicek(u_pi, a)
        Wr = _voigt_faraday_humlicek(u_r, a)
        Wl = _voigt_faraday_humlicek(u_l, a)

        # 归一：Voigt H = Re(W)/sqrt(pi), Faraday F = Im(W)/sqrt(pi)
        sqpi = np.sqrt(np.pi)
        H_pi = Wpi.real / sqpi
        F_pi = Wpi.imag / sqpi
        H_r = Wr.real / sqpi
        F_r = Wr.imag / sqpi
        H_l = Wl.real / sqpi
        F_l = Wl.imag / sqpi

        # 传播与色散系数（按 Unno-Rachkovsky）
        kL = eta0
        eta_I = 1.0 + 0.5 * kL * (H_r + H_l) * (
            1.0 + cos1**2) * 0.5 + kL * H_pi * 0.5 * sin2
        # 为清晰改写：传统写法
        eta_I = 1.0 + kL * (0.5 * (H_r + H_l) * (1.0 + cos1**2) / 2.0 + H_pi *
                            (sin2 / 2.0))
        eta_Q = kL * (H_pi - 0.5 * (H_r + H_l)) * (sin2 * cos2chi) / 2.0
        eta_U = kL * (H_pi - 0.5 * (H_r + H_l)) * (sin2 * sin2chi) / 2.0
        eta_V = kL * (H_r - H_l) * cos1 / 2.0

        rho_Q = kL * (F_pi - 0.5 * (F_r + F_l)) * (sin2 * cos2chi) / 2.0
        rho_U = kL * (F_pi - 0.5 * (F_r + F_l)) * (sin2 * sin2chi) / 2.0
        rho_V = kL * (F_r - F_l) * cos1 / 2.0

        # 紧凑记号
        etaI = eta_I
        etaQ = eta_Q
        etaU = eta_U
        etaV = eta_V
        rhoQ = rho_Q
        rhoU = rho_U
        rhoV = rho_V

        # 稳健的 Δ 表达
        Delta = ((etaI**2 - etaQ**2 - etaU**2 - etaV**2 + rhoQ**2 + rhoU**2 +
                  rhoV**2)**2 + 4.0 *
                 ((etaQ * rhoQ + etaU * rhoU + etaV * rhoV)**2 +
                  (etaU * rhoV - etaV * rhoU)**2 +
                  (etaV * rhoQ - etaQ * rhoV)**2 +
                  (etaQ * rhoU - etaU * rhoQ)**2))
        Delta = np.maximum(Delta, 1e-300)

        # 源函数项
        beta_mu = beta * mu
        denom = 1.0 + beta_mu

        # A_* 项
        etaI2 = etaI**2
        K = etaI2 + rhoQ**2 + rhoU**2 + rhoV**2
        A_I = etaI * K
        A_Q = etaI2 * etaQ + etaI * (rhoV * etaU - rhoU * etaV) + etaQ * (
            rhoQ**2 + rhoU**2 + rhoV**2 - (etaQ**2 + etaU**2 + etaV**2))
        A_U = etaI2 * etaU + etaI * (rhoQ * etaV - rhoV * etaQ) + etaU * (
            rhoQ**2 + rhoU**2 + rhoV**2 - (etaQ**2 + etaU**2 + etaV**2))
        A_V = etaI2 * etaV + etaI * (rhoU * etaQ - rhoQ * etaU) + etaV * (
            rhoQ**2 + rhoU**2 + rhoV**2 - (etaQ**2 + etaU**2 + etaV**2))

        # Stokes（归一到 Ic=1 的出射强度）
        I = (1.0 + (beta_mu * A_I) / Delta) / denom if enable_I else np.zeros(
            (Nlam, Npix))
        Q = ((-beta_mu) * A_Q / (Delta * denom)) if enable_Q else np.zeros(
            (Nlam, Npix))
        U = ((-beta_mu) * A_U / (Delta * denom)) if enable_U else np.zeros(
            (Nlam, Npix))
        V = ((-beta_mu) * A_V / (Delta * denom)) if enable_V else np.zeros(
            (Nlam, Npix))

        # 非磁/未分辨混合：fI, fV（保持 fI ∈ [0,1]）
        I0 = np.ones_like(I)
        I = ld.fI * I + (1.0 - ld.fI) * I0
        Q *= ld.fV
        U *= ld.fV
        V *= ld.fV

        return I, Q, U, V
