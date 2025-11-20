# linemodel.py
import numpy as np


class LineData:
    """
    读取单行谱线参数文件:
    常见行格式: wl0  sigWl  g
    仅使用 wl0, sigWl, g。
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
                # 抽取可解析的浮点列
                vals = []
                for p in parts:
                    try:
                        vals.append(float(p))
                    except Exception:
                        pass
                if len(vals) < 3:
                    continue
                # 取前三列: [wl0, sigWl, g]
                self.wl0 = vals[0]
                self.sigWl = vals[1]
                self.g = vals[2]
                self.numLines += 1
                break
        if self.numLines == 0:
            raise ValueError("未从文件中读取到有效的谱线参数")


class BaseLineModel:
    """
    局部谱线模型接口：
      compute_local_profile(wl_grid, amp, Blos=None, **kwargs) -> dict with keys 'I','V','Q','U'
    说明:
      - wl_grid: (Nlambda,) 或 (Nlambda, Npix)
      - amp: 线项振幅（标量或 (Npix,) 或 (1,Npix)），amp<0 吸收，amp=0 无线型，amp>0 发射
      - Blos: 每像素视向磁场（(Npix,)）
      - 可选 Q/U 输入：
          Bperp: |B_⊥|（(Npix,)）
          chi:   横向场方向角 χ（弧度，(Npix,)），相对于 Q 的参考轴
      - 计算开关（kwargs）:
          enable_V:  默认 True
          enable_QU: 默认 True
      - Ic_weight（可选）：若提供，作为像素加权系数在结果中相乘（例如用于盘面积分）。
        若不提供，则默认全 1。谱线连续谱基线恒为 1。
    """

    def compute_local_profile(self, wl_grid, amp, Blos=None, **kwargs):
        raise NotImplementedError


class GaussianZeemanWeakLineModel(BaseLineModel):
    """
    弱线近似 + 高斯线型（连续谱=1；线型由 amp 单独控制）
    记:
      d = (λ - λ0)/σ,  G = exp(-d^2)

    输出:
      I = 1 + amp * G
      V = Cg * Blos * (amp * G * d / σ)
      Q = -C2 * Bperp^2 * (amp * (G/σ^2) * (1 - 2 d^2)) * cos(2χ)
      U = -C2 * Bperp^2 * (amp * (G/σ^2) * (1 - 2 d^2)) * sin(2χ)

    注意:
      - amp 可为标量或每像素值；若传 (Npix,) 将广播到 (1,Npix) 并与 (Nλ,Npix) 的 wl_grid 对齐。
      - 返回结果若提供 Ic_weight，将在最后整体相乘（权重作用），不改变连续谱基线=1 的定义。
    """

    def __init__(self,
                 line_data: LineData,
                 k_QU: float = 1.0,
                 enable_V: bool = True,
                 enable_QU: bool = True):
        self.ld = line_data
        # V 的比例常数（与常用途径一致）
        self.Cg = -2.0 * 4.6686e-12 * (self.ld.wl0**2) * self.ld.g
        # Q/U 的比例常数（弱场二阶）
        base = 4.6686e-12 * (self.ld.wl0**2) * self.ld.g
        self.C2 = (base**2) * float(k_QU)
        self.enable_V_default = bool(enable_V)
        self.enable_QU_default = bool(enable_QU)

    def compute_local_profile(self, wl_grid, amp, Blos=None, **kwargs):
        # 开关
        enable_V = bool(kwargs.get("enable_V", self.enable_V_default))
        enable_QU = bool(kwargs.get("enable_QU", self.enable_QU_default))

        Bperp = kwargs.get("Bperp", None)
        chi = kwargs.get("chi", None)

        # 像素权重（可选）：用于最终输出的加权，不改变谱线基线=1
        Ic_weight = kwargs.get("Ic_weight", None)

        # 形状处理
        wl_grid = np.asarray(wl_grid, dtype=float)
        if wl_grid.ndim == 1:
            wl_grid = wl_grid[:, None]  # (Nλ,1)
        Nlam, Npix = wl_grid.shape

        # amp 处理并广播到 (1,Npix)
        amp = np.asarray(amp, dtype=float)
        if amp.ndim == 0:
            amp = amp.reshape(1, 1)
        elif amp.ndim == 1:
            amp = amp.reshape(1, -1)
        # 广播检查
        try:
            amp = np.broadcast_to(amp, (1, Npix))
        except ValueError:
            raise ValueError("amp 需为标量或长度 Npix，可广播到 (1,Npix)。")

        # 通用核
        sig = float(self.ld.sigWl)
        d = (wl_grid - self.ld.wl0) / sig
        G = np.exp(-(d * d))

        # Debug print (once)
        if not hasattr(self, '_debug_printed'):
            print(f"[LineModel] wl0={self.ld.wl0}, sig={sig}")
            print(
                f"[LineModel] wl_grid range: {np.min(wl_grid):.4f} - {np.max(wl_grid):.4f}"
            )
            print(f"[LineModel] d range: {np.min(d):.4f} - {np.max(d):.4f}")
            print(f"[LineModel] G max: {np.max(G):.4f}")
            if Blos is not None:
                print(f"[LineModel] Blos max: {np.max(Blos):.4f}")
            self._debug_printed = True

        # I（连续谱=1）
        I = 1.0 + amp * G

        # V
        if enable_V and (Blos is not None):
            Blos_arr = np.asarray(Blos, dtype=float).reshape(1, Npix)
            V = self.Cg * Blos_arr * (amp * G * d / sig)
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
                Q = -self.C2 * Bperp2 * (amp * d2_core) * cos2c
                U = -self.C2 * Bperp2 * (amp * d2_core) * sin2c
            else:
                Q = np.zeros((Nlam, Npix), dtype=float)
                U = np.zeros((Nlam, Npix), dtype=float)
        else:
            Q = np.zeros((Nlam, Npix), dtype=float)
            U = np.zeros((Nlam, Npix), dtype=float)

        # 可选像素权重：最后统一相乘（不改变 I 的基线定义）
        if Ic_weight is not None:
            w = np.asarray(Ic_weight, dtype=float).reshape(1, Npix)
            I = I * w
            V = V * w
            Q = Q * w
            U = U * w

        return {"I": I, "V": V, "Q": Q, "U": U}


class ConstantAmpLineModel(BaseLineModel):
    """
    以恒定强度包裹基础谱线模型的适配器。

    提供一个方便的接口，使用常数振幅运行任意基础谱线模型
    而无需显式传递 amp 参数。这在前向建模中很有用，
    其中亮度分布假设为已知（固定）。

    Parameters
    ----------
    base_model : BaseLineModel
        底层谱线模型对象，应具有
        compute_local_profile(wl_grid, amp, **kwargs) 方法
    amp : float, default=1.0
        恒定振幅值（应用于所有像素）

    Attributes
    ----------
    base_model : BaseLineModel
        底层模型
    amp : float
        恒定振幅

    Examples
    --------
    >>> from core.local_linemodel_basic import LineData, GaussianZeemanWeakLineModel, ConstantAmpLineModel
    >>> ld = LineData('input/lines.txt')
    >>> base = GaussianZeemanWeakLineModel(ld)
    >>> adapter = ConstantAmpLineModel(base, amp=0.5)
    >>>
    >>> # 不需要传递 amp，使用恒定值
    >>> wl_grid = np.linspace(6562.5, 6563.5, 200)  # 仅波长网格
    >>> result = adapter.compute_local_profile(wl_grid, Blos=np.zeros(100))
    >>> print(result['I'].shape)  # (200, 100)
    """

    def __init__(self, base_model: BaseLineModel, amp: float = 1.0):
        """Initialize ConstantAmpLineModel."""
        self.base_model = base_model
        self.amp = float(amp)

    def compute_local_profile(self, wl_grid, amp_unused=None, **kwargs):
        """计算本地谱线型，使用存储的恒定振幅。

        Parameters
        ----------
        wl_grid : np.ndarray
            波长网格
        amp_unused : ignored
            此参数被忽略，使用 self.amp 代替
        **kwargs
            传递给 base_model.compute_local_profile 的其他关键字参数

        Returns
        -------
        dict
            包含 'I', 'V', 'Q', 'U' 键的字典，对应计算的 Stokes 参数
        """
        # 从 kwargs 中移除 amp（如果存在），因为我们要使用 self.amp
        kwargs.pop('amp', None)

        return self.base_model.compute_local_profile(wl_grid, self.amp,
                                                     **kwargs)
