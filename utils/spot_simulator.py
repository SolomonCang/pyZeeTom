# spot_simulator.py (精简版)
# Spot模拟库：多spot配置、几何模型构建、.tomog输出
# 功能：
#   1. 多个spot的管理（位置、磁场、时间演化）
#   2. 将spot映射到disk grid像素
#   3. 生成符合规范的.tomog模型文件
#   4. 注意：谱线合成由 pyzeetom/tomography.py 的0-iter正演处理
#           通过 initTomogFile 参数加载模型

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from types import SimpleNamespace as NS


@dataclass
class SpotConfig:
    """单个spot的配置参数"""
    r: float  # 径向位置（盘面半径单位）
    phi: float  # 初始方位角（弧度）
    amplitude: float  # 振幅（正值=发射，负值=吸收）
    spot_type: str = 'emission'  # 'emission' 或 'absorption'
    radius: float = 0.5  # spot的径向宽度（FWHM）
    width_type: str = 'gaussian'  # 宽度类型 'gaussian' 或 'tophat'
    B_los: float = 0.0  # 视向磁场
    B_perp: float = 0.0  # 横向磁场
    chi: float = 0.0  # 磁场方位角（弧度）
    velocity_shift: float = 0.0  # 附加速度偏移（km/s）

    def __post_init__(self):
        self.r = float(self.r)
        self.phi = float(self.phi)
        self.amplitude = float(self.amplitude)
        self.radius = float(self.radius)
        self.B_los = float(self.B_los)
        self.B_perp = float(self.B_perp)
        self.chi = float(self.chi)
        self.velocity_shift = float(self.velocity_shift)


class SpotSimulator:
    """
    Spot模拟器：
      - 管理多个spot配置
      - 将spot映射到disk grid像素
      - 计算响应函数和磁场分布
      - 输出.tomog模型文件供后续正演使用
    """

    def __init__(self,
                 grid,
                 inclination_rad: float = np.deg2rad(60.0),
                 phi0: float = 0.0,
                 pOmega: float = -0.5,
                 r0_rot: float = 1.0,
                 period_days: float = 1.0):
        """
        初始化spot模拟器

        参数:
        -------
        grid : diskGrid
            网格对象（来自 grid_tom.py）
        inclination_rad : float
            倾角（弧度），默认60度
        phi0 : float
            参考方位角（弧度）
        pOmega : float
            较差转动指数（Ω ∝ r^pOmega）
        r0_rot : float
            较差转动参考半径
        period_days : float
            自转周期（天）
        """
        self.grid = grid
        self.inclination_rad = float(inclination_rad)
        self.phi0 = float(phi0)
        self.pOmega = float(pOmega)
        self.r0_rot = float(r0_rot)
        self.period_days = float(period_days)

        # Spot列表
        self.spots: List[SpotConfig] = []

        # 初始化像素属性数组
        self._init_pixel_arrays()

    def _init_pixel_arrays(self):
        """初始化所有像素属性数组"""
        n = self.grid.numPoints
        self.response = np.ones(n, dtype=float)  # 响应（>1 发射强，<1 弱）
        self.B_los_map = np.zeros(n, dtype=float)
        self.B_perp_map = np.zeros(n, dtype=float)
        self.chi_map = np.zeros(n, dtype=float)

    def add_spot(self, spot_config: SpotConfig):
        """添加单个spot"""
        self.spots.append(spot_config)

    def add_spots(self, spot_configs: List[SpotConfig]):
        """添加多个spot"""
        self.spots.extend(spot_configs)

    def create_spot(self, r: float, phi: float, amplitude: float,
                    **kwargs) -> SpotConfig:
        """
        创建并添加单个spot（快捷方法）

        参数: 参见 SpotConfig
        返回: SpotConfig对象
        """
        spot = SpotConfig(r=r, phi=phi, amplitude=amplitude, **kwargs)
        self.add_spot(spot)
        return spot

    def evolve_spots_to_phase(self, phase: float) -> List[SpotConfig]:
        """
        将spot演化到指定相位（考虑较差转动）

        参数:
        -------
        phase : float
            旋转相位（0~1）

        返回:
        -------
        List[SpotConfig]
            演化后的spot列表
        """
        evolved_spots = []
        for spot in self.spots:
            # 复制当前spot配置
            s = SpotConfig(
                r=spot.r,
                phi=spot.phi,
                amplitude=spot.amplitude,
                spot_type=spot.spot_type,
                radius=spot.radius,
                width_type=spot.width_type,
                B_los=spot.B_los,
                B_perp=spot.B_perp,
                chi=spot.chi,
                velocity_shift=spot.velocity_shift
            )
            
            # 考虑较差转动的相位演化
            # Δφ = 2π * phase * (r/r0_rot)^(pOmega+1)
            # 这是从 pOmega 定义出发：Ω(r) = Ω_ref * (r/r0_rot)^pOmega
            radius_ratio = s.r / self.r0_rot if self.r0_rot > 0 else 1.0
            delphi = 2.0 * np.pi * phase * (radius_ratio**(self.pOmega + 1.0))
            s.phi = s.phi + delphi
            
            # 归一化到 [0, 2π)
            s.phi = s.phi % (2.0 * np.pi)
            
            evolved_spots.append(s)
        
        return evolved_spots

    def _gaussian_weight(self, dr: np.ndarray, sigma: float) -> np.ndarray:
        """高斯权重函数"""
        return np.exp(-0.5 * (dr / sigma)**2)

    def _tophat_weight(self, dr: np.ndarray, radius: float) -> np.ndarray:
        """顶帽权重函数（rect函数）"""
        weight = np.zeros_like(dr)
        weight[np.abs(dr) <= radius] = 1.0
        return weight

    def _compute_azimuthal_weight(self, dphi: np.ndarray,
                                   sigma_phi: float) -> np.ndarray:
        """方位角权重（高斯）"""
        # 处理 2π 周期性
        dphi_wrapped = np.abs(dphi)
        dphi_wrapped = np.minimum(dphi_wrapped, 2.0 * np.pi - dphi_wrapped)
        return np.exp(-0.5 * (dphi_wrapped / sigma_phi)**2)

    def apply_spots_to_grid(self, phase: float = 0.0) -> None:
        """
        应用spot到网格，计算响应和磁场分布

        参数:
        -------
        phase : float
            旋转相位（0~1）
        """
        # 初始化
        self._init_pixel_arrays()
        
        # 获取演化的spot
        spots_at_phase = self.evolve_spots_to_phase(phase)
        
        # 对每个spot应用到网格
        for spot in spots_at_phase:
            self._apply_single_spot(spot)

    def _apply_single_spot(self, spot: SpotConfig) -> None:
        """将单个spot应用到网格"""
        # 计算每个像素到spot的距离
        dr = np.sqrt((self.grid.r - spot.r)**2)
        dphi = self.grid.phi - spot.phi
        
        # 径向权重
        if spot.width_type == 'gaussian':
            sigma_r = spot.radius / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            r_weight = self._gaussian_weight(dr, sigma_r)
        else:  # tophat
            r_weight = self._tophat_weight(dr, spot.radius)
        
        # 方位角权重（高斯，宽度为 spot.radius 在方位角方向）
        sigma_phi = spot.radius / (self.grid.r + 1e-10)  # 防止除零
        phi_weight = self._compute_azimuthal_weight(dphi, sigma_phi)
        
        # 总权重
        weight = r_weight * phi_weight
        
        # 应用响应
        if spot.spot_type == 'emission':
            # 发射：response > 1
            self.response += (spot.amplitude) * weight
        else:  # absorption
            # 吸收：response < 1
            self.response += (spot.amplitude) * weight
        
        # 应用磁场（加权平均）
        weight_sum = np.sum(weight)
        if weight_sum > 0:
            self.B_los_map += spot.B_los * weight
            self.B_perp_map += spot.B_perp * weight
            self.chi_map += spot.chi * weight

    def create_geometry_object(self,
                               phase: float = 0.0) -> NS:
        """
        创建几何对象供VelspaceDiskIntegrator使用

        参数:
        -------
        phase : float
            旋转相位

        返回:
        -------
        geometry : SimpleNamespace
            包含grid、磁场参数等的几何对象
        """
        # 应用spot到网格
        self.apply_spots_to_grid(phase)
        
        # 构造几何对象
        geom = NS()
        geom.grid = self.grid
        geom.area_proj = np.asarray(self.grid.area)
        geom.inclination_rad = self.inclination_rad
        geom.phi0 = self.phi0
        geom.pOmega = self.pOmega
        geom.r0 = self.r0_rot
        geom.period = self.period_days
        
        # 磁场参数
        geom.B_los = self.B_los_map.copy()
        geom.B_perp = self.B_perp_map.copy()
        geom.chi = self.chi_map.copy()
        
        return geom

    def export_to_geomodel(self,
                           filepath: str,
                           phase: float = 0.0,
                           meta: Optional[Dict[str, Any]] = None) -> str:
        """
        将模型导出为.tomog文件供后续正演使用

        参数:
        -------
        filepath : str
            输出文件路径（.tomog格式）
        phase : float
            相位
        meta : dict, optional
            元信息（目标名、观测参数等）

        返回:
        -------
        str
            输出文件路径
        """
        import datetime as dt
        
        # 应用spot到网格
        self.apply_spots_to_grid(phase)
        
        # 构造header
        header = {
            "format": "TOMOG_MODEL",
            "version": 1,
            "created_utc": dt.datetime.utcnow().isoformat() + "Z",
            "source": "SpotSimulator",
            "inclination_deg": float(np.rad2deg(self.inclination_rad)),
            "phi0": float(self.phi0),
            "pOmega": float(self.pOmega),
            "r0_rot": float(self.r0_rot),
            "period": float(self.period_days),
            "nr": int(self.grid.nr) if hasattr(self.grid, 'nr') else -1,
            "phase": float(phase),
        }
        
        if isinstance(meta, dict):
            for k, v in meta.items():
                header[str(k)] = v
        
        # 检查是否有 B_perp 和 chi
        has_B_perp = np.any(self.B_perp_map != 0.0)
        has_chi = np.any(self.chi_map != 0.0)
        
        # 列定义
        columns = [
            "idx", "ring_id", "phi_id", "r", "phi", "area", "Ic_weight",
            "A", "Blos"
        ]
        if has_B_perp:
            columns.extend(["Bperp", "chi"])
        
        # 写文件
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# TOMOG Geometric Model File (generated by SpotSimulator)\n")
            for k in sorted(header.keys()):
                v = header[k]
                if isinstance(v, (list, tuple, np.ndarray)):
                    arr = np.asarray(v).ravel()
                    vstr = ",".join(f"{x:.12g}" for x in arr)
                    f.write(f"# {k}: [{vstr}]\n")
                else:
                    f.write(f"# {k}: {v}\n")
            
            # 尝试补充网格边界
            if hasattr(self.grid, "r_edges"):
                vstr = ",".join(f"{x:.12g}"
                                for x in np.asarray(self.grid.r_edges).ravel())
                f.write(f"# r_edges: [{vstr}]\n")
            
            f.write("# COLUMNS: " + ", ".join(columns) + "\n")
            
            # 写数据
            n = self.grid.numPoints
            for i in range(n):
                row = [
                    i,
                    int(self.grid.ring_id[i])
                    if hasattr(self.grid, "ring_id") else -1,
                    int(getattr(self.grid, "phi_id", np.zeros(n, int))[i]),
                    float(self.grid.r[i]),
                    float(self.grid.phi[i]),
                    float(self.grid.area[i]),
                    float(1.0),  # Ic_weight (始终为1.0)
                    float(self.response[i]),  # response作为A列
                    float(self.B_los_map[i]),
                ]
                if has_B_perp:
                    row.append(float(self.B_perp_map[i]))
                    row.append(float(self.chi_map[i]))
                f.write(" ".join(str(x) for x in row) + "\n")
        
        return filepath


# ============================================================================
# 便捷函数
# ============================================================================


def create_simple_spot_simulator(nr: int = 40,
                                 r_in: float = 0.5,
                                 r_out: float = 4.0,
                                 inclination_deg: float = 60.0,
                                 pOmega: float = -0.5,
                                 r0_rot: float = 1.0,
                                 period_days: float = 1.0) -> SpotSimulator:
    """
    快速创建SpotSimulator

    参数:
    -------
    nr : int
        环数
    r_in, r_out : float
        径向范围
    inclination_deg : float
        倾角（度）
    pOmega : float
        较差转动指数
    r0_rot : float
        参考半径
    period_days : float
        周期（天）

    返回:
    -------
    SpotSimulator
        simulator对象
    """
    from core.grid_tom import diskGrid

    grid = diskGrid(nr=nr, r_in=r_in, r_out=r_out)
    sim = SpotSimulator(grid,
                        inclination_rad=np.deg2rad(inclination_deg),
                        pOmega=pOmega,
                        r0_rot=r0_rot,
                        period_days=period_days)
    return sim


def create_test_spots() -> List[SpotConfig]:
    """创建测试用的spot配置"""
    spots = [
        SpotConfig(r=1.5, phi=0.0, amplitude=2.0, B_los=1000.0),
        SpotConfig(r=2.0, phi=np.pi, amplitude=1.5, B_los=-500.0),
        SpotConfig(r=3.0,
                   phi=np.pi / 2,
                   amplitude=-1.5,
                   spot_type='absorption'),
    ]
    return spots
