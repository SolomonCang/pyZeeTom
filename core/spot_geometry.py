# spot_geometry.py
# 核心模块：处理盘面团块的定义、较差转动演化与亮度/磁场分布生成
# 用于时间相关的物质分布重构

import numpy as np
from typing import List, Tuple, Optional, Callable


class Spot:
    """
    单个团块定义
    
    参数:
    - r: 径向位置（盘面半径单位）
    - phi_initial: 初始方位角（弧度）
    - amplitude: 振幅（正值=发射，负值=吸收）
    - spot_type: 'emission' 或 'absorption'
    - radius: 团块半径（径向宽度）
    - B_amplitude: 磁场振幅（可选）
    - B_direction: 磁场方向 'radial' 或 'azimuthal'（可选）
    """

    def __init__(self,
                 r: float,
                 phi_initial: float,
                 amplitude: float,
                 spot_type: str = 'emission',
                 radius: float = 0.5,
                 B_amplitude: float = 0.0,
                 B_direction: str = 'radial'):
        self.r = float(r)
        self.phi_initial = float(phi_initial)
        self.amplitude = float(amplitude)
        self.spot_type = str(spot_type)
        self.radius = float(radius)
        self.B_amplitude = float(B_amplitude)
        self.B_direction = str(B_direction)

    def __repr__(self):
        return (f"Spot(r={self.r:.2f}, phi={self.phi_initial:.2f}, "
                f"amp={self.amplitude:.2f}, type={self.spot_type})")


class SpotCollection:
    """
    团块集合：管理多个团块并应用较差转动演化
    
    参数:
    - spots: Spot 对象列表
    - pOmega: 较差转动幂律指数（Ω ∝ r^pOmega）
    - r0: 参考半径（默认为1.0）
    - period: 参考周期（天，默认为1.0）
    """

    def __init__(self,
                 spots: Optional[List[Spot]] = None,
                 pOmega: float = -0.5,
                 r0: float = 1.0,
                 period: float = 1.0):
        self.spots = spots if spots is not None else []
        self.pOmega = float(pOmega)
        self.r0 = float(r0)
        self.period = float(period)
        self.omega_ref = 2.0 * np.pi / self.period  # rad/day

    def add_spot(self, spot: Spot):
        """添加团块到集合"""
        self.spots.append(spot)

    def evolve_to_phase(self,
                        phase: float) -> List[Tuple[float, float, float, str]]:
        """
        演化所有团块到指定相位
        
        参数:
        - phase: 相位（0到1，单位：周期）
        
        返回:
        - 演化后的团块位置列表 [(r, phi_evolved, amplitude, type), ...]
        """
        time_days = phase * self.period
        evolved = []

        for spot in self.spots:
            # 较差转动：Ω(r) = Ω_ref × (r/r0)^pOmega
            omega_spot = self.omega_ref * (spot.r / self.r0)**self.pOmega

            # 团块转过的角度
            delta_phi = omega_spot * time_days

            # 更新方位角
            phi_new = (spot.phi_initial + delta_phi) % (2 * np.pi)

            evolved.append((spot.r, phi_new, spot.amplitude, spot.spot_type))

        return evolved

    def get_spots_at_phase(self, phase: float) -> List[Spot]:
        """
        返回演化到指定相位的 Spot 对象列表（保留完整属性）
        
        参数:
        - phase: 相位（0到1）
        
        返回:
        - 演化后的 Spot 对象列表
        """
        time_days = phase * self.period
        evolved_spots = []

        for spot in self.spots:
            omega_spot = self.omega_ref * (spot.r / self.r0)**self.pOmega
            delta_phi = omega_spot * time_days
            phi_new = (spot.phi_initial + delta_phi) % (2 * np.pi)

            # 创建新的 Spot 对象（保留其他属性）
            evolved_spot = Spot(
                r=spot.r,
                phi_initial=phi_new,  # 注意：这里存储演化后的角度
                amplitude=spot.amplitude,
                spot_type=spot.spot_type,
                radius=spot.radius,
                B_amplitude=spot.B_amplitude,
                B_direction=spot.B_direction)
            evolved_spots.append(evolved_spot)

        return evolved_spots

    @staticmethod
    def create_random_spots(n_emission: int = 5,
                            n_absorption: int = 5,
                            r_range: Tuple[float, float] = (0.5, 4.0),
                            amp_emission_range: Tuple[float,
                                                      float] = (1.0, 3.0),
                            amp_absorption_range: Tuple[float,
                                                        float] = (-3.0, -1.0),
                            spot_radius: float = 0.5,
                            B_range: Tuple[float, float] = (500, 2500),
                            seed: Optional[int] = None) -> 'SpotCollection':
        """
        生成随机团块集合
        
        参数:
        - n_emission: 发射团块数量
        - n_absorption: 吸收团块数量
        - r_range: 径向位置范围 (r_min, r_max)
        - amp_emission_range: 发射振幅范围
        - amp_absorption_range: 吸收振幅范围
        - spot_radius: 团块半径
        - B_range: 磁场强度范围
        - seed: 随机数种子
        
        返回:
        - SpotCollection 对象
        """
        if seed is not None:
            np.random.seed(seed)

        spots = []

        # 生成发射团块
        for _ in range(n_emission):
            r = np.random.uniform(*r_range)
            phi = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(*amp_emission_range)
            B_amp = np.random.uniform(*B_range)

            spot = Spot(r=r,
                        phi_initial=phi,
                        amplitude=amp,
                        spot_type='emission',
                        radius=spot_radius,
                        B_amplitude=B_amp,
                        B_direction='radial')
            spots.append(spot)

        # 生成吸收团块
        for _ in range(n_absorption):
            r = np.random.uniform(*r_range)
            phi = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(*amp_absorption_range)
            B_amp = np.random.uniform(*B_range)
            B_sign = np.random.choice([-1, 1])

            spot = Spot(r=r,
                        phi_initial=phi,
                        amplitude=amp,
                        spot_type='absorption',
                        radius=spot_radius,
                        B_amplitude=B_amp * B_sign,
                        B_direction='azimuthal')
            spots.append(spot)

        return SpotCollection(spots=spots)


class TimeEvolvingSpotGeometry:
    """
    时间演化的团块几何：生成指定相位的亮度和磁场分布
    
    参数:
    - grid: diskGrid 对象
    - spot_collection: SpotCollection 对象
    """

    def __init__(self, grid, spot_collection: SpotCollection):
        self.grid = grid
        self.spot_collection = spot_collection
        self._cached_phase = None
        self._cached_distributions = None

    def generate_distributions(
            self, phase: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成指定相位的亮度和磁场分布
        
        参数:
        - phase: 相位（0到1）
        
        返回:
        - (brightness, Br, Bphi): 三个数组，形状为 (grid.numPoints,)
        """
        # 简单缓存机制
        if self._cached_phase == phase and self._cached_distributions is not None:
            return self._cached_distributions

        brightness = np.zeros(self.grid.numPoints, dtype=float)
        Br = np.zeros(self.grid.numPoints, dtype=float)
        Bphi = np.zeros(self.grid.numPoints, dtype=float)

        # 获取演化后的团块
        evolved_spots = self.spot_collection.get_spots_at_phase(phase)

        # 预计算像素坐标
        x_pix = self.grid.r * np.cos(self.grid.phi)
        y_pix = self.grid.r * np.sin(self.grid.phi)

        for spot in evolved_spots:
            # 团块中心坐标
            x_center = spot.r * np.cos(
                spot.phi_initial)  # phi_initial 已更新为演化后的值
            y_center = spot.r * np.sin(spot.phi_initial)

            # 到团块中心的距离
            dist = np.sqrt((x_pix - x_center)**2 + (y_pix - y_center)**2)

            # 亮度分布：高斯型
            spot_brightness = spot.amplitude * np.exp(-(dist / spot.radius)**2)
            brightness += spot_brightness

            # 磁场分布：高斯型
            if spot.B_amplitude != 0.0:
                spot_field = spot.B_amplitude * np.exp(-(dist /
                                                         spot.radius)**2)
                if spot.B_direction == 'radial':
                    Br += spot_field
                elif spot.B_direction == 'azimuthal':
                    Bphi += spot_field

        self._cached_phase = phase
        self._cached_distributions = (brightness, Br, Bphi)

        return brightness, Br, Bphi

    def get_response_func_time_aware(self) -> Callable:
        """
        返回用于 VelspaceDiskIntegrator 的时间相关响应函数
        
        返回:
        - response_func: 函数 f(r, phi, phase) -> brightness
        """

        def response_func(r, phi, phase=0.0):
            """
            时间相关响应函数：根据相位动态计算亮度分布
            
            参数:
            - r: 径向坐标数组
            - phi: 方位角数组
            - phase: 相位（0到1）
            
            返回:
            - brightness: 亮度数组
            """
            # 验证 r, phi 与 grid 一致
            if not (np.array_equal(r, self.grid.r)
                    and np.array_equal(phi, self.grid.phi)):
                raise ValueError("当前实现要求 r, phi 与 grid 完全一致。"
                                 "若需插值，请扩展实现。")

            brightness, _, _ = self.generate_distributions(phase)
            return brightness

        return response_func

    def get_Blos(self,
                 phase: float,
                 inclination_rad: float,
                 phi_obs: float = 0.0) -> np.ndarray:
        """
        计算视向磁场分量
        
        参数:
        - phase: 相位（0到1）
        - inclination_rad: 倾角（弧度）
        - phi_obs: 观测者方位角（弧度，默认0）
        
        返回:
        - Blos: 视向磁场数组，形状 (grid.numPoints,)
        """
        _, Br, Bphi = self.generate_distributions(phase)

        # 视向投影
        # Blos ≈ Br * sin(i) * cos(phi - phi_obs) + Bphi * ...
        # 简化：只考虑径向分量的投影
        Blos = Br * np.sin(inclination_rad) * np.cos(self.grid.phi - phi_obs)

        return Blos
