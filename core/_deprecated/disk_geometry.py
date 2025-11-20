"""Disk geometry container and utilities for tomography.

This module provides the SimpleDiskGeometry class that encapsulates
disk parameters, magnetic field configuration, and stellar parameters
for use with VelspaceDiskIntegrator.

Classes
-------
SimpleDiskGeometry : Disk geometry container with magnetic field support
"""

import numpy as np
from typing import Optional
from core.grid_tom import diskGrid

__all__ = ['SimpleDiskGeometry', 'create_disk_geometry_from_params']


class SimpleDiskGeometry:
    """最小化的盘几何容器，供 VelspaceDiskIntegrator 使用。
    
    提供：grid, area_proj, inclination_rad, phi0, pOmega, r0, period,
         enable_stellar_occultation, stellar_radius, B_los, B_perp, chi, brightness。
    
    Parameters
    ----------
    grid : diskGrid
        盘面网格对象，包含像素位置、面积等信息
    inclination_deg : float, default=60.0
        盘面倾角（度数）
    phi0 : float, default=0.0
        参考角位置（弧度）
    pOmega : float, default=0.0
        差速转动指数 (pOmega = d(ln Omega)/d(ln r))
        - pOmega = 0: Solid body rotation (刚体转动)
        - pOmega = -0.5: v ∝ r^0.5 (Keplerian-like)
        - pOmega = -1.0: Angular momentum conserved
    r0 : float, default=1.0
        参考半径 (R_sun)
    period : float, default=1.0
        自转周期（天）
    enable_stellar_occultation : int, default=0
        是否启用恒星遮挡效应（0=禁用, 1=启用）
    stellar_radius : float, default=1.0
        恒星半径 (R_sun)
    B_los : np.ndarray, optional
        视向磁场分量 (Gauss)，长度应为 grid.numPoints
        如果为 None，初始化为零
    B_perp : np.ndarray, optional
        垂直于视线的磁场强度 (Gauss)
        如果为 None，初始化为零
    chi : np.ndarray, optional
        磁场方向角 (rad)，定义垂直磁场的方向
        如果为 None，初始化为零
    brightness : np.ndarray, optional
        亮度分布 (归一化, 0-1)，用于谱线振幅调制
        如果为 None，初始化为1.0（均匀）
    
    Attributes
    ----------
    grid : diskGrid
        盘面网格
    area_proj : np.ndarray
        投影面积数组
    inclination_rad : float
        倾角（弧度）
    phi0 : float
        参考角位置
    pOmega : float
        差速转动指数
    r0 : float
        参考半径
    period : float
        自转周期
    enable_stellar_occultation : bool
        恒星遮挡标志
    stellar_radius : float
        恒星半径
    B_los : np.ndarray
        视向磁场
    B_perp : np.ndarray
        垂直磁场强度
    chi : np.ndarray
        磁场方向角
    brightness : np.ndarray
        亮度分布 (0-1)
    """

    def __init__(self,
                 grid: diskGrid,
                 inclination_deg: float = 60.0,
                 phi0: float = 0.0,
                 pOmega: float = 0.0,
                 r0: float = 1.0,
                 period: float = 1.0,
                 enable_stellar_occultation: int = 0,
                 stellar_radius: float = 1.0,
                 B_los: Optional[np.ndarray] = None,
                 B_perp: Optional[np.ndarray] = None,
                 chi: Optional[np.ndarray] = None,
                 brightness: Optional[np.ndarray] = None):
        """Initialize SimpleDiskGeometry."""
        self.grid = grid
        self.area_proj = np.asarray(grid.area)
        self.inclination_rad = np.deg2rad(float(inclination_deg))
        self.phi0 = float(phi0)
        self.pOmega = float(pOmega)
        self.r0 = float(r0)
        self.period = float(period)
        self.enable_stellar_occultation = bool(enable_stellar_occultation)
        self.stellar_radius = float(stellar_radius)

        # 验证磁场参数维度
        npix = grid.numPoints
        if B_los is not None:
            B_los = np.asarray(B_los, dtype=float)
            if len(B_los) != npix:
                raise ValueError(
                    f"B_los length ({len(B_los)}) must match grid.numPoints ({npix})"
                )
        if B_perp is not None:
            B_perp = np.asarray(B_perp, dtype=float)
            if len(B_perp) != npix:
                raise ValueError(
                    f"B_perp length ({len(B_perp)}) must match grid.numPoints ({npix})"
                )
        if chi is not None:
            chi = np.asarray(chi, dtype=float)
            if len(chi) != npix:
                raise ValueError(
                    f"chi length ({len(chi)}) must match grid.numPoints ({npix})"
                )
        if brightness is not None:
            brightness = np.asarray(brightness, dtype=float)
            if len(brightness) != npix:
                raise ValueError(
                    f"brightness length ({len(brightness)}) must match grid.numPoints ({npix})"
                )

        # 初始化磁场参数
        self.B_los = B_los if B_los is not None else np.zeros(npix)
        self.B_perp = B_perp if B_perp is not None else np.zeros(npix)
        self.chi = chi if chi is not None else np.zeros(npix)
        self.brightness = brightness if brightness is not None else np.ones(
            npix)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SimpleDiskGeometry(inclination={np.rad2deg(self.inclination_rad):.1f}°, "
            f"pOmega={self.pOmega:.2f}, period={self.period:.3f} d, "
            f"npix={self.grid.numPoints})")

    def get_magnetic_field_summary(self) -> dict:
        """获取磁场和亮度参数摘要。
        
        Returns
        -------
        dict
            包含磁场和亮度统计信息的字典
        """
        return {
            'B_los_min': float(np.min(self.B_los)),
            'B_los_max': float(np.max(self.B_los)),
            'B_los_mean': float(np.mean(self.B_los)),
            'B_perp_min': float(np.min(self.B_perp)),
            'B_perp_max': float(np.max(self.B_perp)),
            'B_perp_mean': float(np.mean(self.B_perp)),
            'chi_min': float(np.min(self.chi)),
            'chi_max': float(np.max(self.chi)),
            'brightness_min': float(np.min(self.brightness)),
            'brightness_max': float(np.max(self.brightness)),
            'brightness_mean': float(np.mean(self.brightness)),
        }


def create_disk_geometry_from_params(par,
                                     grid,
                                     B_los=None,
                                     B_perp=None,
                                     chi=None,
                                     brightness=None,
                                     verbose=0):
    """从参数对象创建 SimpleDiskGeometry。
    
    便利函数：从 readParamsTomog 对象直接创建几何容器。
    
    Parameters
    ----------
    par : readParamsTomog
        参数对象，必须包含以下属性：
        - inclination : float (度)
        - pOmega : float
        - radius : float (R_sun)
        - period : float (天)
        - enable_stellar_occultation : int, optional
    grid : diskGrid
        盘面网格
    B_los : np.ndarray, optional
        视向磁场 (Gauss)
    B_perp : np.ndarray, optional
        垂直磁场强度 (Gauss)
    chi : np.ndarray, optional
        磁场方向角 (rad)
    brightness : np.ndarray, optional
        亮度分布 (0-1)
    verbose : int, default=0
        详细程度 (0=安静, 1=正常, 2=详细)
    
    Returns
    -------
    SimpleDiskGeometry
        创建的几何对象
    """
    npix = grid.numPoints

    if verbose:
        print("[disk_geometry] Creating geometry from params...")
        print(
            f"  Grid: {grid.numPoints} pixels, r ∈ [{grid.r_in:.2f}, {grid.r_out:.2f}] R_sun"
        )

    geom = SimpleDiskGeometry(
        grid=grid,
        inclination_deg=float(getattr(par, 'inclination', 60.0)),
        phi0=0.0,
        pOmega=float(getattr(par, 'pOmega', 0.0)),
        r0=float(getattr(par, 'radius', 1.0)),
        period=float(getattr(par, 'period', 1.0)),
        enable_stellar_occultation=int(
            getattr(par, 'enable_stellar_occultation', 0)),
        stellar_radius=float(getattr(par, 'radius', 1.0)),
        B_los=B_los if B_los is not None else np.zeros(npix),
        B_perp=B_perp if B_perp is not None else np.zeros(npix),
        chi=chi if chi is not None else np.zeros(npix),
        brightness=brightness if brightness is not None else np.ones(npix))

    if verbose:
        print(f"  {geom}")
        if verbose > 1:
            mag_summary = geom.get_magnetic_field_summary()
            print("  Magnetic field:")
            print(
                f"    B_los: [{mag_summary['B_los_min']:.1f}, {mag_summary['B_los_max']:.1f}] G, "
                f"mean={mag_summary['B_los_mean']:.1f} G")
            print(
                f"    B_perp: [{mag_summary['B_perp_min']:.1f}, {mag_summary['B_perp_max']:.1f}] G, "
                f"mean={mag_summary['B_perp_mean']:.1f} G")
            print("  Brightness:")
            print(
                f"    [{mag_summary['brightness_min']:.2f}, {mag_summary['brightness_max']:.2f}], "
                f"mean={mag_summary['brightness_mean']:.2f}")

    return geom
