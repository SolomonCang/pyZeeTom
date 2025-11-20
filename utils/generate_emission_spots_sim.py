# generate_emission_spots_sim.py (精简版)
# Spot模拟配置与.tomog生成工具
# 功能：
#   1. 从参数或代码配置多个spot
#   2. 使用SpotSimulator创建几何模型
#   3. 输出.tomog文件供pyzeetom/tomography.py加载
#   4. 注意：谱线合成由 pyzeetom/tomography.py 的0-iter正演处理
#           通过修改 input/params_tomog.txt 的 initTomogFile 和 initModelPath 接入

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

# 导入核心模块
from core.grid_tom import diskGrid
from utils.spot_simulator import SpotSimulator, SpotConfig


class SpotSimulationConfig:
    """
    Spot模拟配置管理：
      1. 配置grid参数
      2. 配置spot列表
      3. 生成.tomog模型
    """

    def __init__(self, output_dir: str = "./output", verbose: int = 1):
        """
        初始化配置

        参数:
        -------
        output_dir : str
            输出目录（存放.tomog文件）
        verbose : int
            输出详细程度
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = int(verbose)
        self.simulator = None
        self.spot_configs: List[SpotConfig] = []
        
        # 几何参数
        self.grid_params = {}
        self.geom_params = {}

    def setup_grid(self,
                   nr: int = 40,
                   r_in: float = 0.5,
                   r_out: float = 4.0,
                   inclination_deg: float = 60.0,
                   pOmega: float = -0.5,
                   r0_rot: float = 1.0,
                   period_days: float = 1.0) -> SpotSimulator:
        """
        设置grid和几何参数

        返回:
        -------
        SpotSimulator
            初始化的simulator
        """
        # 保存参数
        self.grid_params = {
            'nr': nr,
            'r_in': r_in,
            'r_out': r_out,
        }
        self.geom_params = {
            'inclination_deg': inclination_deg,
            'pOmega': pOmega,
            'r0_rot': r0_rot,
            'period_days': period_days,
        }
        
        # 创建grid
        grid = diskGrid(nr=nr, r_in=r_in, r_out=r_out, verbose=self.verbose)
        
        # 创建simulator
        self.simulator = SpotSimulator(
            grid,
            inclination_rad=np.deg2rad(inclination_deg),
            phi0=0.0,
            pOmega=pOmega,
            r0_rot=r0_rot,
            period_days=period_days)

        if self.verbose:
            print(f"[SpotConfig] Grid setup: nr={nr}, r=[{r_in},{r_out}]")
            print(f"[SpotConfig] Geometry: i={inclination_deg}°, pOmega={pOmega}, "
                  f"r0={r0_rot}R*, P={period_days}d")
        
        return self.simulator

    def add_spot(self, **kwargs) -> SpotConfig:
        """
        添加单个spot

        参数: SpotConfig 的所有参数 (r, phi, amplitude, B_los, B_perp, chi等)
        返回: SpotConfig对象
        """
        spot = SpotConfig(**kwargs)
        self.spot_configs.append(spot)
        if self.simulator is not None:
            self.simulator.add_spot(spot)
        
        if self.verbose:
            print(f"[SpotConfig] Added spot: r={spot.r:.2f}, phi={np.rad2deg(spot.phi):.1f}°, "
                  f"amp={spot.amplitude:.2f}, B_los={spot.B_los:.1f}G")
        
        return spot

    def add_spots(self, spots: List[SpotConfig]) -> None:
        """添加多个spot"""
        for spot in spots:
            self.add_spot(
                r=spot.r,
                phi=spot.phi,
                amplitude=spot.amplitude,
                spot_type=spot.spot_type,
                radius=spot.radius,
                width_type=spot.width_type,
                B_los=spot.B_los,
                B_perp=spot.B_perp,
                chi=spot.chi,
                velocity_shift=spot.velocity_shift)

    def generate_tomog_model(self,
                             output_file: Optional[str] = None,
                             phase: float = 0.0,
                             meta: Optional[Dict[str, Any]] = None) -> str:
        """
        生成.tomog模型文件

        参数:
        -------
        output_file : str, optional
            输出文件路径（.tomog格式）
            若不指定，使用默认 output/spot_model_phase_XXX.tomog
        phase : float
            演化相位 (0~1)
        meta : dict, optional
            额外的元信息

        返回:
        -------
        str
            输出文件路径
        """
        if self.simulator is None:
            raise ValueError("Grid not setup. Call setup_grid() first.")
        
        # 生成默认输出文件名
        if output_file is None:
            output_file = self.output_dir / f"spot_model_phase_{phase:03.0f}.tomog"
        else:
            output_file = Path(output_file)
        
        # 准备元信息
        if meta is None:
            meta = {}
        
        # 添加配置信息到meta
        meta.update({
            'grid_nr': self.grid_params.get('nr'),
            'grid_r_in': self.grid_params.get('r_in'),
            'grid_r_out': self.grid_params.get('r_out'),
            'n_spots': len(self.spot_configs),
        })
        
        # 导出.tomog文件
        result = self.simulator.export_to_geomodel(
            str(output_file),
            phase=phase,
            meta=meta)
        
        if self.verbose:
            print(f"[SpotConfig] Generated .tomog model: {result}")
        
        return result

    def save_config_json(self, output_file: Optional[str] = None) -> str:
        """
        保存配置为JSON文件（便于复现）

        参数:
        -------
        output_file : str, optional
            输出JSON文件路径
            若不指定，使用 output/spot_config.json

        返回:
        -------
        str
            输出文件路径
        """
        if output_file is None:
            output_file = self.output_dir / "spot_config.json"
        else:
            output_file = Path(output_file)
        
        # 配置字典
        config_dict = {
            'grid_params': self.grid_params,
            'geom_params': self.geom_params,
            'spots': [
                {
                    'r': s.r,
                    'phi': float(s.phi),
                    'amplitude': s.amplitude,
                    'spot_type': s.spot_type,
                    'radius': s.radius,
                    'width_type': s.width_type,
                    'B_los': s.B_los,
                    'B_perp': s.B_perp,
                    'chi': float(s.chi),
                    'velocity_shift': s.velocity_shift,
                }
                for s in self.spot_configs
            ]
        }
        
        # 写入JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        if self.verbose:
            print(f"[SpotConfig] Saved configuration: {output_file}")
        
        return str(output_file)

    def load_config_json(self, config_file: str) -> None:
        """
        从JSON文件加载配置

        参数:
        -------
        config_file : str
            JSON配置文件路径
        """
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # 恢复grid
        grid_params = config_dict.get('grid_params', {})
        geom_params = config_dict.get('geom_params', {})
        
        self.setup_grid(
            nr=grid_params.get('nr', 40),
            r_in=grid_params.get('r_in', 0.5),
            r_out=grid_params.get('r_out', 4.0),
            inclination_deg=geom_params.get('inclination_deg', 60.0),
            pOmega=geom_params.get('pOmega', -0.5),
            r0_rot=geom_params.get('r0_rot', 1.0),
            period_days=geom_params.get('period_days', 1.0))
        
        # 恢复spot
        spots = config_dict.get('spots', [])
        for s in spots:
            self.add_spot(**s)
        
        if self.verbose:
            print(f"[SpotConfig] Loaded configuration from {config_file}")

    def get_summary(self) -> str:
        """获取配置摘要"""
        summary = "Spot Simulation Configuration:\n"
        summary += f"  Grid: nr={self.grid_params.get('nr')}, "
        summary += f"r=[{self.grid_params.get('r_in')}, {self.grid_params.get('r_out')}]\n"
        summary += f"  Geometry: i={self.geom_params.get('inclination_deg')}°, "
        summary += f"pOmega={self.geom_params.get('pOmega')}, "
        summary += f"r0={self.geom_params.get('r0_rot')}R*, "
        summary += f"P={self.geom_params.get('period_days')}d\n"
        summary += f"  Spots: {len(self.spot_configs)}\n"
        for i, spot in enumerate(self.spot_configs):
            summary += f"    [{i}] r={spot.r:.2f}, phi={np.rad2deg(spot.phi):.1f}°, "
            summary += f"amp={spot.amplitude:.2f}, B_los={spot.B_los:.1f}G\n"
        return summary


# ============================================================================
# 便捷函数
# ============================================================================


def create_example_spot_config() -> SpotSimulationConfig:
    """创建示例配置"""
    config = SpotSimulationConfig(verbose=1)
    
    # 设置grid和几何参数
    config.setup_grid(
        nr=40,
        r_in=0.5,
        r_out=4.0,
        inclination_deg=60.0,
        pOmega=-0.5,
        r0_rot=1.0,
        period_days=1.0)
    
    # 添加几个示例spot
    config.add_spot(r=1.5, phi=0.0, amplitude=2.0, B_los=1000.0)
    config.add_spot(r=2.0, phi=np.pi, amplitude=1.5, B_los=-500.0)
    config.add_spot(r=3.0, phi=np.pi/2, amplitude=-1.5, spot_type='absorption')
    
    return config
