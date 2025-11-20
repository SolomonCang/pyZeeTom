#!/usr/bin/env python
"""
example_visualize_spot_model.py

完整示例：生成 spot 模型并可视化包括 brightness 在内的所有参数

工作流程：
  1. 生成 spot 配置和 .tomog 模型
  2. 使用 visualize_geomodel.py 可视化 3 个面板：
     - Brightness: spot amplitude（发射/吸收）
     - Blos: 视向磁场
     - Bperp: 横向磁场
"""

import sys
from pathlib import Path
import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils.generate_emission_spots_sim import SpotSimulationConfig
from utils.visualize_geomodel import plot_geomodel
from core.disk_geometry_integrator import VelspaceDiskIntegrator

print("=" * 80)
print("WORKFLOW: Generate Spot Model and Visualize with Brightness Signal")
print("=" * 80)

# Step 1: 生成 spot 配置
print("\n[Step 1] Generating spot model...")
print("-" * 80)

config = SpotSimulationConfig(output_dir="output/simulation", verbose=0)
config.setup_grid(nr=60,
                  r_in=0.0,
                  r_out=1.0,
                  inclination_deg=60.0,
                  pOmega=-0.05,
                  r0_rot=0.5,
                  period_days=1.0)

# 添加 3 个 spot
config.add_spot(r=0.6,
                phi=0.0,
                amplitude=2.0,
                spot_type='emission',
                radius=0.1,
                B_los=1500.0,
                B_perp=500.0,
                chi=0.5)
config.add_spot(r=0.7,
                phi=np.pi,
                amplitude=1.5,
                spot_type='emission',
                radius=0.08,
                B_los=-1000.0,
                B_perp=400.0,
                chi=1.0)
config.add_spot(r=0.5,
                phi=np.pi / 2,
                amplitude=-1.0,
                spot_type='absorption',
                radius=0.12,
                B_los=2000.0,
                B_perp=800.0,
                chi=-0.5)

# 生成 .tomog 文件
tomog_file = config.generate_tomog_model(
    output_file="output/simulation/spot_model_phase_0.00.tomog", phase=0.0)
config.save_config_json("output/simulation/spot_config.json")

print(f"✓ Model file: {tomog_file}")

# Step 2: 可视化模型
print("\n[Step 2] Visualizing model with brightness signal...")
print("-" * 80)

geom, meta, table = VelspaceDiskIntegrator.read_geomodel(tomog_file)

output_fig = "test_output/spot_model_visualization.png"
Path(output_fig).parent.mkdir(parents=True, exist_ok=True)

plot_geomodel(geom,
              meta,
              table,
              projection='polar',
              out_fig=output_fig,
              smooth=False)

print(f"✓ Visualization saved: {output_fig}")

# Step 3: 信息总结
print("\n[Step 3] Summary")
print("=" * 80)

print("\nModel Data Summary:")
print(f"  Grid points: {len(table['r'])}")
print(f"  Radial range: [{min(table['r']):.3f}, {max(table['r']):.3f}]")

print("\nAmplitude (Brightness) - Spot Signature:")
A = table['A']
print(f"  Min: {min(A):.6f} (absorption)")
print(f"  Max: {max(A):.6f} (emission)")
print(f"  Mean: {np.mean(A):.6f}")

print("\nMagnetic Field Summary:")
print(f"  B_los: [{min(table['Blos']):.1f}, {max(table['Blos']):.1f}] G")
print(f"  B_perp: [{min(table['Bperp']):.1f}, {max(table['Bperp']):.1f}] G")

print("\n" + "=" * 80)
print("WORKFLOW COMPLETE!")
print("=" * 80)

print("\nVisualization panels (check " + output_fig + "):")
print("  1. Brightness (left):")
print("     - Red spots: emission regions (amplitude > 1.0)")
print("     - Blue spot: absorption region (amplitude < 1.0)")
print("     - White: unaffected regions (amplitude ≈ 1.0)")
print("  2. Blos (middle): line-of-sight magnetic field")
print("     - Red: positive B_los")
print("     - Blue: negative B_los")
print("  3. Bperp (right): transverse magnetic field")
print("     - Purple/yellow: field strength")

print("\nKey files generated:")
print(f"  - {tomog_file}")
print("  - output/simulation/spot_config.json")
print(f"  - {output_fig}")
