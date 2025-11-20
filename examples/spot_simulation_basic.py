#!/usr/bin/env python3
# examples/spot_simulation_basic.py
"""
Spot Simulation 库的基本示例

演示如何使用 SpotSimulator 和 SpotSimulationPipeline 库
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 添加项目根目录
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from utils.generate_emission_spots_sim import SpotSimulationPipeline

    print("\n" + "=" * 70)
    print("Spot Simulation Basic Example")
    print("=" * 70)

    # 创建管道
    pipeline = SpotSimulationPipeline(output_dir="./test_output/basic",
                                      verbose=1)

    # 设置grid和几何参数
    pipeline.setup_grid(nr=40,
                        r_in=0.5,
                        r_out=4.0,
                        inclination_deg=60.0,
                        pOmega=-0.5,
                        r0_rot=1.0,
                        period_days=1.0)

    # 设置谱线模型（Ha线）
    pipeline.setup_linemodel(wl0=656.3, sigWl=0.5, g=1.17)

    # 添加spot
    pipeline.add_emission_spot(r=1.5,
                               phi_deg=0.0,
                               amplitude=2.0,
                               B_los=1000.0,
                               radius=0.5)

    pipeline.add_emission_spot(r=2.5,
                               phi_deg=180.0,
                               amplitude=1.5,
                               B_los=-500.0,
                               radius=0.6)

    pipeline.add_absorption_spot(r=3.0,
                                 phi_deg=90.0,
                                 amplitude=-1.5,
                                 radius=0.4)

    pipeline.print_summary()

    # 合成光谱
    print("\nSynthesizing spectrum...")
    integrator = pipeline.synthesize_spectrum(phase=0.0,
                                              wl0_nm=656.3,
                                              v_range_kms=200.0,
                                              dv_kms=1.0,
                                              inst_fwhm_kms=0.5)

    print(
        f"Stokes I range: [{integrator.I.min():.4f}, {integrator.I.max():.4f}]"
    )
    print(
        f"Stokes V range: [{integrator.V.min():.4f}, {integrator.V.max():.4f}]"
    )

    # 导出geomodel和光谱
    geomodel_file = pipeline.export_geomodel(phase=0.0)
    spectrum_file = pipeline.export_spectrum(integrator, phase=0.0)

    print("\nOutput files:")
    print(f"  {geomodel_file}")
    print(f"  {spectrum_file}")

    pipeline.save_config()

    print("\n✓ Done!\n")
