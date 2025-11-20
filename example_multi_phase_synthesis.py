#!/usr/bin/env python3
"""
示例：多相位合成数据生成
=============================

利用 utils/spot_simulator.py 中增强的 SpotSimulator 类，
生成多相位、多偏振通道的合成光谱。

配置：
  - 3 个 spot：2 个发射，1 个吸收
  - 相位数组：[-0.3, 0, 0.1, 0.3, 0.4, 0.75, 0.85, 1.3]
  - 偏振通道：['Q', 'V', 'I', 'Q', 'U', 'I', 'V', 'Q']
  - 速度偏移：全零（0 km/s）
  - 盘参数：pOmega=-0.05, period=1.23d, r0=0.84R*, r_in=0, r_out=4, nr=40
  
说明：
  - 相位可以是任意实数（包括负数）
  - 负相位表示反向时间演化
  - 不同相位通过 pOmega 产生不同的 spot 位置
"""

import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.grid_tom import diskGrid
from utils.spot_simulator import (SpotSimulator, SpotConfig,
                                  create_simple_spot_simulator,
                                  add_noise_to_results)


def create_multi_phase_synthesis_example():
    """
    创建并执行多相位合成示例
    
    返回:
    -------
    dict
        包含以下键的字典：
        - 'simulator': SpotSimulator 对象
        - 'result_dict': generate_forward_model() 的返回值
        - 'phases': 相位数组
        - 'pol_channels': 偏振通道列表
    """

    print("=" * 80)
    print("多相位合成数据生成示例")
    print("=" * 80)

    # ========================================================================
    # 步骤 1: 配置网格和几何参数
    # ========================================================================
    print("\n[步骤 1] 配置网格和几何参数...")

    nr = 40
    r_in = 0.0
    r_out = 4.0
    inclination_deg = 60.0
    pOmega = -0.05
    r0_rot = 0.84
    period_days = 1.23

    print(f"  网格参数:")
    print(f"    nr = {nr}, r_in = {r_in}, r_out = {r_out}")
    print(f"  几何参数:")
    print(f"    inclination = {inclination_deg}°")
    print(f"    pOmega = {pOmega}")
    print(f"    r0_rot = {r0_rot} R*")
    print(f"    period = {period_days} days")

    # 创建网格
    grid = diskGrid(nr=nr, r_in=r_in, r_out=r_out, verbose=0)
    print(f"  ✓ 网格创建完成，像素数 = {grid.numPoints}")

    # 创建 SpotSimulator
    simulator = SpotSimulator(grid,
                              inclination_rad=np.deg2rad(inclination_deg),
                              phi0=0.0,
                              pOmega=pOmega,
                              r0_rot=r0_rot,
                              period_days=period_days)
    print("  ✓ SpotSimulator 创建完成")

    # ========================================================================
    # 步骤 2: 添加 3 个 spot（2 发射 + 1 吸收）
    # ========================================================================
    print("\n[步骤 2] 添加 3 个 spot...")

    # Spot 1: 发射，强度 2.0
    spot1 = SpotConfig(r=1.5,
                       phi=0.0,
                       amplitude=2.0,
                       spot_type='emission',
                       radius=0.3,
                       B_los=1200.0,
                       B_perp=500.0,
                       chi=0.0)
    simulator.add_spot(spot1)
    print(f"  ✓ Spot 1 (发射)  : r=1.5, phi=0°, amp=2.0, B_los=1200G")

    # Spot 2: 发射，强度 1.5
    spot2 = SpotConfig(r=2.5,
                       phi=np.pi,
                       amplitude=1.5,
                       spot_type='emission',
                       radius=0.25,
                       B_los=-800.0,
                       B_perp=300.0,
                       chi=np.pi / 4)
    simulator.add_spot(spot2)
    print(f"  ✓ Spot 2 (发射)  : r=2.5, phi=180°, amp=1.5, B_los=-800G")

    # Spot 3: 吸收，强度 -1.2
    spot3 = SpotConfig(r=3.5,
                       phi=np.pi / 2,
                       amplitude=-1.2,
                       spot_type='absorption',
                       radius=0.2,
                       B_los=600.0,
                       B_perp=200.0,
                       chi=np.pi / 2)
    simulator.add_spot(spot3)
    print(f"  ✓ Spot 3 (吸收)  : r=3.5, phi=90°, amp=-1.2, B_los=600G")

    # ========================================================================
    # 步骤 3: 配置多相位参数
    # ========================================================================
    print("\n[步骤 3] 配置多相位参数...")

    # 注：相位可以是任意实数（负数表示反向时间演化）
    # pOmega 会导致不同相位的 spot 位置完全不同
    phases = np.array([-0.3, 0.0, 0.1, 0.3, 0.4, 0.75, 0.85, 1.3])
    vel_shifts = np.zeros_like(phases)  # 所有速度偏移为 0
    pol_channels = ['Q', 'V', 'I', 'Q', 'U', 'I', 'V', 'Q']

    print(f"  相位数组 (长度={len(phases)}):")
    print("  (注：包括负相位，支持任意实数值)")
    for i, (phase, pol) in enumerate(zip(phases, pol_channels)):
        print(
            f"    [{i}] phase={phase:.2f}, pol_channel={pol}, vel_shift={vel_shifts[i]:.1f} km/s"
        )

    simulator.configure_multi_phase_synthesis(phases=phases,
                                              vel_shifts=vel_shifts,
                                              pol_channels=pol_channels)
    print(f"  ✓ 多相位参数配置完成")

    # ========================================================================
    # 步骤 4: 生成合成光谱（无噪声）
    # ========================================================================
    print("\n[步骤 4] 生成合成光谱...")

    result_dict = simulator.generate_forward_model(wl0_nm=656.3, verbose=2)

    results_clean = result_dict['results']
    print(f"  ✓ 生成 {len(results_clean)} 个合成模型")

    # ========================================================================
    # 步骤 5: 显示合成结果摘要
    # ========================================================================
    print("\n[步骤 5] 合成结果摘要...")
    print("-" * 80)

    for i, result in enumerate(results_clean):
        phase = phases[i]
        pol = pol_channels[i]
        vel_shift = vel_shifts[i]

        # 获取对应的 Stokes 参数
        if pol == 'I':
            stokes_data = result.stokes_i
            param_name = "Stokes I"
        elif pol == 'V':
            stokes_data = result.stokes_v
            param_name = "Stokes V"
        elif pol == 'Q':
            stokes_data = result.stokes_q
            param_name = "Stokes Q"
        elif pol == 'U':
            stokes_data = result.stokes_u
            param_name = "Stokes U"

        print(f"[相位 {i}]")
        print(
            f"  phase={phase:.2f}, pol_channel={pol}, vel_shift={vel_shift:.1f} km/s"
        )
        print(
            f"  {param_name}: range=[{stokes_data.min():.6f}, {stokes_data.max():.6f}]"
        )
        print(f"  shape={stokes_data.shape}, mean={stokes_data.mean():.6f}")
        print()

    # ========================================================================
    # 步骤 6: 可选 - 添加噪声
    # ========================================================================
    print("[步骤 6] 添加高斯噪声示例...")

    results_noisy = add_noise_to_results(
        results_clean,
        noise_type='gaussian',
        snr=100.0,  # 100 dB SNR
        seed=42)
    print(f"  ✓ 添加高斯噪声完成 (SNR=100 dB)")

    # 显示加噪后的结果
    print("\n  加噪后的 Stokes V 范围:")
    for i, result in enumerate(results_noisy):
        print(
            f"    [相位 {i}] V: [{result.stokes_v.min():.6f}, {result.stokes_v.max():.6f}]"
        )

    # ========================================================================
    # 步骤 7: 保存结果
    # ========================================================================
    print("\n[步骤 7] 保存结果...")

    output_dir = project_root / "output" / "multi_phase_synthesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存无噪声结果（NPZ 格式）
    output_file_clean = output_dir / "results_clean.npz"
    np.savez(output_file_clean,
             phases=phases,
             pol_channels=pol_channels,
             vel_shifts=vel_shifts,
             description="Clean synthetic spectra from multi-phase synthesis")
    print(f"  ✓ 无噪声结果保存: {output_file_clean}")

    # 保存加噪结果（NPZ 格式）
    output_file_noisy = output_dir / "results_noisy_gaussian_snr100.npz"
    np.savez(output_file_noisy,
             phases=phases,
             pol_channels=pol_channels,
             vel_shifts=vel_shifts,
             noise_type='gaussian',
             snr_db=100.0,
             description="Noisy synthetic spectra (Gaussian noise, SNR=100dB)")
    print(f"  ✓ 加噪结果保存: {output_file_noisy}")

    # 保存配置信息（文本格式）
    config_file = output_dir / "synthesis_config.txt"
    with open(config_file, 'w') as f:
        f.write("多相位合成配置\n")
        f.write("=" * 80 + "\n\n")

        f.write("网格参数:\n")
        f.write(f"  nr = {nr}\n")
        f.write(f"  r_in = {r_in}\n")
        f.write(f"  r_out = {r_out}\n\n")

        f.write("几何参数:\n")
        f.write(f"  inclination = {inclination_deg}°\n")
        f.write(f"  pOmega = {pOmega}\n")
        f.write(f"  r0_rot = {r0_rot} R*\n")
        f.write(f"  period = {period_days} days\n\n")

        f.write("Spot 配置:\n")
        f.write(f"  Spot 1: r=1.5, phi=0°, amp=2.0 (发射), B_los=1200G\n")
        f.write(f"  Spot 2: r=2.5, phi=180°, amp=1.5 (发射), B_los=-800G\n")
        f.write(f"  Spot 3: r=3.5, phi=90°, amp=-1.2 (吸收), B_los=600G\n\n")

        f.write("相位配置:\n")
        for i, (phase, pol,
                vel) in enumerate(zip(phases, pol_channels, vel_shifts)):
            f.write(
                f"  [{i}] phase={phase:.2f}, pol={pol}, vel_shift={vel:.1f} km/s\n"
            )

    print(f"  ✓ 配置信息保存: {config_file}")

    # ========================================================================
    # 完成
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ 多相位合成示例完成")
    print("=" * 80)

    return {
        'simulator': simulator,
        'result_dict': result_dict,
        'results_clean': results_clean,
        'results_noisy': results_noisy,
        'phases': phases,
        'pol_channels': pol_channels,
        'vel_shifts': vel_shifts,
        'output_dir': output_dir
    }


def main():
    """主函数"""
    result = create_multi_phase_synthesis_example()

    print("\n返回的数据结构:")
    print(f"  - simulator: {type(result['simulator'])}")
    print(
        f"  - results_clean: {type(result['results_clean'])}, 长度={len(result['results_clean'])}"
    )
    print(
        f"  - results_noisy: {type(result['results_noisy'])}, 长度={len(result['results_noisy'])}"
    )
    print(f"  - output_dir: {result['output_dir']}")


if __name__ == '__main__':
    main()
