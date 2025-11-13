#!/usr/bin/env python
"""测试 SpecIO 完整读写流程：生成模拟光谱→添加噪声→读取→输出"""

import numpy as np
import os
import sys

# 确保可以导入 core 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import core.SpecIO as SpecIO
from core.grid_tom import diskGrid
from core.velspace_DiskIntegrator import VelspaceDiskIntegrator
from core.local_linemodel_basic import LineData, GaussianZeemanWeakLineModel


def create_directories():
    """创建必要的目录结构"""
    dirs = ['input/inSpec', 'output/outSpec', 'output']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✓ 已创建/确认目录: {', '.join(dirs)}")


def create_mock_line_file(filepath='input/lines_test.txt'):
    """创建模拟的谱线参数文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("# Test line parameters\n")
        f.write("# wl0(nm) sigWl(nm) g\n")
        f.write("500.0  0.02  2.5\n")  # 最少需要 wl0, sigWl, g
    print(f"✓ 已创建谱线参数文件: {filepath}")
    return filepath


def generate_synthetic_profiles(n_phases=5, verbose=True):
    """生成多个相位的合成光谱（无噪声，作为"真实模型"）
    
    Parameters
    ----------
    n_phases : int
        要生成的相位数
    verbose : bool
        是否打印详细信息
        
    Returns
    -------
    profiles : list of dict
        每个dict包含 {'phase', 'v', 'I', 'V', 'wl0'}
    """
    if verbose:
        print(f"\n=== 步骤1：生成 {n_phases} 个相位的合成光谱 ===")

    # 创建谱线模型
    line_file = create_mock_line_file()
    lineData = LineData(line_file)
    line_model = GaussianZeemanWeakLineModel(lineData,
                                             k_QU=1.0,
                                             enable_V=True,
                                             enable_QU=False)

    # 创建简单的盘网格
    nr = 30
    r_out = 1.0
    grid = diskGrid(nr=nr, r_in=0.0, r_out=r_out, verbose=0)

    # 速度网格（LSD风格）
    v_grid = np.linspace(-150, 150, 200)

    profiles = []

    for i in range(n_phases):
        phase = i / n_phases

        # 创建几何（简单刚体转动 + 倾角）
        class SimpleGeom:

            def __init__(self, grid, inc_deg, phase):
                self.grid = grid
                self.area_proj = grid.area * np.cos(np.deg2rad(inc_deg))
                self.inclination_rad = np.deg2rad(inc_deg)
                self.phi0 = 2 * np.pi * phase  # 相位角
                self.pOmega = 0.0  # 刚体转动
                self.r0 = 1.0
                self.period = 1.0

                # 简单的磁场分布：一个径向斑点
                r = grid.r
                phi = grid.phi
                spot_r = 0.5
                spot_phi = 0.0
                dist = np.sqrt(
                    (r * np.cos(phi) - spot_r * np.cos(spot_phi))**2 +
                    (r * np.sin(phi) - spot_r * np.sin(spot_phi))**2)
                self.B_los = 1000.0 * np.exp(-dist**2 / 0.1**2)  # 高斯斑点，峰值1000G

        geom = SimpleGeom(grid, inc_deg=60.0, phase=phase)

        # 包装线模型为常数振幅
        class ConstAmpModel:

            def __init__(self, base_model, amp):
                self.base = base_model
                self.amp = amp

            def compute_local_profile(self,
                                      wl_grid,
                                      amp,
                                      Blos=None,
                                      Ic_weight=None,
                                      **kwargs):
                # 使用固定振幅，忽略传入的 amp 参数
                Npix = wl_grid.shape[1] if wl_grid.ndim > 1 else 1
                amp_fixed = np.full(Npix, self.amp)
                return self.base.compute_local_profile(wl_grid,
                                                       amp=amp_fixed,
                                                       Blos=Blos,
                                                       Ic_weight=Ic_weight,
                                                       **kwargs)

        wrapped_model = ConstAmpModel(line_model, amp=-0.3)

        # 计算合成谱
        inte = VelspaceDiskIntegrator(
            geom=geom,
            wl0_nm=lineData.wl0,
            v_grid=v_grid,
            line_model=wrapped_model,
            inst_fwhm_kms=5.0,
            normalize_continuum=True,
            disk_v0_kms=100.0,
            disk_power_index=0.0,
            disk_r0=r_out,
        )

        profiles.append({
            'phase': phase,
            'v': v_grid.copy(),
            'I': inte.I.copy(),
            'V': inte.V.copy(),
            'wl0': lineData.wl0
        })

        if verbose:
            print(
                f"  相位 {i} (φ={phase:.3f}): I范围 [{inte.I.min():.6f}, {inte.I.max():.6f}], "
                f"V范围 [{inte.V.min():.6e}, {inte.V.max():.6e}]")

    return profiles


def save_model_profiles(profiles, output_dir='output', verbose=True):
    """保存模型光谱（无噪声）到 output/"""
    if verbose:
        print(f"\n=== 步骤2：保存模型光谱到 {output_dir}/ ===")

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i, prof in enumerate(profiles):
        filepath = os.path.join(output_dir, f"model_phase_{i:02d}.lsd")
        header = {
            "phase": f"{prof['phase']:.6f}",
            "wl0_nm": f"{prof['wl0']:.3f}",
            "type": "synthetic_model"
        }
        SpecIO.write_model_spectrum(filepath,
                                    x=prof['v'],
                                    Iprof=prof['I'],
                                    V=prof['V'],
                                    fmt='lsd',
                                    header=header)
        paths.append(filepath)
        if verbose:
            print(f"  ✓ {filepath}")

    return paths


def add_noise_and_save_observations(profiles,
                                    snr_I=100,
                                    snr_V=1000,
                                    output_dir='input/inSpec',
                                    verbose=True):
    """为模型光谱添加噪声，保存为"观测"数据到 input/inSpec/"""
    if verbose:
        print(
            f"\n=== 步骤3：添加噪声并保存到 {output_dir}/ (SNR_I={snr_I}, SNR_V={snr_V}) ==="
        )

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i, prof in enumerate(profiles):
        # 添加高斯噪声
        I_depth = 1.0 - prof['I'].min()  # 谱线深度
        noise_I = (I_depth / snr_I) * np.random.randn(len(prof['I']))
        noise_V = (I_depth / snr_V) * np.random.randn(len(prof['V']))

        I_noisy = prof['I'] + noise_I
        V_noisy = prof['V'] + noise_V

        # 估计 sigma（作为观测误差棒）
        sigma_I = np.full_like(I_noisy, I_depth / snr_I)
        sigma_V = np.full_like(V_noisy, I_depth / snr_V)

        # 写入 LSD 格式（7列：RV, Int, sigma_int, Pol, sigma_pol, Null1, sigma_null1）
        filepath = os.path.join(output_dir, f"obs_phase_{i:02d}.lsd")
        with open(filepath, 'w') as f:
            f.write(
                f"# Synthetic observation with noise (SNR_I={snr_I}, SNR_V={snr_V})\n"
            )
            f.write(f"# phase: {prof['phase']:.6f}\n")
            f.write(f"# wl0_nm: {prof['wl0']:.3f}\n")
            f.write("# RV Int sigma_int Pol sigma_pol Null1 sigma_null1\n")
            for j in range(len(prof['v'])):
                null_val = 0.0
                sigma_null = sigma_V[j]
                f.write(
                    f"{prof['v'][j]:.3f} {I_noisy[j]:.8e} {sigma_I[j]:.3e} "
                    f"{V_noisy[j]:.8e} {sigma_V[j]:.3e} {null_val:.3e} {sigma_null:.3e}\n"
                )

        paths.append(filepath)
        if verbose:
            print(
                f"  ✓ {filepath} (噪声 RMS: I={noise_I.std():.6f}, V={noise_V.std():.6e})"
            )

    return paths


def test_read_observations(obs_files, verbose=True):
    """测试读取带噪声的"观测"文件"""
    if verbose:
        print(f"\n=== 步骤4：读取观测文件 (共{len(obs_files)}个) ===")

    obs_profiles = []
    for fpath in obs_files:
        obs = SpecIO.loadObsProfile(fpath, file_type='lsd_pol')
        if obs is not None:
            obs_profiles.append(obs)
            if verbose:
                print(f"  ✓ 读取 {os.path.basename(fpath)}: {len(obs.wl)} 个速度点, "
                      f"I范围 [{obs.specI.min():.6f}, {obs.specI.max():.6f}], "
                      f"V范围 [{obs.specV.min():.6e}, {obs.specV.max():.6e}]")
        else:
            print(f"  ✗ 读取失败: {fpath}")

    return obs_profiles


def test_write_output_spectra(obs_profiles,
                              output_dir='output/outSpec',
                              verbose=True):
    """测试将读取的观测写出到 output/outSpec/"""
    if verbose:
        print(f"\n=== 步骤5：将读取的观测写出到 {output_dir}/ ===")

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    for i, obs in enumerate(obs_profiles):
        filepath = os.path.join(output_dir, f"rewritten_obs_{i:02d}.lsd")
        header = {
            "source": "read_from_input_inSpec",
            "profile_type": obs.profile_type
        }
        SpecIO.write_model_spectrum(filepath,
                                    x=obs.wl,
                                    Iprof=obs.specI,
                                    V=obs.specV,
                                    sigmaI=obs.specIsig,
                                    fmt='lsd',
                                    header=header)
        output_paths.append(filepath)
        if verbose:
            print(f"  ✓ {filepath}")

    return output_paths


def export_geomodel_example(verbose=True):
    """演示导出几何模型"""
    if verbose:
        print(f"\n=== 步骤6：导出几何模型示例 ===")

    # 快速构造一个积分器
    line_file = 'input/lines_test.txt'
    if not os.path.exists(line_file):
        create_mock_line_file(line_file)

    lineData = LineData(line_file)
    line_model = GaussianZeemanWeakLineModel(lineData,
                                             enable_V=True,
                                             enable_QU=False)

    nr = 20
    grid = diskGrid(nr=nr, r_in=0.0, r_out=1.0, verbose=0)

    class SimpleGeom:

        def __init__(self, grid):
            self.grid = grid
            self.area_proj = grid.area
            self.inclination_rad = np.deg2rad(60.0)
            self.phi0 = 0.0
            self.pOmega = -0.5
            self.r0 = 1.0
            self.period = 1.2
            self.B_los = 500.0 * np.exp(-(
                (grid.r - 0.5)**2 + grid.phi**2) / 0.05)

    geom = SimpleGeom(grid)

    class ConstAmpModel:

        def __init__(self, base_model, amp):
            self.base = base_model
            self.amp = amp

        def compute_local_profile(self,
                                  wl_grid,
                                  amp,
                                  Blos=None,
                                  Ic_weight=None,
                                  **kwargs):
            Npix = wl_grid.shape[1] if wl_grid.ndim > 1 else 1
            amp_fixed = np.full(Npix, self.amp)
            return self.base.compute_local_profile(wl_grid,
                                                   amp=amp_fixed,
                                                   Blos=Blos,
                                                   Ic_weight=Ic_weight,
                                                   **kwargs)

    wrapped_model = ConstAmpModel(line_model, amp=-0.3)
    v_grid = np.linspace(-100, 100, 100)

    inte = VelspaceDiskIntegrator(
        geom=geom,
        wl0_nm=lineData.wl0,
        v_grid=v_grid,
        line_model=wrapped_model,
        inst_fwhm_kms=3.0,
        normalize_continuum=True,
        disk_v0_kms=80.0,
        disk_power_index=-0.5,
        disk_r0=1.0,
    )

    # 导出
    model_path = 'output/geomodel_test.tomog'
    meta = {
        "target": "TestStar",
        "period": 1.2,
        "phase": 0.0,
        "n_rings": nr,
    }
    inte.write_geomodel(model_path, meta=meta)
    if verbose:
        print(f"  ✓ 导出几何模型: {model_path}")

    # 读取验证
    geom_read, meta_read, table = VelspaceDiskIntegrator.read_geomodel(
        model_path)
    if verbose:
        print(f"  ✓ 读取验证: {geom_read.grid.numPoints} 像素, "
              f"目标={meta_read.get('target')}, 周期={meta_read.get('period')} 天")
        print(
            f"    B_los范围: [{table['Blos'].min():.2f}, {table['Blos'].max():.2f}] G"
        )


def main():
    """主测试流程"""
    print("=" * 70)
    print("SpecIO 完整读写流程测试")
    print("=" * 70)

    # 准备目录
    create_directories()

    # 1. 生成合成光谱
    profiles = generate_synthetic_profiles(n_phases=5, verbose=True)

    # 2. 保存模型（无噪声）
    model_paths = save_model_profiles(profiles,
                                      output_dir='output',
                                      verbose=True)

    # 3. 添加噪声并保存为"观测"
    obs_paths = add_noise_and_save_observations(profiles,
                                                snr_I=50,
                                                snr_V=500,
                                                output_dir='input/inSpec',
                                                verbose=True)

    # 4. 读取观测
    obs_profiles = test_read_observations(obs_paths, verbose=True)

    # 5. 写出到 output/outSpec
    output_paths = test_write_output_spectra(obs_profiles,
                                             output_dir='output/outSpec',
                                             verbose=True)

    # 6. 导出几何模型示例
    export_geomodel_example(verbose=True)

    # 总结
    print("\n" + "=" * 70)
    print("测试完成！文件生成总结：")
    print("=" * 70)
    print(f"✓ 模型光谱（无噪声）: {len(model_paths)} 个文件 → output/model_phase_*.lsd")
    print(f"✓ 观测光谱（带噪声）: {len(obs_paths)} 个文件 → input/inSpec/obs_phase_*.lsd")
    print(
        f"✓ 输出光谱（回写）  : {len(output_paths)} 个文件 → output/outSpec/rewritten_obs_*.lsd"
    )
    print(f"✓ 几何模型          : 1 个文件 → output/geomodel_test.tomog")
    print("\n可使用以下命令查看文件：")
    print("  ls -lh output/model_phase_*.lsd")
    print("  ls -lh input/inSpec/obs_phase_*.lsd")
    print("  ls -lh output/outSpec/rewritten_obs_*.lsd")
    print("  head -30 output/geomodel_test.tomog")
    print("=" * 70)


if __name__ == '__main__':
    main()
