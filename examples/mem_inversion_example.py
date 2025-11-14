"""
MEM反演完整示例 — pyZeeTom项目

演示如何使用新的两层MEM模块实现反演。

⚠️ 注意：这是一个**模板示例**，需要根据实际的VelspaceDiskIntegrator
接口进行调整。实际使用时需要确保响应矩阵的计算正确。
"""

import numpy as np
from typing import List

# 导入项目模块
from core.mem_tomography import (MEMTomographyAdapter, MagneticFieldParams,
                                 StokesObservation, SyntheticSpectrum)
from core.grid_tom import diskGrid


def create_mock_observations(n_phases: int = 4) -> List[StokesObservation]:
    """
    创建模拟观测数据。

    在实际应用中，应从文件读取。
    """
    observations = []
    wl = np.linspace(6549.8, 6550.2, 100)  # 模拟波长

    for phase in range(n_phases):
        # 模拟Stokes谱线（高斯轮廓）
        specI = 1.0 - 0.3 * np.exp(-((wl - 6550.0)**2) / 0.001)
        specV = 0.05 * (wl - 6550.0) * np.exp(-((wl - 6550.0)**2) / 0.001)
        specQ = 0.02 * np.sin(phase * 2 * np.pi / n_phases) * specV
        specU = 0.02 * np.cos(phase * 2 * np.pi / n_phases) * specV

        # 噪声
        noise_level = 0.01
        obs = StokesObservation(
            wl=wl,
            specI=specI + np.random.randn(len(wl)) * noise_level,
            specQ=specQ + np.random.randn(len(wl)) * noise_level,
            specU=specU + np.random.randn(len(wl)) * noise_level,
            specV=specV + np.random.randn(len(wl)) * noise_level,
            specI_sig=np.ones(len(wl)) * noise_level,
            specQ_sig=np.ones(len(wl)) * noise_level,
            specU_sig=np.ones(len(wl)) * noise_level,
            specV_sig=np.ones(len(wl)) * noise_level)
        observations.append(obs)

    return observations


def create_mock_synthetic_spectra(
        observations: List[StokesObservation],
        mag_field: MagneticFieldParams) -> List[SyntheticSpectrum]:
    """
    使用积分器生成合成谱线和导数。

    ⚠️ 这是模板，实际实现需要integrator支持导数计算。
    """
    synthetic_specs = []

    for obs in observations:
        # 调用积分器
        # 预期接口（需要根据实际VelspaceDiskIntegrator确认）:
        # result = integrator.integrate_with_derivatives(
        #     Blos=mag_field.Blos,
        #     Bperp=mag_field.Bperp,
        #     chi=mag_field.chi
        # )

        # 临时模拟：生成虚拟合成谱
        wl = obs.wl
        npix = len(mag_field.Blos)
        nlambda = len(wl)

        spec = SyntheticSpectrum(
            wl=wl,
            IIc=0.8 * np.ones(nlambda),  # 虚拟I
            QIc=np.zeros(nlambda),
            UIc=np.zeros(nlambda),
            VIc=0.01 * (wl - wl[50]) * np.exp(-((wl - wl[50])**2) / 0.001),
            dIc_dBlos=np.zeros((nlambda, npix)),
            dIc_dBperp=np.zeros((nlambda, npix)),
            dQc_dBlos=np.zeros((nlambda, npix)),
            dQc_dBperp=np.zeros((nlambda, npix)),
            dQc_dchi=np.zeros((nlambda, npix)),
            dUc_dBlos=np.zeros((nlambda, npix)),
            dUc_dBperp=np.zeros((nlambda, npix)),
            dUc_dchi=np.zeros((nlambda, npix)),
            dVc_dBlos=np.ones((nlambda, npix)) * 0.01,  # dV/dBlos
            dVc_dBperp=np.ones((nlambda, npix)) * 0.005,
            dVc_dchi=np.zeros((nlambda, npix)))
        synthetic_specs.append(spec)

    return synthetic_specs


def assemble_response_matrix(synthetic_specs: List[SyntheticSpectrum],
                             nparam_blos: int,
                             nparam_bperp: int,
                             nparam_chi: int,
                             fit_I: bool = True,
                             fit_V: bool = True) -> np.ndarray:
    """
    从合成谱的导数组装响应矩阵。

    响应矩阵形状：(ndata, nparam)
    其中 nparam = nparam_blos + nparam_bperp + nparam_chi
    """
    ndata_per_spec = len(synthetic_specs[0].wl)
    nspecs = len(synthetic_specs)

    ndata_I = ndata_per_spec * nspecs if fit_I else 0
    ndata_V = ndata_per_spec * nspecs if fit_V else 0
    ndata_total = ndata_I + ndata_V

    nparam = nparam_blos + nparam_bperp + nparam_chi

    Resp = np.zeros((ndata_total, nparam))

    data_idx = 0

    # I分量的导数（如果拟合）
    if fit_I:
        for spec in synthetic_specs:
            spec_data_idx = data_idx
            # dI/dBlos
            Resp[spec_data_idx:spec_data_idx + ndata_per_spec,
                 0:nparam_blos] = spec.dIc_dBlos
            # dI/dBperp
            Resp[spec_data_idx:spec_data_idx + ndata_per_spec,
                 nparam_blos:nparam_blos + nparam_bperp] = spec.dIc_dBperp
            data_idx += ndata_per_spec

    # V分量的导数（如果拟合）
    if fit_V:
        for spec in synthetic_specs:
            spec_data_idx = data_idx
            # dV/dBlos
            Resp[spec_data_idx:spec_data_idx + ndata_per_spec,
                 0:nparam_blos] = spec.dVc_dBlos
            # dV/dBperp
            Resp[spec_data_idx:spec_data_idx + ndata_per_spec,
                 nparam_blos:nparam_blos + nparam_bperp] = spec.dVc_dBperp
            # dV/dchi
            Resp[spec_data_idx:spec_data_idx + ndata_per_spec,
                 nparam_blos + nparam_bperp:] = spec.dVc_dchi
            data_idx += ndata_per_spec

    return Resp


def run_mem_inversion_example():
    """
    MEM反演的完整流程示例。
    """
    print("=" * 60)
    print("pyZeeTom MEM反演示例")
    print("=" * 60)

    # 1. 设置观测数据
    print("\n[1/6] 创建观测数据...")
    observations = create_mock_observations(n_phases=4)
    print(f"  → {len(observations)}个观测相位")

    # 2. 初始化网格和几何
    print("\n[2/6] 初始化网格...")
    grid = diskGrid(nr=15, r_in=0.5, r_out=3.0)
    npix = len(grid.r)
    print(f"  → 像素数: {npix}")

    # 3. 初始化物理参数
    print("\n[3/6] 初始化物理参数...")
    mag_field = MagneticFieldParams(Blos=np.zeros(npix),
                                    Bperp=np.ones(npix) * 0.5,
                                    chi=np.zeros(npix))
    print(
        f"  → Blos范围: [{mag_field.Blos.min():.3f}, {mag_field.Blos.max():.3f}]"
    )
    print(
        f"  → Bperp范围: [{mag_field.Bperp.min():.3f}, {mag_field.Bperp.max():.3f}]"
    )

    # 4. 初始化MEM适配器
    print("\n[4/6] 初始化MEM适配器...")
    adapter = MEMTomographyAdapter(fit_brightness=False,
                                   fit_magnetic=True,
                                   entropy_weights_blos=grid.area,
                                   entropy_weights_bperp=grid.area,
                                   entropy_weights_chi=grid.area,
                                   default_blos=0.1,
                                   default_bperp=0.5,
                                   default_chi=0.0)
    print("  ✓ 适配器已初始化")

    # 5. MEM迭代循环
    print("\n[5/6] 执行MEM迭代...")
    print("-" * 60)

    # 打包初始参数
    pack_result = adapter.pack_image_vector(mag_field=mag_field)
    Image = pack_result[0]
    n_bright = pack_result[1] if len(pack_result) > 1 else 0
    n_blos = pack_result[2] if len(pack_result) > 2 else 0
    n_bperp = pack_result[3] if len(pack_result) > 3 else 0
    n_chi = len(mag_field.chi) if adapter.fit_magnetic else 0

    max_iterations = 10
    convergence_threshold = 1e-3

    for iteration in range(max_iterations):
        # 生成合成谱线（这里使用模拟，实际使用积分器）
        synthetic_specs = create_mock_synthetic_spectra(
            observations, mag_field)

        # 打包数据和响应矩阵
        Data, Fmodel, sig2, Resp = adapter.pack_data_and_response(
            observations, synthetic_specs, fit_I=True, fit_V=True)

        if Resp.shape[1] == 0:
            # 如果响应矩阵未正确构建，手动构建
            Resp = assemble_response_matrix(synthetic_specs,
                                            n_blos,
                                            n_bperp,
                                            n_chi,
                                            fit_I=True,
                                            fit_V=True)

        # 执行MEM迭代
        entropy, chi2, test, Image = adapter.optimizer.iterate(
            Image=Image,
            Fmodel=Fmodel,
            Data=Data,
            sig2=sig2,
            Resp=Resp,
            weights=np.ones_like(Image),
            entropy_params={
                'npix': npix,
                'n_blos': n_blos,
                'n_bperp': n_bperp,
                'n_chi': n_chi
            },
            fixEntropy=0,
            targetAim=None)

        # 解包参数
        adapter.unpack_image_vector(Image,
                                    n_bright,
                                    n_blos,
                                    n_bperp,
                                    n_chi,
                                    mag_field=mag_field)

        # 输出进度
        print(f"  Iter {iteration:2d}: χ²={chi2:10.4f}  S={entropy:10.4f}  "
              f"test={test:.6f}")

        # 收敛检查
        if test < convergence_threshold:
            print(f"  ✓ 收敛！(test < {convergence_threshold})")
            break

    print("-" * 60)

    # 6. 总结结果
    print("\n[6/6] 反演结果...")
    print("\n最终磁场参数：")
    print("  Blos:")
    print(f"    平均值: {mag_field.Blos.mean():.4f} G")
    print(
        f"    范围: [{mag_field.Blos.min():.4f}, {mag_field.Blos.max():.4f}] G")
    print("  Bperp:")
    print(f"    平均值: {mag_field.Bperp.mean():.4f} G")
    print(
        f"    范围: [{mag_field.Bperp.min():.4f}, {mag_field.Bperp.max():.4f}] G"
    )
    print("  chi:")
    print(f"    平均值: {np.rad2deg(mag_field.chi.mean()):.2f}°")
    print(f"    范围: [{np.rad2deg(mag_field.chi.min()):.2f}°, "
          f"{np.rad2deg(mag_field.chi.max()):.2f}°]")

    print(f"\n最终χ²: {chi2:.4f}")
    print(f"最终熵: {entropy:.4f}")
    print(f"收敛指标: {test:.6f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_mem_inversion_example()
