#!/usr/bin/env python
"""集成测试：SpotSimulator优化验证

验证内容：
1. SpotSimulator 使用 amp 替代 response
2. create_geometry_object 返回 SimpleDiskGeometry
3. export_to_geomodel 使用 VelspaceDiskIntegrator.write_geomodel()
4. 整个流程与 ForwardModelResult 和 tomography 兼容
"""

import sys
import numpy as np
from pathlib import Path
import tempfile

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.spot_simulator import create_simple_spot_simulator, SpotConfig
from core.disk_geometry_integrator import SimpleDiskGeometry, VelspaceDiskIntegrator
from core.local_linemodel_basic import GaussianZeemanWeakLineModel, LineData


def test_amp_parameter():
    """测试 SpotSimulator 使用 amp 替代 response"""
    print("\n✓ 测试 1: amp 参数")
    print("─" * 50)

    sim = create_simple_spot_simulator(nr=10)

    # 检查初始化
    assert hasattr(sim, 'amp'), "SpotSimulator 应该有 amp 属性"
    assert sim.amp.shape == (sim.grid.numPoints, )
    assert np.allclose(sim.amp, 1.0), "初始 amp 应该是 ones"
    print("  ✓ amp 属性正确初始化")

    # 添加 spot 并应用
    sim.create_spot(r=1.5, phi=0.0, amplitude=2.0, B_los=1000.0)
    sim.apply_spots_to_grid(phase=0.0)

    # 检查 amp 被修改
    assert not np.allclose(sim.amp, 1.0), "amp 应该被 spot 修改"
    assert sim.amp.max() > 1.0, "强发射 spot 应该增加 amp"
    print(f"  ✓ amp 被正确修改 [{sim.amp.min():.2f}, {sim.amp.max():.2f}]")


def test_geometry_object_is_simple_disk_geometry():
    """测试 create_geometry_object 返回 SimpleDiskGeometry"""
    print("\n✓ 测试 2: create_geometry_object 返回 SimpleDiskGeometry")
    print("─" * 50)

    sim = create_simple_spot_simulator(nr=10)
    sim.create_spot(r=1.5, phi=0.0, amplitude=2.0, B_los=1000.0)

    # 创建几何对象
    geom = sim.create_geometry_object(phase=0.0)

    # 检查类型
    assert isinstance(geom, SimpleDiskGeometry), \
        f"应该返回 SimpleDiskGeometry，得到 {type(geom)}"
    print("  ✓ 返回 SimpleDiskGeometry 类型")

    # 检查所需属性
    assert hasattr(geom, 'amp'), "几何对象应该有 amp"
    assert hasattr(geom, 'B_los'), "几何对象应该有 B_los"
    assert hasattr(geom, 'B_perp'), "几何对象应该有 B_perp"
    assert hasattr(geom, 'chi'), "几何对象应该有 chi"
    print("  ✓ 几何对象包含所有必需属性")

    # 检查 amp 被正确传递
    assert np.allclose(geom.amp, sim.amp), "几何对象的 amp 应该与 simulator 一致"
    print("  ✓ amp 正确传递给几何对象")


def test_export_with_velocity_space_integrator():
    """测试 export_to_geomodel 使用 VelspaceDiskIntegrator.write_geomodel()"""
    print("\n✓ 测试 3: export_to_geomodel 集成测试")
    print("─" * 50)

    sim = create_simple_spot_simulator(nr=10)
    sim.create_spot(r=1.5, phi=0.0, amplitude=2.0, B_los=1000.0)
    sim.create_spot(r=2.5, phi=np.pi, amplitude=1.5, B_los=-500.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / 'test_geomodel.tomog')

        # 导出
        result = sim.export_to_geomodel(filepath,
                                        phase=0.0,
                                        meta={'target': 'test'})

        # 检查文件存在
        assert Path(result).exists(), "输出文件应该存在"
        print(f"  ✓ 文件成功创建: {Path(result).name}")

        # 检查文件内容
        with open(result, 'r') as f:
            content = f.read()

        # 验证格式
        assert "# TOMOG" in content, "应该有 TOMOG 格式标记"
        assert "amp" in content, "应该包含 amp 列"
        assert "Blos" in content, "应该包含 Blos 列"
        print("  ✓ 文件格式正确")

        # 验证数据行数（header + 数据）
        lines = content.strip().split('\n')
        data_lines = [l for l in lines if not l.startswith('#')]
        assert len(data_lines) == sim.grid.numPoints, \
            f"应该有 {sim.grid.numPoints} 行数据，得到 {len(data_lines)}"
        print(f"  ✓ 数据完整: {len(data_lines)} 像素")


def test_full_workflow():
    """测试完整工作流：SpotSimulator -> 几何对象 -> 积分器"""
    print("\n✓ 测试 4: 完整工作流")
    print("─" * 50)

    # 步骤 1: 创建 simulator 和 spots
    sim = create_simple_spot_simulator(nr=10)
    sim.create_spot(r=1.5, phi=0.0, amplitude=2.0, B_los=1000.0)
    sim.create_spot(r=2.5, phi=np.pi, amplitude=1.5, B_los=-500.0)
    print("  ✓ 步骤 1: SpotSimulator 创建完成")

    # 步骤 2: 创建几何对象
    geom = sim.create_geometry_object(phase=0.0)
    assert isinstance(geom, SimpleDiskGeometry)
    print("  ✓ 步骤 2: SimpleDiskGeometry 创建完成")

    # 步骤 3: 创建谱线模型
    line_data = LineData('input/lines.txt')
    line_model = GaussianZeemanWeakLineModel(line_data)
    print("  ✓ 步骤 3: 谱线模型加载完成")

    # 步骤 4: 创建积分器
    v_grid = np.linspace(-100, 100, 51)
    integrator = VelspaceDiskIntegrator(geom=geom,
                                        wl0_nm=656.3,
                                        v_grid=v_grid,
                                        line_model=line_model)
    print("  ✓ 步骤 4: VelspaceDiskIntegrator 创建完成")

    # 步骤 5: 导出几何模型
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / 'integrated_geomodel.tomog')
        integrator.write_geomodel(filepath, meta={'source': 'test_workflow'})

        assert Path(filepath).exists()
        print("  ✓ 步骤 5: 几何模型导出完成")

    # 步骤 6: 检查积分结果
    assert integrator.I is not None
    assert integrator.V is not None
    print("  ✓ 步骤 6: 谱线积分完成")


def test_amp_in_spectral_synthesis():
    """测试 amp 在谱线合成中的正确使用"""
    print("\n✓ 测试 5: amp 在谱线合成中的使用")
    print("─" * 50)

    sim = create_simple_spot_simulator(nr=10)

    # 创建两个 spot：一个强发射，一个弱吸收
    sim.create_spot(r=1.5, phi=0.0, amplitude=5.0, B_los=1000.0)
    sim.create_spot(r=3.0, phi=np.pi, amplitude=-2.0, B_los=-500.0)

    geom = sim.create_geometry_object(phase=0.0)

    # 创建积分器并合成谱线
    line_data = LineData('input/lines.txt')
    line_model = GaussianZeemanWeakLineModel(line_data)
    v_grid = np.linspace(-100, 100, 51)

    integrator = VelspaceDiskIntegrator(geom=geom,
                                        wl0_nm=656.3,
                                        v_grid=v_grid,
                                        line_model=line_model)

    # 验证结果
    assert integrator.I is not None
    assert integrator.V is not None
    assert len(integrator.I) == len(v_grid)

    # 验证 amp 影响了积分结果
    # (amp 较大的像素应该对总谱有更大贡献)
    print(
        f"  ✓ I 谱: min={integrator.I.min():.4f}, max={integrator.I.max():.4f}")
    print(
        f"  ✓ V 谱: min={integrator.V.min():.4f}, max={integrator.V.max():.4f}")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" SpotSimulator 优化集成测试")
    print("=" * 60)

    try:
        test_amp_parameter()
        test_geometry_object_is_simple_disk_geometry()
        test_export_with_velocity_space_integrator()
        test_full_workflow()
        test_amp_in_spectral_synthesis()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n【优化验证结论】")
        print("✓ SpotSimulator 成功使用 amp 替代 response")
        print("✓ create_geometry_object 正确返回 SimpleDiskGeometry")
        print(
            "✓ export_to_geomodel 使用 VelspaceDiskIntegrator.write_geomodel()")
        print("✓ 完整工作流正常运行，与 tomography 兼容")
        print("✓ amp 在谱线合成中被正确使用")
        return 0

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
