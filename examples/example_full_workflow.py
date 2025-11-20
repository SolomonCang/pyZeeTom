#!/usr/bin/env python3
"""
完整工作流示例：参数读取 → 模型加载 → 光谱生成 → 结果保存

演示如何使用 pyzeetom.tomography.main() 完成从参数文件读取、加载几何模型、
生成合成光谱、并保存到 output/outModelSpec 的完整工作流。

工作流步骤：
1. 读取参数文件 (input/params_tomog.txt)
2. 从参数中获取观测信息 (HJD, velR, polchannel)
3. 加载几何模型 (output/simulation/spot_model_phase_0.00.tomog)
4. 生成合成光谱
5. 保存到 output/outModelSpec/，按观测格式组织
6. 验证输出结果

Usage:
    python examples/example_full_workflow.py
    python examples/example_full_workflow.py --params input/params_tomog.txt
    python examples/example_full_workflow.py --params input/params_tomog.txt --verbose 2
"""

import sys
from pathlib import Path
import argparse

# 添加项目根目录到路径
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pyzeetom.tomography as tomog
import core.mainFuncs as mf
import core.SpecIO as SpecIO


def print_section(title):
    """打印分隔符和标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_observation_info(par, obsSet):
    """打印观测参数信息"""
    print_section("观测参数信息")
    print(f"总观测数: {len(obsSet)}")
    print(f"参考时刻 (HJD0): {par.jDateRef:.6f}")
    print(f"自转周期: {par.period:.4f} 天")
    print(f"倾角: {par.inclination:.1f}°")
    print(f"速度范围: [{par.velStart:.1f}, {par.velEnd:.1f}] km/s")
    print(f"光谱分辨率: {par.spectralResolution:.0f}")

    print("\n观测列表:")
    print(
        f"  {'ID':>3} {'HJD':>11} {'Phase':>7} {'VelR':>8} {'PolCh':>5} {'File':>40}"
    )
    print("  " + "-" * 74)

    for i, obs in enumerate(obsSet):
        hjd = par.jDates[i] if i < len(par.jDates) else 0.0
        phase = par.phases[i] if hasattr(par, 'phases') and i < len(
            par.phases) else (i / len(obsSet))
        vel_r = par.velRs[i] if i < len(par.velRs) else 0.0
        pol_ch = str(par.polChannels[i]).upper() if i < len(
            par.polChannels) else 'V'
        fname = Path(par.fnames[i]).name if i < len(par.fnames) else "unknown"

        print(
            f"  {i:3d} {hjd:11.6f} {phase:7.4f} {vel_r:8.2f} {pol_ch:>5} {fname:>40}"
        )


def print_model_info(par):
    """打印几何模型信息"""
    print_section("几何模型信息")

    if hasattr(par, 'initTomogFile') and par.initTomogFile:
        model_path = getattr(par, 'initModelPath', 'unknown')
        print(f"模型文件: {model_path}")
        print("加载模型: 已启用")

        if Path(model_path).exists():
            size = Path(model_path).stat().st_size
            print(f"文件大小: {size / 1024:.1f} KB")
            print("文件状态: OK")
        else:
            print("文件状态: 不存在！")
    else:
        print("加载模型: 禁用（使用默认磁场参数）")


def print_output_structure():
    """打印输出目录结构"""
    print_section("输出目录结构")

    output_dirs = [
        ("output/outModel/", "LSD 格式模型光谱（用于兼容）"),
        ("output/outModelSpec/", "按观测格式的模型光谱（SPEC 或 LSD）"),
        ("output/", "其他输出文件"),
    ]

    print("生成的输出文件将存放在以下目录：\n")
    for dir_path, desc in output_dirs:
        print(f"  {dir_path:<25} {desc}")

    print("\n输出文件命名规则:")
    print("  output/outModelSpec/phase_XXXX_HJDYYY_VRZZ_CH.ext")
    print("    XXXX   = 观测索引 (0000-9999)")
    print("    YYY    = Heliocentric Julian Date (p=小数点)")
    print("    ZZ     = 径向速度修正 (km/s)")
    print("    CH     = 偏振通道 (I/V/Q/U)")
    print("    ext    = 文件扩展名 (.spec 或 .lsd)")


def list_output_files(output_dir="output/outModelSpec"):
    """列出生成的输出文件"""
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"输出目录不存在: {output_dir}")
        return []

    # 同时支持 phase*.spec/lsd（outModel 格式）和 phase_*.spec/lsd（outModelSpec 格式）
    spec_files = list(output_path.glob("phase_*.spec")) + list(
        output_path.glob("phase*.spec"))
    lsd_files = list(output_path.glob("phase_*.lsd")) + list(
        output_path.glob("phase*.lsd"))
    # 移除重复
    files = sorted(set(spec_files + lsd_files))

    print_section(f"生成的光谱文件 ({output_dir})")

    if not files:
        print("  （没有找到文件）")
        return []

    print(f"总计: {len(files)} 个文件\n")
    print(f"  {'文件名':>50} {'大小':>10}")
    print("  " + "-" * 62)

    total_size = 0
    for fpath in files:
        size = fpath.stat().st_size
        total_size += size
        print(f"  {fpath.name:>50} {size:>10} bytes")

    print("  " + "-" * 62)
    print(f"  {'总计':>50} {total_size:>10} bytes")

    return files


def verify_output_format(output_dir="output/outModelSpec"):
    """验证输出文件格式"""
    print_section("输出文件格式验证")

    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"输出目录不存在: {output_dir}")
        return

    # 同时支持两种格式
    spec_files = list(output_path.glob("phase_*.spec")) + list(
        output_path.glob("phase*.spec"))
    lsd_files = list(output_path.glob("phase_*.lsd")) + list(
        output_path.glob("phase*.lsd"))
    files = sorted(set(spec_files + lsd_files))

    if not files:
        print("没有找到输出文件")
        return

    # 检查第一个文件
    test_file = files[0]
    print(f"检查示例文件: {test_file.name}\n")

    try:
        # 读取文件头
        with open(test_file, 'r') as f:
            header_lines = []
            columns_line = None
            for line in f:
                if line.startswith('#'):
                    header_lines.append(line.strip())
                    if 'COLUMNS' in line or 'columns' in line.lower():
                        columns_line = line.strip()
                else:
                    break

        print("文件头信息:")
        for hline in header_lines[:5]:  # 显示前 5 行
            print(f"  {hline}")
        if len(header_lines) > 5:
            print(f"  ... ({len(header_lines) - 5} 行更多)")

        if columns_line:
            print(f"\n列信息: {columns_line}")

        # 读取数据行数
        data_lines = 0
        with open(test_file, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    data_lines += 1

        print(f"\n数据行数: {data_lines}")
        print("✓ 文件格式验证完成")

    except Exception as e:
        print(f"✗ 文件读取失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="pyZeeTom 完整工作流示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python examples/example_full_workflow.py
  python examples/example_full_workflow.py --params input/params_tomog.txt
  python examples/example_full_workflow.py --params input/params_tomog.txt --verbose 2
        """)

    parser.add_argument("--params",
                        "-p",
                        default="input/params_tomog.txt",
                        help="参数文件路径 (默认: input/params_tomog.txt)")
    parser.add_argument("--verbose",
                        "-v",
                        type=int,
                        default=1,
                        help="详细程度 (0=静默, 1=标准, 2=详细)")

    args = parser.parse_args()

    print_section("pyZeeTom 完整工作流示例")
    print(f"参数文件: {args.params}")
    print(f"详细程度: {args.verbose}")

    # ========================================================================
    # 1. 读取参数文件
    # ========================================================================
    print_section("步骤 1: 读取参数文件")

    try:
        par = mf.readParamsTomog(args.params)
        print(f"✓ 成功读取参数文件: {args.params}")
    except Exception as e:
        print(f"✗ 读取参数文件失败: {e}")
        return

    # ========================================================================
    # 2. 读取观测数据
    # ========================================================================
    print_section("步骤 2: 读取观测数据")

    try:
        file_type = getattr(par, 'obsFileType', 'auto')
        pol_channels = getattr(par, 'polChannels', None)
        if pol_channels is not None:
            pol_channels = list(pol_channels)

        obsSet = SpecIO.obsProfSetInRange(list(par.fnames),
                                          par.velStart,
                                          par.velEnd,
                                          par.velRs,
                                          file_type=file_type,
                                          pol_channels=pol_channels)
        print(f"✓ 成功读取 {len(obsSet)} 个观测数据")
    except Exception as e:
        print(f"✗ 读取观测数据失败: {e}")
        return

    # ========================================================================
    # 3. 打印详细信息
    # ========================================================================
    print_observation_info(par, obsSet)
    print_model_info(par)
    print_output_structure()

    # ========================================================================
    # 4. 运行主程序（生成合成光谱）
    # ========================================================================
    print_section("步骤 3: 生成合成光谱")

    try:
        results = tomog.main(par=par,
                             obsSet=obsSet,
                             verbose=args.verbose,
                             run_mem=False)
        print(f"✓ 成功生成 {len(results)} 相位的合成光谱")
    except Exception as e:
        print(f"✗ 生成光谱失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # 5. 验证输出结果
    # ========================================================================
    list_output_files("output/outModel")
    list_output_files("output/outModelSpec")
    verify_output_format("output/outModelSpec")

    # ========================================================================
    # 6. 工作流完成
    # ========================================================================
    print_section("工作流完成！")
    print("✓ 所有步骤已完成")
    print("\n生成的文件位置:")
    print("  • output/outModel/          : LSD 格式模型光谱")
    print("  • output/outModelSpec/      : 按观测格式的模型光谱")
    print("  • output/geomodel_phase0.tomog : 几何模型")
    print("  • output/outFitSummary.txt  : 拟合摘要")


if __name__ == "__main__":
    main()
