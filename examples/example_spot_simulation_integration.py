#!/usr/bin/env python
"""
example_spot_simulation_integration.py

完整的spot模拟集成演示：
  1. 定义spot配置和网格参数
  2. 使用SpotSimulator生成.tomog模型
  3. 修改input/params_tomog.txt配置initTomogFile参数
  4. 运行pyzeetom/tomography.py进行0-iter正演，合成谱线

工作流程：
  模拟配置 → .tomog模型 → 参数配置 → 正演合成

优点：
  - 清晰的模块化流程
  - spot模拟与谱线合成解耦
  - 易于扩展和复现
"""

import sys
from pathlib import Path
import numpy as np
import shutil

# 添加项目根目录到路径
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils.generate_emission_spots_sim import SpotSimulationConfig
from core.mainFuncs import readParamsTomog
from pyzeetom.tomography import main


def create_spot_configuration() -> SpotSimulationConfig:
    """
    创建spot模拟配置
    
    返回:
    -------
    SpotSimulationConfig
        配置对象，包含grid和spot参数
    """
    print("=" * 80)
    print("STEP 1: Create Spot Simulation Configuration")
    print("=" * 80)

    # 创建配置对象
    config = SpotSimulationConfig(output_dir="output/simulation", verbose=1)

    # 设置grid和几何参数（需要与input/params_tomog.txt一致）
    config.setup_grid(
        nr=60,  # 与params_tomog.txt中的nRingsStellarGrid一致
        r_in=0.0,
        r_out=1.0,  # 物理半径（R_sun）
        inclination_deg=60.0,  # 与params_tomog.txt第0行一致
        pOmega=-0.05,  # 与params_tomog.txt第0行一致
        r0_rot=0.5,  # 恒星半径
        period_days=1.0)  # 与params_tomog.txt第0行一致

    # 添加spot（示例配置）
    print("\nAdding spots...")

    # Spot 1: 磁活跃区域（高磁场）
    config.add_spot(
        r=0.6,  # 盘面半径 (R_sun)
        phi=0.0,  # 初始方位角 (弧度)
        amplitude=2.0,  # 发射振幅
        spot_type='emission',
        radius=0.1,  # 高斯宽度
        B_los=1500.0,  # 视向磁场 (Gauss)
        B_perp=500.0,  # 垂直磁场
        chi=0.5)  # 磁场方位角 (弧度)

    # Spot 2: 另一个磁活跃区
    config.add_spot(
        r=0.7,
        phi=np.pi,  # 相对第一个spot 180度
        amplitude=1.5,
        spot_type='emission',
        radius=0.08,
        B_los=-1000.0,  # 反向磁场
        B_perp=400.0,
        chi=1.0)

    # Spot 3: 吸收区（黑子）
    config.add_spot(
        r=0.5,
        phi=np.pi / 2,
        amplitude=-1.0,  # 吸收
        spot_type='absorption',
        radius=0.12,
        B_los=2000.0,
        B_perp=800.0,
        chi=-0.5)

    print(config.get_summary())
    print()

    return config


def generate_tomog_model(config: SpotSimulationConfig,
                         phase: float = 0.0) -> str:
    """
    生成.tomog模型文件
    
    参数:
    -------
    config : SpotSimulationConfig
        spot配置
    phase : float
        演化相位 (0~1)
    
    返回:
    -------
    str
        生成的.tomog文件路径
    """
    print("=" * 80)
    print(f"STEP 2: Generate .tomog Model (phase={phase:.2f})")
    print("=" * 80)

    # 生成.tomog文件
    tomog_file = config.generate_tomog_model(
        output_file="output/simulation/spot_model_phase_0.00.tomog",
        phase=phase,
        meta={
            'simulation_type': 'spot_simulation',
            'timestamp': str(Path.cwd()),
        })

    # 保存配置为JSON（便于复现）
    config.save_config_json("output/simulation/spot_config.json")

    print(f"✓ Generated: {tomog_file}")
    print()

    return tomog_file


def configure_params_tomog(tomog_file: str) -> None:
    """
    修改input/params_tomog.txt以启用模型加载
    
    参数:
    -------
    tomog_file : str
        .tomog文件路径
    """
    print("=" * 80)
    print("STEP 3: Configure input/params_tomog.txt")
    print("=" * 80)

    # 读取参数文件
    params_path = Path("input/params_tomog.txt")
    with open(params_path, 'r') as f:
        lines = f.readlines()

    # 修改第#6行（0-indexed为行号对应的index）
    # 需要找到 #6 注释行，然后修改其后的实际参数行
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 查找 #6 initTomogFile 注释
        if line.strip().startswith("#6"):
            # 保留注释行
            new_lines.append(line)
            i += 1

            # 跳过其他注释行
            while i < len(lines) and (lines[i].strip().startswith("#")
                                      or lines[i].strip() == ""):
                new_lines.append(lines[i])
                i += 1

            # 修改实际参数行
            if i < len(lines):
                new_lines.append(f"1  {tomog_file}\n")
                i += 1
        else:
            new_lines.append(line)
            i += 1

    # 写入修改后的参数文件
    with open(params_path, 'w') as f:
        f.writelines(new_lines)

    print(f"✓ Modified input/params_tomog.txt:")
    print(f"  initTomogFile: 1")
    print(f"  initModelPath: {tomog_file}")
    print()


def run_forward_synthesis(verbose: int = 1) -> None:
    """
    运行0-iter正演模型合成谱线
    
    参数:
    -------
    verbose : int
        输出详细度
    """
    print("=" * 80)
    print("STEP 4: Run Forward Synthesis (0-iter)")
    print("=" * 80)
    print("This step uses pyzeetom/tomography.py to synthesize spectra")
    print("from the loaded .tomog model (0 iterations).")
    print()

    # 读取参数（会自动加载.tomog模型）
    par = readParamsTomog('input/params_tomog.txt', verbose=verbose)

    # 运行正演模型（不进行反演迭代）
    # 注意：这里run_mem=False，即仅进行0-iter正演
    try:
        results = main(par=par, verbose=verbose, run_mem=False)

        print()
        print(f"✓ Synthesis completed: {len(results)} phases")

        # 检查输出
        out_files = [
            Path("out_synth.txt"),
            Path("output/geomodel_phase0.tomog"),
        ]

        for f in out_files:
            if f.exists():
                print(f"✓ Generated output: {f}")

    except Exception as e:
        print(f"✗ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()

    print()


def restore_params_tomog() -> None:
    """恢复params_tomog.txt的原始状态（可选）"""
    print("=" * 80)
    print("OPTIONAL: Restore input/params_tomog.txt to Default")
    print("=" * 80)

    # 备份当前配置
    params_path = Path("input/params_tomog.txt")
    backup_path = Path("input/params_tomog_spot_sim_backup.txt")

    if not backup_path.exists() and params_path.exists():
        shutil.copy(params_path, backup_path)
        print(f"✓ Backed up to {backup_path}")

    # 恢复默认配置
    with open(params_path, 'r') as f:
        lines = f.readlines()

    # 恢复第#6行为默认状态
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("#6"):
            new_lines.append(line)
            i += 1

            while i < len(lines) and (lines[i].strip().startswith("#")
                                      or lines[i].strip() == ""):
                new_lines.append(lines[i])
                i += 1

            if i < len(lines):
                new_lines.append("0  none\n")
                i += 1
        else:
            new_lines.append(line)
            i += 1

    with open(params_path, 'w') as f:
        f.writelines(new_lines)

    print("✓ Restored initTomogFile: 0")
    print()


def main_workflow():
    """执行完整的工作流程"""
    print("\n")
    print("*" * 80)
    print("* pyZeeTom Spot Simulation Integration Workflow")
    print("*" * 80)
    print()

    try:
        # Step 1: 创建spot配置
        config = create_spot_configuration()

        # Step 2: 生成.tomog模型
        tomog_file = generate_tomog_model(config, phase=0.0)

        # Step 3: 修改参数文件
        configure_params_tomog(tomog_file)

        # Step 4: 运行正演合成谱线
        run_forward_synthesis(verbose=1)

        # Step 5（可选）: 恢复默认参数
        # restore_params_tomog()

        print("=" * 80)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("Summary:")
        print("  - Spot simulation: ✓ Complete")
        print("  - .tomog model generation: ✓ Complete")
        print("  - Parameter configuration: ✓ Complete")
        print("  - Forward synthesis: ✓ Complete")
        print()
        print("Output files:")
        print("  - Spot model: output/simulation/spot_model_phase_0.00.tomog")
        print("  - Configuration: output/simulation/spot_config.json")
        print("  - Synthetic spectrum: output/outModel/phase*.lsd")
        print("  - Geomodel: output/geomodel_phase0.tomog")
        print()

    except Exception as e:
        print(f"\n✗ WORKFLOW FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main_workflow()
    sys.exit(exit_code)
