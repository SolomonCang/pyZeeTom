#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spot_forward_workflow.py

完整工作流：使用 SpotSimulator 生成几何模型，通过 forward_tomography 进行光谱合成。

工作流步骤：
1. 读取参考参数文件 (input/params_tomog.txt)
2. 创建 SpotSimulator 并生成多相位 .tomog 模型
3. 调整参数文件，使用生成的 .tomog 文件作为初始模型
4. 调用 forward_tomography() 进行光谱合成
5. 使用 SpecIO.write_model_spectrum() 保存合成的光谱

使用方式：
    python spot_forward_workflow.py [--phases PHASES] [--output DIR] [--verbose LEVEL]

示例：
    # 生成3个相位的模型和光谱
    python spot_forward_workflow.py --phases "0.0,0.25,0.5" --output output/spot_forward
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import core.mainFuncs as mf
import core.SpecIO as SpecIO
from core.grid_tom import diskGrid
from utils.spot_simulator import SpotSimulator, SpotConfig
from pyzeetom.tomography import forward_tomography

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def create_example_spots() -> List[SpotConfig]:
    """创建示例 spot 配置"""
    spots = [
        # 发射 spot 1：低纬度，强视向磁场
        SpotConfig(r=1.5,
                   phi=0.0,
                   amplitude=2.0,
                   spot_type='emission',
                   radius=0.3,
                   width_type='gaussian',
                   B_los=1500.0,
                   B_perp=200.0,
                   chi=np.deg2rad(45.0),
                   velocity_shift=0.0),
        # 发射 spot 2：中纬度，反向磁场
        SpotConfig(r=2.0,
                   phi=np.pi,
                   amplitude=1.5,
                   spot_type='emission',
                   radius=0.4,
                   width_type='gaussian',
                   B_los=-800.0,
                   B_perp=300.0,
                   chi=np.deg2rad(90.0),
                   velocity_shift=0.0),
        # 吸收 spot：高纬度
        SpotConfig(r=3.0,
                   phi=np.pi / 2,
                   amplitude=-1.0,
                   spot_type='absorption',
                   radius=0.35,
                   width_type='gaussian',
                   B_los=500.0,
                   B_perp=100.0,
                   chi=np.deg2rad(135.0),
                   velocity_shift=0.0),
    ]
    return spots


def generate_spot_tomog_models(phases: np.ndarray,
                               output_dir: str = './output/spot_forward',
                               nr: int = 40,
                               r_in: float = 0.5,
                               r_out: float = 4.0,
                               inclination_deg: float = 60.0,
                               pOmega: float = -0.05,
                               r0_rot: float = 1.0,
                               period_days: float = 1.0,
                               verbose: int = 1) -> Dict[str, Any]:
    """
    为多个相位生成 .tomog 模型文件
    
    返回值：
        dict 包含：
            - 'tomog_files': .tomog 文件路径列表
            - 'simulator': SpotSimulator 对象
            - 'phases': 相位数组
    """
    phases = np.atleast_1d(phases)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        sep = '=' * 70
        print(f"\n{sep}")
        print("[SpotForwardWorkflow] 步骤1：生成 Spot 模型文件 (.tomog)")
        print(sep)

    # 创建网格和 simulator
    grid = diskGrid(nr=nr, r_in=r_in, r_out=r_out, verbose=0)
    simulator = SpotSimulator(grid,
                              inclination_rad=np.deg2rad(inclination_deg),
                              phi0=0.0,
                              pOmega=pOmega,
                              r0_rot=r0_rot,
                              period_days=period_days)

    # 添加 spot
    example_spots = create_example_spots()
    simulator.add_spots(example_spots)

    if verbose:
        print(f"  Grid: nr={nr}, r=[{r_in}, {r_out}]")
        print(
            f"  Geometry: i={inclination_deg}°, pOmega={pOmega}, P={period_days}d"
        )
        print(f"  Spots: {len(example_spots)} spots added")

    # 为每个相位生成 .tomog 文件
    tomog_files = []
    for idx, phase in enumerate(phases):
        if verbose:
            print(
                f"\n  [{idx + 1}/{len(phases)}] Generating model for phase {phase:.3f}"
            )

        simulator.apply_spots_to_grid(phase=phase)

        # 生成文件名
        phase_str = f"{phase:.2f}".replace('.', 'p')
        tomog_file = output_path / f"spot_model_phase_{phase_str}.tomog"

        # 导出 .tomog 文件
        simulator.export_to_geomodel(filepath=str(tomog_file),
                                     phase=phase,
                                     meta={
                                         'source': 'spot_forward_workflow',
                                         'phase_index': idx
                                     })

        tomog_files.append(str(tomog_file))
        if verbose:
            file_size = tomog_file.stat().st_size / 1024
            print(f"    ✓ Output: {tomog_file.name} ({file_size:.1f} KB)")

    if verbose:
        print(f"\n  ✓ Generated {len(tomog_files)} .tomog files")

    return {
        'tomog_files': tomog_files,
        'simulator': simulator,
        'phases': phases,
    }


def create_dummy_observation_files(fnames: List[str],
                                   vel_start: float,
                                   vel_end: float,
                                   step: float = 2.0,
                                   verbose: int = 1):
    """创建虚拟观测文件，用于正演合成"""
    # 创建速度网格
    vel_grid = np.arange(vel_start, vel_end + step, step)
    n_points = len(vel_grid)

    # LSD pol format header
    header = f"{n_points} 7"

    if verbose:
        print(
            f"    Creating {len(fnames)} dummy observation files (vel=[{vel_start}, {vel_end}])"
        )

    for fname in fnames:
        path = Path(fname)
        # 总是覆盖，或者检查是否存在
        with open(path, 'w') as f:
            f.write(f"{header}\n")
            for v in vel_grid:
                # RV  Int  sigI  Pol  sigPol  Null1  Null2
                f.write(
                    f"{v:.4f}  1.0000  0.0010  0.0000  0.0001  0.0000  0.0000\n"
                )


def create_observation_file_list(phases: np.ndarray,
                                 output_dir: str,
                                 jdate_ref: float = 0.5,
                                 period_days: float = 1.0,
                                 verbose: int = 1) -> tuple:
    """
    为每个相位创建观测文件列表信息
    
    返回值：
        (fnames, jdates, velRs, polChannels)
    """
    phases = np.atleast_1d(phases)
    output_path = Path(output_dir)

    fnames = []
    jdates = []
    velRs = []
    polChannels = []

    pol_channel_list = ['I', 'V', 'Q', 'U']

    for idx, phase in enumerate(phases):
        # 文件名：output/spot_forward/obs_phase_XX.spec
        phase_str = f"{idx:02d}"
        filename = output_path / f"obs_phase_{phase_str}.spec"

        # 计算 HJD：HJD = HJD_ref + phase * period
        hjd = jdate_ref + phase * period_days

        # 速度偏移（RV）
        vel_r = 0.0

        # 偏振通道（循环）
        pol_channel = pol_channel_list[idx % len(pol_channel_list)]

        fnames.append(str(filename))
        jdates.append(hjd)
        velRs.append(vel_r)
        polChannels.append(pol_channel)

        if verbose > 1:
            print(
                f"    [Phase {idx}] file={filename.name}, HJD={hjd:.3f}, pol={pol_channel}"
            )

    return (fnames, np.array(jdates), np.array(velRs), polChannels)


def create_params_tomog_spotsimu(
        ref_param_file: str = 'input/params_tomog.txt',
        phases: np.ndarray = None,
        tomog_files: List[str] = None,
        output_dir: str = './output/spot_forward',
        output_param_file: str = 'input/params_tomog_spotsimu.txt',
        verbose: int = 1) -> str:
    """
    基于参考参数文件和生成的 .tomog 模型创建新参数文件
    
    工作流：
    1. 读取参考参数文件
    2. 调整参数（initTomogFile, initModelPath, 观测文件列表等）
    3. 写入新参数文件
    
    返回值：
        输出参数文件路径
    """
    if phases is None or len(phases) == 0:
        raise ValueError("phases 不能为空")
    if tomog_files is None or len(tomog_files) != len(phases):
        raise ValueError("tomog_files 数量必须与 phases 一致")

    if verbose:
        sep = '=' * 70
        print(f"\n{sep}")
        print("[SpotForwardWorkflow] 步骤2：创建调整的参数文件")
        print(sep)

    # 步骤1：读取参考参数文件
    if verbose:
        print(f"  Reading reference parameter file: {ref_param_file}")
    par = mf.readParamsTomog(ref_param_file, verbose=0)

    # 步骤2：为第一个相位启用 .tomog 模型初始化
    par.initTomogFile = 1
    par.initModelPath = tomog_files[0]

    # 新增：强制设置观测文件类型为 lsd_pol，因为我们生成的是 LSD 格式的虚拟文件
    par.obsFileType = 'lsd_pol'
    par.specType = 'lsd'

    if verbose:
        print(f"    ✓ initTomogFile = 1")
        print(f"    ✓ initModelPath = {tomog_files[0]}")
        print(f"    ✓ obsFileType = 'lsd_pol'")

    # 步骤3：创建观测文件列表
    fnames, jdates, velRs, polChannels = create_observation_file_list(
        phases,
        output_dir,
        jdate_ref=par.jDateRef,
        period_days=par.period,
        verbose=verbose)

    # 新增：创建虚拟观测文件
    create_dummy_observation_files(fnames,
                                   par.velStart,
                                   par.velEnd,
                                   verbose=verbose)

    # 步骤4：更新观测数据
    par.fnames = np.array(fnames)
    par.jDates = jdates
    par.velRs = velRs
    par.polChannels = polChannels
    par.numObs = len(phases)

    if verbose:
        print(f"  Observation entries updated:")
        print(f"    - numObs = {par.numObs}")
        print(f"    - Phases: {phases.tolist()}")
        print(f"    - Pol channels: {polChannels}")

    # 步骤5：写入新参数文件
    output_path = Path(output_param_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    par.write_params_file(str(output_path), verbose=0)

    if verbose:
        print(f"\n  ✓ Parameter file written: {output_param_file}")

    return str(output_path)


def generate_synthetic_spectra(param_file: str,
                               output_dir: str = './output/spot_forward',
                               verbose: int = 1) -> List[Any]:
    """
    使用 forward_tomography() 生成合成光谱
    
    返回值：
        ForwardModelResult 对象列表
    """
    if verbose:
        sep = '=' * 70
        print(f"\n{sep}")
        print("[SpotForwardWorkflow] 步骤3：使用 forward_tomography() 生成光谱")
        print(sep)
        print(f"  Parameter file: {param_file}")

    # 调用 forward_tomography
    results = forward_tomography(param_file=param_file,
                                 verbose=verbose,
                                 output_dir=output_dir)

    if verbose:
        print(f"\n  ✓ Forward synthesis complete: {len(results)} phases")

    return results


def save_synthetic_spectra(results: List[Any],
                           phases: np.ndarray,
                           pol_channels: List[str],
                           output_dir: str = './output/spot_forward',
                           verbose: int = 1) -> List[str]:
    """
    使用 SpecIO.write_model_spectrum() 保存合成光谱
    
    返回值：
        保存的文件路径列表
    """
    if verbose:
        sep = '=' * 70
        print(f"\n{sep}")
        print("[SpotForwardWorkflow] 步骤4：保存合成光谱")
        print(sep)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for idx, (result, phase,
              pol_channel) in enumerate(zip(results, phases, pol_channels)):

        # 生成输出文件名
        phase_str = f"{idx:02d}"
        output_file = output_path / f"syn_phase_{phase_str}_{pol_channel}.spec"

        if verbose > 1:
            print(
                f"  [{idx + 1}/{len(results)}] phase={phase:.3f}, pol={pol_channel}"
            )

        # 选择合适的 Stokes 参数
        if pol_channel == 'I':
            stokes_data = result.stokes_i
            file_type = 'spec_i'
        elif pol_channel == 'V':
            stokes_data = result.stokes_v
            file_type = 'spec_pol'
        elif pol_channel == 'Q':
            stokes_data = result.stokes_q
            file_type = 'spec_pol'
        elif pol_channel == 'U':
            stokes_data = result.stokes_u
            file_type = 'spec_pol'
        else:
            stokes_data = result.stokes_v
            file_type = 'spec_pol'

        # 使用 write_model_spectrum 保存
        SpecIO.write_model_spectrum(filename=str(output_file),
                                    x=result.wavelength,
                                    Iprof=result.stokes_i,
                                    V=result.stokes_v,
                                    Q=result.stokes_q,
                                    U=result.stokes_u,
                                    sigmaI=None,
                                    fmt='spec',
                                    pol_channel=pol_channel,
                                    file_type_hint=file_type,
                                    header={
                                        'phase': f"{phase:.3f}",
                                        'pol_channel': pol_channel,
                                        'source': 'spot_forward_workflow'
                                    })

        saved_files.append(str(output_file))

        if verbose > 1:
            file_size = output_file.stat().st_size / 1024
            print(f"      ✓ Saved: {output_file.name} ({file_size:.1f} KB)")

    if verbose:
        print(f"\n  ✓ Saved {len(saved_files)} synthetic spectra")

    return saved_files


def full_workflow(phases: np.ndarray = None,
                  output_dir: str = './output/spot_forward',
                  verbose: int = 1) -> Dict[str, Any]:
    """
    完整工作流：生成模型 -> 创建参数文件 -> 前向合成 -> 保存光谱
    
    参数：
    -------
    phases : np.ndarray
        相位数组，默认为 [0.0, 0.25, 0.5, 0.75]
    output_dir : str
        输出目录
    verbose : int
        详细程度
        
    返回值：
    -------
    dict 包含所有中间和最终结果
    """
    if phases is None:
        phases = np.array([0.0, 0.25, 0.5, 0.75])

    phases = np.atleast_1d(phases)

    # 步骤1：生成 .tomog 模型
    step1_result = generate_spot_tomog_models(phases=phases,
                                              output_dir=output_dir,
                                              verbose=verbose)

    # 步骤2：创建参数文件
    param_file = create_params_tomog_spotsimu(
        ref_param_file='input/params_tomog.txt',
        phases=phases,
        tomog_files=step1_result['tomog_files'],
        output_dir=output_dir,
        output_param_file=f"{output_dir}/params_tomog_spotsimu.txt",
        verbose=verbose)

    # 步骤3：生成光谱
    results = generate_synthetic_spectra(param_file=param_file,
                                         output_dir=output_dir,
                                         verbose=verbose)

    # 步骤4：保存光谱
    pol_channels = ['I', 'V', 'Q', 'U'][:len(phases)]
    saved_files = save_synthetic_spectra(results=results,
                                         phases=phases,
                                         pol_channels=pol_channels,
                                         output_dir=output_dir,
                                         verbose=verbose)

    if verbose:
        sep = '=' * 70
        print(f"\n{sep}")
        print("[SpotForwardWorkflow] ✅ 完整工作流完成")
        print(sep)
        print(f"Output directory: {output_dir}")
        print(f"Summary:")
        print(
            f"  - Generated {len(step1_result['tomog_files'])} .tomog models")
        print(f"  - Created parameter file: {param_file}")
        print(f"  - Synthesized {len(results)} spectra")
        print(f"  - Saved {len(saved_files)} spectrum files")
        print(sep)

    return {
        'phases': phases,
        'tomog_files': step1_result['tomog_files'],
        'param_file': param_file,
        'results': results,
        'saved_files': saved_files,
        'output_dir': output_dir
    }


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description=
        'Complete workflow: generate spot models and synthetic spectra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 4 phases with default parameters
  python spot_forward_workflow.py
  
  # Generate 3 custom phases
  python spot_forward_workflow.py --phases "0.0,0.3,0.7"
  
  # Verbose output
  python spot_forward_workflow.py --verbose 2
        """)

    parser.add_argument(
        '--phases',
        type=str,
        default='0.0,0.25,0.5,0.75',
        help='Comma-separated phase values (default: 0.0,0.25,0.5,0.75)')

    parser.add_argument(
        '--output',
        type=str,
        default='./output/spot_forward',
        help='Output directory (default: ./output/spot_forward)')

    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        choices=[0, 1, 2],
                        help='Verbosity level (default: 1)')

    args = parser.parse_args()

    # Parse phases
    try:
        phases = np.array([float(p.strip()) for p in args.phases.split(',')])
    except ValueError as e:
        print(f"Error parsing phases: {e}", file=sys.stderr)
        return 1

    # Run workflow
    try:
        result = full_workflow(phases=phases,
                               output_dir=args.output,
                               verbose=args.verbose)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose > 1:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
