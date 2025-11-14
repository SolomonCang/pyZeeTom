#!/usr/bin/env python
"""Dynamic spectrum visualization tool.

使用项目标准接口读取光谱数据：
- core.mainFuncs.readParamsTomog 读取参数文件，获取相位/JD信息
- core.SpecIO.loadObsProfile 读取观测光谱
- 支持模型光谱（从 output/outModel）和观测光谱的可视化
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from utils.dynamic_spectrum import IrregularDynamicSpectrum
from core.mainFuncs import readParamsTomog, compute_phase_from_jd
from core.SpecIO import loadObsProfile


def load_model_spectra(model_dir, params_file, stokes='I', file_type='auto'):
    """使用标准接口加载模型光谱。
    
    Args:
        model_dir: 模型光谱目录（如 output/outModel）
        params_file: 参数文件路径（如 input/params_tomog.txt）
        stokes: Stokes 参数 ('I', 'V', 'Q', 'U')
        file_type: 文件类型提示（'auto', 'lsd_i', 'spec_i', etc.）
    
    Returns:
        times: 相位数组
        xs: 速度/波长数组列表
        intensities: 光谱强度列表
    """
    model_dir = Path(model_dir)

    # 从参数文件读取相位信息
    params = readParamsTomog(params_file, verbose=0)
    phases = compute_phase_from_jd(params.jDates, params.jDateRef,
                                   params.period)

    print(f"  Loaded {params.numObs} observations from params")
    print(f"  JD range: {params.jDates[0]:.2f} - {params.jDates[-1]:.2f}")
    print(f"  Phase range: {phases[0]:.3f} - {phases[-1]:.3f}")

    times, xs, intensities = [], [], []

    # 根据参数文件中的观测数量查找对应模型文件
    for i in range(params.numObs):
        # 尝试多种命名模式
        phase_id = f"{int(i):03d}"
        candidates = [
            model_dir / f"phase{phase_id}.lsd",
            model_dir / f"phase{phase_id}.spec",
            model_dir / f"phase_{phase_id}.lsd",
            model_dir / f"phase_{phase_id}.spec",
        ]

        file_found = None
        for cand in candidates:
            if cand.exists():
                file_found = cand
                break

        if file_found is None:
            print(
                f"  Warning: model file not found for obs {i} (phase {phases[i]:.3f})"
            )
            continue

        # 尝试使用 SpecIO，如果失败则直接读取
        obs = None
        if file_type != 'direct':
            # 抑制SpecIO的错误输出
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                obs = loadObsProfile(str(file_found), file_type=file_type)

        if obs is None:
            # 直接读取（跳过注释行，兼容4/6列格式）
            try:
                data = np.loadtxt(str(file_found), comments='#')
                wl = data[:, 0]
                specI = data[:, 1] if data.shape[1] > 1 else np.ones_like(wl)
                specV = data[:, 2] if data.shape[1] > 2 else np.zeros_like(wl)
                specQ = data[:, 3] if data.shape[1] > 3 else np.zeros_like(wl)
                specU = data[:, 4] if data.shape[1] > 4 else np.zeros_like(wl)

                # 提取对应 Stokes 分量
                stokes_upper = stokes.upper()
                if stokes_upper == 'I':
                    spec = specI
                elif stokes_upper == 'V':
                    spec = specV
                elif stokes_upper == 'Q':
                    spec = specQ
                elif stokes_upper == 'U':
                    spec = specU
                else:
                    spec = specI

                times.append(phases[i])
                xs.append(wl)
                intensities.append(spec)
            except Exception as e:
                print(f"  Warning: failed to load {file_found}: {e}")
                continue
        else:
            # 使用 SpecIO 读取成功
            stokes_upper = stokes.upper()
            if stokes_upper == 'I':
                spec = obs.specI
            elif stokes_upper == 'V':
                spec = obs.specV if obs.hasV else np.zeros_like(obs.specI)
            elif stokes_upper == 'Q':
                spec = obs.specQ if obs.hasQ else np.zeros_like(obs.specI)
            elif stokes_upper == 'U':
                spec = obs.specU if obs.hasU else np.zeros_like(obs.specI)
            else:
                spec = obs.specI

            times.append(phases[i])
            xs.append(obs.wl)
            intensities.append(spec)

    if len(times) == 0:
        raise FileNotFoundError(f"No model spectra loaded from {model_dir}")

    order = np.argsort(times)
    return (np.array([times[i] for i in order]), [xs[i] for i in order],
            [intensities[i] for i in order])


def main():
    parser = argparse.ArgumentParser(description='动态光谱可视化工具（使用项目标准接口）')
    parser.add_argument('--params',
                        type=str,
                        required=True,
                        help='参数文件路径（如 input/params_tomog.txt）')
    parser.add_argument('--model-dir',
                        type=str,
                        default='output/outModel',
                        help='模型光谱目录')
    parser.add_argument('--stokes',
                        type=str,
                        default='I',
                        choices=['I', 'V', 'Q', 'U'],
                        help='Stokes 参数')
    parser.add_argument('--file-type',
                        type=str,
                        default='auto',
                        help='文件类型提示（auto/lsd_i/lsd_pol/spec_i等）')
    parser.add_argument('--out', type=str, default=None, help='输出图像路径')
    parser.add_argument('--cmap', type=str, default='RdBu_r', help='颜色映射')
    parser.add_argument('--vmin', type=float, default=None, help='颜色范围下限')
    parser.add_argument('--vmax', type=float, default=None, help='颜色范围上限')
    parser.add_argument('--remove-baseline', action='store_true', help='去除基线')
    args = parser.parse_args()

    print(f"Loading spectra from {args.model_dir}...")
    print(f"Using params: {args.params}")

    try:
        times, xs, intensities = load_model_spectra(args.model_dir,
                                                    args.params,
                                                    stokes=args.stokes,
                                                    file_type=args.file_type)
        print(f"✓ Loaded {len(times)} spectra, Stokes {args.stokes}")
    except Exception as e:
        print(f"✗ Error loading spectra: {e}")
        return 1

    # 创建动态谱对象
    dynspec = IrregularDynamicSpectrum(times, xs, intensities)

    if args.remove_baseline:
        print("Removing baseline...")
        dynspec.remove_baseline()

    # 自动设置 vmin/vmax（针对吸收线）
    vmin = args.vmin if args.vmin is not None else 0.95
    vmax = args.vmax if args.vmax is not None else 1.02

    fig, ax = dynspec.plot(cmap=args.cmap, vmin=vmin, vmax=vmax)
    ax.set_ylabel('Rotation Phase')
    ax.set_xlabel('Velocity (km/s)' if xs[0][0] < 1000 else 'Wavelength (Å)')
    ax.set_title(f'Dynamic Spectrum (Stokes {args.stokes})')

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {args.out}")
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
