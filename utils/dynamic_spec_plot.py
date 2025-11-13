"""
动态谱批量绘制工具
------------------
用法示例：
    python dynamic_spec_plot.py --spec_dir input/inSpec --param_file input/params_tomog.txt --out dynamic_spec.png

- 支持 LSD 格式（I 或 pol），自动识别文件夹下所有光谱文件
- 根据参数文件自动换算相位
- 依赖 Dynamic_spec.py 工具库
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ======= 用户可在此区直接设置参数 =======
SPEC_DIR = "input/inSpec"  # 光谱文件夹路径
PARAM_FILE = "input/params_tomog.txt"  # 参数文件路径
OUT_FIG = None  # 输出图片文件名，如"dynamic_spec.png"，None则直接显示
FILE_TYPE = "auto"  # 光谱文件类型：auto/lsd_i/lsd_pol
SORT_BY_PHASE = False  # 是否按相位排序
# ======================================

# 确保依赖库路径优先
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str('/Users/tianqi/Documents/Codes_collection/tinyTools'))
sys.path.insert(0, str(_root / "core"))
from Dynamic_spec import IrregularDynamicSpectrum, load_lsd_spectra
from mainFuncs import readParamsTomog


def parse_args():
    parser = argparse.ArgumentParser(description="批量光谱转动态谱工具")
    parser.add_argument('--spec_dir',
                        type=str,
                        default=SPEC_DIR,
                        help='光谱文件夹路径')
    parser.add_argument('--param_file',
                        type=str,
                        default=PARAM_FILE,
                        help='参数文件路径（如input/params_tomog.txt）')
    parser.add_argument('--out',
                        type=str,
                        default=OUT_FIG,
                        help='输出图片文件名（如dynamic_spec.png），不指定则直接显示')
    parser.add_argument('--file_type',
                        type=str,
                        default=FILE_TYPE,
                        help='光谱文件类型：auto/lsd_i/lsd_pol')
    parser.add_argument('--sort_by_phase',
                        action='store_true',
                        default=SORT_BY_PHASE,
                        help='按相位排序（默认按JD）')
    return parser.parse_args()


def get_phase_from_jd(jd, jd_ref, period):
    return (np.asarray(jd) - float(jd_ref)) / float(period)


def main():
    args = parse_args()
    spec_dir = Path(args.spec_dir)
    param_file = args.param_file
    file_type = args.file_type

    # 读取参数文件，获取JD、相位、文件名
    par = readParamsTomog(param_file)
    fnames = [Path(f) for f in par.fnames]
    jds = np.array(par.jDates)
    phases = np.array(par.phases)
    # jd_ref/period 已在par.phases中用到，无需单独变量

    # 收集文件夹下所有光谱文件
    all_files = sorted([
        p for p in spec_dir.iterdir()
        if p.is_file() and not p.name.startswith('.')
    ])
    # ==== 自动适配参数文件名与实际文件名 ==== #
    import re

    def normalize_name(name):
        """提取文件名中的关键字母和数字，忽略路径、扩展名和分隔符"""
        base = name.split('/')[-1]
        base = base.split('.')[0]  # 去扩展名
        # 提取字母和数字部分
        letters = re.sub(r'[^a-zA-Z]', '', base).lower()
        digits = re.sub(r'[^0-9]', '', base)
        return letters, digits

    def fuzzy_match(param_name, actual_names):
        """模糊匹配参数文件名与实际文件名"""
        p_letters, p_digits = normalize_name(param_name)
        best_match = None
        best_score = 0

        for actual in actual_names:
            a_letters, a_digits = normalize_name(actual)
            # 计算匹配分数：字母和数字分别匹配
            score = 0
            if p_letters in a_letters or a_letters in p_letters:
                score += 1
            if p_digits and a_digits:
                # 数字匹配：去除前导零后比较
                p_num = p_digits.lstrip('0') or '0'
                a_num = a_digits.lstrip('0') or '0'
                if p_num in a_num or a_num in p_num:
                    score += 2  # 数字匹配权重更高

            if score > best_score:
                best_score = score
                best_match = actual

        return best_match if best_score > 0 else None

    file_map = {f.name: f for f in all_files}
    actual_names = list(file_map.keys())
    files_sorted = []

    for f in fnames:
        # 1. 精确匹配
        if f.name in file_map:
            files_sorted.append(file_map[f.name])
        else:
            # 2. 模糊匹配
            matched = fuzzy_match(f.name, actual_names)
            if matched:
                files_sorted.append(file_map[matched])
                print(f"模糊匹配: {f.name} -> {matched}")
            else:
                print(f"警告：未找到与 {f.name} 匹配的光谱文件。")

    if not files_sorted:
        print(f"未在{spec_dir}找到参数文件中指定的光谱文件。")
        sys.exit(1)

    # 获取JD和相位
    times = jds
    if args.sort_by_phase and (phases is not None):
        times = phases

    # 读取光谱
    def time_parser(path, idx):
        # 按文件名在参数文件中的顺序取JD/phase
        name = Path(path).name
        # 允许模糊匹配
        norm = normalize_name(name)
        for i, f in enumerate(fnames):
            if name == f.name or normalize_name(f.name) == norm:
                return times[i]
        return idx

    # 读取所有光谱，获取I/V分量
    # 兼容 LSD I 或 LSD pol (I/V)
    _, _, _, meta = load_lsd_spectra(files_sorted,
                                     time_parser=time_parser,
                                     file_type_hint=file_type,
                                     return_metadata=True)
    times_out, xs, intensities = load_lsd_spectra(files_sorted,
                                                  time_parser=time_parser,
                                                  file_type_hint=file_type)

    # 判断是否有V分量
    has_V = False
    V_list = None
    if 'extras' in meta and isinstance(meta['extras'], dict):
        if 'Pol' in meta['extras'] and meta['extras']['Pol'] is not None:
            V_list = meta['extras']['Pol']
            # 需要确保 V_list 长度与 times_out 对齐
            try:
                has_V = len(V_list) == len(times_out)
            except Exception:
                has_V = False

    # 绘图
    if has_V:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # 计算时间步长
        if len(times_out) > 1:
            dt = np.diff(times_out)
            dt = np.append(dt, dt[-1])  # 最后一个用前一个的
        else:
            dt = np.array([0.1])

        # 计算 I 的全局范围，自动缩放；若跨越1.0，用TwoSlopeNorm以1为中心
        I_all = np.concatenate([np.asarray(v) for v in intensities])
        I_min, I_max = float(np.nanmin(I_all)), float(np.nanmax(I_all))
        norm_I = None
        vmin_I = I_min
        vmax_I = I_max
        if np.isfinite(I_min) and np.isfinite(I_max) and I_min < 1.0 < I_max:
            norm_I = TwoSlopeNorm(vmin=I_min, vcenter=1.0, vmax=I_max)

        # Stokes I
        for i, (t, x, I_val) in enumerate(zip(times_out, xs, intensities)):
            n = len(x)
            # 构造网格边界
            x_edges = np.linspace(x[0], x[-1], n + 1)
            y_edges = [t - dt[i] / 2, t + dt[i] / 2]
            X, Y = np.meshgrid(x_edges, y_edges)
            Z = I_val[np.newaxis, :]
            if norm_I is not None:
                mesh_I = axes[0].pcolormesh(X,
                                            Y,
                                            Z,
                                            cmap='viridis',
                                            shading='flat',
                                            norm=norm_I)
            else:
                mesh_I = axes[0].pcolormesh(X,
                                            Y,
                                            Z,
                                            cmap='viridis',
                                            shading='flat',
                                            vmin=vmin_I,
                                            vmax=vmax_I)

        axes[0].set_xlabel('Velocity (km/s)')
        axes[0].set_ylabel('Time/Phase')
        axes[0].set_title('Stokes I Dynamic Spectrum')
        axes[0].set_ylim(
            min(times_out) - dt[0] / 2,
            max(times_out) + dt[-1] / 2)
        # 添加 I 的 colorbar
        try:
            plt.colorbar(mesh_I, ax=axes[0], label='Stokes I')
        except Exception:
            pass

        # Stokes V
        mesh = None
        for i, (t, x, V_val) in enumerate(zip(times_out, xs, (V_list or []))):
            n = len(x)
            x_edges = np.linspace(x[0], x[-1], n + 1)
            y_edges = [t - dt[i] / 2, t + dt[i] / 2]
            X, Y = np.meshgrid(x_edges, y_edges)
            Z = V_val[np.newaxis, :]
            mesh = axes[1].pcolormesh(X,
                                      Y,
                                      Z,
                                      cmap='RdBu_r',
                                      shading='flat',
                                      vmin=-0.03,
                                      vmax=0.03)

        axes[1].set_xlabel('Velocity (km/s)')
        axes[1].set_ylabel('')
        axes[1].set_title('Stokes V Dynamic Spectrum')
        axes[1].set_ylim(
            min(times_out) - dt[0] / 2,
            max(times_out) + dt[-1] / 2)
        axes[1].set_yticklabels([])

        # 添加colorbar（仅当存在V绘制时）
        if mesh is not None:
            plt.colorbar(mesh, ax=axes[1], label='Stokes V')
        plt.tight_layout()

        if args.out:
            fig.savefig(args.out, dpi=200)
            print(f"已保存动态谱至 {args.out}")
        else:
            plt.show()
    else:
        # 仅I分量
        dynspec = IrregularDynamicSpectrum(times_out, xs, intensities)
        fig, ax = dynspec.plot(title="Dynamic Spectrum",
                               ylim=(min(times_out), max(times_out)))
        if args.out:
            fig.savefig(args.out, dpi=200)
            print(f"已保存动态谱至 {args.out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
