#!/usr/bin/env python
"""Dynamic spectrum visualization tool.

功能：
1. 自适应搜索 .lsd, .s, .spec 等格式的光谱文件。
2. 强制使用 SpecIO 读取光谱，支持全 Stokes 参数。
3. 支持两种绘图模式：
   - image: 动态谱（2D 颜色图）
   - stacked: 堆叠折线图（Waterfall plot），按相位排列
4. Stokes 显示逻辑：
   - stokes='I': 单图显示 Stokes I
   - stokes='V'/'Q'/'U': 双图显示，左侧 Stokes I，右侧 Stokes V/Q/U
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径 (假设脚本位于 tools/ 或类似子目录)
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

# 引入自定义模块
from utils.dynamic_spectrum import IrregularDynamicSpectrum
from core.mainFuncs import readParamsTomog, compute_phase_from_jd
from core.SpecIO import loadObsProfile, ObservationProfile

# ==============================================================================
#                                 参数配置区域
# ==============================================================================
CONFIG = {
    # --- 数据输入配置 ---
    'params_file':
    'input/params_tomog.txt',  # 参数文件路径
    'model_dir':
    'output/outModel',  # 光谱文件所在目录

    # 指定文件列表 (可选)
    # None: 自动搜索 (phase001.lsd, .s, .spec 等)
    'file_list': [
        'output/spot_forward/simuspec/phase_0000_HJD0p500_VRp0p00_I.spec',
        'output/spot_forward/simuspec/phase_0003_HJD1p250_VRp0p00_U.spec',
        'output/spot_forward/simuspec/phase_0002_HJD1p000_VRp0p00_Q.spec',
        'output/spot_forward/simuspec/phase_0001_HJD0p750_VRp0p00_V.spec'
    ],

    # 选择显示的 Stokes 参数
    # 'I': 只画 I
    # 'V', 'Q', 'U': 画两张图，左边 I，右边 V/Q/U
    'stokes':
    'V',
    'file_type':
    'auto',  # 传给 SpecIO 的文件类型提示

    # --- 绘图模式配置 ---
    # 'image':   绘制动态谱（颜色图/热力图）
    # 'stacked': 绘制堆叠折线图（所有谱线画在一张图上，按相位错开）
    'plot_mode':
    'stacked',

    # --- 输出配置 ---
    'out_file':
    None,  # 输出路径 (如 'plot.png')，None 则弹窗

    # --- 'image' 模式专用配置 ---
    'cmap':
    'RdBu_r',  # 颜色映射
    'vmin':
    0.98,  # Stokes I 的颜色下限
    'vmax':
    1.02,  # Stokes I 的颜色上限
    # 偏振分量的颜色范围通常需要更小
    'vmin_pol':
    -0.001,
    'vmax_pol':
    0.001,

    # --- 'stacked' 模式专用配置 ---
    # 缩放系数：控制谱线起伏的高度。
    # Stokes I 公式: Y = Phase + (Intensity - 1.0) * stack_scale
    # Stokes P 公式: Y = Phase + Intensity * stack_scale * pol_scale_mult
    'stack_scale':
    10.0,
    # 偏振分量的额外放大倍数 (V通常100, Q/U可能需要更大，如10000)
    'pol_scale_mult':
    10.0,
    'line_color':
    'black',  # 线条颜色
    'line_width':
    0.6,  # 线条宽度

    # --- 数据处理配置 ---
    'remove_baseline':
    False,  # 是否去除基线 (减去平均谱)
}
# ==============================================================================


def find_file_for_index(base_dir, index):
    """自适应搜索对应索引的光谱文件。"""
    base_dir = Path(base_dir)

    # 可能的前缀和数字格式组合
    stems = [
        f"phase{index:03d}",  # phase001
        f"phase_{index:03d}",  # phase_001
        f"spec{index:03d}",  # spec001
        f"spec_{index:03d}",  # spec_001
        f"{index:03d}",  # 001
        f"phase{index}",  # phase1
        f"spec{index}",  # spec1
    ]

    # 可能的扩展名
    extensions = ['.lsd', '.s', '.spec', '.dat', '.txt']

    for stem in stems:
        for ext in extensions:
            candidate = base_dir / (stem + ext)
            if candidate.exists():
                return candidate
            # 尝试大写后缀
            candidate_upper = base_dir / (stem + ext.upper())
            if candidate_upper.exists():
                return candidate_upper

    return None


def load_model_spectra(model_dir,
                       params_file,
                       file_list=None,
                       file_type='auto'):
    """
    加载光谱数据。
    
    Returns:
        times (np.array): 相位数组
        obs_list (list[ObservationProfile]): SpecIO 读取的光谱对象列表
    """
    model_dir = Path(model_dir)

    # 1. 读取参数文件获取相位信息
    if not Path(params_file).exists():
        raise FileNotFoundError(f"Params file not found: {params_file}")

    params = readParamsTomog(params_file, verbose=0)
    phases = compute_phase_from_jd(params.jDates, params.jDateRef,
                                   params.period)
    num_obs = params.numObs

    print(f"  Params info: {num_obs} observations")
    print(f"  Phase range: {phases[0]:.3f} - {phases[-1]:.3f}")

    # 2. 准备文件列表
    target_files = []
    if file_list is not None:
        limit = min(len(file_list), num_obs)
        for i in range(limit):
            target_files.append(Path(file_list[i]))
    else:
        print(f"  Auto-searching files in {model_dir} ...")
        for i in range(num_obs):
            f_path = find_file_for_index(model_dir, i)
            if f_path is None:
                print(
                    f"  Warning: Could not find file for index {i} (Phase {phases[i]:.3f})"
                )
            target_files.append(f_path)

    # 3. 使用 SpecIO 读取数据
    times = []
    obs_list = []

    for i, f_path in enumerate(target_files):
        if f_path is None: continue
        if not f_path.exists():
            print(f"  Warning: File does not exist: {f_path}")
            continue

        # 强制使用 SpecIO 读取
        try:
            # 注意：loadObsProfile 会自动处理 I, V, Q, U 列
            obs = loadObsProfile(str(f_path), file_type=file_type)
        except Exception as e:
            print(f"  Error loading {f_path.name}: {e}")
            obs = None

        if obs is not None:
            times.append(phases[i])
            obs_list.append(obs)
        else:
            print(f"  Warning: Failed to parse {f_path.name} with SpecIO.")

    if len(times) == 0:
        raise FileNotFoundError("No valid spectra loaded.")

    # 按相位排序
    order = np.argsort(times)
    sorted_times = np.array([times[i] for i in order])
    sorted_obs = [obs_list[i] for i in order]

    return sorted_times, sorted_obs


def get_stokes_data(obs_list, stokes_char):
    """从 obs_list 中提取特定的 Stokes 分量数组列表。"""
    data_list = []
    stokes_char = stokes_char.upper()

    for obs in obs_list:
        if stokes_char == 'I':
            data_list.append(obs.specI)
        elif stokes_char == 'V':
            data_list.append(
                obs.specV if obs.hasV else np.zeros_like(obs.specI))
        elif stokes_char == 'Q':
            data_list.append(
                obs.specQ if obs.hasQ else np.zeros_like(obs.specI))
        elif stokes_char == 'U':
            data_list.append(
                obs.specU if obs.hasU else np.zeros_like(obs.specI))
        else:
            # 默认回退到 I
            data_list.append(obs.specI)
    return data_list


def main():
    # 从 CONFIG 读取
    params_file = CONFIG['params_file']
    model_dir = CONFIG['model_dir']
    file_list = CONFIG['file_list']
    stokes_cfg = CONFIG['stokes'].upper()  # 用户配置的 Stokes ('I', 'V', 'Q', 'U')
    plot_mode = CONFIG['plot_mode']
    out_file = CONFIG['out_file']
    remove_baseline = CONFIG['remove_baseline']

    print("Initializing Spectrum Plotter...")

    try:
        times, obs_list = load_model_spectra(model_dir=model_dir,
                                             params_file=params_file,
                                             file_list=file_list,
                                             file_type=CONFIG['file_type'])
        print(f"✓ Successfully loaded {len(times)} spectra")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    # 提取波长/速度轴 (假设所有文件网格一致，取第一个)
    xs_list = [obs.wl for obs in obs_list]
    x_sample = xs_list[0]

    # 自动判断 X 轴标签
    if np.mean(x_sample) < 2000:
        xlabel = 'Velocity (km/s)' if np.max(
            np.abs(x_sample)) < 1000 else 'Wavelength (nm)'
    else:
        xlabel = 'Wavelength (Å)'

    # 准备数据：总是需要 I，如果 stokes_cfg 不是 I，还需要 Pol
    intensities_I = get_stokes_data(obs_list, 'I')
    intensities_Pol = None

    if stokes_cfg in ['V', 'Q', 'U']:
        intensities_Pol = get_stokes_data(obs_list, stokes_cfg)
        # Check signal strength and suggest scaling
        max_pol = np.max(np.abs(intensities_Pol)) if intensities_Pol else 0.0
        print(f"  Max {stokes_cfg} amplitude: {max_pol:.3e}")

        current_mult = CONFIG.get('pol_scale_mult', 100.0)
        if max_pol > 0 and max_pol * current_mult * CONFIG[
                'stack_scale'] < 0.005:
            suggested_mult = 0.1 / max_pol
            print(
                f"  Warning: {stokes_cfg} signal is very weak. Auto-adjusting pol_scale_mult to {suggested_mult:.1e} for visibility."
            )
            CONFIG['pol_scale_mult'] = suggested_mult

    # 去基线处理 (减去平均谱)
    if remove_baseline:
        print("Removing baseline (subtracting mean profile)...")
        # 处理 I
        mean_I = np.mean(intensities_I, axis=0)
        intensities_I = [spec - mean_I + 1.0
                         for spec in intensities_I]  # I 保持在 1.0 附近

        # 处理 Pol (如果存在)
        if intensities_Pol is not None:
            mean_Pol = np.mean(intensities_Pol, axis=0)
            intensities_Pol = [spec - mean_Pol
                               for spec in intensities_Pol]  # Pol 保持在 0.0 附近

    # ==========================================================================
    # 绘图初始化
    # ==========================================================================
    # 如果是 I，一张图；如果是 V/Q/U，两张图 (1行2列)
    if stokes_cfg == 'I':
        fig, ax_main = plt.subplots(figsize=(6, 10))
        axes = [ax_main]
        data_pairs = [('I', intensities_I)]
    else:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 10), sharey=True)
        axes = [ax_l, ax_r]
        data_pairs = [('I', intensities_I), (stokes_cfg, intensities_Pol)]

    scale = CONFIG['stack_scale']

    # ==========================================================================
    # 循环绘制每个子图 (左图 I, 右图 Pol)
    # ==========================================================================
    for ax_idx, ax in enumerate(axes):
        label, current_intensities = data_pairs[ax_idx]

        # --- 模式 1: Dynamic Spectrum (Image) ---
        if plot_mode == 'image':
            # 动态谱需要规则网格处理，这里调用工具类
            dynspec = IrregularDynamicSpectrum(times, xs_list,
                                               current_intensities)

            # 颜色范围区分
            if label == 'I':
                vmin, vmax = CONFIG['vmin'], CONFIG['vmax']
                cmap = CONFIG['cmap']
            else:
                vmin, vmax = CONFIG['vmin_pol'], CONFIG['vmax_pol']
                cmap = 'RdBu_r'  # 偏振通常用红蓝

            # 注意：IrregularDynamicSpectrum.plot 会创建新 figure，这里我们需要手动画在 ax 上
            # 我们直接使用 pcolormesh
            # 简单起见，假设网格一致，构建 2D 数组
            img_data = np.array(current_intensities)
            # img_data shape: (n_phases, n_pixels)

            # 构造网格
            X, Y = np.meshgrid(x_sample, times)

            im = ax.pcolormesh(X,
                               Y,
                               img_data,
                               cmap=cmap,
                               vmin=vmin,
                               vmax=vmax,
                               shading='auto')
            if ax_idx == 1 or len(axes) == 1:
                plt.colorbar(im, ax=ax, label='Intensity')

            ax.set_title(f'Dynamic Spectrum (Stokes {label})')

        # --- 模式 2: Stacked Lines (Waterfall) ---
        elif plot_mode == 'stacked':
            line_color = CONFIG['line_color']
            lw = CONFIG['line_width']

            for i in range(len(times)):
                phase = times[i]
                x = xs_list[i]
                y = current_intensities[i]

                # 核心公式
                if label == 'I':
                    # Stokes I 通常归一化为 1，减 1 后在 0 附近波动，然后叠加相位
                    y_plot = phase + (y - 1.0) * scale
                else:
                    # Stokes V/Q/U 通常在 0 附近波动，直接叠加
                    pol_mult = CONFIG.get('pol_scale_mult', 100.0)
                    y_plot = phase + (y) * pol_mult * scale

                ax.plot(x, y_plot, color=line_color, linewidth=lw, alpha=0.8)

            ax.set_title(f'Stacked (Stokes {label}, Scale={scale})')

            # 设置 Y 轴范围
            y_min = times[0] - 0.05
            y_max = times[-1] + 0.05
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_sample[0], x_sample[-1])

        # 公共轴标签
        ax.set_xlabel(xlabel)
        if ax_idx == 0:
            ax.set_ylabel('Rotation Phase')

    plt.tight_layout()

    # 输出或显示
    if out_file:
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved image to: {out_file}")
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
