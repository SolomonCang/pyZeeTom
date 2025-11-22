#!/usr/bin/env python
"""
Visualize geomodel.tomog files using core read_geomodel.
Modified to use top-level parameter configuration and Contour plots.
Fix: Compatible with Matplotlib 3.8+ (QuadContourSet.collections deprecation).
Feature: Bperp colormap changed to White -> Orange/Yellow -> Deep Red.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.interpolate import griddata

# ==============================================================================
# 参数空间 (PARAMETER SPACE)
# 在此处修改所有输入参数和配置
# ==============================================================================
PARAM_CONFIG = {
    # 输入文件路径
    # 'model_path': 'output/spot_forward/spot_model_phase_0p00.tomog',
    'model_path': 'output/spot_forward/geomodel_phase_00.tomog',

    # 输出文件路径 (设为 None 则直接显示窗口，设为 'filename.png' 则保存)
    'out_fig': None,

    # 投影方式: 'polar' (极坐标) 或 'cart' (笛卡尔坐标)
    'projection': 'polar',

    # 插值网格分辨率 (数值越大越精细，但在极坐标中心可能会有伪影)
    'grid_size': 200,

    # 等高线层级数量 (数值越大颜色过渡越平滑)
    'contour_levels': 100,

    # 视线方向磁场 (Blos) 的色标最大绝对值 (Gauss)
    'vmax_blos': 500.0,

    # 横向磁场 (Bperp) 的色标最大值 (Gauss)
    'vmax_bperp': 500.0,
}
# ==============================================================================

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

# 尝试导入核心模块
try:
    from core.disk_geometry_integrator import VelspaceDiskIntegrator
except ImportError:
    print("Warning: Could not import 'core.disk_geometry_integrator'.")

    # 定义假的读取器用于演示（如果核心模块缺失）
    class VelspaceDiskIntegrator:

        @staticmethod
        def read_geomodel(path):
            raise NotImplementedError("Core module missing.")


def create_brightness_colormap():
    """
    创建亮度色标：中心为白色（brightness=1），
    低于1为蓝色（吸收），高于1为红色（发射）
    """
    colors = [
        (0.0, 0.0, 1.0),  # 蓝色（吸收）
        (1.0, 1.0, 1.0),  # 白色（归一化）
        (1.0, 0.0, 0.0)  # 红色（发射）
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('brightness', colors, N=n_bins)
    return cmap


def create_bperp_colormap():
    """
    创建 Bperp 色标：白色 -> 橙黄色 -> 深色
    """
    # 定义颜色渐变列表
    # 您可以调整这里的颜色名称或十六进制代码来微调效果
    colors = [
        "white",  # 0.0: 起始为白色
        "#FFD700",  # 0.33: 金色 (Gold)
        "#FF8C00",  # 0.66: 深橙色 (DarkOrange)
        "#8B0000"  # 1.0: 深红色 (DarkRed)
    ]
    cmap = LinearSegmentedColormap.from_list('white_orange_deep',
                                             colors,
                                             N=256)
    return cmap


def set_contour_edge_color(cf, color="face"):
    """
    兼容性辅助函数：设置等高线填充的边缘颜色。
    用于消除等高线之间的细微白线。
    兼容 Matplotlib 旧版本 (.collections) 和新版本 (直接 set_edgecolor)。
    """
    # 新版本 Matplotlib (3.8+)
    if hasattr(cf, 'set_edgecolor'):
        try:
            cf.set_edgecolor(color)
            return
        except Exception:
            pass  # 如果失败，尝试旧方法

    # 旧版本 Matplotlib
    if hasattr(cf, 'collections'):
        for c in cf.collections:
            c.set_edgecolor(color)


def plot_geomodel_contour(geom, meta, table, config):
    """
    使用 Contourf (等高线填充) 绘制几何模型。
    """
    # 提取配置参数
    projection = config['projection']
    grid_size = config['grid_size']
    levels = config['contour_levels']
    vmax_blos = config['vmax_blos']
    vmax_bperp = config['vmax_bperp']
    out_fig = config['out_fig']

    # 提取数据
    r = table['r']
    phi = table['phi']
    Blos = table.get('Blos', np.zeros_like(r))
    Bperp = table.get('Bperp', np.zeros_like(r))

    # 处理亮度数据
    if 'A' in table:
        brightness = table['A']
    elif 'amp' in table:
        brightness = table['amp']
    elif 'brightness' in table:
        brightness = table['brightness']
    elif 'Ic_weight' in table:
        brightness = table['Ic_weight']
        if np.max(brightness) > 0:
            brightness = brightness / np.max(brightness)
    else:
        brightness = np.ones_like(r)

    # 创建色标
    bright_cmap = create_brightness_colormap()
    bperp_cmap = create_bperp_colormap()  # <--- 使用新的橙黄色标

    # 动态设置亮度范围 Norm
    bright_min = np.min(brightness)
    bright_max = np.max(brightness)
    if np.allclose(bright_min, bright_max):
        bright_norm = TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=1.2)
    else:
        vmin = min(bright_min, 1.0 - (bright_max - 1.0))
        vmax = max(bright_max, 1.0 + (1.0 - bright_min))
        bright_norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    # -------------------------------------------------------
    # 数据网格化 (Grid Interpolation)
    # -------------------------------------------------------
    if projection == 'polar':
        # 修复：处理 0/2pi 边界的循环插值问题
        # 将数据在 phi 方向复制一份，确保 griddata 能跨越边界插值

        # 复制 phi < pi/2 的点到 2pi 之后
        mask_low = phi < np.pi / 2
        phi_append_high = phi[mask_low] + 2 * np.pi
        r_append_high = r[mask_low]
        bright_append_high = brightness[mask_low]
        blos_append_high = Blos[mask_low]
        bperp_append_high = Bperp[mask_low]

        # 复制 phi > 3pi/2 的点到 0 之前
        mask_high = phi > 3 * np.pi / 2
        phi_append_low = phi[mask_high] - 2 * np.pi
        r_append_low = r[mask_high]
        bright_append_low = brightness[mask_high]
        blos_append_low = Blos[mask_high]
        bperp_append_low = Bperp[mask_high]

        # 合并数据
        phi_padded = np.concatenate([phi, phi_append_high, phi_append_low])
        r_padded = np.concatenate([r, r_append_high, r_append_low])
        bright_padded = np.concatenate(
            [brightness, bright_append_high, bright_append_low])
        blos_padded = np.concatenate([Blos, blos_append_high, blos_append_low])
        bperp_padded = np.concatenate(
            [Bperp, bperp_append_high, bperp_append_low])

        points_padded = np.column_stack([phi_padded, r_padded])

        # 极坐标网格
        phi_grid_1d = np.linspace(0, 2 * np.pi, grid_size)
        r_grid_1d = np.linspace(0, np.max(r), grid_size)
        X_grid, Y_grid = np.meshgrid(phi_grid_1d, r_grid_1d)

        # 使用 padded 数据进行插值
        print(
            "Interpolating data to grid for contour plotting (with cyclic padding)..."
        )
        Bright_grid = griddata(points_padded,
                               bright_padded, (X_grid, Y_grid),
                               method='cubic',
                               fill_value=1.0)
        Blos_grid = griddata(points_padded,
                             blos_padded, (X_grid, Y_grid),
                             method='cubic',
                             fill_value=0.0)
        Bperp_grid = griddata(points_padded,
                              bperp_padded, (X_grid, Y_grid),
                              method='cubic',
                              fill_value=0.0)
    else:
        # 笛卡尔坐标网格 (保持原样，因为转换到 x,y 后不存在边界断裂问题)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xi = np.linspace(np.min(x), np.max(x), grid_size)
        yi = np.linspace(np.min(y), np.max(y), grid_size)
        X_grid, Y_grid = np.meshgrid(xi, yi)
        points = np.column_stack([x, y])

        # 执行插值
        print("Interpolating data to grid for contour plotting...")
        Bright_grid = griddata(points,
                               brightness, (X_grid, Y_grid),
                               method='cubic',
                               fill_value=1.0)
        Blos_grid = griddata(points,
                             Blos, (X_grid, Y_grid),
                             method='cubic',
                             fill_value=0.0)
        Bperp_grid = griddata(points,
                              Bperp, (X_grid, Y_grid),
                              method='cubic',
                              fill_value=0.0)

    # -------------------------------------------------------
    # 绘图
    # -------------------------------------------------------
    fig = plt.figure(figsize=(18, 5))

    subplot_kw = {'projection': 'polar'} if projection == 'polar' else {}

    ax1 = fig.add_subplot(131, **subplot_kw)
    ax2 = fig.add_subplot(132, **subplot_kw)
    ax3 = fig.add_subplot(133, **subplot_kw)

    # 1. Brightness Plot
    cf1 = ax1.contourf(X_grid,
                       Y_grid,
                       Bright_grid,
                       levels=levels,
                       cmap=bright_cmap,
                       norm=bright_norm)
    set_contour_edge_color(cf1, "face")

    ax1.set_title('Brightness (Absorption ← 1 → Emission)',
                  fontsize=11,
                  pad=20)
    plt.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04, label='Brightness')

    # 2. Blos Plot
    blos_norm = TwoSlopeNorm(vmin=-vmax_blos, vcenter=0, vmax=vmax_blos)
    cf2 = ax2.contourf(X_grid,
                       Y_grid,
                       Blos_grid,
                       levels=levels,
                       cmap='RdBu_r',
                       norm=blos_norm)
    set_contour_edge_color(cf2, "face")

    ax2.set_title('Blos (Line-of-Sight B-field)', fontsize=11, pad=20)
    plt.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04, label='Blos (G)')

    # 3. Bperp Plot
    # 使用新的 bperp_cmap
    cf3 = ax3.contourf(X_grid,
                       Y_grid,
                       Bperp_grid,
                       levels=levels,
                       cmap=bperp_cmap,
                       vmin=0,
                       vmax=vmax_bperp)
    set_contour_edge_color(cf3, "face")

    ax3.set_title('Bperp (Transverse B-field)', fontsize=11, pad=20)
    plt.colorbar(cf3, ax=ax3, fraction=0.046, pad=0.04, label='Bperp (G)')

    # 设置坐标轴标签
    if projection == 'cart':
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('x (R*)')
            ax.set_ylabel('y (R*)')
            ax.set_aspect('equal')

    # 添加元数据信息
    info = []
    if 'iteration' in meta: info.append(f"iter={meta['iteration']}")
    if 'chi2' in meta: info.append(f"chi2={meta['chi2']:.2f}")
    if 'entropy' in meta: info.append(f"S={meta['entropy']:.4f}")
    if info:
        fig.text(0.5,
                 0.95,
                 " | ".join(info),
                 ha='center',
                 va='top',
                 fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if out_fig:
        plt.savefig(out_fig, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {out_fig}")
    else:
        plt.show()
    plt.close()


def main():
    model_file = PARAM_CONFIG['model_path']

    if not Path(model_file).exists():
        print(f"Error: {model_file} not found.")
        print("Generating dummy data for demonstration purposes...")
        # 生成假数据
        n_points = 5000
        r = np.sqrt(np.random.uniform(0, 1, n_points))
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        table = {
            'r': r,
            'phi': phi,
            'brightness': 1.0 + 0.5 * r * np.cos(3 * phi),
            'Blos': 400 * r * np.sin(phi),
            'Bperp': 300 * r
        }
        meta = {'iteration': 0, 'chi2': 1.23, 'entropy': 0.05}
        geom = None
        print("Rendering dummy data...")
        plot_geomodel_contour(geom, meta, table, PARAM_CONFIG)
        return 0

    print(f"Reading {model_file}...")
    try:
        geom, meta, table = VelspaceDiskIntegrator.read_geomodel(model_file)
        print(f"Loaded {len(table['r'])} pixels")
        print(
            f"Rendering (Contour mode, projection={PARAM_CONFIG['projection']})..."
        )

        plot_geomodel_contour(geom, meta, table, PARAM_CONFIG)

        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
