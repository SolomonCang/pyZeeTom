#!/usr/bin/env python
"""Visualize geomodel.tomog files using core read_geomodel."""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.interpolate import griddata

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from core.velspace_DiskIntegrator import VelspaceDiskIntegrator


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


def plot_geomodel(geom,
                  meta,
                  table,
                  projection='polar',
                  out_fig=None,
                  vmax_blos=500.0,
                  vmax_bperp=500.0,
                  smooth=False,
                  grid_size=200):
    """
    Visualize geometry model with 3 panels.
    
    Parameters
    ----------
    smooth : bool
        是否使用插值平滑（散点 vs 网格插值）
    grid_size : int
        插值网格分辨率
    """
    r = table['r']
    phi = table['phi']
    Blos = table.get('Blos', np.zeros_like(r))
    Bperp = table.get('Bperp', np.zeros_like(r))

    # 从.tomog文件中读取亮度（amplitude）数据
    # 优先级：A（spot amplitude）> brightness > Ic_weight > 默认1.0
    if 'A' in table:
        brightness = table['A']
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

    # 动态设置亮度范围
    bright_min = np.min(brightness)
    bright_max = np.max(brightness)
    # 如果数据全为1.0（无spot影响），使用默认范围
    if np.allclose(bright_min, bright_max):
        bright_norm = TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=1.2)
    else:
        # 否则根据实际数据范围设置
        # vcenter设为1.0（无spot影响的基线值）
        # 如果范围很宽，自动调整
        vmin = min(bright_min, 1.0 - (bright_max - 1.0))
        vmax = max(bright_max, 1.0 + (1.0 - bright_min))
        bright_norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig = plt.figure(figsize=(18, 5))

    if projection == 'polar':
        if smooth:
            # 创建规则网格
            phi_grid = np.linspace(0, 2 * np.pi, grid_size)
            r_max = np.max(r)
            r_grid = np.linspace(0, r_max, grid_size)
            Phi_grid, R_grid = np.meshgrid(phi_grid, r_grid)

            # 插值
            points = np.column_stack([phi, r])
            Bright_grid = griddata(points,
                                   brightness, (Phi_grid, R_grid),
                                   method='cubic',
                                   fill_value=1.0)
            Blos_grid = griddata(points,
                                 Blos, (Phi_grid, R_grid),
                                 method='cubic',
                                 fill_value=0.0)
            Bperp_grid = griddata(points,
                                  Bperp, (Phi_grid, R_grid),
                                  method='cubic',
                                  fill_value=0.0)

            ax1 = fig.add_subplot(131, projection='polar')
            ax2 = fig.add_subplot(132, projection='polar')
            ax3 = fig.add_subplot(133, projection='polar')

            sc1 = ax1.pcolormesh(Phi_grid,
                                 R_grid,
                                 Bright_grid,
                                 cmap=bright_cmap,
                                 norm=bright_norm,
                                 shading='auto')
            ax1.set_title('Brightness (Absorption ← 1 → Emission)',
                          fontsize=11,
                          pad=20)
            plt.colorbar(sc1,
                         ax=ax1,
                         fraction=0.046,
                         pad=0.04,
                         label='Brightness')

            blos_norm = TwoSlopeNorm(vmin=-vmax_blos,
                                     vcenter=0,
                                     vmax=vmax_blos)
            sc2 = ax2.pcolormesh(Phi_grid,
                                 R_grid,
                                 Blos_grid,
                                 cmap='RdBu_r',
                                 norm=blos_norm,
                                 shading='auto')
            ax2.set_title('Blos (Line-of-Sight B-field)', fontsize=11, pad=20)
            plt.colorbar(sc2,
                         ax=ax2,
                         fraction=0.046,
                         pad=0.04,
                         label='Blos (G)')

            sc3 = ax3.pcolormesh(Phi_grid,
                                 R_grid,
                                 Bperp_grid,
                                 cmap='plasma',
                                 vmin=0,
                                 vmax=vmax_bperp,
                                 shading='auto')
            ax3.set_title('Bperp (Transverse B-field)', fontsize=11, pad=20)
            plt.colorbar(sc3,
                         ax=ax3,
                         fraction=0.046,
                         pad=0.04,
                         label='Bperp (G)')
        else:
            # 散点图模式
            ax1 = fig.add_subplot(131, projection='polar')
            ax2 = fig.add_subplot(132, projection='polar')
            ax3 = fig.add_subplot(133, projection='polar')

            sc1 = ax1.scatter(phi,
                              r,
                              c=brightness,
                              s=5,
                              cmap=bright_cmap,
                              norm=bright_norm)
            ax1.set_title('Brightness (Absorption ← 1 → Emission)',
                          fontsize=11,
                          pad=20)
            plt.colorbar(sc1,
                         ax=ax1,
                         fraction=0.046,
                         pad=0.04,
                         label='Brightness')

            blos_norm = TwoSlopeNorm(vmin=-vmax_blos,
                                     vcenter=0,
                                     vmax=vmax_blos)
            sc2 = ax2.scatter(phi,
                              r,
                              c=Blos,
                              s=5,
                              cmap='RdBu_r',
                              norm=blos_norm)
            ax2.set_title('Blos (Line-of-Sight B-field)', fontsize=11, pad=20)
            plt.colorbar(sc2,
                         ax=ax2,
                         fraction=0.046,
                         pad=0.04,
                         label='Blos (G)')

            sc3 = ax3.scatter(phi,
                              r,
                              c=Bperp,
                              s=5,
                              cmap='plasma',
                              vmin=0,
                              vmax=vmax_bperp)
            ax3.set_title('Bperp (Transverse B-field)', fontsize=11, pad=20)
            plt.colorbar(sc3,
                         ax=ax3,
                         fraction=0.046,
                         pad=0.04,
                         label='Bperp (G)')

    else:  # Cartesian
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        if smooth:
            # 创建规则网格
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            xi = np.linspace(x_min, x_max, grid_size)
            yi = np.linspace(y_min, y_max, grid_size)
            Xi, Yi = np.meshgrid(xi, yi)

            # 插值
            points = np.column_stack([x, y])
            Bright_grid = griddata(points,
                                   brightness, (Xi, Yi),
                                   method='cubic',
                                   fill_value=1.0)
            Blos_grid = griddata(points,
                                 Blos, (Xi, Yi),
                                 method='cubic',
                                 fill_value=0.0)
            Bperp_grid = griddata(points,
                                  Bperp, (Xi, Yi),
                                  method='cubic',
                                  fill_value=0.0)

            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

            sc1 = ax1.pcolormesh(Xi,
                                 Yi,
                                 Bright_grid,
                                 cmap=bright_cmap,
                                 norm=bright_norm,
                                 shading='auto')
            ax1.set_title('Brightness')
            ax1.set_xlabel('x (R*)')
            ax1.set_ylabel('y (R*)')
            ax1.set_aspect('equal')
            plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

            blos_norm = TwoSlopeNorm(vmin=-vmax_blos,
                                     vcenter=0,
                                     vmax=vmax_blos)
            sc2 = ax2.pcolormesh(Xi,
                                 Yi,
                                 Blos_grid,
                                 cmap='RdBu_r',
                                 norm=blos_norm,
                                 shading='auto')
            ax2.set_title('Blos (G)')
            ax2.set_xlabel('x (R*)')
            ax2.set_ylabel('y (R*)')
            ax2.set_aspect('equal')
            plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

            sc3 = ax3.pcolormesh(Xi,
                                 Yi,
                                 Bperp_grid,
                                 cmap='plasma',
                                 vmin=0,
                                 vmax=vmax_bperp,
                                 shading='auto')
            ax3.set_title('Bperp (G)')
            ax3.set_xlabel('x (R*)')
            ax3.set_ylabel('y (R*)')
            ax3.set_aspect('equal')
            plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)
        else:
            # 散点图模式
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

            sc1 = ax1.scatter(x,
                              y,
                              c=brightness,
                              s=5,
                              cmap=bright_cmap,
                              norm=bright_norm)
            ax1.set_title('Brightness')
            ax1.set_xlabel('x (R*)')
            ax1.set_ylabel('y (R*)')
            ax1.set_aspect('equal')
            plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

            blos_norm = TwoSlopeNorm(vmin=-vmax_blos,
                                     vcenter=0,
                                     vmax=vmax_blos)
            sc2 = ax2.scatter(x, y, c=Blos, s=5, cmap='RdBu_r', norm=blos_norm)
            ax2.set_title('Blos (G)')
            ax2.set_xlabel('x (R*)')
            ax2.set_ylabel('y (R*)')
            ax2.set_aspect('equal')
            plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

            sc3 = ax3.scatter(x,
                              y,
                              c=Bperp,
                              s=5,
                              cmap='plasma',
                              vmin=0,
                              vmax=vmax_bperp)
            ax3.set_title('Bperp (G)')
            ax3.set_xlabel('x (R*)')
            ax3.set_ylabel('y (R*)')
            ax3.set_aspect('equal')
            plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)

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
    parser = argparse.ArgumentParser(
        description="Visualize tomography geometric model")
    parser.add_argument(
        '--model',
        type=str,
        default='output/simulation/spot_model_phase_0.00.tomog',
        help='Model file path')
    parser.add_argument('--out',
                        type=str,
                        default=None,
                        help='Output figure filename')
    parser.add_argument('--projection',
                        type=str,
                        default='polar',
                        choices=['polar', 'cart'],
                        help='Projection type')
    parser.add_argument('--vmax_blos',
                        type=float,
                        default=500.0,
                        help='Blos colorbar range')
    parser.add_argument('--vmax_bperp',
                        type=float,
                        default=500.0,
                        help='Bperp colorbar max')
    parser.add_argument('--smooth',
                        action='store_true',
                        help='Use interpolation for smooth visualization')
    parser.add_argument('--grid-size',
                        type=int,
                        default=200,
                        help='Grid size for interpolation (default: 200)')
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: {args.model} not found")
        return 1

    print(f"Reading {args.model}...")
    try:
        geom, meta, table = VelspaceDiskIntegrator.read_geomodel(args.model)
        print(f"Loaded {len(table['r'])} pixels")
        mode = "smooth" if args.smooth else "scatter"
        print(f"Rendering ({mode} mode, projection={args.projection})...")
        plot_geomodel(geom,
                      meta,
                      table,
                      args.projection,
                      args.out,
                      args.vmax_blos,
                      args.vmax_bperp,
                      smooth=args.smooth,
                      grid_size=args.grid_size)
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
