"""
几何模型可视化工具
------------------
读取 geomodel.tomog 文件，可视化以下内容（3个子图横向排列）：
1. 亮度分布（Stokes I，brightness map）
2. 磁场视向分量 Blos（Stokes V）
3. 磁场横向分量 Bperp 或 Q/U 信息

用法示例：
    python visualize_geomodel.py --model output/geomodel.tomog --out model_viz.png
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from types import SimpleNamespace

# 添加项目路径
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "core"))

# ======= 用户可在此区直接设置参数 =======
MODEL_FILE = "/Users/tianqi/Documents/Codes_collection/ZDI_and/pyZeeTom/output/geomodel_test.tomog"  # 几何模型文件路径
OUT_FIG = None  # 输出图片文件名，如"model_viz.png"，None则直接显示
VMIN_BRIGHT = 0.5  # 亮度色标下限
VMAX_BRIGHT = 1.5  # 亮度色标上限
VMAX_BLOS = 500.0  # Blos色标范围（对称）
VMAX_BPERP = 500.0  # Bperp色标上限
CMAP_BRIGHT = "viridis"  # 亮度色标
CMAP_BLOS = "RdBu_r"  # Blos色标（红蓝对称）
CMAP_BPERP = "plasma"  # Bperp色标
PROJECTION = "polar"  # 投影方式: "polar" (极坐标) 或 "cart" (笛卡尔)
# ======================================


def parse_args():
    parser = argparse.ArgumentParser(description="几何模型可视化工具")
    parser.add_argument('--model',
                        type=str,
                        default=MODEL_FILE,
                        help='几何模型文件路径（geomodel.tomog）')
    parser.add_argument('--out',
                        type=str,
                        default=OUT_FIG,
                        help='输出图片文件名，不指定则直接显示')
    parser.add_argument('--projection',
                        type=str,
                        default=PROJECTION,
                        choices=['polar', 'cart'],
                        help='投影方式：polar (极坐标) 或 cart (笛卡尔)')
    return parser.parse_args()


def read_geomodel(filepath):
    """读取几何模型文件"""
    print(f"  读取文件: {filepath}")

    # 读取文件
    data = []
    inclination = None
    vsini = None
    phase = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                # 尝试从注释中提取元数据
                if 'Inclination:' in line:
                    inclination = float(line.split(':')[1].strip())
                elif 'Vsini:' in line:
                    vsini = float(line.split(':')[1].strip())
                elif 'Phase:' in line:
                    phase = float(line.split(':')[1].strip())
                continue
            # 数据行: r phi area brightness Blos Bperp chi
            parts = line.split()
            if len(parts) >= 7:
                data.append([float(x) for x in parts[:7]])

    # 转为数组并返回模型对象
    if not data:
        raise ValueError("模型数据为空，无法可视化")
    arr = np.array(data, dtype=float)
    r = arr[:, 0]
    phi = arr[:, 1]
    area = arr[:, 2]
    brightness = arr[:, 3]
    Blos = arr[:, 4]
    Bperp = arr[:, 5]
    chi = arr[:, 6]

    return SimpleNamespace(grid_r=r,
                           grid_phi=phi,
                           grid_area=area,
                           brightness=brightness,
                           Blos=Blos,
                           Bperp=Bperp,
                           chi=chi,
                           inclination=inclination,
                           vsini=vsini,
                           phase=phase)


def extract_model_fields(model):
    """从model中提取可视化所需的场"""
    r = model.grid_r
    phi = model.grid_phi
    area = model.grid_area
    brightness = getattr(model, 'brightness', np.ones_like(r))
    Blos = getattr(model, 'Blos', np.zeros_like(r))
    Bperp = getattr(model, 'Bperp', np.zeros_like(r))
    chi = getattr(model, 'chi', np.zeros_like(r))

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return {
        'r': r,
        'phi': phi,
        'x': x,
        'y': y,
        'area': area,
        'brightness': brightness,
        'Blos': Blos,
        'Bperp': Bperp,
        'chi': chi
    }


def plot_polar_map(ax,
                   r,
                   phi,
                   values,
                   vmin=None,
                   vmax=None,
                   cmap='viridis',
                   title='',
                   cbar_label='',
                   norm=None):
    """在极坐标系中绘制2D颜色图（按环逐段绘制，避免phi采样不一致导致的伪影）。"""
    r = np.asarray(r)
    phi = np.asarray(phi)
    values = np.asarray(values)

    # 唯一半径（环中心）
    r_unique = np.unique(r)
    r_unique.sort()
    # 径向边界
    if r_unique.size == 1:
        dr = r_unique[0] * 0.05 if r_unique[0] > 0 else 0.05
        r_edges = np.array([r_unique[0] - dr / 2, r_unique[0] + dr / 2])
    else:
        r_mid = 0.5 * (r_unique[:-1] + r_unique[1:])
        dr_in = r_mid[0] - r_unique[0]
        dr_out = r_unique[-1] - r_mid[-1]
        r_edges = np.concatenate([[r_unique[0] - dr_in], r_mid,
                                  [r_unique[-1] + dr_out]])

    last_mesh = None
    for i_ring, r_c in enumerate(r_unique):
        mask = np.isclose(r, r_c)
        if not np.any(mask):
            continue
        phi_ring = phi[mask]
        val_ring = values[mask]
        order = np.argsort(phi_ring)
        phi_ring = phi_ring[order]
        val_ring = val_ring[order]

        # phi 边界（周期封闭）
        if phi_ring.size > 1:
            phi_edges = np.empty(phi_ring.size + 1, dtype=float)
            phi_edges[1:-1] = 0.5 * (phi_ring[:-1] + phi_ring[1:])
            dphi_head = phi_ring[1] - phi_ring[0]
            dphi_tail = phi_ring[-1] - phi_ring[-2]
            phi_edges[0] = phi_ring[0] - dphi_head / 2
            phi_edges[-1] = phi_ring[-1] + dphi_tail / 2
        else:
            w = np.deg2rad(10.0)
            phi_edges = np.array([phi_ring[0] - w / 2, phi_ring[0] + w / 2])

        r0 = r_edges[i_ring]
        r1 = r_edges[i_ring + 1]
        PHI, R = np.meshgrid(phi_edges, np.array([r0, r1]))
        Z = val_ring[np.newaxis, :]
        if norm is None:
            last_mesh = ax.pcolormesh(PHI,
                                      R,
                                      Z,
                                      cmap=cmap,
                                      vmin=vmin,
                                      vmax=vmax,
                                      shading='flat')
        else:
            last_mesh = ax.pcolormesh(PHI,
                                      R,
                                      Z,
                                      cmap=cmap,
                                      norm=norm,
                                      shading='flat')

    # 将极坐标系统顺时针旋转180°（零角从北移到南）
    ax.set_theta_zero_location('S')
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=20, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if last_mesh is not None:
        cbar = plt.colorbar(last_mesh, ax=ax, pad=0.1, fraction=0.046)
        cbar.set_label(cbar_label, rotation=270, labelpad=20)
    return last_mesh


def plot_cart_map(ax,
                  x,
                  y,
                  values,
                  vmin=None,
                  vmax=None,
                  cmap='viridis',
                  title='',
                  cbar_label='',
                  norm=None):
    """在笛卡尔坐标系中绘制散点图"""
    # 将笛卡尔坐标顺时针旋转90°： (x', y') = (y, -x)
    x_rot = y
    y_rot = -x

    if norm is None:
        scatter = ax.scatter(x_rot,
                             y_rot,
                             c=values,
                             cmap=cmap,
                             vmin=vmin,
                             vmax=vmax,
                             s=30,
                             edgecolors='none',
                             alpha=0.8)
    else:
        scatter = ax.scatter(x_rot,
                             y_rot,
                             c=values,
                             cmap=cmap,
                             norm=norm,
                             s=30,
                             edgecolors='none',
                             alpha=0.8)

    ax.set_aspect('equal')
    ax.set_xlabel('x (rotated)', fontsize=10)
    ax.set_ylabel('y (rotated)', fontsize=10)
    ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 添加colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)

    return scatter


def visualize_model(model_file, projection='polar', out_file=None):
    """可视化几何模型的主函数"""
    print(f"读取几何模型: {model_file}")

    # 读取模型
    model = read_geomodel(model_file)
    fields = extract_model_fields(model)

    print(f"  网格点数: {len(fields['r'])}")
    print(f"  r 范围: [{fields['r'].min():.3f}, {fields['r'].max():.3f}]")
    print(
        f"  亮度范围: [{fields['brightness'].min():.3f}, {fields['brightness'].max():.3f}]"
    )
    print(
        f"  Blos 范围: [{fields['Blos'].min():.1f}, {fields['Blos'].max():.1f}] G"
    )
    print(
        f"  Bperp 范围: [{fields['Bperp'].min():.1f}, {fields['Bperp'].max():.1f}] G"
    )

    # 创建图形
    if projection == 'polar':
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, projection='polar')
        ax2 = fig.add_subplot(132, projection='polar')
        ax3 = fig.add_subplot(133, projection='polar')
        axes = [ax1, ax2, ax3]
        plot_func = plot_polar_map
        coords = (fields['r'], fields['phi'])
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plot_func = plot_cart_map
        coords = (fields['x'], fields['y'])

    # 子图1: 亮度分布（Stokes I）
    plot_func(axes[0],
              coords[0],
              coords[1],
              fields['brightness'],
              vmin=VMIN_BRIGHT,
              vmax=VMAX_BRIGHT,
              cmap=CMAP_BRIGHT,
              title='Brightness (Stokes I)',
              cbar_label='Relative Intensity')

    # 子图2: 磁场视向分量 Blos（Stokes V）
    norm_blos = TwoSlopeNorm(vmin=-VMAX_BLOS, vcenter=0, vmax=VMAX_BLOS)
    plot_func(axes[1],
              coords[0],
              coords[1],
              fields['Blos'],
              norm=norm_blos,
              cmap=CMAP_BLOS,
              title='Line-of-Sight B-field (Stokes V)',
              cbar_label='B$_{los}$ [G]')

    # 子图3: 磁场横向分量 Bperp（Stokes Q/U）
    plot_func(axes[2],
              coords[0],
              coords[1],
              fields['Bperp'],
              vmin=0,
              vmax=VMAX_BPERP,
              cmap=CMAP_BPERP,
              title='Perpendicular B-field (Stokes Q/U)',
              cbar_label='B$_{perp}$ [G]')

    plt.tight_layout()

    if out_file:
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        print(f"✓ 已保存可视化图片至 {out_file}")
    else:
        plt.show()

    return fig, axes


def main():
    args = parse_args()

    model_file = Path(args.model)
    if not model_file.exists():
        print(f"错误：模型文件不存在: {model_file}")
        sys.exit(1)

    try:
        visualize_model(model_file,
                        projection=args.projection,
                        out_file=args.out)
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
