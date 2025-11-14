import os
import numpy as np
from core.grid_tom import diskGrid
from core.local_linemodel_basic import LineData, GaussianZeemanWeakLineModel
from core.spot_geometry import Spot, SpotCollection, TimeEvolvingSpotGeometry
from core.mainFuncs import readParamsTomog

# 参数
output_dir = 'input/inSpec'
os.makedirs(output_dir, exist_ok=True)

# 物理参数
n_spots = 10
r_min, r_max = 1.0, 3.0
phi_initial = 0.0
amp = 2.0  # 发射
spot_radius = 0.25
B_amp = 1000  # Gauss
inclination = 60.0
nr = 60
period = 1.0
pOmega = 0.0
vsini = 100.0
radius = 1.0
r_out = 3.0
SNR = 1000
n_phases = 5
phase_start, phase_end = 0.0, 1.5
phases = np.linspace(phase_start, phase_end, n_phases)
wavelength_domain = True  # 生成波长域(spec_pol)而非速度域(lsd_pol)数据

# 谱线参数

# 自动选择有效谱线参数文件，保证 wl0 有效
line_data = None
for fname in ['input/lines.txt', 'input/lines_test.txt']:
    try:
        ld = LineData(fname)
        if ld.wl0 is not None:
            line_data = ld
            break
    except Exception:
        continue
if line_data is None:
    raise RuntimeError('未找到有效的谱线参数文件')
# 启用QU以便可选择输出Q/U
linemodel = GaussianZeemanWeakLineModel(line_data,
                                        k_QU=1.0,
                                        enable_V=True,
                                        enable_QU=True)

# 网格
grid = diskGrid(nr=nr, r_in=0.0, r_out=r_out, verbose=0)

# 创建spot
rs = np.linspace(r_min, r_max, n_spots)
spots = [
    Spot(r=r,
         phi_initial=phi_initial,
         amplitude=amp,
         spot_type='emission',
         radius=spot_radius,
         B_amplitude=B_amp,
         B_direction='radial') for r in rs
]
spot_collection = SpotCollection(spots=spots,
                                 pOmega=pOmega,
                                 r0=radius,
                                 period=period)
spot_geometry = TimeEvolvingSpotGeometry(grid, spot_collection)

# 物理常数
c = 2.99792458e5  # km/s
# 确保 wl0 有效并为 float
if line_data.wl0 is None:
    raise ValueError("LineData.wl0 为空，无法构造波长网格")
wl0 = float(line_data.wl0)

# 读取 specType 参数以决定输出格式
try:
    par = readParamsTomog('input/params_tomog.txt', verbose=0)
    spec_type = getattr(par, 'specType', 'auto')
    if spec_type.lower() == 'spec':
        wavelength_domain = True
    elif spec_type.lower() == 'lsd':
        wavelength_domain = False
    # 如果 specType='auto'，默认使用波长域
    pol_out = getattr(par, 'polOut', 'V')
except Exception as e:
    print(
        f"Warning: failed to read specType from params, using wavelength domain: {e}"
    )
    wavelength_domain = True
    pol_out = 'V'

print(
    f"Output format: {'spec_pol (wavelength)' if wavelength_domain else 'lsd_pol (velocity)'}"
)

# 构造光谱网格
if wavelength_domain:
    # 波长域：从 wl0-delta_wl 到 wl0+delta_wl
    # delta_wl 对应于 ±200 km/s 的 Doppler 移动
    delta_wl = wl0 * 200.0 / c
    wl_grid = np.linspace(wl0 - delta_wl, wl0 + delta_wl, 401)
    x_grid = wl_grid  # 输出时使用波长
else:
    # 速度域：-200 到 +200 km/s
    v_range = np.linspace(-200, 200, 401)
    wl_grid = wl0 * (1 + v_range / c)
    x_grid = v_range  # 输出时使用速度

for i, phase in enumerate(phases):
    brightness, Br, Bphi = spot_geometry.generate_distributions(phase)
    spots_evolved = spot_collection.get_spots_at_phase(phase)
    incl_rad = np.deg2rad(inclination)
    Blos = Br * np.sin(incl_rad) * np.cos(grid.phi)
    # 横向场幅度，确保根号内非负
    Bperp = np.sqrt(np.maximum(Br**2 + Bphi**2 - Blos**2, 0.0))
    chi = grid.phi
    v0_r0 = vsini / np.sin(incl_rad)
    v_phi = v0_r0 * (grid.r / radius)**pOmega
    v_los = v_phi * np.sin(incl_rad) * np.sin(grid.phi)
    # 计算局部波长：考虑视向速度的 Doppler 移动
    wl_local = wl_grid[:, None] / (1.0 + v_los[None, :] / c)
    amp_local = brightness
    profiles = linemodel.compute_local_profile(wl_local,
                                               amp=amp_local,
                                               Blos=Blos,
                                               Bperp=Bperp,
                                               chi=chi,
                                               Ic_weight=grid.area)
    stokes_i = np.sum(profiles['I'], axis=1)
    stokes_v = np.sum(profiles['V'], axis=1)
    stokes_q = np.sum(profiles['Q'], axis=1)
    stokes_u = np.sum(profiles['U'], axis=1)
    total_area = np.sum(grid.area)
    if total_area > 0:
        stokes_i = stokes_i / total_area
        stokes_v = stokes_v / total_area
        stokes_q = stokes_q / total_area
        stokes_u = stokes_u / total_area
    # 加噪声
    sigma = 1.0 / SNR
    noise_i = np.random.normal(0, sigma, stokes_i.shape)
    noise_v = np.random.normal(0, sigma, stokes_v.shape)
    noise_q = np.random.normal(0, sigma, stokes_q.shape)
    noise_u = np.random.normal(0, sigma, stokes_u.shape)
    stokes_i_noisy = stokes_i + noise_i
    stokes_v_noisy = stokes_v + noise_v
    stokes_q_noisy = stokes_q + noise_q
    stokes_u_noisy = stokes_u + noise_u

    # 保存数据（支持波长域和速度域格式）
    if wavelength_domain:
        # spec_pol 格式：Wav Int Pol Null1 Null2 sigma_int
        fname = os.path.join(output_dir, f'obs_phase_{i:02d}.spec')
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('# Wav(nm) Int Pol Null1 Null2 sigma_int\n')
            for j in range(len(wl_grid)):
                # 根据 polOut 选择要输出的偏振分量
                if pol_out == 'V':
                    pol = stokes_v_noisy[j]
                elif pol_out == 'Q':
                    pol = stokes_q_noisy[j]
                else:  # 'U'
                    pol = stokes_u_noisy[j]
                f.write(
                    f"{wl_grid[j]:.6f} {stokes_i_noisy[j]:.8e} {pol:.8e} 0.0 0.0 {sigma:.3e}\n"
                )
        print(f'写入: {fname} (spec_pol格式，{len(wl_grid)}个波长点)')
    else:
        # lsd_pol 格式：RV Int sigma_int Pol sigma_pol Null1 sigma_null1
        from core.SpecIO import write_model_spectrum
        fname = os.path.join(output_dir, f'obs_phase_{i:02d}.lsd')
        write_model_spectrum(
            fname,
            x_grid,  # 使用 v_range 或 wl_grid（取决于格式）
            stokes_i_noisy,
            V=stokes_v_noisy,
            Q=stokes_q_noisy,
            U=stokes_u_noisy,
            sigmaI=np.full_like(stokes_i_noisy, sigma),
            fmt="lsd",
            pol_channel=pol_out,
            include_null=True)
        print(f'写入: {fname} (lsd_pol格式，{len(x_grid)}个速度点)')
