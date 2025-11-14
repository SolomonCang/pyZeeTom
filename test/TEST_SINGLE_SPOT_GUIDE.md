# 单Spot Stokes IVQU 谱线建模与可视化测试

## 概览

`test_single_spot_stokes_ivqu.py` 是一个综合性的测试脚本，演示如何使用 pyZeeTom 核心模块构建单个发射斑点的模型，并合成Stokes IVQU偏振光谱。

## 功能特点

### 核心功能
1. **参数读取** - 从 `input/params_tomog.txt` 读取观测参数（倾角、自转速率、周期、差速指数等）
2. **谱线参数加载** - 从 `input/lines.txt` 加载谱线信息（H-alpha等）
3. **盘网格生成** - 基于 `core/grid_tom.py` 的 `diskGrid` 创建极坐标网格
4. **Spot建模** - 使用 `core/spot_geometry.py` 的 `Spot` 和 `SpotCollection` 定义发射区
5. **谱线合成** - 利用 `core/local_linemodel_basic.py` 计算Stokes IVQU
6. **可视化** - 生成多相位的谱线图像

### 模型配置（硬编码）
- **Spot位置**：2R*（盘半径）
- **Spot性质**：发射，振幅0.5
- **磁场强度**：1000 Gauss（径向）
- **倾角**：从参数文件读取（默认60°）

## 输出说明

### 1. 可视化谱线图 (PNG)
```
test_output/stokes_spectrum_phase_XX.png
```
- 4个相位（0.00, 0.25, 0.50, 0.75）各生成一张
- 每张图包含4个面板：
  - **Stokes I**：强度谱线（连续谱基线为1.0）
  - **Stokes V**：圆偏振（Zeeman分裂效应）
  - **Stokes Q**：线性偏振分量1
  - **Stokes U**：线性偏振分量2

### 2. 数据存档 (NPZ)
```
test_output/single_spot_stokes_data.npz
```
包含每个相位的合成谱线数据：
- `phases`: 相位数组
- `phase_00`, `phase_01`, `phase_02`, `phase_03`: 各相位的字典，包含：
  - `v`: 速度网格 (km/s)
  - `wl`: 波长网格 (nm)
  - `specI`: Stokes I 强度谱
  - `specV`: Stokes V 圆偏振
  - `specQ`: Stokes Q 线性偏振
  - `specU`: Stokes U 线性偏振

## 核心类与函数

### 类：`SingleSpotModel`
```python
model = SingleSpotModel(params_file, lines_file)
```
完整的单Spot模型封装，包含：
- 参数管理
- 盘网格
- Spot集合
- 像素映射

**主要方法：**
- `make_grid()` - 生成网格
- `make_spot_collection()` - 创建Spot
- `get_spot_properties_at_phase(phase)` - 获取指定相位的Spot属性
- `map_spot_to_pixels(phase)` - 将Spot映射到像素（返回amp, Blos, Bperp, chi）

### 函数：`synthesize_stokes_spectrum()`
```python
result = synthesize_stokes_spectrum(model, phase, v_grid=None, inst_fwhm_kms=0.5)
```
合成Stokes IVQU谱线

**参数：**
- `model`: SingleSpotModel 实例
- `phase`: 相位 (0-1)
- `v_grid`: 速度网格，默认 [-300, 300] km/s，1000点
- `inst_fwhm_kms`: 仪器分辨率，默认 0.5 km/s

**返回：**
Dict with keys: `v`, `wl`, `specI`, `specV`, `specQ`, `specU`

### 函数：`plot_stokes_spectrum()`
```python
fig, axes = plot_stokes_spectrum(spec_result, phase, save_path=None)
```
绘制并保存谱线图

## 运行示例

### 基本运行
```bash
cd /path/to/pyZeeTom
python test/test_single_spot_stokes_ivqu.py
```

### 在Python脚本中调用
```python
from test.test_single_spot_stokes_ivqu import SingleSpotModel, synthesize_stokes_spectrum, plot_stokes_spectrum

# 初始化模型
model = SingleSpotModel('input/params_tomog.txt', 'input/lines.txt')

# 合成第一相位的谱线
result = synthesize_stokes_spectrum(model, phase=0.0)

# 绘制
plot_stokes_spectrum(result, phase=0.0, save_path='test_output/my_plot.png')
```

### 自定义Spot参数
在 `SingleSpotModel.make_spot_collection()` 方法中修改：
```python
# 修改这些参数
r_spot = 2.0           # Spot位置 (R_sun)
phi_initial = 0.0      # 初始方位角
amplitude = 0.5        # 发射强度
radius_spot = 0.3      # Spot半径
B_amplitude = 1000.0   # 磁场强度 (Gauss)
B_direction = 'radial' # 磁场方向
```

## 物理参数说明

### 磁场建模
当前实现采用弱场高斯Zeeman模型：
- **Blos**（视向磁场分量）：与V谱线形成速度依赖的Zeeman分裂
- **Bperp**（横向磁场分量）：与Q/U产生二阶偏振效应
- **chi**（磁场方向角）：Q/U的方向参考

### 投影效应
- 所有像素的Stokes谱线按面积加权平均
- 磁场在视向上的投影：$B_{los} = B \sin(i) \cos(\phi - \phi_0)$
- 其中i是倾角，φ是方位角

## 扩展建议

### 添加多个Spot
```python
spot1 = Spot(r=2.0, phi_initial=0.0, ...)
spot2 = Spot(r=2.5, phi_initial=np.pi, ...)
self.spot_collection = SpotCollection(spots=[spot1, spot2], ...)
```

### 自定义谱线模型
如需使用MEM等其他谱线模型，替换：
```python
line_model = GaussianZeemanWeakLineModel(...)
# 换为
line_model = YourCustomLineModel(...)
```

### 时间演化
若要模拟磁场随时间变化的演化，修改 `synthesize_stokes_spectrum()` 中的时间参数

## 已知限制

1. **逐像素计算** - 当前实现对每个像素逐个调用谱线模型，性能可优化（向量化）
2. **简单高斯卷积** - 仪器分辨率采用高斯卷积，实际可能需更复杂的核
3. **常数磁场** - 当前Spot内磁场强度为常数，可扩展为空间变化
4. **刚性假设** - Spot形状保持为高斯，可扩展为其他几何

## 依赖关系

```
core/grid_tom.py              ← diskGrid
core/spot_geometry.py         ← Spot, SpotCollection
core/local_linemodel_basic.py ← LineData, GaussianZeemanWeakLineModel
core/mainFuncs.py             ← readParamsTomog
```

## 测试结果示例

```
======================================================================
 单Spot Stokes IVQU 谱线建模与可视化测试
======================================================================

[1] 读取输入文件...
[2] 模型参数...
[3] 合成Stokes IVQU谱线... ✓
[4] 绘制谱线图... ✓
[5] 谱线统计信息:
    速度范围: [-300.0, 300.0] km/s
    波长范围: [655.6233, 656.9367] nm
    Stokes I 范围: [0.360000, 0.379996]
    Stokes V 范围: [-0.000187, 0.000187]
[6] 保存数据... ✓
```

---

**最后更新**：2025-11-14  
**作者**：pyZeeTom Testing Suite  
**相关文档**：docs/PROJECT_STRUCTURE.md, docs/QUICK_START.md
