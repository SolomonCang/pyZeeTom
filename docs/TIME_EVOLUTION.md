# 时间演化与差速转动功能文档

## 概述

本文档说明 pyZeeTom 项目中新增的时间演化和差速转动支持功能。这些功能允许在多个观测相位下，正确模拟星周物质的差速转动导致的盘面结构变化。

## 新增参数

### params_tomog.txt 文件（第13行）

**jDateRef** - 参考时刻的 Julian Date (HJD0)
- 用于计算观测相位：`phase = (JD - JD0) / period`
- 对于刚体转动，`phase` 直接对应观测角度偏移：`Δφ = 2π × phase`
- 对于差速转动，每个环的相位演化由各自的角速度决定

示例：
```
#13 jDateRef  (HJD0, reference epoch for phase calculation: phase = (HJD - HJD0) / period)
2450000.5
```

### 差速转动参数（第0行）

**pOmega** - 差速转动幂律指数
- 定义角速度随半径的变化规律：`Ω(r) = Ω_ref × (r/r0)^pOmega`
- **pOmega = 0.0**：刚体转动（所有环同步）
- **pOmega = -0.5**：开普勒型（类太阳盘，常用于吸积盘）
- **pOmega = -1.0**：恒定角动量

示例：
```
#0 inclination  vsini  period  pOmega
60.0  25.0  1.0  -0.5
```

## 核心功能模块

### 1. 相位计算（core/mainFuncs.py）

#### `compute_phase_from_jd(jd, jd_ref, period)`
计算观测相位。

**参数：**
- `jd`: 观测的 Julian Date（可以是单个值或数组）
- `jd_ref`: 参考时刻 HJD0
- `period`: 自转周期（天）

**返回：**
- `phase`: 相位值，`phase = (jd - jd_ref) / period`

**示例：**
```python
from core.mainFuncs import compute_phase_from_jd

jd_ref = 2450000.0
period = 2.0
jd = 2450002.0  # 2天后

phase = compute_phase_from_jd(jd, jd_ref, period)
# 结果：phase = 1.0（完成一个周期）
```

#### readParamsTomog 类
读取参数文件后，自动计算每个观测的相位并存储在 `self.phases` 数组中。

```python
par = readParamsTomog('params_tomog.txt')
print(par.phases)  # 每个观测的相位数组
```

### 2. 盘面结构演化（core/grid_tom.py）

#### `diskGrid.rotate_to_phase(phase, pOmega=0.0, r0=1.0, period=1.0)`
根据相位和差速转动参数，计算时间演化后的像素方位角。

**参数：**
- `phase`: 观测相位
- `pOmega`: 差速转动幂律指数
- `r0`: 参考半径（通常为星球半径）
- `period`: 参考半径处的转动周期

**返回：**
- `phi_new`: 更新后的方位角数组（弧度）

**物理意义：**

1. **刚体转动**（pOmega=0）：
   ```
   所有环转动相同角度：
   Δφ = 2π × phase
   ```

2. **差速转动**（pOmega≠0）：
   ```
   每环转动角度与半径相关：
   Δφ(r) = 2π × phase × (r/r0)^pOmega
   
   例如 pOmega=-0.5（开普勒型）：
   - 内圈（r < r0）：转得更快
   - 外圈（r > r0）：转得更慢
   ```

**示例：**
```python
from core.grid_tom import diskGrid

grid = diskGrid(nr=60, r_in=0.0, r_out=1.0)

# 刚体转动 1/4 周期
phi_rigid = grid.rotate_to_phase(0.25, pOmega=0.0)

# 开普勒型差速转动 1/2 周期
phi_kepler = grid.rotate_to_phase(0.5, pOmega=-0.5, r0=1.0)
```

### 3. 速度场积分（core/velspace_DiskIntegrator.py）

#### VelspaceDiskIntegrator
积分器现在支持时间相关的盘面结构。

**新增参数：**
- `time_phase`: 观测相位，用于计算差速转动导致的结构演化

**工作原理：**
1. 从几何对象获取差速转动参数（pOmega, r0, period）
2. 根据 `time_phase` 计算演化后的方位角
3. 使用演化后的方位角计算：
   - 响应函数（亮度/辐射分布）
   - 视向投影（观测几何）
4. 生成正确的 Stokes 谱线

**示例：**
```python
from core.velspace_DiskIntegrator import VelspaceDiskIntegrator

inte = VelspaceDiskIntegrator(
    geom=geom,
    wl0_nm=500.0,
    v_grid=velocity_grid,
    line_model=line_model,
    disk_power_index=-0.5,  # pOmega
    disk_r0=1.0,
    time_phase=0.5,  # 当前观测相位
)

# inte.I, inte.V 已考虑时间演化
```

### 4. 主流程集成（pyzeetom/tomography.py）

#### SimpleDiskGeometry 类
几何容器现在包含差速转动参数：

```python
geom = SimpleDiskGeometry(
    grid,
    inclination_deg=60.0,
    pOmega=-0.5,      # 差速转动指数
    r0=1.0,           # 参考半径
    period=1.0        # 转动周期
)
```

#### main() 函数
主流程自动为每个观测使用正确的相位：

```python
results = []
for i, obs in enumerate(obsSet):
    current_phase = par.phases[i]  # 当前观测的相位
    
    inte = VelspaceDiskIntegrator(
        geom=geom,
        time_phase=current_phase,  # 传递相位
        # ... 其他参数
    )
    results.append((v_grid, inte.I, inte.V))
```

## 使用示例

### 完整工作流程

1. **准备参数文件** (`input/params_tomog.txt`)：
```
# 倾角 vsini 周期 差速指数
60.0  25.0  1.0  -0.5

# ... 其他参数 ...

# 参考时刻
2450000.5

# 观测列表（文件名 JD velR）
obs/phase000.lsd  2450000.500  0.0
obs/phase001.lsd  2450000.625  0.0
obs/phase002.lsd  2450000.750  0.0
# ... 更多观测
```

2. **运行正演**：
```python
from pyzeetom import tomography

results = tomography.main()

# results 包含每个相位的合成谱
# 每个观测已正确考虑差速转动导致的结构演化
```

3. **查看结果**：
```python
for i, (v, I, V) in enumerate(results):
    print(f"Phase {i}: I range = [{I.min():.4f}, {I.max():.4f}]")
```

## 物理场景说明

### 刚体转动（pOmega=0）
- 适用于小尺度结构（如表面亮斑、磁场结构）
- 整个盘面作为一个整体转动
- 所有位置在相同时间转过相同角度

### 差速转动（pOmega≠0）
- 适用于延展盘面（如吸积盘、行星环）
- 不同半径处有不同的角速度
- 内圈和外圈的相对位置随时间变化

**开普勒型（pOmega=-0.5）示例：**

假设观测跨越 2 个周期（相位 0 到 2）：
- 在 r=r0 处：转动 2×2π = 4π（2周）
- 在 r=0.5r0 处：转动 2×2π×(0.5)^(-0.5) ≈ 5.66π（约2.83周）
- 在 r=2r0 处：转动 2×2π×(2)^(-0.5) ≈ 2.83π（约1.41周）

这导致盘面结构随时间"撕扯变形"。

## 测试

运行测试验证功能：
```bash
python test/test_phase_evolution.py
```

测试覆盖：
- ✓ 相位计算正确性
- ✓ 刚体转动
- ✓ 差速转动
- ✓ 周期性
- ✓ 参数文件集成

## 注意事项

1. **相位基准**：所有相位计算相对于 `jDateRef`，确保此值设置正确
2. **周期单位**：周期 `period` 单位为天
3. **pOmega 符号**：
   - 负值（如 -0.5）：内圈转得快
   - 正值（罕见）：外圈转得快
   - 零：刚体转动
4. **数值精度**：差速转动可能导致数值不稳定，建议 `|pOmega| ≤ 1.0`

## 后续扩展

未来可能的增强：
- [ ] 支持非幂律型的角速度分布
- [ ] 时间相关的响应函数（随相位变化的亮度分布）
- [ ] 反演中的差速转动参数拟合
- [ ] 可视化工具显示盘面演化动画

## 参考文献

关于差速转动的物理背景，参考：
- Donati et al. (2003) - ZDI with differential rotation
- Keplerian disk dynamics - standard astrophysical fluid dynamics texts
