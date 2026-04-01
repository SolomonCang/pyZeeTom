# Veiling Effect 使用指南

## 物理背景

**Veiling（掩饰效应）**是指吸积盘或其他热源产生的额外连续谱辐射稀释恒星吸收线的现象，常见于：

### 物理模型

$$I_{\text{obs}}(\lambda) = \frac{I_{\text{star}}(\lambda) + r \cdot I_{\text{cont}}}{1 + r}$$

其中：

### 效果

1. **谱线深度稀释**：
   $$d_{\text{obs}} = \frac{d_0}{1 + r}$$

2. **等值宽度减小**：
   $$\text{EW}_{\text{obs}} = \frac{\text{EW}_0}{1 + r}$$

3. **Stokes 参数稀释**：所有 Stokes 参数（I, V, Q, U）都被相同因子稀释


## 实现方式

### ✅ 正确方式：使用 `veiling_factor` 参数

```python
from core.physical_model import create_physical_model

# 创建带 veiling 效应的模型
model = create_physical_model(
    par,
    wl0_nm=656.28,
    line_model=line_model,
    veiling_factor=1.0,  # r=1.0: 线深度减半
    verbose=1
)
```

### ❌ 错误方式：使用 `Ic_weight`

**不要**用 `Ic_weight` 模拟 veiling 效应！原因：


## 使用示例

### 示例 1：基本用法

```python
from core.mainFuncs import readParamsTomog
from core.local_linemodel_basic import LineData, GaussianZeemanWeakLineModel
from core.physical_model import create_physical_model

# 读取参数
par = readParamsTomog('input/params_tomog.txt')
line_data = LineData('input/lines.txt')
line_model = GaussianZeemanWeakLineModel(line_data)

# 不同 veiling 强度对比
veiling_factors = [0.0, 0.5, 1.0, 2.0]

for r in veiling_factors:
    model = create_physical_model(
        par,
        wl0_nm=line_data.wl0,
        line_model=line_model,
        veiling_factor=r,
        verbose=0
    )
    
    # 计算光谱
    integrator = model.integrator
    integrator.compute_spectrum()
    
    # 分析结果
    line_depth = 1.0 - np.min(integrator.I)
    print(f"r={r:.1f}: Line depth = {line_depth:.4f}")
```

### 示例 2：在 Forward Tomography 中使用

```python
from pyzeetom import tomography

# 在参数文件中添加 veiling 配置（如果支持）
# 或在代码中指定
results = tomography.forward_tomography(
    'input/params_tomog.txt',
    veiling_factor=0.8,  # 传递给底层 integrator
    verbose=1
)
```

### 示例 3：相位依赖的 Veiling（推荐方式）

**新功能**：`veiling_factor` 现在支持传入函数！

```python
import numpy as np

# 定义相位依赖的 veiling 函数
def veiling_hotspot(phase):
    """Hotspot 模型：吸积柱在相位 0.25 时正对观测者"""
    phase_max = 0.25
    width = 0.15
    
    # 处理相位折叠
    delta = phase - phase_max
    if delta > 0.5:
        delta -= 1.0
    elif delta < -0.5:
        delta += 1.0
    
    r_max = 2.0
    r_min = 0.1
    return r_max * np.exp(-0.5 * (delta / width)**2) + r_min

# 直接传入函数（不需要循环！）
model = create_physical_model(
    par,
    wl0_nm=line_data.wl0,
    line_model=line_model,
    veiling_factor=veiling_hotspot,  # ← 传入函数
    verbose=0
)

# 计算不同相位的光谱
integrator = model.integrator
phases = np.linspace(0, 1, 20)

for phase in phases:
    result = integrator.compute_spectrum_single_phase(phase)
    print(f"Phase {phase:.2f}: r={result['veiling_factor']:.2f}, "
          f"depth={1.0 - np.min(result['I']):.4f}")
```

**物理模型示例**：

```python
# 模型 1: 吸积盘遮掩效应
def veiling_disk_occultation(phase, inclination_deg=60):
    phi = 2 * np.pi * phase
    inc_rad = np.deg2rad(inclination_deg)
    proj = np.abs(np.cos(phi) * np.sin(inc_rad))
    return 1.5 * proj + 0.2

# 模型 2: 正弦调制（简单周期变化）
def veiling_sinusoidal(phase):
    return 1.0 + 0.5 * np.sin(2 * np.pi * phase)

# 使用
model = create_physical_model(..., veiling_factor=veiling_disk_occultation)
```


## 参数说明

### `veiling_factor` (float or callable, default=0.0)

**类型 1: 常数 (float)**

| 值 | 物理含义 | 效果 |
|---|----------|------|
| `r = 0.0` | 无 veiling（纯恒星） | 原始线深度 |
| `r = 0.5` | 中等额外连续谱 | 线深度 = 原始 × 0.67 |
| `r = 1.0` | 强 veiling | 线深度 = 原始 × 0.5 |
| `r = 2.0` | 极强 veiling | 线深度 = 原始 × 0.33 |

**类型 2: 相位依赖函数 (callable)**

```python
# 签名: veiling_factor(phase) -> float
# phase: 0-1 (轨道相位)
def my_veiling(phase):
    return 1.0 + 0.5 * np.sin(2*np.pi*phase)
```

**典型值范围**：

## 验证方法

### 基本 Veiling 效应

```bash
python examples/06_veiling_effect_demo.py
```

该脚本会：
1. 计算不同 veiling 因子下的光谱
2. 验证线深度/EW 的稀释关系：$d_{\text{obs}} = d_0 / (1 + r)$
3. 绘制对比图

### 相位依赖 Veiling

```bash
python examples/07_phase_dependent_veiling.py
```

该脚本演示：
1. **Hotspot 模型**：局域吸积区域旋转
2. **盘遮掩模型**：视角几何调制
3. **正弦模型**：周期性变化
4. 动态谱图展示相位演化
## 验证方法

运行演示脚本验证实现：

```bash
python examples/06_veiling_effect_demo.py
```

该脚本会：
1. 计算不同 veiling 因子下的光谱
2. 验证线深度/EW 的稀释关系：$d_{\text{obs}} = d_0 / (1 + r)$
3. 绘制对比图


## 注意事项

1. **归一化模式**：
   - `normalize_continuum=True` 时，veiling 在归一化后应用
   - 连续谱保持在 1.0，谱线被稀释

2. **所有 Stokes 参数**：
   - I, V, Q, U 都被相同因子 $(1+r)$ 稀释
   - 保持偏振度不变（相对关系保持）

3. **与 `amp` 的区别**：
   - `amp`：改变局部发射/吸收**强度**
   - `veiling_factor`：在**观测端**叠加额外连续谱

4. **物理解释**：
   - Veiling 是**观测效应**，不改变恒星本身的物理状态
   - 适用于描述多组分系统（恒星 + 吸积盘）


### Q: 相位依赖的 veiling 如何工作？

**A:** 
1. 定义函数 `f(phase) -> r`
2. 传递给 `veiling_factor` 参数
3. 在 `compute_spectrum_single_phase(phase)` 中自动调用
4. 每个相位使用对应的 `r(phase)` 值

```python
def my_veiling(phase):
    return 1.0 + 0.5 * np.cos(2*np.pi*phase)

model = create_physical_model(..., veiling_factor=my_veiling)
result = model.integrator.compute_spectrum_single_phase(0.25)
# 自动使用 r = my_veiling(0.25)
```

### Q: 能否实现波长依赖的 veiling？

**A:** 当前实现假设 $r$ 在整个谱线范围内恒定（灰 veiling）。如需波长依赖（非灰），需要：
1. 修改 `compute_spectrum` 接受 $r(\lambda)$
2. 在归一化步骤前应用波长相关的稀释

注：大多数情况下灰 veiling 假设已足够（短波段范围内）t` 会同时放大连续谱和线轮廓，导致：

正确的 veiling 应该：

### Q: 能否实现波长依赖的 veiling？

**A:** 当前实现假设 $r$ 在整个谱线范围内恒定（灰 veiling）。如需波长依赖（非灰），需要：
1. 修改 `compute_spectrum` 接受 $r(\lambda)$
2. 在归一化步骤前应用波长相关的稀释

### Q: 如何从观测反推 veiling 参数？

**A:** 使用 MEM 反演时，可将 `veiling_factor` 作为全局参数优化：
```python
## 更新日志

  - ✅ 初始实现，支持恒定 veiling 参数
  - ✅ **新增**：相位依赖的 veiling（函数接口）
  - ✅ 添加 `compute_spectrum_single_phase()` 方法
  - ✅ 三种物理模型演示（hotspot, disk occultation, sinusoidal）
  
  - [ ] 波长依赖的 veiling（非灰 veiling）
  - [ ] 在反演中优化 veiling 参数（全局参数拟合）
  - [ ] 支持多组分 veiling（恒星+盘+热点）ve, r0=0.5)
```


## 参考文献

1. Hartigan, P., et al. 1995, ApJ, 452, 736 - "Disk Accretion and Mass Loss from Young Stars"
2. Basri, G., & Batalha, C. 1990, ApJ, 363, 654 - "Veiling in T Tauri stars"
3. White, R. J., & Basri, G. 2003, ApJ, 582, 1109 - "Very Low Mass Stars and Brown Dwarfs"


## 更新日志

  - [ ] 波长依赖的 veiling
  - [ ] 相位依赖的 veiling（自动化）
  - [ ] 在反演中优化 veiling 参数
# Veiling Effect 使用指南

## 物理背景

**Veiling（掩饰效应）**是指吸积盘或其他热源产生的额外连续谱辐射稀释恒星吸收线的现象，常见于：
- **T Tauri 星**：吸积盘产生强连续辐射
- **Be 星**：星周盘发射
- **激变变星**：吸积盘热辐射

### 物理模型

$$I_{\text{obs}}(\lambda) = \frac{I_{\text{star}}(\lambda) + r \cdot I_{\text{cont}}}{1 + r}$$

其中：
- $I_{\text{star}}(\lambda)$：本征恒星光谱（含吸收线）
- $I_{\text{cont}}$：连续谱水平（归一化时 = 1）
- $r$：veiling 参数（$r \geq 0$）

### 效果

1. **谱线深度稀释**：
    $$d_{\text{obs}} = \frac{d_0}{1 + r}$$

2. **等值宽度减小**：
    $$\text{EW}_{\text{obs}} = \frac{\text{EW}_0}{1 + r}$$

3. **Stokes 参数稀释**：所有 Stokes 参数（I, V, Q, U）都被相同因子稀释

---

## 实现方式

### ✅ 正确方式：使用 `veiling_factor` 参数

```python
from core.physical_model import create_physical_model

# 创建带 veiling 效应的模型
model = create_physical_model(
     par,
     wl0_nm=656.28,
     line_model=line_model,
     veiling_factor=1.0,  # r=1.0: 线深度减半
     veiling_region='stellar',  # 'all', 'stellar', 或 'disk'
     verbose=1
)
```

### ❌ 错误方式：使用 `Ic_weight`

**不要**用 `Ic_weight` 模拟 veiling 效应！原因：
- `Ic_weight` 是几何权重，会同时缩放连续谱和线轮廓
- Veiling 只影响**谱线相对深度**，不改变连续谱的几何分布

---

## 功能特性

pyZeeTom 的 veiling 实现支持三种模式：

### 1. 常数 Veiling（基本用法）

适用于恒定吸积率、无相位变化的系统。

```python
# 不同 veiling 强度对比
veiling_factors = [0.0, 0.5, 1.0, 2.0]

for r in veiling_factors:
     model = create_physical_model(
          par,
          wl0_nm=line_data.wl0,
          line_model=line_model,
          veiling_factor=r,  # 常数
          verbose=0
     )
    
     integrator = model.integrator
     integrator.compute_spectrum()
    
     line_depth = 1.0 - np.min(integrator.I)
     print(f"r={r:.1f}: depth={line_depth:.4f}")
```

**典型值范围**：
- T Tauri 星：$r = 0.1 \sim 3.0$（可达更高）
- 弱吸积系统：$r < 0.5$
- 强吸积系统：$r > 1.0$

### 2. 相位依赖 Veiling（高级功能）

适用于局域吸积区（热点）、视角调制、脉动吸积等场景。

```python
import numpy as np

# 定义相位依赖的 veiling 函数
def veiling_hotspot(phase):
     """局域吸积热点模型"""
     phase_max = 0.25  # 热点在相位 0.25 正对观测者
     width = 0.15
     delta = phase - phase_max
     if delta > 0.5:
          delta -= 1.0
     elif delta < -0.5:
          delta += 1.0
     r_max, r_min = 2.0, 0.1
     return r_max * np.exp(-0.5 * (delta / width)**2) + r_min

# 传入函数（不是函数调用结果！）
model = create_physical_model(
     par,
     wl0_nm=line_data.wl0,
     line_model=line_model,
     veiling_factor=veiling_hotspot,  # ← 传入函数对象
     verbose=0
)

# 计算不同相位的光谱
integrator = model.integrator
for phase in np.linspace(0, 1, 20):
     result = integrator.compute_spectrum_single_phase(phase)
     print(f"Phase {phase:.2f}: r={result['veiling_factor']:.2f}")
```

**常用物理模型**：

```python
# 模型 1: 盘遮掩效应（视角几何）
def veiling_disk_occultation(phase, r_max=1.5, inclination_deg=60):
     phi = 2 * np.pi * phase
     inc_rad = np.deg2rad(inclination_deg)
     proj = np.abs(np.cos(phi) * np.sin(inc_rad))
     return r_max * proj + 0.2

# 模型 2: 正弦调制（周期性吸积）
def veiling_sinusoidal(phase, r_mean=1.0, amplitude=0.8):
     phi = 2 * np.pi * phase
     return max(0.0, r_mean + amplitude * np.sin(phi))

# 模型 3: 双热点系统
def veiling_double_hotspot(phase):
     r1 = 2.0 * np.exp(-0.5 * ((phase - 0.2) / 0.1)**2)
     r2 = 1.5 * np.exp(-0.5 * ((phase - 0.7) / 0.1)**2)
     return r1 + r2 + 0.1
```

### 3. 区域 Veiling（物理精确性）

**关键概念**：在 T Tauri 星等系统中，吸积连续谱主要来自恒星表面的边界层/激波，因此**只稀释恒星吸收线**，不影响吸积盘自身的发射。

```python
# 物理上正确的 T Tauri 星模型
model = create_physical_model(
     par,
     wl0_nm=line_data.wl0,
     line_model=line_model,
     veiling_factor=1.5,
     veiling_region='stellar',  # ← 只对 r ≤ R* 的恒星光球层应用 veiling
     verbose=0
)
```

**三种区域模式**：

| `veiling_region` | 物理含义 | 适用场景 |
|-----------------|---------|---------|
| `'all'` (默认) | 全局稀释所有光谱 | 传统近似，计算简单 |
| `'stellar'` | 只稀释 r ≤ R* 的恒星成分 | **T Tauri 星**（推荐）、Herbig Ae/Be |
| `'disk'` | 只稀释 r > R* 的盘成分 | 外部照射、特殊几何 |

**效果对比**（以 r=1.5 为例）：
- **'all' 模式**：稀释因子 = 1/(1+1.5) = 0.40（全部流量）
- **'stellar' 模式**：稀释因子 ≈ 0.88（仅 10% 的恒星流量被稀释）
- **'disk' 模式**：稀释因子 ≈ 0.42（仅 90% 的盘流量被稀释）

---

## 完整示例

### 综合演示脚本

运行以下脚本可查看所有 veiling 功能：

```bash
python examples/06_veiling_comprehensive_demo.py
```

该脚本演示：
1. **Part 1**: 常数 veiling 效应（r = 0, 0.5, 1.0, 2.0）
    - 验证 $d = d_0/(1+r)$ 关系
    - 线深度和等值宽度对比

2. **Part 2**: 相位依赖 veiling
    - Hotspot 模型：局域吸积区旋转
    - Disk Occultation 模型：视角几何调制
    - Sinusoidal 模型：周期性变化
    - 动态谱图可视化

3. **Part 3**: 区域 veiling
    - 'all', 'stellar', 'disk' 三种模式对比
    - 系统几何示意图
    - 流量预算分析

---

## 参数说明

### `veiling_factor` (float or callable, default=0.0)

**类型 1: 常数 (float)**

| 值 | 物理含义 | 效果 |
|---|----------|------|
| `r = 0.0` | 无 veiling（纯恒星） | 原始线深度 |
| `r = 0.5` | 中等额外连续谱 | 线深度 = 原始 × 0.67 |
| `r = 1.0` | 强 veiling | 线深度 = 原始 × 0.5 |
| `r = 2.0` | 极强 veiling | 线深度 = 原始 × 0.33 |

**类型 2: 相位依赖函数 (callable)**

```python
# 函数签名: veiling_factor(phase) -> float
# phase: 0-1 (轨道相位)
def my_veiling(phase):
     return 1.0 + 0.5 * np.sin(2*np.pi*phase)
```

### `veiling_region` (str, default='all')

| 值 | 说明 | 推荐场景 |
|---|------|---------|
| `'all'` | 全局应用 veiling | 快速计算、不需要高精度 |
| `'stellar'` | 仅应用于 r ≤ R* | **T Tauri 星**、Herbig Ae/Be（推荐）|
| `'disk'` | 仅应用于 r > R* | 特殊几何、外部照射 |

---

## 注意事项

1. **归一化模式**：
    - `normalize_continuum=True` 时，veiling 在归一化后应用
    - 连续谱保持在 1.0，谱线被稀释

2. **所有 Stokes 参数**：
    - I, V, Q, U 都被相同因子 $(1+r)$ 稀释
    - 保持偏振度不变（相对关系保持）

3. **与 `amp` 的区别**：
    - `amp`：改变局部发射/吸收**强度**
    - `veiling_factor`：在**观测端**叠加额外连续谱

4. **物理解释**：
    - Veiling 是**观测效应**，不改变恒星本身的物理状态
    - 适用于描述多组分系统（恒星 + 吸积盘）

5. **区域 veiling 的物理基础**：
    - 吸积连续谱来自边界层/激波（靠近恒星表面）
    - 这些连续谱与恒星光叠加，稀释恒星吸收线
    - 但不影响吸积盘外围的发射线

---

## 常见问题

### Q: 如何选择 veiling_region 参数？

**A:** 
- **T Tauri 星、Herbig Ae/Be**：使用 `'stellar'`（物理上最准确）
- **快速原型、不需要高精度**：使用 `'all'`（计算简单）
- **特殊几何（如外部照射盘）**：使用 `'disk'`

### Q: 相位依赖的 veiling 如何工作？

**A:** 
1. 定义函数 `f(phase) -> r`
2. 传递给 `veiling_factor` 参数（传入函数对象，不是调用结果）
3. 在 `compute_spectrum_single_phase(phase)` 中自动调用
4. 每个相位使用对应的 `r(phase)` 值

```python
def my_veiling(phase):
     return 1.0 + 0.5 * np.cos(2*np.pi*phase)

model = create_physical_model(..., veiling_factor=my_veiling)
result = model.integrator.compute_spectrum_single_phase(0.25)
# 自动使用 r = my_veiling(0.25)
```

### Q: 能否实现波长依赖的 veiling？

**A:** 当前实现假设 $r$ 在整个谱线范围内恒定（灰 veiling）。如需波长依赖（非灰），需要：
1. 修改 `compute_spectrum` 接受 $r(\lambda)$
2. 在归一化步骤前应用波长相关的稀释

注：大多数情况下灰 veiling 假设已足够（短波段范围内）。

### Q: 如何从观测反推 veiling 参数？

**A:** 使用 MEM 反演时，可将 `veiling_factor` 作为全局参数优化：
```python
# 未来功能：在反演中优化 veiling
result = tomography.inversion_tomography(
     'input/params_tomog.txt',
     optimize_veiling=True,  # 待实现
     veiling_initial=1.0
)
```

### Q: 'stellar' 模式为什么稀释因子不等于理论值？

**A:** 因为只有部分流量（恒星光球层）被 veiling 稀释，盘的贡献不受影响。有效稀释因子为：

$$\text{dilution}_{\text{eff}} = \frac{f_{\text{stellar}}}{1+r} + f_{\text{disk}}$$

其中 $f_{\text{stellar}}$ 和 $f_{\text{disk}}$ 是恒星和盘的流量占比。

---

## 参考文献

1. Hartigan, P., et al. 1995, ApJ, 452, 736 - "Disk Accretion and Mass Loss from Young Stars"
2. Basri, G., & Batalha, C. 1990, ApJ, 363, 654 - "Veiling in T Tauri stars"
3. White, R. J., & Basri, G. 2003, ApJ, 582, 1109 - "Very Low Mass Stars and Brown Dwarfs"
4. Herczeg, G. J., & Hillenbrand, L. A. 2014, ApJ, 786, 97 - "UV Excess Measures of Accretion"

---

## 更新日志

- **2025-12-01**: 
  - ✅ 初始实现，支持恒定 veiling 参数
  - ✅ 新增：相位依赖的 veiling（函数接口）
  - ✅ 新增：区域 veiling（'all', 'stellar', 'disk'）
  - ✅ 添加 `compute_spectrum_single_phase()` 方法
  - ✅ 综合演示脚本整合（`06_veiling_comprehensive_demo.py`）
  
- **未来计划**：
  - [ ] 波长依赖的 veiling（非灰 veiling）
  - [ ] 在反演中优化 veiling 参数（全局参数拟合）
  - [ ] 支持多组分 veiling（恒星+盘+热点）
  - [ ] 与观测数据的自动拟合工具
