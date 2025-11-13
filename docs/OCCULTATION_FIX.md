# 恒星遮挡功能修正说明

## 问题描述

原始实现中，恒星遮挡区域随观测相位（phase）变化，这与物理模型不符。

### 错误原因

在 `core/grid_tom.py` 的 `compute_stellar_occultation_mask` 函数中：
```python
# 错误代码
phi_phase = self.phi + 2.0 * np.pi * phase
```

这行代码导致网格像素的方位角随 `phase` 旋转，进而导致遮挡区域也随之旋转。

### 物理模型约定

正确的物理图像应该是：
1. **观察者方向固定**：观察者始终从某个固定方向（例如 φ_obs=0）观测
2. **盘面坐标固定**：网格像素的坐标 (r, φ) 在盘面坐标系中固定不变
3. **遮挡区域固定**：在盘面坐标系中，被恒星遮挡的区域位置固定（背向观察者）
4. **Spot运动**：spot随差速转动进出遮挡区域，导致可见spot数量变化
5. **光谱调制**：spot进出遮挡区产生周期性光谱变化

## 修正方案

### 1. 修改 `core/grid_tom.py`

**函数签名变更**：
```python
# 修正前
def compute_stellar_occultation_mask(self, phase, inclination_deg, stellar_radius=1.0, verbose=0)

# 修正后
def compute_stellar_occultation_mask(self, phi_obs, inclination_deg, stellar_radius=1.0, verbose=0)
```

**关键改动**：
- 移除 `phase` 参数（时间相关）
- 添加 `phi_obs` 参数（观察者方向，空间固定）
- 移除错误的 `phi_phase = self.phi + 2π*phase`
- 直接使用 `self.phi`（盘面固有坐标）

**新的遮挡计算逻辑**：
```python
# 像素笛卡尔坐标（盘面坐标系，固定）
x_disk = self.r * np.cos(self.phi)  # 不再随phase变化
y_disk = self.r * np.sin(self.phi)
z_disk = 0  # 赤道盘

# 观察者视线方向（固定）
n_obs = (sin(i) * cos(phi_obs), sin(i) * sin(phi_obs), cos(i))

# 像素沿视线方向的距离
dist_along_view = r_pixel · n_obs

# 垂直距离
r_perp = sqrt(|r_pixel|² - dist_along_view²)

# 遮挡判据
occluded = (r_perp < R*) & (dist_along_view < 0)
```

### 2. 修改 `core/velspace_DiskIntegrator.py`

**调用处变更**：
```python
# 修正前
occultation_mask = self.grid.compute_stellar_occultation_mask(
    phase=phase_for_occult,
    inclination_deg=inclination_deg,
    stellar_radius=stellar_radius)

# 修正后
phi_obs = getattr(self.geom, "phi_obs", 0.0)  # 从几何对象获取
occultation_mask = self.grid.compute_stellar_occultation_mask(
    phi_obs=phi_obs,
    inclination_deg=inclination_deg,
    stellar_radius=stellar_radius)
```

### 3. 更新测试脚本

在 `test/test_stellar_occultation.py` 中：
- 添加遮挡几何示意图（`plot_occultation_geometry`）
- 展示遮挡区域在盘面坐标系中的固定位置
- 说明物理模型约定

## 验证结果

### 修正前（错误）
- 遮挡像素数在不同相位略有浮动：1797-1801
- 原因：遮挡区域随phase旋转，数值误差导致浮动

### 修正后（正确）
- 遮挡像素数在所有相位**完全一致**：1801/7202 (25.0%)
- 遮挡区域在盘面坐标系中固定在 φ≈180° 方向
- 符合物理约定：观察者固定，盘面转动

## 物理图像总结

```
时刻 t=0:
  - 观察者: φ=0° (固定)
  - Spot位置: φ≈0° (初始)
  - 遮挡区: φ≈180° (固定)
  - Spot可见: ✓ (在前侧)

时刻 t=0.5T:
  - 观察者: φ=0° (固定)
  - Spot位置: φ≈180° (转动到后侧)
  - 遮挡区: φ≈180° (固定)
  - Spot可见: ✗ (被遮挡)
```

关键点：
- **观察者不动**：始终从 φ=0° 看
- **遮挡区不动**：始终在 φ≈180°（背向观察者）
- **Spot在动**：随盘面转动，进出遮挡区
- **光谱变化**：spot进出遮挡区导致可见信号变化

## 影响范围

需要更新的代码：
- ✅ `core/grid_tom.py` - 遮挡计算核心
- ✅ `core/velspace_DiskIntegrator.py` - 调用处
- ✅ `test/test_stellar_occultation.py` - 测试脚本

对主流程的影响：
- `tomography.py` 和 `SimpleDiskGeometry` 需要确保有 `phi_obs` 属性（默认0.0）
- 如果启用遮挡（`enable_stellar_occultation=1`），需确保几何对象提供：
  - `phi_obs`: 观察者方向（弧度）
  - `inclination_rad`: 倾角（弧度）
  - `stellar_radius`: 恒星半径

## 后续改进建议

1. **参数化观察者方向**：
   - 当前默认 `phi_obs=0`
   - 可扩展为多观察方向的场景（多站观测）

2. **非赤道盘**：
   - 当前假设 z=0 无限薄盘
   - 可扩展为有厚度的盘（需考虑 z 坐标）

3. **非球形恒星**：
   - 当前假设恒星为球体
   - 可考虑离心率、扁率等效应

4. **相对论效应**：
   - 高速转动时的光行时效应
   - 引力弯曲效应（强场情况）

---
**修正时间**: 2025-11-13  
**修正人员**: AI Assistant  
**验证状态**: ✅ 已通过测试
