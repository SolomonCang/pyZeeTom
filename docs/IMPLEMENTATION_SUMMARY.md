# 时间演化功能实现总结

## 更新概述

针对 `input/params_tomog.txt` 中新增的 HJD0 参考点参数，完成了相位计算和差速转动导致的盘面结构时间演化功能。

## 修改的文件

### 1. core/mainFuncs.py
- ✅ 新增 `compute_phase_from_jd()` 函数
  - 根据 JD、JD0 和 period 计算观测相位
  - 支持单个值和数组输入
  
- ✅ 修改 `readParamsTomog` 类
  - 在 `__init__` 中自动计算并存储 `self.phases` 数组
  - 每个观测时刻对应一个相位值

### 2. core/grid_tom.py
- ✅ 新增 `diskGrid.rotate_to_phase()` 方法
  - 根据相位和差速转动参数更新像素方位角
  - 支持刚体转动（pOmega=0）和差速转动（pOmega≠0）
  - 正确处理开普勒型（pOmega=-0.5）等常见情况
  - 返回演化后的方位角数组，不修改原始网格

### 3. core/velspace_DiskIntegrator.py
- ✅ 修改 `VelspaceDiskIntegrator.__init__()`
  - 在初始化时计算演化后的方位角 `phi_evolved`
  - 响应函数和投影计算使用 `phi_evolved` 而非原始 `grid.phi`
  - 正确实现时间相关的盘面结构

### 4. pyzeetom/tomography.py
- ✅ 修改 `SimpleDiskGeometry` 类
  - 新增 `pOmega`, `r0`, `period` 属性
  - 这些参数被传递给积分器用于时间演化计算
  
- ✅ 修改 `main()` 函数
  - 从参数对象获取差速转动参数
  - 为每个观测时刻传递正确的 `time_phase`
  - 确保每个观测使用对应的盘面结构

### 5. 测试文件
- ✅ 创建 `test/test_phase_evolution.py`
  - 测试相位计算函数
  - 测试刚体转动
  - 测试差速转动
  - 测试周期性
  - 测试与参数文件的集成
  - **所有测试通过 ✓**

### 6. 文档
- ✅ 创建 `docs/TIME_EVOLUTION.md`
  - 详细的功能说明
  - 参数定义和物理意义
  - 使用示例和代码片段
  - 注意事项和最佳实践
  
- ✅ 更新 `README.md`
  - 在"主要特性"中添加新功能说明
  - 添加指向详细文档的链接

## 核心物理概念

### 相位计算
```
phase = (JD - JD0) / period
```
- JD: 观测时刻的 Julian Date
- JD0: 参考时刻（第13行参数）
- period: 自转周期（第0行参数）

### 差速转动
角速度幂律：`Ω(r) = Ω_ref × (r/r0)^pOmega`

各像素的角位移：
```python
if pOmega == 0:
    # 刚体转动：所有环同步
    Δφ = 2π × phase
else:
    # 差速转动：与半径相关
    Δφ(r) = 2π × phase × (r/r0)^pOmega
```

常见 pOmega 值：
- **0.0**: 刚体转动（表面亮斑、磁场）
- **-0.5**: 开普勒型（吸积盘）
- **-1.0**: 恒定角动量

### 时间演化效果

以 pOmega=-0.5、phase=0.5（半个周期）为例：
- r=r0 处：转动 π rad
- r=0.5r0 处：转动 π×√2 ≈ 4.44 rad（内圈转得快）
- r=2r0 处：转动 π/√2 ≈ 2.22 rad（外圈转得慢）

这导致盘面结构随时间"撕扯变形"。

## 使用方法

### 参数文件设置
```
# params_tomog.txt

# 第0行：包含差速转动指数 pOmega
60.0  25.0  1.0  -0.5

# 第13行：参考时刻 jDateRef
2450000.5

# 第14+行：观测列表
obs/phase000.lsd  2450000.500  0.0
obs/phase001.lsd  2450000.625  0.0
obs/phase002.lsd  2450000.750  0.0
```

### Python 代码
```python
from pyzeetom import tomography

# 运行正演，自动考虑时间演化
results = tomography.main()

# 每个结果对应一个观测时刻
# 已正确计算差速转动导致的结构变化
```

## 测试验证

运行测试：
```bash
python test/test_phase_evolution.py
```

测试结果：
```
✓ compute_phase_from_jd 测试通过
✓ 刚体转动测试通过 (Δφ = 1.570796 rad)
✓ 差速转动测试通过 (pOmega=-0.5)
  - 内圈平均Δφ=3.482 rad
  - 外圈平均Δφ=2.514 rad
✓ 周期性测试通过
✓ 参数集成测试通过
```

## 兼容性说明

### 向后兼容
- 如果参数文件**没有**第13行（jDateRef），则 `par.phases = None`
- 积分器收到 `time_phase=None` 时，使用原始网格方位角
- 即：老的参数文件仍能正常运行，只是不考虑时间演化

### 刚体转动特例
- 设置 `pOmega=0.0` 即可使用刚体转动
- 所有环同步转动，与之前行为一致

## 代码质量

- ✅ 所有修改均有详细注释
- ✅ 遵循项目代码风格
- ✅ 使用 NumPy 数组广播，性能优化
- ✅ 完整的单元测试覆盖
- ✅ 详细的用户文档

## 后续建议

1. **可视化工具**：创建动画展示盘面演化
2. **反演支持**：在 MEM 反演中拟合 pOmega 参数
3. **非幂律模型**：支持更复杂的角速度分布
4. **性能优化**：对于大网格，缓存演化后的方位角

## 总结

本次更新成功实现了：
- ✅ 相位自动计算
- ✅ 差速转动的盘面结构时间演化
- ✅ 完整的测试验证
- ✅ 详细的用户文档
- ✅ 向后兼容性

所有功能已集成到主流程，用户只需在参数文件中添加第13行 `jDateRef` 即可启用时间演化功能。
