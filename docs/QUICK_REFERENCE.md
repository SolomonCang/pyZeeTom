# 时间演化功能快速参考

## 参数文件修改

在 `params_tomog.txt` 中：

```
# 第0行：添加 pOmega（差速转动指数）
inclination  vsini  period  pOmega
60.0         25.0   1.0     -0.5

# 第13行：添加 jDateRef（参考时刻）
jDateRef  (HJD0)
2450000.5
```

## pOmega 值速查

| pOmega | 类型 | 说明 | 适用场景 |
|--------|------|------|---------|
| 0.0 | 刚体转动 | 所有环同步 | 表面亮斑、小尺度磁场 |
| -0.5 | 开普勒型 | 内快外慢（吸积盘标准） | 吸积盘、行星环 |
| -1.0 | 恒定角动量 | 极端差速 | 特殊物理场景 |

## 相位计算公式

```
phase = (JD - JD0) / period
```

## 角位移计算

### 刚体转动（pOmega=0）
```
Δφ = 2π × phase
```
所有半径处转动相同角度

### 差速转动（pOmega≠0）
```
Δφ(r) = 2π × phase × (r/r0)^pOmega
```
不同半径处转动不同角度

## 代码示例

### 手动计算相位
```python
from core.mainFuncs import compute_phase_from_jd

phase = compute_phase_from_jd(
    jd=2450001.0,      # 观测时刻
    jd_ref=2450000.0,  # 参考时刻
    period=2.0         # 周期（天）
)
# 结果：phase = 0.5
```

### 手动计算方位角演化
```python
from core.grid_tom import diskGrid

grid = diskGrid(nr=60, r_out=1.0)
phi_new = grid.rotate_to_phase(
    phase=0.25,    # 1/4 周期
    pOmega=-0.5,   # 开普勒型
    r0=1.0,        # 参考半径
    period=1.0     # 周期
)
```

### 自动流程
```python
from pyzeetom import tomography

# 自动读取参数文件并计算所有相位
results = tomography.main('input/params_tomog.txt')
```

## 快速诊断

### 检查相位是否正确计算
```python
par = readParamsTomog('params_tomog.txt')
print(par.phases)  # 应该是一个数组
```

### 检查差速转动效果
```python
# 内圈应转得比外圈快（对于 pOmega<0）
grid = diskGrid(nr=10, r_out=2.0)
phase = 0.5
phi = grid.rotate_to_phase(phase, pOmega=-0.5, r0=1.0)

inner_idx = grid.r < 1.0
outer_idx = grid.r > 1.0

delta_phi = (phi - grid.phi) % (2*np.pi)
print(f"内圈平均Δφ: {delta_phi[inner_idx].mean():.3f}")
print(f"外圈平均Δφ: {delta_phi[outer_idx].mean():.3f}")
# 内圈应该更大
```

## 常见问题

### Q: 如何关闭时间演化？
A: 有三种方法：
1. 不在参数文件添加第13行（jDateRef）
2. 设置 `pOmega=0.0`（刚体转动）
3. 所有观测使用相同的 JD

### Q: phase 可以是负数或大于1吗？
A: 可以！
- 负数：观测时刻在参考时刻之前
- >1：观测跨越多个周期
- 所有值都是有效的

### Q: 如何选择 pOmega？
A: 根据物理场景：
- 表面现象（亮斑、磁场）：0.0
- 吸积盘：-0.5
- 不确定：先用 0.0（刚体），再尝试 -0.5

### Q: 参考半径 r0 应该设为多少？
A: 通常设为星球半径（`radius` 参数值）

### Q: 时间演化会影响计算速度吗？
A: 几乎不影响，计算开销可忽略

## 单位注意事项

- **JD, JD0**: Julian Date（天数，可以很大如2450000）
- **period**: 天
- **pOmega**: 无量纲（幂律指数）
- **phi**: 弧度
- **r, r0**: 与 grid 单位一致（通常归一化为星球半径）

## 测试命令

```bash
# 运行时间演化测试
python test/test_phase_evolution.py

# 运行完整测试套件
python -m pytest test/

# 运行特定测试
python -m pytest test/test_phase_evolution.py -v
```

## 更多信息

- 详细文档：[docs/TIME_EVOLUTION.md](TIME_EVOLUTION.md)
- 实现总结：[docs/IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- 项目说明：[README.md](../README.md)
