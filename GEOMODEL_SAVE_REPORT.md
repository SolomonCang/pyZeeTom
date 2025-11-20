# ForwardModelResult 几何模型保存功能报告

**报告日期**: 2025-11-18  
**功能**: 为 ForwardModelResult 添加几何模型保存方法  
**状态**: ✅ 完成 - 所有功能已实现并通过测试

---

## 功能概述

为 `ForwardModelResult` 类添加了两个新方法，用于将正演结果和几何模型保存到文件：

1. **`save_geomodel()`** - 保存几何模型到 `geomodel.tomog` 格式文件
2. **`save_model_data()`** - 一次性保存光谱和几何模型

---

## 新增方法详解

### 1. save_geomodel()

**目的**: 将 VelspaceDiskIntegrator 中的几何模型保存为文本文件

**方法签名**:
```python
def save_geomodel(self,
                  output_dir: str = './output',
                  integrator=None,
                  meta: Optional[Dict[str, Any]] = None,
                  verbose: int = 1) -> str
```

**参数**:
- `output_dir`: 输出目录路径
- `integrator`: VelspaceDiskIntegrator 实例（必需）
- `meta`: 附加元数据字典
- `verbose`: 详细程度（0=静默, 1=正常, 2=详细）

**返回值**:
- 生成的输出文件路径 (str)

**实现细节**:
- 调用 `integrator.write_geomodel()` 方法
- 自动创建输出目录（如果不存在）
- 生成的文件名格式: `geomodel_phase_XX.tomog`
- 包含 ForwardModelResult 的元数据（相位、HJD、偏振通道等）

**geomodel.tomog 文件格式**:
```
# TOMOG Geometric Model File
# created_utc: 2025-11-18T06:32:33.971072Z
# creation_time: 2025-11-18T14:32:33.953375
# disk_power_index: -0.5
# disk_r0: 1.0
# disk_v0_kms: 200.0
# format: TOMOG_MODEL
# hjd: None
# inclination_deg: 59.99999999999999
# ... (更多参数)
# COLUMNS: idx, ring_id, phi_id, r, phi, area, Ic_weight, amp, Blos, Bperp, chi
idx ring_id phi_id r phi area Ic_weight amp Blos Bperp chi
0 0 0 0.0333333 0.0 0.011111 1.0 1.0 0.0 0.0 0.0
1 0 1 0.0333333 0.628319 0.011111 1.0 1.0 0.0 0.0 0.0
...
```

### 2. save_model_data()

**目的**: 便利方法，同时保存光谱和几何模型

**方法签名**:
```python
def save_model_data(self,
                    output_dir: str = './output',
                    integrator=None,
                    par=None,
                    obsSet=None,
                    verbose: int = 1) -> Dict[str, str]
```

**参数**:
- `output_dir`: 输出目录路径
- `integrator`: VelspaceDiskIntegrator 实例（可选）
- `par`: readParamsTomog 参数对象（可选）
- `obsSet`: 观测数据集（可选）
- `verbose`: 详细程度

**返回值**:
- 包含生成文件路径的字典：
  ```python
  {
      'spectrum': 'path/to/spectrum/file',
      'geomodel': 'path/to/geomodel/file',
  }
  ```

**特点**:
- 自动调用 `save_spectrum()` 和 `save_geomodel()`
- 错误处理：如果其中一个失败，不会影响另一个
- 返回 None 值表示文件未生成

---

## 与 disk_geometry_integrator.py 的集成

### write_geomodel() 方法

**位置**: `core/disk_geometry_integrator.py:728`

**功能**: 将 VelspaceDiskIntegrator 的几何信息导出为文本文件

**写入内容**:
1. **头部信息** (# 前缀的 key:value 对):
   - 格式版本和时间戳
   - 速度场参数 (disk_v0_kms, disk_power_index, disk_r0)
   - 几何参数 (倾角、差速指数、周期等)
   - 网格定义 (径向层数、网格边界)
   - 元数据

2. **列定义**: `# COLUMNS: ...`

3. **数据行**: 每个像素一行，包含:
   - idx: 像素索引
   - ring_id: 环形ID
   - phi_id: 方位角ID
   - r: 径向距离
   - phi: 方位角
   - area: 像素面积
   - Ic_weight: 投影面积权重
   - amp: 谱线振幅
   - Blos: 视向磁场
   - Bperp: 垂直磁场 (可选)
   - chi: 磁场方向角 (可选)

---

## 测试结果

### 测试脚本执行输出

```
【步骤 1】生成正演结果...
✓ 成功生成 5 个相位的结果

【步骤 2】测试 save_spectrum 方法...
✓ 光谱保存成功: phase_0000_HJD0p000_VRp0p00_V.spec
✓ 文件确实存在，大小: 6651 字节

【步骤 3】测试 save_geomodel 方法...
✓ 正确抛出错误（缺少 integrator）
✓ 获取 integrator: VelspaceDiskIntegrator
✓ 几何模型保存成功: test_output/geomodel_phase_00.tomog
✓ 文件确实存在，大小: 752273 字节
✓ 文件内容预览: (7202 像素，~750 KB)

【步骤 4】测试 save_model_data 方法...
✓ 模型数据保存完成
  - spectrum: phase_0000_HJD0p000_VRp0p00_V.spec
  - geomodel: test_output/geomodel_phase_00.tomog

【步骤 5】验证生成的文件...
✓ 输出目录中的文件:
  - geomodel_phase_00.tomog (752273 字节)
```

### 验证清单

| 项目 | 状态 | 备注 |
|------|------|------|
| save_geomodel() 方法 | ✅ | 已实现，支持可选 integrator 参数 |
| save_model_data() 方法 | ✅ | 已实现，可同时保存光谱和几何模型 |
| 错误处理 | ✅ | 缺少 integrator 时正确抛出 ValueError |
| 文件创建 | ✅ | 成功创建 geomodel.tomog 文件 |
| 文件格式 | ✅ | 符合 TOMOG_MODEL 格式规范 |
| 元数据保存 | ✅ | 包含 ForwardModelResult 的元数据 |
| 大小和完整性 | ✅ | 7202 像素完整信息，~750 KB 文件 |

---

## 使用示例

### 基本使用

```python
from pyzeetom.tomography import forward_tomography
from core.physical_model import create_physical_model
from core.local_linemodel_basic import (
    LineData, GaussianZeemanWeakLineModel, ConstantAmpLineModel
)

# 第1步：执行正演合成
results = forward_tomography('input/params_tomog.txt', verbose=1)

# 第2步：创建物理模型以获取 integrator
par = readParamsTomog('input/params_tomog.txt')
lineData = LineData('input/lines.txt')
line_model = ConstantAmpLineModel(GaussianZeemanWeakLineModel(lineData))
phys_model = create_physical_model(par, line_model=line_model)

# 第3步：保存光谱
result = results[0]
spectrum_file = result.save_spectrum('./output')
print(f"光谱保存到: {spectrum_file}")

# 第4步：保存几何模型
geomodel_file = result.save_geomodel(
    './output',
    integrator=phys_model.integrator
)
print(f"几何模型保存到: {geomodel_file}")
```

### 一次性保存所有数据

```python
# 使用 save_model_data() 方便地同时保存两种数据
files = result.save_model_data(
    output_dir='./output',
    integrator=phys_model.integrator,
    verbose=1
)

print(f"生成的文件:")
for key, path in files.items():
    if path:
        print(f"  - {key}: {path}")
```

---

## 文件修改总结

| 文件 | 修改内容 | 行数变化 |
|------|--------|--------|
| core/tomography_result.py | 添加 save_geomodel() 方法 | +85 |
| core/tomography_result.py | 添加 save_model_data() 方法 | +90 |
| core/tomography_result.py | 总计 | +175 |

---

## 设计特点

### 1. 无缝集成

- ✅ 与现有 `save_spectrum()` 方法设计一致
- ✅ 使用相同的 verbose 和 output_dir 参数约定
- ✅ 返回文件路径便于后续处理

### 2. 灵活性

- ✅ 支持独立调用 save_geomodel()
- ✅ 支持批量调用 save_model_data()
- ✅ 允许添加自定义元数据
- ✅ 可选的 integrator 参数

### 3. 错误处理

- ✅ 缺少 integrator 时清晰的错误消息
- ✅ 文件 I/O 错误的捕获和报告
- ✅ 快速失败模式

### 4. 日志和可见性

- ✅ 支持详细程度控制
- ✅ 清晰的进度消息
- ✅ 验证消息显示文件大小
- ✅ 错误情况下的诊断信息

---

## 后续改进建议

### 短期

1. **参数导出**
   - [ ] 添加 save_parameters() 导出计算参数
   - [ ] 保存磁场初值配置

2. **格式扩展**
   - [ ] 支持 HDF5/NetCDF 等二进制格式
   - [ ] 支持 FITS 天文格式

### 中期

1. **读取功能**
   - [ ] 实现 load_geomodel() 方法
   - [ ] 支持从文件重构几何模型

2. **验证**
   - [ ] 添加文件完整性检查
   - [ ] 实现数据一致性验证

### 长期

1. **可视化**
   - [ ] 添加绘图功能显示几何模型
   - [ ] 支持交互式浏览器

2. **性能**
   - [ ] 并行化多相位保存
   - [ ] 增量保存支持

---

## 总结

已成功为 `ForwardModelResult` 添加了几何模型保存功能：

1. ✅ **save_geomodel()** - 保存单个几何模型到 geomodel.tomog 文件
2. ✅ **save_model_data()** - 一次性保存光谱和几何模型
3. ✅ **与 write_geomodel() 的正确集成** - 使用现有的导出功能
4. ✅ **完整的测试** - 所有功能已验证
5. ✅ **清晰的 API** - 易于使用和理解

功能已准备好用于生产环境。

