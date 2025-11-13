# pyZeeTom 模型导出与光谱输出接口说明

本文档简要描述新增的几何模型导出与光谱写入功能。

---

## 快速测试

运行完整的读写流程测试（自动生成模拟数据）：

```bash
python test/test_spec_io_workflow.py
```

此测试将：
1. 生成 5 个相位的合成光谱（无噪声）→ `output/model_phase_*.lsd`
2. 添加噪声作为"观测"数据 → `input/inSpec/obs_phase_*.lsd`
3. 读取观测并重写到 → `output/outSpec/rewritten_obs_*.lsd`
4. 导出一个示例几何模型 → `output/geomodel_test.tomog`

---

## 一、几何模型导出（geomodel.tomog）

**功能**：在几何建立完成后，将每个像素的物理属性（强度响应、磁场等）及网格结构导出为文本文件，便于：
- 保存完整物理模型以供复现分析；
- 在不同观测相位间共享基础结构；
- 用于后续反演/MEM算法的输入。

**接口**：`VelspaceDiskIntegrator.write_geomodel(filepath, meta=None)`
- 定义位置：`core/velspace_DiskIntegrator.py`
- 调用时机：建立积分器后，可在任意相位调用（通常在第一个相位）。
- 文件结构：
  ```
  # TOMOG Geometric Model File
  # format: TOMOG_MODEL
  # version: 1
  # created_utc: 2025-11-13T...Z
  # wl0_nm: 500.0
  # inclination_deg: 60.0
  # phi0: 0.0
  # pOmega: -0.5
  # r0_rot: 1.0
  # period: 1.0
  # nr: 60
  # r_edges: [0.0, 0.0833, ...]
  # target: HD12345  （若传入meta）
  # ...
  # COLUMNS: idx, ring_id, phi_id, r, phi, area, Ic_weight, A, Blos [, Bperp, chi]
  <数据行：每像素一行>
  ```
- 头部包含可重建模型所需的全部参数（几何、网格分辨率、差速转动参数等）。
- 每行为一个像素，包含：
  - `idx`：像素序号
  - `ring_id`, `phi_id`：环号与方位序号
  - `r`, `phi`, `area`：极坐标半径（R_star单位）、方位角（弧度）、像素面积
  - `Ic_weight`：几何权重（投影面积×响应）
  - `A`：响应因子（>1发射增强，<1减弱）
  - `Blos`：视向磁场（G）
  - `Bperp`, `chi`：垂向磁场与方位角（可选，若 geom 提供）

**读取接口**：`VelspaceDiskIntegrator.read_geomodel(filepath)` → `(geom_like, meta, table)`
- 返回三元组：
  - `geom_like`：重建的几何对象（兼容积分器所需属性）
  - `meta`：头部字典
  - `table`：像素数据表（dict of arrays）

**典型用例**：
```python
# 导出
inte.write_geomodel("output/geomodel_phase0.tomog", meta={
    "target": "HD12345",
    "period": 1.2,
    "phase0": 0.0,
})

# 读取
geom, meta, table = VelspaceDiskIntegrator.read_geomodel("output/geomodel_phase0.tomog")
print(f"Target: {meta['target']}, numPoints: {geom.grid.numPoints}")
```

---

## 二、模型光谱输出

**功能**：将正演积分器结果（I, V, Q, U）写入标准格式文件，便于：
- 与观测数据对比（LSD或spec格式）；
- 检查多相位模型演化；
- 为反演提供"伪观测"输入。

### 2.1 单文件写入

**接口**：`SpecIO.write_model_spectrum(filename, x, Iprof, V=None, Q=None, U=None, sigmaI=None, fmt='lsd', header=None)`
- 定义位置：`core/SpecIO.py`
- 参数：
  - `filename`：输出路径
  - `x`：速度（km/s，fmt='lsd'）或波长（nm，fmt='spec'）
  - `Iprof`：Stokes I 归一化轮廓
  - `V, Q, U`：Stokes 分量（可选，默认填0）
  - `sigmaI`：I的误差（可选，默认填0）
  - `fmt`：`'lsd'` 或 `'spec'`，决定列名
  - `header`：自定义头部键值对（如 `{"phase_index": "0", "JD": "2459001.5"}`）
- 输出格式（示例）：
  ```
  # phase_index: 0
  # JD: 2459001.5
  # RV Int sigma_int V Q U
  -150.0  1.00023e+00  0.0e+00  -1.234e-03  0.0e+00  0.0e+00
  -148.0  9.99876e-01  0.0e+00  -1.256e-03  0.0e+00  0.0e+00
  ...
  ```

### 2.2 批量写入（多相位序列）

**接口**：`SpecIO.save_results_series(results, basepath='model_phase', fmt='lsd')` → `List[文件路径]`
- 适用场景：`tomography.main()` 返回的 `[(v, I, V), ...]` 列表。
- 自动生成文件名：`{basepath}_00.{fmt}`, `{basepath}_01.{fmt}`, ...
- 返回：写入的文件路径列表。

**典型用例**：
```python
results = tomography.main(par)
paths = SpecIO.save_results_series(results, basepath="output/model_phase", fmt="lsd")
print(f"已写入 {len(paths)} 个相位模型光谱：{paths[0]}, ...")
```

---

## 三、在 tomography.py 中的集成示例

主流程 `pyzeetom.tomography.main()` 在返回结果前增加了：
1. **导出第一相位几何模型**（若 `verbose=True`）：
   - 输出到 `output/geomodel_phase0.tomog`
   - 包含目标名称、周期、观测相位等元信息
2. **保存所有相位模型光谱**（若 `verbose=True`）：
   - 输出到 `output/model_phase_00.lsd`, `output/model_phase_01.lsd`, ...
   - 格式为 LSD（速度空间），包含 I, V 分量

用户可在主流程外自行调用这两个接口以自定义输出路径/格式。

---

## 四、兼容性与扩展性

- **文件格式**：纯文本，跨平台兼容，便于版本控制与手工审阅。
- **反演接口预留**：`geomodel.tomog` 头部保留全部网格/几何参数，后续MEM反演可直接读取此结构。
- **老接口不变**：原 `readObs.py` 重命名为 `SpecIO.py`，所有读入接口（`loadObsProfile`/`obsProfSetInRange`）保持不变，确保向后兼容。

---

## 五、快速上手

```python
from pyzeetom import tomography
import core.SpecIO as SpecIO
from core.velspace_DiskIntegrator import VelspaceDiskIntegrator

# 运行正演
results = tomography.main(par, obsSet, lineData, verbose=1)
# → 自动导出 output/geomodel_phase0.tomog 和 output/model_phase_*.lsd

# 手动读取模型
geom, meta, table = VelspaceDiskIntegrator.read_geomodel("output/geomodel_phase0.tomog")
print(f"恢复几何：{geom.grid.numPoints} 像素，周期 {meta.get('period')} 天")
```

---

如需进一步定制（如支持 Q/U 写入、多相位模型合并等），请参考 `SpecIO.py` 中的函数文档字符串。
