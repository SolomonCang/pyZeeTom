
# pyZeeTom Copilot 快速指南与项目架构

## 一、项目目标与物理场景
本项目旨在开发一个能够反演4个Stokes量（I, Q, U, V）偏振光谱的tomography工具。
核心物理场景：
- 中心天体+星周物质（尘埃团、盘、行星等）刚体/差速环绕运动
- 观测者与中心天体同系，仅通过自转带来的不同phase观测不同视角
- 每相位可获得Stokes I/V/Q/U偏振光谱
- 当前主攻正演模型，后续将引入MEM等反演方法

---

## 二、核心架构与目录功能说明

### 1. core/ 物理与数值核心
| 文件名 | 主要功能 |
| ------ | -------- |
| grid_tom.py | 环状/盘面网格生成与管理，`diskGrid`类，像素属性一维数组存储（r, phi, area, ring_id等） |
| local_linemodel_basic.py | 基础谱线模型，支持Stokes I/V/Q/U，弱场高斯Zeeman等 |
| local_linemodel_ME.py | MEM相关谱线模型（实验/扩展用） |
| velspace_DiskIntegrator.py | 速度空间积分，将局部谱线投影到观测空间，合成全局Stokes谱 |
| mainFuncs.py | 参数解析（如`readParamsTomog`），兼容旧版接口 |
| mem_generic.py | MEM反演通用模块，最大熵算法基础 |
| mem_tomography.py | MEM反演主流程，调度与接口 |
| SpecIO.py | 光谱数据读写，支持多格式 |
| spot_geometry.py | 斑点/结构几何建模与操作 |
| readObs.py | 观测数据读取，标准化为`ObservationProfile`结构 |

### 2. pyzeetom/ 主入口与流程调度
| 文件名 | 主要功能 |
| ------ | -------- |
| tomography.py | Tomography主入口，串联参数读取、网格生成、谱线合成、反演等 |
| __init__.py | 包初始化 |

### 3. test/ 单元测试
pytest风格，覆盖主要流程和边界情况。

### 4. utils/ 辅助工具与可视化
| 文件名 | 主要功能 |
| ------ | -------- |
| dynamic_spec_plot.py | 动态光谱可视化 |
| dynamic_spectrum.py | 动态光谱生成与处理 |
| generate_emission_spots_sim.py | 生成模拟发射斑点分布 |
| visualize_geomodel.py | 几何模型可视化 |

### 5. examples/ 示例脚本
典型流程与用法示例。

### 6. input/ 输入参数与观测数据
| 文件名/目录 | 主要功能 |
| ----------- | -------- |
| params_*.txt | 主控参数文件，控制流程、模型、观测等 |
| lines*.txt | 谱线参数文件（如中心波长、宽度、g因子） |
| inSpec/ | 观测数据（多格式），如LSD/spec/pol等 |

### 7. output/ 主要输出
| 文件名/目录 | 主要功能 |
| ----------- | -------- |
| model_phase_*.lsd | 每相位合成模型谱线（LSD格式） |
| geomodel_*.tomog | 每相位几何模型输出 |
| mem_inversion_result.npz | MEM反演结果（numpy存档） |
| outFitSummary.txt | 汇总信息 |
| outModel/ | 其他模型输出 |

### 8. test_output/ 测试输出
各测试脚本输出结果，便于回归与对比。

### 9. docs/ 文档
开发、接口、格式、测试等详细说明文档。

### 10. tbd/ 待定/实验性代码
实验性谱线模型、参数等。

---

## 三、文件与格式约定
- **参数文件**（如`params_tomog.txt`）：前14行为主控参数，第5行为谱线模型参数，第11行为谱线参数文件，第12行为观测格式，14行后为观测文件名、JD、velR等。
- **谱线参数文件**（如`lines.txt`）：每行`wl0 sigWl g`
- **观测数据**：支持多格式，需包含`wl, specI, specV, specQ, specU, sigma`等。
- **输出文件**：每相位生成`.model`文件，汇总信息写入`outFitSummary.txt`。

---

## 四、开发与风格约定
- 所有像素属性、谱线参数等均以一维数组存储，广播到(Nlambda, Npix)形状
- I分量基线固定为1.0，盘积分时用Ic_weight加权
- `Blos`为视向分量，Q/U用`Bperp`和`chi`（弧度），弱场近似下Q/U为二阶项
- `doppl_deriv`等接口需支持向量化，便于反演/优化
- 主入口为`pyzeetom/tomography.py`，运行需确保`PYTHONPATH`包含仓库根

### 光谱输出一致性（SpecIO）
- 解析类型由`SpecIO._heuristic_guess`与`_assign_columns_by_type`统一：
	- 6列仅对应`spec_pol`（Wav Int Pol Null1 Null2 sigma_int）；`lsd_i`严格3列。
- 写出时优先使用`SpecIO.write_model_spectrum(..., file_type_hint=...)`明确结构：
	- `file_type_hint`支持：`spec_pol`, `spec_i`, `spec_i_simple`, `lsd_pol`, `lsd_i`, `lsd_i_simple`
	- 当不考虑磁场（V/Q/U未生成）时，偏振/Null列自动零填充，保持结构一致。
	- 不再“强制匹配输入结构（force）”，请显式传入`file_type_hint`以保证输出结构与需求对齐。

---

## 五、典型扩展点
- **自定义谱线模型**：继承`BaseLineModel`，实现`compute_local_profile(wl_grid, amp, Blos, ...)`，返回dict含I/V/Q/U
- **新观测格式**：实现兼容`ObservationProfile`的数据读取器
- **反演/优化**：MEM等反演方法以模块化方式集成，接口与正演流程解耦

---

## 六、测试与调试
- 单元测试见`test/`目录，推荐`pytest`运行
- 可在REPL中import各模块，手动构造参数测试

---

## 七、注意事项
- amp/Blos等长度需与像素数一致，否则抛出ValueError
- 速度单位以km/s为主，部分旧代码可能为m/s，注意核查

---

## 八、后续规划
- 正演流程完善后，将引入MEM等反演方法，结构将拆分为参数解析、谱线模型、积分器、优化/熵等子模块
- 详细参数文件格式、反演接口等将随开发进度补充

---

如需进一步说明或示例，请具体指出需求。
