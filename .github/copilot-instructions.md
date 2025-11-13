## Copilot 快速指南 — pyZeeTom 项目架构与开发约定

### 项目目标
本项目旨在开发一个能够反演4个Stokes量（I, Q, U, V）偏振光谱的tomography工具。核心物理场景为：
- 存在一个中心天体，周围有星周物质（如尘埃团、盘、行星、小天体等）以刚体或差速方式环绕运动。
- 观测者与中心天体处于同一惯性系，只能通过天体自转带来的不同“phase”观测不同视角。
- 每一观测相位可获得Stokes I及VQU分量的偏振光谱。
- 当前主攻正演模型（forward modeling），后续将引入MEM等方法实现反演（inversion）。

---

### 物理与几何建模
- **环状几何模型**：星周物质被划分为多个同心环（ring），每环可有不同物理属性，支持刚体同步自转和差速运动。
- **相位采样**：通过天体自转，采样不同观测相位（phase），每相位合成一组Stokes光谱。
- **Stokes谱线**：支持I, V, Q, U四个分量，局部线型模型支持弱场高斯Zeeman等。

---

### 核心代码结构
- `core/grid_tom.py`：环状/盘面网格生成，`diskGrid`类，像素属性一维数组存储（r, phi, area, ring_id等）。
- `core/local_linemodel_basic.py` / `local_linemodel_ME.py`：局部谱线模型，支持Stokes I/V/Q/U，弱场高斯Zeeman等。
- `core/velspace_DiskIntegrator.py`：速度空间积分，将局部谱线投影到观测空间，合成全局Stokes谱。
- `core/readObs.py`：观测数据读取，支持多格式（LSD/spec/pol等），标准化为`ObservationProfile`结构。
- `pyzeetom/tomography.py`：主流程入口，串联参数读取、网格生成、谱线合成等。
- `core/mainFuncs.py`：参数解析（`readParamsTomog`），兼容旧版接口。
- `memSimple3.py`等：MEM反演相关，当前未启用，后续反演阶段将重构。

---

### 文件与格式约定
- **参数文件**（如`params_tomog.txt`）：
  - 前14行为主控参数（部分磁场/亮度参数暂未用），第5行定义谱线模型参数（如`lineAmpConst k_QU enableV enableQU`）。
  - 第11行可指定谱线参数文件（如`lines.txt`），第12行可指定观测格式（如`lsd_i`/`spec_i`/`auto`）。
  - 14行后为观测文件名、JD、velR等。
- **谱线参数文件**（如`lines.txt`）：每行`wl0 sigWl g`
- **观测数据**：支持多格式，自动识别，需包含`wl, specI, specV, specQ, specU, sigma`等。
- **输出文件**：每相位生成`.model`文件（LSD格式），汇总信息写入`outFitSummary.txt`。

---

### 代码风格与开发约定
- **数组广播**：所有像素属性、谱线参数等均以一维数组存储，广播到(Nlambda, Npix)形状。
- **Stokes基线**：I分量基线固定为1.0（吸收线amp<0），盘积分时用Ic_weight加权，不改变基线定义。
- **磁场约定**：`Blos`为视向分量，Q/U用`Bperp`和`chi`（弧度），弱场近似下Q/U为二阶项。
- **导数接口**：`doppl_deriv`等接口需支持向量化，便于后续反演/优化。
- **路径与导入**：主入口为`pyzeetom/tomography.py`，运行需确保`PYTHONPATH`包含仓库根。

---

### 典型扩展点
- **自定义谱线模型**：继承`BaseLineModel`，实现`compute_local_profile(wl_grid, amp, Blos, ...)`，返回dict含I/V/Q/U。
- **新观测格式**：实现兼容`ObservationProfile`的数据读取器，确保能被主流程识别。
- **反演/优化**：后续MEM等反演方法将以模块化方式集成，接口与正演流程解耦。

---

### 测试与调试
- 单元测试：见`test/`目录，推荐`pytest`运行。
- 交互调试：可在REPL中import各模块，手动构造参数测试。

---

### 注意事项
- **广播错误**：amp/Blos等长度需与像素数一致，否则抛出ValueError。
- **单位一致性**：速度单位以km/s为主，部分旧代码可能为m/s，注意核查。
- **废弃模块**：`memSimple3.py`、`resp_tom.py`等仅兼容旧流程，勿在新流程调用。

---

### 后续规划
- 正演流程完善后，将引入MEM等反演方法，结构将拆分为参数解析、谱线模型、积分器、优化/熵等子模块。
- 详细参数文件格式、反演接口等将随开发进度补充。

如需进一步说明或示例，请具体指出需求。
