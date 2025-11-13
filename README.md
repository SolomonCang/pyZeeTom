
# pyZeeTom

**pyZeeTom** 是一个用于反演和正演4个Stokes量（I, Q, U, V）偏振光谱的tomography工具。

## 项目简介

本项目面向如下物理场景：
- 存在一个中心天体，周围有星周物质（尘埃团、盘、行星、小天体等）以刚体或差速方式环绕运动。
- 观测者与中心天体处于同一惯性系，只能通过天体自转带来的不同“phase”观测不同视角。
- 每一观测相位可获得Stokes I及VQU分量的偏振光谱。
- 当前主攻正演模型，后续将引入MEM等方法实现反演。

## 主要特性

- 多种观测数据格式支持（LSD/spec/pol/I/V/Q/U）
- 弱场高斯Zeeman线型与自定义谱线模型
- 环状/盘面网格，支持刚体与差速运动
- 速度空间积分，合成全局Stokes谱
- **时间演化支持**：差速转动导致的盘面结构变化
- **相位计算**：自动根据 JD、JD0 和周期计算观测相位
- 结构清晰，便于扩展反演/优化模块

## 快速开始

### 安装

```bash
pip install -e .
# 或包含开发依赖
pip install -e .[dev]
```

### 运行正演合成

```python
from pyzeetom import tomography
tomography.run_tomography('input/params_tomog.txt')
```

### 输入文件说明

1. **参数文件**（如`params_tomog.txt`）：主控参数、谱线模型参数、观测文件列表等
2. **谱线参数文件**（如`lines.txt`）：每行`wl0 sigWl g`
3. **观测数据**：支持多格式，需含Stokes I/V/Q/U及误差

示例见 `input/` 目录。

## 核心结构


### 模型构筑基本逻辑
- **观察者与网格结构不动**：观测者与中心天体处于同一惯性系，盘面网格（r, φ）结构在所有相位保持不变。
- **物质结构随自转/差速转动演化**：盘面上的物质（如团块、亮度/磁场分布）会因恒星自转或差速转动而在 φ 方向发生平移或剪切，具体由 $\phi(t) = \phi_0 + \Omega(r) t$ 控制。
- **每一观测相位**：重新计算物质分布（如团块位置），但网格本身和投影几何不变，仅物质属性（如亮度、磁场）随时间演化。

> 详细结构与开发约定见 [copilot-instructions.md](.github/copilot-instructions.md)

## 输入参数说明

### 核心恒星参数（params_tomog.txt 前14行）

**行0: 视向参数与转动**
- `inclination` (deg)：倾角，影响视向速度投影 `vlos = vφ·sin(i)·sin(φ)` 和盘面投影面积
- `vsini` (km/s)：赤道视向速度，计算赤道线速度 `veq = vsini/sin(i)`
- `period` (day)：参考半径 r₀ 处自转周期，定义参考角速度 `Ω₀ = 2π/P`
- `pOmega`：差速转动幂律指数，控制角速度径向分布 `Ω(r) = Ω₀·(r/r₀)^pOmega`
  - `pOmega=0.0`：刚体转动
  - `pOmega=-0.5`：类开普勒（盘）
  - `pOmega=-1.0`：恒定角动量

**行1: 物理尺度与网格定义（⭐重要修改）**
- `mass` (M☉)：恒星质量，用于计算同步轨道半径并输出
- `radius` (R☉)：恒星半径，作为参考半径 r₀
- `Vmax` (km/s，可选)：**模式1**：直接指定盘面最大速度，>0 时忽略 r_out
- `r_out` (R*，可选)：**模式2**：盘外半径（单位：恒星半径），Vmax=0 时从此计算
- `enable_occultation` (0/1，可选)：恒星遮挡开关，1=启用（考虑无限薄赤道盘被恒星本体遮挡）

**网格构建两种模式：**
1. **直接速度模式**：设 `Vmax > 0`（如300 km/s），网格由 Vmax 和 nr 直接定义，不依赖 radius/vsini/inclination
2. **物理派生模式**：设 `Vmax = 0`，指定 `r_out`，根据差速公式计算：
   - `Vmax = veq · (r_out/radius)^(pOmega+1)`
   - 网格外半径 `r_out_grid` 由上式反推

**行2: 网格分辨率**
- `nRingsStellarGrid`：径向环数，每环宽度 `Δr = r_out_grid/nr`

**行3-4: 反演控制（当前正演阶段未启用）**
- `targetForm/targetValue/numIterations`：MEM反演目标
- `test_aim`：收敛阈值

**行5: 谱线模型参数**
- `lineAmpConst`：谱线振幅（<0吸收，>0发射），控制线深 `I = 1 + amp·G(λ)`
- `k_QU`：Q/U二阶项比例因子，调节弱场横向分量灵敏度
- `enableV`：是否计算 Stokes V（1=启用，0=禁用）
- `enableQU`：是否计算 Stokes Q/U（1=启用，0=禁用）

**行6-10: 初始化与反演设置（遗留字段，当前未用）**
- 保留与旧版MEM反演兼容性

**行11: 仪器与谱线文件**
- `spectralResolution`：**光谱分辨率**（如65000），自动转换为 FWHM：`FWHM (km/s) = c/R`
- `lineParamFile`：谱线参数文件路径（格式：`wl0 sigWl g`）

**行12: 速度范围与观测格式**
- `velStart/velEnd` (km/s)：速度网格范围，定义合成谱的采样区间
- `obsFileType`：观测格式提示（`auto`/`lsd_i`/`lsd_pol`/`spec_i`等）

**行13: 时间参考点**
- `jDateRef` (HJD)：参考历元 HJD₀，用于相位计算 `phase = (JD - HJD₀)/period`

**行14+: 观测序列**
- 每行：`filename  JD  velR`
  - `JD`：观测时刻（Heliocentric Julian Date）
  - `velR` (km/s)：径向速度修正（加到观测速度轴）

### 谱线参数文件（lines.txt）

- `wl0` (Å)：谱线中心波长
- `sigWl` (Å)：谱线高斯宽度 σ，控制本征线宽
- `g`：朗德因子，决定Zeeman分裂强度
  - Stokes V 系数：`Cg = -2.0 × 4.6686e-12 × wl0² × g`
  - Q/U 系数：`C2 = (4.6686e-12 × wl0² × g)² × k_QU`

### 正演计算流程

1. **网格生成**（`grid_tom.py`）：
   - 等Δr分环，每环像素数自适应（∝ r），每像素面积 `dA = r·Δr·Δφ`
   - 根据 Vmax 或 r_out 确定网格外半径
2. **速度场计算**（`velspace_DiskIntegrator.py`）：
   - 外侧（r≥r₀）：幂律 `vφ(r) = veq·(r/r₀)^(pOmega+1)`
   - 内侧（r<r₀）：余弦减速序列，确保 r=0 处 v=0
   - 视向速度：`vlos = vφ·sin(i)·sin(φ + φ₀)`
3. **恒星遮挡**（可选，`grid_tom.py`）：
   - 将盘面像素投影到观测者视角
   - 若投影距离 < R* 且在恒星后方，该像素被遮挡（权重设为0）
4. **局部谱线合成**（`local_linemodel_basic.py`）：
   - 弱场高斯Zeeman线型，每像素计算 I/V/Q/U 分量
   - 归一化：`d = (λ-λ₀)/σ`，`G = exp(-d²)`
   - Stokes I：`1 + amp·G`（基线=1）
   - Stokes V：`Cg·Blos·amp·G·d/σ`（一阶线性项）
   - Stokes Q/U：`-C2·Bperp²·amp·G(1-2d²)/σ²·cos/sin(2χ)`（二阶项）
5. **盘积分**（`velspace_DiskIntegrator.py`）：
   - 按 vlos 将像素投影到观测速度网格
   - 权重：`w = Ic_weight·area_proj`（连续谱强度×投影面积）
   - 遮挡像素权重为0，不参与积分
   - 累加生成全局 Stokes 谱
6. **仪器卷积与输出**：
   - 高斯卷积（FWHM = c/R）模拟仪器分辨率
   - 归一化基线为1.0
7. **时间演化**（`spot_geometry.py`）：
   - 团块位置演化：`φ(t) = φ₀ + Ω(r)·t`
   - 每观测相位重新计算团块分布，实现差速剪切效应
8. **同步轨道输出**（`mainFuncs.py`）：
   - 根据质量和周期计算：`r_sync³ = G·M·P²/(4π²)`
   - 输出到屏幕供参考

## 文件与格式约定

- **参数文件**：前14行为主控参数，第5行为谱线模型参数，第11/12行可指定谱线/观测格式，14行后为观测文件名、JD、velR等
- **谱线参数文件**：每行`wl0 sigWl g`
- **观测数据**：需包含`wl, specI, specV, specQ, specU, sigma`等
- **输出文件**：每相位生成`.model`文件，汇总信息写入`outFitSummary.txt`

## 测试与开发

```bash
# 运行全部测试
python -m pytest test/
# 单元测试示例
python -m pytest -q test/test_tomography_random_spots.py
```

### 代码风格与注意事项
- 所有像素/谱线参数均以一维数组存储，广播到(Nlambda, Npix)
- I分量基线固定为1.0，盘积分时用Ic_weight加权
- 磁场约定：Blos为视向分量，Q/U用Bperp和chi（弧度）
- 速度单位以km/s为主
- 废弃模块（如memSimple3.py、resp_tom.py）仅兼容旧流程，勿在新流程调用

详细开发约定见 [copilot-instructions.md](.github/copilot-instructions.md)

## 反演与扩展

- 反演（如MEM最大熵方法）将在正演流程完善后集成，接口将与正演解耦
- 支持自定义谱线模型、观测格式、优化方法等扩展

## 参考文档

- [项目结构](PROJECT_STRUCTURE.md)
- [时间演化与差速转动](docs/TIME_EVOLUTION.md) ⭐ 新功能！
- [AI代理/开发约定](.github/copilot-instructions.md)
- [MEM说明](core/MEM_README.md)

## 引用

如使用本项目，请引用：

```bibtex
@software{pyzeetom,
  title = {pyZeeTom: Stellar Spectropolarimetric Tomography},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/SolomonCang/pyZeeTom}
}
```

## 许可

[待添加许可证]

## 贡献

欢迎提交 Issue 和 Pull Request！

开发指南见 [copilot-instructions.md](.github/copilot-instructions.md)
