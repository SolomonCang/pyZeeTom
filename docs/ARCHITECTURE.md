# pyZeeTom 项目架构详解

**最后更新**: 2025-11-15  
**版本**: Phase 2.5.4.1（重构完成）

---

## 目录
1. [项目概述](#项目概述)
2. [核心架构设计](#核心架构设计)
3. [模块详解](#模块详解)
4. [数据流与工作流](#数据流与工作流)
5. [物理模型](#物理模型)
6. [扩展与集成](#扩展与集成)

---

## 项目概述

**pyZeeTom** 是一个用于反演和正演4个Stokes量（I, Q, U, V）偏振光谱的tomography工具。

### 物理场景
- **中心天体+星周物质**：存在一个中心天体，周围有星周物质（尘埃团、盘、行星、小天体等）以刚体或差速方式环绕运动
- **相位观测**：观测者与中心天体处于同一惯性系，只能通过天体自转带来的不同"phase"观测不同视角
- **多通道观测**：每一观测相位可获得Stokes I及VQU分量的偏振光谱
- **工作模式**：当前主攻正演模型，后续将引入MEM等方法实现反演

### 主要特性
- ✓ 多种观测数据格式支持（LSD/spec/pol/I/V/Q/U）
- ✓ 弱场高斯Zeeman线型与自定义谱线模型
- ✓ 环状/盘面网格，支持刚体与差速运动
- ✓ 速度空间积分，合成全局Stokes谱
- ✓ 时间演化支持：差速转动导致的盘面结构变化
- ✓ 相位计算：自动根据JD、JD0和周期计算观测相位
- ✓ MEM反演：最大熵方法重建磁场分布

---

## 核心架构设计

### 分层架构

```
┌─────────────────────────────────────────────────┐
│  用户接口层 (UI Layer)                           │
│  pyzeetom/tomography.py                         │
│  - forward_tomography()                         │
│  - inversion_tomography()                       │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  工作流执行引擎 (Workflow Layer)                  │
│  - tomography_forward.py                        │
│  - tomography_inversion.py                      │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  配置与结果容器 (Config & Result Layer)          │
│  - tomography_config.py (ForwardModelConfig,   │
│                         InversionConfig)       │
│  - tomography_result.py (ForwardModelResult,   │
│                         InversionResult)       │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  物理计算核心 (Physics & Integration Layer)      │
│  - velspace_DiskIntegrator.py                  │
│    (速度空间磁盘积分器)                         │
│  - local_linemodel_basic.py                    │
│    (谱线模型)                                  │
│  - mem_tomography.py                           │
│    (MEM适配层)                                 │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  基础工具库 (Utility Layer)                      │
│  - grid_tom.py (网格生成与管理)                 │
│  - disk_geometry.py (盘几何)                    │
│  - SpecIO.py (光谱IO)                          │
│  - mainFuncs.py (参数解析)                     │
│  - mem_generic.py (通用MEM算法)                │
│  - iteration_manager.py (反演迭代控制)         │
│  - mem_optimization.py (MEM优化)              │
│  - mem_monitoring.py (监控与日志)             │
└─────────────────────────────────────────────────┘
```

---

## 模块详解

### 1. 用户接口层

#### `pyzeetom/tomography.py` (235 行)
主入口模块，提供两个核心API：

```python
def forward_tomography(
    param_file: str = 'input/params_tomog.txt',
    verbose: int = 1,
    output_dir: str = './output'
) -> List[ForwardModelResult]
```
执行正演频谱合成，返回每相位的正演结果。

```python
def inversion_tomography(
    param_file: str = 'input/params_tomog.txt',
    obs_file: str = None,
    verbose: int = 1,
    output_dir: str = './output'
) -> InversionResult
```
执行MEM反演工作流，返回重建的磁场分布。

**关键点**：
- 薄包装设计，委托给工作流引擎
- 统一的参数处理和错误控制
- 自动路径推导

---

### 2. 工作流执行引擎

#### `core/tomography_forward.py` (246 行)
正演工作流主引擎。

**核心函数**：
```python
def run_forward_synthesis(
    config: ForwardModelConfig,
    verbose: bool = False
) -> List[ForwardModelResult]
```

**工作流步骤**：
1. 验证配置完整性
2. 创建磁盘积分器（VelspaceDiskIntegrator）
3. 对每相位执行频谱合成
4. 收集并返回结果

**关键操作**：
- 相位迭代
- 速度空间积分
- Stokes谱合成
- 结果聚合

---

#### `core/tomography_inversion.py` (1026 行)
MEM反演工作流主引擎。

**核心函数**：
```python
def run_mem_inversion(
    config: InversionConfig,
    verbose: bool = False
) -> InversionResult
```

**工作流步骤**：
1. 初始化反演迭代管理器
2. 对每个相位和每个像素执行MEM优化
3. 监控收敛性
4. 中间结果保存
5. 返回最终反演结果

**关键组件**：
- 参数编码/解码（磁场参数打包）
- MEM优化器适配层
- 迭代收敛控制
- 结果聚合

---

### 3. 配置与结果容器

#### `core/tomography_config.py` (621 行)
统一的配置对象。

**核心类**：

```python
@dataclass
class ForwardModelConfig:
    """正演配置容器"""
    par: Any                    # 参数对象
    obsSet: List[Any]          # 观测数据集
    lineData: BasicLineData     # 谱线参数
    geom: SimpleDiskGeometry    # 盘几何
    line_model: Any            # 谱线模型
    velEq: float = 100.0       # 赤道速度 (km/s)
    pOmega: float = 0.0        # 差速转动指数
    radius: float = 1.0        # 中心半径
    # ... 更多参数
    
    def validate(self) -> bool
    def create_summary(self) -> str
    @classmethod
    def from_par(cls, par, obsSet, lineData, **kwargs)
```

```python
@dataclass
class InversionConfig:
    """反演配置容器"""
    forward_config: ForwardModelConfig
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    entropy_regularization: float = 0.1
    # ... 更多参数
    
    def validate(self) -> bool
```

**优势**：
- 类型安全与IDE自动补全
- 内置验证逻辑
- 清晰的参数文档
- 便利的序列化/反序列化

---

#### `core/tomography_result.py` (16 KB)
统一的结果容器。

```python
@dataclass
class ForwardModelResult:
    """正演结果"""
    phase_index: int
    stokes_i: np.ndarray
    stokes_v: np.ndarray
    stokes_q: np.ndarray
    stokes_u: np.ndarray
    wavelength: np.ndarray
    model_parameters: dict
    
    def create_summary(self) -> str
    def save_to_file(self, filename: str)
```

```python
@dataclass
class InversionResult:
    """反演结果"""
    forward_results: List[ForwardModelResult]
    B_los: np.ndarray
    B_perp: np.ndarray
    chi: np.ndarray
    final_entropy: float
    convergence_flag: bool
    
    def create_summary(self) -> str
    def save_to_file(self, output_dir: str)
```

---

### 4. 物理计算核心

#### `core/velspace_DiskIntegrator.py` (702 行)
速度空间磁盘积分器 - 核心物理模块。

**核心类**：
```python
class VelspaceDiskIntegrator:
    """盘模型速度空间积分器
    
    功能：
    - 盘网格的速度映射
    - 局部Stokes谱生成
    - 速度空间积分合成
    - 多相位处理
    """
    
    def __init__(self, grid, line_model, verbose=False)
    
    def compute_spectrum_single_phase(
        self, 
        phase: float,
        Blos: np.ndarray,
        Bperp: np.ndarray,
        chi: np.ndarray
    ) -> dict
        # 返回: {'I': I_spec, 'V': V_spec, 'Q': Q_spec, 'U': U_spec}
    
    def derivative_Blos(self, ...) -> np.ndarray
        # Stokes参数对Blos的导数
    
    def derivative_Bperp_chi(self, ...) -> Tuple[np.ndarray, np.ndarray]
        # Stokes参数对Bperp和chi的导数
```

**关键算法**：
1. **盘速度场建模**：
   - 外侧：幂律Ω(r) = Ω₀(r/r₀)^p
   - 内侧：自适应减速序列

2. **局部谱线计算**：
   - 使用注入的谱线模型（line_model）
   - 支持I/V/Q/U多通道

3. **速度空间积分**：
   - 盘网格像素求和
   - 卷积平滑（FWHM调宽）
   - 速度到观测频率映射

4. **导数计算**：
   - 自动计算Stokes参数对磁场参数的导数
   - 支持反演优化

---

#### `core/local_linemodel_basic.py` (230 行)
谱线模型 - 弱场近似。

**核心类**：
```python
class LineData:
    """谱线参数容器（从文件读取）"""
    wl0: float      # 谱线中心波长
    sigWl: float    # 谱线宽度
    g: float        # Landé g因子
    
    def __init__(self, filename: str)
```

```python
class GaussianZeemanWeakLineModel(BaseLineModel):
    """弱场近似 + 高斯线型
    
    记: d = (λ - λ0)/σ, G = exp(-d²)
    
    输出:
      I = 1 + amp × G
      V = Cg × Blos × (amp × G × d / σ)
      Q = -C2 × Bperp² × (amp × (G/σ²) × (1 - 2d²)) × cos(2χ)
      U = -C2 × Bperp² × (amp × (G/σ²) × (1 - 2d²)) × sin(2χ)
    
    参数:
      - amp: 线项振幅（正=发射，负=吸收）
      - Blos: 视向磁场 (km/s)
      - Bperp: 垂直磁场
      - chi: 磁场方位角（弧度）
    """
    
    def compute_local_profile(
        self,
        wl_grid: np.ndarray,
        amp: np.ndarray,
        Blos: np.ndarray = None,
        **kwargs
    ) -> dict
```

**扩展点**：继承`BaseLineModel`可实现自定义谱线模型。

---

#### `core/mem_tomography.py` (554 行)
MEM反演适配层。

**核心类**：
```python
class MEMTomographyAdapter:
    """通用MEM算法与项目特定参数化的适配层
    
    功能：
    - Stokes I, Q, U, V 谱线的拟合
    - 磁场参数 (Blos, Bperp, chi) 的熵定义
    - 数据打包/解包
    - 响应矩阵构建
    """
    
    def __init__(self, config, grid, line_model, obs_data)
    
    def compute_synthetic(self, B_los, B_perp, chi) -> SyntheticSpectrum
        # 合成Stokes谱
    
    def compute_gradients(self, B_los, B_perp, chi) -> dict
        # 计算残差梯度（用于MEM优化）
    
    def pack_parameters(self, B_los, B_perp, chi) -> np.ndarray
        # 打包为优化向量
    
    def unpack_parameters(self, x: np.ndarray) -> Tuple[np.ndarray, ...]
        # 解包回物理参数
```

**集成点**：与`mem_generic.py`的通用MEM优化器集成。

---

### 5. 基础工具库

#### `core/grid_tom.py` (358 行)
网格生成与管理。

**核心类**：
```python
class diskGrid:
    """等Δr分层盘网格（每环宽度一致）
    
    属性（一维数组存储）：
    - r: 圆柱半径
    - phi: 方位角
    - dr: 径向像素宽度
    - dphi: 角向像素宽度
    - area: 像素面积
    - ring_id: 环编号
    - phi_id: 角向编号
    """
    
    def __init__(
        self,
        nr: int = 60,
        r_in: float = 0.0,
        r_out: float = 5.0,
        target_pixels_per_ring: Optional[Union[int, List]] = None
    )
    
    @property
    def numPoints(self) -> int
        # 总像素数
    
    def get_ring(self, ring_idx: int) -> dict
        # 获取指定环的所有像素
```

**设计亮点**：
- 一维数组存储，支持向量化运算
- 灵活的每环像素数控制
- 自动计算等面积或等数量配置

---

#### `core/disk_geometry.py` (7.8 KB)
盘几何与动力学参数容器。

```python
class SimpleDiskGeometry:
    """盘几何与动力学参数
    
    包含：
    - diskGrid 实例
    - 动力学参数 (velEq, pOmega, r0)
    - 物理参数 (inclination, posang等)
    """
```

---

#### `core/SpecIO.py` (728 行)
光谱数据读写（支持多格式）。

**核心函数**：
```python
def obsProfSetInRange(
    fnames: List[str],
    vel_start: float,
    vel_end: float,
    vel_rs: float,
    file_type: str = 'auto',
    pol_channels: Optional[Dict] = None
) -> List[ObservationProfile]
    # 读取观测数据集
```

```python
def write_model_spectrum(
    filename: str,
    wavelength: np.ndarray,
    spec_i: np.ndarray,
    spec_v: np.ndarray = None,
    spec_q: np.ndarray = None,
    spec_u: np.ndarray = None,
    file_type_hint: str = 'spec_i'
)
    # 写出模型谱
```

**支持格式**：
- `lsd_i`: LSD intensity-only format
- `lsd_pol`: LSD full polarimetry (I,V,Q,U,σ)
- `spec_i`: Simple spectrum (λ, I)
- `spec_pol`: Full polarimetry spectrum

---

#### `core/mainFuncs.py` (37 KB)
参数解析与兼容性层。

```python
def readParamsTomog(filename: str) -> ParamObject
    """读取参数文件（向后兼容旧版格式）
    
    返回包含所有配置参数的对象
    """

def parseParamLine(s: str) -> Tuple[str, str]
    """解析参数行"""

# ... 其他参数处理函数
```

---

#### `core/mem_generic.py` (17 KB)
通用最大熵方法（MEM）算法。

**核心类**：
```python
class MEMOptimizer:
    """通用MEM优化算法
    
    支持：
    - 最大化熵约束下的极大似然法
    - 自适应收敛控制
    - Lagrange乘子管理
    """
    
    def iterate(
        self,
        model_fn,
        residual_fn,
        data,
        x0: np.ndarray,
        lambda_coeff: float = 1.0
    ) -> Tuple[np.ndarray, float, dict]
```

**设计原则**：
- 完全项目无关的通用实现
- 通过回调函数接受项目特定的物理模型
- 易于与其他项目集成

---

#### `core/iteration_manager.py` (13 KB)
反演迭代控制与管理。

```python
class IterationManager:
    """管理MEM迭代过程
    
    功能：
    - 迭代计数与收敛判定
    - 中间结果保存
    - 参数收敛曲线跟踪
    - 自适应步长控制
    """
    
    def update(self, residual: float, entropy: float, params: np.ndarray)
    def should_continue(self) -> bool
    def get_summary(self) -> str
```

---

#### `core/mem_optimization.py` (19 KB)
MEM优化加速与缓存。

**核心类**：
```python
class ResponseMatrixCache:
    """响应矩阵缓存，避免重复计算"""
    
class DataPipeline:
    """数据流管理，优化内存使用"""
```

**Week 2优化**：
- 缓存响应矩阵（避免重复计算）
- 流式处理观测数据
- 自动内存管理

---

#### `core/mem_monitoring.py` (12 KB)
反演监控与日志。

```python
class MEMMonitor:
    """监控MEM反演过程
    
    记录：
    - 每次迭代的残差、熵、磁场
    - 收敛历史
    - 性能指标
    """
```

---

## 数据流与工作流

### 正演工作流 (Forward Synthesis)

```
输入文件
├── params_tomog.txt (参数)
├── lines.txt (谱线参数)
└── inSpec/*.lsd (观测数据)
       │
       ▼
[pyzeetom/tomography.py::forward_tomography]
       │
       ├─ mainFuncs.readParamsTomog(params_tomog.txt)
       │  └─ ParamObject {velEq, pOmega, radius, ...}
       │
       ├─ SpecIO.obsProfSetInRange(inSpec)
       │  └─ [ObservationProfile, ...]
       │
       ├─ LineData(lines.txt)
       │  └─ LineData {wl0, sigWl, g}
       │
       ▼
[ForwardModelConfig]
       │
       ├─ SimpleDiskGeometry (盘几何)
       │  └─ diskGrid + 动力学参数
       │
       ├─ GaussianZeemanWeakLineModel (谱线模型)
       │
       ├─ validate() (配置验证)
       │
       ▼
[tomography_forward.run_forward_synthesis]
       │
       ├─ FOR each phase in [phase_0, phase_1, ...]
       │  ├─ VelspaceDiskIntegrator.compute_spectrum_single_phase
       │  │  ├─ 为每个网格像素计算速度和磁场投影
       │  │  ├─ 调用 line_model.compute_local_profile
       │  │  │  └─ 返回 {I, V, Q, U} (Nλ,)
       │  │  ├─ 速度空间积分合成
       │  │  └─ 返回合成谱 {I, V, Q, U}
       │  │
       │  └─ ForwardModelResult
       │     └─ {phase_index, stokes_i/v/q/u, wavelength, ...}
       │
       ▼
输出文件
├── output/model_phase_0.lsd
├── output/model_phase_1.lsd
└── output/outFitSummary.txt
```

### 反演工作流 (MEM Inversion)

```
正演结果 (ForwardModelResult)
       │
       ├─ Stokes谱 {I, V, Q, U}
       ├─ 观测数据 {Iobs, Vobs, Qobs, Uobs}
       ├─ 初始磁场猜测 {Blos_0, Bperp_0, chi_0}
       │
       ▼
[InversionConfig]
       │
       ├─ forward_config (ForwardModelConfig)
       ├─ max_iterations, convergence_threshold
       ├─ entropy_regularization
       │
       ▼
[tomography_inversion.run_mem_inversion]
       │
       ├─ IterationManager (迭代控制)
       │
       ├─ FOR iteration = 0, 1, 2, ...
       │  ├─ FOR each pixel in diskGrid
       │  │  ├─ MEMTomographyAdapter.compute_synthetic
       │  │  │  ├─ 调用 VelspaceDiskIntegrator.compute_spectrum
       │  │  │  └─ 返回合成谱
       │  │  │
       │  │  ├─ MEMOptimizer.iterate (单步MEM优化)
       │  │  │  ├─ 计算残差: χ² = Σ((S_syn - S_obs)²/σ²)
       │  │  │  ├─ 计算熵: H = -Σ p_i log(p_i)
       │  │  │  ├─ 最大化: Q = H - λ·χ²
       │  │  │  ├─ 更新参数: Blos, Bperp, chi
       │  │  │  └─ 返回 x_new, χ²_new, ...
       │  │  │
       │  │  └─ 更新磁场 {Blos, Bperp, chi}
       │  │
       │  ├─ IterationManager.update (收敛判定)
       │  │  ├─ 检查 |Δχ²| < threshold
       │  │  ├─ 检查最大迭代次数
       │  │  └─ 保存中间结果
       │  │
       │  └─ 中间结果保存 (可选)
       │
       ▼
[InversionResult]
       │
       ├─ forward_results (正演结果)
       ├─ B_los (最终视向磁场)
       ├─ B_perp (最终垂直磁场)
       ├─ chi (最终磁场方位角)
       ├─ final_entropy (最终熵)
       ├─ convergence_flag (是否收敛)
       │
       ▼
输出文件
├── output/mem_inversion_result.npz
├── output/inversion_summary.txt
└── output/inversion_intermediate_*.npz
```

---

## 物理模型

### 1. 盘速度场

**外侧** (r ≥ r₀)：幂律自转
$$\Omega(r) = \Omega_0 \left(\frac{r}{r_0}\right)^p$$

**内侧** (r < r₀)：自适应减速序列
- 用余弦或其他光滑函数实现过渡
- 避免物理奇点

**线速度**：
$$v_\phi(r) = r \cdot \Omega(r)$$

### 2. 谱线模型（弱场近似）

设高斯谱线中心为λ₀，宽度为σ，无量纲偏差：
$$d = \frac{\lambda - \lambda_0}{\sigma}$$

则高斯基：
$$G(d) = \exp(-d^2)$$

#### 强度 (Stokes I)
$$I(\lambda) = I_c + a \cdot G(d)$$
其中 $I_c = 1$（连续谱）， $a$ 为振幅（正=发射，负=吸收）

#### 圆偏振 (Stokes V)
$$V(\lambda) = C_g \cdot B_\text{los} \cdot a \cdot G(d) \cdot \frac{d}{\sigma}$$
其中 $C_g$ 为Zeeman系数， $B_\text{los}$ 为视向磁场

#### 线性偏振 (Stokes Q, U)
$$Q(\lambda) = -C_2 \cdot B_\perp^2 \cdot a \cdot \frac{G(d)}{\sigma^2} \cdot (1 - 2d^2) \cdot \cos(2\chi)$$
$$U(\lambda) = -C_2 \cdot B_\perp^2 \cdot a \cdot \frac{G(d)}{\sigma^2} \cdot (1 - 2d^2) \cdot \sin(2\chi)$$

其中：
- $B_\perp$ 为垂直平面磁场强度
- $\chi$ 为磁场方位角（弧度）
- $C_2$ 为二阶Zeeman系数

### 3. 速度到观测频率映射

多普勒偏移：
$$v = c \frac{\lambda - \lambda_\text{ref}}{\lambda_\text{ref}}$$

或频率空间：
$$\nu = \nu_0 \left(1 - \frac{v}{c}\right)$$

### 4. 盘积分与求和

每像素对观测谱的贡献：
$$S_\text{obs}(\lambda) = \sum_i w_i \cdot S_\text{local}(i, \lambda)$$

其中 $w_i$ 为像素权重（面积/可见性因子等）。

---

## 扩展与集成

### 自定义谱线模型

1. 继承 `BaseLineModel`
2. 实现 `compute_local_profile()` 方法
3. 返回 `{'I': ..., 'V': ..., 'Q': ..., 'U': ...}`

```python
class MyCustomLineModel(BaseLineModel):
    def compute_local_profile(self, wl_grid, amp, **kwargs):
        # 自定义计算逻辑
        return {'I': I, 'V': V, 'Q': Q, 'U': U}
```

### 新观测格式支持

1. 在 `SpecIO.py` 中添加解析函数
2. 返回 `ObservationProfile` 对象
3. 集成到 `obsProfSetInRange()` 中

### 反演方法扩展

1. 在 `core/` 下创建新模块（如 `tomography_mcmc.py`）
2. 实现类似 `run_mem_inversion()` 的接口
3. 使用现有的配置与结果容器
4. 在主入口 (`pyzeetom/tomography.py`) 中暴露新接口

---

## 典型开发流程

### 步骤1：问题诊断
- 使用 `tomography_config.validate()` 检查参数
- 查看 `tomography_result.create_summary()` 理解输出

### 步骤2：正演验证
```python
from pyzeetom import tomography
results = tomography.forward_tomography('input/params.txt', verbose=2)
```

### 步骤3：反演调试
```python
results = tomography.inversion_tomography('input/params.txt', verbose=2)
```

### 步骤4：模型扩展
- 修改 `disk_geometry.py` 添加新的几何模型
- 继承 `BaseLineModel` 实现自定义谱线
- 在 `tomography_config.py` 中配置参数

### 步骤5：性能优化
- 使用 `mem_optimization.py` 的缓存和流管理
- 利用 `mem_monitoring.py` 跟踪性能指标
- 根据 `iteration_manager.py` 调整收敛参数

---

## 文件大小与复杂度概览

| 模块 | 大小 | 主要责任 |
|-----|------|--------|
| mainFuncs.py | 37 KB | 参数解析、兼容性 |
| velspace_DiskIntegrator.py | 27 KB | 核心物理积分 |
| SpecIO.py | 27 KB | 光谱IO |
| tomography_inversion.py | 34 KB | MEM反演流程 |
| tomography_config.py | 21 KB | 配置容器 |
| mem_optimization.py | 19 KB | MEM优化加速 |
| mem_tomography.py | 19 KB | MEM适配层 |
| mem_generic.py | 17 KB | 通用MEM算法 |
| tomography_result.py | 16 KB | 结果容器 |
| grid_tom.py | 14 KB | 网格生成 |
| iteration_manager.py | 13 KB | 迭代控制 |
| mem_monitoring.py | 12 KB | 监控与日志 |
| local_linemodel_basic.py | 8 KB | 谱线模型 |
| tomography_forward.py | 7.1 KB | 正演流程 |
| disk_geometry.py | 7.8 KB | 盘几何 |

**总计**：约 327 KB 的核心代码库。

---

## 参考文献与设计原则

### 核心算法
- Skilling & Bryan (1984): Maximum Entropy Image Reconstruction
- Hobson & Lasenby (1998): Magnetic field inversion using entropy methods

### 设计模式
- **分层架构**：UI → Config → Workflow → Physics → Tools
- **配置对象**：类型安全的参数封装
- **结果容器**：统一的输出结构
- **适配器模式**：通用算法与项目特定物理的解耦
- **回调设计**：MEM优化器与项目无关

### 扩展性原则
- 新谱线模型：继承 `BaseLineModel`
- 新几何模型：修改 `disk_geometry.py`
- 新观测格式：扩展 `SpecIO.py`
- 新反演方法：创建新工作流模块

---

**文档维护**: 每次重大重构后更新  
**最后修改**: 2025-11-15  
**贡献者**: pyZeeTom development team
