
# pyZeeTom Copilot 快速指南与项目架构

**最后更新**: 2025-11-15  
**版本**: Phase 2.5.4.1（重构完成）

## 快速导航

- 📐 **完整架构文档**: 见 `docs/ARCHITECTURE.md` 
- 🎯 **快速开始**: [快速开始](#快速开始)
- 🔧 **核心模块**: [核心架构](#核心架构)
- 📊 **数据流**: [数据流与工作流](#数据流与工作流)
- 🧪 **开发指南**: [开发与风格约定](#开发与风格约定)

---

## 项目概述

**pyZeeTom** 是一个用于反演和正演4个Stokes量（I, Q, U, V）偏振光谱的tomography工具。

### 物理场景
- **中心天体+星周物质**：存在一个中心天体，周围有星周物质（尘埃团、盘、行星等）以刚体或差速方式环绕运动
- **相位观测**：观测者与中心天体处于同一惯性系，仅通过天体自转带来的不同"phase"观测不同视角  
- **多通道观测**：每一观测相位可获得Stokes I及VQU分量的偏振光谱
- **工作模式**：正演模型 + MEM反演方法

---

## 快速开始

### 正演合成
```python
from pyzeetom import tomography
results = tomography.forward_tomography('input/params_tomog.txt', verbose=1)
# 返回 List[ForwardModelResult]，每个元素对应一个观测相位
```

### MEM反演
```python
result = tomography.inversion_tomography('input/params_tomog.txt', verbose=1)
# 返回 InversionResult，包含重建的磁场分布 (B_los, B_perp, chi)
```

---

## 核心架构

### 分层设计

```
┌─ pyzeetom/tomography.py ─────────────────┐  用户接口层
│  forward_tomography() / inversion_tomography()
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐  工作流执行层
│  tomography_forward.py                   │  
│  tomography_inversion.py                 │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐  配置与结果层
│  tomography_config.py (Config objects)   │
│  tomography_result.py (Result objects)   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐  物理计算层
│  velspace_DiskIntegrator.py (核心积分)   │
│  local_linemodel_basic.py (谱线模型)    │
│  mem_tomography.py (MEM适配)            │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐  基础工具层
│  grid_tom.py (网格)                      │
│  disk_geometry.py (盘几何)               │
│  SpecIO.py (光谱IO)                      │
│  mainFuncs.py (参数解析)                │
│  mem_generic.py (MEM算法)               │
│  iteration_manager.py (迭代控制)        │
│  mem_optimization.py (缓存加速)         │
│  mem_monitoring.py (监控日志)           │
└─────────────────────────────────────────┘
```

### 1. core/ 物理与数值核心

| 文件 | 大小 | 功能 |
|------|------|------|
| **velspace_DiskIntegrator.py** | 27 KB | 速度空间积分、盘模型、Stokes谱合成 |
| **tomography_inversion.py** | 34 KB | MEM反演工作流执行引擎 |
| **tomography_config.py** | 21 KB | 正演/反演配置容器（dataclass） |
| **SpecIO.py** | 27 KB | 光谱数据读写（多格式支持） |
| **mainFuncs.py** | 37 KB | 参数解析、向后兼容 |
| **mem_tomography.py** | 19 KB | MEM反演适配层（项目特定参数化） |
| **mem_optimization.py** | 19 KB | MEM优化加速、缓存、数据流管理 |
| **mem_generic.py** | 17 KB | 通用MEM算法（项目无关） |
| **tomography_result.py** | 16 KB | 正演/反演结果容器 |
| **grid_tom.py** | 14 KB | 环状盘网格生成（等Δr分层） |
| **iteration_manager.py** | 13 KB | MEM迭代控制、收敛判定、中间保存 |
| **mem_monitoring.py** | 12 KB | 反演监控、性能指标、日志 |
| **local_linemodel_basic.py** | 8 KB | 弱场高斯Zeeman谱线模型 |
| **tomography_forward.py** | 7.1 KB | 正演工作流执行 |
| **disk_geometry.py** | 7.8 KB | 盘几何与动力学参数 |

### 2. pyzeetom/ 主入口与流程调度

| 文件 | 功能 |
|------|------|
| **tomography.py** | 主入口，提供 `forward_tomography()` 和 `inversion_tomography()` API |
| **__init__.py** | 包初始化 |

---

## 数据流与工作流

### 正演工作流 (Forward Synthesis)

```
输入数据
├── params_tomog.txt (主控参数)
├── lines.txt (谱线参数: wl0, sigWl, g)
└── inSpec/*.lsd (观测数据)
       │
       ▼
readParamsTomog() / SpecIO.obsProfSetInRange() / LineData()
       │
       ├─ ParamObject (动力学参数、格式等)
       ├─ [ObservationProfile] (观测谱集合)
       └─ LineData (谱线参数)
       │
       ▼
ForwardModelConfig (配置容器)
       │
       ├─ SimpleDiskGeometry (盘网格 + 动力学)
       ├─ GaussianZeemanWeakLineModel (谱线模型)
       └─ validate()
       │
       ▼
run_forward_synthesis() [tomography_forward.py]
       │
       ├─ FOR each phase:
       │  ├─ VelspaceDiskIntegrator.compute_spectrum_single_phase()
       │  │  ├─ 计算每像素速度和磁场投影
       │  │  ├─ 调用 line_model.compute_local_profile()
       │  │  │  └─ 返回 {I, V, Q, U}
       │  │  └─ 速度空间积分合成
       │  │
       │  └─ ForwardModelResult(相位结果)
       │
       ▼
输出文件
├── output/model_phase_0.lsd
├── output/model_phase_1.lsd
└── output/outFitSummary.txt
```

### 反演工作流 (MEM Inversion)

```
正演结果 + 观测数据
       │
       ├─ 合成Stokes谱 {I, V, Q, U}
       ├─ 观测Stokes谱 {Iobs, Vobs, Qobs, Uobs}
       └─ 初始磁场猜测 {Blos_0, Bperp_0, chi_0}
       │
       ▼
InversionConfig (配置容器)
       │
       ├─ forward_config
       ├─ max_iterations, convergence_threshold
       └─ entropy_regularization
       │
       ▼
run_mem_inversion() [tomography_inversion.py]
       │
       ├─ IterationManager (迭代控制)
       │
       ├─ FOR iteration:
       │  ├─ FOR each pixel:
       │  │  ├─ MEMTomographyAdapter.compute_synthetic()
       │  │  ├─ MEMOptimizer.iterate()
       │  │  │  ├─ 计算 χ² = Σ((S_syn - S_obs)²/σ²)
       │  │  │  ├─ 最大化 Q = H - λ·χ²
       │  │  │  └─ 更新 (Blos, Bperp, chi)
       │  │  │
       │  │  └─ 更新磁场
       │  │
       │  ├─ 收敛判定
       │  └─ 中间保存 (可选)
       │
       ▼
InversionResult
       │
       ├─ B_los (最终视向磁场)
       ├─ B_perp (最终垂直磁场)
       ├─ chi (最终磁场方位角)
       ├─ final_entropy
       └─ convergence_flag
       │
       ▼
输出文件
├── output/mem_inversion_result.npz
├── output/inversion_summary.txt
└── output/inversion_intermediate_*.npz
```

---

## 物理模型

### 盘速度场

**外侧** (r ≥ r₀): 幂律自转
$$\Omega(r) = \Omega_0 \left(\frac{r}{r_0}\right)^p, \quad v_\phi = r \cdot \Omega(r)$$

**内侧** (r < r₀): 自适应减速序列（光滑过渡）

### 谱线模型（弱场近似）

设无量纲偏差 $d = (\lambda - \lambda_0) / \sigma$，高斯基 $G(d) = \exp(-d^2)$

#### Stokes I（强度）
$$I(\lambda) = 1 + a \cdot G(d)$$

#### Stokes V（圆偏振）
$$V(\lambda) = C_g \cdot B_{\text{los}} \cdot a \cdot G(d) \cdot \frac{d}{\sigma}$$

#### Stokes Q, U（线性偏振）
$$Q(\lambda) = -C_2 \cdot B_\perp^2 \cdot a \cdot \frac{G(d)}{\sigma^2} \cdot (1-2d^2) \cdot \cos(2\chi)$$
$$U(\lambda) = -C_2 \cdot B_\perp^2 \cdot a \cdot \frac{G(d)}{\sigma^2} \cdot (1-2d^2) \cdot \sin(2\chi)$$

其中：
- $a$ 为振幅（正=发射，负=吸收）
- $B_{\text{los}}$ 为视向磁场
- $B_\perp, \chi$ 为垂直磁场与方位角

---

## 开发与风格约定

### 命名与单位约定
- 所有像素属性（r, phi, Blos等）都是一维数组，与像素数一致
- 速度单位：km/s（主要）
- 磁场：Gauss
- 方位角：弧度

### 数组形状约定
- 网格像素：(Npix,)
- 波长/频率：(Nlambda,)
- Stokes谱：(Nlambda,) 或 (Nlambda, Nphase)
- 磁场参数求导：(Nlambda, Npix)

### 配置对象设计
```python
# 使用 dataclass 而非字典
@dataclass
class ForwardModelConfig:
    par: Any
    obsSet: List[Any]
    lineData: BasicLineData
    # ... 参数与类型注解
    
    def validate(self) -> bool:
        # 验证参数一致性
        pass
```

### 光谱输出一致性

使用 `SpecIO.write_model_spectrum()` 时须明确指定输出格式：
```python
SpecIO.write_model_spectrum(
    filename='output/model.lsd',
    wavelength=wl,
    spec_i=I_spec,
    spec_v=V_spec,
    file_type_hint='lsd_pol'  # 明确指定格式
)
```

支持格式：
- `lsd_i`: LSD 仅强度（3列）
- `lsd_pol`: LSD 完全偏振（I,V,Q,U,σ）
- `spec_i`: 简单谱（λ, I）
- `spec_pol`: 谱+偏振（Wav, Int, Pol, σ）

### 主入口约定
- 用户入口: `pyzeetom/tomography.py`
- 运行前确保 `PYTHONPATH` 包含项目根目录

---

## 典型扩展点

### 自定义谱线模型
继承 `BaseLineModel` 并实现 `compute_local_profile()`:
```python
from core.local_linemodel_basic import BaseLineModel

class MyLineModel(BaseLineModel):
    def compute_local_profile(self, wl_grid, amp, Blos=None, **kwargs):
        # 自定义计算逻辑
        return {'I': I, 'V': V, 'Q': Q, 'U': U}

# 在配置中使用
config.line_model = MyLineModel()
```

### 新观测格式支持
在 `SpecIO.py` 中扩展：
```python
def load_custom_format(filename):
    # 解析自定义格式
    return ObservationProfile(...)

# 集成到 obsProfSetInRange()
```

### 新反演方法
创建新工作流模块（如 `tomography_mcmc.py`）:
```python
def run_mcmc_inversion(config: InversionConfig) -> InversionResult:
    # 使用现有的 ForwardModelConfig / InversionResult 容器
    pass

# 在主入口暴露接口
```

---

## 核心文件速查

| 需求 | 文件 | 关键函数/类 |
|------|------|----------|
| 正演合成 | tomography_forward.py | `run_forward_synthesis()` |
| MEM反演 | tomography_inversion.py | `run_mem_inversion()` |
| 参数解析 | mainFuncs.py | `readParamsTomog()` |
| 光谱IO | SpecIO.py | `obsProfSetInRange()`, `write_model_spectrum()` |
| 网格生成 | grid_tom.py | `diskGrid` |
| 速度积分 | velspace_DiskIntegrator.py | `VelspaceDiskIntegrator` |
| 谱线模型 | local_linemodel_basic.py | `GaussianZeemanWeakLineModel` |
| MEM算法 | mem_generic.py | `MEMOptimizer` |
| 迭代控制 | iteration_manager.py | `IterationManager` |

---

## 注意事项

⚠️ **常见错误**
- ❌ 磁场数组长度与像素数不一致 → ValueError
- ❌ 速度单位混淆（km/s vs m/s）
- ❌ 谱线参数文件格式不规范 → 解析失败
- ❌ 观测数据格式指定错误 → 数据读取失败

✅ **最佳实践**
- 总是使用 `config.validate()` 检查参数
- 使用 `result.create_summary()` 理解输出
- 使用 `verbose=2` 进行调试
- 保存中间结果便于问题追踪

---

## 完整文档

更多细节请参考 **`docs/ARCHITECTURE.md`**，包括：
- 详细的物理模型推导
- 数据流图表
- 模块间接口说明
- 参考文献与设计原则
- 性能优化指南
