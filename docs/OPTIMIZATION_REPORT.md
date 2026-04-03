# pyZeeTom 架构优化报告

**版本**：v0.3.2  
**报告日期**：2026-04-03  
**分析范围**：`core/`、`pyzeetom/`、`utils/`、`tests/`、`docs/`、`examples/`

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [现状评估](#2-现状评估)
3. [优化建议](#3-优化建议)
   - 3.1 [拆分超大模块](#31-拆分超大模块)
   - 3.2 [提升测试覆盖率](#32-提升测试覆盖率)
   - 3.3 [规范化错误处理](#33-规范化错误处理)
   - 3.4 [统一配置管理](#34-统一配置管理)
   - 3.5 [性能优化](#35-性能优化)
   - 3.6 [包结构与公共 API 改善](#36-包结构与公共-api-改善)
   - 3.7 [SpecIO 模块重构](#37-specio-模块重构)
   - 3.8 [整合 `tbd/` 实验代码](#38-整合-tbd-实验代码)
   - 3.9 [持续集成与代码质量门禁](#39-持续集成与代码质量门禁)
4. [优先级路线图](#4-优先级路线图)
5. [附录：代码指标一览](#5-附录代码指标一览)

---

## 1. 执行摘要

pyZeeTom 是一个专注于偏振光谱层析反演的科学计算软件包，其分层架构设计清晰、物理建模严谨、类型标注完整，已初步具备良好的工程基础。然而在向生产级科学软件演进的过程中，仍存在以下几类系统性问题：

| 问题类型 | 严重程度 | 主要表现 |
|----------|----------|----------|
| 代码规模失控 | ⚠️ 高 | 3 个文件超过 1000 行，职责边界模糊 |
| 测试覆盖极低 | ⚠️ 高 | 仅 1 个测试文件，物理核心完全未测试 |
| 错误处理不规范 | 🔶 中 | `except Exception: pass` 吞噬真实错误 |
| 配置入口分裂 | 🔶 中 | 文本格式与 JSON 格式并存，迁移不完整 |
| 性能潜力未释放 | 🔶 中 | 响应矩阵缓存存在但未默认启用 |
| API 封装不完整 | 🟡 低 | `main()` CLI 入口未实现，内部模块直接暴露 |

---

## 2. 现状评估

### 2.1 架构优势

- **良好的分层设计**：用户 API 层（`pyzeetom/`）→ 工作流层（`tomography_*.py`）→ 物理计算层（`disk_geometry_integrator.py`）→ 基础工具层（`grid_tom.py`、`SpecIO.py`）形成清晰的单向依赖。
- **广泛使用 `dataclass`**：`ForwardModelConfig`、`InversionConfig`、`PhysicalModel` 等均以 `@dataclass` 定义，类型安全且可序列化。
- **适配器模式合理运用**：`MEMTomographyAdapter` 将通用的 MEM 数学引擎（`mem_generic.py`）与项目特定的磁场物理量解耦，扩展性强。
- **类型标注覆盖率高**：经统计约 97% 的函数携带类型标注，有利于静态分析和 IDE 辅助。
- **文档质量优秀**：`ARCHITECTURE.md`、`README.md` 及各模块 docstring 内容翔实，包含物理公式推导。

### 2.2 主要问题

#### 文件规模与职责

```
core/tomography_inversion.py  1505 行  ← 参数编解码、响应矩阵、收敛检测混杂
core/mainFuncs.py             1286 行  ← JSON/文本解析、参数验证、兼容层全部堆叠
core/disk_geometry_integrator.py 1223 行  ← 几何计算与积分核心混合
```

#### 测试覆盖

- 全项目仅有 **`tests/test_config_json.py`**（375 行，34 个测试用例）。
- 物理核心（Stokes 积分、MEM 迭代、收敛判断、线模型）**完全没有单元测试**。
- 错误路径仅覆盖 1 处。

#### 其他

- `SpecIO.py` 内嵌于 `grid_tom.py` 文件末尾（非独立文件），同时承担解析、验证、I/O、格式检测多种职责。
- `tbd/local_linemodel_ME.py`（397 行 Unno-Rachkovsky 线模型）游离于主代码库之外，无法通过正常 import 使用。
- `pyzeetom/tomography.py` 中的 `main()` CLI 入口在 `pyproject.toml` 中已声明但尚未实现。

---

## 3. 优化建议

### 3.1 拆分超大模块

#### 3.1.1 `tomography_inversion.py`（1505 行）

当前该文件同时包含：参数打包/解包工具函数、响应矩阵数值微分计算、同步反演主循环、初始化逻辑。建议拆分为：

```
core/
├── tomography_inversion.py      ← 仅保留 run_mem_inversion() 等公共入口（~200 行）
├── inversion_parameter_codec.py ← _pack_parameters / _unpack_parameters（~120 行）
└── inversion_response_matrix.py ← _compute_response_matrix（~150 行）
```

**拆分原则**：每个文件只解决一个清晰的子问题；公共工具函数（如 `_pack_parameters`）与 `mem_tomography.py` 中 `MagneticFieldParams.to_vector()` 存在逻辑重复，拆分后可统一到 `inversion_parameter_codec.py`，消除重复代码。

#### 3.1.2 `mainFuncs.py`（1286 行）

`readParamsTomog` 类同时负责文本格式解析、JSON 格式解析、参数合法性验证、向后兼容别名维护：

```
core/
├── config_loader.py    ← 仅负责文件 I/O（文本/JSON 读写）
├── config_validator.py ← 参数合法性检查（数值范围、必填项）
└── mainFuncs.py        ← 保留 readParamsTomog 作为兼容门面，委托上面两个模块
```

**拆分后可同步实现参数校验的集中化**，避免默认值分散在构造函数、`from_json()`、`ForwardModelConfig` 三处。

#### 3.1.3 `disk_geometry_integrator.py`（1223 行）

建议将速度场计算（`disk_velocity_rigid_inner`）、几何投影（`SimpleDiskGeometry`）与频谱积分核心（`VelspaceDiskIntegrator`）分离：

```
core/
├── disk_geometry.py   ← SimpleDiskGeometry + 速度场工具函数
└── disk_integrator.py ← VelspaceDiskIntegrator（依赖 disk_geometry）
```

---

### 3.2 提升测试覆盖率

当前仅对 JSON 配置加载进行了系统测试，建议按以下优先级补充测试：

#### 优先级 P0（阻塞物理正确性）

```python
# tests/test_line_model.py
def test_stokes_v_antisymmetric():
    """Stokes V should be antisymmetric under B_los sign flip."""
    model = GaussianZeemanWeakLineModel(line_data)
    v_pos = model.compute_local_profile(wl, amp=1.0, Blos=+100.0)['V']
    v_neg = model.compute_local_profile(wl, amp=1.0, Blos=-100.0)['V']
    np.testing.assert_allclose(v_pos, -v_neg, rtol=1e-10)

def test_zero_field_stokes_qu_vanish():
    """Q and U must be zero when B_perp == 0."""
    result = model.compute_local_profile(wl, amp=1.0, Blos=0, Bperp=0, chi=0)
    np.testing.assert_allclose(result['Q'], 0, atol=1e-12)
    np.testing.assert_allclose(result['U'], 0, atol=1e-12)
```

```python
# tests/test_grid.py
def test_grid_pixel_count_matches_rings():
    """Total pixel count must equal sum over rings of their phi bins."""
    grid = diskGrid(nRings=5, Vmax=50.0, vsini=30.0, ...)
    assert grid.numPoints == sum(grid.nphis)
```

#### 优先级 P1（MEM 核心数学）

```python
# tests/test_mem_generic.py
def test_entropy_decreases_toward_default():
    """When Image == default, entropy gradient should be zero."""

def test_chi2_gradient_numerical_vs_analytical():
    """Analytical chi2 gradient must match finite-difference approximation."""
```

#### 优先级 P2（集成测试）

```python
# tests/test_forward_integration.py
def test_forward_roundtrip_single_phase():
    """Run full forward synthesis with synthetic config, verify spectra shapes."""
```

**目标**：将核心物理代码行覆盖率从当前的 ~5% 提升至 **≥60%**（可用 `pytest-cov` 追踪）。

建议在 `pyproject.toml` 中添加覆盖率配置：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=core --cov=pyzeetom --cov-report=term-missing"

[project.optional-dependencies]
dev = [
  "pytest>=7.0",
  "pytest-cov>=4.0",
  "matplotlib>=3.6",
]
```

---

### 3.3 规范化错误处理

#### 问题位置

`core/SpecIO.py`（嵌入于 `grid_tom.py`）中存在吞噬异常的模式：

```python
# 现状（行 84–88）：
for p in parts:
    try:
        float(p)
        nums += 1
    except Exception:   # ← 过于宽泛，掩盖真实错误
        pass
```

#### 建议改法

```python
# 改后：明确只捕获 ValueError
for p in parts:
    try:
        float(p)
        nums += 1
    except ValueError:
        pass
```

对于更复杂的格式检测逻辑，建议引入自定义异常层次：

```python
# core/exceptions.py（新建）
class PyZeeTomError(Exception):
    """Base exception for all pyZeeTom errors."""

class SpecParseError(PyZeeTomError):
    """Raised when spectral data cannot be parsed."""
    def __init__(self, filename: str, reason: str):
        super().__init__(f"Cannot parse '{filename}': {reason}")

class ConfigValidationError(PyZeeTomError):
    """Raised when configuration parameters are invalid."""
```

自定义异常层次的好处：用户可在自己的代码中精确捕获 `SpecParseError` 而不是裸 `Exception`，极大提升可调试性。

---

### 3.4 统一配置管理

#### 当前痛点

- 参数默认值分散在三处：`readParamsTomog.__init__`、`readParamsTomog.from_json`、`ForwardModelConfig`；
- JSON 路径与文本路径解析逻辑存在细微不一致（向后兼容补丁分布各处）；
- `velRs`、`lineParamFile` 等字段只在文本路径中设置，JSON 路径可能漏掉。

#### 建议方案

**集中默认值**：引入一个 `PARAM_DEFAULTS` 字典（或 `@dataclass` with `field(default=...)`），作为所有解析路径的单一来源：

```python
# core/config_defaults.py（新建）
PARAM_DEFAULTS = {
    "obsFileType": "auto",
    "lineParamFile": "input/lines.txt",
    "enable_stellar_occultation": 0,
    "fit_B_los": True,
    "fit_B_perp": True,
    "fit_chi": True,
    # ...
}
```

**废弃文本格式（长期）**：在下一个次要版本中，为文本格式解析路径添加 `DeprecationWarning`，并提供 `pyzeetom convert-config old.txt new.json` CLI 命令辅助迁移：

```python
# pyzeetom/tomography.py
def forward_tomography(param_file, ...):
    if not str(param_file).lower().endswith('.json'):
        import warnings
        warnings.warn(
            "Text parameter files are deprecated. "
            "Please migrate to JSON format using `pyzeetom convert-config`.",
            DeprecationWarning,
            stacklevel=2,
        )
```

---

### 3.5 性能优化

#### 3.5.1 默认启用响应矩阵缓存

`mem_optimization.py` 中已实现 `ResponseMatrixCache`（LRU 缓存），但在 `tomography_inversion.py` 中的使用是可选的。建议将其设为**默认启用**：

```python
# core/tomography_inversion.py - 在 _invert_simultaneous 中
from core.mem_optimization import ResponseMatrixCache

cache = ResponseMatrixCache(max_size=10)  # 默认启用

def _compute_response_matrix_cached(integrator, B_los, B_perp, chi, ...):
    key = cache.compute_key(B_los, B_perp, chi)
    if key in cache:
        return cache[key]
    result = _compute_response_matrix(integrator, B_los, B_perp, chi, ...)
    cache[key] = result
    return result
```

#### 3.5.2 向量化像素积分

`VelspaceDiskIntegrator` 中对每个圆环像素逐一调用 `line_model.compute_local_profile()` 形成内层循环。对于 `GaussianZeemanWeakLineModel`，该计算完全可以向量化（一次性对所有像素计算高斯指数）：

```python
# 现状（示意）：
for ipix in range(npix):
    profile = line_model.compute_local_profile(wl, amp[ipix], Blos[ipix], ...)
    stokes_i[:] += profile['I'] * area[ipix]

# 优化后：一次性计算所有像素
# d[ipix, iwl] = (wl[iwl] - wl_shifted[ipix]) / sigma
d = (wl_grid[np.newaxis, :] - wl_shifted[:, np.newaxis]) / sigma
G = np.exp(-d**2)                              # (npix, nlambda)
stokes_i = np.sum(amp[:, np.newaxis] * G * area[:, np.newaxis], axis=0)
```

向量化可将内层循环从 Python 级降至 NumPy C 级，**对典型网格（~500 像素 × ~200 波长点）可获得 10–50× 加速**。

#### 3.5.3 提前收敛退出

`mem_iteration_manager.py` 中的 `ConvergenceChecker` 支持停滞检测，但 `run_mem_inversion` 主循环中未始终利用该结果提前退出。建议确保每次迭代后检查 `ConvergenceChecker.is_stagnant()` 并在满足条件时立即跳出。

#### 3.5.4 数值微分并行化（长期）

响应矩阵计算（`_compute_response_matrix`）需对每个像素的每个磁场参数做有限差分，共约 `3 × Npix` 次正演调用。该步骤为高度并行的 embarrassingly parallel 问题，可使用 `concurrent.futures.ProcessPoolExecutor` 或 `joblib` 实现多核并行：

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
    futures = [pool.submit(_perturb_and_integrate, integrator, ipix, delta)
               for ipix in range(npix)]
    results = [f.result() for f in futures]
```

---

### 3.6 包结构与公共 API 改善

#### 3.6.1 实现 CLI `main()` 入口

`pyproject.toml` 已声明 `pyzeetom = "pyzeetom.tomography:main"`，但 `main()` 函数尚未实现。建议补充：

```python
# pyzeetom/tomography.py
def main():
    """Command-line interface entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="pyZeeTom — Polarization Tomography Tool"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pyzeetom forward config.json
    fwd = subparsers.add_parser("forward", help="Run forward synthesis")
    fwd.add_argument("config", help="Parameter file (.json or .txt)")
    fwd.add_argument("-v", "--verbose", type=int, default=1)
    fwd.add_argument("-o", "--output-dir", default="./output")

    # pyzeetom inversion config.json
    inv = subparsers.add_parser("inversion", help="Run MEM inversion")
    inv.add_argument("config", help="Parameter file (.json or .txt)")
    inv.add_argument("-v", "--verbose", type=int, default=1)
    inv.add_argument("-o", "--output-dir", default="./output")

    # pyzeetom convert-config old.txt new.json
    conv = subparsers.add_parser("convert-config", help="Convert text config to JSON")
    conv.add_argument("input", help="Input text parameter file")
    conv.add_argument("output", help="Output JSON file")

    args = parser.parse_args()
    if args.command == "forward":
        forward_tomography(args.config, verbose=args.verbose, output_dir=args.output_dir)
    elif args.command == "inversion":
        inversion_tomography(args.config, verbose=args.verbose, output_dir=args.output_dir)
    elif args.command == "convert-config":
        _convert_config(args.input, args.output)
```

#### 3.6.2 收紧 `__init__.py` 公共 API

当前 `pyzeetom/__init__.py` 几乎为空。建议显式声明公共 API：

```python
# pyzeetom/__init__.py
from pyzeetom.tomography import forward_tomography, inversion_tomography

__version__ = "0.3.2"
__all__ = ["forward_tomography", "inversion_tomography"]
```

这样用户可以直接 `from pyzeetom import forward_tomography`，而不必了解内部模块结构。

#### 3.6.3 统一版本号管理

当前 `pyproject.toml` 中版本号为 `0.1.0`，而自定义指令文档显示版本为 `v0.3.2`，两者不一致。建议：

```toml
# pyproject.toml
[project]
version = "0.3.2"
```

并引入 `pyzeetom/__version__.py` 作为单一来源，供 `__init__.py` 和 `pyproject.toml` 共同引用（或使用 `setuptools-scm` 从 git tag 自动生成版本）。

---

### 3.7 SpecIO 模块重构

当前 `SpecIO.py` 的代码被**嵌入在 `core/grid_tom.py` 文件末尾**，且承担多种职责。建议以下重构：

#### 3.7.1 分离为独立文件

```
core/
├── grid_tom.py          ← 仅保留 diskGrid 和网格 IO（write_model_grid/load_model_grid）
├── spec_io.py           ← 新建，从 grid_tom.py 末部迁移 SpecIO 内容（735 行 → 独立文件）
└── SpecIO.py            ← 可保留为向后兼容的重导出包装：
                            from core.spec_io import *  # noqa
```

#### 3.7.2 引入格式注册表

当前格式检测采用硬编码 if/elif 链。建议改用注册表模式，便于用户扩展自定义格式：

```python
# core/spec_io.py

class FormatRegistry:
    """Registry for spectral file format parsers."""
    _parsers: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass):
            cls._parsers[name] = klass
            return klass
        return decorator

    @classmethod
    def detect_and_parse(cls, text: str, hint: str = "auto") -> pd.DataFrame:
        ...

@FormatRegistry.register("lsd_pol")
class LsdPolParser:
    """Parser for full-polarimetry LSD format (I, V, Q, U, σ)."""
    ...

@FormatRegistry.register("lsd_i")
class LsdIParser:
    """Parser for intensity-only LSD format."""
    ...
```

用户添加新格式只需实现解析器类并调用 `@FormatRegistry.register("my_format")`，无需修改核心代码。

---

### 3.8 整合 `tbd/` 实验代码

`tbd/local_linemodel_ME.py` 实现了比现有弱场近似更完整的 Unno-Rachkovsky 线转移模型（含 Voigt 轮廓、源函数、辐射转移方程），是项目物理精度提升的关键。

#### 建议路径

1. **代码审查**：确认 `local_linemodel_ME.py` 的接口是否符合 `BaseLineModel.compute_local_profile()` 规范；
2. **迁移到 `core/`**：将其移入 `core/local_linemodel_ME.py`，并在 `core/local_linemodel_basic.py` 末尾注册：
   ```python
   # core/local_linemodel_basic.py
   __all__ = [
       "BaseLineModel",
       "GaussianZeemanWeakLineModel",
       "ConstantAmpLineModel",
       # ME model will be added here once integrated
   ]
   ```
3. **添加集成测试**：验证 ME 模型在弱场极限下与高斯近似的输出一致。
4. **删除 `tbd/` 目录**：迁移完成后清理，避免维护两份代码。

---

### 3.9 持续集成与代码质量门禁

目前项目已有 `.github/` 目录，建议补充或完善 CI 配置：

#### GitHub Actions 工作流建议

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest --cov=core --cov=pyzeetom --cov-fail-under=60

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install ruff mypy
      - run: ruff check core/ pyzeetom/
      - run: mypy core/ pyzeetom/ --ignore-missing-imports
```

#### 代码风格工具

建议在 `pyproject.toml` 中统一配置 `ruff`（速度快，兼容 flake8/isort/pyupgrade 规则）：

```toml
[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]   # 长行在物理公式注释中难以避免

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_ignores = true
```

---

## 4. 优先级路线图

### 阶段 1：紧急修复（1–2 周）

- [ ] **修复 `except Exception` 为 `except ValueError`**（`SpecIO` 解析逻辑，~5 行改动）
- [ ] **实现 `main()` CLI 入口**（`pyzeetom/tomography.py`，~60 行）
- [ ] **同步版本号**：`pyproject.toml` → `0.3.2`
- [ ] **收紧 `pyzeetom/__init__.py`**：显式导出 `forward_tomography`、`inversion_tomography`

### 阶段 2：核心质量提升（2–4 周）

- [ ] **添加物理单元测试**：线模型对称性、网格像素计数、MEM 熵梯度数值验证
- [ ] **默认启用响应矩阵缓存**
- [ ] **引入 `core/exceptions.py`** 自定义异常层次
- [ ] **集中参数默认值** 到 `core/config_defaults.py`
- [ ] **配置 ruff + mypy**，加入 CI

### 阶段 3：架构重构（1–2 个月）

- [ ] **拆分 `tomography_inversion.py`**：参数编解码、响应矩阵、反演主循环独立文件
- [ ] **拆分 `mainFuncs.py`**：加载器与验证器分离
- [ ] **`SpecIO` 迁移为独立文件 `core/spec_io.py`** + 引入格式注册表
- [ ] **向量化像素积分循环**（`VelspaceDiskIntegrator`）
- [ ] **迁移 `tbd/local_linemodel_ME.py`** 进 `core/`

### 阶段 4：生产化（长期）

- [ ] **废弃文本格式配置**，全面迁移 JSON
- [ ] **响应矩阵并行化**（`ProcessPoolExecutor`）
- [ ] **生成 API 参考文档**（`sphinx` 或 `mkdocs`）
- [ ] **发布 PyPI 包**（完善 `pyproject.toml` 元信息、添加 `CHANGELOG.md`）

---

## 5. 附录：代码指标一览

### 文件规模

| 文件 | 行数 | 主要职责 | 重构建议 |
|------|------|----------|----------|
| `core/tomography_inversion.py` | 1505 | 反演主循环 + 参数编解码 + 响应矩阵 | 拆分为 3 个文件 |
| `core/mainFuncs.py` | 1286 | 参数解析（文本+JSON）+ 兼容层 | 拆分为加载器+验证器 |
| `core/disk_geometry_integrator.py` | 1223 | 圆盘几何 + 速度积分 | 拆分为几何+积分器 |
| `core/tomography_result.py` | 926 | 结果容器 + 输出工具 | 可接受，考虑拆出输出工具 |
| `core/physical_model.py` | 782 | 物理模型统一初始化 | 可接受 |
| `core/tomography_config.py` | 771 | 配置 dataclass | 可接受 |
| `core/SpecIO.py`（嵌入） | 735 | 谱数据 I/O | 迁移为独立文件 |
| `core/mem_optimization.py` | 570 | 缓存、数据管道、稳定性监控 | 可接受 |
| `core/mem_generic.py` | 525 | 通用 MEM 优化器 | 良好 |

### 架构质量评级

| 维度 | 评级 | 说明 |
|------|------|------|
| 模块化 | **A** | 分层清晰，依赖方向正确 |
| 类型安全 | **A−** | 97% 类型标注，dataclass 覆盖广 |
| 文档质量 | **A−** | ARCHITECTURE.md 详尽，docstring 全面 |
| 错误处理 | **B+** | 97 处显式 raise，但存在异常吞噬 |
| 代码组织 | **B** | 3 个超大文件，参数打包逻辑重复 |
| 测试覆盖 | **D** | 仅 ~5% 行覆盖率，物理核心零覆盖 |
| 性能设计 | **B+** | 缓存基础设施完备，但未默认启用 |
| 可扩展性 | **A−** | 适配器+策略模式，新格式/线模型扩展便捷 |

---

*本报告基于对 pyZeeTom v0.3.2 源代码的静态分析，建议结合实际运行性能剖析（如 `cProfile`）进一步确认性能瓶颈位置，以确保优化工作事半功倍。*
