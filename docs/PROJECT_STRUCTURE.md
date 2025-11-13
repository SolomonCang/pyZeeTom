# pyZeeTom 项目结构简明指南

## 目录结构

```
pyZeeTom/
├── core/                          # 核心模块
│   ├── grid_tom.py               # ✓ 盘面网格生成 (diskGrid)
│   ├── readObs.py                # ✓ 观测数据读取 (多格式支持)
│   ├── local_linemodel_basic.py  # ✓ 基础局部线模型 (弱场高斯Zeeman)
│   ├── velspace_DiskIntegrator.py # ✓ 速度空间盘面积分
│   ├── mainFuncs.py              # ✓ 主函数集 (含旧版接口)
│   ├── memSimple3.py             # ✗ [已废弃] MEM 优化算法
│   └── MEM_README.md             # MEM 模块说明文档
│
├── pyzeetom/
│   ├── __init__.py               # 包初始化
│   └── tomography.py             # ✓ 新版简化主入口
│
├── tbd/                          # 待定/实验性功能
│   ├── local_linemodel_ME.py     # Milne-Eddington 线模型
│   └── lines_ME.txt              # ME 模型谱线参数
│
├── examples/                     # 示例配置
│   ├── params_example.tomog      # 参数文件模板
│   └── lines.txt                 # 谱线参数文件
│
├── test/                         # 测试
│   └── test_grid_tom.py
│
├── tomography.py                 # 根级薄包装 (调用 pyzeetom/)
├── pyproject.toml               # 项目配置
└── .github/
    └── copilot-instructions.md   # AI 编码代理指南
```

## 模块状态标识

- ✓ **活跃使用**: 当前流程核心模块
- ✗ **已废弃**: 保留用于兼容性，不推荐使用
- ⚠ **实验性**: 功能不完整或待验证

## 核心流程 (简化版本)

```
参数读取 (mainFuncs.readParamsTomog)
    ↓
观测数据读取 (readObs.py)
    ↓
盘面网格构建 (grid_tom.diskGrid)
    ↓
局部线模型初始化 (local_linemodel_basic)
    ↓
速度空间积分 (velspace_DiskIntegrator)
    ↓
生成合成谱 (I/V)
```

## 快速开始

### 1. 准备输入文件

```bash
examples/
  ├── params_example.tomog   # 参数配置
  ├── lines.txt              # 谱线参数
  └── obs_*.dat              # 观测数据
```

### 2. 运行合成

```python
from pyzeetom import tomography

# 方式1: 使用默认配置
tomography.run_tomography('params_example.tomog')

# 方式2: 通过主模块
from core import mainFuncs
params = mainFuncs.readParamsTomog('params.tomog')
# ... 后续处理
```

### 3. 输出文件

- `outFitSummary.txt`: 拟合摘要
- `phase_*.model`: 各相位模型谱 (LSD格式)

## 重要约定

### 1. 数组形状
- **像素数组**: 一维 `(Npix,)`
- **波长网格**: `(Nlambda,)` 或广播为 `(Nlambda, 1)`
- **谱线输出**: `(Nlambda, Npix)` 或 `(Nlambda,)` (积分后)

### 2. 物理量约定
- **连续谱基线**: 固定为 1.0
- **吸收线**: `amp < 0`
- **速度单位**: km/s
- **磁场**: `Blos` (视向), `Bperp` (垂直) + 角度 `chi`

### 3. 文件格式
- **参数文件**: 行 0-13 保留字段，行 14+ 观测列表
- **谱线文件**: 至少包含 `wl0, sigWl, g` 列
- **观测数据**: 支持 LSD/Spec, pol/I/simple 格式

## 废弃功能说明

### MEM 优化 (memSimple3.py)
- **用途**: 最大熵方法迭代优化亮度/磁场分布
- **现状**: 已被直接合成方法替代
- **保留原因**: 兼容 `mainFuncs.readParamsZDI`
- **详见**: `core/MEM_README.md`

### 响应函数 (resp_tom.py)
- **用途**: 谱线响应计算接口
- **现状**: 已被解析导数替代

## 扩展开发

### 添加新线模型

```python
from core.local_linemodel_basic import BaseLineModel

class MyLineModel(BaseLineModel):
    def compute_local_profile(self, wl_grid, amp, Blos, **kwargs):
        """
        参数:
            wl_grid: (Nlambda, Npix) 波长网格
            amp: (Npix,) 线强
            Blos: (Npix,) 视向磁场
        返回:
            {'I': array, 'V': array}  # (Nlambda, Npix)
        """
        # 实现你的线模型
        pass
```

### 添加观测读取器

参考 `readObs.py` 中的格式探测逻辑，确保返回：
- `wl`: 波长数组
- `specI`, `specV`: Stokes 谱
- `specIsig`, `specVsig`: 误差

## 测试

```bash
# 运行所有测试
python -m pytest test/

# 快速测试
python -m pytest -q test/test_grid_tom.py
```

## 常见问题

### Q: MEM 相关导入错误？
A: 检查是否误用了旧版流程。新版不需要 `memSimple3` 或 `memSaim3`。

### Q: 数组形状不匹配？
A: 确保像素数组为一维，需要广播时显式 reshape。

### Q: 如何切换线模型？
A: 修改参数文件第5行的线模型参数，或在代码中传入不同的线模型对象。

---

**详细说明**: `.github/copilot-instructions.md`
**MEM 说明**: `core/MEM_README.md`
