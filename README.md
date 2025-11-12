
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
2. **谱线参数文件**（如`lines.txt`）：每行`wl0 [strength] sigWl g [limbDark]`
3. **观测数据**：支持多格式，需含Stokes I/V/Q/U及误差

示例见 `input/` 目录。

## 核心结构

- `core/grid_tom.py`：环状/盘面网格生成
- `core/local_linemodel_basic.py` / `local_linemodel_ME.py`：局部谱线模型
- `core/velspace_DiskIntegrator.py`：速度空间积分
- `core/readObs.py`：观测数据读取与标准化
- `pyzeetom/tomography.py`：主流程入口

> 详细结构与开发约定见 [copilot-instructions.md](.github/copilot-instructions.md)

## 文件与格式约定

- **参数文件**：前14行为主控参数，第5行为谱线模型参数，第11/12行可指定谱线/观测格式，14行后为观测文件名、JD、velR等
- **谱线参数文件**：每行`wl0 [strength] sigWl g [limbDark]`，最少需`wl0, sigWl, g`
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
