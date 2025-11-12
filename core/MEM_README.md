# MEM 模块说明

## 状态：已废弃 (Deprecated)

`memSimple3.py` 和 `memSaim3.py` 是旧版本 ZDI (Zeeman Doppler Imaging) 流程的一部分，**在当前简化流程中已不再使用**。

## 历史背景

这些模块实现了基于 Skilling & Bryan (1984, MNRAS 211, 111) 的最大熵方法 (MEM) 图像重建算法，用于：
- 恒星表面亮度分布映射
- 磁场几何重建（球谐系数）
- 同时拟合 Stokes I 和 V 谱线轮廓

### 熵类型
1. **标准熵** (Img[0:n1]): 用于亮度映射（正值，无上限）
2. **填充因子熵** (Img[n1:n2]): 限制上限的正值熵
3. **磁熵** (Img[n2:ntot]): 正负双向熵（Hobson & Lasenby 1998）

## 当前流程

新的简化流程位于 `pyzeetom/tomography.py`，直接基于：
- 局部线模型 (`local_linemodel_basic.py`, `local_linemodel_ME.py`)
- 盘面积分 (`velspace_DiskIntegrator.py`)
- 网格构建 (`grid_tom.py`)

**不再需要 MEM 迭代优化**，直接生成合成谱线。

## 保留原因

1. **兼容性**: `core/mainFuncs.py` 中的 `readParamsZDI` 函数仍使用 MEM（旧版本接口）
2. **参考价值**: 完整的 MEM 实现可供研究和算法参考
3. **潜在恢复**: 如需恢复迭代优化功能，可基于此代码重构

## 主要函数

- `mem_iter()`: 主迭代函数
- `packDataVector()`, `packResponseMatrix()`: 数据打包
- `packImageVector()`, `unpackImageVector()`: 参数向量转换
- `constantsMEM`: 控制常数类
- `control()`: Skilling-Bryan 控制过程
- `searchDir()`, `diagDir()`: 搜索方向计算

## 如果需要使用

如果确实需要使用 MEM 功能，请参考 `core/mainFuncs.py` 中的旧版本实现。但注意：
- 需要磁场几何对象 (`magGeom`)
- 需要亮度映射对象 (`briMap`)
- 需要响应矩阵计算（谱线对参数的导数）

## 替代方案

对于反演/拟合需求，建议：
1. 使用标准最小二乘法
2. 采用 MCMC 或其他贝叶斯方法
3. 基于神经网络的反演方法

---

**如有疑问，请参考 `.github/copilot-instructions.md` 中的项目结构说明。**
