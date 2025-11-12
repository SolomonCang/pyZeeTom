# 测试运行指南

本项目提供了两个便捷的测试运行脚本：

## 快速开始

### Bash 脚本 (macOS/Linux 推荐)
```bash
./run_tests.sh
```

### Python 脚本 (跨平台)
```bash
python run_tests.py
```

## 使用示例

### 基础用法
```bash
# 运行所有测试（详细模式）
./run_tests.sh
# 或
python run_tests.py

# 简洁输出
./run_tests.sh -q
python run_tests.py -q

# 超详细输出
./run_tests.sh -vv
python run_tests.py -vv
```

### 仅运行几何网格测试
```bash
./run_tests.sh --grid-only
python run_tests.py --grid-only
```

### 启用代码覆盖率
```bash
./run_tests.sh --cov
python run_tests.py --cov
```

### 查看帮助
```bash
./run_tests.sh -h
python run_tests.py -h
```

## 依赖管理

脚本会自动检查并提示安装缺失的依赖：

- **pytest**: 测试框架（必需）
- **matplotlib**: 几何图生成（可选，图像测试需要）
- **pytest-cov**: 代码覆盖率（可选，仅 --cov 时需要）

### 手动安装开发依赖
```bash
# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows

# 安装开发依赖
pip install -e .[dev]
```

## 生成几何图片示例

测试期间会在临时目录生成图片。若需保存示例图：

```bash
python -c "from core.grid_tom import visualize_grid; \
visualize_grid(nr=20, r_in=0, r_out=200, \
save_path='grid_example.png', show=False)"
```

生成的 `grid_example.png` 展示了盘网格结构。

## 故障排查

### 虚拟环境未激活
如果看到警告，请先激活虚拟环境：
```bash
source .venv/bin/activate
```

### matplotlib 未安装
几何图生成测试会被跳过，但不影响其他测试。若需运行该测试：
```bash
pip install matplotlib
```

### pytest 未找到
脚本会自动尝试安装，若失败请手动运行：
```bash
pip install -e .[dev]
```

## 测试文件结构

```
test/
├── test_grid_tom.py       # 几何网格测试（含图像生成）
└── (其他测试文件...)
```

## 持续集成

可在 CI 环境中使用：
```yaml
# GitHub Actions 示例
- name: Run tests
  run: python run_tests.py --cov
```

## 更多选项

所有选项说明：

| 选项 | 说明 |
|------|------|
| `-q, --quiet` | 简洁输出模式 |
| `-vv, --verbose` | 详细输出模式 |
| `--cov, --coverage` | 启用代码覆盖率报告 |
| `--grid-only` | 仅运行几何网格测试 |
| `-h, --help` | 显示帮助信息 |
