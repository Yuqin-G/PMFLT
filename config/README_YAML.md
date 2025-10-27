# YAML Configuration Files

本项目现在支持使用YAML格式的配置文件，提供更清晰和易于维护的配置管理方式。

## 📁 文件结构

```
config/
├── models/
│   ├── fedpha.yaml          # FedPHA模型配置
│   ├── fedclip.yaml         # FedCLIP模型配置
│   ├── fedotp.yaml          # FedOTP模型配置
│   ├── fedpgp.yaml          # FedPGP模型配置
│   ├── fedtpg.yaml          # FedTPG模型配置
│   ├── coop.yaml            # CoOp模型配置
│   ├── vlp.yaml             # VLP模型配置
│   ├── kgcoop.yaml          # kgCoOp模型配置
│   ├── pfedmoap.yaml        # pFedMoAP模型配置
│   ├── promptfl.yaml        # PromptFL模型配置
│   ├── promptfolio.yaml     # PromptFolio模型配置
│   └── cbm.yaml             # CBM模型配置
├── yaml_loader.py           # YAML配置加载器
└── factory.py               # 配置工厂（已更新支持YAML）
```

## 🚀 使用方法

### 1. 直接使用YAML加载器

```python
from config.yaml_loader import get_model_config

# 加载FedPHA配置
cfg = get_model_config('fedpha')
print(f"Model name: {cfg.MODEL.NAME}")
print(f"N_CTX: {cfg.MODEL.N_CTX}")
print(f"PFL: {cfg.MODEL.PFL}")
```

### 2. 使用配置工厂（推荐）

```python
from config.factory import get_model_config

# 使用YAML配置（默认）
cfg = get_model_config('fedpha', use_yaml=True)

# 使用Python配置（回退选项）
cfg = get_model_config('fedpha', use_yaml=False)
```

### 3. 完整的配置创建流程

```python
from config.factory import create_config

# 创建完整的配置（包含基础配置、模型配置和实验配置）
cfg = create_config(
    model_name='fedpha',
    exp_name='dtd_16_1',
    use_yaml=True  # 使用YAML配置
)

# 使用配置
print(f"Model: {cfg.MODEL.NAME}")
print(f"Dataset: {cfg.DATASET.NAME_SPACE}")
print(f"Learning rate: {cfg.OPTIM.LR}")
```

## 📝 YAML配置格式

### 基本结构

```yaml
# 模型基本配置
MODEL:
  NAME: 'fedpha'
  N_CTX: 4
  D_CTX: 1
  CTX_INIT: ""
  DEPTH: 0
  PFL: true  # 个性化联邦学习

# 训练器特定配置
TRAINER:
  GL_SVDMSE:
    N_CTX: 4
    CSC: false
    CTX_INIT: false
    PREC: "fp16"
    CLASS_TOKEN_POSITION: "end"
    N: 1
    lambda_orthogonal: 1
    alpha: 1.0
    ratio: 0.8
```

### 数据类型

- **字符串**: `'fedpha'`, `"fp16"`
- **数字**: `4`, `1.0`, `0.8`
- **布尔值**: `true`, `false`
- **空值**: `""` (空字符串)

## 🔧 配置特点

### 1. 向后兼容
- 保留了所有原有的Python配置文件
- 如果YAML加载失败，会自动回退到Python配置
- 现有代码无需修改即可使用

### 2. 易于维护
- YAML格式更直观，易于阅读和编辑
- 支持注释，便于理解配置项的作用
- 结构化层次清晰

### 3. 类型安全
- 自动转换为CfgNode对象
- 保持与原有配置系统完全兼容
- 支持嵌套配置结构

## 🧪 测试

运行测试脚本验证YAML配置：

```bash
# 简单测试（仅验证YAML文件有效性）
python test_yaml_simple.py

# 完整测试（需要安装yacs和pyyaml）
python test_yaml_config.py
```

## 📋 支持的模型

| 模型名称 | YAML文件 | 个性化FL | 特殊配置 |
|---------|----------|----------|----------|
| FedPHA | fedpha.yaml | ✅ | GL_SVDMSE |
| FedCLIP | fedclip.yaml | ❌ | FEDCLIP |
| FedOTP | fedotp.yaml | ✅ | GLP_OT |
| FedPGP | fedpgp.yaml | ✅ | FEDPGP |
| FedTPG | fedtpg.yaml | ❌ | - |
| CoOp | coop.yaml | ❌ | - |
| VLP | vlp.yaml | ❌ | - |
| kgCoOp | kgcoop.yaml | ❌ | - |
| pFedMoAP | pfedmoap.yaml | ✅ | PFEDMOAP |
| PromptFL | promptfl.yaml | ❌ | PROMPTFL |
| PromptFolio | promptfolio.yaml | ✅ | PLOT |
| CBM | cbm.yaml | ❌ | CBM |

## 🔄 迁移指南

### 从Python配置迁移到YAML

1. **无需修改现有代码**：配置工厂会自动使用YAML配置
2. **手动指定**：在调用时设置 `use_yaml=True`
3. **回退机制**：如果YAML加载失败，自动使用Python配置

### 添加新的YAML配置

1. 在 `config/models/` 目录下创建新的 `.yaml` 文件
2. 按照现有格式编写配置内容
3. 在 `yaml_loader.py` 的 `MODEL_CONFIG_MAP` 中添加映射
4. 运行测试验证配置正确性

## 🎯 优势

1. **可读性更强**：YAML格式比Python代码更直观
2. **维护性更好**：非程序员也能轻松修改配置
3. **版本控制友好**：YAML文件的diff更清晰
4. **跨语言支持**：YAML是通用格式，其他语言也能读取
5. **向后兼容**：不影响现有代码的使用

## ⚠️ 注意事项

1. YAML文件中的布尔值使用 `true`/`false`（小写）
2. 字符串值建议用引号包围，特别是包含特殊字符时
3. 缩进必须使用空格，不能使用制表符
4. 注释使用 `#` 符号
5. 如果修改了YAML文件，建议运行测试脚本验证
