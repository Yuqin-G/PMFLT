# YAML配置清理指南

如果完全使用YAML配置，可以删除以下内容：

## 🗑️ 可以删除的导入

### 原来的导入（可以删除）：
```python
# 这些导入都可以删除
from .models.fedtpg import get_fedtpg_cfg, get_coop_cfg, get_vlp_cfg, get_kgcoop_cfg
from .models.fedpgp import get_fedpgp_cfg
from .models.fedotp import get_fedotp_cfg
from .models.promptfl import get_promptfl_cfg
from .models.fedclip import get_fedclip_cfg
from .models.pfedmoap import get_pfedmoap_cfg
from .models.fedpha import get_fedpha_cfg
from .models.promptfolio import get_promptfolio_cfg
from .models.cbm import get_cbm_cfg
```

### 简化后的导入（保留）：
```python
from yacs.config import CfgNode as CN
from .base import get_base_cfg
from .datasets.experiments import get_experiment_configs
from .yaml_loader import get_model_config
```

## 🗑️ 可以删除的代码

### 1. Python配置字典（可以删除）：
```python
# 这个字典可以完全删除
model_configs = {
    'fedtpg': get_fedtpg_cfg,
    'coop': get_coop_cfg,
    'vlp': get_vlp_cfg,
    'kgcoop': get_kgcoop_cfg,
    'fedpgp': get_fedpgp_cfg,
    'fedotp': get_fedotp_cfg,
    'promptfl': get_promptfl_cfg,
    'fedclip': get_fedclip_cfg,
    'pfedmoap': get_pfedmoap_cfg,
    'fedpha': get_fedpha_cfg,
    'promptfolio': get_promptfolio_cfg,
    'cbm': get_cbm_cfg,
}
```

### 2. 回退机制（可以删除）：
```python
# 这些回退逻辑可以删除
if not use_yaml:
    # ... 回退到Python配置的代码
```

### 3. 简化的函数（替换）：
```python
# 原来的复杂函数
def get_model_config(model_name, use_yaml=True):
    # ... 复杂的回退逻辑

# 简化为
def get_model_config(model_name):
    return get_yaml_model_config(model_name)
```

## 📁 可以删除的文件

### Python配置文件（可选删除）：
```
config/models/
├── fedpha.py          # 可以删除
├── fedclip.py         # 可以删除
├── fedotp.py          # 可以删除
├── fedpgp.py          # 可以删除
├── fedtpg.py          # 可以删除
├── pfedmoap.py        # 可以删除
├── promptfl.py        # 可以删除
├── promptfolio.py     # 可以删除
└── cbm.py             # 可以删除
```

### 保留的文件：
```
config/models/
├── fedpha.yaml        # 保留
├── fedclip.yaml       # 保留
├── fedotp.yaml        # 保留
├── fedpgp.yaml        # 保留
├── fedtpg.yaml        # 保留
├── coop.yaml          # 保留
├── vlp.yaml           # 保留
├── kgcoop.yaml        # 保留
├── pfedmoap.yaml      # 保留
├── promptfl.yaml      # 保留
├── promptfolio.yaml   # 保留
└── cbm.yaml           # 保留
```

## 🔄 迁移步骤

### 步骤1：备份现有文件
```bash
cp config/factory.py config/factory_backup.py
```

### 步骤2：使用YAML-only版本
```bash
cp config/factory_yaml_only.py config/factory.py
```

### 步骤3：测试配置加载
```python
from config.factory import get_model_config
cfg = get_model_config('fedpha')
print(cfg.MODEL.NAME)  # 应该输出: fedpha
```

### 步骤4：删除Python配置文件（可选）
```bash
# 如果确定不再需要Python配置，可以删除
rm config/models/*.py
```

## ⚠️ 注意事项

1. **向后兼容**：删除Python配置文件后，将无法回退到Python配置
2. **测试充分**：确保所有YAML配置文件都正确且完整
3. **备份重要**：删除前务必备份重要文件
4. **渐进迁移**：建议先测试YAML-only版本，确认无误后再删除Python文件

## 📊 清理效果

| 项目 | 清理前 | 清理后 | 减少 |
|------|--------|--------|------|
| 导入语句 | 12个 | 4个 | -8个 |
| 函数复杂度 | 高 | 低 | 简化 |
| 文件数量 | 24个 | 12个 | -12个 |
| 维护成本 | 高 | 低 | 降低 |

## 🎯 最终效果

清理后的配置系统：
- ✅ 只使用YAML配置文件
- ✅ 统一的配置加载接口
- ✅ 更简洁的代码结构
- ✅ 更低的维护成本
- ✅ 更好的可读性
