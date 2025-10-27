"""
YAML configuration loader for model configurations
"""
import os
import yaml
from yacs.config import CfgNode as CN

def load_yaml_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert dictionary to CfgNode
    cfg = CN(config_dict)
    return cfg

def get_model_config_from_yaml(model_name):
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    
    # Construct YAML file path
    yaml_path = os.path.join(models_dir, f'{model_name}.yaml')
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    
    return load_yaml_config(yaml_path)

# Model configuration mapping
MODEL_CONFIG_MAP = {
    'fedpha': 'fedpha',
    'fedclip': 'fedclip', 
    'fedotp': 'fedotp',
    'fedpgp': 'fedpgp',
    'fedtpg': 'fedtpg',
    'coop': 'coop',
    'vlp': 'vlp',
    'kgcoop': 'kgcoop',
    'pfedmoap': 'pfedmoap',
    'promptfl': 'promptfl',
    'promptfolio': 'promptfolio',
    'cbm': 'cbm'
}

def get_model_config(model_name):
    return get_model_config_from_yaml(model_name)
    if model_name not in MODEL_CONFIG_MAP:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_CONFIG_MAP.keys())}")
    
    config_file = MODEL_CONFIG_MAP[model_name]
    return get_model_config_from_yaml(config_file)
