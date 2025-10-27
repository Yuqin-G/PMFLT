"""
Configuration factory functions - YAML only version
Unified management of all configuration creation and merging
Now fully based on YAML configuration files
"""
from yacs.config import CfgNode as CN
from .base import get_base_cfg
from .datasets.experiments import get_experiment_configs
from .yaml_loader import get_model_config as get_yaml_model_config

def merge_configs_safely(base_cfg, model_cfg):
    """
    安全地合并配置，处理嵌套结构
    
    Args:
        base_cfg: 基础配置对象
        model_cfg: 模型配置对象
    """
    def merge_cfg_recursive(base_cfg, model_cfg, path=""):
        """递归合并 CfgNode"""
        for key in model_cfg.keys():
            current_path = f"{path}.{key}" if path else key
            
            if key in base_cfg:
                if isinstance(base_cfg[key], CN) and isinstance(model_cfg[key], CN):
                    # 递归合并嵌套的 CfgNode
                    merge_cfg_recursive(base_cfg[key], model_cfg[key], current_path)
                else:
                    # 直接覆盖
                    base_cfg[key] = model_cfg[key]
            else:
                # 直接添加新键
                base_cfg[key] = model_cfg[key]
    
    # 直接使用 CfgNode 进行合并
    merge_cfg_recursive(base_cfg, model_cfg)

def get_model_config(model_name):
    """
    Get model configuration based on model name (YAML only)
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        CfgNode: Model configuration object
    """
    return get_yaml_model_config(model_name)

def get_experiment_config(exp_name):
    """Get experiment configuration based on experiment name"""
    experiments = get_experiment_configs()
    
    if exp_name not in experiments:
        raise ValueError(f"Unknown experiment name: {exp_name}")
    
    return experiments[exp_name]

def create_config(model_name, exp_name, **kwargs):
    """
    Create complete configuration by merging base, model, and experiment configs
    
    Args:
        model_name (str): Name of the model
        exp_name (str): Name of the experiment
        **kwargs: Additional configuration parameters
        
    Returns:
        CfgNode: Complete configuration object
    """
    # Get base configuration
    cfg = get_base_cfg()
    
    # Get model configuration (YAML only)
    model_cfg = get_model_config(model_name)
    
    # Get experiment configuration
    exp_cfg = get_experiment_config(exp_name)
    
    # Merge configurations safely
    merge_configs_safely(cfg, model_cfg)
    
    # Handle experiment configuration with dot notation
    for key, value in exp_cfg.items():
        if '.' in key:
            # Handle nested configuration, e.g., 'DATASET.NAME_SPACE'
            keys = key.split('.')
            current = cfg
            for k in keys[:-1]:
                if k not in current:
                    current[k] = CN()
                current = current[k]
            current[keys[-1]] = value
        else:
            cfg[key] = value
    
    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    
    return cfg

def update_config_from_args(cfg, args):
    """
    Update configuration from command line arguments
    
    Args:
        cfg: Configuration object to update
        args: Parsed command line arguments
    """
    if hasattr(args, 'model_name') and args.model_name:
        cfg.MODEL.NAME = args.model_name
    
    if hasattr(args, 'backbone') and args.backbone:
        cfg.MODEL.BACKBONE = args.backbone
    
    if hasattr(args, 'num_epoch') and args.num_epoch:
        cfg.OPTIM.MAX_EPOCH = args.num_epoch

    if hasattr(args, 'comm_round') and args.comm_round:
        cfg.FEDERATED.COMM_ROUND = args.comm_round

    if hasattr(args, 'local_epoch') and args.local_epoch:
        cfg.FEDERATED.LOCAL_EPOCH = args.local_epoch
    
    if hasattr(args, 'use_swanlab') and args.use_swanlab:
        cfg.EXPERIMENT.USE_SWANLAB = args.use_swanlab

    if hasattr(args, 'save_gpu_memory') and args.save_gpu_memory:
        cfg.EXPERIMENT.SAVE_GPU_MEMORY = args.save_gpu_memory

    if hasattr(args, 'use_swanlab') and args.use_swanlab:
        cfg.EXPERIMENT.USE_SWANLAB = args.use_swanlab
    
    if hasattr(args, 'print_freq') and args.print_freq:
        cfg.EXPERIMENT.PRINT_FREQ = args.print_freq
    
    if hasattr(args, 'depth_ctx') and args.depth_ctx:
        cfg.MODEL.D_CTX = args.depth_ctx
    
    if hasattr(args, 'model_depth') and args.model_depth:
        cfg.MODEL.DEPTH = args.model_depth
    
    if hasattr(args, 'batch_size') and args.batch_size:
        cfg.DATASET.BATCH_SIZE = args.batch_size
        cfg.DATASET.TRAIN_BATCH_SIZE = args.batch_size
        # cfg.DATASET.TEST_BATCH_SIZE = args.batch_size
    
    if hasattr(args, 'num_cls_per_client') and args.num_cls_per_client:
        cfg.FEDERATED.NUM_CLASS_PER_CLIENT = args.num_cls_per_client
    
    if hasattr(args, 'avail_percent') and args.avail_percent:
        cfg.FEDERATED.AVAIL_PERCENT = args.avail_percent
    
    if hasattr(args, 'num_shots') and args.num_shots:
        cfg.DATASET.NUM_SHOTS = args.num_shots
    
    if hasattr(args, 'w') and args.w:
        cfg.MODEL.W = args.w
    
    if hasattr(args, 'data_mode') and args.data_mode:
        cfg.FEDERATED.DATA_MODE = args.data_mode
    
    if hasattr(args, 'alpha') and args.alpha:
        cfg.FEDERATED.ALPHA = args.alpha
    
    if hasattr(args, 'num_clients') and args.num_clients:
        cfg.FEDERATED.NUM_CLIENTS = args.num_clients
    
    if hasattr(args, 'single_target') and args.single_target:
        cfg.FEDERATED.SINGLE_TARGET = args.single_target
    
    if hasattr(args, 'target_domain') and args.target_domain:
        cfg.DATASET.TARGET_DOMAINS = [args.target_domain]
    
    return cfg
