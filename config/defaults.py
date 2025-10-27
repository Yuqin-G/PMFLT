"""
Simplified default configuration file
Using the new registry-based system
"""
from .simple import create_config

def get_cfg_defaults():
    """Get default configuration using the simplified system"""
    return create_config(
        model_name='fedtpg',
        exp_name='caltech101',
        DATASET={
            'ROOT': '/data2/gzh/data',
            'NUM_SHOTS': 4
        },
        EXPERIMENT={
            'OUTPUT_DIR': './output'
        },
        SYSTEM={
            'SEED': -1,
            'USE_CUDA': True,
            'VERBOSE': True
        },
        OPTIMIZER={
            'NAME': 'sgd',
            'LR': 0.003,
            'WEIGHT_DECAY': 1e-5,
            'MOMENTUM': 0.9,
            'MAX_EPOCH': 1000
        },
        TRAIN={
            'NUM_CLASS_PER_CLIENT': 10,
            'AVAIL_PERCENT': 1.0,
            'SPLIT': 'base',
            'W': 8.0
        },
        FEDERATED={
            'DATA_MODE': 'few_shot',
            'ALPHA': 1.0,
            'NUM_CLIENTS': 10
        }
    )

# For backward compatibility, keep the original _C variable
_C = get_cfg_defaults()