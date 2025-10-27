"""
Experiment configuration definitions
"""

def get_experiment_configs():
    """Get all experiment configurations"""
    experiments = {
        # Cross-dataset experiments
        'cross_data': {
            'DATASET.NAME_SPACE': ["imagenet"],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': [
                'caltech101', 'oxford_flowers', 'fgvc_aircraft', 'ucf101', 
                'oxford_pets', 'food101', 'dtd', 'stanford_cars', 'sun397', 'eurosat'
            ],
            'DATASET.NUM_CLASS': 1000
        },
        
        # Cross-domain experiments
        'cross_domain': {
            'DATASET.NAME_SPACE': ["imagenet"],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': [
                'imagenet-v2', 'imagenet-s', 'imagenet-a', 'imagenet-r', 'imagenet'
            ]
        },
        
        # Cross-class experiments
        'cross_cls': {
            'DATASET.NAME_SPACE': [
                'caltech101', 'oxford_flowers', 'fgvc_aircraft', 'ucf101', 
                'oxford_pets', 'food101', 'dtd', 'stanford_cars', 'sun397'
            ],
            'DATASET.SPLIT.TRAIN': 'base',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': [
                'caltech101', 'oxford_flowers', 'fgvc_aircraft', 'ucf101', 
                'oxford_pets', 'food101', 'dtd', 'stanford_cars', 'sun397'
            ]
        },
        
        # Single dataset experiments
        'caltech101': {
            'DATASET.NAME_SPACE': ['caltech101'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['caltech101'],
            'DATASET.NUM_CLASS': 100
        },
        
        'oxford_flowers': {
            'DATASET.NAME_SPACE': ['oxford_flowers'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['oxford_flowers'],
            'DATASET.NUM_CLASS': 102
        },
        
        'fgvc_aircraft': {
            'DATASET.NAME_SPACE': ['fgvc_aircraft'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['fgvc_aircraft'],
            'DATASET.NUM_CLASS': 100
        },
        
        'ucf101': {
            'DATASET.NAME_SPACE': ['ucf101'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['ucf101'],
            'DATASET.NUM_CLASS': 101
        },
        
        'oxford_pets': {
            'DATASET.NAME_SPACE': ['oxford_pets'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['oxford_pets'],
            'DATASET.NUM_CLASS': 37
        },
        
        'food101': {
            'DATASET.NAME_SPACE': ['food101'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['food101'],
            'DATASET.NUM_CLASS': 101
        },
        
        'dtd': {
            'DATASET.NAME_SPACE': ['dtd'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['dtd'],
            'DATASET.NUM_CLASS': 47
        },
        
        'stanford_cars': {
            'DATASET.NAME_SPACE': ['stanford_cars'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['stanford_cars'],
            'DATASET.NUM_CLASS': 196
        },
        
        'sun397': {
            'DATASET.NAME_SPACE': ['sun397'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['sun397'],
            'DATASET.NUM_CLASS': 397
        },
        
        'eurosat': {
            'DATASET.NAME_SPACE': ['eurosat'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['eurosat'],
            'DATASET.NUM_CLASS': 10
        },
        
        'cifar100': {
            'DATASET.NAME_SPACE': ['cifar100'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['cifar100'],
            'DATASET.NUM_CLASS': 100
        },
        
        'cifar10': {
            'DATASET.NAME_SPACE': ['cifar10'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['cifar10'],
            'DATASET.NUM_CLASS': 10
        },
        
        'resisc45': {
            'DATASET.NAME_SPACE': ['resisc45'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['resisc45'],
            'DATASET.NUM_CLASS': 45
        },
        
        'cub200': {
            'DATASET.NAME_SPACE': ['cub200'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['cub200'],
            'DATASET.NUM_CLASS': 200
        },
        
        # Domain shift experiments
        'pacs': {
            'DATASET.DOMAIN': True,
            'DATASET.NAME_SPACE': ['pacs'],
            'DATASET.SOURCE_DOMAINS': ['art_painting', 'cartoon', 'photo', 'sketch'],
            'DATASET.TARGET_DOMAINS': ['art_painting', 'cartoon', 'photo', 'sketch'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['pacs'],
            'DATASET.NUM_DOMAINS': 4,
            'DATASET.NUM_CLASS': 7,
        },

        'domainnet': {
            'DATASET.DOMAIN': True,
            'DATASET.NAME_SPACE': ['domainnet'],
            'DATASET.SOURCE_DOMAINS': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
            'DATASET.TARGET_DOMAINS': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
            'DATASET.SPLIT.TRAIN': 'all',
            'DATASET.SPLIT.TEST': 'all',
            'DATASET.TESTNAME_SPACE': ['domainnet'],
            'DATASET.NUM_DOMAINS': 6,
            'DATASET.NUM_CLASS': 345,
        },
    }
    
    return experiments
