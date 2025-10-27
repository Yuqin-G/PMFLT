"""
Dataset configuration definitions
"""
from yacs.config import CfgNode as CN

def get_dataset_configs():
    """Get all dataset configurations"""
    datasets = {
        'imagenet': {
            'num_classes': 1000,
            'description': 'ImageNet-1K dataset'
        },
        'caltech101': {
            'num_classes': 100,
            'description': 'Caltech-101 dataset'
        },
        'oxford_flowers': {
            'num_classes': 102,
            'description': 'Oxford Flowers-102 dataset'
        },
        'eurosat': {
            'num_classes': 10,
            'description': 'EuroSAT dataset'
        },
        'oxford_pets': {
            'num_classes': 37,
            'description': 'Oxford Pets dataset'
        },
        'fgvc_aircraft': {
            'num_classes': 100,
            'description': 'FGVC Aircraft dataset'
        },
        'food101': {
            'num_classes': 101,
            'description': 'Food-101 dataset'
        },
        'dtd': {
            'num_classes': 47,
            'description': 'Describable Textures Dataset'
        },
        'ucf101': {
            'num_classes': 101,
            'description': 'UCF-101 dataset'
        },
        'stanford_cars': {
            'num_classes': 196,
            'description': 'Stanford Cars dataset'
        },
        'sun397': {
            'num_classes': 397,
            'description': 'SUN397 dataset'
        },
        'resisc45': {
            'num_classes': 45,
            'description': 'RESISC45 dataset'
        },
        'cifar100': {
            'num_classes': 100,
            'description': 'CIFAR-100 dataset'
        },
        'cifar10': {
            'num_classes': 10,
            'description': 'CIFAR-10 dataset'
        },
        'cub200': {
            'num_classes': 200,
            'description': 'CUB-200 dataset'
        }
    }
    
    return datasets
