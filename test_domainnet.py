#!/usr/bin/env python3
"""测试DomainNet数据集能否正常导入和加载"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_domainnet_import():
    """测试DomainNet模块导入"""
    print("测试DomainNet模块导入...")
    
    try:
        from dataloader.domain_datasets import DomainNet
        print("✅ DomainNet导入成功")
        return True
    except ImportError as e:
        print(f"❌ DomainNet导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 导入过程中出现其他错误: {e}")
        return False

def test_domainnet_data_structure():
    """测试DomainNet数据集结构"""
    print("\n测试DomainNet数据集结构...")
    
    # 检查数据目录
    domainnet_paths = [
        "/data2/gzh/data/domainnet",
        "/data2/gzh/FL_CLIP/data/domainnet",
        "/data2/gzh/FL_CLIP/FLVLM-main/data/domainnet",
        "./data/domainnet"
    ]
    
    found_domainnet = False
    domain_stats = {}
    
    for path in domainnet_paths:
        if os.path.exists(path):
            print(f"✅ 找到DomainNet目录: {path}")
            found_domainnet = True
            
            # 检查子目录
            subdirs = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
            print("\n📊 各域数据统计:")
            print("=" * 50)
            
            total_images = 0
            for subdir in subdirs:
                subdir_path = os.path.join(path, subdir)
                if os.path.exists(subdir_path):
                    # 统计图像文件数量
                    image_count = 0
                    class_count = 0
                    
                    for item in os.listdir(subdir_path):
                        item_path = os.path.join(subdir_path, item)
                        if os.path.isdir(item_path):
                            # 这是一个类别目录
                            class_count += 1
                            # 统计该类别下的图像文件
                            for file in os.listdir(item_path):
                                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                                    image_count += 1
                    
                    domain_stats[subdir] = {
                        'classes': class_count,
                        'images': image_count
                    }
                    total_images += image_count
                    
                    print(f"  📁 {subdir:12} | 类别数: {class_count:3d} | 图像数: {image_count:6d}")
                else:
                    print(f"  ❌ {subdir}/ 不存在")
                    domain_stats[subdir] = {'classes': 0, 'images': 0}
            
            print("=" * 50)
            print(f"  📊 总计      | 类别数: {sum(stats['classes'] for stats in domain_stats.values()):3d} | 图像数: {total_images:6d}")
            print("=" * 50)
            
            # 显示每个域的详细信息
            print("\n📋 各域详细信息:")
            for domain, stats in domain_stats.items():
                if stats['images'] > 0:
                    print(f"  {domain}: {stats['classes']} 个类别, {stats['images']} 张图像")
                else:
                    print(f"  {domain}: 无数据")
            
            break  # 找到第一个有效路径就停止
        else:
            print(f"❌ 目录不存在: {path}")
    
    if not found_domainnet:
        print("⚠️  未找到DomainNet数据集目录，可能需要下载数据集")
    
    return found_domainnet

def test_domainnet_config():
    """测试DomainNet配置"""
    print("\n测试DomainNet配置...")
    
    # 创建DomainNet配置示例
    domainnet_config = {
        "DATASET": {
            "NAME": "DomainNet",
            "SOURCE_DOMAINS": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
            "TARGET_DOMAINS": ["sketch"],  # 示例：使用sketch作为目标域
            "BETA": 0.5,
            "PATH": "/data2/gzh/data/domainnet"
        },
        "DATA": {
            "BATCH_SIZE": 32,
            "NUM_WORKERS": 0
        },
        "FEDERATED": {
            "DATA_MODE": "domain",
            "NUM_CLIENTS": 6
        }
    }
    
    print("✅ DomainNet配置示例:")
    print(f"  - 数据集名称: {domainnet_config['DATASET']['NAME']}")
    print(f"  - 源域列表: {domainnet_config['DATASET']['SOURCE_DOMAINS']}")
    print(f"  - 目标域列表: {domainnet_config['DATASET']['TARGET_DOMAINS']}")
    print(f"  - 数据路径: {domainnet_config['DATASET']['PATH']}")
    print(f"  - 数据模式: {domainnet_config['FEDERATED']['DATA_MODE']}")
    print(f"  - 客户端数量: {domainnet_config['FEDERATED']['NUM_CLIENTS']}")
    
    return True

def test_domainnet_dataset_loading():
    """测试DomainNet数据集加载（不实际加载，只测试函数调用）"""
    print("\n测试DomainNet数据集加载...")
    
    try:
        # 测试导入必要的模块
        import torch
        import torchvision.transforms as transforms
        print("✅ PyTorch和torchvision导入成功")
        
        # 测试DomainNet类是否存在
        from dataloader.domain_datasets import DomainNet
        print("✅ DomainNet类可用")
        
        # 检查DomainNet的初始化参数
        import inspect
        sig = inspect.signature(DomainNet.__init__)
        params = list(sig.parameters.keys())
        print(f"✅ DomainNet初始化参数: {params}")
        
        # 测试数据变换
        transform_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ])
        print("✅ 数据变换定义成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 数据集加载测试失败: {e}")
        return False

def test_domainnet_with_server():
    """测试DomainNet与服务器集成"""
    print("\n测试DomainNet与服务器集成...")
    
    try:
        from federated.server_base import ServerBase
        from config.defaults import get_cfg_default
        
        # 创建配置
        cfg = get_cfg_default()
        cfg.DATASET.NAME = "DomainNet"
        cfg.DATASET.SOURCE_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        cfg.DATASET.TARGET_DOMAINS = ["sketch"]
        cfg.DATASET.PATH = "/data2/gzh/data/domainnet"
        cfg.FEDERATED.DATA_MODE = "domain"
        cfg.FEDERATED.NUM_CLIENTS = 6
        cfg.DATASET.BETA = 0.5
        cfg.DATASET.TRAIN_BATCH_SIZE = 32
        cfg.DATASET.TEST_BATCH_SIZE = 128
        cfg.DATASET.NUM_WORKERS = 0
        
        print("✅ 配置创建成功")
        
        # 检查服务器是否有prepare_domain_data方法
        if hasattr(ServerBase, 'prepare_domain_data'):
            print("✅ ServerBase有prepare_domain_data方法")
        else:
            print("❌ ServerBase缺少prepare_domain_data方法")
            return False
        
        # 检查配置是否正确设置
        if cfg.DATASET.NAME == "DomainNet":
            print("✅ 数据集名称配置正确")
        else:
            print("❌ 数据集名称配置错误")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 服务器集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_domainnet_file_structure():
    """测试DomainNet文件结构"""
    print("\n测试DomainNet文件结构...")
    
    # 检查domain_datasets.py文件
    domain_datasets_path = "dataloader/domain_datasets.py"
    if os.path.exists(domain_datasets_path):
        with open(domain_datasets_path, 'r') as f:
            content = f.read()
        
        # 检查DomainNet相关类和方法
        checks = [
            ("class DomainNet", "DomainNet类"),
            ("def __init__", "初始化方法"),
            ("def __getitem__", "数据获取方法"),
            ("def __len__", "长度方法"),
            ("clipart", "clipart域支持"),
            ("infograph", "infograph域支持"),
            ("painting", "painting域支持"),
            ("quickdraw", "quickdraw域支持"),
            ("real", "real域支持"),
            ("sketch", "sketch域支持")
        ]
        
        for pattern, description in checks:
            if pattern in content:
                print(f"✅ {description}存在")
            else:
                print(f"❌ {description}不存在")
        
        return True
    else:
        print("❌ domain_datasets.py文件不存在")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DomainNet数据集导入和加载测试")
    print("=" * 60)
    
    tests = [
        ("DomainNet模块导入", test_domainnet_import),
        ("DomainNet数据集结构", test_domainnet_data_structure),
        ("DomainNet配置", test_domainnet_config),
        ("DomainNet数据集加载", test_domainnet_dataset_loading),
        ("DomainNet与服务器集成", test_domainnet_with_server),
        ("DomainNet文件结构", test_domainnet_file_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试 {test_name} 出现异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 DomainNet数据集可以正常导入和使用!")
        print("\n下一步建议:")
        print("1. 确保DomainNet数据集已下载到正确位置")
        print("2. 测试完整的数据加载流程")
        print("3. 验证域适应功能")
    else:
        print("\n⚠️  部分测试失败，需要进一步检查")
        print("\n可能的问题:")
        print("1. 缺少必要的依赖包")
        print("2. DomainNet数据集未下载或路径不正确")
        print("3. 代码实现不完整")
