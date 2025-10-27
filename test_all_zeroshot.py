#!/usr/bin/env python3
"""
测试所有方法的 zero-shot 性能
"""

import sys
import os
import time
import subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.base import get_base_cfg
from federated.server_base import ServerBase

def test_zeroshot_performance(algo_name, dataset="dtd", backbone="ViT-B/16"):
    """测试单个算法的 zero-shot 性能"""
    print(f"\n{'='*60}")
    print(f"测试 {algo_name} 的 Zero-Shot 性能")
    print(f"数据集: {dataset}, Backbone: {backbone}")
    print(f"{'='*60}")
    
    try:
        # 创建配置
        cfg = get_base_cfg()
        
        # 设置基本参数
        cfg.MODEL.NAME = algo_name
        cfg.MODEL.BACKBONE = backbone
        cfg.DATASET.NAME_SPACE = [dataset]
        cfg.DATASET.TESTNAME_SPACE = [dataset]
        cfg.DATASET.NUM_SHOTS = 16
        cfg.FEDERATED.NUM_CLASS_PER_CLIENT = 10
        cfg.EXPERIMENT.NAME = f"zeroshot_{algo_name}_{dataset}"
        cfg.EXPERIMENT.OUTPUT_DIR = "./outputs"
        cfg.EXPERIMENT.SEED = 42
        
        # 冻结配置
        cfg.freeze()
        
        # 记录开始时间
        start_time = time.time()
        
        # 创建服务器实例并运行测试
        server = ServerBase(cfg)
        
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n{algo_name} Zero-Shot 测试完成，耗时: {elapsed_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"\n{algo_name} 测试失败: {str(e)}")
        return False

def main():
    # 定义所有要测试的算法
    algorithms = [
        "coop",           # 基础 CoOp
        "kgcoop",         # 知识图谱 CoOp
        "fedtpg",         # FedTPG
        "fedpha",         # FedPHA
        "fedpgp",         # FedPGP
        "pfedmoap",       # pFedMoAP
        "fedotp",         # FedOTP
        "promptfl",       # PromptFL
        "promptfolio",    # PromptFolio
        "fedclip",        # FedCLIP
        "vlp",            # VLP
        "cbm"             # CBM
    ]
    
    # 定义要测试的数据集
    datasets = ["dtd"]  # 可以添加更多数据集
    
    # 定义要测试的backbone
    backbones = ["ViT-B/16"]  # 可以添加更多backbone
    
    print("开始测试所有方法的 Zero-Shot 性能")
    print(f"算法数量: {len(algorithms)}")
    print(f"数据集: {datasets}")
    print(f"Backbone: {backbones}")
    print("="*80)
    
    # 记录总体开始时间
    total_start_time = time.time()
    
    # 存储测试结果
    results = {}
    
    # 测试每个算法
    for algo in algorithms:
        for dataset in datasets:
            for backbone in backbones:
                key = f"{algo}_{dataset}_{backbone}"
                print(f"\n开始测试: {key}")
                
                success = test_zeroshot_performance(algo, dataset, backbone)
                results[key] = success
                
                # 添加短暂延迟避免资源冲突
                time.sleep(2)
    
    # 记录总体结束时间
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    
    # 打印总结
    print(f"\n{'='*80}")
    print("Zero-Shot 性能测试总结")
    print(f"{'='*80}")
    print(f"总测试时间: {total_elapsed:.2f}秒")
    print(f"成功测试: {sum(results.values())}/{len(results)}")
    print("\n详细结果:")
    
    for key, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {key}: {status}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
