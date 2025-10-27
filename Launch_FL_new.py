import os
import torch
import random
import argparse
import numpy as np
import swanlab
from config.factory import create_config
from utils import setup_logger, analysis_results
from federated.server import Server


def print_args(args, cfg):
    # print("***************")
    # print("** Arguments **")
    # print("***************")
    # optkeys = list(args.__dict__.keys())
    # optkeys.sort()
    # for key in optkeys:
    #     print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def setup_cfg(args):
    """Setup configuration using factory system"""
    # Create configuration using factory
    cfg = create_config(
        model_name=args.model_name,
        exp_name=args.exp_name,
        use_yaml=True
    )

    # Update configuration from command line arguments
    from config.factory import update_config_from_args
    cfg = update_config_from_args(cfg, args)
    
    # Set additional parameters that might not be in args
    cfg.EXPERIMENT.OUTPUT_DIR = args.output_dir if args.output_dir else './output'
    
    cfg.freeze()
    return cfg

def main(args):
    seeds = args.seeds
    cfg = setup_cfg(args)



    if (cfg.FEDERATED.DATA_MODE == "dirichlet"):
        setup_logger(os.path.join(cfg.EXPERIMENT.OUTPUT_DIR, cfg.EXPERIMENT.NAME, cfg.MODEL.NAME, cfg.FEDERATED.DATA_MODE, str(cfg.FEDERATED.NUM_CLIENTS)+"_"+str(cfg.FEDERATED.ALPHA), str(cfg.EXPERIMENT.SEED)))
    else:
        setup_logger(os.path.join(cfg.EXPERIMENT.OUTPUT_DIR, cfg.EXPERIMENT.NAME, cfg.MODEL.NAME, cfg.FEDERATED.DATA_MODE, str(cfg.FEDERATED.NUM_CLIENTS)+"_"+str(cfg.DATASET.NUM_SHOTS), str(cfg.EXPERIMENT.SEED)))

    results = []

    cfg.defrost()

    if args.use_all == 0:

        print("\n" + "="*60)
        print("Base2Novel".center(60))
        print("="*60 + "\n")

        cfg.DATASET.SPLIT.TRAIN = 'base'
        cfg.DATASET.SPLIT.TEST = 'base&new'

    if args.single_target == 1:
        cfg.DATASET.TARGET_DOMAINS = [args.target_domain]
        cfg.DATASET.SOURCE_DOMAINS = [domain for domain in cfg.DATASET.SOURCE_DOMAINS if domain not in cfg.DATASET.TARGET_DOMAINS]
        cfg.DATASET.NUM_DOMAINS = cfg.DATASET.NUM_DOMAINS - 1
        cfg.DATASET.NAME_SPACE = cfg.DATASET.NAME_SPACE * cfg.DATASET.NUM_DOMAINS

        print("\n" + "="*60)
        print("Leave-one-domain-out".center(60))
        print("="*60 + "\n")

        print("=" * 60)
        print("Name Space:", cfg.DATASET.NAME_SPACE)
        print("Source Domains:", cfg.DATASET.SOURCE_DOMAINS)
        print("Target Domains:", cfg.DATASET.TARGET_DOMAINS)
        print("=" * 60)

    cfg.freeze()

    print_args(args, cfg)

    for seed in seeds:
        print("\n" + "="*60)
        print(f"🌱 SEED = {seed}".center(60))
        print("="*60 + "\n")

        if args.use_swanlab:
            experiment_name = f"{cfg.MODEL.NAME}-{cfg.DATASET.NAME_SPACE[0]}-{cfg.FEDERATED.DATA_MODE}-seed{seed}"
            if args.use_all == 0:
                experiment_name = "Base2Novel-" + experiment_name
            if args.single_target == 1:
                experiment_name = "Domain-" + experiment_name
            swanlab.init(
                project="FLVLM",
                experiment_name=experiment_name,
                reinit=True,
                # mode="offline"
            )
            swanlab.log({
                "use_all": args.use_all,
                "single_target": args.single_target,
                "num_clients": cfg.FEDERATED.NUM_CLIENTS,
                "comm_round": cfg.FEDERATED.COMM_ROUND,
                "local_epoch": cfg.FEDERATED.LOCAL_EPOCH,
                "batch_size": cfg.DATASET.BATCH_SIZE,
                "num_shots": cfg.DATASET.NUM_SHOTS,
                "alpha": cfg.FEDERATED.ALPHA,
                "seed": seed
            })

        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        fl_server = Server(cfg)

        if args.eval_only:
            if args.model_name != 'clip':
                fl_server.load_model(args.model_dir, epoch=args.load_epoch)
            if fl_server.cfg.DATASET.SPLIT.TEST == 'base&new':
                fl_server.test("all")
            else:
                fl_server.test(fl_server.cfg.DATASET.SPLIT.TEST)
            return

        import time
        start_time = time.time()

        if not args.no_train:
            result = fl_server.train()

        end_time = time.time()
        total_seconds = end_time - start_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        print(f"Total training time: {hours} hours {minutes} minutes")

        results.append(result)

    analysis_results(cfg, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with Simplified Configuration")
    
    # Basic arguments
    parser.add_argument("--root", type=str, default="/data2/gzh/data", help="path to dataset")
    parser.add_argument("--use_swanlab", action="store_true", default=False, help="use swanlab for logging")
    parser.add_argument("--save_gpu_memory", action="store_true", default=False, help="save GPU memory")
    parser.add_argument("--exp_name", type=str, default="cross_cls", help="experiment name")
    parser.add_argument("--model_name", type=str, default="fedtpg", help="model name")
    parser.add_argument("--num_shots", type=int, default=64, help="number of samples each class")
    parser.add_argument("--use_all", type=int, default=1, help="use all classes for training and testing")
    parser.add_argument("--comm_round", type=int, default=30, help="number of communication rounds")
    parser.add_argument("--local_epoch", type=int, default=1, help="number of local epochs")
    parser.add_argument("--depth_ctx", type=int, default=1, help="depth of ctx")
    parser.add_argument("--model_depth", type=int, default=0, help="number of self-attention modules in prompt net")
    parser.add_argument("--n_ctx", type=int, default=16, help="length of ctx")
    parser.add_argument("--num_epoch", type=int, default=30, help="number of running epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--avail_percent", type=float, default=1.0, help="avail_percent")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory")
    parser.add_argument("--seeds", "--seed", type=int, nargs="+", default=[43], help="list of seeds")
    parser.add_argument("--w", type=float, default=0.0, help="weight of regularization for KgCoOp")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="name of CNN backbone")
    
    # Evaluation arguments
    parser.add_argument("--mode", type=str, default="basic", help="basic, full")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--per-class", action="store_true", help="do not call trainer.train()")
    
    # Federated learning arguments
    parser.add_argument("--data_mode", type=str, default="few_shot", help="few_shot, dirichlet, domain")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha, only used in dirichlet mode")
    parser.add_argument("--num_clients", type=int, default=10, help="num_clients")
    parser.add_argument("--single_target", type=int, default=0, help="single_target")
    parser.add_argument("--target_domain", type=str, default="", help="target_domain")

    args = parser.parse_args()
    
    
    main(args)
