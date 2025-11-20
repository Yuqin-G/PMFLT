# PMFLT: Pretrained-Model based Federated Learning Toolbox

PMFLT is a research-oriented federated learning toolbox designed for experiments with pre-trained models. It provides standardized training pipelines and modular components, enabling easy exploration, comparison, and development of federated learning methods.

## 📖 Introduction

This project implements multiple federated learning approaches for vision-language models, including:

- **PromptFL**: Let Federated Participants Cooperatively Learn Prompts Instead of Models – Federated Learning in Age of Foundation Model (TMC 2023)
- **FedTPG**: Federated Text-driven Prompt Generation for Vision-Language Models (ICLR 2024)
- **FedOTP**: Global and Local Prompts Cooperation via Optimal Transport for Federated Learning (CVPR 2024)
- **FedPGP**: Harmonizing Generalization and Personalization in Federated Prompt Learning (ICML 2024)
- **PromptFolio**: Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method (NeurIPS 2024)
- **pFedMoAP**: Mixture of Experts Made Personalized: Federated Prompt Learning for Vision-Language Models (ICLR 2025)
- **FedPHA**: Federated Prompt Learning for Heterogeneous Client Adaptation (ICML 2025)
- **FedMVP**: Federated Multi-modal Visual Prompt Tuning for Vision-Language Models (ICCV 2025)


## 🏗️ Project Structure

```
FLVLM-main/
├── config/              # Configuration files
│   ├── models/         # Model-specific configs
│   └── factory.py      # Configuration factory
├── federated/          # Federated learning core code
│   ├── client_*.py     # Client implementations
│   ├── server_*.py     # Server implementations
│   └── base_trainer.py # Base trainer
├── model/              # Model definitions
├── dataloader/         # Data loaders
├── Launch_FL_new.py    # Main entry point
└── requirements.txt    # Dependencies
```

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 1.12.1+
- CUDA 10.2+ (recommended)

### Setup

```bash
# Create conda environment
conda create -n flvlm python=3.8
conda activate flvlm

# Install PyTorch
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Data Preparation

Please refer to [CoOP](https://github.com/KaiyangZhou/CoOp/tree/main) for data preparation.

### Training


#### Base2Novel (Few-shot)
```bash
algo="promptfl"
exp_name="dtd"
comm_round=30 shot=16 num_clients=10
CUDA_VISIBLE_DEVICES=0 python -u Launch_FL_new.py --use_all 0 --num_shots $shot --num_clients $num_clients --comm_round $comm_round --exp_name $exp_name --model_name $model_name --seeds 0 1 2 
```

#### Dirichelet
```bash
algo="promptfl"
exp_name="cifar100"
comm_round=50 num_clients=100 avail_percent=0.1
python -u Launch_FL_new.py --print_freq 3 --data_mode "dirichlet" --alpha 0.5 --avail_percent $avail_percent --num_clients $num_clients --comm_round $comm_round --exp_name $exp_name --model_name $algo --seeds 0 1 2
```

#### Domain Generalization
```bash
python Launch_FL_new.py --target_domain photo --single_target 1 --num_clients 6 --exp_name pacs --comm_round 1 --model_name promptfl --seed 0 
```

### Key Arguments

- `--root`: Root directory path to datasets
- `--exp_name`: Experiment name (cross_cls, cross_data, cross_domain)
- `--model_name`: Model name (fedtpg, fedpha, fedpgp, fedotp, fedclip, fedmvp, promptfl, pfedmoap, promptfolio)
- `--num_clients`: Number of clients
- `--comm_round`: Number of communication rounds
- `--local_epoch`: Number of local epochs
- `--num_shots`: Number of samples per class
- `--batch_size`: Batch size
- `--data_mode`: Data partitioning mode (few_shot, dirichlet, domain)
- `--alpha`: Dirichlet distribution parameter (only for dirichlet mode)
- `--seeds`: Random seed list (default [43])
- `--use_swanlab`: Enable SwanLab for experiment tracking

## 🙏 Acknowledgments

This project builds on the following open-source projects:

- [CoOp](https://github.com/KaiyangZhou/CoOp) - Context Optimization
- [CLIP](https://github.com/openai/CLIP) - Contrastive Language-Image Pre-training

## 📧 Contact

For questions or suggestions, please reach out via GitHub Issues.
