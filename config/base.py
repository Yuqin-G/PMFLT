"""
Base configuration file
Contains common configurations for all federated learning methods
"""
from yacs.config import CfgNode as CN

def get_base_cfg():
    """Get base configuration"""
    cfg = CN()
    
    # ===================
    # Experiment Configuration
    # ===================
    cfg.EXPERIMENT = CN()
    cfg.EXPERIMENT.NAME = ""
    cfg.EXPERIMENT.OUTPUT_DIR = "./output"
    cfg.EXPERIMENT.RESUME = ""  # Path to resume training
    cfg.EXPERIMENT.SEED = -1  # Negative value for random seed, positive for fixed seed
    
    # Training and testing configuration
    cfg.EXPERIMENT.MODE = "basic"  # Evaluation mode
    # cfg.EXPERIMENT.MODE = "full"  # Evaluation mode
    cfg.EXPERIMENT.PRINT_FREQ = 1  # Frequency to print evaluation info (batch)
    cfg.EXPERIMENT.CHECKPOINT_FREQ = 0  # Frequency to save model (epoch)
    cfg.EXPERIMENT.PER_CLASS_RESULT = False  # Whether to compute per-class results
    cfg.EXPERIMENT.COMPUTE_CMAT = False  # Whether to compute confusion matrix
    cfg.EXPERIMENT.NO_TEST = False  # Whether to skip testing
    cfg.EXPERIMENT.FINAL_MODEL = "last_step"  # Final model selection method
    cfg.EXPERIMENT.USE_SWANLAB = False  # Whether to use swanlab for logging
    cfg.EXPERIMENT.SAVE_GPU_MEMORY = False  # Whether to save GPU memory
    
    # ===================
    # Data Configuration
    # ===================
    cfg.DATASET = CN()
    cfg.DATASET.ROOT = "/data2/gzh/data"
    cfg.DATASET.DOMAIN = False  # Whether to use domain generalization
    cfg.DATASET.NAME_SPACE = []  # List of training datasets
    cfg.DATASET.TESTNAME_SPACE = []  # List of test datasets
    cfg.DATASET.NUM_SHOTS = 16  # Number of samples per class
    cfg.DATASET.BATCH_SIZE = 128  # Batch size for data loading (legacy)
    cfg.DATASET.TRAIN_BATCH_SIZE = 128  # Batch size for training data loading
    cfg.DATASET.TEST_BATCH_SIZE = 128  # Batch size for test data loading
    cfg.DATASET.NUM_WORKERS = 8  # Number of workers for data loading
    
    # Data split configuration
    cfg.DATASET.SPLIT = CN()
    cfg.DATASET.SPLIT.TRAIN = "base"
    cfg.DATASET.SPLIT.TEST = "base&new"
    cfg.DATASET.NUM_DOMAINS = 0
    cfg.DATASET.SOURCE_DOMAINS = []
    cfg.DATASET.TARGET_DOMAINS = []
    
    
    # ===================
    # Model Configuration
    # ===================
    cfg.MODEL = CN()
    cfg.MODEL.NAME = 'fedpha'  # Model name
    cfg.MODEL.BACKBONE = "ViT-B/16" 

    # Common model parameters
    cfg.MODEL.N_CTX = 16  # Number of context vectors
    cfg.MODEL.D_CTX = 1  # Number of context vector layers
    cfg.MODEL.CTX_INIT = ""  # Initialization words
    cfg.MODEL.DEPTH = 0  # Number of self-attention modules
    cfg.MODEL.W = 8.0  # Knowledge guidance weight
    cfg.MODEL.PFL = False # Personalized FL or Generalized FL
    
    # ===================
    # Optimizer Configuration
    # ===================
    cfg.OPTIM = CN()
    cfg.OPTIM.NAME = "sgd"
    cfg.OPTIM.LR = 0.003
    cfg.OPTIM.WEIGHT_DECAY = 1e-5
    cfg.OPTIM.MOMENTUM = 0.9
    
    # Learning rate scheduler
    cfg.OPTIM.LR_SCHEDULER = "cosine"
    cfg.OPTIM.STEPSIZE = (-1, )  # -1 or 0 means stepsize equals max_epoch
    cfg.OPTIM.GAMMA = 0.1
    cfg.OPTIM.MAX_EPOCH = 1000
    
    # ===================
    # Federated Learning Configuration
    # ===================
    cfg.FEDERATED = CN()
    cfg.FEDERATED.DATA_MODE = "few_shot"  # Data distribution mode: few_shot or dirichlet
    cfg.FEDERATED.ALPHA = 0.5  # Alpha parameter for Dirichlet distribution
    cfg.FEDERATED.NUM_CLIENTS = 10  # Number of clients
    # cfg.FEDERATED.NUM_CLASS_PER_CLIENT = 10  # Number of classes per client
    cfg.FEDERATED.AVAIL_PERCENT = 1.0  # Percentage of clients participating in training
    cfg.FEDERATED.COMM_ROUND = 30  # Number of communication rounds
    cfg.FEDERATED.LOCAL_EPOCH = 1  # Number of local epochs
    
    # ===================
    # Trainer Configuration
    # ===================
    cfg.TRAINER = CN()
    cfg.TRAINER.LOCAL_TRAINING = False # Whether to perform local training after federated training
    # cfg.TRAINER.LOCAL_TRAINING = True # Whether to perform local training after federated training
    # cfg.TRAINER.NUM_LOCAL_EPOCHS = 1  # Number of epochs for local training
    cfg.TRAINER.NUM_LOCAL_EPOCHS = 10  # Number of epochs for local training
    # This will be populated by specific model configurations

    return cfg
