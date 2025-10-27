""" Federated Text-driven Prompt Generation for Vision-Language Models (ICLR 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import time
import numpy as np
import datetime
from scipy.stats import hmean
from dataloader.dm_federated import TestDataManager, TrainDataManager 
from federated.utils import *
import copy
from utils import get_classes_per_client, get_domain_client_num
from federated.client import Client
from federated.client_fedpgp import Client_FedPGP
from federated.client_fedotp import Client_FedOTP
from federated.client_promptfl import Client_PromptFL
from federated.client_fedclip import Client_FedCLIP
from federated.client_pfedmoap import Client_pFedMoAP
from federated.client_fedtpg import Client_FedTPG
from federated.client_fedpha import Client_FedPHA
from federated.client_promptfolio import Client_PromptFolio

import math
import random
import dill
from federated.base_trainer import TrainerBase

class ServerBase(TrainerBase):
    """Base class for federated learning servers with common functionality"""

    def __init__(self, cfg):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")



        self.start_round = self.round = 0
        self.comm_round = cfg.FEDERATED.COMM_ROUND
        self.local_epoch = cfg.FEDERATED.LOCAL_EPOCH
        self.pfl = cfg.MODEL.PFL

        if (cfg.MODEL.W == 8 and cfg.MODEL.NAME == "coop"):
            self.output_dir = os.path.join(cfg.EXPERIMENT.OUTPUT_DIR, cfg.EXPERIMENT.NAME, "kgcoop", cfg.FEDERATED.DATA_MODE, str(cfg.FEDERATED.NUM_CLIENTS)+"_"+str(cfg.FEDERATED.ALPHA), str(cfg.EXPERIMENT.SEED))
        else:
            self.output_dir = os.path.join(cfg.EXPERIMENT.OUTPUT_DIR, cfg.EXPERIMENT.NAME, cfg.MODEL.NAME, cfg.FEDERATED.DATA_MODE, str(cfg.FEDERATED.NUM_CLIENTS)+"_"+str(cfg.FEDERATED.ALPHA), str(cfg.EXPERIMENT.SEED))

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()

        cfg.defrost()

        self.evaluator = Classification(cfg)

        # # 测试 CLIP 模型的 zero-shot 性能
        # print("=" * 50)
        # print("测试 CLIP 模型的 Zero-Shot 性能")
        # print("=" * 50)
        # # 直接使用原始 CLIP 模型进行 zero-shot 测试
        # self.clip_model_ = self.clip_model.to(self.device)
        # self.clip_model_.eval()
        # # 进行 zero-shot 测试
        # self.test(self.evaluator, "test")
        # print("=" * 50)
        # print("Zero-Shot 测试完成")
        # print("=" * 50)
        # exit()

        self.clients = []
        self.init_server(cfg)
        self.available_clients = [False] * self.num_clients

        self.cfg = cfg


    def build_data_loader(self):
        cfg = self.cfg
        dm = TestDataManager(cfg, cfg.DATASET.SPLIT.TRAIN)
        self.test_datasets = dm.test_datasets
        self.test_loaders = dm.test_loaders
        self.all_classnames = dm.all_classnames[0]
        self.all_classes = list(range(len(self.all_classnames)))
        self.global_classnames = dm.global_classnames[0]
        self.global_classes = list(range(len(self.global_classnames)))
        self.new_classes = [idx for idx in self.all_classes if idx not in self.global_classes]
        self.new_classnames = [self.all_classnames[i] for i in self.new_classes]

        print("--------")
        print("global class:", len(self.global_classnames))
        print("all class:", len(self.all_classnames))
        print("new class:", len(self.new_classnames))
        print("--------")

    def build_model(self):
        pass

    def calculate_dataset_client_info(self, cfg, dataname, source_domains=None, num_clients=None):
        if cfg.FEDERATED.DATA_MODE == "few_shot":
            if cfg.DATASET.SPLIT.TRAIN == 'base':
                m = math.ceil(cfg.DATASET.NUM_CLASS / 2)
            elif cfg.DATASET.SPLIT.TRAIN == 'all':
                m = cfg.DATASET.NUM_CLASS
            else:
                m = cfg.DATASET.NUM_CLASS
                
            global_classes = np.arange(m)
            num_class_per_client = np.around(m / num_clients).astype(int)

            return {
                'mode': 'few_shot',
                'global_classes': global_classes,
                'num_clients': num_clients,
                "source_domains": source_domains,
                'total_classes': m,
                'num_class_per_client': num_class_per_client,
            }

        elif cfg.FEDERATED.DATA_MODE == "dirichlet":

            dm = TrainDataManager(cfg, dataname, dirichlet_initial=True, num_clients=num_clients, source_domains=source_domains, target_domains=source_domains)
            net_dataidx_map_train = dm.net_dataidx_map_train
            net_dataidx_map_test = dm.net_dataidx_map_test
            train_data_cls_counts = dm.train_data_cls_counts
            test_data_cls_counts = dm.test_data_cls_counts
            classes_per_client = get_classes_per_client(train_data_cls_counts)

            return {
                'mode': 'dirichlet',
                'global_classes': cfg.DATASET.NUM_CLASS,
                'num_clients': num_clients,
                'net_dataidx_map_train': net_dataidx_map_train,
                'net_dataidx_map_test': net_dataidx_map_test,
                "source_domains": source_domains,
                'train_data_cls_counts': train_data_cls_counts,
                'test_data_cls_counts': test_data_cls_counts,
                'classes_per_client': classes_per_client,
            }
        else:
            raise ValueError(f"Unsupported data mode: {cfg.FEDERATED.DATA_MODE}")

    def get_available_data_for_client(self, cfg, client_id, last_num_clients, dataset_info):

        if dataset_info['mode'] == 'dirichlet':
            available_cls = dataset_info['classes_per_client'][client_id - last_num_clients]
            available_data = [
                dataset_info['net_dataidx_map_train'][client_id - last_num_clients], 
                dataset_info['net_dataidx_map_test'][client_id - last_num_clients]
            ]
            # if (client_id <= 4):
            if (client_id <= 100):
                print("Client id:", client_id, " available classes:", available_cls)
                print("train_data_cls_counts: ", dataset_info['train_data_cls_counts'][client_id - last_num_clients])
                print("test_data_cls_counts: ", dataset_info['test_data_cls_counts'][client_id - last_num_clients])
            # if (client_id == 4):
                # print("Only show the first 5 clients")


        elif dataset_info['mode'] == 'few_shot':
            if client_id == dataset_info['num_clients'] - 1:
                available_cls = dataset_info['global_classes'][
                    (client_id - last_num_clients) * dataset_info['num_class_per_client']:
                ]
            else:
                available_cls = dataset_info['global_classes'][
                    int((client_id - last_num_clients) * dataset_info['num_class_per_client']):
                    int((client_id - last_num_clients + 1) * dataset_info['num_class_per_client'])
                ]
            print("Client id:", client_id, " available classes:", available_cls)
            available_data = None
        else:
            raise ValueError(f"Unsupported mode: {dataset_info['mode']}")
            
        return available_cls, available_data

    def create_client_instance(self, cfg, client_id, dataname, available_cls, available_data, dataset_info):
        pass

    def init_server(self, cfg):
        available_datasets = cfg.DATASET.NAME_SPACE
        self.num_clients = cfg.FEDERATED.NUM_CLIENTS
        client_id, last_num_clients = 0, 0

        if (cfg.DATASET.DOMAIN == True):
            num_client_per_domain = get_domain_client_num(cfg.DATASET.NUM_DOMAINS, cfg.FEDERATED.NUM_CLIENTS)
            print("Source Domains:", cfg.DATASET.SOURCE_DOMAINS)
            print("Number of clients per domain:", num_client_per_domain)

        for idx, dataname in enumerate(available_datasets):
            if (cfg.DATASET.DOMAIN == True):
                dataset_info = self.calculate_dataset_client_info(cfg, dataname, num_clients=int(num_client_per_domain[idx]), source_domains=[cfg.DATASET.SOURCE_DOMAINS[idx]])
            else:
                dataset_info = self.calculate_dataset_client_info(cfg, dataname, num_clients=cfg.FEDERATED.NUM_CLIENTS)

            if dataset_info['mode'] == 'few_shot':
                print(f"Dataset {dataname}: {dataset_info['total_classes']} classes")
                print(f"Number of clients for this dataset: {dataset_info['num_clients']}")
            
            current_num_clients = last_num_clients + dataset_info['num_clients']

            while client_id < current_num_clients:
                available_cls, available_data = self.get_available_data_for_client(
                    cfg, client_id, last_num_clients, dataset_info
                )
                client = self.create_client_instance(
                    cfg, len(self.clients), dataname, available_cls, available_data, dataset_info
                )
                self.clients.append(client)
                client_id += 1

            # print(client_id, end="")

            last_num_clients = current_num_clients

        print(f'total number of clients:{self.num_clients}')

    def distribute(self, idx, initial=False):
        if (initial or self.available_clients[idx] == False):
            global_weights = copy.deepcopy(self.model.state_dict())
            self.clients[idx].model.load_state_dict(global_weights, strict=False)
            self.available_clients[idx] = True
            return

        if hasattr(self.global_model, 'state_dict'):
            self.clients[idx].download(self.global_model.state_dict())
        else:
            self.clients[idx].download(self.global_model)

    def model_inference(self, input, classnames, dataname):
        # 对于 zero-shot 测试，直接使用 CLIP 模型
        if hasattr(self, 'clip_model_') and self.clip_model_ is not None:
            # 使用原始 CLIP 模型进行 zero-shot 推理
            # 构建文本提示
            import clip
            templates = {
                'dtd': 'a photo of a {}, a type of texture.',
                'oxford_pets': 'a photo of a {}, a type of pet.',
                'oxford_flowers': 'a photo of a {}, a type of flower.',
                'fgvc_aircraft': 'a photo of a {}, a type of aircraft.',
                'eurosat': 'a centered satellite photo of {}.',
                'stanford_cars': 'a photo of a {}.',
                'food101': 'a photo of {}, a type of food.',
                'sun397': 'a photo of a {}.',
                'caltech101': 'a photo of a {}.',
                'ucf101': 'a photo of a person doing {}.',
                'resisc45': 'a photo of a {}.',
                'cub200': 'a photo of a {}.',
                'cifar100': 'a photo of a {}.',
                'cifar10': 'a photo of a {}.',
                'imagenet': 'a photo of a {}.',
            }
            
            template = templates.get(dataname, 'a photo of a {}.')
            
            text_prompts = [template.format(name.replace("_", " ")) for name in classnames]
            text_tokens = clip.tokenize(text_prompts).to(input.device)
            
            with torch.no_grad():
                image_features = self.clip_model_.encode_image(input)
                text_features = self.clip_model_.encode_text(text_tokens)
                
                if len(image_features.shape) == 3:
                    # for fedotp，image_features 的形状是 [sequence_length, batch_size, feature_dim]
                    # we need to take the first position (CLS token) and transpose back to [batch_size, feature_dim]
                    image_features = image_features[0]  # [batch_size, feature_dim]
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                logit_scale = self.clip_model_.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                
            return logits
        else:
            # 使用包装后的模型
            return self.model(input, classnames, dataname)[0]

    def parse_batch(self, batch):
        input = batch["img"]
        label = batch["label"]
        cname = batch["cname"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, cname

    def after_round(self, round):
        if (self.cfg.EXPERIMENT.PRINT_FREQ != 0):
            if ((round + 1) % self.cfg.EXPERIMENT.PRINT_FREQ == 0):
            # if ((round + 1) % 1 == 0):
                results_local, results_global = self.evaluate(mode="basic", pfl=self.pfl, gfl=True, round=round)
                if (self.pfl == True):
                    local_acc = np.mean(results_local['local_acc'])
                else:
                    local_acc = np.mean(results_global['local_acc'])
                global_acc = np.mean(results_global[f'{self.cfg.DATASET.SPLIT.TRAIN}_acc'])
                log_summary = {
                    f"eval/local_acc": local_acc,
                    f"eval/global_acc": global_acc,
                }
                self.log_scalar(log_summary, round + 1)

    def before_train(self):
        directory = self.output_dir
        if self.cfg.EXPERIMENT.RESUME:
            directory = self.cfg.RESUME

        # Initialize SwanLab logging
        if (self.cfg.EXPERIMENT.USE_SWANLAB):
            self.init_writer(self.output_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")
        last_epoch = (self.round + 1) == self.comm_round
        if last_epoch:
            self.save_model(self.round, self.output_dir)
        do_test = not self.cfg.EXPERIMENT.NO_TEST
        if do_test:
            if self.cfg.EXPERIMENT.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            # eval_based on each dataset
            self.local_test()
            if self.cfg.DATASET.SPLIT.TEST=='base&new':
                self.test('all')
            else:
                self.test(self.cfg.DATASET.SPLIT.TEST)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
    
    def personalization(self):
        # For General FL, after completing training on all clients, perform additional local training to obtain personalized models.
        print("\n" + "="*60)
        print("Local training to obtain PFL models...".center(60))
        print("="*60 + "\n")

        self.pfl = True
        for idx in range(len(self.clients)):
            for epoch in range(self.cfg.TRAINER.NUM_LOCAL_EPOCHS):
                self.clients[idx].train(epoch)

        results_local, _ = self.evaluate(mode=self.cfg.EXPERIMENT.MODE, pfl=self.pfl, gfl=False, distribute=False)
        return results_local
    
    @torch.no_grad()
    def test(self, evaluator, split="all"):
        """Test the model on the specified split using the global parts."""
        self.set_model_mode("test")

        acc_list, correct_list, total_list = [], [], []

        dm = TestDataManager(self.cfg, split)
        datasets = dm.test_datasets
        data_loaders = dm.test_loaders

        # Support evaluation on multiple test datasets
        for i, data_loader in enumerate(data_loaders):
            evaluator.reset()
            # print(f"Evaluate on the *{split}* set of {self.cfg.DATASET.TESTNAME_SPACE[i]}")

            classnames = datasets[i].classnames
            dataname = datasets[i].data_name

            for batch_idx, batch in enumerate(data_loader):
                inputs, labels, cnames = self.parse_batch(batch)

                if (hasattr(self.model, 'forward_global')):
                    outputs = self.model.forward_global(inputs)
                elif (hasattr(self.model, 'require_classnames')):
                    outputs = self.model(inputs, classnames, dataname)
                else:
                    outputs = self.model(inputs)
                evaluator.process(outputs, labels)

            results = evaluator.evaluate()
            acc = results["accuracy"]
            corret = results["correct"]
            total = results["total"]
            acc_list.append(acc)
            correct_list.append(corret)
            total_list.append(total)

        acc_mean = np.mean(acc_list)
        correct_mean = np.mean(correct_list)
        total_mean = np.mean(total_list)

        print(f"Accuracy of *General FL* model on *{split}* test set: {acc_mean:.2f} ({int(correct_mean)}/{int(total_mean)})")

        return round(acc_mean, 2)

    def evaluate_pfl(self, mode="basic", distribute=True):
        """
        Personal FL evaluation (use local model to evaluate)
        """
        results_local = {"local_acc": [], "all_acc": [], "base_acc": [], "new_acc": [], "hmean": []}

        for idx in range(len(self.clients)): 
            if (distribute):
                self.distribute(idx)
            results_local["local_acc"].append(self.clients[idx].test(self.evaluator, split="local", use_part="local"))
            if (mode == "full"):
                results_local[f"{self.cfg.DATASET.SPLIT.TRAIN}_acc"].append(self.clients[idx].test(self.evaluator, split=self.cfg.DATASET.SPLIT.TRAIN, use_part="local"))
            
            if (self.cfg.EXPERIMENT.SAVE_GPU_MEMORY):
                self.clients[idx].model.cpu()

        print(f"Average accuracy of *Personal FL* models on *local* test set: {np.mean(results_local['local_acc']):.2f}")

        if (mode == "full"):
            print(f"Average accuracy of *Personal FL* models on *{self.cfg.DATASET.SPLIT.TRAIN}* test set: {np.mean(results_local[f'{self.cfg.DATASET.SPLIT.TRAIN}_acc']):.2f}")

        if (mode=="full" and self.cfg.DATASET.SPLIT.TEST == "base&new"):
            for idx in range(len(self.clients)):

                if (hasattr(self.clients[idx].model.prompt_learner, 'init_embedding')):
                    self.clients[idx].model.prompt_learner.init_embedding(classnames=self.clients[idx].new_classnames, clip_model=self.clip_model)

                results_local["new_acc"].append(self.clients[idx].test(self.evaluator, split="new", use_part="local"))

                if (hasattr(self.clients[idx].model.prompt_learner, 'init_embedding')):
                    self.clients[idx].model.prompt_learner.init_embedding(classnames=self.clients[idx].global_classnames, clip_model=self.clip_model)

                if (self.cfg.EXPERIMENT.SAVE_GPU_MEMORY):
                    self.clients[idx].model.cpu()

            results_local["hmean"] = [round(hmean([a, b]), 2) for a, b in zip(results_local['base_acc'], results_local['new_acc'])]
            print(f"Harmonic mean of *Personal FL* models on *base* and *new* test set: {np.mean(results_local['hmean']):.2f}")

        return results_local

    def evaluate_gfl(self, mode="basic"):
        """
        General FL evaluation (use global model to evaluate)
        """
        results_global = {"local_acc": [], "all_acc": [], "base_acc": [], "new_acc": [], "hmean": []}

        for idx in range(len(self.clients)): 
            self.distribute(idx)
            if mode == "full" or not self.cfg.MODEL.PFL:
                print(f"Client id: {idx} ", end="")
                results_global["local_acc"].append(self.clients[idx].test(self.evaluator, split="local", use_part="global"))

            if (self.cfg.EXPERIMENT.SAVE_GPU_MEMORY):
                self.clients[idx].model.cpu()

        if (mode == "full" or not self.cfg.MODEL.PFL):
            print(f"Average accuracy of *General FL* models on *local* test set: {np.mean(results_global['local_acc']):.2f}")

        results_global[f"{self.cfg.DATASET.SPLIT.TRAIN}_acc"].append(self.test(self.evaluator, split=self.cfg.DATASET.SPLIT.TRAIN))

        if (self.cfg.DATASET.SPLIT.TEST == "base&new"):
            if (hasattr(self.model.prompt_learner, 'init_embedding')):
                self.model.prompt_learner.init_embedding(classnames=self.new_classnames, clip_model=self.clip_model)

            results_global["new_acc"].append(self.test(self.evaluator, split="new"))

            if (hasattr(self.model.prompt_learner, 'init_embedding')):
                self.model.prompt_learner.init_embedding(classnames=self.global_classnames, clip_model=self.clip_model)

            results_global["hmean"] = [round(hmean([a, b]), 2) for a, b in zip(results_global['base_acc'], results_global['new_acc'])]
            print(f"Harmonic mean of *General FL* models on *base* and *new* test set: {np.mean(results_global['hmean']):.2f}")

        return results_global

    def evaluate(self, mode="basic", pfl=True, gfl=True, round=None, distribute=True):
        """
        Basic mode includes the following common evaluations in other papers:
            1. (Personalization) The average accuracy of the *Personal* FL models on the *local* test set
            2. (Generalization) The average accuracy of the *General* FL models on the *all* test set or *base&new* test set

        Full mode includes the following evaluations:
            1. (Personalization) The average accuracy of the *Personal* FL models on the *local* test set
            2. (Personalization) The average accuracy of the *Personal* FL models on the *all* test set or *base&new* test set
            3. (Generalization) The average accuracy of the *Personal* FL models on the *local* test set
            4. (Generalization) The average accuracy of the *General* FL models on the *all* test set or *base&new* test set
        """

        print("\n" + "="*60)
        print("🎯 Start Evaluation...".center(60))
        # print("="*60 + "\n")

        if (round is not None):
            print(f"Evaluate at round {round + 1}")
        print(f"Evaluate in {mode} mode")
        print(f"Evaluate Personal FL: {pfl}")
        print(f"Evaluate General FL: {gfl}")
        print("")
        
        results_local = {"local_acc": [], "all_acc": [], "base_acc": [], "new_acc": [], "hmean": []}
        results_global = {"local_acc": [], "all_acc": [], "base_acc": [], "new_acc": [], "hmean": []}

        if (gfl):
            results_global = self.evaluate_gfl(mode)
        if (pfl):
            results_local = self.evaluate_pfl(mode, distribute)

        print("\n" + "="*60 + "\n")
        # print("🎯 Evaluation Done...".center(60))
        # print("="*60 + "\n")

        return results_local, results_global

        for idx in range(len(self.clients)): 
            self.distribute(idx)
            self.clients[idx].model.training = False
            if (self.cfg.MODEL.PFL):
                results_local["local_acc"].append(self.clients[idx].test(self.evaluator, split="local", use_part="local"))
                if (mode == "full"):
                    results_local[f"{self.cfg.DATASET.SPLIT.TRAIN}_acc"].append(self.clients[idx].test(self.evaluator, split=self.cfg.DATASET.SPLIT.TRAIN, use_part="local"))
                    results_global["local_acc"].append(self.clients[idx].test(self.evaluator, split="local", use_part="global"))
                    
        results_global[f"{self.cfg.DATASET.SPLIT.TRAIN}_acc"].append(self.clients[0].test(self.evaluator, split=self.cfg.DATASET.SPLIT.TRAIN, use_part="global"))

        if (self.cfg.MODEL.PFL):
            print(f"Average accuracy of *Personal FL* models on *local* test set: {np.mean(results_local['local_acc']):.2f}")
            if (mode == "full"):
                print(f"Average accuracy of *Personal FL* models on *{self.cfg.DATASET.SPLIT.TRAIN}* test set: {np.mean(results_local[f'{self.cfg.DATASET.SPLIT.TRAIN}_acc']):.2f}")

        if (mode == "full"):
            print(f"Average accuracy of *General FL* models on *local* test set: {np.mean(results_local['local_acc']):.2f}")

        if (self.cfg.DATASET.SPLIT.TEST == "base&new"):
            print("\n" + "="*60)
            print("Base2Novel: Evaluate on new set".center(60))
            print("="*60 + "\n")

            for idx in range(len(self.clients)):
                self.clients[idx].model.prompt_learner.init_embedding(classnames=self.clients[idx].new_classnames, clip_model=self.clip_model)
                self.clients[idx].model.training = False
                self.clients[idx].model.to(self.device)

                if (self.cfg.MODEL.PFL and mode == "full"):
                        results_local["new_acc"].append(self.clients[idx].test(self.evaluator, split="new", use_part="local"))

            results_global["new_acc"].append(self.clients[0].test(self.evaluator, split="new", use_part="global")) 

            if (mode == "full" and self.cfg.MODEL.PFL):
                print(f"Average accuracy of *Personal FL* models on *new* test set: {np.mean(results_local['new_acc']):.2f}")

            if (self.cfg.MODEL.PFL and mode == "full"):
                    results_local["hmean"] = [round(hmean([a, b]), 2) for a, b in zip(results_local['base_acc'], results_local['new_acc'])]

            results_global["hmean"] = [round(hmean([a, b]), 2) for a, b in zip(results_global['base_acc'], results_global['new_acc'])]

            if (mode == "full" and self.cfg.MODEL.PFL):
                print(f"Harmonic mean of *Personal FL* models on *base* and *new* test set: {np.mean(results_local['hmean']):.2f}")
            print(f"Harmonic mean of *General FL* models on *base* and *new* test set: {np.mean(results_global['hmean']):.2f}")

        return results_local, results_global
