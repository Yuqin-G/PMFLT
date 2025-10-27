""" FedTPG服务器实现
"""
import copy
import numpy as np
from federated.server_base import ServerBase
from federated.client_fedtpg import Client_FedTPG
from model.custom_coop import CoOpCLIP
from model.custom_vlp import VLPCLIP
from model.FedTPG import FedTPG
from federated.utils import *
from model.FedTPG import load_clip_to_cpu

class Server_FedTPG(ServerBase):
    """FedTPG联邦学习服务器"""

    def create_client_instance(self, cfg, client_id, dataname, available_cls, available_data, dataset_info):
        return Client_FedTPG(cfg, client_id, dataname, available_cls, available_data=available_data, dataset_info=dataset_info)

    def build_model(self):

        cfg = self.cfg

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE})")

        clip_model = load_clip_to_cpu(cfg)
        self.clip_model = clip_model
        self.model_name = cfg.MODEL.NAME
        global_classnames = self.global_classnames

        print(cfg.DATASET.NAME_SPACE)
        print(global_classnames)
        print(len(global_classnames))

        if cfg.MODEL.NAME == "coop":
            self.model = CoOpCLIP(cfg, self.global_classnames, clip_model, device=self.device)
        elif cfg.MODEL.NAME == "vlp":
            self.model = VLPCLIP(cfg, self.global_classnames, clip_model, device=self.device)
        else:
            self.model = FedTPG(cfg, self.global_classnames, clip_model, device=self.device)

        self.w = cfg.MODEL.W

        # Turn off gradients for non-prompt_learner parameters
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

    def distribute(self, idx, initial=False):
        if (initial or self.available_clients[idx] == False):
            self.clients[idx].model = copy.deepcopy(self.model)
            self.clients[idx].optim = build_optimizer(self.clients[idx].model.prompt_learner, self.cfg.OPTIM)
            self.clients[idx].sched = build_lr_scheduler(self.clients[idx].optim, self.cfg.OPTIM)
            self.clients[idx].register_model("prompt_learner", self.model.prompt_learner, self.clients[idx].optim, self.clients[idx].sched)
            self.clients[idx].w = self.w
            self.available_clients[idx] = True
            return
        if hasattr(self.global_model, 'state_dict'):
            self.clients[idx].download(self.global_model.state_dict())
        else:
            self.clients[idx].download(self.global_model)

    def train(self):

        self.before_train()
        self.global_model = copy.deepcopy(self.model.prompt_learner)

        for round in range(self.start_round, self.comm_round):
            self.epoch = round
            num_selected = max(int(self.cfg.FEDERATED.AVAIL_PERCENT * self.num_clients), 1)
            idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)
            w_glob = None

            total_samples = sum([len(self.clients[idx].train_loader.dataset) for idx in idxs_users])

            for idx in idxs_users:
                if (round == 0):
                    self.distribute(idx, initial=True)
                else:
                    self.distribute(idx)
                
                for _ in range(self.local_epoch):
                    w_local = self.clients[idx].train(round)

                if (self.cfg.EXPERIMENT.SAVE_GPU_MEMORY):
                    self.clients[idx].model.cpu()

                num_samples = len(self.clients[idx].train_loader.dataset)
                weight = num_samples / total_samples

                if w_glob is None:
                    w_glob = copy.deepcopy(w_local)
                    for k in w_glob.keys():
                        w_glob[k] *= weight
                else:
                    for k in w_glob.keys():
                        w_glob[k] += copy.deepcopy(w_local[k]) * weight
                
            self.global_model.load_state_dict(w_glob, strict=False)
            self.model.prompt_learner.load_state_dict(self.global_model.state_dict())
        
            self.after_round(round)

        # results_local, results_global = self.evaluate(mode=self.cfg.EXPERIMENT.MODE, pfl=self.pfl, gfl=True)
        results_local, results_global = self.evaluate(mode="full", pfl=self.pfl, gfl=True)

        if (not self.pfl and self.cfg.TRAINER.LOCAL_TRAINING):
            # For General FL, after completing training on all clients, perform additional local training to obtain personalized models.
            results_local = self.personalization()

        return results_local, results_global
