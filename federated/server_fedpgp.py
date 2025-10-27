""" FedPGP服务器实现
"""
import copy
import numpy as np
from federated.server_base import ServerBase
from model.FedPGP import FedPGP, load_clip_to_cpu
from federated.utils import *
from federated.client_fedpgp import Client_FedPGP

class Server_FedPGP(ServerBase):

    def create_client_instance(self, cfg, client_id, dataname, available_cls, available_data, dataset_info):
        return Client_FedPGP(cfg, client_id, dataname, available_cls, available_data=available_data, dataset_info=dataset_info)

    def build_model(self):
        cfg = self.cfg
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE})")
        clip_model = load_clip_to_cpu(cfg)
        self.clip_model = clip_model
        self.model_name = cfg.MODEL.NAME
        print(cfg.DATASET.NAME_SPACE)
        global_classnames = self.global_classnames

        print(global_classnames)
        print(len(global_classnames))
        self.model = FedPGP(cfg, global_classnames, clip_model, device=self.device)

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
            self.available_clients[idx] = True
            return
        if hasattr(self.global_model, 'state_dict'):
            self.clients[idx].download(self.global_model.state_dict())
        else:
            self.clients[idx].download(self.global_model)

    def train(self):
        self.before_train()
        self.global_model = copy.deepcopy(self.model.prompt_learner.sigma)
        
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

                w_local0 = copy.deepcopy(w_local['prompt_learner.sigma'])
                w_local1 = copy.deepcopy(w_local['prompt_learner.U'])
                w_local2 = copy.deepcopy(w_local['prompt_learner.V'])

                num_samples = len(self.clients[idx].train_loader.dataset)
                weight = num_samples / total_samples

                if w_glob is None:
                    w_glob = copy.deepcopy(w_local0) * weight
                else:
                    w_glob += copy.deepcopy(w_local0) * weight

            self.global_model = copy.deepcopy(w_glob)
            self.model.prompt_learner.sigma.data.copy_(self.global_model.data)

            self.after_round(round)


        results_local, results_global = self.evaluate(mode=self.cfg.EXPERIMENT.MODE)
        # results_local, results_global = self.evaluate(mode="full")
        return results_local, results_global
        

