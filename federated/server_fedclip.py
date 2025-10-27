""" FedCLIP服务器实现
"""
import copy
import numpy as np
from federated.server_base import ServerBase
from model.FedCLIP import FedCLIP, load_clip_to_cpu
from federated.utils import *
from federated.client_fedclip import Client_FedCLIP

class Server_FedCLIP(ServerBase):

    def create_client_instance(self, cfg, client_id, dataname, available_cls, available_data, dataset_info):
        return Client_FedCLIP(cfg, client_id, dataname, available_cls, available_data=available_data, dataset_info=dataset_info)

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
        self.model = FedCLIP(cfg, global_classnames, clip_model, device=self.device)

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
        if (initial):
            self.clients[idx].model = copy.deepcopy(self.model)
            self.clients[idx].optim = build_optimizer(self.clients[idx].model.prompt_learner, self.cfg.OPTIM)
            self.clients[idx].sched = build_lr_scheduler(self.clients[idx].optim, self.cfg.OPTIM)
            self.clients[idx].register_model("prompt_learner", self.model.prompt_learner, self.clients[idx].optim, self.clients[idx].sched)
            return
        if hasattr(self.global_model, 'state_dict'):
            self.clients[idx].download(self.global_model.state_dict())
        else:
            self.clients[idx].download(self.global_model)

    def train(self):
        self.before_train()
        self.global_model = copy.deepcopy(self.model.img_adap)

        for epoch in range(self.start_round, self.comm_round):
            self.epoch = epoch
            num_selected = max(int(self.cfg.FEDERATED.AVAIL_PERCENT * self.num_clients), 1)
            idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)

            w_glob = None
            total_samples = sum([len(self.clients[idx].train_loader.dataset) for idx in idxs_users])
            
            for idx in idxs_users:
                if (epoch == 0):
                    self.distribute(idx, initial=True)
                else:
                    self.distribute(idx)
                w_local = self.clients[idx].train(epoch)
                
                # Calculate weight based on sample size
                num_samples = len(self.clients[idx].train_loader.dataset)
                weight = num_samples / total_samples

                if w_glob is None:
                    w_glob = copy.deepcopy(w_local) * weight
                else:
                    for k in w_glob.keys():
                        w_glob[k] += copy.deepcopy(w_local[k]) * weight
                        
            self.global_model.load_state_dict(w_glob, strict=False)

        self.model.img_adap.load_state_dict(self.global_model.state_dict())
        
        local_acc = []
        global_acc = []
        all_acc = []

        for idx in idxs_users: 
            self.distribute(idx)
            self.clients[idx].model.training = False
            local_acc.append(self.clients[idx].local_test(self.evaluator))
            global_acc.append(self.clients[idx].test(self.evaluator, "all"))
        
        print("local_acc", local_acc)
        print("local_acc_mean", np.mean(local_acc))
        print("global_acc", global_acc)
        print("all_acc_mean", np.mean(global_acc))

        return local_acc, global_acc
