""" pFedMoAP服务器实现
"""
import copy
import random
import numpy as np
from federated.server_base import ServerBase
from model.pFedMoAP import pFedMoAP, load_clip_to_cpu
from federated.utils import *
from federated.client_pfedmoap import Client_pFedMoAP

class Server_pFedMoAP(ServerBase):

    def create_client_instance(self, cfg, client_id, dataname, available_cls, available_data, dataset_info):
        return Client_pFedMoAP(cfg, client_id, dataname, available_cls, available_data=available_data, dataset_info=dataset_info)

    def build_model(self):
        cfg = self.cfg
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE})")
        clip_model = load_clip_to_cpu(cfg)
        self.clip_model = clip_model
        self.model_name = cfg.MODEL.NAME
        global_classnames = self.global_classnames

        print(cfg.DATASET.NAME_SPACE)
        print("Global classnames: ", global_classnames)
        print("Number of global classes: ", len(global_classnames))

        self.model = pFedMoAP(cfg, global_classnames, clip_model, device=self.device)

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
        self.local_prompts = [[] for i in range(self.num_clients)]
        self.local_gatings = [{} for i in range(self.num_clients)]
        self.set_sparse_selection(self.num_clients)
        self.num_experts = min(10, self.num_clients) 

        self.global_model = copy.deepcopy(self.model.prompt_learner.ctx)

        for round in range(self.start_round, self.comm_round):
            self.round = round
            num_selected = max(int(self.cfg.FEDERATED.AVAIL_PERCENT * self.num_clients), 1)
            idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)
            w_glob = None
            
            for idx in idxs_users:
                # ensuring training only the local prompts (no non-local) for the first time each client is selected
                if (round == 0):
                    self.distribute(idx, initial=True)
                else:
                    self.distribute(idx)

                if (round > 0):
                    if self.local_prompts[idx] != []:
                        # local_prompts[idx] != [] -> local_gatings[idx] != {}
                        self.clients[idx].model.load_state_dict(self.local_gatings[idx], strict=False)
                        selected_experts = self.sparse_selection(idx, self.local_prompts, method="nearest")
                        self.clients[idx].download_nonlocal_ctx([self.local_prompts[iii] for iii in selected_experts])

                # train
                for _ in range(self.local_epoch):
                    w_local = self.clients[idx].train(round)
                
                if (self.cfg.EXPERIMENT.SAVE_GPU_MEMORY):
                    self.clients[idx].model.cpu()

                # upload
                if self.local_prompts[idx] != []:
                    self.local_gatings[idx] = {name: copy.deepcopy(w_local[name]) for name in w_local if 'gating' in name}  # gating dict
                self.local_prompts[idx] = copy.deepcopy(w_local['prompt_learner.ctx'])  # prompts
            
            self.reset_distance_cache(update_indices=idxs_users)

            total_samples = sum([len(self.clients[idx].train_loader.dataset) for idx in idxs_users])
            global_prompt = None
            for idx in idxs_users:
                num_samples = len(self.clients[idx].train_loader.dataset)
                local_prompt = self.local_prompts[idx]
                weight = num_samples / total_samples
                if global_prompt is None:
                    global_prompt = copy.deepcopy(local_prompt) * weight
                else:
                    global_prompt += copy.deepcopy(local_prompt) * weight

            self.global_model = copy.deepcopy(global_prompt)
            self.model.prompt_learner.ctx.data.copy_(self.global_model.data)

            self.after_round(round)

        results_local, results_global = self.evaluate(mode=self.cfg.EXPERIMENT.MODE, pfl=self.pfl, gfl=True)

        return results_local, results_global

    def set_sparse_selection(self, num_clients):
            self.num_clients = num_clients
            self.shuffled_all_indices = list(range(self.num_clients))
            random.shuffle(self.shuffled_all_indices)
            self.random_selection_condition = lambda idx, current_idx, ctxs: idx != current_idx and ctxs[idx] != []
            self.reset_distance_cache()

    def reset_distance_cache(self, update_indices=None):
        if update_indices is None:
            self.distance_cache = {i: {j: None for j in range(self.num_clients)} for i in range(self.num_clients)}
        else:
            for idx in update_indices:
                self.distance_cache[idx] = {j: None for j in range(self.num_clients)}
                for i in range(self.num_clients):
                    self.distance_cache[i][idx] = None

    def _get_dist_from_cache(self, idx, x):
        if x in self.distance_cache[idx]:
            return self.distance_cache[idx][x]
        elif idx in self.distance_cache[x]:
            return self.distance_cache[x][idx]
        return None

    def sparse_selection(self, idx, ctxs, method="random"):
        def random_selection(idx, ctxs):
            selected_indices = []
            for x in self.shuffled_all_indices:
                if self.random_selection_condition(x, idx, ctxs):
                    selected_indices.append(x)
                if len(selected_indices) == self.num_experts - 1: # exclude the current client
                    break
            return selected_indices
        
        if method == "random":
            return random_selection(idx, ctxs)
        
        if method == "nearest":
            if ctxs[idx] == []:
                return random_selection(idx, ctxs)
            trained_indices = [i for i in range(len(ctxs)) if ctxs[i] != []]

            # print("num_experts: ", self.num_experts)

            if len(trained_indices) <= self.num_experts:
                return [i for i in trained_indices if i != idx]

            distances = []
            for a_trained_idx in trained_indices:
                if a_trained_idx == idx:
                    continue
                dist = self._get_dist_from_cache(idx, a_trained_idx)
                if dist is None:
                    dist = torch.norm(ctxs[idx] - ctxs[a_trained_idx])
                    self.distance_cache[idx][a_trained_idx] = dist
                    self.distance_cache[a_trained_idx][idx] = dist
                distances.append(dist)
            
            distances = torch.stack(distances)
            print(f"distances: {distances}")
            indices_for_smallest_dist = torch.topk(distances, self.num_experts-1, largest=False)[1]

            # Map the indices in the distances array back to the original client indices
            selected_experts = [trained_indices[i.item()] for i in indices_for_smallest_dist]
            return selected_experts
            return [int(i.item()) for i in indices_for_smallest_dist]

            # raise NotImplementedError(f"Method: {method} has not been implemented yet")
        raise ValueError(f"Unknown sparse selection method for experts: {method}")
