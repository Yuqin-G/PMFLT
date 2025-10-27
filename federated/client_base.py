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

import torch
import torch.nn.functional as F
import os
from dataloader.dm_federated import TrainDataManager, TestDataManager
from federated.utils import *
from federated.base_trainer import TrainerBase


class ClientBase(TrainerBase):
    """Base class for federated learning clients with common functionality"""
    
    def __init__(self, cfg, client_id, dataname, available_cls, available_data=None, dataset_info=None):
        super().__init__()
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Basic attributes
        self.comm_round = cfg.FEDERATED.COMM_ROUND
        self.local_epoch = cfg.FEDERATED.LOCAL_EPOCH

        self.client_id = client_id
        self.cfg = cfg
        
        # Initialize global test data
        self.build_data_loader(dataname, available_cls, available_data, dataset_info)
    
    def build_data_loader(self, dataname, available_cls, available_data, dataset_info):

        # print(dataset_info)

        if self.cfg.FEDERATED.DATA_MODE == "few_shot":
            dm = TrainDataManager(
                self.cfg, dataname, 
                available_classes=available_cls, 
                global_classes=dataset_info['global_classes'],
                num_clients=dataset_info['num_clients'],
                source_domains=dataset_info['source_domains'],
                target_domains=dataset_info['source_domains']
            )
        elif self.cfg.FEDERATED.DATA_MODE == "dirichlet":
            dm = TrainDataManager(
                self.cfg, dataname, 
                available_classes=None, 
                dirichlet_initial=False, 
                available_train_data=available_data[0], 
                available_test_data=available_data[1],
                num_clients=dataset_info['num_clients'],
                source_domains=dataset_info['source_domains'],
                target_domains=dataset_info['source_domains']
            )

        self.dm = dm
        self.global_classnames = dm.available_classnames
        self.available_classes = dm.available_classes
        self.all_classnames = dm.all_classnames
        self.all_classes = list(range(len(self.all_classnames)))
        self.global_classnames = dm.global_classnames
        self.global_classes = dataset_info['global_classes']

        if (self.global_classes is not None) and (dataset_info['mode'] != 'dirichlet'):
            self.new_classes = [idx for idx in self.all_classes if idx not in self.global_classes]
            self.new_classnames = [self.all_classnames[i] for i in self.new_classes]
        
        self.dataset = dm.dataset
        self.train_loader = dm.train_loader
        self.test_loader = dm.test_loader
        self.data_name = dataname
    
    def build_model(self, clip_model):
        """Build the model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build_model method")
    
    def parse_batch(self, batch):
        """Parse a batch of data."""
        input = batch["img"]
        label = batch["label"]
        cname = batch["cname"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, cname
    
    def load_model(self, directory, epoch=None):
        """Load model from directory."""
        if not directory:
            print("Note that load {} failed".format(directory))
            return

        names = self.get_model_names()
        model_name = names[0]

        filepath = os.path.join(directory, f"{model_name}.pth.tar")
        
        if not os.path.isfile(filepath):
            print("Not found: {}".format(filepath))
            return

        print("Loading {} from {}".format(model_name, filepath))
        checkpoint = torch.load(filepath, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]
        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        print("Loading weights to {} "
              "(epoch = {})".format(model_name, epoch))
        self._models[model_name].load_state_dict(state_dict, strict=False)
    
    def save_model(self, epoch, directory, is_best=False):
        """Save model to directory."""
        names = self.get_model_names()
        model_name = names[0]
        model_dict = self._models[model_name].state_dict()

        optim_dict = None
        if self._optims[model_name] is not None:
            optim_dict = self._optims[model_name].state_dict()

        sched_dict = None
        if self._scheds[model_name] is not None:
            sched_dict = self._scheds[model_name].state_dict()

        save_file = {
            "state_dict": model_dict,
            "optimizer": optim_dict,
            "scheduler": sched_dict,
            "epoch": epoch,
        }

        save_path = os.path.join(directory, f"{model_name}.pth.tar")
        torch.save(save_file, save_path)
        print("Model saved to {}".format(save_path))
    
    @torch.no_grad()
    def test(self, evaluator, split="all", use_part="local"):
        """Test the model on the specified split using different parts."""
        self.set_model_mode("test")

        acc_list, correct_list, total_list = [], [], []
        if (split == "local"):
            datasets = [self.dataset]
            data_loaders = [self.test_loader]
        else:
            dm = TestDataManager(self.cfg, split)
            datasets = dm.test_datasets
            data_loaders = dm.test_loaders

        # Support evaluation on multiple test datasets
        for i, data_loader in enumerate(data_loaders):
            evaluator.reset()
            # print(f"Evaluate on the *{split}* set of {self.cfg.DATASET.TESTNAME_SPACE[i]}")

            # classnames = datasets[i].classnames
            classnames = datasets[i].global_classnames
            dataname = datasets[i].data_name

            for batch_idx, batch in enumerate(data_loader):
                inputs, labels, cnames = self.parse_batch(batch)
                
                if (hasattr(self.model, 'require_classnames')):
                    outputs = self.model(inputs, classnames, dataname)
                elif (use_part == "local"):
                    outputs = self.model(inputs)
                elif (use_part == "global"):
                    if (hasattr(self.model, 'forward_global')):
                        outputs = self.model.forward_global(inputs)
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

        if (self.print_info):
            if (use_part == "local"):
                print(f"Client id: {self.client_id} Accuracy of *Personal FL* model on *{split}* test set: {acc_mean:.2f} ({int(correct_mean)}/{int(total_mean)})")
            else:
                print(f"Accuracy of *General FL* model on *{split}* test set: {acc_mean:.2f} ({int(correct_mean)}/{int(total_mean)})")

        if (self.cfg.EXPERIMENT.SAVE_GPU_MEMORY):
            self.model = self.model.cpu()

        return round(acc_mean, 2)

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]