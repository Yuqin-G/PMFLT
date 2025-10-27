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

import copy
from model.FedTPG import FedTPG
from model.custom_coop import CoOpCLIP
from model.custom_vlp import VLPCLIP
from federated.utils import *
import torch.nn.functional as F
from federated.client_base import ClientBase

class Client_FedTPG(ClientBase):
    """A local client with frozen clip and FL meta_net and private training data"""
    
    def build_model(self):
        """Server distribute the model to client"""
        pass

    def train(self, num_round):
        """Train the model for one round."""
        self.set_model_mode("train")
        losses = MetricMeter()

        dataname = self.data_name
        classnames = self.global_classnames
        
        for batch in self.train_loader:
            loss, acc = self.forward_backward(batch, dataname, classnames)
            
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }
        losses.update(loss_summary)

        info = []
        info += [f"epoch [{num_round + 1}/{self.comm_round}]"]
        info += [f"client_id [{self.client_id}]"]
        info += [f"{dataname}"]
        info += [f"{losses}"]
        info += [f"lr {self.get_current_lr():.4e}"]
        print(" ".join(info))

        self.update_lr()
        local_updates = self.model.prompt_learner.state_dict()
        return local_updates

    def download(self, global_net):
        """Load global meta network."""
        self.model.prompt_learner.load_state_dict(global_net)


    def forward_backward(self, batch, dataname, classnames):
        """Forward and backward pass."""
        images, labels, cnames = self.parse_batch(batch)

        output, score = self.model(images, classnames, dataname)
        loss = F.cross_entropy(output, labels) + self.w * score
        self.model_backward_and_update(loss)

        return loss,compute_accuracy(output, labels)[0].item()
