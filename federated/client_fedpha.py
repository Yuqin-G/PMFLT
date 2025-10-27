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

from model.FedPHA import FedPHA
from federated.utils import *
from federated.client_base import ClientBase
import torch.nn.functional as F

class Client_FedPHA(ClientBase):
    
    def build_model(self):
        """Server distribute the model to client"""
        pass

    def download(self, global_model):
        """Download global model"""
        self.model.prompt_learner.ctx_global.data.copy_(global_model.data)
        self.model.prompt_learner.ctx_global.requires_grad = True

    def forward_backward(self, batch, dataname, classnames):
        """Forward and backward pass."""

        image, label, _ = self.parse_batch(batch)
        output, global_features, local_features, projected_local_features, output_global= self.model(image)

        pull_loss = F.mse_loss(local_features, projected_local_features)
        alpha = self.model.alpha
        push_loss = F.relu(alpha - torch.norm(local_features - global_features, dim=-1)).mean()
        lambda_pull = 1.0
        lambda_push = 1.0
        loss = F.cross_entropy(output, label)
        loss2 = F.cross_entropy(output_global, label)
        loss += loss2
        loss += lambda_pull * pull_loss + lambda_push * push_loss
        self.model_backward_and_update(loss)

        return loss, compute_accuracy(output, label)[0].item()

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
        local_updates = self.model.state_dict()
        return local_updates


