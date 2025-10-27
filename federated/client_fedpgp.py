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

from model.FedPGP import FedPGP
from federated.utils import *
import torch.nn.functional as F
from federated.client_base import ClientBase

class Client_FedPGP(ClientBase):
    """A local client with frozen clip and FL meta_net and private training data"""
    
    def build_model(self):
        """Server distribute the model to client"""
        pass

    def download(self, global_model):
        """Download global model"""
        if isinstance(global_model, dict):
            self.model.prompt_learner.sigma.load_state_dict(global_model)
        else:
            self.model.prompt_learner.sigma.data.copy_(global_model.data)

    def forward_backward(self, batch, dataname, classnames):
        """Forward and backward pass."""
        cos = torch.nn.CosineSimilarity(dim=-1)
        images, labels, cnames = self.parse_batch(batch)

        text_features_0, text_features_sigma, text_features_UV, text_features, output = self.model(images)
        posi = cos(text_features_0, text_features_sigma)
        nega = cos(text_features_sigma, text_features)

        logits = torch.cat((posi.reshape(-1, 1), nega.reshape(-1, 1)), dim=1)
        logits /= self.cfg.TRAINER.FEDPGP.temp
        target = torch.zeros(logits.size(0)).to(self.device).long()
        loss2 = F.cross_entropy(logits, target)
        loss = F.cross_entropy(output, labels)
        loss += self.cfg.TRAINER.FEDPGP.mu*loss2
        self.model_backward_and_update(loss)

        return loss, compute_accuracy(output, labels)[0].item()

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


