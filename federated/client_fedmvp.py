import copy
from model.FedMVP import FedMVP
from federated.utils import *
import torch.nn.functional as F
from federated.client_base import ClientBase

class Client_FedMVP(ClientBase):
    """A local client with frozen clip and FL meta_net and private training data"""
    
    def build_model(self):
        """Server distribute the model to client"""
        pass

    def download(self, global_net):
        """Load global meta network."""
        self.model.prompt_learner.load_state_dict(global_net)

    def train(self, num_round, early_stop_threshold=0.5):
        """Train the model for one round."""
        self.set_model_mode("train")
        losses = MetricMeter()

        dataname = self.data_name
        classnames = self.global_classnames

        # Track if LoRA is activated
        if not self.model.prompt_learner.use_lora:  
            print(f"Training without LoRA fine-tuning for client {self.client_id}")

        # Store previous loss to detect stabilization
        prev_loss = float('inf')

        for batch in self.train_loader:
            loss, acc = self.forward_backward(batch, dataname, classnames)

            # Check if loss is below threshold and LoRA isn't already enabled
            if loss.item() < early_stop_threshold and not self.model.prompt_learner.use_lora:
                print(f"Early stopping condition met! Switching to LoRA fine-tuning at client {self.client_id}")

                # Freeze `crossattn_vis`
                for param in self.model.prompt_learner.crossattn.parameters():
                    param.requires_grad = False  
                # Enable LoRA fine-tuning
                for param in self.model.prompt_learner.cross_loraparams.parameters():
                    param.requires_grad = True              
                self.model.prompt_learner.use_lora = True
                print("use_lora set to True")
            
                lora_params = 0
                for param in [self.model.prompt_learner.cross_loraparams]:
                    if hasattr(param, 'parameters'):
                        lora_params += sum(p.numel() for p in param.parameters() if p.requires_grad)

                print(f"Number of LoRA Trainable Parameters: {lora_params}")

            prev_loss = loss.item()  

        # Log training info
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }
        losses.update(loss_summary)

        info = [
            f"epoch [{num_round + 1}/{self.comm_round}]",
            f"client_id [{self.client_id}]",
            f"{dataname}",
            f"{losses}",
            f"lr {self.get_current_lr():.4e}",
        ]
        print(" ".join(info))

        self.update_lr()
        local_updates = self.model.prompt_learner.state_dict()
        return local_updates

    def forward_backward(self, batch, dataname, classnames):
        """Forward and backward pass."""
        images, labels, cnames = self.parse_batch(batch)
        image_output, vis_score = self.model(images, classnames, dataname)
        loss_ori = F.cross_entropy(image_output, labels)
        loss = loss_ori + self.cfg.TRAINER.FEDMVP.ALPHA*vis_score
        self.model_backward_and_update(loss)
        return loss,compute_accuracy(image_output, labels)[0].item()


