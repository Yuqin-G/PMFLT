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

from model.FedCLIP import FedCLIP
from dataloader.dm_federated import TrainDataManager, TestDataManager
from federated.utils import *
import torch.nn.functional as F
from federated.base_trainer import TrainerBase

class Client_FedCLIP(TrainerBase):
    """A local client with frozen clip and FL meta_net and private training data"""
    def __init__(self, cfg, client_id, dataname, available_cls, global_cls=None, available_data=None):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.client_id = client_id

        # self.id = -1
        self.cfg = cfg
        self.build_data_loader(dataname, available_cls, available_data)
        self.build_model(clip_model)
        # self.model.get_tokenized_classnames(classnames=self.available_classes)


    def build_data_loader(self, dataname, available_cls, available_data, global_cls=None):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        if (self.cfg.FEDERATED.DATA_MODE == "few_shot"):
            dm = TrainDataManager(self.cfg, dataname, available_cls)
        elif (self.cfg.FEDERATED.DATA_MODE == "dirichlet"):
            dm = TrainDataManager(self.cfg, dataname, available_cls, dirichlet_initial=False, available_train_data=available_data[0], available_test_data=available_data[1])

        self.train_loader = dm.train_loader
        # self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.available_classes = dm.available_classes
        self.all_classnames = dm.all_classnames
        self.data_name = dm.data_name

    def build_model(self, clip_model):
        cfg = self.cfg

        # classnames = self.dm.dataset.classnames

        # print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE})")
        self.model_name = cfg.MODEL.NAME
        # print("Building custom CLIP")
        # self.model = FedCLIP(cfg, self.available_classes, clip_model, device = self.device)
        self.model = FedCLIP(cfg, device = self.device)

        self.w = cfg.TRAIN.W

        print(f"Client_id: {self.client_id}")
        # print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "img_adap" not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            # print(name)
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated: {enabled}")
        self.model.to(self.device)

        self.optim = build_optimizer(self.model.img_adap, cfg.OPTIM)
        # import torch.optim as optim
        # self.optim = optim.Adam(params=[{'params': self.model.img_adap.parameters()}], lr=1e-2, betas=(
            # 0.9, 0.98), eps=1e-6, weight_decay=1e-5)
        # self.optim = optim.Adam(params=[{'params': self.model.img_adap.parameters()}], lr=1e-3, betas=(
            # 0.9, 0.98), eps=1e-6, weight_decay=1e-5)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("img_adap", self.model.img_adap, self.optim, self.sched)


    def train(self, num_round):
        self.set_model_mode("train")
        losses = MetricMeter()

        # lab2cname= self.dataset.lab2cname
        dataname = self.data_name
        # classnames = self.available_classes
        classnames = self.all_classnames

        for batch in self.train_loader:
            loss, acc = self.forward_backward(batch, dataname, classnames)
            self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }
        losses.update(loss_summary)

        info = []
        info += [f"epoch [{num_round + 1}/{self.max_epoch}]"]
        info += [f"client_id [{self.client_id}]"]
        info += [f"{dataname}"]
        info += [f"{losses}"]
        info += [f"lr {self.get_current_lr():.4e}"]
        print(" ".join(info))

        self.update_lr()
        local_updates = self.model.img_adap.state_dict()
        return local_updates


    def load_meta(self, global_net):
        self.model.img_adap.load_state_dict(global_net)


    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def get_similarity(self, image_features, text_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity

    def forward_backward(self, batch, dataname, classnames):
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        image, label, cname = self.parse_batch(batch)
        # logits_per_image, logits_per_text, image_features, text_features = self.model(image, classnames)
        logits_per_image, logits_per_text, image_features, text_features = self.model(image, cname)
        
        ground_truth = torch.arange(
            len(image), dtype=torch.long, device=self.device)

        
        
        loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2

        # text_features = self.model.text_features

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # return loss, compute_accuracy(pred, label)[0].item()
        return loss, compute_accuracy(similarity, label)[0].item()



    def get_current_lr(self, names=None):
        # current_lr = self.sched.get_last_lr()
        # return current_lr[0]
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]

    def model_inference(self, input, classnames):
        return self.model(input, classnames)


    @torch.no_grad()
    def test(self, evaluator, split):
        """A generic testing pipeline."""
        self.set_model_mode("eval")

        dm = TestDataManager(self.cfg, split)

        data_loaders = dm.test_loaders
        datasets = dm.test_datasets
        acc_list = []
        for i, data_loader in enumerate(data_loaders):
            evaluator.reset()
            print(f"Evaluate on the *{split}* set of {self.cfg.DATASET.TESTNAME_SPACE[i]}")
            classnames = datasets[i].classnames
            dataname = datasets[i].data_name
            # self.model.get_tokenized_classnames(classnames)

            # for batch_idx, batch in enumerate(tqdm(data_loader)):
            for batch_idx, batch in enumerate(data_loader):
                inputs, labels, cnames = self.parse_batch(batch)
                # logits_per_image, logits_per_text, image_features, text_features = self.model_inference(inputs, classnames)
                logits_per_image, logits_per_text, image_features, text_features = self.model_inference(inputs, cnames)
                # text_features = self.model.text_features

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                outputs = similarity

                evaluator.process(outputs, labels)

            results = evaluator.evaluate()

            acc_list.append(list(results.values())[0])
        acc_mean = np.mean(acc_list)
        print(f"accuracy: {acc_mean}")
        return acc_mean

    @torch.no_grad()
    def local_test(self, evaluator):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        acc_dict = {}
        evaluator.reset()

        print(f"Evaluate on the *{self.client_id}th* client of {self.data_name}")
        # classnames = self.available_classes
        classnames = self.all_classnames
        dataname = self.data_name
        test_loader = self.test_loader

        # for batch_idx, batch in enumerate(tqdm(test_loader)):
        for batch_idx, batch in enumerate(test_loader):
            inputs, labels, cnames = self.parse_batch(batch)
            logits_per_image, logits_per_text, image_features, text_features = self.model_inference(inputs, classnames)
            # text_features = self.model.text_features

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            outputs = similarity

            # print(outputs)
            # print(labels)
            # print(outputs.shape)
            # print(labels.shape)

            evaluator.process(outputs, labels)

        results = evaluator.evaluate()
        acc= list(results.values())[0]

        if dataname not in acc_dict:
            acc_dict[dataname]= [acc]
        else:
            acc_dict[dataname].append(acc)
        acc_list = []
        for key in acc_dict.keys():
            acc_list.append(np.mean(acc_dict[key]))
            print(f"acc of {key}: {np.mean(acc_dict[key])}")
        print(f"local accuracy: {np.mean(acc_list)}")
        return np.mean(acc_list)
