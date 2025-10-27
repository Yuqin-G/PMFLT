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

import clip
import copy
import random
from model.Cbm import CBM
from model.FedTPG import FedTPG
from model.custom_coop import CoOpCLIP
from model.custom_vlp import VLPCLIP
from model.FedPGP import FedPGP
from dataloader.dm_federated import TrainDataManager
from federated.utils import *
import torch.nn.functional as F
from federated.base_trainer import TrainerBase
from torch.utils.data import Dataset, DataLoader
from dataloader.dm_federated import TestDataManager 
from tqdm import tqdm

class Client(TrainerBase):
    """A local client with frozen clip and FL meta_net and private training data"""
    def __init__(self, cfg, client_id, dataname, available_cls, global_cls=None, available_data=None):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.max_epoch = cfg.OPTIMIZER.MAX_EPOCH
        self.client_id = client_id

        # self.id = -1
        self.cfg = cfg
        self.build_data_loader(dataname, available_cls, available_data, global_cls)
        self.build_model()

        self.gateway = Gateway(self.device).to(self.device)
        self.raw_concepts = []

        if (self.available_classes_indices is None):
            self.available_classes_indices = list(range(self.total_classes))
        # self.ori_protos = []
        # self.ori_covs = []


    def build_data_loader(self, dataname, available_cls, available_data, global_cls=None):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        if (self.cfg.FEDERATED.DATA_MODE == "few_shot") or (self.client_id == -1):
            dm = TrainDataManager(self.cfg, dataname, available_cls)
        elif (self.cfg.FEDERATED.DATA_MODE == "dirichlet"):
            dm = TrainDataManager(self.cfg, dataname, available_cls, dirichlet_initial=False, available_train_data=available_data[0], available_test_data=available_data[1])

        self.train_loader = dm.train_loader
        # self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.available_classes = dm.available_classnames
        self.all_classnames = dm.all_classnames
        # self.data_name = dm.data_name
        self.dm = dm
        self.available_classes_indices = available_cls

        dataset_classnum = {'imagenet': 1000, 'caltech101':100, 'oxford_flowers': 102,'eurosat':10, 'oxford_pets':37, 'fgvc_aircraft': 100,
                                    'food101': 101, 'dtd': 47, 'ucf101':101,'stanford_cars':196,'sun397':397, 'resisc45': 45, 'cifar100': 100, "cifar10": 10, "cub200": 200}

        self.data_name = dataname
        self.total_classes = dataset_classnum[dataname]

    def build_model(self, clip_model):
        cfg = self.cfg

        # print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE})")
        # clip_model = load_clip_to_cpu(cfg)
        self.model_name = cfg.MODEL.NAME
        # print("Building custom CLIP")
        if cfg.MODEL.NAME == 'fedtpg':
            self.model = FedTPG(cfg, clip_model, device = self.device)
        elif cfg.MODEL.NAME == 'coop':
            self.model = CoOpCLIP(cfg, clip_model, device = self.device)
        elif cfg.MODEL.NAME == 'vlp':
            self.model = VLPCLIP(cfg, clip_model, device = self.device)
        elif cfg.MODEL.NAME == 'fedpgp':
            self.model = FedPGP(cfg, clip_model, device = self.device)
        elif cfg.MODEL.NAME == 'cbm':
            # self.model = CBM(cfg, clip_model,device = self.device)
            self.model = CBM(cfg, device = self.device)

        self.w = cfg.MODEL.W

        print(f"Client_id: {self.client_id}")
        # print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated: {enabled}")
        # NOTE: only give prompt_learner to the optimizer

        # params = ([p for p in self.model.prompt_learner.parameters()])
        if not (cfg.MODEL.NAME == 'cbm'):
            self.model.to(self.device)
            self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIMIZER)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIMIZER)
            self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIMIZER)
        # self.sched = build_lr_scheduler(self.optim, cfg.OPTIMIZER)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
    
    
    def train_cbm(self, cluster=True):
        # concept_num = 1000
        # concept_num = 500
        concept_num = 100
        # concept_num = 200

        if (cluster == True):
            attributes_embeddings, class_names = self.cluster(self.available_classes_indices, concept_num) # available classes 是 目录
            self.bottleneck = attributes_embeddings.to(self.device)

        concept_num = self.bottleneck.shape[0]

        # Building pseudo features
        self._compute_relations()
        self._build_feature_set()

        self.pesudo_train_loader = DataLoader(self._feature_trainset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=self.cfg.DATA.NUM_WORKERS, pin_memory=True)

        self.stage=1
        self.Explainer = Explainer(self.bottleneck.shape[-1], concept_num, self.total_classes, self.device)

        self.fc = nn.Linear(512, self.total_classes, bias=True)
        # self.Explainer = Explainer(attributes_embeddings.shape[-1], concept_num, self.total_classes, self.device)
        # self.generate_explainer(attributes_embeddings.shape[-1], concept_num, self.total_classes)
        print(self.Explainer.explainer.weight.shape)
        print(self.Explainer.fc.weight.shape)

        # self.explainer, self.fc = self._train(self.Explainer, self.train_loader, self.test_loader) # 从图像特征到类别的映射，使用马氏距离作为正则化项
        # self.explainer, self.fc = self._train(self.Explainer, self.feature_train_loader, self.test_loader) # 从图像特征到类别的映射，使用马氏距离作为正则化项
        # self.explainer, self.fc = self._train(self.Explainer, self.pesudo_train_loader, self.test_loader) # 从图像特征到类别的映射，使用马氏距离作为正则化项
        self._train(self.Explainer, self.pesudo_train_loader, self.test_loader) # 从图像特征到类别的映射，使用马氏距离作为正则化项
        
        # self.train_fc(self.fc, self.pesudo_train_loader, self.test_loader)


    def train_fc(self, fc, pesudo_train_loader, test_loader=None, num_epochs=30, lr=5e-3):
        """
        用pesudo_train_loader直接训练从feature dim到class_num dim的分类器
        """
        fc = fc.to(self.device)
        fc.train()
        optimizer = torch.optim.Adam(fc.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for batch in pesudo_train_loader:
                features, labels = batch[1], batch[2]
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = fc(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * features.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += features.size(0)

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples * 100
            print(f"epoch:{epoch}, train loss:{avg_loss:.6f}, train accuracy: {avg_acc:.5f}")



    def generate_explainer(self, feature_dim, cpt_num, out_dim, bias=True):
        self.explainer = nn.Linear(feature_dim, cpt_num, bias=bias)
        self.fc = nn.Linear(cpt_num, out_dim, bias=bias).to(self.device)

    
    def cluster(self, concept_cls, num_attributes=None, only_get_name_embeddings=False):
        self.stage = 0
        # Attributes selection
        print("start attribute selection: ", self.data_name)
        print(concept_cls)
        attributes, names, counter = self.dm.get_attributes(self.data_name.lower(), concept_cls)
        

        attribute_embeddings = []
        self.data_name = self.data_name.lower()
        
        if self.data_name == 'cub200': prompt = "A photo of {}, a kind of bird."
        elif self.data_name == 'cifar100': prompt = "A photo of a {}"
        elif self.data_name == "resisc45": prompt = "A photo of {}, a kind of satellite photo."
        elif self.data_name == 'dtd': prompt = "A photo of {}, a kind of texture."
        elif self.data_name == "fgvc_aircraft": prompt = "A photo of {}, a kind of aircraft."
        elif self.data_name == "oxford_flowers": prompt = "A photo of {}, a kind of flower."
        elif self.data_name == "food101": prompt = "A photo of {}, food."
        elif self.data_name == "stanford_cars": prompt = "A photo of {}, car."
        elif self.data_name == "ucf101": prompt = "A photo of {}, a kind of action."
        elif self.data_name == 'oxford_pets': prompt = "A photo of {}, pet."

        elif self.data_name == 'tinyimagenet': prompt = "A photo of {}."
        elif self.data_name == 'imagenet100': prompt = "A good photo of {}."

        print("====================")
        print(prompt)
        print(self.dm.get_prefix())
        print("====================")

        # extract text features
        for i in range((len(attributes) // self.cfg.DATA.BATCH_SIZE) + 1):
        # Prompting batch by batch
            sub_attributes = attributes[i * self.cfg.DATA.BATCH_SIZE: (i + 1) * self.cfg.DATA.BATCH_SIZE]
            clip_attributes_embeddings = clip.tokenize([self.dm.get_prefix()+ attr for attr in sub_attributes]).to(self.device)
            attribute_embeddings += [embedding.detach().cpu() for embedding in self.model.encode_text(clip_attributes_embeddings)]
        class_name_embeddings = clip.tokenize([ prompt.format(name) for name in names]).to(self.device)


        print(self.data_name)
        print("concept_cls: ", concept_cls)
        # print("self.available_classes_indices: ", self.available_classes_indices)
        
        # # 调试信息：检查训练集原始数据
        # print("=== 训练集原始数据调试 ===")
        # train_labels_raw = []
        # for i, batch in enumerate(self.train_loader):
        #     images, targets, cnames = self.parse_batch(batch)
        #     train_labels_raw.extend(targets.cpu().numpy())
        #     if i >= 5:  # 只检查前几个batch
        #         break
        # print(f"训练集前几个batch的标签: {np.unique(train_labels_raw)}")
        
        train_features, train_labels = self.get_image_embeddings(self.train_loader)
        test_features, test_labels = self.get_image_embeddings(self.test_loader)

        # print(len(train_features))
        # print(len(train_labels))

        # print(len(test_features))
        # print(len(test_labels))

        # 调试信息：打印标签分布
        print(f"训练集标签: {np.unique(train_labels)}")
        print(f"测试集标签: {np.unique(test_labels)}")
        print(f"训练集标签数量: {len(np.unique(train_labels))}")
        print(f"测试集标签数量: {len(np.unique(test_labels))}")

        # exit()

        self.train_features = train_features
        self.train_labels = train_labels

        # print(self.available_classes_indices[0])
        # self.train_labels = train_labels + self.available_classes_indices[0]
        # print(train_labels)

        # train_labels = train_labels + self.available_classes_indices[0]

        unique_labels = np.unique(train_labels)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        train_labels = np.array([label_map[label] for label in train_labels])
        test_labels = np.array([label_map[label] for label in test_labels])

        # print(train_labels)

        if only_get_name_embeddings:
            return class_name_embeddings

        attribute_embeddings = torch.stack(attribute_embeddings).float()
        attribute_embeddings = attribute_embeddings / attribute_embeddings.norm(dim=-1, keepdim=True)

        print ("num_attributes: ", attribute_embeddings.shape[0])
        print("attribute_embeddings.shape: ", attribute_embeddings.shape)

        print("Clustering...")
        mu = torch.mean(attribute_embeddings, dim=0)
        sigma_inv = torch.linalg.inv(torch.cov(attribute_embeddings.T))
        configs = {
            'mu': mu,
            'sigma_inv': sigma_inv,
            'mean_distance': np.mean([self.mahalanobis_distance(embed, mu, sigma_inv).cpu() for embed in attribute_embeddings])
        }

        # out_dim = self.total_classes
        out_dim = len(self.available_classes_indices)

        print(out_dim)
        print(attribute_embeddings.shape)

        linear_model = ['linear', 'bn', 'linear']
        self.gateway.gate = self.gateway.generate_gate(linear_model, input_dim=attribute_embeddings.shape[-1], output_dim=out_dim, num_attributes=attribute_embeddings.shape[0])
        self.gateway = self.gateway.to(self.device)

        # self.now_train_features = train_features
        # self.now_train_labels = train_labels

        # Feature dataloader, construct Learnt Embeddings
        train_score_dataset = FeatureDataset(train_features, train_labels)
        train_loader = DataLoader(train_score_dataset, batch_size=self.cfg.DATA.BATCH_SIZE,drop_last=False,shuffle=True,num_workers=self.cfg.DATA.NUM_WORKERS)
        test_score_dataset = FeatureDataset(test_features, test_labels) 
        test_loader = DataLoader(test_score_dataset, batch_size=self.cfg.DATA.BATCH_SIZE,drop_last=False, shuffle=False,num_workers=self.cfg.DATA.NUM_WORKERS)

        self.feature_train_loader = train_loader

        best_model = self._train(self.gateway, train_loader, test_loader, regularizer='mahalanobis',configs=configs) # 从图像特征到类别的映射，使用马氏距离作为正则化项
        # best_model = self._train(self.gateway, train_loader, test_loader,regularizer='cosine',configs=configs) # 从图像特征到类别的映射，使用马氏距离作为正则化项

        print("best_model: ", best_model)
        # 训练好的线性层权重矩阵就是聚类中心，每一行代表一个类别对应的概念空间中心 

        # print(best_model[0])

        centers = best_model[0].weight.detach().cpu().numpy() 
        # centers = best_model.weight.detach().cpu().numpy() 
        self.gateway.gate = None # reset

        selected_idxes = []
        print("select {} attributes out of {}".format(num_attributes, len(attribute_embeddings)))
        for center in centers:
            center = center / torch.tensor(center).norm().numpy()
            distances = np.sum((attribute_embeddings.numpy() - center.reshape(1, -1)) ** 2, axis=1)
            # sorted_idxes = np.argsort(distances)[::-1]
            sorted_idxes = np.argsort(distances)
            for elem in sorted_idxes:
                if elem not in selected_idxes:
                    selected_idxes.append(elem)
                    break

        # print(selected_idxes)
        selected_idxes = np.array(selected_idxes[:num_attributes])
        for j in selected_idxes: self.raw_concepts.append(attributes[j])

        torch.save(self.raw_concepts, "./save/{}_{}_concepts.pkl".format(self.data_name, self.client_id))
        torch.save(self.available_classes_indices, "./save/{}_{}_available_classes_indices.pkl".format(self.data_name, self.client_id))
        torch.save(self.available_classes, "./save/{}_{}_available_classes.pkl".format(self.data_name, self.client_id))

        return attribute_embeddings[selected_idxes].clone().detach(), class_name_embeddings.clone().detach()


    def _train(self,model,train_loader,test_loader,regularizer=None,configs=None):

        FB_lr_init = 0.001

        if self.stage:
            lr = FB_lr_init
        # else: lr = 0.01
        else: lr = 0.005
        
        self._epoch_num = 30

        if self.stage: 
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=0.0000)
            for name, param in model.named_parameters():
                print(f"参数名: {name}, 是否需要梯度: {param.requires_grad}, 尺寸: {param.size()}")
        else: optimizer = torch.optim.Adam(model.gate.parameters(),lr=lr )
        
        # 查看训练的参数

        scheduler = None
        milestones = [30,40,50]
        if self.stage: 
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        best_model, best_acc = self._train_function(model,train_loader, test_loader, optimizer,scheduler,regularizer,configs)
        return best_model
    
    def _train_function(self, model, train_loader, test_loader, optimizer, scheduler=None, regularizer=None, configs=None):  

        model.train()
        loss_function = torch.nn.CrossEntropyLoss().to(self.device)

        last_best_acc, best_acc = None, 0
        if self.stage == 0: 
            # epochs = 1
            epochs = 30
            # epochs = 100
        # elif self._cur_task: epochs = 60 # ["FB_epoch_inc"]
        else: epochs = 60 # self.args["FB_epoch"]

        random_class_order_list = list(range(self.total_classes))
        random.shuffle(random_class_order_list)

        best_model = model
        for epoch in range(epochs):
            losses,correct,total = 0, 0, 0
            for idx, batch in enumerate(train_loader):
                if self.stage == 0: 
                    inputs, targets = batch[0], batch[1]
                    # print(inputs.shape)
                    # print(targets.shape)
                else: 
                    inputs, targets = batch[1], batch[2]
                    # inputs, targets = batch[0], batch[1]

                inputs = inputs.float().to(self.device)
                targets = targets.long().to(self.device)

                if self.stage: # training CBL & Classifier
                    # if self._cur_task > 0 and self.args['sg_num']: 
                    # sg_num = 1
                    sg_num = 0
                    if sg_num: 
                        sg_inputs, sg_targets = self._sample_gussian(idx, random_class_order_list, sg_num) # sg_num=1
                        inputs = torch.cat([inputs, sg_inputs], dim=0)
                        targets = torch.cat([targets, sg_targets], dim=0)
                    else: sg_inputs = None

                    # logits, CSV = model.forward_explainer(inputs)
                    logits, CSV = model.forward(inputs)
                    
                    loss = loss_function(logits, targets) # ce-loss
                    sim = 1
                    # sim = 0.5
                    # if self.args['sim']: 
                    loss += sim * self._similiarity_loss(inputs, self.bottleneck, CSV)
                    loss += self._sparse_linear_loss(self.Explainer.fc)
                    
                else: # Learning to search concepts
                    # targets -= offset
                    if (inputs.shape[0] == 1):
                        continue
                    logits = model(inputs)
                    loss = loss_function(logits,targets)
                    loss += self._compute_loss(regularizer,configs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _,preds = torch.max(logits,dim=1)
                correct += preds.eq(targets.expand_as(preds)).sum()
                total += len(targets)
            # if self.stage == 0: end = time.time()
            # else: b_end = time.time()
            train_acc = (correct*100 / total)
            if scheduler: scheduler.step()
            if epoch % self.cfg.EXPERIMENT.PRINT_FREQ == 0 or epoch == epochs - 1: 
                # test_accuracy = self._compute_accuracy(model, test_loader, offset) 
                test_accuracy = self._compute_accuracy(model, test_loader) 
                if test_accuracy > best_acc: 
                    best_acc = test_accuracy
                    if self.stage == 0: best_model = copy.deepcopy(model.gate)  
                    # else: best_model = [copy.deepcopy(model.explainer), copy.deepcopy(model.unity)]
                    else: best_model = [copy.deepcopy(model.explainer), copy.deepcopy(model.fc)]
                
                # logging.info('task: %d, epoch:%d, train loss:%.6f,train accuracy: %.5f,test_accuracy:%.5f'  
                            #  % (self._cur_task, epoch, losses/len(train_loader), train_acc, test_accuracy))
                print('epoch:%d, train loss:%.6f,train accuracy: %.5f,test_accuracy:%.5f'  
                             % (epoch, losses/len(train_loader), train_acc, test_accuracy))
                # if last_best_acc is not None and best_acc == last_best_acc and not self.stage:
                    # print("early stop")
                    # break
                # last_best_acc = best_acc
        return best_model, best_acc


    def _build_feature_set(self):
        # vectors_train = []
        # labels_train = []

        # vectors_train = {}
        # labels_train = {}

        print("constructing pseudo features...")
        # for class_idx in range(self._known_classes, self._total_classes):
        # for class_idx in self.available_classes_indices:
            # print([class_idx])
            # print(self.data_name)
            # dm = TrainDataManager(self.cfg, self.data_name, [class_idx])
            # vectors, labels = self.get_image_embeddings(dm.train_loader) # class_idx全部的数据
            # vectors_train[class_idx] = vectors
            # labels_train[class_idx] = [class_idx]*len(vectors)
            # print(vectors.shape)
            # vectors_train.append(vectors) # [[vector1], [vector2], ...], 由于vector仅储存新类，因此我们在后面要对index进行修正
            # labels_train.append([class_idx]*len(vectors))

        # 按类把相应的feature拼成一整个
        vectors_train = [None for _ in range(self.total_classes)]
        labels_train = [None for _ in range(self.total_classes)]


        for class_idx in range(self.total_classes):
            idxs = np.where(self.train_labels == class_idx)[0]
            # print(class_idx)
            # print(self.train_labels)
            # print("idxs: ", idxs)
            if len(idxs) > 0:
                vectors_train[class_idx] = np.concatenate([self.train_features[i][None, ...] for i in idxs], axis=0)
                labels_train[class_idx] = np.array([class_idx] * len(idxs))
            else:
                vectors_train[class_idx] = np.empty((0, self.train_features.shape[1]))
                labels_train[class_idx] = np.empty((0,), dtype=int)

        # print("sep")
        # print(len(self.ori_protos))
        
        for class_idx in range(self.total_classes):
            if class_idx not in self.available_classes_indices:
                new_idx = self._relations[class_idx]
                # print(new_idx)
                if vectors_train[new_idx] is not None and len(vectors_train[new_idx]) > 0:
                    vectors_train[class_idx] = vectors_train[new_idx] - self.ori_protos[new_idx].cpu().numpy() + self.ori_protos[class_idx].cpu().numpy()
                    labels_train[class_idx] = np.array([class_idx] * len(vectors_train[class_idx]))
                    # print("len(vectors_train[class_idx]): ", len(vectors_train[class_idx]))
                else:
                    print(f"Warning: No reference data for class {class_idx} (reference class {new_idx})")
                    vectors_train[class_idx] = np.empty((0, self.train_features.shape[1] if len(self.train_features) > 0 else 512))
                    labels_train[class_idx] = np.empty((0,), dtype=int)

                # vectors_train.append(vectors_train[new_idx] - self.ori_protos[new_idx].cpu().numpy()
                #                      + self.ori_protos[class_idx].cpu().numpy())
                # labels_train.append([class_idx] * len(vectors_train[-1])) # 新添加的vectors, 更新对应的标签
        
        # torch.save(vectors_train, "vectors_train.pkl")
        # torch.save(labels_train, "labels_train.pkl")

        vectors_train = np.concatenate(vectors_train)
        labels_train = np.concatenate(labels_train)

        # torch.save(vectors_train, "vectors_train1.pkl")
        # torch.save(labels_train, "labels_train1.pkl")
        
        print(f"Final dataset size: {len(vectors_train)} samples")
        self._feature_trainset = Pesudo_FeatureDataset(vectors_train, labels_train)


    # 仅用本地数据训练explainer的最后一层
    def personalization(self, epochs=10, lr=1e-3):
        """
        仅用本地数据训练explainer的最后一层（fc层）
        """
        if not hasattr(self, 'Explainer'):
            print("未找到Explainer模型，无法训练。")
            return

        # 只训练最后一层
        for name, param in self.Explainer.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        local_dataset = Pesudo_FeatureDataset(self.train_features, self.train_labels)

        train_loader = torch.utils.data.DataLoader(local_dataset, batch_size=64, shuffle=True)

        device = next(self.Explainer.parameters()).device

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.Explainer.parameters()), lr=lr)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.fc.parameters()), lr=lr)

        # self.Explainer.train()
        self.fc.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for batch in train_loader:
                # print(batch)
                # print(batch.shape)
                x, y = batch[1], batch[2]
                x = x.to(device).float()
                y = y.to(device).long()
                optimizer.zero_grad()
                outputs, _ = self.Explainer.forward(x)
                # outputs = self.fc.forward(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
            avg_loss = total_loss / total if total > 0 else 0
            acc = correct / total if total > 0 else 0
            if epoch % 1 == 0 or epoch == epochs - 1:
                print(f"epoch:{epoch}, train loss:{avg_loss:.6f}, train accuracy: {acc*100:.5f}")

        print("本地explainer最后一层训练完成。")

    def _compute_relations(self):
        # calculate the similarity matrix of old classes to new classes.
        # protos = torch.stack(self.ori_protos[self.available_classes_indices]).float().to(self.device)
        # names_emb = self._network.convnet.encode_text(self.names).float().to(self.device)
        # names_emb = self.class_name_embeddings.float().to(self.device)

        # protos = torch.stack([self.ori_protos[i] for i in self.available_classes_indices]).float().to(self.device)
        protos = torch.stack([self.ori_protos[i] for i in range(self.total_classes)]).float().to(self.device)
        names_emb = self.model.encode_text(self.class_name_embeddings).float().to(self.device)

        protos /= protos.norm(dim=-1, keepdim=True)
        names_emb /= names_emb.norm(dim=-1, keepdim=True)

        other_class_idx = list(set(range(self.total_classes)) - set(self.available_classes_indices))

        # simi_matrix = names_emb[other_class_idx] @ protos.T
        # self._relations = torch.argmax(simi_matrix, dim=1) + self.available_classes_indices[0]
        # simi_matrix = simi_matrix[other_class_idx]

        # print(names_emb.shape)
        # print(protos.shape)

        simi_matrix = names_emb @ protos.T # (total_class, 512) @ (512, total_class) = (total_class, total_class)
        # torch.save(simi_matrix, "simi_matrix0.pkl")

        print("simi_matrix.shape: ", simi_matrix.shape)
        print("self.available_classes_indices: ", self.available_classes_indices)
        print("train_labels: ", len(self.train_labels))

        mask = torch.zeros_like(simi_matrix)
        mask[:, self.available_classes_indices] = 1
        # mask[:, other_class_idx] = 0
        # torch.save(mask, "mask.pkl")
        
        # print("simi_matrix: ", simi_matrix)
        simi_matrix = simi_matrix * mask

        # torch.save(simi_matrix, "simi_matrix.pkl")

        
        # print("simi_matrix: ", simi_matrix.shape) # (total_class, now_class)
        # print("simi_matrix: ", simi_matrix)


        self._relations = torch.argmax(simi_matrix, dim=1)

        # self._relations = self._relations[other_class_idx]
        # print("self._relations: ", self._relations.shape)
        print("self._relations: ", self._relations)


        self._relations = self._relations.cpu().numpy()


    def _compute_loss(self,regularizer,configs):
        if regularizer == 'mahalanobis':
            mahalanobis_loss = (self.mahalanobis_distance(self.gateway.gate[0].weight/self.gateway.gate[0].weight.data.norm(dim=-1, keepdim=True), configs['mu'].to(self.device), configs['sigma_inv'].to(self.device)) 
                                - configs['mean_distance']) / (configs['mean_distance']**5)
            return torch.abs(mahalanobis_loss)
        elif regularizer == 'cosine':
            weight = self.gateway.gate[0].weight/self.gateway.gate[0].weight.data.norm(dim=-1, keepdim=True)
            lambda1 = 0.1 
            return lambda1 * torch.sum((weight - configs['mu'].unsqueeze(0).to(self.device)) ** 2, dim=-1).mean()
        else :
            return 0

    def mahalanobis_distance(self,x,mu,sigma_inv):
        if self.stage: 
            x -= mu
            mahal = torch.sqrt(x @ sigma_inv @ x.T)
        else: 
            x = x - mu.unsqueeze(0)
            mahal = torch.diag(x @ sigma_inv @ x.T).mean()
        return mahal

    
    def _similiarity_loss(self, inputs, bottleneck, csv, sg=None, cpt_targets=None):
        # concept alignment
        # print(inputs.shape) # (batch, 512)

        if len(inputs.shape) > 2: img_feats = self._network.extract_vector(inputs.to(self.device)).float()
        else: img_feats = inputs.float()
        distance_loss = nn.MSELoss()

        if sg is not None: img_feats = torch.cat((img_feats,sg),dim=0).float()

        # print(img_feats.shape) # (batch, 512)
        target_feats = img_feats @ bottleneck.T  # (batch, 512) @ (512, cpt_num) = (batch, cpt_num)

        # print(bottleneck.shape) # (cpt_num, 512)
        # print(target_feats.shape) # (batch, cpt_num)
        # print(csv.shape) # (batch, cpt_num)

        target_feats, csv = target_feats**3, csv**3

        target_feats = target_feats / torch.norm(target_feats, p=2, dim=0, keepdim=True)
        csv = csv / torch.norm(csv, p=2, dim=0, keepdim=True)

        # print(target_feats.shape)
        # print(csv.shape)
        similarities = torch.sum(csv * target_feats, dim=0)
        
        return -similarities.mean() 
                    
    def _sparse_linear_loss(self, unity):
        # weight_mat = torch.cat([unity[i].weight for i in range(len(unity))]).view(self._total_classes,-1)
        weight_mat = self.Explainer.fc.weight.view(self.total_classes, -1)

        # lam = 1e-2
        # lam = 1e-3
        lam = 5e-4
        # lam = 3e-4
        # lam = 1e-4
        # lam = 1e-6
        # lam = 1e-5
        # lam = 0.0001
        # lam = 0.001
        alpha = 0.99
        loss = lam * alpha * weight_mat.norm(p=1) + 0.5 * lam * (1-alpha) * (weight_mat**2).sum()
        # loss+=self.gamma * ((self.classifier.weight-self.original_model_weight)**2).sum()+self.gamma*((self.classifier.bias-self.original_model_bias)**2).sum()    
        return loss

    def _compute_accuracy(self, model, test_loader, offset=None):
        model.eval()
        with torch.no_grad():
            predictions = []
            labels = []
            for idx, batch in enumerate(test_loader):
                results = None
                # print("stage: ", self.stage)
                if self.stage == 0: 
                    inputs, targets = batch[0], batch[1]
                else: 
                    # inputs, targets = batch[1], batch[2]
                    # inputs, targets = batch[0], batch[1]
                    inputs, targets ,cnames = self.parse_batch(batch)
                    targets += self.available_classes_indices[0]

                inputs = inputs.float().to(self.device)
                # print(targets)
                # targets += self.available_classes_indices[0]
                # print(targets)
                if len(inputs.shape) > 2: inputs = self.model.encode_image(inputs).float()
                if self.stage:
                    logits, CSV = model.forward(inputs)
                else:
                    logits = model.gate(inputs)
                pred = torch.argmax(logits, dim=-1)
                predictions.append(pred)
                labels.append(targets)
            predictions = torch.cat(predictions)
            labels = torch.cat(labels).to(self.device)
            # if self.stage == 0: labels -= offset
        acc = (torch.sum(predictions == labels) / len(predictions) * 100)
        return acc

    @torch.no_grad()
    def test(self, evaluator=None, split="test"):
        """A generic testing pipeline."""
        self.set_model_mode("eval")

        dm = TestDataManager(self.cfg, split)

        data_loaders = dm.test_loaders
        datasets = dm.test_datasets
        acc_list = []
        for i, data_loader in enumerate(data_loaders):
            evaluator.reset()
            print(f"Evaluate on the *{split}* set of {self.cfg.DATA.TESTNAME_SPACE[i]}")
            classnames = datasets[i].classnames
            dataname = datasets[i].data_name

            # for batch_idx, batch in enumerate(tqdm(data_loader)):
            for batch_idx, batch in enumerate(data_loader):
                inputs, labels, cnames = self.parse_batch(batch)

                inputs = inputs.float().to(self.device)
                if len(inputs.shape) > 2: inputs = self.model.encode_image(inputs).float()
                logits, CSV = self.Explainer.forward(inputs)
                # logits = self.fc(inputs)
                # print("test logits.shape: ", logits.shape)
                evaluator.process(logits, labels)

            results = evaluator.evaluate()

            acc_list.append(list(results.values())[0])
        acc_mean = np.mean(acc_list)
        print(f"avg accuracy: {acc_mean}")
        return acc_mean

    @torch.no_grad()
    def local_test(self, evaluator):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        acc_dict = {}

        evaluator.reset()
        print(f"Evaluate on the *{self.client_id}th* client of {self.data_name}")
        classnames = self.available_classes
        dataname = self.data_name
        test_loader = self.test_loader

        # for batch_idx, batch in enumerate(tqdm(test_loader)):
        for batch_idx, batch in enumerate(test_loader):
            inputs, labels, cnames = self.parse_batch(batch)
            inputs = inputs.float().to(self.device)

            if len(inputs.shape) > 2: inputs = self.model.encode_image(inputs).float()
            # labels += self.available_classes_indices[0]
            logits, CSV = self.Explainer.forward(inputs)
            # logits = self.fc(inputs)
            # print("local test logits.shape: ", logits.shape)

            # idx = self.available_classes_indices
            # logits = logits[:, idx[0]:idx[-1]+1]

            evaluator.process(logits, labels)

        results = evaluator.evaluate()
        acc= list(results.values())[0]

        if dataname not in acc_dict:
            acc_dict[dataname]= [acc]
        else:
            acc_dict[dataname].append(acc)
        acc_list = []
        for key in acc_dict.keys():
            acc_list.append(np.mean(acc_dict[key]))
            print(f"avg acc of {key}: {np.mean(acc_dict[key])}")
        print(f"avg local accuracy: {np.mean(acc_list)}")
        return np.mean(acc_list)

    def get_image_embeddings(self, loader):

        with torch.no_grad():
            features = []
            labels = []
            # for i, (_, images,targets) in enumerate(loader):
            for i, batch in enumerate(loader):
                images, targets, cnames = self.parse_batch(batch)
                # print(images)
                # print(targets)
                # print(cnames)
                images, targets = images.to(self.device), targets.to(self.device)
                # images: [batch_size, 3, 224, 224]
                # image_features = self._network.extract_vector(images)
                image_features = self.model.encode_image(images)
                # if self.stage: image_features = self._network.unifier(image_features.float())
                # [batch_size, 768]
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # [batch_size, 768]
                features.append(image_features.cpu())
                labels.append(targets.cpu())
            features = torch.cat(features)
            labels = torch.cat(labels)
        features = np.array(features)
        labels = np.array(labels)

        return features,labels

    def building_protos(self):
        # prototype construction
        ori_protos, ori_covs = [], []
        print("building protos...")
        with torch.no_grad():
            train_features,train_labels = self.get_image_embeddings(self.train_loader)
            train_features,train_labels = torch.from_numpy(train_features).float(),torch.from_numpy(train_labels).float()
            # for i in range(self._known_classes, self._total_classes):
            for i in self.available_classes_indices:
                index = torch.nonzero(train_labels == i)
                index = index.squeeze()
                class_data = train_features[index]
                cls_mean = class_data.mean(dim=0).to(self.device)
                # cls_cov = torch.cov(class_data.t()).to(self.device) + 1e-4* torch.eye(class_data.shape[-1], device=self.device)

                ori_protos.append(cls_mean)
                # ori_covs.append(cls_cov)
        
        # return ori_protos, ori_covs
        return ori_protos, None


    def train(self,num_round):
        self.set_model_mode("train")
        losses = MetricMeter()

        # lab2cname= self.dataset.lab2cname
        dataname = self.data_name
        # classnames = self.available_classes
        classnames = self.global_classnames
        # print(classnames)
        # batch = next(iter(self.train_loader))
        for batch in self.train_loader:
            loss,acc = self.forward_backward(batch, dataname, classnames)
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
        local_updates = self.model.prompt_learner.state_dict()
        return local_updates

    def load_meta(self, global_net):
        self.model.prompt_learner.load_state_dict(global_net)


    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def forward_backward(self, batch, dataname,classnames):
        images, labels, cnames = self.parse_batch(batch)

        # print(images.shape)

        output, score = self.model(images,classnames, dataname)
        # print(output.shape)
        loss = F.cross_entropy(output, labels) + self.w*score
        return loss,compute_accuracy(output, labels)[0].item()

    def parse_batch(self, batch):
        input = batch["img"]
        label = batch["label"]
        cname = batch["cname"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, cname


    def get_current_lr(self, names=None):
        # current_lr = self.sched.get_last_lr()
        # return current_lr[0]
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]
    def model_inference(self, input, classnames, dataname):
        # return self.model(input,classnames, dataname)
        return self.model(input, classnames, dataname)[0]

class FeatureDataset(Dataset):
    # contains features of images
    def __init__(self, features, targets, group_array=None):
        self.features = torch.tensor(features)
        self.targets = torch.tensor(targets)
        self.group_array = group_array

    def __getitem__(self, idx):
        if self.group_array is not None:
            return self.features[idx], self.targets[idx], self.group_array[idx]
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)

class Explainer(nn.Module):

    def __init__(self, feature_dim, cpt_num, out_dim, device, bias=True):
        super(Explainer, self).__init__()
        self.device = device
        self.explainer = nn.Linear(feature_dim, cpt_num, bias=bias).to(self.device)
        self.fc = nn.Linear(cpt_num, out_dim, bias=bias).to(self.device)
        
    def forward(self, x):
        # if len(x.shape) > 2: x = self.extract_vector(x).float()
        # if len(x.shape) > 2: x = self.model.encode_image(x).float()
        # if sg is not None: x = torch.cat((x,sg),dim=0).float()

        # print(x.shape)
        x = x / x.norm(dim=1, keepdim=True)

        csv = self.explainer(x)

        mean = torch.mean(csv, dim=0, keepdim=True)
        std = torch.std(csv, dim=0, keepdim=True)
        
        norm_csv = csv - mean
        norm_csv /= std

        logits = self.fc(csv)
        
        return logits, csv
        # logits = None
        # for fc in self.unity:
            # logits = fc(csv) if logits is None else torch.concat((logits,fc(csv)),dim=1)



# class Gateway(BaseNet):
class Gateway(nn.Module):
    # input_dim = attribute_embeddings.shape[-1]
    # output_dim = _total_classes
    def __init__(self, device, pretrained=False):
        super(Gateway, self).__init__()
        # super().__init__(args, pretrained)
        # super().__init__(device, pretrained)
        # self.args = args
        self.device = device
        self.convnet = None
        self.gate = None
        # self.heads = None
        self.heads = nn.ModuleList()
        
    def forward(self, x):
        results = self.gate(x)
        return results
    
    def update_gateway(self,type,output_dim,input_dim=None,num_attributes=None):
        if self.gate is None:
            self.gate = self.generate_gate(type,input_dim,output_dim,num_attributes)
            self.gate = self.gate.to(self.device)
        else:
           self.gate = self.expand(self.gate,output_dim).to(self.device)

    def expand(self,last,out_dim):
        nb_output = last.out_features
        nb_input = last.in_features
    
        new = nn.Linear(nb_input,out_dim,bias=True if last.bias is not None else False)
        new.weight.data[:nb_output] = copy.deepcopy(last.weight.data)
        if last.bias is not None:
            new.bias.data[:nb_output] = copy.deepcopy(last.bias.data)
        return new
        
    def addi(self,out_dim,attributes_embeddings=None):
        model = self.gate
        if self.heads == None: self.heads = model.to(self.device)
        else:
            nb_output = self.heads.out_features
            # adding trained_fc
            self.heads = self.expand(self.heads,out_dim)
            # reinit way
            self.heads.weight.data[nb_output:] = copy.deepcopy(model.weight.data)
            if model.bias is not None:
                self.heads.bias.data[nb_output:] = copy.deepcopy(model.bias.data)
            
            self.heads = self.heads.to(self.device)
        
    def addi_heads(self):
        self.heads.append(self.gate)
    
    def generate_gate(self,mode,input_dim,output_dim,num_attributes=None):
        if mode == ['linear', 'bn', 'linear']:
            fc = nn.Sequential(
                nn.Linear(input_dim, num_attributes, bias=False),
                nn.BatchNorm1d(num_attributes),
                nn.Linear(num_attributes, output_dim)
            )
        elif mode == ['bn', 'linear']:
            
            fc = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, output_dim, bias=False)
            )
            # if self.mode == "multi": self.heads.append(fc)
        elif mode == ['linear', 'linear']:
            fc = nn.Sequential(
                nn.Linear(input_dim, num_attributes, bias=False),
                nn.Linear(num_attributes, output_dim)
            )
        elif mode == ['linear']:
            fc = nn.Sequential(nn.Linear(num_attributes, output_dim, bias=False))
        else:
            raise NotImplementedError
        return fc

from PIL import Image
class Pesudo_FeatureDataset(Dataset):
    def __init__(self, features, labels, raw=None, use_path=False, trsf=None):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.raw = raw
        self.use_path = use_path
        self.trsf = trsf
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.raw is not None:
            if self.use_path:
                raw = Image.open(self.raw[idx]).convert("RGB")
                raw = self.trsf(raw)
                # image = self.trsf(pil_loader(self.images[idx]))
            else:
                raw = self.trsf(Image.fromarray(self.raw[idx]))
        
            return idx, feature, label, raw
        else: return idx, feature, label
