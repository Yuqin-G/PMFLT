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
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from torch.utils.data import Sampler
from dataloader.fed_datasets import *
from dataloader.resisc45 import RESISC45
from dataloader.cifar100 import CIFAR100
from dataloader.cifar10 import CIFAR10
from dataloader.cub200 import CUB200
from dataloader.domain_datasets import PACS, DomainNet, OfficeHomeDG

import numpy as np
from clip.clip import _transform
import random
import torchvision.transforms as T
from tqdm import tqdm
from collections import defaultdict
import random
import math

class DatasetSplit(TorchDataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idx, image, label = self.dataset[self.idxs[item]]
        return idx, image, label

class DatasetWrapper(TorchDataset):

    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        label = item.label
        classname = item.classname
        # print(item.impath)
        
        if isinstance(item.impath, str):
            try:
                img = Image.open(item.impath).convert("RGB")
            except:
                img = item.impath
        elif isinstance(item.impath, (np.ndarray, torch.Tensor)):
            if isinstance(item.impath, torch.Tensor):
                img_array = item.impath.numpy()
            else:
                img_array = item.impath
            img = Image.fromarray(img_array)
        else:
            img = item.impath

        if self.transform is not None:
            img = self.transform(img)

        output = {
            "img": img,
            "label": label,
            "cname":classname
        }
        return output

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts

class ClassSampler(Sampler):
    def __init__(self, data_cnames, classes, k, seed):

        self.classes = classes
        self.data_cnames = data_cnames
        self.k = k
        self.seed = seed

        class_to_indices = self._create_class_to_indices()
        indices = []
        for indices_per_class in class_to_indices.values():
            indices += indices_per_class
        self.indices = indices

    def _create_class_to_indices(self):
        if self.seed >= 0:
            random.seed(self.seed)
        class_to_indices = {c: [] for c in self.classes}
        for c in class_to_indices.keys():
            valid_idx = np.where(np.array(self.data_cnames)==c)[0].tolist()
            try:
                sample_idx = random.sample(valid_idx, self.k)
            except:
                sample_idx = valid_idx
            class_to_indices[c] = sample_idx

        return class_to_indices

    def __iter__(self):
        random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class PrecomputedDirichletSampler(Sampler):
    def __init__(self, data_indices, seed=42):
        self.data_indices = data_indices
        self.seed = seed
        
        if self.seed >= 0:
            random.seed(self.seed)
    
    def __iter__(self):
        random.shuffle(self.data_indices)
        return iter(self.data_indices)
    
    def __len__(self):
        return len(self.data_indices)


def partition_data_dirichlet(y_train, y_test, n_parties, beta=0.5, min_require_size=10, seed=42):
    if len(np.unique(y_train)) <= 10:
        K = 10
    elif len(np.unique(y_train)) <= 100:
        K = 100
    else:
        K = len(np.unique(y_train))
    
    random.seed(seed)

    N_train = y_train.shape[0]
    N_test = y_test.shape[0]
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}
    
    min_size = 0
    min_require_size = min_require_size
    
    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        idx_batch_test = [[] for _ in range(n_parties)]
        
        for k in range(K):
            train_idx_k = np.where(y_train == k)[0]
            test_idx_k = np.where(y_test == k)[0]
            
            np.random.shuffle(train_idx_k)
            np.random.shuffle(test_idx_k)
            
            # 生成Dirichlet分布比例
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            
            # 确保每个客户端的数据量不超过总数据的1/n_parties
            proportions = np.array([p * (len(idx_j) < N_train / n_parties) 
                                  for p, idx_j in zip(proportions, idx_batch_train)])
            proportions = proportions / proportions.sum()
            
            # 计算分割点
            proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
            
            # 分配数据
            idx_batch_train = [idx_j + idx.tolist() 
                             for idx_j, idx in zip(idx_batch_train, 
                                                 np.split(train_idx_k, proportions_train))]
            idx_batch_test = [idx_j + idx.tolist() 
                            for idx_j, idx in zip(idx_batch_test, 
                                                np.split(test_idx_k, proportions_test))]
        
        # 计算最小样本数
        min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
        min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
        min_size = min(min_size_train, min_size_test)
    
    # 打乱每个客户端的数据
    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        net_dataidx_map_test[j] = idx_batch_test[j]
    
    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train)
    test_data_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test)
    
    return net_dataidx_map_train, net_dataidx_map_test, train_data_cls_counts, test_data_cls_counts


def build_data_loader(
        cfg,
        data_source,
        data_cnames,
        classes=None,
        batch_size=64,
        num_shots=None,
        tfm=None,
):
    dataset_wrapper = DatasetWrapper(data_source, transform=tfm)
    # Build sampler
    sampler = None
    if classes is not None:
        sampler = ClassSampler(
            data_cnames,
            classes,
            num_shots,
            cfg.EXPERIMENT.SEED
        )

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    # print(len(data_loader))
    # assert len(data_loader) > 0
    # exit()
    return data_loader


def build_dirichlet_data_loader(
        cfg,
        data_source,
        data_indices,
        batch_size=64,
        tfm=None,
):
    dataset_wrapper = DatasetWrapper(data_source, transform=tfm)
    
    sampler = PrecomputedDirichletSampler(
        data_indices,
        cfg.EXPERIMENT.SEED
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    return data_loader

class TrainDataManager:

    def __init__(
            self,
            cfg,
            dataname,
            available_classes=None,
            global_classes=None,
            dirichlet_initial=False,
            available_train_data=None,
            available_test_data=None,
            source_domains=None,
            target_domains=None,
            num_clients=None,
    ):
        if (cfg.FEDERATED.DATA_MODE == "dirichlet"):
            available_classes = None

        # Load dataset
        if (dataname == "pacs"):
            dataset = PACS(cfg, source_domains, target_domains)
        elif (dataname == "domainnet"):
            dataset = DomainNet(cfg, source_domains, target_domains)
        elif (dataname == "officehome"):
            dataset = OfficeHomeDG(cfg, source_domains, target_domains)

        if dataname == 'caltech101':
            dataset = Caltech101(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'oxford_flowers':
            dataset = OxfordFlowers(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'eurosat':
            dataset = EuroSAT(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'oxford_pets':
            dataset = OxfordPets(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'fgvc_aircraft':
            dataset = FGVCAircraft(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'food101':
            dataset = Food101(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'dtd':
            dataset = DescribableTextures(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'ucf101':
            dataset = UCF101(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'stanford_cars' or dataname == 'stanfordcars':
            dataset = StanfordCars(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'sun397':
            dataset = SUN397(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'cifar100':
            dataset = CIFAR100(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'cifar10':
            dataset = CIFAR10(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'resisc45':
            dataset = RESISC45(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'cub200':
            dataset = CUB200(cfg, available_classes, global_classes, relabel=False)

        elif dataname == 'imagenet':
            dataset = ImageNet(cfg, available_classes, global_classes, relabel=False)

        tfm = _transform(224)

        if (cfg.FEDERATED.DATA_MODE == "dirichlet" and dirichlet_initial == True):
            
            y_train = np.array([item.label for item in dataset.train])
            y_test = np.array([item.label for item in dataset.test])

            net_dataidx_map_train, net_dataidx_map_test, train_data_cls_counts, test_data_cls_counts = partition_data_dirichlet(
                y_train, y_test, 
                n_parties=num_clients, 
                beta=cfg.FEDERATED.ALPHA, 
                min_require_size=10
            )
            self.net_dataidx_map_train = net_dataidx_map_train
            self.net_dataidx_map_test = net_dataidx_map_test
            self.train_data_cls_counts = train_data_cls_counts
            self.test_data_cls_counts = test_data_cls_counts
            return

        if (cfg.FEDERATED.DATA_MODE == "few_shot"):
            train_loader = build_data_loader(
                cfg,
                data_source=dataset.train,
                data_cnames = dataset.train_cnames,
                classes=dataset.classnames,
                batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
                num_shots=cfg.DATASET.NUM_SHOTS,
                tfm=tfm,
            )

            test_loader = build_data_loader(
                cfg,
                data_source=dataset.test,
                data_cnames = dataset.test_cnames,
                batch_size=cfg.DATASET.TEST_BATCH_SIZE,
                tfm=tfm,
            )

            train_labels = [item.label for item in dataset.train]
            test_labels = [item.label for item in dataset.test]

        elif (cfg.FEDERATED.DATA_MODE == "dirichlet"):

            train_loader = build_dirichlet_data_loader(
                cfg,
                data_source=dataset.train,
                data_indices=available_train_data,
                batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
                tfm=tfm,
            )
            
            test_loader = build_dirichlet_data_loader(
                cfg,
                data_source=dataset.test,
                data_indices=available_test_data,
                batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
                tfm=tfm,
            )

        self.train_loader = train_loader
        self.available_classnames = dataset.classnames
        self.all_classnames = dataset.all_classnames
        self.global_classnames = dataset.global_classnames
        self.data_name = dataname
        self.test_loader = test_loader
        self.available_classes = available_classes
        self.dataset = dataset

        
class TestDataManager:

    def __init__(
            self,
            cfg,
            split
    ):
        # dataset_classnum = {'imagenet': 1000, 'caltech101':100, 'oxford_flowers': 102,'eurosat':10, 'oxford_pets':37, 'fgvc_aircraft': 100,
                            # 'food101': 101, 'dtd': 47, 'ucf101':101,'stanford_cars':196,'sun397':397 ,'imagenet-a': 200,'imagenet-s': 200,'imagenet-r': 200,'imagenet-v2': 200}

        dataset_classnum = {'imagenet': 1000, 'caltech101':100, 'oxford_flowers': 102,'eurosat':10, 'oxford_pets':37, 'fgvc_aircraft': 100,
                                    'food101': 101, 'dtd': 47, 'ucf101':101,'stanford_cars':196,'sun397':397, 
                                    'resisc45': 45, 'cifar100': 100, "cifar10": 10, "cub200": 200, "pacs": 7, "officehome": 65, "domainnet": 345}
        # Load dataset
        
        available_datasets = cfg.DATASET.TESTNAME_SPACE
        # split = cfg.DATASET.SPLIT.TEST
        test_loaders, test_datasets, global_classnames, all_classnames = [],[],[],[]
        for dataname in available_datasets:

            if (dataname == "pacs"):
                dataset = PACS(cfg)
            elif (dataname == "domainnet"):
                dataset = DomainNet(cfg)

            all_cls_idx = np.arange(dataset_classnum[dataname])
            m = math.ceil(dataset_classnum[dataname] / 2)
            if split == 'base':
                available_classes = all_cls_idx[:m]
            elif split == 'new':
                available_classes = all_cls_idx[m:]
            else:
                available_classes = None

            if dataname == 'caltech101':
                dataset = Caltech101(cfg, available_classes)
            elif dataname == 'oxford_flowers':
                dataset = OxfordFlowers(cfg, available_classes)
            elif dataname == 'eurosat':
                dataset = EuroSAT(cfg, available_classes)
            elif dataname == 'oxford_pets':
                dataset = OxfordPets(cfg, available_classes)
            elif dataname == 'fgvc_aircraft':
                dataset = FGVCAircraft(cfg, available_classes)
            elif dataname == 'food101':
                dataset = Food101(cfg, available_classes)
            elif dataname == 'dtd':
                dataset = DescribableTextures(cfg, available_classes)
            elif dataname == 'ucf101':
                dataset = UCF101(cfg, available_classes)
            elif dataname == 'stanford_cars':
                dataset = StanfordCars(cfg, available_classes)
            elif dataname == 'sun397':
                dataset = SUN397(cfg, available_classes)
            elif dataname == 'cifar100':
                dataset = CIFAR100(cfg, available_classes)
            elif dataname == 'cifar10':
                dataset = CIFAR10(cfg, available_classes)
            elif dataname == 'resisc45':
                dataset = RESISC45(cfg, available_classes)
            elif dataname == 'cub200':
                dataset = CUB200(cfg, available_classes)
            elif dataname == 'imagenet':
                dataset = ImageNet(cfg, available_classes)
            elif dataname == 'imagenet-a':
                dataset = ImageNetA(cfg)
            elif dataname == 'imagenet-r':
                dataset = ImageNetR(cfg)
            elif dataname == 'imagenet-s':
                dataset = ImageNetSketch(cfg)
            elif dataname == 'imagenet-v2':
                dataset = ImageNetV2(cfg)

            tfm = _transform(224)

            test_loader = build_data_loader(
                cfg,
                data_source=dataset.test,
                data_cnames=dataset.test_cnames,
                batch_size=cfg.DATASET.TEST_BATCH_SIZE,
                tfm=tfm,
            )
            test_loaders.append(test_loader)
            test_datasets.append(dataset)
            global_classnames.append(dataset.classnames)
            all_classnames.append(dataset.all_classnames)

        self.test_loaders = test_loaders
        self.test_datasets = test_datasets
        self.data_name = dataname
        self.all_classnames = all_classnames
        self.global_classnames = global_classnames






