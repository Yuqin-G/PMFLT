"""CUB-200-2011 数据集工具

参考resisc45.py风格重写
"""

import os
import torch
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Optional, Tuple
import pickle

# =========================
# CUB-200-2011 Dataset 定义
# =========================

class CUB200Dataset(Dataset):
    """
    CUB-200-2011 数据集，兼容ImageFolder风格，支持train/test划分。
    加载速度优化：元数据缓存到pickle文件，避免每次都读取和merge csv。
    """
    base_folder = "CUB_200_2011/images"
    url = "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        loader: Callable = default_loader,
        download: bool = True,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        self._load_metadata_fast()

    def _get_cache_path(self):
        """返回元数据缓存文件路径"""
        cache_dir = os.path.join(self.root, "CUB_200_2011")
        os.makedirs(cache_dir, exist_ok=True)
        split = "train" if self.train else "test"
        return os.path.join(cache_dir, f"cub200_{split}_meta.pkl")

    def _load_metadata_fast(self):
        """优先从pickle缓存加载元数据，加速启动"""
        cache_path = self._get_cache_path()
        if os.path.isfile(cache_path):
            # 直接加载缓存
            with open(cache_path, "rb") as f:
                meta = pickle.load(f)
            self.classes = meta["classes"]
            self.imgs = meta["imgs"]
            return

        # 否则正常读取并缓存
        # 读取图片路径
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        # 读取类别名
        self.classes = []
        with open(
            os.path.join(self.root, "CUB_200_2011", "classes.txt"), "r"
        ) as fo:
            for line in fo.readlines():
                label_id, classname = line.split(" ")
                classname = (
                    classname.split(".")[-1].replace("\n", "").replace("_", " ")
                )
                self.classes.append(classname)

        # 读取标签
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        # 读取train/test划分
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        data = data.merge(train_test_split, on="img_id")

        if self.train:
            data = data[data.is_training_img == 1].reset_index(drop=True)
        else:
            data = data[data.is_training_img == 0].reset_index(drop=True)

        # 构造imgs: [(impath, label)]
        self.imgs = []
        for idx, row in data.iterrows():
            impath = os.path.join(self.root, self.base_folder, row.filepath)
            label = int(row.target) - 1  # 标签从0开始
            self.imgs.append((impath, label))

        # 缓存到pickle
        meta = {
            "classes": self.classes,
            "imgs": self.imgs,
        }
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(meta, f)
        except Exception as e:
            print(f"Warning: 保存CUB200元数据缓存失败: {e}")

    def _check_integrity(self):
        # 只检查元数据文件是否存在，极大加快启动速度
        meta_files = [
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            os.path.join(self.root, "CUB_200_2011", "classes.txt"),
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
        ]
        for f in meta_files:
            if not os.path.isfile(f):
                print(f"Missing file: {f}")
                return False
        # 只检查前2张图片是否存在，进一步加快速度
        try:
            images = pd.read_csv(
                os.path.join(self.root, "CUB_200_2011", "images.txt"),
                sep=" ",
                names=["img_id", "filepath"],
            )
            for idx, row in images.head(2).iterrows():
                filepath = os.path.join(self.root, self.base_folder, row.filepath)
                if not os.path.isfile(filepath):
                    print(filepath)
                    return False
        except Exception:
            return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(
            os.path.join(self.root, self.filename), "r:gz"
        ) as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        impath, label = self.imgs[idx]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx


from dataloader.utils import Datum, subsample_classes

class CUB200:
    def __init__(self, cfg, available_classes=None, global_classes=None, relabel=True):
        self.data_name = "cub200"
        location = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))

        # 加载Dataset时已加速
        train_dataset = CUB200Dataset(root=location, train=True)
        test_dataset = CUB200Dataset(root=location, train=False)

        # 构造Datum对象
        train = [
            Datum(
                impath=train_dataset.imgs[idx][0],
                label=label,
                classname=train_dataset.classes[label],
            )
            for idx, (impath, label) in enumerate(train_dataset.imgs)
        ]
        test = [
            Datum(
                impath=test_dataset.imgs[idx][0],
                label=label,
                classname=test_dataset.classes[label],
            )
            for idx, (impath, label) in enumerate(test_dataset.imgs)
        ]

        # 支持选取部分类
        if available_classes is not None:
            output, cnames = subsample_classes(train, [], test, available_classes=available_classes, relabel=relabel)
            train = output[0]
            test = output[2]
            train_cnames = cnames[0]
            test_cnames = cnames[2]
        else:
            output, cnames = subsample_classes(train, [], test, available_classes=[i for i in range(len(train_dataset.classes))], relabel=relabel)
            train = output[0]
            test = output[2]
            train_cnames = cnames[0]
            test_cnames = cnames[2]


        self.classnames = []
        if available_classes is not None:
            for idx, i in enumerate(available_classes):
                cname = train_dataset.classes[i]
                if cname not in self.classnames:
                    self.classnames.append(cname)
        else:
            self.classnames = train_dataset.classes

        # print(self.classnames)

        self.all_classnames = train_dataset.classes
        self.global_classnames = self.all_classnames
        
        if global_classes is not None:
            indices = global_classes.tolist()
            self.global_classnames = [self.all_classnames[i] for i in indices]
        
        self.train = train
        self.test = test
        self.train_cnames = train_cnames
        self.test_cnames = test_cnames
