import os
import torch
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from dataloader.utils import Datum, subsample_classes

class CIFAR10:
    def __init__(self, cfg, available_classes=None, global_classes=None, relabel=True):
        self.data_name = "cifar10"
        location = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        train_dataset = PyTorchCIFAR10(root=location+"/cifar10", train=True, download=False)
        test_dataset = PyTorchCIFAR10(root=location+"/cifar10", train=False, download=False)

        train = []
        for idx in range(len(train_dataset)):
            img, label = train_dataset[idx]
            cname = train_dataset.classes[label]
            impath = train_dataset.data[idx] if hasattr(train_dataset, 'data') else None
            train.append(Datum(impath=impath, label=label, classname=cname))

        test = []
        for idx in range(len(test_dataset)):
            img, label = test_dataset[idx]
            cname = test_dataset.classes[label]
            impath = test_dataset.data[idx] if hasattr(test_dataset, 'data') else None
            test.append(Datum(impath=impath, label=label, classname=cname))

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

        self.all_classnames = train_dataset.classes
        self.global_classnames = self.all_classnames
        
        if global_classes is not None:
            indices = global_classes.tolist()
            self.global_classnames = [self.all_classnames[i] for i in indices]
        
        self.train = train
        self.test = test
        self.train_cnames = train_cnames
        self.test_cnames = test_cnames
