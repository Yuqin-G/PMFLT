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
import os
import sys
import time
import numpy as np

CUSTOM_TEMPLATES = {
    'oxford_pets': 'a photo of a {}, a type of pet.',
    'oxford_flowers': 'a photo of a {}, a type of flower.',
    'fgvc_aircraft': 'a photo of a {}, a type of aircraft.',
    'dtd': '{} texture.',
    'eurosat': 'a centered satellite photo of {}.',
    'stanford_cars': 'a photo of a {}.',
    'food101': 'a photo of {}, a type of food.',
    'sum397': 'a photo of a {}.',
    'caltech101': 'a photo of a {}.',
    'ucf101': 'a photo of a person doing {}.',
    "resisc45": "a photo of a {}.",
    "cub200": "a photo of a {}.",
    "cifar100": "a photo of a {}.",
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}


def analysis_results(cfg, results, mode=0):
    print("\n" + "="*60)
    print("📈 Analysis Results...".center(60))
    print("="*60 + "\n")
    # personalization & generalization (used in several other papers)
    if mode == 0:
        local_all = {"local_acc": [], f"{cfg.DATASET.SPLIT.TRAIN}_acc": [], "new_acc": [], "hmean": []}
        global_all = {"local_acc": [], f"{cfg.DATASET.SPLIT.TRAIN}_acc": [], "new_acc": [], "hmean": []}
        for idx, result in enumerate(results):
            result_local = result[0]
            result_global = result[1]

            if (result_local["local_acc"] != []):
                local_all["local_acc"].append((np.mean(result_local["local_acc"]).round(2)))

            if (result_local[f"{cfg.DATASET.SPLIT.TRAIN}_acc"] != []):
                local_all[f"{cfg.DATASET.SPLIT.TRAIN}_acc"].append((np.mean(result_local[f"{cfg.DATASET.SPLIT.TRAIN}_acc"]).round(2)))

            if (result_global["local_acc"] != []):
                global_all["local_acc"].append((np.mean(result_global["local_acc"]).round(2)))

            global_all[f"{cfg.DATASET.SPLIT.TRAIN}_acc"].append((np.mean(result_global[f"{cfg.DATASET.SPLIT.TRAIN}_acc"]).round(2)))

            if (cfg.DATASET.SPLIT.TEST == "base&new"):
                if (result_local["new_acc"] != []):
                    local_all["new_acc"].append(np.mean(result_local["new_acc"]))
                    local_all["hmean"].append(np.mean(result_local["hmean"]))

                global_all["new_acc"].append(np.mean(result_global["new_acc"]))
                global_all["hmean"].append(np.mean(result_global["hmean"]))

        if (local_all["local_acc"] != []):
            print_results("Personalization Performance", local_all)
        print_results("Generalization Performance", global_all)

        return local_all, global_all


def print_results(title, results_dict):
    print(title)
    print("-" * 60)
    if not results_dict:
        print("(No results found)\n")
        return
    for key, value in results_dict.items():
        if len(value) > 0:
            print(f"=> {key}:")
            mean, std = np.mean(value), np.std(value)
            print(f"mean ± std: {mean:.2f}±{std:.2f}")
            print(f"values: {np.array2string(np.array(value), precision=2)}")
            # print("-" * 60)
    print()

    
def get_classes_per_client(train_data_cls_counts):
    classes_per_client = {}
    
    for client_id, class_counts in train_data_cls_counts.items():
        classes = [class_id for class_id, count in class_counts.items() if count > 0]
        classes_per_client[client_id] = sorted(classes)
    
    return classes_per_client

def get_domain_client_num(domain_num, total_client_num):
    # if equal_dis:
    n_clients = total_client_num // domain_num
    not_allocated_num = total_client_num % domain_num

    domain_client_num = np.ones(domain_num) * n_clients
    remain_num = np.random.randint(domain_num, size=not_allocated_num)
    for i in range(len(remain_num)):
        domain_client_num[remain_num[i]] += 1

    return domain_client_num

def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items



def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class Logger:
    """Write console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def setup_logger(output=None):
    if output is None:
        return

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        # fpath = os.path.join(output, "log.txt")
        fpath = os.path.join(output, "log.log")

    if os.path.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S.log")

    sys.stdout = Logger(fpath)