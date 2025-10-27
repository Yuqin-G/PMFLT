""" CBM服务器实现
"""
import copy
import numpy as np
import torch
import os
from federated.server_base import ServerBase

def get_prefix(self):
    name = self.data_name.lower()
    # if attribute == 'cbm': return ""
    # print(name)
    if name == 'cifar100': return "A bad photo of an object with "
    elif name == 'cub200': return "The bird has "
    elif name == 'resisc45': return "A satellite photo of an object with"

    elif name == 'imagenet-r': return "A picture of an object with "
    elif name == 'imagenet-a': return "A photo of an object with "

    elif name == "food101": return "A photo of the food with"
    elif name == "oxford_flowers": return "A photo of a flower with"
    elif name == "ucf101": return "A photo of {}, a kind of action."
    elif name == 'fgvc_aircraft': return "A photo of {}, a kind of aircraft."
    elif name == 'dtd': return "A photo of {}, a kind of texture."
    elif name == "stanford_cars": return "A photo of the car with"

    elif name == 'pets': return "A photo of the pet with" # "The pet has/with"
    elif name == 'tinyimagenet': return "A photo of the object with "
    elif name == 'imagenet100': return "A good photo of an object with "
    else:
        raise NotImplementedError

def get_attributes(self, attribute, indice):
    # attribute is delivered by args["attibute"]
    name = None

    print("attribute: ", attribute)

    if attribute == 'random':
        '''
        Generate random attributes
        '''
        import urllib.request
        import random

        word_url = "https://www.mit.edu/~ecprice/wordlist.10000"
        response = urllib.request.urlopen(word_url)
        long_txt = response.read().decode()
        word_list = long_txt.splitlines()

        random_words = []
        for i in range(512):
            words = random.choices(word_list, k=random.randint(1, 5))
            random_words.append(' '.join(words))
        print(len(random_words))

        attributes = random_words
        print("random selection!")
        return attributes

    elif attribute == 'cifar100':
        path = "./data/cifar100/concepts.json"
        name = "./data/cifar100/cifar_label2class.json"
    elif attribute == 'cub200': 
        path = "./data/cub200/cub200_4o_simple_cpts.json"
        name = "./data/cub200/cub200_label2class.json"
    elif attribute == "resisc45":
        path = "./data/resisc45/resisc45_4o_simple_cpts.json"
        name = "./data/resisc45/resisc45_label2class.json"

    elif attribute == 'imagenet-r': path = "./data/Imagenet-R/INR_simple_gpt4o.json"
    elif attribute == 'imagenet-a': path = "./data/Imagenet-A/INA_simple_gpt4o.json"

    elif attribute == "dtd": 
        path = "./data/dtd/dtd_4o_simple_cpts.json"
        name = "./data/dtd/dtd_label2class.json"
    elif attribute == "pets": 
        path = "./data/oxford-iiit-pet/pets_simple_gpt4o.json"
        name = "./data/oxford-iiit-pet/pets_label2class.json"
    elif attribute == "ucf101": 
        path = "./data/ucf101/ucf101_4o_simple_cpts.json"
        name = "./data/ucf101/ucf101_label2class.json"

    elif attribute == "fgvc_aircraft": 
        path = "./data/aircraft/aircraft_4o_simple_cpts.json"
        name = "./data/aircraft/aircraft_label2class.json"
    
    elif attribute == "food101": 
        path = "./data/food/food_4o_simple_cpts.json"
        name = "./data/food/food_label2class.json"

    elif attribute == "oxford_flowers": 
        path = "./data/flower/flower_4o_simple_cpts.json"
        name = "./data/flower/flower_label2class.json"

    elif attribute == "stanford_cars": 
        path = "./data/cars/cars_4o_simple_cpts.json"
        name = "./data/cars/cars_label2class.json"

    elif attribute == "tinyimagenet": 
        path = "./data/tinyimagenet/15_tinyimagenet_simple_gpt4o.json"
        name = "./data/tinyimagenet/Tinyimagenet_label2class.json"
    elif attribute == "imagenet100":
        path = "./data/imagenet100/IN100_4o_simple_cpts.json"
        name = "./data/imagenet100/imagenet100_label2class.json"
    else:
        raise NotImplementedError

    print(name)

    attr, cpt_count = [],[0]
    fo = open(path, "r", encoding="utf-8")
    name = open(name, "r", encoding="utf-8")
    attributes = json.load(fo)
    names = json.load(name)
    print(names)
    name = [names[str(idx)] for idx in indice]
    for idx in indice:
        cpt_count.append(cpt_count[-1] + len(attributes[str(idx)]))
        for item in attributes[str(idx)]: 
            attr.append(item)
    return attr, name, cpt_count

class Server_CBM(ServerBase):
    """CBM联邦学习服务器"""

    def train(self):
        """CBM训练方法 - 默认使用train_cbm"""
        return self.train_cbm()

    def train_cbm(self):
        """CBM训练方法"""
        self.before_train()

        class_name_embeddings_path = "./save/{}_class_name_embeddings_{}.pkl".format(self.cfg.EXPERIMENT.NAME, self.cfg.DATASET.NUM_SHOTS)
        ori_protos_path = "./save/{}_ori_protos_{}.pkl".format(self.cfg.EXPERIMENT.NAME, self.cfg.DATASET.NUM_SHOTS)

        if (self.cfg.MODEL.NAME == 'cbm'):
            if os.path.exists(class_name_embeddings_path):
                print("Loading")
                self.class_name_embeddings = torch.load(class_name_embeddings_path)
                self.ori_protos = torch.load(ori_protos_path)
            else:
                print("Building")
                client_all = Client(self.cfg, -1, self.dataname, None, global_cls=None, available_data=None)
                print(client_all.available_classes_indices)
                self.class_name_embeddings = client_all.cluster(client_all.available_classes_indices, 50, only_get_name_embeddings=True)
                self.ori_protos, self.ori_covs = client_all.building_protos()
                torch.save(self.class_name_embeddings, "./save/{}_class_name_embeddings_{}.pkl".format(self.cfg.EXPERIMENT.NAME, self.cfg.DATASET.NUM_SHOTS))
                torch.save(self.ori_protos, "./save/{}_ori_protos_{}.pkl".format(self.cfg.EXPERIMENT.NAME, self.cfg.DATASET.NUM_SHOTS))
            self.ori_covs = None 

        for epoch in range(1):
            self.epoch = epoch
            num_selected = max(int(self.cfg.FEDERATED.AVAIL_PERCENT * self.num_clients), 1)
            idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)

            local_acc = []
            all_acc = []

            for idx in idxs_users:
                self.clients[idx].ori_protos = copy.deepcopy(self.ori_protos)
                self.clients[idx].ori_covs = copy.deepcopy(self.ori_covs)
                self.clients[idx].class_name_embeddings = copy.deepcopy(self.class_name_embeddings)

                w_local = self.clients[idx].train_cbm()

                local_acc.append(self.clients[idx].local_test(self.evaluator))
                all_acc.append(self.clients[idx].test(self.evaluator, "all"))

        print("local_acc", local_acc)
        print("local_acc_mean", np.mean(local_acc))
        print("all_acc", all_acc)
        print("all_acc_mean", np.mean(all_acc))

        self.after_train()

    def train_cbm1(self):
        """CBM训练方法1（带bottleneck）"""
        self.before_train()
        
        for epoch in range(1):
            self.epoch = epoch
            num_selected = max(int(self.cfg.FEDERATED.AVAIL_PERCENT * self.num_clients), 1)
            idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)

            local_acc = []
            all_acc = []
            new_local_acc = []

            bottleneck = None
            concept_num = 100
            bottleneck_file_path = "./save/bottleneck/{}_shot{}_num{}_bottleneck.pkl".format(self.cfg.EXPERIMENT.NAME, self.cfg.DATASET.NUM_SHOTS, concept_num)

            # 检查bottleneck文件是否存在
            if os.path.exists(bottleneck_file_path):
                print(f"Loading bottleneck from {bottleneck_file_path}")
                bottleneck = torch.load(bottleneck_file_path).to(self.device)
                print(f"Loaded bottleneck shape: {bottleneck.shape}")
            else:
                print(f"Bottleneck file not found, training from scratch...")
                for idx in idxs_users:
                    print("cluster id:", idx)
                    attributes_embeddings, class_names = self.clients[idx].cluster(self.clients[idx].available_classes_indices, concept_num)
                    bottleneck = attributes_embeddings.to(self.device) if bottleneck is None else torch.concat((bottleneck, attributes_embeddings.to(self.device)), dim=0)

                # 确保save目录存在
                os.makedirs("./save/bottleneck", exist_ok=True)
                torch.save(bottleneck.cpu(), bottleneck_file_path)
                print(f"Saved bottleneck to {bottleneck_file_path}")

            print("bottleneck.shape", bottleneck.shape)

            for idx in idxs_users:
                _ = self.clients[idx].cluster(self.clients[idx].available_classes_indices, only_get_name_embeddings=True) 
                bottleneck = torch.load(bottleneck_file_path).to(self.device)
                self.clients[idx].ori_protos = copy.deepcopy(self.ori_protos)
                self.clients[idx].class_name_embeddings = copy.deepcopy(self.class_name_embeddings)
                self.clients[idx].bottleneck = copy.deepcopy(bottleneck)

                w_local = self.clients[idx].train_cbm(cluster=False)

                torch.save(self.clients[idx].Explainer, "./save/{}_{}_explainer_5e-4.pkl".format(self.clients[idx].data_name, self.clients[idx].client_id))

                local_acc.append(self.clients[idx].local_test(self.evaluator))
                all_acc.append(self.clients[idx].test(self.evaluator, "all"))

                self.clients[idx].personalization(epochs=10, lr=1e-3)
                new_local_acc.append(self.clients[idx].local_test(self.evaluator))

        print("local_acc", local_acc)
        print("local_acc_mean", np.mean(local_acc))
        print("new_local_acc", new_local_acc)
        print("new_local_acc_mean", np.mean(new_local_acc))
        print("all_acc", all_acc)
        print("all_acc_mean", np.mean(all_acc))
    
        return new_local_acc, all_acc

    def train_cbm_dir(self):
        """CBM Dirichlet训练方法"""
        self.before_train()
        
        for epoch in range(1):
            self.epoch = epoch
            num_selected = max(int(self.cfg.FEDERATED.AVAIL_PERCENT * self.num_clients), 1)
            idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)

            local_acc = []
            all_acc = []
            new_local_acc = []

            bottleneck = None
            concept_num = 100
            bottleneck_file_path = "./save/bottleneck_dir/{}_shot{}_num{}_bottleneck.pkl".format(self.cfg.EXPERIMENT.NAME, self.cfg.DATASET.NUM_SHOTS, concept_num)

            # 检查bottleneck文件是否存在
            if os.path.exists(bottleneck_file_path):
                print(f"Loading bottleneck from {bottleneck_file_path}")
                bottleneck = torch.load(bottleneck_file_path).to(self.device)
                print(f"Loaded bottleneck shape: {bottleneck.shape}")
            else:
                print(f"Bottleneck file not found, training from scratch...")
                for idx in idxs_users:
                    print("cluster id:", idx)
                    attributes_embeddings, class_names = self.clients[idx].cluster(self.clients[idx].available_classes_indices, concept_num)
                    bottleneck = attributes_embeddings.to(self.device) if bottleneck is None else torch.concat((bottleneck, attributes_embeddings.to(self.device)), dim=0)
                    print("before merging bottleneck.shape", bottleneck.shape)
                    bottleneck = torch.unique(bottleneck, dim=0)
                    print("after merging bottleneck.shape", bottleneck.shape)

                # 确保save目录存在
                os.makedirs("./save/bottleneck_dir", exist_ok=True)
                torch.save(bottleneck.cpu(), bottleneck_file_path)
                print(f"Saved bottleneck to {bottleneck_file_path}")

            print("bottleneck.shape", bottleneck.shape)

            for idx in idxs_users:
                _ = self.clients[idx].cluster(self.clients[idx].available_classes_indices, only_get_name_embeddings=True) 
                bottleneck = torch.load(bottleneck_file_path).to(self.device)
                self.clients[idx].ori_protos = copy.deepcopy(self.ori_protos)
                self.clients[idx].class_name_embeddings = copy.deepcopy(self.class_name_embeddings)
                self.clients[idx].bottleneck = copy.deepcopy(bottleneck)

                w_local = self.clients[idx].train_cbm(cluster=False)

                local_acc.append(self.clients[idx].local_test(self.evaluator))
                all_acc.append(self.clients[idx].test(self.evaluator, "all"))

                self.clients[idx].personalization(epochs=10, lr=1e-3)
                new_local_acc.append(self.clients[idx].local_test(self.evaluator))

        print("local_acc", local_acc)
        print("local_acc_mean", np.mean(local_acc))
        print("new_local_acc", new_local_acc)
        print("new_local_acc_mean", np.mean(new_local_acc))
        print("all_acc", all_acc)
        print("all_acc_mean", np.mean(all_acc))
    
        return new_local_acc, all_acc
