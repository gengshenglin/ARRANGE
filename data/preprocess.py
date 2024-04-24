from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from torch.utils.data import Subset
import random
import torch
import numpy as np


def mnist_preprocess_varying(mnist, client_num: int, distribution_ratios):
    clients_data = []
    all_idxs = set(range(len(mnist)))  # 包含所有可能索引的集合
    
    for i in range(client_num):
        per_samples_num = int(distribution_ratios[i] * len(mnist))
        
        selected_idxs = set(random.sample(all_idxs, min(per_samples_num, len(all_idxs))))
        
        mnist_data = Subset(mnist, list(selected_idxs))
        clients_data.append(mnist_data)
        
        all_idxs -= selected_idxs

    return clients_data



# def mnist_preprocess(mnist, client_num: int, sample_rate: float = 0.1):
#     # return [Subset(mnist,random.sample(range(len(mnist)),int(sample_rate*len(mnist)))) for _ in range(client_num)]

#     clients_data = []

#     per_samples_num = int(sample_rate * len(mnist))
#     per_others_num = len(mnist) - per_samples_num

#     for i in range(client_num):

#         mnist_data, other_data = random_split(mnist, [per_samples_num, per_others_num])
#         clients_data.append(mnist_data)
#     return clients_data

def mnist_preprocess(mnist, client_num: int, sample_rate: float = 0.1):
    clients_data = []
    all_idxs = set(range(len(mnist)))  
    per_samples_num = int(sample_rate * len(mnist))
    for i in range(client_num):
        selected_idxs = set(random.sample(all_idxs, min(per_samples_num, len(all_idxs))))
    
        mnist_data = Subset(mnist, list(selected_idxs))
        clients_data.append(mnist_data)
    
        all_idxs -= selected_idxs
    return clients_data

def mnist_preprocess_non_iid(mnist, client_num: int, sample_rate: float = 0.2, num_classes=10):
    clients_data = []
    all_idxs = {k: [] for k in range(num_classes)}
    per_client_samples = int(sample_rate * len(mnist))

    for idx, (_, label) in enumerate(mnist):
        all_idxs[label].append(idx)

    for i in range(client_num):
        client_idxs = []

        while len(client_idxs) < per_client_samples:
            for label in range(num_classes):
                num_samples = random.randint(1, per_client_samples - len(client_idxs))
                sampled_idxs = random.sample(all_idxs[label], min(num_samples, len(all_idxs[label])))
                client_idxs.extend(sampled_idxs)

                all_idxs[label] = [idx for idx in all_idxs[label] if idx not in sampled_idxs]

                if len(client_idxs) >= per_client_samples:
                    break

        mnist_data = Subset(mnist, client_idxs[:per_client_samples])
        clients_data.append(mnist_data)

    return clients_data


def mnist_preprocess_two_classes_varying_sizes(mnist, client_num: int, sample_rate:float,num_classes: int):
    clients_data = []
    
    class_idxs = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(mnist):
        class_idxs[label].append(idx)
    
    for _ in range(client_num):
       selected_classes = random.sample(range(10), num_classes)
       total_samples_per_client = int(sample_rate * len(mnist))
       available_samples_class_1 = len(class_idxs[selected_classes[0]])
       available_samples_class_2 = len(class_idxs[selected_classes[1]])
       min_available_samples = min(available_samples_class_1, available_samples_class_2)

       if total_samples_per_client > min_available_samples:
           total_samples_per_client = min_available_samples

       samples_first_class = random.randint(1, total_samples_per_client - 1)
       samples_second_class = total_samples_per_client - samples_first_class

       selected_idxs = random.sample(class_idxs[selected_classes[0]], samples_first_class)
       selected_idxs.extend(random.sample(class_idxs[selected_classes[1]], samples_second_class))

       mnist_data = Subset(mnist, selected_idxs)
       clients_data.append(mnist_data)
       
    return clients_data

