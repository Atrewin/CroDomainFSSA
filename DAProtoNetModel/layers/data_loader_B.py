import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
from utils.logger import *

class EvalDataset(data.Dataset):
    """
    FewRel Dataset
    return
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, data_mode="val"):
        self.root = root
        path = getDatesetFilePath(root,name)
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path, encoding='utf-8'))  # ok
        self.classes = list(self.json_data.keys())

        self.part_data(data_mode)
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate  # unknow
        self.encoder = encoder  # @jinhui 用来index id 化

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'])  # 单词下标化
        return word
    def part_data(self,data_mode):
        target_classes = ["pos", "neg"]
        spilt = [0.5, 0.7]
        if "test" == data_mode:
            spilt = [0.7, 1]
        if "val" == data_mode:
            spilt = [0.5, 0.7]
        if data_mode == "train":
            spilt = [0, 0.5]
        for i, class_name in enumerate(target_classes):
            length = 1000#len(self.json_data[class_name]) 有异步问题
            self.json_data[class_name] = self.json_data[class_name][int(length*spilt[0]):int(length*spilt[1])]

    def __additem__(self, d, word):
        d['word'].append(word)

    def __getitem__(self, index):
        target_classes = ["pos", "neg"]  # 固定为[pos, neg], 那么意味着pos的index为0，neg的index为1。
        support_set = {'word': []}
        support_label = []

        for i, class_name in enumerate(target_classes):
            indices = [index]
            for j in indices:
                word = self.__getraw__(
                    self.json_data[class_name][j])
                word = torch.tensor(word).long()
                self.__additem__(support_set, word)
            support_label += [i]

        return support_set, support_label

    def __len__(self):
        return len(self.json_data["pos"])

def getDatesetFilePath(root, name):
    if "book" in name:
        name = "books/" + name
    elif "dvd" in name:
        name = "dvd/" + name
    elif "electr" in name:
        name = "electronics/" + name
    elif "kitchen" in name:
        name  = "kitchenAndhousewares/" + name

    path = os.path.join(root, name + ".json")
    return path





def collate_eval_fn(data):
    batch_support = {'word': []}
    batch_support_labels = []

    support_sets, support_labels= zip(*data)
    for i in range(len(support_sets)):# i = batch
        for k in support_sets[i]:# k = ["word", "graphFeature"]
            batch_support[k] += support_sets[i][k]
        batch_support_labels += support_labels[i]

    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)

    # @jinhui 个人感觉不应该在这里就变为tensor
    batch_support_labels = torch.tensor(batch_support_labels)
    return batch_support, batch_support_labels


def get_loader_val(name, encoder, N, K, Q, batch_size,
        num_workers=1, na_rate=0, root='./data/domain_data/processed_data', data_mode=None):
    dataset = EvalDataset(name, encoder, N, K, Q, na_rate, root, data_mode=data_mode)#已经ID化 @jinhui 改变了FewRelDataset
    # temp = dataset[0]# 参看数据format
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_eval_fn)# 这里的多线程会导致random到相同的值)
    return data_loader# 为什么初始化的时候会被读取一次





if __name__ == '__main__':

    pass
