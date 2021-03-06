import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
from utils.logger import *

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    return
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path, encoding='utf-8'))# ok
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate# unknow
        self.encoder = encoder# @jinhui 用来index id 化

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'])  # 单词下标化
        return word

    def __additem__(self, d, word):
        d['word'].append(word)

    def __getitem__(self, index):
        target_classes = ["pos", "neg"]  # 固定为[pos, neg], 那么意味着pos的index为0，neg的index为1。
        support_set = {'word': []}
        query_set = {'word': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word = self.__getraw__(
                        self.json_data[class_name][j])
                word = torch.tensor(word).long()
                if count < self.K:
                    self.__additem__(support_set, word)
                else:
                    self.__additem__(query_set, word)
                count += 1

            query_label += [i] * self.Q

        return support_set, query_set, query_label
    
    def __len__(self):
        return 1000000000


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
class FewGraphDataset(data.Dataset):
    """
    FewRel Dataset add Graph feature
    return
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root, isNewGraphFeature=True):
        self.root = root
        path = getDatesetFilePath(root, name)

        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path, encoding='utf-8'))  # ok
        self.isNewGraphFeature = isNewGraphFeature
        if not isNewGraphFeature:
            self.graph_feature = self.load_graphFeature(self.graphFeaturePath(root,name))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate  # unknow
        self.encoder = encoder  # @jinhui 用来index id 化

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'])  # 单词下标化
        return word

    def getGraphFeature(self, class_name, j):

        if self.isNewGraphFeature:
            graphFeature = self.json_data[class_name][j]["graphFeature"]
        else:
            if class_name == "pos":
                graphFeature = self.graph_feature[j]
            else:
                graphFeature = self.graph_feature[j + 1000]
        graphFeature = torch.tensor(graphFeature).type(torch.FloatTensor)
        return graphFeature

    def __additem__(self, d, word, graphFeature):
        d['word'].append(word)
        d["graphFeature"].append(graphFeature)

    def __getitem__(self, index):
        try:# 有些样本缺失了graphFeature
            target_classes = ["pos", "neg"]  # 固定为[pos, neg], 那么意味着pos的index为0，neg的index为1。
            support_set = {'word': [], "graphFeature": []}
            query_set = {'word': [], "graphFeature": []}
            query_label = []
            support_label = []
            for i, class_name in enumerate(target_classes):
                indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))),
                    self.K + self.Q, False)
                # if index%100 < 10:
                    # if class_name == "pos":
                    #     prx = ""
                    #     for indice in indices:
                    #         prx += (str(indice) + "-")
                        # print(str(index) + " " + class_name + ":         " + prx)#@jinhui check
                count = 0
                for j in indices:  # 如数据是进行了"neg", "pos"和并的
                    word = self.__getraw__(
                        self.json_data[class_name][j])
                    word = torch.tensor(word).long()
                    graphFeature = self.getGraphFeature(class_name, j)


                    if count < self.K:
                        self.__additem__(support_set, word, graphFeature)

                    else:
                        self.__additem__(query_set, word, graphFeature)
                    count += 1
                support_label += [i] * self.K
                query_label += [i] * self.Q

                # for i2 in indices:#@jinhui check
                #     self.DICT[i2] = 1
                #     self.dict_val[i2] = 1
        except:
            # prx = ""
            # for indice in indices:
            #     prx += (str(indice) + "-")
            # print("bad indices: " + prx)#@jinhui check
            index = random.randint(0,self.__len__())
            support_set, query_set, query_label, support_label = self.__getitem__(index)


        return support_set, query_set, query_label, support_label

    def __len__(self):
        return 1000000000

    def load_graphFeature(self, path):
        X_s_ = np.load(open(path, 'rb'), allow_pickle=True)
        return X_s_

    def graphFeaturePath(self, root, name):
        # path = os.path.join(root, name + ".json")
        # if not os.path.exists(path):
        #     print("[ERROR] Data file does not exist!")
        #     assert (0)
        # self.json_data = json.load(open(path, encoding='utf-8'))  # ok

        name = name.split("_")[0]
        path = os.path.join(root, 'graph_features/sf_' + name + '_small_5000.np')
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        return path


def collate_fn(data):
    batch_support = {'word': [], "graphFeature":[]}
    batch_query = {'word': [], "graphFeature":[]}
    batch_label = []
    batch_support_lsbels = []
    support_sets, query_sets, query_labels,  support_lsbels= zip(*data)
    for i in range(len(support_sets)):# i = batch
        for k in support_sets[i]:# k = ["word", "graphFeature"]
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]

        batch_label += query_labels[i]
        batch_support_lsbels += support_lsbels[i]

    for k in batch_support:# k = ["word", "graphFeature"]
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    # @jinhui 个人感觉不应该在这里就变为tensor
    batch_label = torch.tensor(batch_label)
    batch_support_lsbels = torch.tensor(batch_support_lsbels)
    return batch_support, batch_query, batch_label, batch_support_lsbels


def get_loader(name, encoder, N, K, Q, batch_size, 
        num_workers=1, collate_fn=collate_fn, na_rate=0, root='./data/domain_data/processed_data', opt=None):
    dataset = FewGraphDataset(name, encoder, N, K, Q, na_rate, root, isNewGraphFeature=opt.isNewGraphFeature)#已经ID化 @jinhui 改变了FewRelDataset
    # temp = dataset[0]# 参看数据format
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,# 这里的多线程会导致random到相同的值
            collate_fn=collate_fn)
    return iter(data_loader)# 为什么初始化的时候会被读取一次




def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label



class FewRelUnsupervisedDataset(data.Dataset):
    """
        FewRel Unsupervised Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = getDatesetFilePath(root, name)

        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        # self.json_data = json.load(open(path, encoding='utf-8'))# no self.json_data = json.load(open(path, encoding='utf-8'))
        self.json_data = json.load(open(path, encoding='utf-8'), encoding='utf-8')
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word= self.encoder.tokenize(item['tokens'])
        return word

    def __additem__(self, d, word, graphFeature):
        d['word'].append(word)
        #@jinhui
        d['graphFeature'].append(torch.tensor([0] * 100).type(torch.FloatTensor))  # 为了和label数据统一， 又不想花时间查结果
        return d
    def getGraphFeature(self):


        pass


    def __getitem__(self, index):
        total = self.N * self.K   # 获取 support set共有的样本个数作为domain classifier的输入
        support_set = {'word': [],
                       'graphFeature':[]
                       }

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word = self.__getraw__(self.json_data[j])   # 下标化
            word = torch.tensor(word).long()

            graphFeature  = self.getGraphFeature()
            support_set = self.__additem__(support_set, word, graphFeature)# 真的是传引用吗？



        return support_set
    
    def __len__(self):
        return 1000000000


def collate_fn_unsupervised(data):
    batch_support = {'word': [],
                     "graphFeature":[]
                     }
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support


def get_loader_unsupervised(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_unsupervised, na_rate=0, root='./data/domain_data/processed_data'):
    dataset = FewRelUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)



if __name__ == '__main__':

    pass
