# 导入相关模块
from torch.utils.data import DataLoader, Dataset
import numpy as np, pickle
import argparse
import os,sys
from os.path import normpath,join
Base_DIR=normpath(join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0,Base_DIR)#添加环境变量，因为append是从列表最后开始添加路径，可能前面路径有重复，最好用sys.path.insert(Base_DIR)从列表最前面开始添加

from data.data_process_utils.concept_util import getReviewsConceptTriples, rawTriples2index, getGraphMaps# 为什么train找不到这个包


class ReviewGraphDataset(Dataset):  # 继承Dataset
    def __init__(self, domainList, data_root):  # __init__是初始化该类的一些基础参数

        self.rawDataset = getReviewsConceptTriples(domainList, data_root)  # 这个很耗时间 主要是conceptTriples的字典太大了，检索空间太大，建议将结果保存下来，往后直接读取

        self.maps = getGraphMaps(domainList)
        # self.relation_map = maps[0]# 文件目录 @jinhui 目前是硬绑定
        # self.concept_map = maps[1]# 来自总的大图
        # self.unique_nodes_mapping = maps[2]# 被reviewconcept过滤过的
    def __len__(self):  # 返回整个数据集的大小
        return len(self.rawDataset)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        review = self.rawDataset[index]  # 根据索引index获取该review
        maps = self.maps
        reviewTriples = rawTriples2index(review, maps)
        # 封装到concept_util了
        # for triple in review:# 这个步骤也很慢，推荐之后转换好直接读取
        #     try:
        #         # 到word2int
        #         srcMap = self.concept_map[triple[0]]
        #         relMap = self.relation_map[triple[1]]
        #         distMap = self.concept_map[triple[2]]
        #         #  到int2node_index # 存在数组越界的情况unique_nodes_mapping：8090 concept_map：118651# 实际上是数据不一致的问题
        #
        #         srcMap, distMap = self.unique_nodes_mapping[srcMap], self.unique_nodes_mapping[distMap]
        #     except:
        #         await = 0 # 实际上是数据不一致的问题 主要是前后数据没有连起来，导致字典为空的查询
        #         continue
        #     triple = [srcMap, relMap, distMap]
        #     reviewTriples.append(triple)

        return np.array(reviewTriples)  # 返回该review


def getDataSet(domainList, data_root):

    reviewGraphDataset = ReviewGraphDataset(domainList, data_root)

    return reviewGraphDataset





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/domain_data/processed_data",
                        help="data_json_dir")

    opt = parser.parse_args()

    print('Extracting review concept triples from domain/domains.')
    domainList = ["books"]  # , "dvd", "electronics", "kitchen"
    dataset = getDataSet(domainList, data_root=opt.data_path)
    sample = dataset[1]
    print("  ")