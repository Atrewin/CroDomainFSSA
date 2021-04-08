import  time

import os,sys
from os.path import normpath,join,dirname
# print(__file__)#获取的是相对路径
# print(os.path.abspath(__file__))#获得的是绝对路径
# print(os.path.dirname(os.path.abspath(__file__)))#获得目录的绝对路径
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#获得的是Test的绝对路径
import numpy as np, pickle, argparse

from os.path import normpath,join,dirname
Base_DIR=normpath(join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0,Base_DIR)#添加环境变量，因为append是从列表最后开始添加路径，可能前面路径有重复，最好用sys.path.insert(Base_DIR)从列表最前面开始添加

from utils import json_util#找包的优先级情况是什么样的？
from utils.path_util import from_project_root


# 是从data_linkConceptNet 迁移过来的，未来需要做代码整合
def getDomainDataURL(domainList, data_root="data/domain_data/processed_data"):
    root = data_root
    root = from_project_root(root)
    urlList = []
    for domain in domainList:
        if domain == "books":
            labelURL = join(root, domain, "book_reviews.json")
            unlabeled = join(root, domain, "book_unlabeled_reviews.json")

        elif domain == "dvd":
            labelURL = join(root, domain, "dvd_reviews.json")
            unlabeled = join(root, domain, "dvd_unlabeled_reviews.json")

        elif domain == "electronics":
            labelURL = join(root, domain, "electronics_reviews.json")
            unlabeled = join(root, domain, "electronics_unlabeled_reviews.json")

        elif domain == "kitchen":
            labelURL = join(root, "kitchenAndhousewares", "kitchen_reviews.json")
            unlabeled = join(root, "kitchenAndhousewares", "kitchen_unlabeled_reviews.json")
        else:
            erro = 0/0

        labelURL = from_project_root(labelURL)
        unlabeled = from_project_root(unlabeled)
        urlList.append(labelURL)
        # 暂时不需要它的参与
        # urlList.append(unlabeled)
    return urlList

def getAllConcepts(urlList):
    conceptSet= set()
    for url in urlList:
        json_data = json_util.load(url)
        if "unlabeled" not in url:
            # labled 的情况
            for mode in ["pos", "neg"]:
                reivewJsonList = json_data[mode]
                for reviewJson in reivewJsonList:
                    reviewConcepts = reviewJson["concepts"]
                    conceptSet.update(reviewConcepts)
        else:
            pass
            for reviewJson in json_data:
                if "concepts" in reviewJson.keys():
                    reviewConcepts = reviewJson["concepts"]
                    conceptSet.update(reviewConcepts)


    return list(conceptSet)


import random


def getAllOpinionConceptTriple(urlList):
    opinionConceptTripleList= [] #特意不仅进行去重处理, 但是加上unlabelled的时候有160000+
    for url in urlList:
        json_data = json_util.load(url)
        if "unlabeled" not in url:
            # labled 的情况
            for mode in ["pos", "neg"]:
                reivewJsonList = json_data[mode]
                for reviewJson in reivewJsonList:
                    reviewOpinionConceptTriples = reviewJson["opinionConceptTriples"]
                    # 这里不要改变原数据, 需要再考虑在哪里进行替换
                    # temp = []
                    # romdomRel = random.randint(0, 9)
                    # for index, triple in enumerate(reviewOpinionConceptTriples):
                    #     triple[1] = str(romdomRel)
                    #     reviewOpinionConceptTriples[index] = triple
                    #     pass
                    opinionConceptTripleList.extend(reviewOpinionConceptTriples)
        else:
            pass
            # # 不需要那么多unlabelled数据
            reivewJsonList = json_data
            for reviewJson in reivewJsonList:
                if "opinionConceptTriples" in reviewJson.keys():
                    reviewOpinionConceptTriple = reviewJson["opinionConceptTriples"]
                    opinionConceptTripleList.extend(reviewOpinionConceptTriple)


    return list(opinionConceptTripleList)

def getReviewConceptNetTriples(conceptList):# concept2ConceptGraphDict太大的是导致的结果是查询的时间很大
    # return [len(concepts), len(triples)]
    dataRoot = "data/domain_data/init_data"
    conceptGraphDict_keep_url = join(dataRoot, "conceptGraphDict.json")
    conceptGraphDict_keep_url = from_project_root(conceptGraphDict_keep_url)

    if os.path.exists(conceptGraphDict_keep_url):
        concept2ConceptGraphDict = json_util.load(conceptGraphDict_keep_url)["concept2ConceptGraphDict"]
    else:
        concept2ConceptGraphDict = {}
    reviewConceptNetTriples = []
    for concept in conceptList:
        # 因为前面连接的字典可能早不到这个图所以需要先检查concept2ConceptGraphDict[concept]是否存在
        if concept in concept2ConceptGraphDict.keys():
            concept2ConceptGraphDict[concept]
            reviewConceptNetTriples.extend(concept2ConceptGraphDict[concept])



    return reviewConceptNetTriples

def changeOpinionConceptRel(reviewOpinionConceptTriples):

    romdomRel = random.randint(0, 9)
    romdomRel = str(romdomRel)
    for index, triple in enumerate(reviewOpinionConceptTriples):
        triple[1] = str(romdomRel)
        reviewOpinionConceptTriples[index] = triple

    return reviewOpinionConceptTriples

def getReviewsConceptTriples(domainList, data_root):# 耗时很大，需要加入提醒交互打印
    # 如果要分开领域来训练,那么通过 urlList来限制
    # return [len(reviews), len(triples)]

    urlList = getDomainDataURL(domainList, data_root)

    count = 0
    reviewsConceptTriples = []
    for url in urlList:
        json_data = json_util.load(url)
        if "unlabeled" not in url:
            # labled 的情况
            for mode in ["pos", "neg"]:
                reivewJsonList = json_data[mode]
                for reviewJson in reivewJsonList:
                    reviewOpinionConceptTriples = reviewJson["opinionConceptTriples"]
                    reviewOpinionConceptTriples = changeOpinionConceptRel(reviewOpinionConceptTriples)

                    if "conceptNetTriples" in reviewJson.keys():
                        reviewConceptNetTriples = reviewJson["conceptNetTriples"]# 这个函数放到外面集中执行，将结果存到json
                    else:
                        reviewConceptNetTriples = []
                    reviewConceptNetTriples.extend(reviewOpinionConceptTriples)
                    if len(reviewConceptNetTriples) != 0:
                        reviewsConceptTriples.append(reviewConceptNetTriples)
                    count += 1
                    print('\r当前进度：{0}'.format(count), end='', flush=True)
        else:
            pass
            # # 不需要unlabelled数据先
            for reviewJson in json_data:
                reviewOpinionConceptTriples = reviewJson["opinionConceptTriples"]
                reviewOpinionConceptTriples = changeOpinionConceptRel(reviewOpinionConceptTriples)

                if "conceptNetTriples" in reviewJson.keys():
                    reviewConceptNetTriples = reviewJson["conceptNetTriples"]  # 这个函数放到外面集中执行，将结果存到json
                else:
                    reviewConceptNetTriples = []
                reviewConceptNetTriples.extend(reviewOpinionConceptTriples)

                if len(reviewConceptNetTriples) != 0:
                    reviewsConceptTriples.append(reviewConceptNetTriples)
                count += 1
                print('\r当前进度：{0}'.format(count), end='', flush=True)

    return reviewsConceptTriples




