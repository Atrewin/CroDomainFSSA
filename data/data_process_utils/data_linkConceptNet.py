# -*- coding: utf-8 -*-
import sys
import time
import os
from os.path import normpath,join,dirname
from utils.path_util import from_project_root
from utils import json_util
import requests
import argparse
import sys
import importlib
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from data.data_process_utils.concept_util import getReviewConceptNetTriples
def conceptNetAPI(word):
    url = "http://api.conceptnet.io/c/en/" + word + "?offset=0&limit=50"
    edges = requests.get(url).json()["edges"]# 被限制了20条, 需要都拿到吗？
    triples = []
    for edge in edges:
        import traceback

        try:
            if edge["start"]["language"] == "en" and edge["end"]["language"] == "en":# 边缘节点不会有[language]
                startConcept = edge["start"]["label"]
                endConcept = edge["end"]["label"]
                rel = edge["rel"]["label"]
                triples.append((startConcept, rel, endConcept))
            pass
        except:
            pass

    return triples
def addConceptNetTripe2reviewJson(reivewJsonList):

    for index, reviewJson in enumerate(reivewJsonList):

        reviewConcepts = reviewJson["concepts"]
        conceptNetTriples = getReviewConceptNetTriples(reviewConcepts)
        reviewJson["conceptNetTriples"] = conceptNetTriples
        reivewJsonList[index] = reviewJson
        pass
    return reivewJsonList
def addConceptNetTriple2JsonData(data_file):
    """
    :param data_file:
    :return:
    """
    url = from_project_root(data_file)
    json_data = json_util.load(url)
    if "unlabeled" not in data_file:
        # labled 的情况
        for mode in ["pos", "neg"]:
            reivewJsonList = json_data[mode]
            reivewJsonList = addConceptNetTripe2reviewJson(reivewJsonList)
            json_data[mode] = reivewJsonList
            pass
        # 保存数据
        json_util.dump(json_data, url)

    else:
        json_data = addConceptNetTripe2reviewJson(json_data)
        json_util.dump(json_data, url)

    print("addConceptNetTriple2JsonData to " + data_file)

def getAllConcepts(urlList):
    conceptSet= set()
    for url in urlList:
        json_data = json_util.load(url)
        if "unlabeled" not in url:
            # labled 的情况
            for mode in ["pos", "neg"]:
                reivewJsonList = json_data[mode]
                for reviewJson in reivewJsonList:
                    if "concepts" in reviewJson.keys():
                        reviewConcepts = reviewJson["concepts"]
                    else:
                        reviewConcepts = []
                    conceptSet.update(reviewConcepts)
        else:
            for reviewJson in json_data:
                if "concepts" in reviewJson.keys():
                    reviewConcepts = reviewJson["concepts"]
                    conceptSet.update(reviewConcepts)


    return list(conceptSet)

def getDomainDataURL(domainList):
    root = "data/domain_data/processed_data"
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
        urlList.append(unlabeled)
    return urlList
global num
num = 0
def getConceptGraphDict(allConcepts, presentConceptGraphDict):
    ConceptGraphDict = {}
    global num
    for concept in allConcepts:
        if concept not in presentConceptGraphDict.keys():
            conceptTripleList = conceptNetAPI(concept)
            time.sleep(1)
            ConceptGraphDict[concept] = conceptTripleList

        print('\r当前进度：{0}'.format(num), end='', flush=True)
        num = num + 1

    return ConceptGraphDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/domain_data/processed_data",
                        help="data_json_dir")
    parser.add_argument("--start", default=0,
                        help="beginIndex")
    parser.add_argument("--end", default=2000,
                        help="endIndex")
    opt = parser.parse_args()

    # 统一获取后分配的方案
    domainList = ["books", "dvd", "electronics", "kitchen"]
    dataRoot = opt.data_path
    concept_keep_url = join(dataRoot, "ConceptsSetDict.json")
    concept_keep_url = from_project_root(concept_keep_url)

    conceptGraphDict_keep_url = join(dataRoot, "conceptGraphDict.json")
    conceptGraphDict_keep_url = from_project_root(conceptGraphDict_keep_url)

    # TODO save allconcept
    urlList = getDomainDataURL(domainList)
    allConcepts = getAllConcepts(urlList)# 有接近两万个，而concept每小时最多拿到3600个
    ConceptsSetDict = {}
    ConceptsSetDict["allDomain"] = allConcepts
    json_util.dump(ConceptsSetDict, concept_keep_url)


    # TODO link conceptNet
    ConceptsSetDict = json_util.load(concept_keep_url)
    allConcepts = ConceptsSetDict["allDomain"]


    start = opt.start
    end = opt.end
    end = len(allConcepts)
    saveSetp = 500

    conceptGraphDict = {
        "tag": "none",
        "concept2ConceptGraphDict": {}
    }
    for i in range(start,end,saveSetp):
        j = min(i+saveSetp,end)
        if os.path.exists(conceptGraphDict_keep_url):
            conceptGraphDict = json_util.load(conceptGraphDict_keep_url)
        newConcept2ConceptGraphDict = getConceptGraphDict(allConcepts[i:j], conceptGraphDict["concept2ConceptGraphDict"])
        conceptGraphDict["concept2ConceptGraphDict"].update(newConcept2ConceptGraphDict)
        time.sleep(3)
        conceptGraphDict["tag"] = str(start) + "_" + str(i+saveSetp)
        json_util.dump(conceptGraphDict,conceptGraphDict_keep_url)
        print('\r当前进度：{0}{1}%'.format('▉' * int((i+saveSetp)/(end-start)*10), ((i+saveSetp)/(end-start)*100)), end='', flush=True)



    print("完成获取concept"+ str(start) + "-" + str(end) + "的获取")

    # TODO 回写到reviewJosnData
    urlList = getDomainDataURL(domainList)
    # 不用把，这个只是一个技巧文件，原则上不需要
    for url in urlList:
        addConceptNetTriple2JsonData(url)

    print("完成获取conceptTriple写入JsonData")
