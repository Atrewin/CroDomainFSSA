# -*- coding: utf-8 -*-
import sys
import  time
from os.path import normpath,join,dirname
from utils.path_util import from_project_root
from utils import json_util
import requests
import argparse
import sys
import importlib
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

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
        conceptNetTriple = []
        for concept in reviewConcepts:
            conceptTripleList = conceptNetAPI(concept)
            time.sleep(2)
            conceptNetTriple.append(conceptTripleList)

        reviewJson["conceptNetTriple"] = conceptNetTriple
        reivewJsonList[index] = reviewJson
        pass

def getConceptNetTriple(data_file):
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
        return json_data
    else:
        return addConceptNetTripe2reviewJson(json_data)

    print(" ")

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
def getConceptGraphDict(allConcepts):
    ConceptGraphDict = {}
    global num
    for concept in allConcepts:
        conceptTripleList = conceptNetAPI(concept)
        time.sleep(2)
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
    # 每篇独立获取方案， 将有严重的重复访问导致的访问量限制问题
    # json_data = getConceptNetTriple(opt.json_url)
    # # 保存数据
    # json_util.dump(json_data, opt.json_url)

    # 统一获取后分配的方案
    domainList = ["books", "dvd", "electronics", "kitchen"]
    dataRoot = opt.data_path
    concept_keep_url = join(dataRoot, "ConceptsSetDict.json")
    concept_keep_url = from_project_root(concept_keep_url)

    conceptGraphDict_keep_url = join(dataRoot, "conceptGraphDict.json")
    conceptGraphDict_keep_url = from_project_root(conceptGraphDict_keep_url)

    #
    urlList = getDomainDataURL(domainList)
    allConcepts = getAllConcepts(urlList)# 有接近两万个，而concept明天最多拿到3600个
    # ConceptsSetDict = {}
    # ConceptsSetDict["allDomain"] = allConcepts
    # json_util.dump(ConceptsSetDict, concept_keep_url)
    #
    # ConceptsSetDict = json_util.load(concept_keep_url)
    # allConcepts = ConceptsSetDict["allDomain"]


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
        newConcept2ConceptGraphDict = getConceptGraphDict(allConcepts[i:j])
        conceptGraphDict["concept2ConceptGraphDict"].update(newConcept2ConceptGraphDict)
        time.sleep(3)
        conceptGraphDict["tag"] = str(start) + "_" + str(i+saveSetp)
        json_util.dump(conceptGraphDict,conceptGraphDict_keep_url)
        print('\r当前进度：{0}{1}%'.format('▉' * int((i+saveSetp)/(end-start)*10), ((i+saveSetp)/(end-start)*100)), end='', flush=True)
        conceptGraphDict = json_util.load(conceptGraphDict_keep_url)


    print("完成获取concept"+ str(start) + end + "的获取")
