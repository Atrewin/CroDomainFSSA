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
from concept_util import getDomainDataURL, getAllOpinionConceptTriple



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/domain_data/processed_data",
                        help="data_json_dir")

    opt = parser.parse_args()

    # 统一获取后分配的方案
    domainList = ["books", "dvd", "electronics", "kitchen"]# , "dvd", "electronics", "kitchen"
    dataRoot = opt.data_path
    concept_keep_url = join(dataRoot, "ConceptsSetDict.json")
    concept_keep_url = from_project_root(concept_keep_url)

    conceptGraphDict_keep_url = join(dataRoot, "conceptGraphDict.json")
    conceptGraphDict_keep_url = from_project_root(conceptGraphDict_keep_url)


    conceptGraphDict = {
        "tag": "none",
        "concept2ConceptGraphDict": {}
    }



    # print("完成获取concept"+ str(start) + "-" + str(end) + "的获取")

    if os.path.exists(conceptGraphDict_keep_url):
        conceptGraphDict = json_util.load(conceptGraphDict_keep_url)
    print("  ")

    dataRoot = "extension/Graph-Embedding"
    conceptnet_keep_url = join(dataRoot, "conceptnet_english_ours.txt")
    conceptnet_keep_url = from_project_root(conceptnet_keep_url)


    with open(conceptnet_keep_url, mode='w', encoding="utf-8") as f:
        concept2ConceptGraphDict = conceptGraphDict["concept2ConceptGraphDict"]
        for conceptList in concept2ConceptGraphDict.values():

            for triple in conceptList:
                f.write(triple[0])
                f.write('\t')
                f.write(triple[2])
                f.write('\t')
                f.write(triple[1])
                f.write('\n')
            pass

        domainList = ["books", "dvd", "electronics", "kitchen"]#
        urlList = getDomainDataURL(domainList)
        opinionConceptTriples = getAllOpinionConceptTriple(urlList)
        for triple in opinionConceptTriples:
            f.write(triple[0])
            f.write('\t')
            f.write(triple[2])
            f.write('\t')
            f.write(triple[1])
            f.write('\n')
        pass

        # 加入10条备用边
        for i in range(10):
            f.write("conceptA")
            f.write('\t')
            f.write("conceptB")
            f.write('\t')
            f.write(str(i))
            f.write('\n')



        f.close()

    # 当前的opinionConcept是加到末尾的,考虑是否要打乱
    print("")
    print("")


