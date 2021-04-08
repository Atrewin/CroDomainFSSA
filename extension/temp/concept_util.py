# import  time
#
# import os,sys
# from os.path import normpath,join,dirname
# # print(__file__)#获取的是相对路径
# # print(os.path.abspath(__file__))#获得的是绝对路径
# # print(os.path.dirname(os.path.abspath(__file__)))#获得目录的绝对路径
# # print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#获得的是Test的绝对路径
#
# Base_DIR=normpath(join(os.path.dirname(os.path.abspath(__file__)), '../..'))
# sys.path.insert(0,Base_DIR)#添加环境变量，因为append是从列表最后开始添加路径，可能前面路径有重复，最好用sys.path.insert(Base_DIR)从列表最前面开始添加
# print(Base_DIR)
# from os.path import normpath,join,dirname
# from utils.path_util import from_project_root
# from utils import json_util
#
# # 是从data_linkConceptNet 迁移过来的，未来需要做代码整合
# def getDomainDataURL(domainList):
#     root = "data/domain_data/processed_data"
#     urlList = []
#     for domain in domainList:
#         if domain == "books":
#             labelURL = join(root, domain, "book_reviews.json")
#             unlabeled = join(root, domain, "book_unlabeled_reviews.json")
#
#         elif domain == "dvd":
#             labelURL = join(root, domain, "dvd_reviews.json")
#             unlabeled = join(root, domain, "dvd_unlabeled_reviews.json")
#
#         elif domain == "electronics":
#             labelURL = join(root, domain, "electronics_reviews.json")
#             unlabeled = join(root, domain, "electronics_unlabeled_reviews.json")
#
#         elif domain == "kitchen":
#             labelURL = join(root, "kitchenAndhousewares", "kitchen_reviews.json")
#             unlabeled = join(root, "kitchenAndhousewares", "kitchen_unlabeled_reviews.json")
#         else:
#             erro = 0/0
#
#         labelURL = from_project_root(labelURL)
#         unlabeled = from_project_root(unlabeled)
#         urlList.append(labelURL)
#         urlList.append(unlabeled)
#     return urlList
#
# def getAllConcepts(urlList):
#     conceptSet= set()
#     for url in urlList:
#         json_data = json_util.load(url)
#         if "unlabeled" not in url:
#             # labled 的情况
#             for mode in ["pos", "neg"]:
#                 reivewJsonList = json_data[mode]
#                 for reviewJson in reivewJsonList:
#                     reviewConcepts = reviewJson["concepts"]
#                     conceptSet.update(reviewConcepts)
#         else:
#             reviewConcepts = reviewJson["concepts"]
#             conceptSet.update(reviewConcepts)
#
#
#     return list(conceptSet)
