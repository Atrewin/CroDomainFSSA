# coding=utf-8
import sys
from os.path import normpath,join,dirname

# sys.getdefaultencoding()    # 查看设置前系统默认编码
# sys.setdefaultencoding('utf-8')
# sys.getdefaultencoding()    # 查看设置后系统默认编码
# print("---"*15)
# print(__file__)
# print(normpath(join(dirname(__file__), '../..')), flush=True)# 指向的是你文件运行的路径，如果在命令行跑那么它是根据你启动的路径来确认的
sys.path.append(normpath(join(dirname(__file__), '../..')))
# 命令行中带的坑： 要加个PYTHONPATH=. 指向python工程的根目录，就可以省去很多麻烦（这是pycharm帮我们集成了的）
# 运行的环境路径和工程链接路径的差异性   #核心冲突，命令行认为的项目根目录和实际的项目根目录

# 使用统一基于工程根目录的方式组织文件目录
from utils.path_util import from_project_root
from utils import json_util

import argparse

# 获取doc的entities
import stanza
# stanza.download("en")
nlp = stanza.Pipeline('en', use_gpu=True)

def sent_process(reviews):
    """
    :param reviews: 评论
    :return:
    """
    processed_reviews = []
    for review in reviews:
        review = review.replace("...", " ")
        review = review.replace("  ", " ")
        review = review.replace(")", " ")
        review = review.replace("(", " ")
        review = review.replace(".", " ")
        review = review.replace("\"", " ")
        review = review.replace("&quot;", " ")
        review = review.replace("!", " ")
        review = review.replace(":", " ")
        review = review.replace(",", " ")
        processed_reviews.append(review)

    return processed_reviews


def sentence_extractor(xml_url):
    """
    :param xml_url:
    :return: reviews: 数组类型，
    """
    tag = False
    reviews = []
    single_review = []
    f = open(xml_url, encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        if line.strip() == "<review_text>":
            tag = True
            single_review = []
            continue
        if line.strip() == "</review_text>":
            reviews.append(" ".join(single_review))
            tag = False
        if tag:
            single_review.append(line.strip())

    return reviews
class Filter():
    def __init__(self):
        # 选择concept
        self.uposKeepType = ["NOUN", "VERB", "ADJ"]# "ADV" 是否可以做更多的concept类型过滤 upos tag #用xpos会更加精确
        self.xposKeepType = ["JJ", "JJR", "JJS", "NNS", "NN", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]  # 考虑是否需要副词的参与

    pass

    def isKeepConcept(self, token):
        return token.words[0].upos in self.uposKeepType and token.words[0].xpos in self.xposKeepType
        pass

    def isKeepConceptTriple(self, dependency, methodSelect=1):

        src = dependency[0]
        rel = dependency[1]
        dist = dependency[2]


        if methodSelect == 1:
            # 方案1: 利用以形容词为中心的过滤

            # 方案1：保留opinionToken为中心的所有关系
            uposKeepType = ["ADJ"]
            filterRelType = ["obl:npmod", "det", "cc", "det:predet"]# 先写在这里为了可读性， 后期需要写道函数外，避免频繁读写
            keepRelType = ["nsubj", "amod", "advcl"]
            if src.id != 0 and (src.upos != "PUNCT" and dist.upos != "PUNCT") and rel in keepRelType:  # root 是没有upos等信息
                # 如果一个节点是形容词 且它的邻居节点是一个concept就保留
                if (src.upos in uposKeepType and dist.upos in self.uposKeepType) or (dist.upos in uposKeepType and src.upos in self.uposKeepType):
                    return True
            return False

        elif methodSelect == 2:
            # 方法2：利用 jj nusubj NN 强约束 过滤太厉害 来自模式匹配

            srcType = ["NOUN","ADJ"]
            distType = ["ADJ"]
            relType = ["nsubj","amod"] # amod 可能是和ADJ是重复的条件 advcl
            if src.id != 0 :  # root 是没有upos等信息
                if src.upos in srcType and rel in relType and  dist.upos in distType:
                    return True
            return False


        elif methodSelect == 3:
            # 方案3 还可利用rel类型进行强匹配

            relType = ["nsubj","amod"] # amod 可能是和ADJ是重复的条件 advcl
            if src.id != 0 :  # root 是没有upos等信息
                if src.upos in self.uposKeepType and rel in relType and  dist.upos in self.uposKeepType:
                    return True
            return False

        elif methodSelect == 4:
            # 强模式拼接
            # pattern one
            srcType = ["NOUN"]
            distType = ["ADJ"]
            relType = ["nsubj"]  # amod 可能是和ADJ是重复的条件 advcl
            if src.id != 0:  # root 是没有upos等信息
                if src.upos in srcType and rel in relType and dist.upos in distType:
                    return True

            # pattern two






            return False

            pass
        pass


def getReviewConceptsAndTriples(review, nlp, tripleSelectet=1):
    doc = nlp(review)
    concepts = []
    opinionConceptTriples = []
    filter = Filter()

    for sentence in doc.sentences:
        for token in sentence.tokens:
            if filter.isKeepConcept(token):
                concepts.append(token.words[0].lemma)# 不知道是否可以使用词性还原spent -> send


        for dependency in sentence.dependencies:
            # 需要定义什么样的语法关系需要保留下来
            if filter.isKeepConceptTriple(dependency, tripleSelectet):
                src = dependency[0]
                rel = dependency[1]
                dist = dependency[2]
                opinionConceptTriples.append((src.lemma, rel, dist.lemma))


            pass
    concepts = set(concepts)
    opinionConceptTriples = set(opinionConceptTriples)
    return list(concepts), list(opinionConceptTriples)

def getReviewJsons(rawReviews):
    # 文本预处理
    process_reviews = sent_process(rawReviews)
    reviews_arr = []
    for index, review in enumerate(process_reviews):
        # 获取tokens
        tokens = review.strip().split(" ")  # 可能需要进一步文本预处理
        # 获取 reviewConcept 和conceptTriple
        review = process_reviews[index]# rawReviews[index] 中间会有写review的报错， 是nlp中的问题
        tripleSelectet = 1
        concepts, opinionConceptTriples = getReviewConceptsAndTriples(review, nlp, tripleSelectet)
        reviewJson = {"tokens": tokens,
                      "concepts": concepts,
                      "opinionConceptTriples": opinionConceptTriples
                      }
        reviews_arr.append(reviewJson)
        print('\r当前进度：{0}'.format(index), end='', flush=True)


    return reviews_arr

# 标注数据格式转换
def reviews2json(pos_reviews, neg_reviews):
    """
    :param pos_reviews:
    :param neg_reviews:
    :return: json数据, 格式: {"pos":[{"tokens":***}], "neg":[{"tokens":***}]}
    """
    json_data = {}
    json_data['pos'] = getReviewJsons(pos_reviews)
    json_data['neg'] = getReviewJsons(neg_reviews)

    return json_data


#  未标注数据格式转换
def unlabeledReviews2json(unlabeled_reviews):
    """
    :param unlabeled_reviews:
    :return: list(数组)
    """
    return getReviewJsons(unlabeled_reviews)


def main():
    # 参数获取
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_xml_url", default="data/domain_data/init_data/books/positive.review", type=str,
                        help="Neg xml url")
    parser.add_argument("--neg_xml_url", default="data/domain_data/init_data/books/negative.review", type=str,
                        help="Pos xml url")
    parser.add_argument("--unlabeled_xml_url", default="data/domain_data/init_data/books/book.unlabeled", type=str,
                        help="unlabeled_xml_url")
    parser.add_argument("--keep_url", default="data/domain_data/processed_data/books/reviews.json", type=str,
                        help="save json url")
    parser.add_argument("--unlabeled_keep_url", default="data/domain_data/processed_data/books/unlabeled_reviews.json",
                        type=str, help="unlabeled_keep_url")
    opt = parser.parse_args()


    pos_xml_url = from_project_root(opt.pos_xml_url)
    neg_xml_url = from_project_root(opt.neg_xml_url)
    unlabeled_xml_url = from_project_root(opt.unlabeled_xml_url)
    keep_url = from_project_root(opt.keep_url)
    unlabeled_keep_url = from_project_root(opt.unlabeled_keep_url)

    # 从xml文件中抽取具体的reviews, return: 数组 array: raw doc
    pos_reviews = sentence_extractor(pos_xml_url)
    neg_reviews = sentence_extractor(neg_xml_url)
    unlabeled_reviews = sentence_extractor(unlabeled_xml_url)

    # 预处理 和 抽取文本的entities都封装到reviews2json内部因为entities的时候还需要使用到标点符号的分割
    # 将reviews转换成json结构 //包含tokens化；get entities
    json_data = reviews2json(pos_reviews, neg_reviews)
    unlabeled_json_data = unlabeledReviews2json(unlabeled_reviews)

    # 保存数据
    json_util.dump(json_data, keep_url)
    json_util.dump(unlabeled_json_data, unlabeled_keep_url)


if __name__ == '__main__':
    main()
