# coding=utf-8
import sys
from os.path import normpath,join,dirname
sys.path.append(normpath(join(dirname(__file__), '../..')))

from utils.path_util import from_project_root
from utils import json_util

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pos_xml_url", default="domain_data/init_data/books/negative.review", type=str, help="Neg xml url")
parser.add_argument("--neg_xml_url", default="domain_data/init_data/books/positive.review", type=str, help="Pos xml url")
parser.add_argument("--unlabeled_xml_url", default="domain_data/init_data/books/book.unlabeled", type=str, help="unlabeled_xml_url")
parser.add_argument("--keep_url", default="domain_data/processed_data/books/reviews.json", type=str, help="save json url")
parser.add_argument("--unlabeled_keep_url", default="domain_data/processed_data/books/unlabeled_reviews.json", type=str, help="unlabeled_keep_url")
opt = parser.parse_args()


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
        review = review.replace(".", "")
        review = review.replace("\"", "")
        review = review.replace("&quot;", "")
        review = review.replace("!", "")
        review = review.replace(":", "")
        review = review.replace(",", "")
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
    f = open(xml_url)
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


# 标注数据格式转换
def reviews2json(pos_reviews, neg_reviews):
    """
    :param pos_reviews:
    :param neg_reviews:
    :return: json数据, 格式: {"pos":[{"tokens":***}], "neg":[{"tokens":***}]}
    """
    json_data = {}
    pos_arr = []
    for review in pos_reviews:
        tokens = review.strip().split(" ")  # 可能需要进一步文本预处理
        tokens = {"tokens": tokens}
        pos_arr.append(tokens)
    json_data['pos'] = pos_arr

    neg_arr = []
    for review in neg_reviews:
        tokens = review.strip().split(" ")  # 可能需要进一步文本预处理
        tokens = {"tokens": tokens}
        neg_arr.append(tokens)
    json_data['neg'] = neg_arr

    return json_data


#  未标注数据格式转换
def unlabeledReviews2json(unlabeled_reviews):
    """
    :param unlabeled_reviews:
    :return: list(数组)
    """
    json_data = []
    for review in unlabeled_reviews:
        tokens = review.strip().split(" ")  # 可能需要进一步文本预处理
        tokens = {"tokens": tokens}
        json_data.append(tokens)

    return json_data


def main():
    # 参数获取
    pos_xml_url = from_project_root(opt.pos_xml_url)
    neg_xml_url = from_project_root(opt.neg_xml_url)
    unlabeled_xml_url = from_project_root(opt.unlabeled_xml_url)
    keep_url = from_project_root(opt.keep_url)
    unlabeled_keep_url = from_project_root(opt.unlabeled_keep_url)

    # 从xml文件中抽取具体的reviews, return: 数组 array
    pos_reviews = sentence_extractor(pos_xml_url)
    neg_reviews = sentence_extractor(neg_xml_url)
    unlabeled_reviews = sentence_extractor(unlabeled_xml_url)

    # 文本预处理
    pos_reviews = sent_process(pos_reviews)
    neg_reviews = sent_process(neg_reviews)
    unlabeled_reviews = sent_process(unlabeled_reviews)

    # 将reviews转换成json结构
    json_data = reviews2json(pos_reviews, neg_reviews)
    unlabeled_json_data = unlabeledReviews2json(unlabeled_reviews)

    # 保存数据
    json_util.dump(json_data, keep_url)
    json_util.dump(unlabeled_json_data, unlabeled_keep_url)


if __name__ == '__main__':
    main()
