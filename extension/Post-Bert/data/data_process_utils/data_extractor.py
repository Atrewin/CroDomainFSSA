# coding=utf-8
import sys
from os.path import normpath,join,dirname
sys.path.append(normpath(join(dirname(__file__), '..')))


import os
import random
from glob import glob

random.seed(0)

import argparse


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

def construction2pair(source_unlabeled_reviews, target_unlabeled_reviews):
    """
        :param source_unlabeled_reviews, target_unlabeled_reviews: list：str
        :return: list(数组)
        """
    samples = []
    random.shuffle(source_unlabeled_reviews)
    random.shuffle(target_unlabeled_reviews)

    #考虑变成sentence 为单位

    maxConcatLength = min(len(source_unlabeled_reviews), len(target_unlabeled_reviews))
    half = int(maxConcatLength / 2)
    # 复现的原文好像是句子力度，且是target-target 50% target-source 50% @jinhui 需要确认
    # 为什么只有3600个sample
    for i in range(maxConcatLength):
        # 暂时不考虑预处理效率问题 其实很快不需要考虑
        if(i < half/2):
            concat_sentence = "CLS " + source_unlabeled_reviews[i] + " SPE " + target_unlabeled_reviews[i] + " SPE"
            samples.append((concat_sentence, 0))
        elif(i < half):
            concat_sentence = "CLS " + target_unlabeled_reviews[i] + " SPE " + source_unlabeled_reviews[i] + " SPE"
            samples.append((concat_sentence, 0))
        elif(i < half*1.5):
            concat_sentence = "CLS " + source_unlabeled_reviews[i] + " SPE " + source_unlabeled_reviews[i] + " SPE"
            samples.append((concat_sentence, 1))
        else:
            concat_sentence = "CLS " + target_unlabeled_reviews[i] + " SPE " + target_unlabeled_reviews[i] + " SPE"
            samples.append((concat_sentence, 1))
        pass
    random.shuffle(samples)
    return samples
    pass


def construction2MASK(source_unlabeled_reviews, target_unlabeled_reviews):
    """
            :param source_unlabeled_reviews, target_unlabeled_reviews: list：str
            :return: list(数组)
            """

    random.shuffle(source_unlabeled_reviews)
    random.shuffle(target_unlabeled_reviews)

    maxConcatLength = min(len(source_unlabeled_reviews), len(target_unlabeled_reviews))

    samples = source_unlabeled_reviews[:maxConcatLength] + target_unlabeled_reviews[:maxConcatLength]
    random.shuffle(samples)

    return samples
    pass
def saveFormatData(samples, path, mode="train", formatFor = "DSP"):
    if formatFor == "DSP":
        out_fname = 'train' if mode == 'train' else 'dev'
        if not os.path.exists(path):
            os.makedirs(path)
        f1 = open(os.path.join(path, out_fname + '.input0'), 'w', encoding="utf-8")
        f2 = open(os.path.join(path, out_fname + '.label'), 'w', encoding="utf-8")
        for sample in samples:
            f1.write(sample[0] + '\n')
            f2.write(str(sample[1]) + '\n')
        f1.close()
        f2.close()

    elif formatFor == "MASK":
        out_fname = mode
        if not os.path.exists(path):
            os.makedirs(path)
        f1 = open(os.path.join(path, out_fname + '.raw'), 'w', encoding="utf-8")
        for sample in samples:
            f1.write(sample + '\n' + '\n')

        f1.close()
    pass

def process2DSP(opt):


    source_unlabeled_xml_url = opt.source_unlabeled_xml_url
    target_unlabeled_xml_url = opt.target_unlabeled_xml_url

    sentence_pair_keep_url = opt.sentence_pair_keep_url

    # 从xml文件中抽取具体的reviews, return: 数组 array

    source_unlabeled_reviews = sentence_extractor(source_unlabeled_xml_url)
    target_unlabeled_reviews = sentence_extractor(target_unlabeled_xml_url)

    # 文本预处理 @jinhui 不知道在BERT场景下需不需要文本预处理

    source_unlabeled_reviews = sent_process(source_unlabeled_reviews)
    target_unlabeled_reviews = sent_process(target_unlabeled_reviews)

    # 构造正负样本 (review1 + SEP + review, 0/1 )
    samples = construction2pair(source_unlabeled_reviews, target_unlabeled_reviews)

    # 切分train dev
    split = int(0.6 * len(samples))
    train_samples = samples[0:split]
    dev_samples = samples[split:]

    # 保存数据
    sentence_pair_keep_url = os.path.join(sentence_pair_keep_url, opt.process_mode)
    saveFormatData(train_samples, sentence_pair_keep_url, "train", formatFor="DSP")
    saveFormatData(dev_samples, sentence_pair_keep_url, "dev", formatFor="DSP")

    # # 将reviews转换成json结构 @jinhui 在BERT场景下可能不需要文本预处理 可能不需要转为json格式


def process2MASK(opt):
    source_unlabeled_xml_url = opt.source_unlabeled_xml_url
    target_unlabeled_xml_url = opt.target_unlabeled_xml_url

    sentence_pair_keep_url = opt.sentence_pair_keep_url

    # 从xml文件中抽取具体的reviews, return: 数组 array
    source_unlabeled_reviews = sentence_extractor(source_unlabeled_xml_url)
    target_unlabeled_reviews = sentence_extractor(target_unlabeled_xml_url)

    # 文本预处理 @jinhui 不知道在BERT场景下需不需要文本预处理

    source_unlabeled_reviews = sent_process(source_unlabeled_reviews)
    target_unlabeled_reviews = sent_process(target_unlabeled_reviews)

    # 合并为一个文件
    samples = construction2MASK(source_unlabeled_reviews, target_unlabeled_reviews)

    # 切分train valid test
    split = int(0.6 * len(samples))
    valid = int(0.8 * len(samples))

    train_samples = samples[0:split]
    valid_samples = samples[split:valid]
    test_samples = samples[valid:]

    sentence_pair_keep_url = os.path.join(sentence_pair_keep_url, opt.process_mode)
    saveFormatData(train_samples, sentence_pair_keep_url, "train", formatFor="MASK")
    saveFormatData(valid_samples, sentence_pair_keep_url, "valid", formatFor="MASK")
    saveFormatData(test_samples, sentence_pair_keep_url, "test", formatFor="MASK")

    pass
def main():
    # 参数获取
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_unlabeled_xml_url", default="../domain_data/init_data/kitchenAndhousewares/unlabeled.review",
                        type=str,
                        help="unlabeled_xml_url")
    parser.add_argument("--target_unlabeled_xml_url", default="../domain_data/init_data/books/book.unlabeled",
                        type=str,
                        help="unlabeled_xml_url")

    parser.add_argument("--sentence_pair_keep_url", default="../domain_data/processed_data/book2kitchen",
                        type=str, help="sentence pair keep path")

    parser.add_argument("--process_mode", default="MASK",
                        type=str, help="data format for task type")
    opt = parser.parse_args()



    # processing
    if opt.process_mode == "DSP":
        process2DSP(opt)
    elif opt.process_mode == "MASK":
        process2MASK(opt)


if __name__ == '__main__':
    main()
