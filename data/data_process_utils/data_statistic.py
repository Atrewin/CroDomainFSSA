# coding=utf-8
import sys
from os.path import normpath,join,dirname
sys.path.append(normpath(join(dirname(__file__), '..')))
import argparse
from utils.path_util import from_project_root
from utils import json_util

parser = argparse.ArgumentParser()
parser.add_argument("--json_url", default="data/book_reviews.json", help="data_json_file")
opt = parser.parse_args()


def get_data_length(data_file):
    """
    :param data_file:
    :return:
    """
    url = from_project_root(data_file)
    json_data = json_util.load(url)
    category_list = ["pos", "neg"]
    max_len = 0
    min_len = 99999
    sum_len = 0
    num_instances = 0
    for key in category_list:
        sens = json_data[key]
        for sen in sens:
            num_instances += 1
            token_len = len(sen["tokens"])
            if max_len < token_len:
                max_len = token_len
            if min_len > token_len:
                min_len = token_len
            sum_len += token_len

    return max_len, min_len, sum_len//num_instances


def main():
    # 计算句子最大长度，最小长度，以及平均长度
    max_len, min_len, avg_len = get_data_length(opt.json_url)
    print("max_length:{}, min_length:{}, avg_length:{}".format(max_len, min_len, avg_len))


if __name__ == '__main__':
    main()
