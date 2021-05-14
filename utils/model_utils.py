
import json
from DAProtoNetModel.layers.sentence_encoder import *
import sys
import os
from os.path import normpath,join,dirname
Base_DIR=normpath(join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(0,Base_DIR)#添加环境变量，因为append是从列表最后开始添加路径，可能前面路径有重复，最好用sys.path.insert(Base_DIR)从列表最前面开始添加
import numpy as np

def getSentenceEncoder(encoder_name, opt):

    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        max_length = opt.max_length
        sentence_encoder = CNNSentenceEncoder(glove_mat, glove_word2id, max_length,hidden_size=opt.hidden_size)
    elif encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        max_length = opt.max_length
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, max_length, cat_entity_rep=opt.cat_entity_rep, mask_entity=opt.mask_entity)
    elif encoder_name == 'roberta':
        pretrain_ckpt = opt.pretrain_ckpt
        max_length = opt.max_length
        filepath, tempfilename = os.path.split(pretrain_ckpt)
        sentence_encoder =RobertaSentenceEncoder(filepath,tempfilename, max_length, cat_entity_rep=opt.cat_entity_rep)
    elif encoder_name == 'roberta_newGraph':
        pretrain_ckpt = opt.pretrain_ckpt
        max_length = opt.max_length
        filepath, tempfilename = os.path.split(pretrain_ckpt)
        sentence_encoder = RobertaNewgraphSentenceEncoder(filepath,tempfilename, max_length, hidden_size=opt.hidden_size, cat_entity_rep=opt.cat_entity_rep)

    elif encoder_name == "bert_newGraph":

        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        max_length = opt.max_length
        sentence_encoder = BertGraphSentenceEncoder(pretrain_ckpt, max_length, cat_entity_rep=opt.cat_entity_rep,
                                               mask_entity=opt.mask_entity)

    elif encoder_name == "graph":
        max_length = opt.max_length
        sentence_encoder = GraphSentenceEncoder(max_length, cat_entity_rep=opt.cat_entity_rep,
                                                    mask_entity=opt.mask_entity)


    else:
        raise NotImplementedError

    return sentence_encoder