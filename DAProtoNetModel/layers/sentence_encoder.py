import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '..')))
import torch
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
import torch.nn as nn
from DAProtoNetModel.layers.network.CNNEncoder import cnnEncoder

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50,
            pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length,
                word_embedding_dim, pos_embedding_dim)
        self.in_encoder = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, hidden_size)  # 编码领域不变特征
        self.sp_encoder = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, hidden_size)  # 编码领域特定特征
        self.drop = nn.Dropout(0.2)
        self.word2id = word2id

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.drop(x)
        in_x = self.in_encoder(x)
        sp_x = self.sp_encoder(x)
        return in_x, sp_x

    def tokenize(self, raw_tokens):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        return indexed_tokens


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False): 
        nn.Module.__init__(self)
        self.in_bert = BertModel.from_pretrained(pretrain_path)
        self.sp_bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity

        self.in_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)
        self.sp_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)
        # self.drop = nn.Dropout(0.2)

    def forward(self, inputs):
        # 方式一
        # word_embed, _ = self.bert(inputs['word'])
        # word_embed = self.drop(word_embed)
        # in_x = self.in_cnn_encoder(word_embed)
        # sp_x = self.sp_cnn_encoder(word_embed)
        # 方式二
        _, in_x = self.in_bert(inputs['word'])
        _, sp_x = self.sp_bert(inputs['word'])
        # in_x = self.drop(in_x)
        # sp_x = self.drop(sp_x)
        return in_x, sp_x

    def tokenize(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        for token in raw_tokens:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        return indexed_tokens


class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        for token in raw_tokens:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens


class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path,checkpoint_name, max_length, cat_entity_rep=False):
        nn.Module.__init__(self)
        # 也可以考虑直接取Bert CSL 和SLP的方案
        from fairseq.models.roberta import RobertaModel
        self.in_roberta = RobertaModel.from_pretrained(pretrain_path, checkpoint_file=checkpoint_name)
        self.sp_roberta = RobertaModel.from_pretrained(pretrain_path, checkpoint_file=checkpoint_name)

        self.max_length = max_length
        # self.tokenizer = 直接只用in_roberta内置的方法 具体查看fairseq 的接口
        self.cat_entity_rep = cat_entity_rep

        # 后面的Bert特征怎么取到的问题
        # CNN的方案 如果不用那么就没必要增加GPU负担
        # self.in_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)
        # self.sp_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)


    def forward(self, inputs):

        in_x = self.in_roberta.extract_features(inputs['word'])#B, N ,D

        sp_x = self.sp_roberta.extract_features(inputs['word'])

        # 方式二
        return in_x[:,1,:], sp_x[:,1,:]

    def tokenize(self, raw_tokens):
        # token -> index #查看到raw_tokens本身有CLS

        tokens = 'CLS'
        for token in raw_tokens:
            tokens += " " + token
        indexed_tokens = self.in_roberta.encode(tokens).tolist()


        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(2)#该用什么填充呢？
        indexed_tokens = indexed_tokens[:self.max_length]

        return indexed_tokens

class RobertaPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.roberta = RobertaForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def forward(self, inputs):
        x = self.roberta(inputs['word'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)
        return indexed_tokens 
