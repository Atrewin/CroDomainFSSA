import sys,os
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(os.path.abspath(__file__)), '..')))
import torch.nn as nn


from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
import torch.nn as nn
from DAProtoNetModel.layers.network.CNNEncoder import cnnEncoder


class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path,checkpoint_name, max_length, cat_entity_rep=False):
        nn.Module.__init__(self)
        # 也可以考虑直接取Bert CSL 和SLP的方案
        from fairseq.models.roberta import RobertaModel
        self.in_roberta = RobertaModel.from_pretrained(pretrain_path, checkpoint_file=checkpoint_name)

        self.max_length = max_length
        # self.tokenizer = 直接只用in_roberta内置的方法 具体查看fairseq 的接口
        self.cat_entity_rep = cat_entity_rep

        self.in_encoder = self._in_encoder

        # 后面的Bert特征怎么取到的问题
        # CNN的方案 如果不用那么就没必要增加GPU负担
        # self.in_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)
        # self.sp_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)

        # 调用分类器来获得语义信息方案
        # embedding_size = 768
        # self.in_roberta.register_classification_head('emb_sentence', num_classes=embedding_size)
        # self.sp_roberta.register_classification_head('emb_sentence', num_classes=embedding_size)


    def forward(self, inputs):

        # ## 方式一 使用CNN
        # word_embed = self.in_roberta.extract_features(inputs['word'])#@jinhui 疑问：有没有办法可以限制它的更新？
        # # word_embed = self.drop(word_embed)
        # in_x = self.in_cnn_encoder(word_embed)
        # sp_x = self.sp_cnn_encoder(word_embed)
        #
        # return in_x, sp_x
        # # 方式二
        in_x = self._in_encoder(inputs)#B, S_N ,D

        return in_x
        # # 方案三 调用分类器
        # in_x = self.in_roberta.predict('emb_sentence',inputs['word'])#B, D
        # sp_x = self.sp_roberta.predict('emb_sentence', inputs['word'])
        # return  in_x, sp_x # 4,768

    def _in_encoder(self,inputs):
        in_x = self.in_roberta.extract_features(inputs['word'])[:, 1, :]  # B, S_N ,D
        return in_x

    def _sp_encoder(self,inputs):
        sp_x = self.sp_roberta.extract_features(inputs['word'])[:, 1, :]#@jinhui
        return  sp_x


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


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False):
        nn.Module.__init__(self)
        self.in_bert = BertModel.from_pretrained(pretrain_path)
        self.sp_bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        self.in_encoder = self._in_encoder
        self.sp_encoder = self._sp_encoder
        self.in_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)

        # self.drop = nn.Dropout(0.2)

    def forward(self, inputs):
        # 方式一
        # word_embed, _ = self.bert(inputs['word'])
        # word_embed = self.drop(word_embed)
        # in_x = self.in_cnn_encoder(word_embed)
        # sp_x = self.sp_cnn_encoder(word_embed)
        # 方式二
        in_x = self._in_encoder(inputs)  # 是因为后面需要多态调用才增加的封装

        # in_x = self.drop(in_x)
        # sp_x = self.drop(sp_x)
        return in_x

    def _in_encoder(self, inputs):
        _, in_x = self.in_bert(inputs['word'])
        return in_x

    def _sp_encoder(self, inputs):
        _, sp_x = self.sp_bert(inputs['word'])
        return sp_x

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
